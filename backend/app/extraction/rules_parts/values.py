"""Typed value parsing for the rules extractor: document context (decimal separator, day/month order, currency),
and validators that turn a piece of text into an invoice number, a date, an amount or a party name - or nothing.

A value that fails its type check yields nothing; callers then fall through to the next candidate.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.extraction.normalize import (
    CurrencyGuess,
    DateMatch,
    DecimalHint,
    MoneyMatch,
    clean_text_value,
    date_order_prior,
    document_currency,
    find_dates,
    find_money_values,
    infer_date_order,
    infer_decimal_separator,
)
from app.extraction.rules_parts import vocab

DateOrder = Literal["DMY", "MDY", "auto"]


@dataclass(frozen=True, slots=True)
class DocContext:
    decimal_hint: DecimalHint
    date_order: DateOrder  # explicit, else inferred from unambiguous dates, else "auto"
    order_prior: Literal["DMY", "MDY"] | None  # convention used to read dates that stay ambiguous
    currency: CurrencyGuess | None
    default_dollar: str


def build_context(texts: Sequence[str], default_dollar: str, date_order: DateOrder) -> DocContext:
    """Document-level hints computed once and passed into every parse (the regional helpers live in normalize)."""
    decimal_hint = infer_decimal_separator(texts)
    currency = document_currency("\n".join(texts), default_dollar=default_dollar)
    order: DateOrder = date_order if date_order != "auto" else (infer_date_order(texts) or "auto")
    prior = date_order_prior(texts, currency.code if currency else None, decimal_hint) if order == "auto" else None
    return DocContext(decimal_hint, order, prior, currency, default_dollar)


# ---------------------------------------------------------------------------------------------------- dates


@dataclass(frozen=True, slots=True)
class DateValue:
    match: DateMatch
    ambiguous: bool


def first_date(text: str, ctx: DocContext) -> DateValue | None:
    """The first date in ``text`` read with the document's day/month order; an ambiguous reading uses the
    convention prior and is flagged."""
    found = find_dates(text, ctx.date_order)
    if not found:
        return None
    m = found[0]
    if not m.parsed.ambiguous:
        return DateValue(m, False)
    if ctx.order_prior == "DMY":
        again = find_dates(text[m.start : m.end], "DMY")
        if again:
            redo = again[0]
            m = DateMatch(parsed=redo.parsed, start=m.start + redo.start, end=m.start + redo.end)
    return DateValue(m, True)


def is_pure_date(text: str, ctx: DocContext) -> DateValue | None:
    """A date that makes up the whole of ``text`` (apart from separators and a trailing time)."""
    value = first_date(text, ctx)
    if value is None:
        return None
    rest = (text[: value.match.start] + " " + text[value.match.end :]).strip(" :,-|()")
    rest = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp][Mm])?\b", "", rest).strip()
    return value if not rest else None


# ---------------------------------------------------------------------------------------------------- money

def money_values(text: str, ctx: DocContext) -> list[MoneyMatch]:
    """The amounts in ``text`` read with the document's decimal separator and dollar."""
    return find_money_values(text, ctx.decimal_hint, ctx.default_dollar)


def has_fraction(m: MoneyMatch) -> bool:
    exponent = m.value.as_tuple().exponent
    return isinstance(exponent, int) and exponent < 0


def labelled_amount(segment: str, ctx: DocContext) -> MoneyMatch | None:
    """The amount printed after a money label: the rightmost amount of the segment. Registration / account
    numbers are not amounts; an integer without a currency marker is only accepted at the end of the segment."""
    values = money_values(segment, ctx)
    if not values:
        return None
    value = values[-1]
    before = segment[: value.start]
    if vocab.ID_WORDS_RE.search(before):
        return None
    # A bare integer counts only as the whole value of the label ("Amount due 500"), never inside other words
    # ("Shipping Method: Express 2", "Total (3 items) 110").
    bare_integer = not has_fraction(value) and value.currency is None
    if bare_integer and (segment[value.end :].strip(" )]*.") or not money_only(before)):
        return None
    return value


def pure_amount(text: str, ctx: DocContext) -> MoneyMatch | None:
    """An amount that makes up the whole of ``text`` (apart from currency markers)."""
    values = money_values(text, ctx)
    if len(values) != 1:
        return None
    value = values[0]
    rest = (text[: value.start] + text[value.end :]).strip(" :()")
    if rest and not re.fullmatch(r"[A-Z]{3}", rest):
        return None
    if not has_fraction(value) and value.currency is None and len(text.strip()) <= 2:
        return None
    return value


_QUALIFIER_WORDS = re.compile(
    r"\b(?:incl|excl|inc|ex|including|excluding|on|of|at|vat|gst|hst|tax|taxes)\b\.?", re.IGNORECASE
)


def money_only(text: str) -> bool:
    """True when ``text`` holds nothing but separators, currency codes/symbols, percentages and tax qualifiers
    ("(EUR)", "@ 20%", "(incl. VAT)") - i.e. a caption with no value of its own."""
    rest = re.sub(r"\d+(?:[.,]\d+)?\s?%", " ", text)
    rest = _QUALIFIER_WORDS.sub(" ", rest)
    rest = re.sub(r"\b[A-Z]{3}\b", " ", rest)
    return not re.sub(r"[$€£₹¥@()\[\]:#.,\-–/*\s]", "", rest)


# ------------------------------------------------------------------------------------------ invoice numbers

_ID_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_/.#:]*")
_PREFIX_TOKENS = frozenset({"", "no", "nr", "number", "num", "n°", "nº", "#"})


@dataclass(frozen=True, slots=True)
class IdValue:
    value: str
    start: int


def invoice_number(text: str, ctx: DocContext, *, strict: bool) -> IdValue | None:
    """An invoice-number token at the start of ``text``: id-like, at least one digit, not a date or an amount.
    ``strict`` (values found below/above a label) also requires the text to be just that token."""
    tokens = [(m.group(0), m.start()) for m in re.finditer(r"\S+", text)]
    i = 0
    while i < len(tokens) and tokens[i][0].lower().strip(":.#-") in _PREFIX_TOKENS:
        i += 1
    if i >= len(tokens) or (strict and len(tokens) - i > 1):
        return None
    raw, offset = tokens[i]
    core = raw.strip(":;,()[]").lstrip("#").rstrip(".")
    m = _ID_TOKEN.fullmatch(core)
    if not core or m is None or len(core) > 40 or not any(ch.isdigit() for ch in core):
        return None
    if find_dates(core, ctx.date_order) or re.fullmatch(r"\d{1,3}(?:,\d{3})*\.\d{2}", core):
        return None
    if not any(ch.isalpha() for ch in core) and len(re.sub(r"\D", "", core)) > 15:
        return None
    return IdValue(core, offset + raw.find(core))


# ---------------------------------------------------------------------------------------------- party names

_ADDRESS_CUT = re.compile(r",\s*(?=\d|(?:p\.?\s?o\.?\s*)?box\b)", re.IGNORECASE)
_LEADER = re.compile(r"^[\s.:·…_]+")  # dotted leaders: "Customer ........: Seaside Resorts"
_TRAILING_ATTN = re.compile(r"[\s,;]+(?:attn|attention|c/o)\b.*$", re.IGNORECASE)


def party_name(text: str) -> str | None:
    """A party name from a label value or a block line, or ``None`` when the text is not name-like (an id, a
    date, an amount, an address, contact details, a title, another key)."""
    s = clean_text_value(_LEADER.sub("", text))
    if s is None:
        return None
    s = _TRAILING_ATTN.sub("", s)
    cut = _ADDRESS_CUT.search(s)
    if cut:
        s = s[: cut.start()]
    s = s.strip(" ,;:")
    if not s or vocab.ATTENTION_RE.match(s) or vocab.CONTACT_RE.search(s):
        return None
    compact = s.replace(" ", "")
    letters = sum(ch.isalpha() for ch in compact)
    if letters < 2 or letters / len(compact) < 0.6:
        return None
    if vocab.KEY_RE.match(s) or is_title(s) or find_dates(s):
        return None
    labels = vocab.find_field_labels(s)
    if labels and sum(lab.end - lab.start for lab in labels) >= 0.5 * len(s.strip("() ")):
        return None  # "(Bill to)", "Ship To" - another label, not a name
    words = re.findall(r"[a-z]+", s.lower())
    if words and all(w in vocab.SECTION_WORDS for w in words):
        return None
    if len(s.split()) == 1 and any(ch.isdigit() for ch in s):
        return None  # "INV-0042", "29ABCDE1234F1Z5"
    return s


def is_title(text: str) -> str | None:
    """The document type a title chunk names ("TAX INVOICE" -> "invoice"), else ``None``."""
    words = re.findall(r"[a-z&]+", text.lower())
    if not words or len(words) > 6 or not all(w in vocab.TITLE_WORDS for w in words):
        return None
    joined = " ".join(words)
    for phrase, kind in vocab.TITLE_CORE.items():
        if re.search(rf"\b{phrase}\b", joined):
            return kind
    return None


def is_address(text: str) -> bool:
    return bool(vocab.STREET_RE.search(text) and re.search(r"\d", text)) or bool(vocab.POSTCODE_RE.search(text))
