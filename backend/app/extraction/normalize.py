"""Parsing and normalization of amounts, dates, currencies, percentages, invoice numbers and party names.

These helpers are deliberately conservative: when a token could be an amount *or* something else (a date, a phone
number, an identifier, a percentage, a year) it is not an amount. Callers combine them with label context.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Literal

from rapidfuzz import fuzz

DateOrder = Literal["DMY", "MDY", "auto"]
DecimalHint = Literal[",", "."] | None

# ------------------------------------------------------------------------------------------------ currencies

ISO_CURRENCIES: frozenset[str] = frozenset(
    {
        "USD", "EUR", "GBP", "INR", "CAD", "AUD", "NZD", "JPY", "CNY", "CHF", "SGD", "HKD", "AED", "SAR", "QAR",
        "KWD", "BHD", "OMR", "SEK", "NOK", "DKK", "PLN", "CZK", "HUF", "RON", "BGN", "TRY", "ILS", "ZAR", "NGN",
        "KES", "EGP", "MAD", "BRL", "MXN", "ARS", "CLP", "COP", "PEN", "PKR", "BDT", "LKR", "NPR", "MYR", "IDR",
        "THB", "VND", "KRW", "TWD", "RUB", "UAH",
    }
)
# Codes that are also common words/acronyms in invoice text: only trusted right next to an amount.
_RISKY_CODES = frozenset({"PHP", "CAD", "ARS", "TRY", "MAD", "PEN", "COP"})

# symbol -> (ISO code, confidence, inferred?)
_SYMBOLS: dict[str, tuple[str, float, bool]] = {
    "US$": ("USD", 0.95, False),
    "USD$": ("USD", 0.95, False),
    "CA$": ("CAD", 0.95, False),
    "CAD$": ("CAD", 0.95, False),
    "C$": ("CAD", 0.9, False),
    "AU$": ("AUD", 0.95, False),
    "A$": ("AUD", 0.9, False),
    "NZ$": ("NZD", 0.95, False),
    "HK$": ("HKD", 0.95, False),
    "S$": ("SGD", 0.9, False),
    "R$": ("BRL", 0.9, False),
    "€": ("EUR", 0.95, False),
    "£": ("GBP", 0.95, False),
    "₹": ("INR", 0.95, False),
    "₩": ("KRW", 0.9, False),
    "₱": ("PHP", 0.9, False),
    "₺": ("TRY", 0.9, False),
    "₪": ("ILS", 0.9, False),
    "₦": ("NGN", 0.9, False),
    "฿": ("THB", 0.9, False),
    "₫": ("VND", 0.9, False),
    "₽": ("RUB", 0.9, False),
    "zł": ("PLN", 0.9, False),
    "Rs.": ("INR", 0.7, True),
    "Rs": ("INR", 0.7, True),
    "₨": ("INR", 0.6, True),
    "¥": ("JPY", 0.6, True),
}
_MULTI_SYMBOLS = sorted((s for s in _SYMBOLS if len(s) > 1), key=len, reverse=True)
_SYMBOL_RE = "|".join(re.escape(s) for s in _MULTI_SYMBOLS) + r"|[$€£₹₩₱₺₪₦฿₫₽₨¥]"
_ISO_RE = "|".join(sorted(ISO_CURRENCIES))


@dataclass(frozen=True, slots=True)
class CurrencyGuess:
    code: str
    confidence: float
    raw: str
    inferred: bool


@dataclass(frozen=True, slots=True)
class MoneyMatch:
    value: Decimal
    start: int
    end: int
    raw: str
    currency: str | None = None  # ISO code of an adjacent currency marker, if any


@dataclass(frozen=True, slots=True)
class ParsedDate:
    value: date
    ambiguous: bool
    raw: str


@dataclass(frozen=True, slots=True)
class DateMatch:
    parsed: ParsedDate
    start: int
    end: int


# --------------------------------------------------------------------------------------------------- masking

_MONTHS: dict[str, int] = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4, "april": 4, "may": 5,
    "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MON = (
    r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
# A date may be glued to a time when OCR drops the space ("2024-03-0510:32"): digits right after the date only
# end it when they form a time.
_GLUED_TIME = r"(?=\d{1,2}:\d{2}(?!\d))"
_DATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ymd",
        re.compile(
            r"(?<![\d./-])(?P<y>(?:19|20)\d{2})(?P<sep>[-/.])(?P<m>\d{1,2})(?P=sep)(?P<d>\d{1,2})"
            rf"(?:(?![\d])|{_GLUED_TIME})"
        ),
    ),
    (
        "num",
        re.compile(
            r"(?<![\d./-])(?P<a>\d{1,2})(?P<sep>[/.-])(?P<b>\d{1,2})(?P=sep)(?P<y>\d{4}|\d{2})"
            rf"(?:(?![\d]|[/-]\d)|{_GLUED_TIME})"
        ),
    ),
    (
        "dmy_text",
        re.compile(
            r"(?<![\w])(?P<d>\d{1,2})(?:st|nd|rd|th)?[\s/.-]*(?:of\s+)?" + _MON + r"[\s/.,-]*(?P<y>\d{4}|\d{2})(?!\d)",
            re.IGNORECASE,
        ),
    ),
    (
        "mdy_text",
        re.compile(
            r"(?<![\w])" + _MON + r"[\s/.-]*(?P<d>\d{1,2})(?:st|nd|rd|th)?,?[\s/.-]*(?P<y>\d{4})(?!\d)",
            re.IGNORECASE,
        ),
    ),
)
_TIME_RE = re.compile(r"(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp][Mm])?(?!\d)")
_PERCENT_RE = re.compile(r"(?<![\w.,])(?P<num>\d+(?:[.,]\d+)?)\s?%")
_PHONE_RE = re.compile(
    r"(?<![\w])(?:\+\d{1,3}[\s-]?)?(?:\(\d{1,4}\)[\s-]?)?\d{2,5}(?:[\s-]\d{2,5}){1,4}(?![\w.,])"
)
_LONG_DIGITS_RE = re.compile(r"(?<![\w.,])\d{10,}(?![\w.,])")


def _mask(text: str, spans: Iterable[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for i in range(start, end):
            chars[i] = " "
    return "".join(chars)


def _phone_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for m in _PHONE_RE.finditer(text):
        token = m.group(0)
        digits = sum(ch.isdigit() for ch in token)
        looks_phone = token.startswith("+") or "(" in token or "-" in token
        if digits >= 8 and looks_phone:
            spans.append(m.span())
    return spans


def _non_amount_spans(text: str) -> list[tuple[int, int]]:
    spans = [(m.start, m.end) for m in find_dates(text)]
    # times are looked for with the dates blanked out, so a time glued to a date ("2024-03-0510:32") is one too
    spans += [m.span() for m in _TIME_RE.finditer(_mask(text, spans))]
    spans += [m.span() for m in _PERCENT_RE.finditer(text)]
    spans += _phone_spans(text)
    spans += [m.span() for m in _LONG_DIGITS_RE.finditer(text)]
    return spans


# "1 234,56" / "12 345 678.90": thousands grouped with a regular space. Only this exact shape (a 1-3 digit head,
# 3-digit groups, two decimals) is read as one amount; anything else stays separate numbers ("Qty 12 345").
_SPACE_GROUPED_RE = re.compile(r"(?<![\d.,])\d{1,3}(?: \d{3})+[.,]\d{2}(?!\d)")


def _join_space_groups(text: str) -> str:
    """``text`` with the regular spaces inside space-grouped amounts replaced by NBSP (same length, so offsets into
    ``text`` stay valid); the number parser already treats NBSP as a group separator."""
    if " " not in text:
        return text
    return _SPACE_GROUPED_RE.sub(lambda m: m.group(0).replace(" ", "\N{NO-BREAK SPACE}"), text)


# ----------------------------------------------------------------------------------------------------- money

_GROUP_CHARS = "   '’"
_NUMBER = r"\d{1,3}(?:[,.   '’]\d{2,3})+(?:[.,]\d{1,4})?|\d+(?:[.,]\d{1,4})?"
_MONEY_RE = re.compile(
    r"(?<![\w.,/\-−])"
    r"(?P<open>\()?\s?"
    r"(?P<neg1>[-−])?\s?"
    rf"(?P<cur1>{_SYMBOL_RE}|(?:{_ISO_RE})(?![A-Za-z]))?\s?"
    r"(?P<neg2>[-−])?\s?"
    rf"(?P<num>{_NUMBER})"
    rf"(?:\s?(?P<cur2>{_SYMBOL_RE}|(?:{_ISO_RE})(?![A-Za-z])))?"
    r"(?P<close>\))?"
    r"(?P<trail>-(?![\d\w]))?"
    r"(?![\w/])"
)


def _interpret_number(token: str, decimal_hint: DecimalHint) -> Decimal | None:
    t = token
    for ch in _GROUP_CHARS:
        t = t.replace(ch, "")
    has_comma, has_dot = "," in t, "." in t
    try:
        if has_comma and has_dot:
            dec = "," if t.rfind(",") > t.rfind(".") else "."
            grp = "." if dec == "," else ","
            if t.count(dec) != 1:
                return None
            int_part, frac = t.split(dec)
            int_part = int_part.replace(grp, "")
            if not (int_part.isdigit() and frac.isdigit()):
                return None
            return Decimal(f"{int_part}.{frac}")
        if has_comma or has_dot:
            sep = "," if has_comma else "."
            parts = t.split(sep)
            if len(parts) > 2:
                if all(p.isdigit() for p in parts) and all(len(p) in (2, 3) for p in parts[1:]) and len(parts[-1]) == 3:
                    return Decimal("".join(parts))
                return None
            int_part, frac = parts
            if not (int_part.isdigit() and frac.isdigit()):
                return None
            if len(frac) == 3:
                treat_as_group = (sep == ",") != (decimal_hint == ",")
                if sep == "." and decimal_hint != ",":
                    treat_as_group = False
                if treat_as_group:
                    return Decimal(int_part + frac)
                return Decimal(f"{int_part}.{frac}")
            if len(frac) > 4:
                return None
            return Decimal(f"{int_part}.{frac}")
        if not t.isdigit():
            return None
        return Decimal(t)
    except InvalidOperation:
        return None


def _symbol_code(raw: str | None, default_dollar: str) -> str | None:
    if not raw:
        return None
    raw = raw.strip()
    if raw in _SYMBOLS:
        return _SYMBOLS[raw][0]
    if raw == "$":
        return default_dollar
    upper = raw.upper()
    return upper if upper in ISO_CURRENCIES else None


_LEADING_GROUPS = ("open", "neg1", "cur1", "neg2", "num")


def find_money_values(text: str, decimal_hint: DecimalHint = None, default_dollar: str = "USD") -> list[MoneyMatch]:
    """Every amount-like token in ``text`` (dates, times, phones, percentages, ids and bare years excluded).

    ``start``/``raw`` begin at the amount's own first character (bracket, sign, currency marker or digit), never at
    the blanked-out date / percentage before it. Amounts grouped with regular spaces ("1 234,56") are one amount.
    """
    if not text:
        return []
    joined = _join_space_groups(text)
    masked = _mask(joined, _non_amount_spans(joined))
    out: list[MoneyMatch] = []
    for m in _MONEY_RE.finditer(masked):
        num = m.group("num")
        value = _interpret_number(num, decimal_hint)
        if value is None:
            continue
        cur_raw = m.group("cur1") or m.group("cur2")
        has_marker = cur_raw is not None
        has_fraction = any(sep in num for sep in ",.")
        if not has_marker and not has_fraction and len(num) == 4 and 1900 <= int(num) <= 2100:
            continue  # a bare year
        open_paren, close_paren = m.group("open"), m.group("close")
        negative = bool(m.group("neg1") or m.group("neg2") or m.group("trail") or (open_paren and close_paren))
        if negative:
            value = -value
        # the optional "\s?" between the leading groups can match blanked-out characters: start at the first group
        start = next(m.start(group) for group in _LEADING_GROUPS if m.group(group) is not None)
        end = m.end()
        if open_paren and not close_paren:
            start = m.start("neg1") if m.group("neg1") else (m.start("cur1") if m.group("cur1") else m.start("num"))
        if close_paren and not open_paren:
            end = m.end("cur2") if m.group("cur2") else m.end("num")
        out.append(
            MoneyMatch(
                value=value,
                start=start,
                end=end,
                raw=text[start:end],
                currency=_symbol_code(cur_raw, default_dollar),
            )
        )
    return out


def parse_money(text: str, decimal_hint: DecimalHint = None) -> Decimal | None:
    """The first amount in ``text``, or ``None``."""
    values = find_money_values(text, decimal_hint)
    return values[0].value if values else None


def parse_number(text: str, decimal_hint: DecimalHint = None) -> Decimal | None:
    """The first plain number (quantity, rate) in ``text``; dates/phones/percentages excluded."""
    if not text:
        return None
    joined = _join_space_groups(text)
    masked = _mask(joined, _non_amount_spans(joined))
    for m in re.finditer(rf"(?<![\w.,])(?P<neg>[-−])?(?P<num>{_NUMBER})(?![\d])", masked):
        value = _interpret_number(m.group("num"), decimal_hint)
        if value is not None:
            return -value if m.group("neg") else value
    return None


def parse_percentage(text: str) -> Decimal | None:
    """The first percentage in ``text`` as a number of percent ("VAT 20%" -> 20)."""
    if not text:
        return None
    m = _PERCENT_RE.search(text)
    if not m:
        return None
    return _interpret_number(m.group("num"), "," if "," in m.group("num") else ".")


def infer_decimal_separator(texts: Iterable[str]) -> DecimalHint:
    """The decimal separator a document uses, from amounts whose reading is unambiguous."""
    votes: Counter[str] = Counter()
    token_re = re.compile(r"\d[\d.,  '’]*\d")
    for text in texts:
        if not text:
            continue
        masked = _mask(text, _non_amount_spans(text))
        for m in token_re.finditer(masked):
            tok = m.group(0)
            if "," in tok and "." in tok:
                votes["," if tok.rfind(",") > tok.rfind(".") else "."] += 1
                continue
            for sep in ",.":
                if tok.count(sep) == 1:
                    frac = tok.rsplit(sep, 1)[1]
                    if len(frac) in (1, 2):
                        votes[sep] += 1
    if not votes:
        return None
    (best, n_best), *rest = votes.most_common()
    if rest and rest[0][1] == n_best:
        return None
    return best  # type: ignore[return-value]


def quantize_money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def amounts_equal(a: Decimal | None, b: Decimal | None, tolerance: Decimal = Decimal("0.02")) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) <= tolerance


# ----------------------------------------------------------------------------------------------------- dates


def _year(raw: str) -> int:
    y = int(raw)
    if len(raw) == 2:
        y += 2000 if y < 70 else 1900
    return y


def _make(y: int, m: int, d: int) -> date | None:
    if not 1990 <= y <= 2100:
        return None
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _parse_match(kind: str, m: re.Match[str], order: DateOrder) -> ParsedDate | None:
    raw = m.group(0).strip(" ,.")
    if kind == "ymd":
        value = _make(int(m.group("y")), int(m.group("m")), int(m.group("d")))
        return ParsedDate(value, False, raw) if value else None
    if kind in ("dmy_text", "mdy_text"):
        month = _MONTHS[m.group("mon").lower().rstrip(".")]
        value = _make(_year(m.group("y")), month, int(m.group("d")))
        return ParsedDate(value, False, raw) if value else None
    a, b, y = int(m.group("a")), int(m.group("b")), _year(m.group("y"))
    sep = m.group("sep")
    if a > 12 and b > 12:
        return None
    if a > 12:
        value, ambiguous = _make(y, b, a), False
    elif b > 12 or a == b:
        value, ambiguous = _make(y, a, b), False
    elif sep == "." or order == "DMY":
        value, ambiguous = _make(y, b, a), False
    elif order == "MDY":
        value, ambiguous = _make(y, a, b), False
    else:
        value, ambiguous = _make(y, a, b), True
    return ParsedDate(value, ambiguous, raw) if value else None


def find_dates(text: str, order: DateOrder = "auto") -> list[DateMatch]:
    """Every calendar date in ``text`` with its position (overlaps resolved in favour of the longer match)."""
    if not text:
        return []
    candidates: list[DateMatch] = []
    for kind, pattern in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            parsed = _parse_match(kind, m, order)
            if parsed is not None:
                end = m.start() + len(m.group(0).rstrip(" ,."))
                candidates.append(DateMatch(parsed=parsed, start=m.start(), end=end))
    candidates.sort(key=lambda c: (-(c.end - c.start), c.start))
    chosen: list[DateMatch] = []
    for c in candidates:
        if all(c.end <= k.start or c.start >= k.end for k in chosen):
            chosen.append(c)
    return sorted(chosen, key=lambda c: c.start)


def parse_date(text: str, order: DateOrder = "auto") -> ParsedDate | None:
    """The first date in ``text``. Numeric day/month order: ``order`` if given, the unambiguous reading when one
    part is > 12, DMY for dotted dates, otherwise MDY flagged ``ambiguous``."""
    found = find_dates(text, order)
    return found[0].parsed if found else None


def infer_date_order(texts: Iterable[str]) -> Literal["DMY", "MDY"] | None:
    """Day/month order a document uses, from numeric slash/dash dates whose reading is unambiguous."""
    votes: Counter[str] = Counter()
    pattern = _DATE_PATTERNS[1][1]
    for text in texts:
        for m in pattern.finditer(text or ""):
            if m.group("sep") == ".":
                continue
            a, b = int(m.group("a")), int(m.group("b"))
            if a > 12 >= b:
                votes["DMY"] += 1
            elif b > 12 >= a:
                votes["MDY"] += 1
    if not votes:
        return None
    (best, n_best), *rest = votes.most_common()
    if rest and rest[0][1] == n_best:
        return None
    return best  # type: ignore[return-value]


# Currencies whose documents write numeric dates month-first; every other currency's region is day-first.
_MDY_CURRENCIES = frozenset({"USD", "PHP"})
_MIXED_ORDER_CURRENCIES = frozenset({"CAD"})
_TEXT_DMY = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?[\s.-]+(?:of\s+)?[A-Za-z]{3,9}\.?[\s,.-]+\d{2,4}\b")
_TEXT_MDY = re.compile(r"\b[A-Za-z]{3,9}\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b")


def _text_date_style(texts: Iterable[str]) -> Literal["DMY", "MDY"] | None:
    dmy = mdy = 0
    for text in texts:
        for m in find_dates(text):
            raw = m.parsed.raw
            if any(ch.isalpha() for ch in raw):
                if _TEXT_DMY.fullmatch(raw) or raw[:1].isdigit():
                    dmy += 1
                elif _TEXT_MDY.fullmatch(raw) or raw[:1].isalpha():
                    mdy += 1
    if dmy == mdy:
        return None
    return "DMY" if dmy > mdy else "MDY"


def date_order_prior(
    texts: Iterable[str], currency: str | None, decimal_hint: DecimalHint
) -> Literal["DMY", "MDY"] | None:
    """Real-world convention for numeric dates the document itself leaves ambiguous: the style of its written-out
    dates, else the currency's region (USD -> month first, most others day first), else a decimal comma (Europe).

    Both extractors read an ambiguous "05/03/2024" with this prior, so they agree on it (and both flag it)."""
    style = _text_date_style(texts)
    if style is not None:
        return style
    if currency in _MDY_CURRENCIES:
        return "MDY"
    if currency is not None and currency not in _MIXED_ORDER_CURRENCIES:
        return "DMY"
    if decimal_hint == ",":
        return "DMY"
    return None


# -------------------------------------------------------------------------------------------------- currency

_CURRENCY_LABEL_RE = re.compile(rf"\bcurrency\b\s*[:\-]?\s*\(?\s*(?P<code>{_ISO_RE})\b", re.IGNORECASE)
_AMOUNTS_IN_RE = re.compile(rf"\b(?:amounts?|prices?|figures?)\s+(?:are\s+)?in\s+(?P<code>{_ISO_RE})\b", re.IGNORECASE)
_CODE_RE = re.compile(rf"(?<![A-Za-z])(?P<code>{_ISO_RE})(?![A-Za-z])")
_CODE_NEAR_AMOUNT_RE = re.compile(
    rf"(?<![A-Za-z])(?P<a>{_ISO_RE})\s?\$?\s?[-−(]?\d|\d\)?\s?(?P<b>{_ISO_RE})(?![A-Za-z])|\((?P<c>{_ISO_RE})\)"
)
_MULTI_SYMBOL_RE = re.compile(r"(?<![A-Za-z])(?:" + "|".join(re.escape(s) for s in _MULTI_SYMBOLS if "$" in s) + ")")
_ALPHA_SYMBOL_RE = re.compile(r"(?<![A-Za-z])(?:Rs\.?|zł)(?=\s?\d)")


def detect_currency(text: str, default_dollar: str = "USD") -> CurrencyGuess | None:
    """Currency of a document: explicit label > ISO code > unambiguous symbol > ambiguous symbol > bare "$"."""
    if not text:
        return None
    m = _CURRENCY_LABEL_RE.search(text) or _AMOUNTS_IN_RE.search(text)
    if m:
        return CurrencyGuess(m.group("code").upper(), 0.98, m.group(0), False)

    near = Counter(
        (mm.group("a") or mm.group("b") or mm.group("c")) for mm in _CODE_NEAR_AMOUNT_RE.finditer(text)
    )
    anywhere = Counter(c for c in (mm.group("code") for mm in _CODE_RE.finditer(text)) if c not in _RISKY_CODES)
    codes = near or anywhere
    if codes:
        (code, _), *others = codes.most_common()
        confidence = (0.95 if near else 0.85) if not others else 0.75
        return CurrencyGuess(code, confidence, code, False)

    remaining = text
    for mm in _MULTI_SYMBOL_RE.finditer(text):
        code, conf, inferred = _SYMBOLS[mm.group(0)]
        return CurrencyGuess(code, conf, mm.group(0), inferred)
    for symbol in ("€", "£", "₹", "₩", "₱", "₺", "₪", "₦", "฿", "₫", "₽"):
        if symbol in remaining:
            code, conf, inferred = _SYMBOLS[symbol]
            return CurrencyGuess(code, conf, symbol, inferred)
    am = _ALPHA_SYMBOL_RE.search(text)
    if am:
        key = am.group(0) if am.group(0) in _SYMBOLS else am.group(0).rstrip(".")
        code, conf, inferred = _SYMBOLS.get(key, _SYMBOLS["Rs"])
        return CurrencyGuess(code, conf, am.group(0), inferred)
    for symbol in ("₨", "¥"):
        if symbol in remaining:
            code, conf, inferred = _SYMBOLS[symbol]
            return CurrencyGuess(code, conf, symbol, inferred)
    if "$" in remaining:
        return CurrencyGuess(default_dollar, 0.5, "$", True)
    return None


_DOLLAR_MARKERS: dict[str, re.Pattern[str]] = {
    "CAD": re.compile(
        r"\b(?:GST/HST|HST|QST|TVQ|TPS)\b|\b\d{9}\s?RT\s?\d{4}\b|(?s:\bGST\b.*\bPST\b|\bPST\b.*\bGST\b)"
        r"|\b[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z]\s\d[ABCEGHJ-NPRSTV-Z]\d\b"
    ),
    "AUD": re.compile(r"\b(?:ABN|ACN)\b|\b[Pp][Tt][Yy]\.?\s*[Ll][Tt][Dd]\b|\b(?:NSW|VIC|QLD|TAS|ACT)\s+\d{4}\b"),
    "NZD": re.compile(r"\bNZBN\b|\bNew Zealand\b"),
    "SGD": re.compile(r"\bUEN\b|\bSingapore\b"),
    "HKD": re.compile(r"\bHong Kong\b"),
}


def dollar_country(text: str) -> str | None:
    """Which dollar a "$"-only document means, from unmistakable local markers (Canadian HST/QST and business
    numbers, Australian ABN / "Pty Ltd", ...); ``None`` unless exactly one country is indicated."""
    hits = [code for code, pattern in _DOLLAR_MARKERS.items() if pattern.search(text)]
    return hits[0] if len(hits) == 1 else None


def document_currency(text: str, default_dollar: str = "USD") -> CurrencyGuess | None:
    """``detect_currency`` with a bare "$" refined by unmistakable local markers (still inferred, at most 0.6)."""
    currency = detect_currency(text, default_dollar=default_dollar)
    if currency is not None and currency.inferred and currency.raw == "$":
        local = dollar_country(text)
        if local is not None and local != currency.code:
            currency = CurrencyGuess(local, min(currency.confidence + 0.1, 0.6), currency.raw, True)
    return currency


# ----------------------------------------------------------------------------------------- names and numbers

_LEGAL_SUFFIXES = frozenset(
    {
        "inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "corp", "corporation", "co", "company", "gmbh",
        "plc", "pvt", "private", "sa", "sas", "sarl", "bv", "nv", "ag", "kg", "srl", "spa", "pty", "oy", "ab",
    }
)


def normalize_name(value: str) -> str:
    """Comparable form of a party name: accents/punctuation/case removed, '&' -> 'and', leading 'the' and
    trailing legal-form suffixes dropped ("ACME Corp." -> "acme")."""
    s = unicodedata.normalize("NFKD", value)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold().replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    original = s.split()
    tokens = list(original)
    if tokens and tokens[0] == "the" and len(tokens) > 1:
        tokens = tokens[1:]
    while tokens and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens or original)


def names_match(a: str | None, b: str | None, threshold: float = 0.88) -> bool:
    if not a or not b:
        return False
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    score = max(fuzz.ratio(na, nb), fuzz.token_sort_ratio(na, nb)) / 100.0
    return score >= threshold


def normalize_invoice_number(value: str) -> str:
    """Comparable form of an invoice number: upper-case alphanumerics only ("inv-2024/0042" -> "INV20240042")."""
    return re.sub(r"[^0-9A-Za-z]", "", value).upper()


def clean_text_value(value: str | None) -> str | None:
    """Strip whitespace and stray separator punctuation from an extracted text value; empty -> None."""
    if value is None:
        return None
    s = re.sub(r"\s+", " ", value).strip().strip(":;,|-–—").strip()
    return s or None
