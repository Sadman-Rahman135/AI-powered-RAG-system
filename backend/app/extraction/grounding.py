"""Grounding: a value the source text does not support is rejected to ``None``; it is never kept and never guessed.

``SourceIndex`` indexes one source text once and answers two kinds of question:

* *typed support* (``supports_*``, ``text_support``, ``id_support``): does the document contain this value, however it
  is written? ``1234.56`` is supported by "1.234,56 €", ``2024-03-05`` by "5 March 2024", an ambiguous "03/04/2024"
  supports both readings. These are the checks ``ground_result`` applies to any ``ExtractionResult``.
* *verbatim occurrence* (``occurs_*``): does the exact string a model copied out of the document occur in it? The LLM
  extractor uses these to enforce the "copy exactly as printed" contract.

Rules per field kind
    money         the amount equals an amount found in the text (exact to half a cent, document decimal separator
                  respected, dates/percentages/phone numbers/bare years are not amounts). A sign may be dropped but
                  never invented.
    dates         the date is one the text prints, under either day/month order when the printed date is ambiguous.
    invoice no.   its normalised form equals a normalised id-like token (or a run of adjacent tokens); a fragment of a
                  longer token only counts from five characters up and only where it does not cut a digit/letter run,
                  and it is a full ("tolerant") match only when it covers at least 80 % of the token - a shorter one
                  ("2024-0042" of "INV-2024-0042-A") is only "partial" support, which the LLM extractor scores
                  below the review threshold.
    names         the value occurs word-aligned in the text, or fuzzily (OCR noise) in a source line that is about as
                  long as the value. A short line can never vouch for a longer invented name.
    currency      its ISO code or a symbol mapping to it is printed. A bare "$" supports only the currency that
                  ``detect_currency`` yields for the document.
    tax rate      the number is printed as a percentage.
    line items    the amount is supported, otherwise the whole item is dropped; unsupported secondary values
                  (description, quantity, unit price) of a kept item are set to ``None``.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from functools import cached_property
from typing import Literal

from rapidfuzz import fuzz

from app.domain.schemas import (
    DATE_FIELDS,
    MONEY_FIELDS,
    TEXT_FIELDS,
    ExtractionResult,
    ExtractionSource,
    FieldMeta,
    LineItem,
    MetaNote,
)
from app.extraction.normalize import (
    ISO_CURRENCIES,
    CurrencyGuess,
    DecimalHint,
    date_order_prior,
    detect_currency,
    document_currency,
    find_dates,
    find_money_values,
    infer_date_order,
    infer_decimal_separator,
    normalize_invoice_number,
    normalize_name,
    parse_number,
    parse_percentage,
)

Support = Literal["exact", "tolerant", "partial"]
DateSupport = Literal["clear", "ambiguous"]
CurrencySupport = Literal["explicit", "inferred"]

MONEY_TOLERANCE = Decimal("0.005")
FUZZY_THRESHOLD = 90  # rapidfuzz partial_ratio needed for a fuzzy name match (OCR noise)
_MIN_FUZZY_CHARS = 4  # shorter names must match as whole words
_MIN_LINE_SHARE = 0.85  # a source line must be at least this long, relative to the value, to vouch for it
_MIN_FRAGMENT_CHARS = 5  # a fragment of a longer id token must be at least this long
_MIN_FRAGMENT_SHARE = 0.8  # ...and cover this share of the token to count as a full (tolerant) match
_PARTIAL_LIST_CONFIDENCE = 0.75  # ceiling for a line-item list that lost an item or a value

_SCALAR_FIELDS: tuple[str, ...] = (*TEXT_FIELDS, *DATE_FIELDS, *MONEY_FIELDS, "tax_rate", "currency")
_REJECTED = MetaNote.UNGROUNDED_VALUE_REJECTED
_SOURCE_OF_METHOD = {
    "rules": ExtractionSource.RULES,
    "llm": ExtractionSource.LLM,
    "hybrid": ExtractionSource.HYBRID,
    "spreadsheet": ExtractionSource.SPREADSHEET,
    "manual": ExtractionSource.MANUAL,
}

# Codes that are also ordinary words or acronyms: trusted only next to an amount or when the document says so
# (``detect_currency`` applies the same caution).
_WORD_LIKE_CODES = frozenset({"PHP", "CAD", "ARS", "TRY", "MAD", "PEN", "COP"})
_CODE_TOKEN_RE = re.compile(r"(?<![A-Za-z])[A-Z]{3}(?![A-Za-z])")
_BARE_DOLLAR_RE = re.compile(r"(?<![A-Za-z])\$")
_NUMBER_TOKEN_RE = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)*")
_PERCENT_RE = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)?\s?%")
_ID_TOKEN_RE = re.compile(r"[^\W_]+(?:[-_/.][^\W_]+)*")
_DIGIT_SPAN_RE = re.compile(r"\d(?:.*\d)?", re.DOTALL)
_PLAIN_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_LINE_BREAKS_RE = re.compile(r"[\r\n\f]+")


@dataclass(frozen=True, slots=True)
class _Boundary:
    """Look-arounds that make a verbatim string match only as a whole word / number / identifier / marker."""

    left: str
    right: str
    applies: Callable[[str], bool]  # whether a look-around is needed next to this edge character


_WORD = _Boundary(r"(?<!\w)", r"(?!\w)", lambda ch: ch.isalnum() or ch == "_")
_LETTER = _Boundary(r"(?<![^\W\d_])", r"(?![^\W\d_])", str.isalpha)
_NUMBER = _Boundary(r"(?<![\d.,])", r"(?!\d)(?![.,]\d)", lambda ch: True)
_ID = _Boundary(r"(?<![^\W_])(?<![^\W_][-_/.])", r"(?![^\W_])(?![-_/.][^\W_])", str.isalnum)


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _fold(text: str) -> str:
    return _collapse(text).casefold()


def amount_core(text: str, decimal_hint: DecimalHint = None) -> str | None:
    """The digits-and-separators part of the first amount in ``text`` ("(1.579,00 €)" -> "1.579,00")."""
    matches = find_money_values(text, decimal_hint)
    if not matches:
        return None
    core = _DIGIT_SPAN_RE.search(matches[0].raw)
    return core.group(0) if core else None


def plain_number_core(text: str) -> str | None:
    """The first plain number in ``text`` as printed ("7,5 %" -> "7,5")."""
    match = _PLAIN_NUMBER_RE.search(text)
    return match.group(0) if match else None


def _embedded(needle: str, token: str) -> bool:
    """``needle`` sits inside ``token`` without cutting a run of digits or a run of letters at either end."""
    start = token.find(needle)
    while start != -1:
        end = start + len(needle)
        left_ok = start == 0 or token[start - 1].isdigit() != needle[0].isdigit()
        right_ok = end == len(token) or token[end].isdigit() != needle[-1].isdigit()
        if left_ok and right_ok:
            return True
        start = token.find(needle, start + 1)
    return False


class SourceIndex:
    """One source text, indexed once for every grounding question (the heavier indexes are built lazily)."""

    def __init__(self, text: str, *, default_dollar: str = "USD") -> None:
        self.text = text
        self.default_dollar = default_dollar
        self.decimal_hint: DecimalHint = infer_decimal_separator([text])
        self.date_order = infer_date_order([text])
        self._folded = _fold(text)

    @cached_property
    def order_prior(self) -> Literal["DMY", "MDY"] | None:
        """How to read a numeric date the document leaves ambiguous - the regional convention the rules use too
        (``date_order_prior``); ``None`` when the document settles the order itself or nothing indicates one."""
        if self.date_order is not None:
            return None
        currency = document_currency(self.text, self.default_dollar)
        return date_order_prior([self.text], currency.code if currency else None, self.decimal_hint)

    # ------------------------------------------------------------------------------------ verbatim occurrence

    def _occurs(self, value: str, boundary: _Boundary) -> bool:
        needle = _fold(value)
        if not needle:
            return False
        left = boundary.left if boundary.applies(needle[0]) else ""
        right = boundary.right if boundary.applies(needle[-1]) else ""
        return re.search(f"{left}{re.escape(needle)}{right}", self._folded) is not None

    def occurs_number(self, core: str) -> bool:
        """``core`` (digits and separators as printed) occurs as a whole number, not inside a longer one."""
        return self._occurs(core, _NUMBER)

    def occurs_literal(self, value: str) -> bool:
        """``value`` occurs word-aligned, ignoring case and whitespace."""
        return self._occurs(value, _WORD)

    def occurs_currency(self, value: str) -> bool:
        """A currency symbol or code occurs; codes must not be part of a longer word ("EUR" in "EUROPE")."""
        return self._occurs(value, _LETTER)

    def text_support(self, value: str) -> Support | None:
        """Support for a name or description: exact word-aligned occurrence, or an OCR-tolerant fuzzy match."""
        if self.occurs_literal(value):
            return "exact"
        return "tolerant" if self._fuzzy_name_match(value) else None

    def id_support(self, value: str) -> Support | None:
        """Support for an invoice number: whole-token occurrence ("exact"), equality after normalisation or a
        fragment covering most of a longer token ("tolerant"), a shorter aligned fragment ("partial")."""
        if self._occurs(value, _ID):
            return "exact"
        normalized = normalize_invoice_number(value)
        if not normalized:
            return None
        if normalized in self._id_windows:
            return "tolerant"
        if len(normalized) < _MIN_FRAGMENT_CHARS:
            return None
        hosts = [
            len(token) for token in self._id_tokens if len(token) > len(normalized) and _embedded(normalized, token)
        ]
        if not hosts:
            return None
        return "tolerant" if len(normalized) >= _MIN_FRAGMENT_SHARE * min(hosts) else "partial"

    def locate(self, snippet: str) -> int | None:
        """Offset in ``text`` of ``snippet`` (case and whitespace-insensitive), or ``None``."""
        words = snippet.split()
        if not words:
            return None
        match = re.search(r"\s+".join(re.escape(word) for word in words), self.text, re.IGNORECASE)
        return match.start() if match else None

    # ---------------------------------------------------------------------------------------- typed support

    @property
    def _decimal_readings(self) -> tuple[DecimalHint, ...]:
        """The document's decimal separator, or both readings when it never settles the question."""
        return (self.decimal_hint,) if self.decimal_hint else (".", ",")

    @cached_property
    def _amounts(self) -> tuple[tuple[Decimal, ...], tuple[bool, ...]]:
        """Sorted magnitudes of every amount in the text, with whether each was printed negative."""
        seen: set[tuple[Decimal, bool]] = set()
        for hint in self._decimal_readings:
            for match in find_money_values(self.text, hint, self.default_dollar):
                seen.add((abs(match.value), match.value < 0))
        ordered = sorted(seen)
        return tuple(magnitude for magnitude, _ in ordered), tuple(negative for _, negative in ordered)

    def supports_money(self, value: Decimal) -> bool:
        """``value`` equals an amount in the text to half a cent; a negative needs a printed negative."""
        if not value.is_finite():
            return False
        magnitudes, negatives = self._amounts
        target = abs(value)
        position = bisect_left(magnitudes, target - MONEY_TOLERANCE)
        while position < len(magnitudes) and magnitudes[position] <= target + MONEY_TOLERANCE:
            if value >= 0 or negatives[position]:
                return True
            position += 1
        return False

    @cached_property
    def _numbers(self) -> frozenset[Decimal]:
        found: set[Decimal] = set()
        for token in _NUMBER_TOKEN_RE.findall(self.text):
            for hint in self._decimal_readings:
                number = parse_number(token, hint)
                if number is not None:
                    found.add(number)
        return frozenset(found)

    def supports_number(self, value: Decimal) -> bool:
        """``value`` (a quantity) is a number printed in the text."""
        return value.is_finite() and abs(value) in self._numbers

    @cached_property
    def _percents(self) -> frozenset[Decimal]:
        found = (parse_percentage(match.group(0)) for match in _PERCENT_RE.finditer(self.text))
        return frozenset(percent for percent in found if percent is not None)

    def supports_percent(self, value: Decimal) -> bool:
        """``value`` is printed in the text as a percentage."""
        return value.is_finite() and value in self._percents

    @cached_property
    def _dates(self) -> dict[date, bool]:
        """Every date the text can be read as, mapped to whether only an ambiguous reading reaches it."""
        readings: dict[date, bool] = {}

        def record(day: date, ambiguous: bool) -> None:
            readings[day] = readings.get(day, True) and ambiguous

        for match in find_dates(self.text, self.date_order or "auto"):
            parsed = match.parsed
            record(parsed.value, parsed.ambiguous)
            if parsed.ambiguous:  # both parts are <= 12, so the other day/month order is a real date too
                record(date(parsed.value.year, parsed.value.day, parsed.value.month), True)
        return readings

    def supports_date(self, value: date) -> DateSupport | None:
        """``value`` is a date the text prints (either order when the printed date is ambiguous)."""
        if value not in self._dates:
            return None
        return "ambiguous" if self._dates[value] else "clear"

    @cached_property
    def _currency_guess(self) -> CurrencyGuess | None:
        return detect_currency(self.text, self.default_dollar)

    @cached_property
    def _explicit_currencies(self) -> frozenset[str]:
        """Currencies the text states: markers next to amounts (not a bare "$") and standalone ISO codes."""
        next_to_amounts = {
            match.currency
            for match in find_money_values(self.text, self.decimal_hint, self.default_dollar)
            if match.currency and not _BARE_DOLLAR_RE.search(match.raw)
        }
        standalone = (set(_CODE_TOKEN_RE.findall(self.text)) & ISO_CURRENCIES) - _WORD_LIKE_CODES
        return frozenset(next_to_amounts | standalone)

    def supports_currency(self, code: str) -> CurrencySupport | None:
        """``code`` is printed as a code or symbol; "inferred" when only a bare "$"-style symbol supports it."""
        if code in self._explicit_currencies:
            return "explicit"
        guess = self._currency_guess
        if guess is not None and guess.code == code:
            return "inferred" if guess.inferred else "explicit"
        return None

    # ------------------------------------------------------------------------------------------- internals

    @cached_property
    def _lines(self) -> tuple[str, ...]:
        return tuple(line for line in _LINE_BREAKS_RE.split(self.text) if line.strip())

    @cached_property
    def _name_windows(self) -> tuple[tuple[int, ...], tuple[str, ...]]:
        """Normalised source lines plus every two and three consecutive lines joined (names wrap), de-duplicated and
        sorted by length (so a length floor is a bisect), as ``(lengths, windows)``."""
        lines = [name for line in self._lines if (name := normalize_name(line))]
        windows = set(lines)
        for size in (2, 3):
            windows.update(" ".join(lines[start : start + size]) for start in range(len(lines) - size + 1))
        ordered = sorted(windows, key=len)
        return tuple(len(window) for window in ordered), tuple(ordered)

    def _fuzzy_name_match(self, value: str) -> bool:
        normalized = normalize_name(value)
        if len(normalized) < _MIN_FUZZY_CHARS:
            return False
        # partial_ratio aligns the shorter string inside the longer one, so a short line that happens to be
        # contained in an invented long name would score 100. Only a line about as long as the value may vouch.
        lengths, windows = self._name_windows
        start = bisect_left(lengths, math.ceil(_MIN_LINE_SHARE * len(normalized)))
        # score_cutoff lets rapidfuzz reject hopeless pairs without a full alignment; any() stops at the first hit
        return any(
            fuzz.partial_ratio(normalized, window, score_cutoff=FUZZY_THRESHOLD) >= FUZZY_THRESHOLD
            for window in windows[start:]
        )

    @cached_property
    def _id_line_tokens(self) -> tuple[tuple[str, ...], ...]:
        """Per line, the normalised id-like tokens (letters, digits and inner - _ / . separators)."""
        return tuple(
            tuple(normalize_invoice_number(token) for token in _ID_TOKEN_RE.findall(line)) for line in self._lines
        )

    @cached_property
    def _id_tokens(self) -> tuple[str, ...]:
        return tuple(token for tokens in self._id_line_tokens for token in tokens)

    @cached_property
    def _id_windows(self) -> frozenset[str]:
        """Each id-like token, and runs of two or three adjacent tokens on one line, with separators removed."""
        windows: set[str] = set()
        for tokens in self._id_line_tokens:
            for start in range(len(tokens)):
                joined = ""
                for token in tokens[start : start + 3]:
                    joined += token
                    windows.add(joined)
        return frozenset(windows)


# --------------------------------------------------------------------------------------------- ground_result


@dataclass(frozen=True, slots=True)
class _Verdict:
    grounded: bool
    note: MetaNote | None = None  # an info note to attach to a grounded field


def _check_scalar(name: str, value: object, index: SourceIndex) -> _Verdict:
    if name == "invoice_number":
        return _Verdict(index.id_support(str(value)) is not None)
    if name in TEXT_FIELDS:
        return _Verdict(index.text_support(str(value)) is not None)
    if name in MONEY_FIELDS:
        return _Verdict(isinstance(value, Decimal) and index.supports_money(value))
    if name == "tax_rate":
        return _Verdict(isinstance(value, Decimal) and index.supports_percent(value))
    if name in DATE_FIELDS:
        date_support = index.supports_date(value) if isinstance(value, date) else None
        note = MetaNote.AMBIGUOUS_DATE_ORDER if date_support == "ambiguous" else None
        return _Verdict(date_support is not None, note)
    currency_support = index.supports_currency(str(value))
    note = MetaNote.CURRENCY_INFERRED if currency_support == "inferred" else None
    return _Verdict(currency_support is not None, note)


def _with_note(notes: list[str], note: MetaNote) -> list[str]:
    return notes if note.value in notes else [*notes, note.value]


def ground_result(
    result: ExtractionResult, source: str | SourceIndex, *, default_dollar: str = "USD"
) -> ExtractionResult:
    """A copy of ``result`` in which every value the source text does not support is gone.

    A rejected scalar becomes ``None``; its original goes to ``rejected[field]`` and its ``FieldMeta`` stays with
    confidence 0 and the ``ungrounded_value_rejected`` note, so validation can report it. Supported values keep their
    confidence. ``default_dollar`` is what a bare "$" means (``DEFAULT_DOLLAR_CURRENCY``). Calling it twice changes
    nothing the second time.
    """
    index = source if isinstance(source, SourceIndex) else SourceIndex(source, default_dollar=default_dollar)
    grounded = result.model_copy(deep=True)
    fallback_source = _SOURCE_OF_METHOD.get(grounded.method, ExtractionSource.LLM)
    for name in _SCALAR_FIELDS:
        value = getattr(grounded.data, name)
        if value is None:
            continue
        verdict = _check_scalar(name, value, index)
        meta = grounded.meta.get(name)
        if verdict.grounded:
            if verdict.note is not None and meta is not None:
                meta.notes = _with_note(meta.notes, verdict.note)
            continue
        setattr(grounded.data, name, None)
        grounded.rejected[name] = value
        evidence = meta.evidence if meta is not None and meta.evidence and index.occurs_literal(meta.evidence) else None
        grounded.meta[name] = FieldMeta(
            confidence=0.0,
            source=meta.source if meta is not None else fallback_source,
            evidence=evidence,
            notes=_with_note(meta.notes if meta is not None else [], _REJECTED),
        )
    _ground_line_items(grounded, index, fallback_source)
    grounded.compute_overall_confidence()
    return grounded


def _ground_line_items(result: ExtractionResult, index: SourceIndex, fallback_source: ExtractionSource) -> None:
    items = result.data.line_items
    if not items:
        return
    kept: list[LineItem] = []
    dropped: list[LineItem] = []
    nulled = False
    for position, item in enumerate(items):
        if item.amount is None or not index.supports_money(item.amount):
            dropped.append(item)
            continue
        unsupported: list[str] = []
        if item.unit_price is not None and not index.supports_money(item.unit_price):
            unsupported.append("unit_price")
        if item.quantity is not None and not index.supports_number(item.quantity):
            unsupported.append("quantity")
        if item.description is not None and index.text_support(item.description) is None:
            unsupported.append("description")
        for field in unsupported:
            result.rejected[f"line_items[{position}].{field}"] = getattr(item, field)
        nulled = nulled or bool(unsupported)
        kept.append(item.model_copy(update=dict.fromkeys(unsupported)) if unsupported else item)
    result.data.line_items = kept
    if dropped:
        result.rejected["line_items"] = [*result.rejected.get("line_items", []), *dropped]
    if dropped or nulled:
        meta = result.meta.get("line_items") or FieldMeta(confidence=_PARTIAL_LIST_CONFIDENCE, source=fallback_source)
        result.meta["line_items"] = meta.model_copy(
            update={
                "confidence": min(meta.confidence, _PARTIAL_LIST_CONFIDENCE) if kept else 0.0,
                "notes": _with_note(meta.notes, _REJECTED),
            }
        )
