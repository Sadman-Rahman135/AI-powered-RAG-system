"""Splitting a long PDF that holds many invoices into one ``Segment`` per invoice.

Per page we read cheap signals - the invoice number (label + value), "Page k of n", "continued" markers, an
invoice header (title, invoice-number or bill-to label) and a totals line. A new segment starts when

- the page says "Page 1 of n";
- the invoice number changes (OCR look-alikes such as O/0 and I/1 compare equal) - unless the page carries the
  next "Page k of n" of the current invoice, or a "continued" marker ties it to the previous page;
- the page has an invoice header while the current segment already has its totals.

Pages without signals continue the current segment. Every page is scanned once with a handful of regular
expressions (label lookups only on lines a prefilter selects), so 1000 pages take a fraction of a second.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.extraction.normalize import normalize_invoice_number
from app.extraction.rules_parts import kv, values, vocab
from app.extraction.rules_parts.layout import build_text_lines
from app.extraction.rules_parts.values import DocContext
from app.ingestion.types import Line, Page

_CTX = DocContext(None, "auto", None, None, "USD")
_NUMBER_ONLY = frozenset({"invoice_number"})
_HEADER_LINES = 20
_CONFUSABLE = str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2"})
_CONTINUES_NEXT = re.compile(
    r"\bcontinued\s+(?:on\b|overleaf)|\bsee\s+(?:next|following)\s+page\b|\bcarried\s+forward\b|\bc/f\b",
    re.IGNORECASE,
)
_CONTINUES_PREV = re.compile(
    r"\(\s*cont(?:inued|'?d|\.)?\s*\)|\bcontinued\s+from\b|\bcontinuation\b|\bbrought\s+forward\b|\bb/f\b"
    r"|^\W*(?:[\w#.:-]+\s+){0,4}continued\W*$",
    re.IGNORECASE,
)


@dataclass(slots=True)
class Segment:
    page_start: int
    page_end: int
    pages: list[Page]
    invoice_number_hint: str | None


@dataclass(slots=True)
class _Signals:
    invoice_number: str | None = None
    page_k: int | None = None
    page_n: int | None = None
    continues_prev: bool = False
    continues_next: bool = False
    has_header: bool = False
    has_totals: bool = False


@dataclass(slots=True)
class _Open:
    pages: list[Page] = field(default_factory=list)
    number: str | None = None
    has_totals: bool = False
    last_k: int | None = None
    last_n: int | None = None
    continues_next: bool = False


def _same_number(a: str, b: str) -> bool:
    na, nb = normalize_invoice_number(a), normalize_invoice_number(b)
    return na == nb or na.translate(_CONFUSABLE) == nb.translate(_CONFUSABLE)


def _invoice_number(lines: list[Line], i: int) -> str | None:
    """The invoice number of the label on line ``i``: same line, else the cell under (or over) it."""
    first = max(i - 1, 0)
    text_lines = build_text_lines(lines[first : i + 2])
    scan = {tl.index for tl in text_lines if tl.source == i - first}
    found = kv.collect(text_lines, _CTX, set(), fields=_NUMBER_ONLY, scan=scan)
    return str(found[0].value) if found else None


_TITLE_HINT = re.compile(r"invoice|receipt|credit|bill|rechnung|factur|fattura|faktura|gutschrift", re.IGNORECASE)


def _title(line: Line) -> bool:
    """A title cell ("INVOICE", "TAX INVOICE", "RECEIPT"...) on the line."""
    if not any(_TITLE_HINT.search(span.text) for span in line.spans):
        return False
    return any(values.is_title(part) for part in line.text.split("  ") if part.strip())


def _totals(text: str) -> bool:
    for lab in vocab.find_field_labels(text):
        if lab.field == "total" and values.labelled_amount(text[lab.end :], _CTX) is not None:
            return True
    return False


def _signals(page: Page) -> _Signals:
    """Signals of one page. Prefilters run on a plain single-spaced join of the words; ``Line.text`` (with its
    column gaps) is rendered only for the few lines that need cells."""
    sig = _Signals()
    lines = page.lines
    last = len(lines) - 1
    for i, line in enumerate(lines):
        flat = " ".join(span.text for span in line.spans)
        if not flat.strip():
            continue
        lower = flat.lower()
        if sig.invoice_number is None and (
            vocab.INVOICE_NUMBER_LABEL_RE.search(flat)
            or ("nvoice" in lower and vocab.INVOICE_NUMBER_LABEL_RE.search(vocab.ocr_label_view(flat)))
        ):
            sig.invoice_number = _invoice_number(lines, i)
            sig.has_header = sig.has_header or sig.invoice_number is not None
        if sig.page_k is None and "page" in lower:
            marker = vocab.PAGE_OF_RE.search(flat)
            if marker:
                sig.page_k, sig.page_n = int(marker.group(1)), int(marker.group(2))
        if "contin" in lower or "/f" in lower or "forward" in lower:
            sig.continues_next = sig.continues_next or bool(_CONTINUES_NEXT.search(flat))
            near_top = i <= max(4, last // 3)
            sig.continues_prev = sig.continues_prev or (near_top and bool(_CONTINUES_PREV.search(flat)))
        if i < _HEADER_LINES and not sig.has_header and (vocab.PARTY_LABEL_RE.search(flat) or _title(line)):
            sig.has_header = True
        if not sig.has_totals and vocab.TOTAL_LABEL_RE.search(flat):
            sig.has_totals = _totals(line.text)
    return sig


def _starts_new(sig: _Signals, current: _Open) -> bool:
    if sig.page_k == 1:
        return True
    in_sequence = current.last_k in (None, (sig.page_k or 0) - 1) and current.last_n in (None, sig.page_n)
    if sig.page_k is not None and sig.page_k > 1 and in_sequence:
        return False  # the next page of the current invoice
    if sig.invoice_number and current.number and not _same_number(sig.invoice_number, current.number):
        tied = sig.continues_prev or current.continues_next
        return not tied or (sig.has_header and current.has_totals)
    if sig.continues_prev or current.continues_next:
        return False
    return sig.has_header and current.has_totals


def segment_pages(pages: Sequence[Page]) -> list[Segment]:
    """Group consecutive pages into invoices (see the module docstring for the rules)."""
    segments: list[Segment] = []
    current: _Open | None = None

    def close(seg: _Open) -> None:
        segments.append(Segment(seg.pages[0].number, seg.pages[-1].number, seg.pages, seg.number))

    for page in pages:
        sig = _signals(page)
        if current is None or _starts_new(sig, current):
            if current is not None:
                close(current)
            current = _Open()
        current.pages.append(page)
        current.number = current.number or sig.invoice_number
        current.has_totals = current.has_totals or sig.has_totals
        current.last_k = sig.page_k if sig.page_k is not None else current.last_k
        current.last_n = sig.page_n if sig.page_n is not None else current.last_n
        current.continues_next = sig.continues_next
    if current is not None:
        close(current)
    return segments
