"""Document-level signals: document type, currency evidence, the supplier fallback and the invoice-date fallback.

Fallbacks are heuristics without a label, so their confidence is capped (supplier <= 0.65, date <= 0.6).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from app.domain.schemas import DocumentType, MetaNote
from app.extraction.labels import COLUMN_ALIASES, LabelMatch, normalize_header
from app.extraction.normalize import find_dates
from app.extraction.rules_parts import values, vocab
from app.extraction.rules_parts.kv import labels_of
from app.extraction.rules_parts.layout import TextLine, best_chunk, median_height
from app.extraction.rules_parts.values import DocContext

SUPPLIER_WITH_SUFFIX = 0.65
SUPPLIER_PLAIN = 0.55
SUPPLIER_FOOTER = 0.5
DATE_FALLBACK = 0.6
_TITLE_LINES = 25
_SUPPLIER_LINES = 18
_FOOTER_LINES = 6
_KINDS = {"invoice": DocumentType.INVOICE, "receipt": DocumentType.RECEIPT, "credit_note": DocumentType.CREDIT_NOTE}
_KEYWORDS = (
    (re.compile(r"\bcredit\s*(?:note|memo)\b", re.IGNORECASE), DocumentType.CREDIT_NOTE),
    (
        re.compile(r"(?<!upon )(?<!on )(?<!of )\breceipt\b(?!\s+of\b)", re.IGNORECASE),
        DocumentType.RECEIPT,
    ),
    (re.compile(r"\binvoice\b", re.IGNORECASE), DocumentType.INVOICE),
)


@dataclass(slots=True)
class Found:
    value: object
    confidence: float
    line: int
    pos: int
    notes: list[str]


def document_type(lines: Sequence[TextLine]) -> Found | None:
    """A title cell ("TAX INVOICE", "SALES RECEIPT", "CREDIT NOTE") near the top, else the first type keyword."""
    if not lines:
        return None
    first_page = lines[0].page
    for line in lines[:_TITLE_LINES]:
        if line.page != first_page:
            break
        for chunk in line.chunks:
            kind = values.is_title(chunk.text)
            if kind:
                return Found(_KINDS[kind], 0.9, line.index, chunk.start, [])
    for line in lines:
        hits = [(m.start(), kind) for pattern, kind in _KEYWORDS for m in [pattern.search(line.text)] if m]
        if hits:
            pos, kind = min(hits, key=lambda h: h[0])
            return Found(kind, 0.6, line.index, pos, [])
    return None


def currency(lines: Sequence[TextLine], ctx: DocContext) -> Found | None:
    """The document currency (``detect_currency`` over all text); inferred guesses carry ``CURRENCY_INFERRED``."""
    guess = ctx.currency
    if guess is None:
        return None
    notes = [MetaNote.CURRENCY_INFERRED.value] if guess.inferred else []
    code_re = re.compile(rf"(?<![A-Za-z]){guess.code}(?![A-Za-z])")
    raw = guess.raw.strip()
    for line in lines:
        m = code_re.search(line.text) if not guess.inferred else None
        pos = m.start() if m else line.text.find(raw) if raw else -1
        if pos >= 0:
            return Found(guess.code, guess.confidence, line.index, pos, notes)
    return Found(guess.code, guess.confidence, lines[0].index, 0, notes) if lines else None


# -------------------------------------------------------------------------------------------- supplier


def party_block_cells(lines: Sequence[TextLine], label_line: TextLine, lab: LabelMatch) -> set[tuple[int, int]]:
    """(line, cell start) of a party label and of the contiguous block of cells below it in its column."""
    chunk = label_line.chunk_at(lab.start)
    cells = {(label_line.index, chunk.start)}
    last_y = label_line.y1
    for j in range(label_line.index + 1, min(len(lines), label_line.index + 8)):
        nxt = lines[j]
        if nxt.page != label_line.page or nxt.y0 - last_y > 1.2 * label_line.height:
            break
        cell = best_chunk(nxt, chunk.x0, max(chunk.x1, chunk.x0 + 20 * label_line.char_width), label_line.char_width)
        if cell is None:
            continue
        cells.add((j, cell.start))
        last_y = nxt.y1
    return cells


def _company_words(name: str) -> set[str]:
    return set(re.findall(r"[a-z]+", name.lower().replace(".", "")))


def _column_word(text: str) -> bool:
    h = normalize_header(text)
    return any(h in aliases for aliases in COLUMN_ALIASES.values())


def _supplier_like(text: str) -> str | None:
    if values.is_title(text) or vocab.KEY_RE.match(text) or vocab.PAGE_OF_RE.search(text) or _column_word(text):
        return None
    if any(lab.field != "currency" for lab in vocab.find_field_labels(text)):
        return None
    if values.is_address(text) or vocab.CONTACT_RE.search(text) or vocab.TAX_ID_RE.search(text):
        return None
    if re.search(r"\bthank", text, re.IGNORECASE):
        return None
    return values.party_name(text)


def _suffix_score(name: str) -> int:
    words = _company_words(name)
    return 2 if words & vocab.COMPANY_SUFFIXES else 1 if words & vocab.BUSINESS_WORDS else 0


def _footer_supplier(lines: Sequence[TextLine], first_page: int) -> Found | None:
    """Letterhead printed as a footer ("Northwind Consulting LLC | 200 Main St | billing@..."): only a name with a
    company suffix counts."""
    page_lines = [ln for ln in lines if ln.page == first_page]
    for line in reversed(page_lines[-_FOOTER_LINES:]):
        for chunk in line.chunks:
            offset = 0
            for part in re.split(r"\s*[|•·]\s*", chunk.text):
                name = _supplier_like(part) if part.strip() else None
                if name is not None and _suffix_score(name) == 2:
                    pos = chunk.start + chunk.text.find(part, offset)
                    return Found(name, SUPPLIER_FOOTER, line.index, pos, [])
                offset += len(part)
    return None


def supplier_fallback(
    lines: Sequence[TextLine], ctx: DocContext, excluded: set[tuple[int, int]], stop: int | None
) -> Found | None:
    """The most prominent party at the top of the first page - above (or beside) the bill-to block and the item
    table, not a title, address, contact, tax-id or column-header cell; a company suffix is preferred. Without a
    candidate at the top, a footer line naming a company."""
    if not lines:
        return None
    first_page = lines[0].page
    typical = median_height(lines)
    best: tuple[tuple[int, int, int], Found] | None = None
    for line in lines[:_SUPPLIER_LINES]:
        if line.page != first_page or (stop is not None and line.index >= stop):
            break
        if any(values.has_fraction(m) for m in values.money_values(line.text, ctx)):
            break  # amounts start: the letterhead is above
        for chunk in line.chunks:
            if (line.index, chunk.start) in excluded:
                continue
            name = _supplier_like(chunk.text)
            if name is None:
                continue
            suffix = _suffix_score(name)
            spans = line.spans[chunk.first : chunk.last + 1]
            prominent = 1 if max(s.height for s in spans) >= 1.25 * typical else 0
            key = (suffix, prominent, -line.index)
            if best is None or key > best[0]:
                conf = SUPPLIER_WITH_SUFFIX if suffix == 2 else SUPPLIER_PLAIN
                best = (key, Found(name, conf, line.index, chunk.start, []))
    return best[1] if best else _footer_supplier(lines, first_page)


# -------------------------------------------------------------------------------------- invoice date


def _blocked(line: TextLine, start: int) -> bool:
    """The nearest label left of a date names another date type (due, delivery, order, period...)."""
    left = [lab for lab in labels_of(line) if lab.end <= start]
    if not left:
        return False
    return left[-1].field in ("due_date", vocab.OTHER_DATE)


def invoice_date_fallback(
    lines: Sequence[TextLine], ctx: DocContext, claimed: set[tuple[int, int]], stop: int | None
) -> Found | None:
    """The first date on the first page that no other date label claims (confidence <= 0.6)."""
    if not lines:
        return None
    first_page = lines[0].page
    for line in lines:
        if line.page != first_page or (stop is not None and line.index >= stop):
            break
        for m in find_dates(line.text, ctx.date_order):
            if (line.index, m.start) in claimed or _blocked(line, m.start):
                continue
            value = values.first_date(line.text[m.start : m.end], ctx)
            if value is None:
                continue
            notes = [MetaNote.AMBIGUOUS_DATE_ORDER.value] if value.ambiguous else []
            parsed: date = value.match.parsed.value
            return Found(parsed, DATE_FALLBACK, line.index, m.start, notes)
    return None
