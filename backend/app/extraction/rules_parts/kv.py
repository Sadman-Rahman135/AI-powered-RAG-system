"""Key -> value search: every field label yields at most one typed candidate value.

Search order for a label (``labels.find_labels`` + ``vocab`` fixes):

1. the same line: text between the label and the next label / key on that line;
2. party names (supplier, customer): the first name-like line of the block below the label, in its column,
   skipping attention/contact lines (preferred over a far-away value in another column of the label's line);
3. other fields whose label is a bare caption cell: the nearest line below (or above, when the document prints
   values above their captions - decided once per document by a vote) with a cell in the caption's column.

Every value is validated by type; a label whose value fails yields nothing.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from app.domain.schemas import MetaNote
from app.extraction.labels import COLUMN_ALIASES, LabelMatch, normalize_header
from app.extraction.normalize import parse_percentage
from app.extraction.rules_parts import values, vocab
from app.extraction.rules_parts.layout import Chunk, TextLine, best_chunk
from app.extraction.rules_parts.values import DocContext
from app.extraction.rules_parts.vocab import DATE_FIELDS, MONEY_FIELDS, PARTY_FIELDS

SAME_LINE = 0.9
BELOW = 0.8
OTHER_SIDE = 0.7
AMBIGUOUS_CAP = 0.7
_STACKED = ("invoice_number", *DATE_FIELDS, *MONEY_FIELDS)
_FAR_GAP_CHARS = 30.0  # a same-line party value further away than this is another column's text
_NEAR_GAP_CHARS = 12.0  # ...and one this close is the label's own value, even with a block below


@dataclass(slots=True)
class Candidate:
    field: str
    value: Any
    confidence: float
    line: int  # TextLine index of the evidence (value) line
    span: int | None  # index of the span holding the value, within that line
    label_line: int
    label: LabelMatch
    how: str  # same_line | block | below | above
    notes: list[str] = dataclasses.field(default_factory=list)
    rate: Decimal | None = None  # tax lines: the percentage printed with the label
    after: str = ""  # text after the label (total qualifiers such as "(incl. VAT)")
    date_start: int | None = None  # dates: character offset of the date in the evidence line


@dataclass(slots=True)
class _Pending:
    line: TextLine
    label: LabelMatch
    chunk: Chunk
    after: str


def _separate(line: TextLine, label: LabelMatch) -> bool:
    """A following label ends a party value only when it starts a new cell or is a "Key:" ("Bill To: Total
    Solutions Ltd" is a name)."""
    if line.chunk_at(label.start).start == label.start:
        return True
    return line.text[label.end : label.end + 3].lstrip().startswith(":")


def _segment_end(line: TextLine, labels: list[LabelMatch], k: int, name: str) -> int:
    lab = labels[k]
    end = len(line.text)
    for nxt in labels[k + 1 :]:
        if name in PARTY_FIELDS and not _separate(line, nxt):
            continue
        end = nxt.start
        break
    segment = line.text[lab.end : end]
    for key in vocab.KEY_RE.finditer(segment):
        if segment[: key.start()].strip(" .:#-"):
            return lab.end + key.start()  # "Customer: Globex   Terms: Net 30": another key/value pair starts
        # nothing before the key: it is the label's own ("Shipping charges: 12.00", "Discount amount: 5.00")
    return end


def _caption_only(line: TextLine, lab: LabelMatch, name: str) -> Chunk | None:
    """The label's cell when it holds nothing but the label (and separators / currency or rate qualifiers; for a
    party: other party labels and section words too - "Buyer (Bill to)", "Customer Details")."""
    chunk = line.chunk_at(lab.start)
    if name in PARTY_FIELDS:
        text = line.text[chunk.start : chunk.end]
        for other in labels_of(line):
            if other.field in PARTY_FIELDS and chunk.start <= other.start < chunk.end:
                text = text.replace(other.text, " ")
        words = re.findall(r"[a-z]+", text.lower())
        return chunk if all(w in vocab.SECTION_WORDS for w in words) else None
    before = line.text[chunk.start : lab.start].strip(" :#*-•")
    rest = line.text[lab.end : chunk.end]
    if before:
        return None
    if name in MONEY_FIELDS:
        return chunk if values.money_only(rest) else None
    return chunk if not rest.strip(" :#.-") else None


def _with_notes(line: TextLine, confidence: float, notes: list[str]) -> float:
    if line.conf is not None and line.conf < 0.8:
        notes.append(MetaNote.LOW_OCR_CONFIDENCE.value)
        return confidence * line.conf
    return confidence


def _candidate(
    name: str, value: Any, conf: float, line: TextLine, pos: int, label_line: TextLine, lab: LabelMatch, how: str,
    notes: list[str] | None = None, **extra: Any,
) -> Candidate:
    notes = list(notes or [])
    conf = _with_notes(line, conf, notes)
    return Candidate(name, value, conf, line.index, line.span_index_at(pos), label_line.index, lab, how, notes, **extra)


def _typed(name: str, text: str, ctx: DocContext, *, strict: bool) -> tuple[Any, int, list[str], float] | None:
    """(value, offset in text, notes, confidence cap) for a typed field read from ``text``."""
    if name == "invoice_number":
        number = values.invoice_number(text, ctx, strict=strict)
        return (number.value, number.start, [], 1.0) if number else None
    if name in DATE_FIELDS:
        date_value = values.is_pure_date(text, ctx) if strict else values.first_date(text, ctx)
        if date_value is None:
            return None
        if date_value.ambiguous:
            notes = [MetaNote.AMBIGUOUS_DATE_ORDER.value]
            return date_value.match.parsed.value, date_value.match.start, notes, AMBIGUOUS_CAP
        return date_value.match.parsed.value, date_value.match.start, [], 1.0
    money = values.pure_amount(text, ctx) if strict else values.labelled_amount(text, ctx)
    if money is None:
        return None
    amount = abs(money.value) if name == "discount" else money.value
    return amount, money.start, [], 1.0


def _rate(line: TextLine, lab: LabelMatch, end: int) -> Decimal | None:
    return parse_percentage(line.text[lab.start : end])


def _column_word(text: str) -> bool:
    h = normalize_header(text)
    return any(h in aliases for aliases in COLUMN_ALIASES.values())


def _party_block(
    lines: list[TextLine], line: TextLine, chunk: Chunk, skip: set[int]
) -> tuple[TextLine, Chunk, str] | None:
    """First name-like cell below a party label, in the label's column, skipping attention/contact lines; the
    block ends at the item table (``skip``) or a column-header cell."""
    tol = line.char_width
    last_y = line.y1
    for j in range(line.index + 1, min(len(lines), line.index + 12)):
        nxt = lines[j]
        if nxt.page != line.page or nxt.y0 - last_y > 2.6 * line.height or j in skip:
            return None
        cell = best_chunk(nxt, chunk.x0, chunk.x1, tol)
        if cell is None:
            continue
        last_y = nxt.y1
        text = cell.text
        if vocab.ATTENTION_RE.match(text) or vocab.CONTACT_RE.search(text):
            continue
        if any(lab.start <= 1 for lab in vocab.find_field_labels(text)) or _column_word(text):
            return None
        name = values.party_name(text)
        return (nxt, cell, name) if name else None
    return None


def _neighbour(
    lines: list[TextLine], pending: _Pending, direction: int, ctx: DocContext
) -> tuple[TextLine, Chunk, tuple[Any, int, list[str], float]] | None:
    """Value in the nearest cell below (``direction=1``) or above (``-1``) a caption cell, if it validates."""
    line, chunk = pending.line, pending.chunk
    tol = line.char_width
    j = line.index + direction
    while 0 <= j < len(lines) and lines[j].page == line.page:
        nxt = lines[j]
        gap = nxt.y0 - line.y1 if direction > 0 else line.y0 - nxt.y1
        if gap > 3.0 * line.height:
            return None
        if gap > -0.3 * line.height:
            cell = best_chunk(nxt, chunk.x0, chunk.x1, tol)
            if cell is not None:
                if any(lab.start <= 1 for lab in vocab.find_field_labels(cell.text)):
                    return None
                typed = _typed(pending.label.field, cell.text, ctx, strict=True)
                return (nxt, cell, typed) if typed else None
        j += direction
    return None


def collect(
    lines: list[TextLine],
    ctx: DocContext,
    skip: set[int],
    *,
    fields: frozenset[str] | None = None,
    scan: set[int] | None = None,
) -> list[Candidate]:
    """Candidates for every labelled field of the document. Lines in ``skip`` (table rows) are ignored;
    ``fields`` / ``scan`` restrict the search to some fields / label lines (the page segmenter's quick look)."""
    out: list[Candidate] = []
    pending: list[_Pending] = []
    for line in lines:
        if line.index in skip or (scan is not None and line.index not in scan):
            continue
        labels = labels_of(line)
        for k, lab in enumerate(labels):
            name = lab.field
            if fields is not None and name not in fields:
                continue
            if name == "currency" or (lab.text == "#" and line.chunk_at(lab.start).start != lab.start):
                continue  # a bare "#" names the invoice number only at the start of a cell ("# INV-000123")
            end = _segment_end(line, labels, k, name)
            segment = line.text[lab.end : end]
            after = line.text[lab.end :]
            if name in PARTY_FIELDS:
                out.extend(_party(lines, line, lab, segment, end, skip))
                continue
            typed = _typed(name, segment, ctx, strict=False)
            rate = _rate(line, lab, end) if name == "tax" else None
            if typed is not None:
                value, offset, notes, cap = typed
                out.append(
                    _candidate(
                        name, value, min(SAME_LINE, cap), line, lab.end + offset, line, lab, "same_line", notes,
                        rate=rate, after=after, date_start=lab.end + offset if name in DATE_FIELDS else None,
                    )
                )
                continue
            if name == "tax" and rate is not None:
                out.append(_candidate("tax_rate", rate, 0.85, line, lab.start, line, lab, "same_line"))
            chunk = _caption_only(line, lab, name)
            if chunk is not None and name in _STACKED:
                pending.append(_Pending(line, lab, chunk, after))
    out.extend(_resolve_pending(lines, pending, ctx))
    return out


def labels_of(line: TextLine) -> list[LabelMatch]:
    if line.labels_cache is None:
        line.labels_cache = vocab.find_field_labels(line.text)
    return line.labels_cache


def _party(
    lines: list[TextLine], line: TextLine, lab: LabelMatch, segment: str, end: int, skip: set[int]
) -> list[Candidate]:
    """Party value: in the label's own cell, else a near cell to its right ("Bill To: | Globex"), else the block
    below the label, else a cell further right on the same line (not beyond another column's distance)."""
    head = segment.split("  ")
    text = next((part for part in head if part.strip(" :#-")), "")
    chunk = line.chunk_at(lab.start)
    pos = lab.end + segment.find(text) if text else end
    name = values.party_name(text) if text else None
    gap = (line.x_at(pos) - chunk.x1) / line.char_width if text else 0.0
    if name is not None and (pos < chunk.end or gap <= _NEAR_GAP_CHARS):
        return [_candidate(lab.field, name, SAME_LINE, line, pos, line, lab, "same_line")]
    caption = _caption_only(line, lab, lab.field)
    block = _party_block(lines, line, caption, skip) if caption is not None else None
    if block is not None:
        nxt, cell, block_name = block
        return [_candidate(lab.field, block_name, BELOW, nxt, cell.start, line, lab, "block")]
    if name is not None and gap <= _FAR_GAP_CHARS:
        return [_candidate(lab.field, name, SAME_LINE, line, pos, line, lab, "same_line")]
    return []


def _resolve_pending(lines: list[TextLine], pending: list[_Pending], ctx: DocContext) -> list[Candidate]:
    if not pending:
        return []
    found = {id(p): (_neighbour(lines, p, 1, ctx), _neighbour(lines, p, -1, ctx)) for p in pending}
    below_votes = sum(1 for b, _ in found.values() if b is not None)
    above_votes = sum(1 for _, a in found.values() if a is not None)
    above_first = above_votes > below_votes
    out: list[Candidate] = []
    for p in pending:
        below, above = found[id(p)]
        primary, secondary = (above, below) if above_first else (below, above)
        how = "above" if above_first else "below"
        hit, conf = primary, BELOW
        if hit is None and p.label.field not in MONEY_FIELDS:
            hit, conf, how = secondary, OTHER_SIDE, ("below" if above_first else "above")
        if hit is None:
            continue
        nxt, cell, (value, offset, notes, cap) = hit
        name = p.label.field
        rate = parse_percentage(p.chunk.text) if name == "tax" else None
        out.append(
            _candidate(
                name, value, min(conf, cap), nxt, cell.start + offset, p.line, p.label, how, notes,
                rate=rate, after=p.after, date_start=cell.start + offset if name in DATE_FIELDS else None,
            )
        )
    return out


def claimed_dates(candidates: list[Candidate]) -> set[tuple[int, int]]:
    """(line, offset) of every date taken by a date label - excluded from the invoice-date fallback."""
    return {(c.line, c.date_start) for c in candidates if c.field in DATE_FIELDS and c.date_start is not None}
