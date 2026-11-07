"""Deterministic, layout-aware invoice field extraction (no LLM).

For one document - or one segment of a long multi-invoice PDF:

1. lines -> ``TextLine`` records (text, span offsets, cells, page), see ``rules_parts.layout``;
2. document context computed once: decimal separator, day/month order (given, else inferred from unambiguous
   dates, else a regional convention flagged as ambiguous) and currency (``detect_currency``);
3. the line-item table (``line_items``); its rows are not searched for labels;
4. label candidates per field: the value on the label's line, the block below a party label, or the value
   under (or above) a caption cell, each validated by type (``rules_parts.kv``);
5. selection: label specificity, totals preference, tax lines summed, arithmetic consistency raising the
   confidence of the amounts involved - never filling a missing one (``rules_parts.select``);
6. fallbacks without a label (supplier at the top of page 1, first unclaimed date) with capped confidence.

Every value comes with ``FieldMeta(source="rules", evidence=<source line text>, page=<page of that line>)``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

from app.domain.schemas import (
    DocumentType,
    ExtractionResult,
    ExtractionSource,
    ExtractionWarning,
    FieldMeta,
    InvoiceData,
    MetaNote,
)
from app.extraction.line_items import LineItemTable, find_line_items
from app.extraction.rules_parts import doc, kv, select, vocab
from app.extraction.rules_parts.kv import Candidate, labels_of
from app.extraction.rules_parts.layout import TextLine, build_text_lines
from app.extraction.rules_parts.select import MoneyPicks, Pick
from app.extraction.rules_parts.values import DocContext, build_context
from app.ingestion.types import Line, Span

DateOrderArg = Literal["DMY", "MDY", "auto"]
EVIDENCE_LIMIT = 200
_CHAR_W = 6.0
_LINE_H = 12.0
_ITEMS_CONSISTENT = 0.9


@dataclass(slots=True)
class FieldHit:
    """A field value and where it was read: ``lines`` are TextLine indices (the first holds the value)."""

    value: Any
    confidence: float
    lines: list[int]
    span: int | None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RulesOutput:
    data: InvoiceData
    hits: dict[str, FieldHit]
    lines: list[TextLine]
    warnings: list[str]
    table_sources: set[int] = field(default_factory=set)  # caller line indices of the item table's rows


def lines_from_text(text: str) -> list[Line]:
    """Pseudo-geometry for plain text: one span per whitespace-separated token, ``x = column * 6.0``,
    ``y = line index * 12.0`` (blank lines keep their row but produce no ``Line``; tabs become column gaps)."""
    out: list[Line] = []
    for index, raw in enumerate(text.replace("\r\n", "\n").replace("\r", "\n").split("\n")):
        row = raw.replace("\t", " \t").expandtabs(8).replace("\f", " ")
        y0 = index * _LINE_H
        spans = [
            Span(m.group(0), m.start() * _CHAR_W, y0, m.end() * _CHAR_W, y0 + 10.0) for m in re.finditer(r"\S+", row)
        ]
        if spans:
            out.append(Line(spans=spans))
    return out


# ------------------------------------------------------------------------------------------------ hits


def _pick_hit(pick: Pick | None) -> FieldHit | None:
    if pick is None:
        return None
    lines = list(dict.fromkeys(c.line for c in pick.members))
    span = pick.members[0].span if pick.members else None
    return FieldHit(pick.value, pick.confidence, lines, span, list(dict.fromkeys(pick.notes)))


def _found_hit(found: doc.Found | None, lines: list[TextLine]) -> FieldHit | None:
    if found is None:
        return None
    line = lines[found.line]
    notes = list(found.notes)
    confidence = found.confidence
    if line.conf is not None and line.conf < 0.8:
        notes.append(MetaNote.LOW_OCR_CONFIDENCE.value)
        confidence *= line.conf
    return FieldHit(found.value, confidence, [found.line], line.span_index_at(found.pos), notes)


def _party_cells(lines: list[TextLine]) -> tuple[set[tuple[int, int]], int | None]:
    """Cells of every party label (and its block) on the first page - never the supplier fallback - and the last
    line of the first customer block (the fallback looks above and beside it, not below)."""
    excluded: set[tuple[int, int]] = set()
    block_end: int | None = None
    for line in lines:
        if line.page != lines[0].page:
            break
        for lab in labels_of(line):
            if lab.field in vocab.PARTY_FIELDS:
                cells = doc.party_block_cells(lines, line, lab)
                excluded |= cells
                if lab.field == "customer_name" and block_end is None:
                    block_end = max(i for i, _ in cells)
    return excluded, block_end


def _items_sum(table: LineItemTable | None) -> Decimal | None:
    if table is None or not table.items or any(i.amount is None for i in table.items):
        return None
    return sum((i.amount for i in table.items if i.amount is not None), Decimal("0"))


def _tax_bases(money: MoneyPicks) -> list[Decimal]:
    if money.subtotal is None:
        return []
    subtotal = Decimal(str(money.subtotal.value))
    discount = abs(Decimal(str(money.discount.value))) if money.discount else Decimal("0")
    shipping = Decimal(str(money.shipping.value)) if money.shipping else Decimal("0")
    return [subtotal - discount, subtotal - discount + shipping, subtotal]


def _dates(groups: dict[str, list[Candidate]], hits: dict[str, FieldHit], lines: list[TextLine],
           ctx: DocContext, cands: list[Candidate], table: LineItemTable | None) -> None:
    for name in ("invoice_date", "due_date"):
        hit = _pick_hit(select.pick_labelled(groups.get(name, [])))
        if hit is not None:
            hits[name] = hit
    if "invoice_date" not in hits:
        stop = table.header_line if table is not None else None
        fallback = doc.invoice_date_fallback(lines, ctx, kv.claimed_dates(cands), stop)
        hit = _found_hit(fallback, lines)
        if hit is not None and ("due_date" not in hits or hit.value != hits["due_date"].value):
            hits["invoice_date"] = hit


def _parties(groups: dict[str, list[Candidate]], hits: dict[str, FieldHit], lines: list[TextLine],
             ctx: DocContext, table: LineItemTable | None) -> None:
    for name in vocab.PARTY_FIELDS:
        hit = _pick_hit(select.pick_labelled(groups.get(name, [])))
        if hit is not None:
            hits[name] = hit
    if "supplier_name" not in hits:
        excluded, block_end = _party_cells(lines)
        stops = [table.header_line] if table is not None and table.header_line is not None else []
        if block_end is not None:
            stops.append(block_end + 1)
        hit = _found_hit(doc.supplier_fallback(lines, ctx, excluded, min(stops, default=None)), lines)
        customer = hits.get("customer_name")
        if hit is not None and (customer is None or customer.value != hit.value):
            hits["supplier_name"] = hit


def _money(groups: dict[str, list[Candidate]], hits: dict[str, FieldHit], lines: list[TextLine],
           table: LineItemTable | None, warnings: list[str]) -> None:
    money = select.pick_money(groups, lines, _items_sum(table))
    for name, pick in money.as_dict().items():
        hit = _pick_hit(pick)
        if hit is not None:
            hits[name] = hit
    rate_pick = select.pick_tax_rate(money.tax, groups.get("tax_rate", []), _tax_bases(money))
    rate = _pick_hit(rate_pick)
    if rate is not None and rate_pick is not None:
        source = rate_pick.members[0]  # the percentage is printed with the label
        rate.lines = [source.label_line]
        rate.span = lines[source.label_line].span_index_at(source.label.start)
        hits["tax_rate"] = rate
    if table is None:
        warnings.append(ExtractionWarning.NO_LINE_ITEMS_TABLE.value)
        return
    confidence = max(table.confidence, _ITEMS_CONSISTENT) if money.items_consistent else table.confidence
    first = table.header_line if table.header_line is not None else min(table.lines)
    hits["line_items"] = FieldHit(table.items, confidence, [first], 0, [])


def run_rules(
    lines: Sequence[Line],
    *,
    default_dollar: str = "USD",
    date_order: DateOrderArg = "auto",
    page_numbers: Sequence[int] | None = None,
) -> RulesOutput:
    """Field values and their provenance (``extract_invoice_rules`` wraps this into an ``ExtractionResult``)."""
    text_lines = build_text_lines(lines, page_numbers)
    if not text_lines:
        return RulesOutput(InvoiceData(), {}, [], [ExtractionWarning.NO_LINE_ITEMS_TABLE.value])
    ctx = build_context([ln.text for ln in text_lines], default_dollar, date_order)
    table = find_line_items(text_lines, ctx)
    cands = kv.collect(text_lines, ctx, set(table.lines) if table else set())
    groups = select.by_field(cands)
    hits: dict[str, FieldHit] = {}
    warnings: list[str] = []

    number = _pick_hit(select.pick_labelled(groups.get("invoice_number", [])))
    if number is not None:
        hits["invoice_number"] = number
    _dates(groups, hits, text_lines, ctx, cands, table)
    _parties(groups, hits, text_lines, ctx, table)
    _money(groups, hits, text_lines, table, warnings)
    for name, found in (("currency", doc.currency(text_lines, ctx)), ("document_type", doc.document_type(text_lines))):
        hit = _found_hit(found, text_lines)
        if hit is not None:
            hits[name] = hit

    data = InvoiceData(**{name: hit.value for name, hit in hits.items()})
    sources = {text_lines[i].source for i in table.lines} if table is not None else set()
    return RulesOutput(data, hits, text_lines, warnings, sources)


def _evidence(hit: FieldHit, lines: list[TextLine]) -> str:
    text = " | ".join(dict.fromkeys(lines[i].text.strip() for i in hit.lines))
    return text[:EVIDENCE_LIMIT]


def to_result(output: RulesOutput) -> ExtractionResult:
    meta = {
        name: FieldMeta(
            confidence=round(min(max(hit.confidence, 0.0), 1.0), 4),
            source=ExtractionSource.RULES,
            evidence=_evidence(hit, output.lines),
            page=output.lines[hit.lines[0]].page,
            notes=list(dict.fromkeys(hit.notes)),
        )
        for name, hit in output.hits.items()
        if not (name == "document_type" and hit.value == DocumentType.UNKNOWN)
    }
    result = ExtractionResult(data=output.data, meta=meta, method="rules", warnings=list(output.warnings))
    result.compute_overall_confidence()
    return result


def extract_invoice_rules(
    lines: Sequence[Line],
    *,
    default_dollar: str = "USD",
    date_order: DateOrderArg = "auto",
    page_numbers: Sequence[int] | None = None,
) -> ExtractionResult:
    """Extract invoice fields from layout lines (digital PDF words, OCR boxes or ``lines_from_text``).

    ``page_numbers`` gives the page of each line when ``lines`` span several pages; ``date_order`` forces the
    day/month order ("auto" infers it); ``default_dollar`` is the ISO code a bare "$" stands for.
    """
    return to_result(run_rules(lines, default_dollar=default_dollar, date_order=date_order, page_numbers=page_numbers))
