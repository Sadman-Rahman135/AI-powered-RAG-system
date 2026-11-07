"""Fixture builders for the pipeline tests.

Real files are generated at test time: digital invoice PDFs with PyMuPDF ``insert_htmlbox`` and XLSX ledgers with
openpyxl (literal cell values). ``canned_record`` builds ``ExtractedRecord`` objects so the processor tests can run
without the extraction modules (they monkeypatch ``app.pipeline.processor.extract_records``).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import openpyxl
import pymupdf

from app.db.session import session_scope
from app.domain.schemas import (
    INVOICE_FIELDS,
    DocumentType,
    ExtractionResult,
    ExtractionSource,
    FieldMeta,
    InvoiceData,
    LineItem,
    SourceKind,
    field_is_present,
)
from app.extraction.service import ExtractedRecord
from app.ingestion.sniff import HEAD_BYTES, detect_file_kind, detect_mime_type
from app.pipeline.intake import create_document
from app.storage import get_storage, suffix_for

SUPPLIER = "Acme Supplies Ltd"
CUSTOMER = "Globex Corporation"
CSS = "* {font-family: sans-serif; font-size: 11px;}"
ITEMS_A = (("Widget A", "2", "25.00", "50.00"), ("Widget B", "1", "50.00", "50.00"))
LEDGER_HEADER = ("Invoice No", "Invoice Date", "Supplier", "Customer", "Subtotal", "Tax", "Total", "Currency")

Item = tuple[str, str, str, str]  # description, quantity, unit price, amount


# ------------------------------------------------------------------------------------------------- PDFs


def _items_table(items: Sequence[Item]) -> str:
    right = "style='text-align:right'"
    rows = "".join(
        f"<tr><td>{d}</td><td {right}>{q}</td><td {right}>{p}</td><td {right}>{a}</td></tr>" for d, q, p, a in items
    )
    return (
        "<table style='width:100%'><tr><th style='text-align:left'>Description</th>"
        f"<th {right}>Qty</th><th {right}>Unit Price</th><th {right}>Amount</th></tr>{rows}</table>"
    )


def _totals(subtotal: Decimal, tax: Decimal, total: Decimal, tax_rate: str, currency: str) -> str:
    right = "style='text-align:right'"
    return (
        "<table style='width:100%'>"
        f"<tr><td style='width:60%'></td><td>Subtotal:</td><td {right}>{subtotal:.2f}</td></tr>"
        f"<tr><td></td><td>Tax ({tax_rate}%):</td><td {right}>{tax:.2f}</td></tr>"
        f"<tr><td></td><td>Total:</td><td {right}>{currency} {total:.2f}</td></tr></table>"
    )


def invoice_pages(
    number: str,
    *,
    supplier: str = SUPPLIER,
    customer: str = CUSTOMER,
    items: Sequence[Item] = ITEMS_A,
    tax_rate: str = "10",
    currency: str = "USD",
    invoice_date: str = "2024-03-05",
    total: Decimal | None = None,
    split: bool = False,
) -> list[str]:
    """HTML of one invoice: one page, or (``split``) two pages marked "Page 1 of 2" / "Page 2 of 2" with the
    first item on page 1 and the rest plus the totals on page 2. ``total`` overrides the computed total."""
    subtotal = sum((Decimal(amount) for *_, amount in items), Decimal(0))
    tax = (subtotal * Decimal(tax_rate) / 100).quantize(Decimal("0.01"))
    printed_total = subtotal + tax if total is None else total
    header = (
        f"<h2>{supplier}</h2><p>12 Harbour Road, Springfield</p><h1>INVOICE</h1>"
        f"<table><tr><td>Invoice No:</td><td>{number}</td></tr>"
        f"<tr><td>Invoice Date:</td><td>{invoice_date}</td></tr></table>"
        f"<p>Bill To: {customer}</p>"
    )
    totals = _totals(subtotal, tax, printed_total, tax_rate, currency)
    if not split:
        return [header + _items_table(items) + totals]
    return [
        header + _items_table(items[:1]) + "<p>Page 1 of 2</p>",
        _items_table(items[1:]) + totals + "<p>Page 2 of 2</p>",
    ]


def write_pdf(path: Path, pages: Sequence[str]) -> Path:
    """A digital PDF with one A4 page per HTML string (an empty string makes a blank page)."""
    doc = pymupdf.open()
    for html in pages:
        page = doc.new_page(width=595, height=842)
        if html:
            page.insert_htmlbox(pymupdf.Rect(50, 50, 545, 800), html, css=CSS)
    doc.save(path)
    doc.close()
    return path


def write_ledger_xlsx(path: Path, rows: Sequence[Sequence[Any]], *, sheet: str = "Ledger") -> Path:
    """An XLSX ledger: the ``LEDGER_HEADER`` row, then ``rows`` as literal cell values."""
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    assert worksheet is not None
    worksheet.title = sheet
    worksheet.append(list(LEDGER_HEADER))
    for row in rows:
        worksheet.append(list(row))
    workbook.save(path)
    return path


def ledger_row(number: str, subtotal: int, tax: int, total: int, *, customer: str = CUSTOMER) -> tuple[Any, ...]:
    return (number, date(2024, 3, 5), SUPPLIER, customer, subtotal, tax, total, "USD")


# ----------------------------------------------------------------------------------------- canned records

_SOURCES = {"rules": ExtractionSource.RULES, "spreadsheet": ExtractionSource.SPREADSHEET, "llm": ExtractionSource.LLM}


def canned_record(
    number: str | None = "INV-1001",
    *,
    supplier: str | None = SUPPLIER,
    customer: str | None = CUSTOMER,
    invoice_date: date | None = date(2024, 3, 5),
    subtotal: str | None = "100.00",
    tax: str | None = "10.00",
    total: str | None = "110.00",
    currency: str | None = "USD",
    items: Sequence[Item] = ITEMS_A,
    source_kind: SourceKind = SourceKind.INVOICE_DOCUMENT,
    page_start: int | None = 1,
    page_end: int | None = 1,
    sheet_name: str | None = None,
    row_number: int | None = None,
    confidence: float = 0.9,
    warnings: Sequence[str] = (),
) -> ExtractedRecord:
    """An extracted record whose present fields all carry a FieldMeta with ``confidence``."""
    method = "spreadsheet" if source_kind is SourceKind.LEDGER_ROW else "rules"
    data = InvoiceData(
        invoice_number=number,
        invoice_date=invoice_date,
        supplier_name=supplier,
        customer_name=customer,
        subtotal=None if subtotal is None else Decimal(subtotal),
        tax=None if tax is None else Decimal(tax),
        total=None if total is None else Decimal(total),
        currency=currency,
        document_type=DocumentType.INVOICE if source_kind is SourceKind.INVOICE_DOCUMENT else DocumentType.UNKNOWN,
        line_items=[
            LineItem(description=d, quantity=Decimal(q), unit_price=Decimal(p), amount=Decimal(a))
            for d, q, p, a in items
        ],
    )
    meta = {
        name: FieldMeta(confidence=confidence, source=_SOURCES[method], page=page_start)
        for name in INVOICE_FIELDS
        if field_is_present(data, name)
    }
    result = ExtractionResult(data=data, meta=meta, method=method, warnings=list(warnings))
    result.compute_overall_confidence()
    return ExtractedRecord(
        result=result,
        source_kind=source_kind,
        page_start=page_start,
        page_end=page_end,
        sheet_name=sheet_name,
        row_number=row_number,
        text="",
    )


def canned_ledger_row(number: str, total: str, *, subtotal: str, tax: str, row_number: int) -> ExtractedRecord:
    return canned_record(
        number,
        subtotal=subtotal,
        tax=tax,
        total=total,
        items=(),
        source_kind=SourceKind.LEDGER_ROW,
        sheet_name="Ledger",
        row_number=row_number,
        confidence=0.95,
    )


# ---------------------------------------------------------------------------------------------- uploads


def upload(path: Path, *, filename: str | None = None) -> tuple[UUID, UUID]:
    """Store ``path`` and create its document and queued job, as the upload endpoint does; returns their ids."""
    data = path.read_bytes()
    name = filename or path.name
    kind = detect_file_kind(data[:HEAD_BYTES], name)
    mime = detect_mime_type(data[:HEAD_BYTES], kind)
    stored = get_storage().save_bytes(data, suffix=suffix_for(kind, mime))
    with session_scope() as session:
        intake = create_document(session, filename=name, stored=stored, kind=kind, content_type=mime)
        return intake.document.id, intake.job.id
