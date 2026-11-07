"""The shared contracts behave as every other module assumes."""

from __future__ import annotations

import json
import uuid
from datetime import date
from decimal import Decimal

import numpy as np
from sqlalchemy import inspect, select

from app.config import get_settings
from app.db.models import ALL_TABLES, Chunk, Document, Invoice
from app.db.session import get_engine
from app.domain.schemas import (
    CORE_FIELDS,
    INVOICE_FIELDS,
    REQUIRED_FIELDS,
    ExtractionResult,
    ExtractionSource,
    FieldMeta,
    InvoiceData,
    LineItem,
)
from app.ingestion.types import (
    CorruptFileError,
    FileKind,
    IngestionError,
    Line,
    Page,
    ParsedDocument,
    Sheet,
    Span,
)


def test_empty_invoice_data_has_every_field_null() -> None:
    dumped = InvoiceData().model_dump(mode="json")
    assert set(INVOICE_FIELDS) == set(dumped)
    for name in INVOICE_FIELDS:
        if name == "line_items":
            assert dumped[name] == []
        elif name == "document_type":
            assert dumped[name] == "unknown"
        else:
            assert dumped[name] is None, name
    assert set(CORE_FIELDS) <= set(INVOICE_FIELDS)
    assert set(REQUIRED_FIELDS) <= set(CORE_FIELDS)


def test_money_serializes_as_plain_string() -> None:
    data = InvoiceData(total=Decimal("110.00"), subtotal=Decimal("1E+2"), line_items=[LineItem(amount=Decimal("5"))])
    dumped = json.loads(data.model_dump_json())
    assert dumped["total"] == "110.00"
    assert dumped["subtotal"] == "100"
    assert dumped["line_items"][0]["amount"] == "5"
    assert InvoiceData.model_validate(dumped).total == Decimal("110.00")


def test_currency_must_be_iso_code() -> None:
    import pydantic
    import pytest

    with pytest.raises(pydantic.ValidationError):
        InvoiceData(currency="usd")
    assert InvoiceData(currency="EUR").currency == "EUR"


def test_overall_confidence_penalizes_missing_required() -> None:
    full = ExtractionResult(
        data=InvoiceData(
            invoice_number="A1",
            invoice_date=date(2024, 3, 5),
            supplier_name="Acme",
            customer_name="Globex",
            subtotal=Decimal("100.00"),
            tax=Decimal("10.00"),
            total=Decimal("110.00"),
            currency="USD",
            line_items=[LineItem(description="x", amount=Decimal("100.00"))],
        ),
        meta={n: FieldMeta(confidence=0.9, source=ExtractionSource.RULES) for n in CORE_FIELDS},
    )
    assert full.compute_overall_confidence() == 0.9
    empty = ExtractionResult()
    assert empty.compute_overall_confidence() < 0.3


def test_line_text_preserves_column_gaps() -> None:
    line = Line(spans=[Span("Total:", 0, 0, 30, 10), Span("110.00", 80, 0, 110, 10)])
    assert line.text == "Total:  110.00"
    tight = Line(spans=[Span("Invoice", 41.0, 97, 79.7, 115), Span("No:", 83.2, 97, 101.7, 115)])
    assert tight.text == "Invoice No:"
    page = Page(number=1, lines=[tight, line])
    assert page.text == "Invoice No:\nTotal:  110.00"


def test_parsed_document_counts_and_sheet_text() -> None:
    doc = ParsedDocument(kind=FileKind.PDF, pages=[Page(number=1, method="text"), Page(number=2, method="ocr")])
    assert doc.page_count == 2 and doc.ocr_page_count == 1
    rows = [["Invoice No", "Total"], ["INV-1", Decimal("12.50")], [None, date(2024, 1, 2)]]
    sheet = Sheet(name="S", index=0, rows=rows)
    assert sheet.text == "Invoice No\tTotal\nINV-1\t12.50\n\t2024-01-02"
    assert ParsedDocument(kind=FileKind.XLSX, sheets=[sheet]).page_count == 1


def test_ingestion_errors_carry_codes() -> None:
    err = CorruptFileError("bad xref")
    assert isinstance(err, IngestionError)
    assert err.code == "corrupt_file" and err.permanent and err.message == "bad xref"
    assert IngestionError("x", code="custom", permanent=False).code == "custom"


def test_init_db_creates_all_tables(db_session) -> None:
    names = set(inspect(get_engine()).get_table_names())
    assert set(ALL_TABLES) <= names


def test_embedding_and_invoice_round_trip(db_session) -> None:
    dim = get_settings().embedding_dim
    doc = Document(
        filename="a.pdf", content_type="application/pdf", file_kind="pdf", size_bytes=10, sha256="0" * 64,
        storage_key="k",
    )
    db_session.add(doc)
    db_session.flush()
    vec = np.linspace(0, 1, dim, dtype=np.float32)
    db_session.add(Chunk(document_id=doc.id, text="hello", embedding=vec))
    inv = Invoice(document_id=doc.id)
    inv.apply_invoice_data(
        InvoiceData(invoice_number="inv-0042", supplier_name="ACME Corp.", total=Decimal("110.00"), currency="USD")
    )
    db_session.add(inv)
    db_session.commit()
    chunk = db_session.scalars(select(Chunk)).one()
    assert np.allclose(chunk.embedding, vec)
    loaded = db_session.get(Invoice, inv.id)
    assert loaded is not None
    data = loaded.to_invoice_data()
    assert data.total == Decimal("110.00") and data.invoice_number == "inv-0042"
    assert loaded.invoice_number_norm == "INV0042"
    assert isinstance(loaded.id, uuid.UUID)
