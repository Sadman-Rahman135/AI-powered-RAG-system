"""Builders for the API tests: small real upload files (made in memory) and directly seeded database rows."""

from __future__ import annotations

import io
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

import openpyxl
import pymupdf
from PIL import Image
from sqlalchemy.orm import Session

from app.db.models import (
    CrossVerification,
    Document,
    DocumentPage,
    Invoice,
    ProcessingEvent,
    ProcessingJob,
    ValidationIssueRow,
)
from app.domain.schemas import (
    DocumentType,
    ExtractionSource,
    FieldMeta,
    InvoiceData,
    LineItem,
    ReviewStatus,
    SourceKind,
)
from app.search.chunking import build_chunks
from app.search.embeddings import HashingEmbedder
from app.search.service import index_document

BASE_TIME = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)

# ------------------------------------------------------------------------------------------------ files


def pdf_bytes(*lines: str) -> bytes:
    """A one-page digital PDF with ``lines`` of text."""
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72.0
    for line in lines or ("Hello",):
        page.insert_text((72, y), line, fontsize=11)
        y += 16
    data = doc.tobytes()
    doc.close()
    return data


def png_bytes(size: tuple[int, int] = (64, 48)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (250, 250, 250)).save(buffer, format="PNG")
    return buffer.getvalue()


def xlsx_bytes() -> bytes:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["Invoice No", "Supplier", "Total"])
    sheet.append(["LED-1", "Acme Corp", 110.0])
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def csv_bytes() -> bytes:
    return b"Invoice No,Date,Supplier,Total\nINV-1,2024-01-02,Acme Corp,110.00\nINV-2,2024-01-03,Initech,75.25\n"


EXE_BYTES = (
    b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00" + b"\xb8" + bytes(64) + b"PE\x00\x00" + bytes(256)
)


# ------------------------------------------------------------------------------------------------ rows


def seed_document(
    session: Session,
    *,
    filename: str = "invoice.pdf",
    file_kind: str = "pdf",
    status: str = "completed",
    created_offset_s: int = 0,
    job_status: str | None = "succeeded",
    sha256: str | None = None,
    storage_key: str | None = None,
    **columns: Any,
) -> Document:
    """A document (and, unless ``job_status`` is None, one job) committed so the API sessions see it."""
    created = BASE_TIME + timedelta(seconds=created_offset_s)
    document = Document(
        filename=filename,
        content_type={"pdf": "application/pdf", "image": "image/png", "csv": "text/csv"}.get(
            file_kind, "application/octet-stream"
        ),
        file_kind=file_kind,
        size_bytes=columns.pop("size_bytes", 1234),
        sha256=sha256 or f"{abs(hash((filename, created_offset_s))):064x}"[:64],
        storage_key=storage_key or f"2026/10/{UUID(int=created_offset_s + 1).hex}.pdf",
        status=status,
        created_at=created,
        updated_at=created,
        **columns,
    )
    session.add(document)
    session.flush()
    if job_status is not None:
        finished = job_status in ("succeeded", "failed")
        session.add(
            ProcessingJob(
                document_id=document.id,
                status=job_status,
                attempt=1 if job_status != "queued" else 0,
                max_attempts=3,
                stage="indexing" if finished else None,
                progress_current=1 if finished else 0,
                progress_total=1 if finished else 0,
                error="boom" if job_status == "failed" else None,
                created_at=created,
                started_at=created if job_status != "queued" else None,
                finished_at=created + timedelta(seconds=2) if finished else None,
            )
        )
    session.commit()
    return document


def invoice_data(**overrides: Any) -> InvoiceData:
    values: dict[str, Any] = {
        "invoice_number": "INV-2024-001",
        "invoice_date": date(2024, 3, 5),
        "due_date": date(2024, 4, 4),
        "supplier_name": "Acme Corp",
        "customer_name": "Globex Ltd",
        "subtotal": Decimal("100.00"),
        "tax": Decimal("10.00"),
        "tax_rate": Decimal("10"),
        "total": Decimal("110.00"),
        "currency": "USD",
        "document_type": DocumentType.INVOICE,
        "line_items": [
            LineItem(
                description="Office chairs",
                quantity=Decimal("2"),
                unit_price=Decimal("50.00"),
                amount=Decimal("100.00"),
            )
        ],
    }
    values.update(overrides)
    return InvoiceData(**values)


def seed_invoice(
    session: Session,
    document: Document,
    data: InvoiceData | None = None,
    *,
    record_index: int = 0,
    review_status: str = ReviewStatus.AUTO_APPROVED.value,
    issues: tuple[tuple[str, str, str | None], ...] = (),
    source_kind: str = SourceKind.INVOICE_DOCUMENT.value,
    confidence: float = 0.9,
    created_offset_s: int = 0,
) -> Invoice:
    """An invoice of ``document`` with ``issues`` given as ``(code, severity, field)``; committed."""
    data = data or invoice_data()
    created = BASE_TIME + timedelta(seconds=created_offset_s)
    invoice = Invoice(
        document_id=document.id,
        record_index=record_index,
        source_kind=source_kind,
        page_start=1,
        page_end=1,
        extraction_method="rules",
        overall_confidence=confidence,
        review_status=review_status,
        created_at=created,
        updated_at=created,
    )
    invoice.apply_invoice_data(data)
    invoice.set_field_meta(
        {
            name: FieldMeta(confidence=confidence, source=ExtractionSource.RULES, evidence=f"{name} evidence", page=1)
            for name in ("invoice_number", "invoice_date", "supplier_name", "total")
        }
    )
    session.add(invoice)
    session.flush()
    for code, severity, field in issues:
        session.add(
            ValidationIssueRow(
                document_id=document.id,
                invoice_id=invoice.id,
                code=code,
                severity=severity,
                field=field,
                message=f"{code} on {field}",
                details={"expected": "100.00", "actual": "120.00"} if code == "TOTAL_MISMATCH" else {},
            )
        )
    session.commit()
    return invoice


def seed_document_issue(
    session: Session, document: Document, code: str = "DUPLICATE_FILE", severity: str = "warning"
) -> None:
    session.add(
        ValidationIssueRow(
            document_id=document.id, invoice_id=None, code=code, severity=severity, message=code, details={}
        )
    )
    session.commit()


def seed_pages(session: Session, document: Document, texts: list[str], *, method: str = "text") -> None:
    for number, text in enumerate(texts, start=1):
        session.add(
            DocumentPage(
                document_id=document.id,
                page_number=number,
                method=method,
                text=text,
                char_count=len(text),
                ocr_confidence=0.91 if method == "ocr" else None,
            )
        )
    session.commit()


def seed_event(session: Session, document: Document, stage: str, message: str, level: str = "info") -> None:
    session.add(ProcessingEvent(document_id=document.id, stage=stage, level=level, message=message, data={"n": 1}))
    session.commit()


def index(session: Session, document: Document, page_texts: list[str]) -> None:
    """Index a seeded document's pages and invoices exactly as the pipeline does."""
    invoices = [(invoice.id, invoice.to_invoice_data()) for invoice in document.invoices]
    drafts = build_chunks(list(enumerate(page_texts, start=1)), invoices, document.filename)
    index_document(session, document.id, drafts, HashingEmbedder(384))
    session.commit()


def seed_cross_verification(
    session: Session, left: Invoice, right: Invoice, *, status: str = "mismatch"
) -> CrossVerification:
    row = CrossVerification(
        left_invoice_id=left.id,
        right_invoice_id=right.id,
        kind="pdf_excel",
        status=status,
        fields=[
            {"field": "total", "left": "110.00", "right": "120.00", "status": "mismatch", "note": "difference 10.00"}
        ],
        line_items={"left_count": 1, "right_count": 0, "matched": 0, "status": "missing"},
        trigger="auto",
    )
    session.add(row)
    session.commit()
    return row
