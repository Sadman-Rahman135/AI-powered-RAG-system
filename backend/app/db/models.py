"""ORM models. See docs/DATABASE.md for the ER diagram and index rationale.

PostgreSQL-only extras (created by the Alembic migration, not mapped here): ``chunks.search_tsv`` generated
tsvector + GIN index, HNSW index on ``chunks.embedding``, trigram GIN indexes on invoice party names.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import numpy as np
from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import get_settings
from app.db.base import Base, BigIntPK, JSONType, utcnow
from app.db.types import EmbeddingVector
from app.domain.schemas import (
    DocumentType,
    ExtractionSource,
    FieldMeta,
    InvoiceData,
    LineItem,
    ReviewStatus,
    SourceKind,
)

TZDateTime = DateTime(timezone=True)
MoneyColumn = Numeric(18, 4)


class DocumentStatus:
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"
    ALL = (QUEUED, PROCESSING, COMPLETED, NEEDS_REVIEW, FAILED)


class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ALL = (QUEUED, RUNNING, SUCCEEDED, FAILED)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    file_kind: Mapped[str] = mapped_column(String(10), index=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    storage_key: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(20), default=DocumentStatus.QUEUED, index=True)
    page_count: Mapped[int | None] = mapped_column(Integer)
    ocr_page_count: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(50))
    error_message: Mapped[str | None] = mapped_column(Text)
    processing_ms: Mapped[int | None] = mapped_column(Integer)
    duplicate_of_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("documents.id", ondelete="SET NULL"), index=True
    )
    extraction_method: Mapped[str | None] = mapped_column(String(20))
    llm_provider: Mapped[str | None] = mapped_column(String(50))
    llm_model: Mapped[str | None] = mapped_column(String(100))
    prompt_version: Mapped[str | None] = mapped_column(String(40))
    warnings: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(TZDateTime, default=utcnow, onupdate=utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(TZDateTime)

    pages: Mapped[list[DocumentPage]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="DocumentPage.page_number",
    )
    invoices: Mapped[list[Invoice]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True, order_by="Invoice.record_index"
    )
    jobs: Mapped[list[ProcessingJob]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ProcessingJob.created_at",
    )
    events: Mapped[list[ProcessingEvent]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True, order_by="ProcessingEvent.id"
    )

    __table_args__ = (Index("ix_documents_status_created", "status", "created_at"),)


class DocumentPage(Base):
    """Extracted text per PDF page / image frame / spreadsheet sheet."""

    __tablename__ = "document_pages"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    document_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page_number: Mapped[int] = mapped_column(Integer)  # 1-based; sheets use their 1-based index
    label: Mapped[str | None] = mapped_column(String(255))  # sheet name
    method: Mapped[str] = mapped_column(String(10))  # text | ocr | sheet | none
    text: Mapped[str] = mapped_column(Text, default="")
    ocr_confidence: Mapped[float | None] = mapped_column(Float)
    char_count: Mapped[int] = mapped_column(Integer, default=0)

    document: Mapped[Document] = relationship(back_populates="pages")

    __table_args__ = (UniqueConstraint("document_id", "page_number"),)


class Invoice(Base):
    """One extracted record: an invoice document or one spreadsheet ledger row."""

    __tablename__ = "invoices"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    record_index: Mapped[int] = mapped_column(Integer, default=0)
    source_kind: Mapped[str] = mapped_column(String(20), default=SourceKind.INVOICE_DOCUMENT.value)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    sheet_name: Mapped[str | None] = mapped_column(String(255))
    row_number: Mapped[int | None] = mapped_column(Integer)

    invoice_number: Mapped[str | None] = mapped_column(String(100))
    invoice_number_norm: Mapped[str | None] = mapped_column(String(100), index=True)
    invoice_date: Mapped[date | None] = mapped_column(Date, index=True)
    due_date: Mapped[date | None] = mapped_column(Date)
    supplier_name: Mapped[str | None] = mapped_column(String(255))
    supplier_name_norm: Mapped[str | None] = mapped_column(String(255), index=True)
    customer_name: Mapped[str | None] = mapped_column(String(255))
    subtotal: Mapped[Decimal | None] = mapped_column(MoneyColumn)
    discount: Mapped[Decimal | None] = mapped_column(MoneyColumn)
    shipping: Mapped[Decimal | None] = mapped_column(MoneyColumn)
    tax: Mapped[Decimal | None] = mapped_column(MoneyColumn)
    tax_rate: Mapped[Decimal | None] = mapped_column(Numeric(9, 4))
    total: Mapped[Decimal | None] = mapped_column(MoneyColumn, index=True)
    currency: Mapped[str | None] = mapped_column(String(3), index=True)
    document_type: Mapped[str] = mapped_column(String(20), default=DocumentType.UNKNOWN.value)
    line_items: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    field_meta: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)

    extraction_method: Mapped[str] = mapped_column(String(20), default="rules")
    overall_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    review_status: Mapped[str] = mapped_column(String(20), default=ReviewStatus.NEEDS_REVIEW.value, index=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    reviewed_by: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(TZDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(TZDateTime, default=utcnow, onupdate=utcnow)

    document: Mapped[Document] = relationship(back_populates="invoices")
    issues: Mapped[list[ValidationIssueRow]] = relationship(
        back_populates="invoice", cascade="all, delete-orphan", passive_deletes=True, order_by="ValidationIssueRow.id"
    )

    __table_args__ = (Index("ix_invoices_number_supplier", "invoice_number_norm", "supplier_name_norm"),)

    # --- conversions -----------------------------------------------------------------------------------------
    def to_invoice_data(self) -> InvoiceData:
        return InvoiceData(
            invoice_number=self.invoice_number,
            invoice_date=self.invoice_date,
            due_date=self.due_date,
            supplier_name=self.supplier_name,
            customer_name=self.customer_name,
            subtotal=_dec(self.subtotal),
            discount=_dec(self.discount),
            shipping=_dec(self.shipping),
            tax=_dec(self.tax),
            tax_rate=_dec(self.tax_rate),
            total=_dec(self.total),
            currency=self.currency,
            document_type=DocumentType(self.document_type or DocumentType.UNKNOWN.value),
            line_items=[LineItem.model_validate(li) for li in (self.line_items or [])],
        )

    def apply_invoice_data(self, data: InvoiceData) -> None:
        """Copy values onto the typed columns and refresh the normalized lookup keys."""
        from app.extraction.normalize import normalize_invoice_number, normalize_name

        self.invoice_number = data.invoice_number
        self.invoice_number_norm = normalize_invoice_number(data.invoice_number) if data.invoice_number else None
        self.invoice_date = data.invoice_date
        self.due_date = data.due_date
        self.supplier_name = data.supplier_name
        self.supplier_name_norm = normalize_name(data.supplier_name) if data.supplier_name else None
        self.customer_name = data.customer_name
        self.subtotal = data.subtotal
        self.discount = data.discount
        self.shipping = data.shipping
        self.tax = data.tax
        self.tax_rate = data.tax_rate
        self.total = data.total
        self.currency = data.currency
        self.document_type = data.document_type.value
        self.line_items = [li.model_dump(mode="json") for li in data.line_items]

    def field_meta_models(self) -> dict[str, FieldMeta]:
        return {k: FieldMeta.model_validate(v) for k, v in (self.field_meta or {}).items()}

    def set_field_meta(self, meta: dict[str, FieldMeta]) -> None:
        self.field_meta = {k: v.model_dump(mode="json") for k, v in meta.items()}


def _dec(value: Decimal | None) -> Decimal | None:
    """Strip storage padding (Numeric(18,4) returns 110.0000) to the printed scale, min 2 decimals."""
    if value is None:
        return None
    d = Decimal(value)
    normalized = d.normalize()
    exponent = normalized.as_tuple().exponent
    if isinstance(exponent, int) and exponent > -2:
        return d.quantize(Decimal("0.01"))
    return normalized


class ValidationIssueRow(Base):
    __tablename__ = "validation_issues"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    document_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    invoice_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("invoices.id", ondelete="CASCADE"), index=True
    )
    code: Mapped[str] = mapped_column(String(40), index=True)
    severity: Mapped[str] = mapped_column(String(10), index=True)
    field: Mapped[str | None] = mapped_column(String(40))
    message: Mapped[str] = mapped_column(Text)
    details: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, default=utcnow)

    invoice: Mapped[Invoice | None] = relationship(back_populates="issues")


class Correction(Base):
    """Audit trail of manual corrections; survives re-processing (invoice link is set NULL)."""

    __tablename__ = "corrections"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    document_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    invoice_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("invoices.id", ondelete="SET NULL"), index=True
    )
    field: Mapped[str] = mapped_column(String(40))
    old_value: Mapped[Any] = mapped_column(JSONType, nullable=True)
    new_value: Mapped[Any] = mapped_column(JSONType, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text)
    corrected_by: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(TZDateTime, default=utcnow)


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(20), default=JobStatus.QUEUED, index=True)
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    stage: Mapped[str | None] = mapped_column(String(20))
    progress_current: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    worker_id: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(TZDateTime, default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    heartbeat_at: Mapped[datetime | None] = mapped_column(TZDateTime)

    document: Mapped[Document] = relationship(back_populates="jobs")


class ProcessingEvent(Base):
    """Processing history: one row per stage transition / notable event."""

    __tablename__ = "processing_events"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    document_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("processing_jobs.id", ondelete="SET NULL"))
    stage: Mapped[str] = mapped_column(String(20))
    level: Mapped[str] = mapped_column(String(10), default="info")
    message: Mapped[str] = mapped_column(Text)
    data: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, default=utcnow, index=True)

    document: Mapped[Document] = relationship(back_populates="events")


class Chunk(Base):
    """Searchable text unit (page text window or invoice summary) with its embedding."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    document_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    invoice_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("invoices.id", ondelete="CASCADE"), index=True
    )
    page_number: Mapped[int | None] = mapped_column(Integer)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    kind: Mapped[str] = mapped_column(String(20), default="page_text")  # page_text | invoice_summary | sheet
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[np.ndarray | None] = mapped_column(EmbeddingVector(get_settings().embedding_dim), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, default=utcnow)


class CrossVerification(Base):
    __tablename__ = "cross_verifications"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    left_invoice_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("invoices.id", ondelete="CASCADE"), index=True)
    right_invoice_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("invoices.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(20))  # pdf_pdf | pdf_excel | other
    status: Mapped[str] = mapped_column(String(20), index=True)  # match | mismatch | partial
    fields: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    line_items: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    trigger: Mapped[str] = mapped_column(String(10), default="auto")  # auto | manual
    created_at: Mapped[datetime] = mapped_column(TZDateTime, default=utcnow)

    left_invoice: Mapped[Invoice] = relationship(foreign_keys=[left_invoice_id])
    right_invoice: Mapped[Invoice] = relationship(foreign_keys=[right_invoice_id])

    __table_args__ = (UniqueConstraint("left_invoice_id", "right_invoice_id"),)


ALL_TABLES = (
    "documents",
    "document_pages",
    "invoices",
    "validation_issues",
    "corrections",
    "processing_jobs",
    "processing_events",
    "chunks",
    "cross_verifications",
)

__all__ = [
    "ALL_TABLES",
    "Chunk",
    "Correction",
    "CrossVerification",
    "Document",
    "DocumentPage",
    "DocumentStatus",
    "ExtractionSource",
    "Invoice",
    "JobStatus",
    "ProcessingEvent",
    "ProcessingJob",
    "ValidationIssueRow",
]
