"""ORM rows -> API response models, with the related data each needs loaded in batches (a fixed number of queries
per page, never one per row)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.schemas import (
    CrossVerificationOut,
    CrossVerifySide,
    DocumentDetailOut,
    DocumentOut,
    ExtractionInfo,
    FieldComparisonOut,
    FieldMetaOut,
    InvoiceOut,
    IssueCounts,
    IssueOut,
    JobOut,
    LineItemComparisonOut,
    LineItemOut,
)
from app.db.models import CrossVerification, Document, Invoice, ProcessingJob, ValidationIssueRow


def job_out(job: ProcessingJob | None) -> JobOut | None:
    return JobOut.model_validate(job) if job is not None else None


def latest_jobs(session: Session, document_ids: Sequence[UUID]) -> dict[UUID, ProcessingJob]:
    """The most recent job of each document."""
    latest: dict[UUID, ProcessingJob] = {}
    if not document_ids:
        return latest
    statement = (
        select(ProcessingJob)
        .where(ProcessingJob.document_id.in_(document_ids))
        .order_by(ProcessingJob.created_at, ProcessingJob.id)
    )
    for job in session.scalars(statement):
        latest[job.document_id] = job
    return latest


def issue_out(row: ValidationIssueRow) -> IssueOut:
    return IssueOut.model_validate(row)


def _issue_counts(session: Session, document_ids: Sequence[UUID]) -> dict[UUID, IssueCounts]:
    counts: dict[UUID, IssueCounts] = defaultdict(IssueCounts)
    if not document_ids:
        return counts
    statement = (
        select(ValidationIssueRow.document_id, ValidationIssueRow.severity, func.count())
        .where(ValidationIssueRow.document_id.in_(document_ids))
        .group_by(ValidationIssueRow.document_id, ValidationIssueRow.severity)
    )
    for document_id, severity, count in session.execute(statement):
        if severity in IssueCounts.model_fields:
            setattr(counts[document_id], severity, count)
    return counts


def _invoice_counts(session: Session, document_ids: Sequence[UUID]) -> dict[UUID, int]:
    if not document_ids:
        return {}
    statement = (
        select(Invoice.document_id, func.count())
        .where(Invoice.document_id.in_(document_ids))
        .group_by(Invoice.document_id)
    )
    return {document_id: count for document_id, count in session.execute(statement)}


def _document_fields(document: Document) -> dict[str, Any]:
    return {name: getattr(document, name) for name in DocumentOut.model_fields if hasattr(Document, name)}


def documents_out(session: Session, documents: Sequence[Document]) -> list[DocumentOut]:
    ids = [document.id for document in documents]
    jobs, invoices, issues = latest_jobs(session, ids), _invoice_counts(session, ids), _issue_counts(session, ids)
    return [
        DocumentOut(
            **_document_fields(document),
            invoice_count=invoices.get(document.id, 0),
            issue_counts=issues.get(document.id, IssueCounts()),
            job=job_out(jobs.get(document.id)),
        )
        for document in documents
    ]


def _issues_by_invoice(session: Session, invoice_ids: Sequence[UUID]) -> dict[UUID, list[IssueOut]]:
    grouped: dict[UUID, list[IssueOut]] = defaultdict(list)
    if not invoice_ids:
        return grouped
    statement = (
        select(ValidationIssueRow).where(ValidationIssueRow.invoice_id.in_(invoice_ids)).order_by(ValidationIssueRow.id)
    )
    for row in session.scalars(statement):
        assert row.invoice_id is not None
        grouped[row.invoice_id].append(issue_out(row))
    return grouped


def invoices_out(session: Session, rows: Sequence[tuple[Invoice, str]]) -> list[InvoiceOut]:
    """``rows`` are ``(invoice, filename of its document)``; issues are read fresh from the database."""
    issues = _issues_by_invoice(session, [invoice.id for invoice, _ in rows])
    return [_invoice_out(invoice, filename, issues.get(invoice.id, [])) for invoice, filename in rows]


def invoice_out(session: Session, invoice: Invoice) -> InvoiceOut:
    filename = session.scalar(select(Document.filename).where(Document.id == invoice.document_id)) or ""
    return invoices_out(session, [(invoice, filename)])[0]


def _invoice_out(invoice: Invoice, filename: str, issues: list[IssueOut]) -> InvoiceOut:
    data = invoice.to_invoice_data()  # money with storage padding stripped ("110.0000" -> "110.00")
    return InvoiceOut(
        id=invoice.id,
        document_id=invoice.document_id,
        filename=filename,
        record_index=invoice.record_index,
        source_kind=invoice.source_kind,
        page_start=invoice.page_start,
        page_end=invoice.page_end,
        sheet_name=invoice.sheet_name,
        row_number=invoice.row_number,
        invoice_number=data.invoice_number,
        invoice_date=data.invoice_date,
        due_date=data.due_date,
        supplier_name=data.supplier_name,
        customer_name=data.customer_name,
        subtotal=data.subtotal,
        discount=data.discount,
        shipping=data.shipping,
        tax=data.tax,
        tax_rate=data.tax_rate,
        total=data.total,
        currency=data.currency,
        document_type=data.document_type.value,
        line_items=[LineItemOut.model_validate(item.model_dump()) for item in data.line_items],
        field_meta={name: FieldMetaOut.model_validate(meta) for name, meta in (invoice.field_meta or {}).items()},
        extraction_method=invoice.extraction_method,
        overall_confidence=invoice.overall_confidence,
        review_status=invoice.review_status,
        issues=issues,
        created_at=invoice.created_at,
        updated_at=invoice.updated_at,
        reviewed_at=invoice.reviewed_at,
        reviewed_by=invoice.reviewed_by,
    )


def document_detail_out(session: Session, document: Document) -> DocumentDetailOut:
    (summary,) = documents_out(session, [document])
    invoices = session.scalars(
        select(Invoice).where(Invoice.document_id == document.id).order_by(Invoice.record_index, Invoice.id)
    ).all()
    document_issues = session.scalars(
        select(ValidationIssueRow)
        .where(ValidationIssueRow.document_id == document.id, ValidationIssueRow.invoice_id.is_(None))
        .order_by(ValidationIssueRow.id)
    ).all()
    return DocumentDetailOut(
        **summary.model_dump(),
        invoices=invoices_out(session, [(invoice, document.filename) for invoice in invoices]),
        document_issues=[issue_out(row) for row in document_issues],
        extraction=ExtractionInfo(
            method=document.extraction_method,
            llm_provider=document.llm_provider,
            llm_model=document.llm_model,
            prompt_version=document.prompt_version,
        ),
    )


def cross_verifications_out(session: Session, rows: Sequence[CrossVerification]) -> list[CrossVerificationOut]:
    invoice_ids = {row.left_invoice_id for row in rows} | {row.right_invoice_id for row in rows}
    sides: dict[UUID, CrossVerifySide] = {}
    if invoice_ids:
        statement = (
            select(Invoice.id, Invoice.document_id, Invoice.invoice_number, Document.filename)
            .join(Document, Document.id == Invoice.document_id)
            .where(Invoice.id.in_(invoice_ids))
        )
        for invoice_id, document_id, number, filename in session.execute(statement):
            sides[invoice_id] = CrossVerifySide(document_id=document_id, filename=filename, invoice_number=number)
    return [
        CrossVerificationOut(
            id=row.id,
            left_invoice_id=row.left_invoice_id,
            right_invoice_id=row.right_invoice_id,
            left=sides[row.left_invoice_id],
            right=sides[row.right_invoice_id],
            kind=row.kind,
            status=row.status,
            fields=[FieldComparisonOut.model_validate(field) for field in row.fields or []],
            line_items=LineItemComparisonOut.model_validate(row.line_items or {}),
            trigger=row.trigger,
            created_at=row.created_at,
        )
        for row in rows
    ]
