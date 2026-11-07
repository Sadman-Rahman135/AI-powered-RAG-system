"""Manual corrections and approvals of extracted records, with their audit trail.

``lock_for_review`` opens every review write: it takes the reconciliation lock first in the transaction (the lock
order the pipeline uses, so a correction and a document being processed cannot deadlock), then loads the record
with a row lock (``SELECT ... FOR UPDATE``), and refuses the write (409 ``already_processing``) while the record's
document is queued or being processed - processing replaces its records.

``apply_corrections`` writes one ``Correction`` per field whose value really changes (old and new values as JSON,
the comment, who), marks those fields ``FieldMeta(confidence=1.0, source="manual")``, refreshes the overall
confidence and re-validates the record (issues, review status, document status). ``approve`` marks a record
``reviewed`` and records the approval in the same audit trail (field ``review_status``), so an approval comment is
kept. None of them commits; callers do.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.api.errors import AlreadyProcessingError, NotFoundError
from app.api.schemas import EDITABLE_FIELDS, InvoicePatch
from app.db.models import Correction, Document, DocumentStatus, Invoice, ProcessingEvent
from app.domain.schemas import DocumentType, ExtractionResult, ExtractionSource, FieldMeta, InvoiceData, ReviewStatus
from app.pipeline.review import refresh_document_status, revalidate_invoice, serialize_reconciliation

REVIEW_STAGE = "review"
_BUSY_STATUSES = (DocumentStatus.QUEUED, DocumentStatus.PROCESSING)


def locked_invoice(invoice_id: UUID) -> Select[Invoice]:
    """The record, locked for this transaction (``FOR UPDATE`` on PostgreSQL; SQLite has one writer anyway)."""
    return select(Invoice).where(Invoice.id == invoice_id).with_for_update()


def lock_for_review(session: Session, invoice_id: UUID) -> Invoice:
    """Begin a review write: reconciliation lock, then the record's row lock, then the processing check."""
    serialize_reconciliation(session)  # must be the transaction's first statement (lock order)
    invoice = session.scalars(locked_invoice(invoice_id)).one_or_none()
    if invoice is None:
        raise NotFoundError("Invoice not found")
    status = session.scalar(select(Document.status).where(Document.id == invoice.document_id))
    if status in _BUSY_STATUSES:
        raise AlreadyProcessingError(
            "The document is being processed; correct or approve its records when processing has finished"
        )
    return invoice


def _requested_changes(patch: InvoicePatch) -> dict[str, Any]:
    """The values the patch sets, in display order (``document_type: null`` means ``unknown``)."""
    provided = patch.fields.model_fields_set
    changes: dict[str, Any] = {}
    for name in EDITABLE_FIELDS:
        if name in provided:
            value = getattr(patch.fields, name)
            changes[name] = DocumentType.UNKNOWN if name == "document_type" and value is None else value
    if patch.line_items is not None:
        changes["line_items"] = [item.model_dump() for item in patch.line_items]
    return changes


def _value(data: InvoiceData, name: str) -> Any:
    """A comparable value: decimals compare numerically ("110.0" equals "110.00"), line items field by field."""
    if name == "line_items":
        return [item.model_dump() for item in data.line_items]
    return getattr(data, name)


def apply_corrections(session: Session, invoice: Invoice, patch: InvoicePatch, actor: str) -> list[Correction]:
    """Apply ``patch`` to ``invoice``; returns the corrections written (empty when nothing actually changed)."""
    current = invoice.to_invoice_data()
    corrected = InvoiceData.model_validate({**current.model_dump(), **_requested_changes(patch)})
    changed = [name for name in (*EDITABLE_FIELDS, "line_items") if _value(current, name) != _value(corrected, name)]
    if not changed:
        return []

    before, after = current.model_dump(mode="json"), corrected.model_dump(mode="json")
    corrections = [
        Correction(
            document_id=invoice.document_id,
            invoice_id=invoice.id,
            field=name,
            old_value=before[name],
            new_value=after[name],
            comment=patch.comment,
            corrected_by=actor,
        )
        for name in changed
    ]
    session.add_all(corrections)

    invoice.apply_invoice_data(corrected)
    meta = invoice.field_meta_models()
    for name in changed:
        meta[name] = FieldMeta(confidence=1.0, source=ExtractionSource.MANUAL)
    invoice.set_field_meta(meta)
    invoice.overall_confidence = ExtractionResult(data=corrected, meta=meta).compute_overall_confidence()
    session.add(
        ProcessingEvent(
            document_id=invoice.document_id,
            stage=REVIEW_STAGE,
            message=f"{len(changed)} field(s) corrected",
            data={"invoice_id": str(invoice.id), "fields": changed, "by": actor},
        )
    )
    session.flush()
    revalidate_invoice(session, invoice)
    return corrections


def approve(session: Session, invoice: Invoice, comment: str | None, actor: str) -> bool:
    """Mark ``invoice`` reviewed by ``actor``; returns False (and changes nothing) when it already is."""
    if invoice.review_status == ReviewStatus.REVIEWED:
        return False
    session.add(
        Correction(
            document_id=invoice.document_id,
            invoice_id=invoice.id,
            field="review_status",
            old_value=invoice.review_status,
            new_value=ReviewStatus.REVIEWED.value,
            comment=comment,
            corrected_by=actor,
        )
    )
    invoice.review_status = ReviewStatus.REVIEWED.value
    invoice.reviewed_at = datetime.now(UTC)
    invoice.reviewed_by = actor
    session.add(
        ProcessingEvent(
            document_id=invoice.document_id,
            stage=REVIEW_STAGE,
            message="Record approved",
            data={"invoice_id": str(invoice.id), "by": actor},
        )
    )
    session.flush()
    refresh_document_status(session, session.get_one(Document, invoice.document_id))
    return True
