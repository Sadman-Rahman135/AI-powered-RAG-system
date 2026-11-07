"""Manual cross-verification of two records, and the list of stored comparisons (automatic and manual)."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query
from sqlalchemy import or_, select
from sqlalchemy.orm import Session, aliased

from app.api.deps import API_PREFIX, ListLimit, Offset, SessionDep, guard
from app.api.errors import NotFoundError, ValidationFailedError
from app.api.schemas import (
    ComparisonStatusValue,
    CrossVerificationList,
    CrossVerificationOut,
    CrossVerifyRequest,
    ErrorResponse,
)
from app.api.serializers import cross_verifications_out
from app.db.base import utcnow
from app.db.models import CrossVerification, Document, Invoice
from app.ingestion.types import FileKind
from app.pipeline.progress import as_utc
from app.validation.cross_verify import compare_invoices, comparison_kind

router = APIRouter(prefix=API_PREFIX, tags=["cross-verification"], dependencies=guard("default"))
TRIGGER_MANUAL = "manual"


def _pair_by_documents(session: Session, left_id: UUID, right_id: UUID) -> list[tuple[Invoice, Invoice]]:
    """Records of two documents paired by normalized invoice number; with no shared number, a document holding a
    single record on each side is compared directly."""
    if left_id == right_id:
        raise ValidationFailedError("Choose two different documents")
    records: dict[UUID, list[Invoice]] = {}
    for document_id in (left_id, right_id):
        if session.get(Document, document_id) is None:
            raise NotFoundError("Document not found", details={"document_id": str(document_id)})
        records[document_id] = list(
            session.scalars(
                select(Invoice).where(Invoice.document_id == document_id).order_by(Invoice.record_index, Invoice.id)
            )
        )
    left, right = records[left_id], records[right_id]
    by_number: dict[str, Invoice] = {}
    for invoice in right:
        if invoice.invoice_number_norm:
            by_number.setdefault(invoice.invoice_number_norm, invoice)
    pairs = [
        (invoice, by_number[invoice.invoice_number_norm])
        for invoice in left
        if invoice.invoice_number_norm and invoice.invoice_number_norm in by_number
    ]
    if pairs:
        return pairs
    if len(left) == 1 and len(right) == 1:
        return [(left[0], right[0])]
    raise ValidationFailedError(
        "The documents share no invoice number; choose the two records to compare by invoice id",
        details={"left_records": len(left), "right_records": len(right)},
    )


def _pair_by_invoices(session: Session, left_id: UUID, right_id: UUID) -> tuple[Invoice, Invoice]:
    if left_id == right_id:
        raise ValidationFailedError("Choose two different records")
    pair = []
    for invoice_id in (left_id, right_id):
        invoice = session.get(Invoice, invoice_id)
        if invoice is None:
            raise NotFoundError("Invoice not found", details={"invoice_id": str(invoice_id)})
        pair.append(invoice)
    return pair[0], pair[1]


def _upload_order(session: Session, invoice: Invoice) -> tuple[datetime | None, str, int, str]:
    """The pipeline's pair order: the record of the earlier-uploaded document is ``left``."""
    document = session.get_one(Document, invoice.document_id)
    return as_utc(document.created_at), str(document.id), invoice.record_index, str(invoice.id)


def _compare_and_store(session: Session, first: Invoice, second: Invoice) -> CrossVerification:
    """Compare a pair and store the result in the pair's one row (unique per pair, whichever way round).

    An existing row - automatic or manual - is updated in place and keeps its orientation; a new pair is oriented
    like the pipeline orients it (earlier upload on the left). Either way the row becomes ``manual``, which the
    pipeline keeps (and refreshes) when it re-validates the records.
    """
    row = session.scalars(
        select(CrossVerification).where(
            or_(
                (CrossVerification.left_invoice_id == first.id) & (CrossVerification.right_invoice_id == second.id),
                (CrossVerification.left_invoice_id == second.id) & (CrossVerification.right_invoice_id == first.id),
            )
        )
    ).first()
    if row is not None:
        left, right = (first, second) if row.left_invoice_id == first.id else (second, first)
    else:
        left, right = sorted((first, second), key=lambda invoice: _upload_order(session, invoice))
        row = CrossVerification(left_invoice_id=left.id, right_invoice_id=right.id)
        session.add(row)
    kinds = [FileKind(session.get_one(Document, invoice.document_id).file_kind) for invoice in (left, right)]
    result = compare_invoices(left.to_invoice_data(), right.to_invoice_data()).to_dict()
    row.kind = comparison_kind(kinds[0], kinds[1])
    row.status = result["status"]
    row.fields = result["fields"]
    row.line_items = result["line_items"]
    row.trigger = TRIGGER_MANUAL
    row.created_at = utcnow()  # when this comparison was made
    session.flush()
    return row


@router.post(
    "/cross-verify",
    summary="Compare two records field by field",
    responses={
        404: {"model": ErrorResponse, "description": "An id does not exist"},
        422: {"model": ErrorResponse, "description": "Not exactly one pair of ids, or nothing to pair"},
    },
)
def cross_verify(body: CrossVerifyRequest, session: SessionDep) -> CrossVerificationOut:
    """Compare two records given by invoice id, or two documents given by document id.

    Documents are paired by normalized invoice number (a single record on each side is compared directly). When
    the documents share several invoice numbers, every pair is compared and stored and the response is the first
    pair in record order; ``GET /cross-verifications?document_id=...`` lists them all. A pair has one stored
    comparison: asking again (either way round) updates it, and ``left`` is the earlier upload unless the pair's
    row already had the other orientation.
    """
    if body.left_invoice_id is not None and body.right_invoice_id is not None:
        pairs = [_pair_by_invoices(session, body.left_invoice_id, body.right_invoice_id)]
    else:
        assert body.left_document_id is not None and body.right_document_id is not None
        pairs = _pair_by_documents(session, body.left_document_id, body.right_document_id)
    rows = [_compare_and_store(session, left, right) for left, right in pairs]
    session.commit()
    return cross_verifications_out(session, rows[:1])[0]


@router.get("/cross-verifications", summary="List stored comparisons")
def list_cross_verifications(
    session: SessionDep,
    invoice_id: UUID | None = None,
    document_id: UUID | None = None,
    status_filter: Annotated[ComparisonStatusValue | None, Query(alias="status")] = None,
    limit: ListLimit = 500,
    offset: Offset = 0,
) -> CrossVerificationList:
    """Newest first; ``invoice_id`` / ``document_id`` match either side."""
    statement = select(CrossVerification)
    if invoice_id is not None:
        statement = statement.where(
            or_(CrossVerification.left_invoice_id == invoice_id, CrossVerification.right_invoice_id == invoice_id)
        )
    if document_id is not None:
        left, right = aliased(Invoice), aliased(Invoice)
        statement = (
            statement.join(left, left.id == CrossVerification.left_invoice_id)
            .join(right, right.id == CrossVerification.right_invoice_id)
            .where(or_(left.document_id == document_id, right.document_id == document_id))
        )
    if status_filter is not None:
        statement = statement.where(CrossVerification.status == status_filter)
    rows = session.scalars(
        statement.order_by(CrossVerification.created_at.desc(), CrossVerification.id).limit(limit).offset(offset)
    ).all()
    return CrossVerificationList(items=cross_verifications_out(session, rows))
