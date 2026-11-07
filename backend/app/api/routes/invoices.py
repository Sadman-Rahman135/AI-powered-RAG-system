"""Extracted records: list/filter, get, PATCH corrections, approve, correction history."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import APIRouter, Body, Query
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.api.corrections import apply_corrections, approve, lock_for_review
from app.api.deps import API_PREFIX, ActorDep, Limit, ListLimit, Offset, SessionDep, guard
from app.api.errors import NotFoundError
from app.api.queries import contains, sort_order
from app.api.schemas import (
    ApproveRequest,
    CorrectionList,
    CorrectionOut,
    ErrorResponse,
    InvoiceList,
    InvoiceOut,
    InvoicePatch,
    QueryAmount,
    QueryText,
    ReviewStatusValue,
)
from app.api.serializers import invoice_out, invoices_out
from app.db.models import Correction, Document, Invoice, ValidationIssueRow
from app.extraction.normalize import normalize_invoice_number

router = APIRouter(prefix=f"{API_PREFIX}/invoices", tags=["invoices"], dependencies=guard("default"))

InvoiceSort = Literal[
    "created_at", "-created_at", "updated_at", "-updated_at", "invoice_date", "-invoice_date", "due_date", "-due_date",
    "total", "-total", "invoice_number", "-invoice_number", "supplier_name", "-supplier_name", "customer_name",
    "-customer_name", "overall_confidence", "-overall_confidence",
]  # fmt: skip
_NOT_FOUND: dict[int | str, dict[str, Any]] = {404: {"model": ErrorResponse, "description": "No such invoice"}}
_BUSY: dict[int | str, dict[str, Any]] = {
    409: {"model": ErrorResponse, "description": "The document is queued or being processed"}
}
_TEXT = Annotated[QueryText, Query(max_length=200)]


def get_invoice_or_404(session: Session, invoice_id: UUID) -> Invoice:
    invoice = session.get(Invoice, invoice_id)
    if invoice is None:
        raise NotFoundError("Invoice not found")
    return invoice


@router.get("", summary="List and filter extracted records")
def list_invoices(
    session: SessionDep,
    q: Annotated[
        QueryText, Query(max_length=200, description="Number, supplier, customer or filename contains")
    ] = None,
    supplier: _TEXT = None,
    customer: _TEXT = None,
    invoice_number: Annotated[QueryText, Query(max_length=100)] = None,
    currency: Annotated[QueryText, Query(max_length=3)] = None,
    date_from: date | None = None,
    date_to: date | None = None,
    total_min: QueryAmount = None,
    total_max: QueryAmount = None,
    review_status: ReviewStatusValue | None = None,
    has_errors: bool | None = None,
    document_id: UUID | None = None,
    limit: Limit = 20,
    offset: Offset = 0,
    sort: InvoiceSort = "-created_at",
) -> InvoiceList:
    """Every filter is optional and they combine with AND; text filters are case-insensitive substrings
    (``invoice_number`` also matches the normalized number, so ``inv2024/1`` finds ``INV-2024-1``)."""
    statement = select(Invoice, Document.filename).join(Document, Document.id == Invoice.document_id)
    if q and q.strip():
        term = q.strip()
        statement = statement.where(
            or_(
                contains(Invoice.invoice_number, term),
                contains(Invoice.supplier_name, term),
                contains(Invoice.customer_name, term),
                contains(Document.filename, term),
            )
        )
    if supplier and supplier.strip():
        statement = statement.where(contains(Invoice.supplier_name, supplier.strip()))
    if customer and customer.strip():
        statement = statement.where(contains(Invoice.customer_name, customer.strip()))
    if invoice_number and invoice_number.strip():
        number = invoice_number.strip()
        normalized = normalize_invoice_number(number)
        condition = contains(Invoice.invoice_number, number)
        statement = statement.where(
            or_(condition, Invoice.invoice_number_norm == normalized) if normalized else condition
        )
    if currency and currency.strip():
        statement = statement.where(Invoice.currency == currency.strip().upper())
    if date_from is not None:
        statement = statement.where(Invoice.invoice_date >= date_from)
    if date_to is not None:
        statement = statement.where(Invoice.invoice_date <= date_to)
    if total_min is not None:
        statement = statement.where(Invoice.total >= total_min)
    if total_max is not None:
        statement = statement.where(Invoice.total <= total_max)
    if review_status is not None:
        statement = statement.where(Invoice.review_status == review_status)
    if document_id is not None:
        statement = statement.where(Invoice.document_id == document_id)
    if has_errors is not None:
        errors = (
            select(ValidationIssueRow.id)
            .where(ValidationIssueRow.invoice_id == Invoice.id, ValidationIssueRow.severity == "error")
            .exists()
        )
        statement = statement.where(errors if has_errors else ~errors)
    total = session.scalar(select(func.count()).select_from(statement.subquery())) or 0
    rows = session.execute(
        statement.order_by(sort_order(Invoice, sort), Invoice.document_id, Invoice.record_index, Invoice.id)
        .limit(limit)
        .offset(offset)
    ).all()
    return InvoiceList(
        items=invoices_out(session, [(invoice, filename) for invoice, filename in rows]),
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{invoice_id}", summary="Get one record", responses=_NOT_FOUND)
def get_invoice(invoice_id: UUID, session: SessionDep) -> InvoiceOut:
    return invoice_out(session, get_invoice_or_404(session, invoice_id))


@router.patch(
    "/{invoice_id}",
    summary="Correct fields of a record",
    responses={**_NOT_FOUND, **_BUSY, 422: {"model": ErrorResponse, "description": "A value is not valid"}},
)
def correct_invoice(invoice_id: UUID, patch: InvoicePatch, session: SessionDep, actor: ActorDep) -> InvoiceOut:
    """Set the given ``fields`` (``null`` clears one) and optionally replace all ``line_items``.

    Amounts are decimal strings (``"1234.50"``), dates ``YYYY-MM-DD``, currency an ISO code. Each changed field is
    recorded in the correction history with ``comment`` and the caller (``X-Actor`` header, else the API key
    fingerprint), becomes ``source: manual`` with confidence 1.0, and the record is validated again: the response
    carries the fresh issues and review status.
    """
    invoice = lock_for_review(session, invoice_id)
    apply_corrections(session, invoice, patch, actor)
    session.commit()  # also ends the transaction (and its locks) when nothing changed
    return invoice_out(session, invoice)


@router.post("/{invoice_id}/approve", summary="Approve a record after review", responses={**_NOT_FOUND, **_BUSY})
def approve_invoice(
    invoice_id: UUID, session: SessionDep, actor: ActorDep, body: Annotated[ApproveRequest | None, Body()] = None
) -> InvoiceOut:
    """Marks the record ``reviewed`` (by the caller, now) and updates the document status; approving a reviewed
    record again changes nothing."""
    invoice = lock_for_review(session, invoice_id)
    approve(session, invoice, body.comment if body else None, actor)
    session.commit()
    return invoice_out(session, invoice)


@router.get("/{invoice_id}/corrections", summary="Correction and approval history", responses=_NOT_FOUND)
def list_corrections(
    invoice_id: UUID, session: SessionDep, limit: ListLimit = 500, offset: Offset = 0
) -> CorrectionList:
    invoice = get_invoice_or_404(session, invoice_id)
    rows = session.scalars(
        select(Correction)
        .where(Correction.invoice_id == invoice.id)
        .order_by(Correction.id)
        .limit(limit)
        .offset(offset)
    ).all()
    return CorrectionList(items=[CorrectionOut.model_validate(row) for row in rows])
