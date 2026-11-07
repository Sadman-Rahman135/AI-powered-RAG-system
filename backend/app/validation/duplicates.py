"""Duplicate detection: the same invoice showing up in another document.

``find_duplicate_candidates`` looks up records carrying the same normalized invoice number in *other* documents;
``duplicate_issues`` decides per candidate whether it is a duplicate. Spreadsheet ledger rows are never duplicates of
an invoice document: they are the counterparts that cross-verification compares against.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Invoice
from app.domain.schemas import DEFAULT_SEVERITY, InvoiceData, IssueCode, Severity, SourceKind, ValidationIssue
from app.extraction.normalize import amounts_equal, names_match

__all__ = ["duplicate_issues", "find_duplicate_candidates"]

_MESSAGES = {
    IssueCode.DUPLICATE_INVOICE: "Another document contains an invoice with the same number, supplier and total",
    IssueCode.POSSIBLE_DUPLICATE: (
        "Another document contains an invoice with the same number and supplier but a different or unknown total"
    ),
}
_SEVERITY_ORDER = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}


def find_duplicate_candidates(
    session: Session, *, invoice_number_norm: str, exclude_document_id: UUID
) -> list[Invoice]:
    """Records with this normalized invoice number in documents other than ``exclude_document_id``.

    Any source kind is returned (ledger rows included); ``duplicate_issues`` and the caller decide what counts.
    """
    if not invoice_number_norm:
        return []  # a number made only of punctuation identifies nothing
    statement = (
        select(Invoice)
        .where(Invoice.invoice_number_norm == invoice_number_norm, Invoice.document_id != exclude_document_id)
        .order_by(Invoice.created_at, Invoice.id)
    )
    return list(session.scalars(statement))


def duplicate_issues(
    data: InvoiceData,
    candidates: Sequence[Invoice],
    *,
    source_kind: SourceKind = SourceKind.INVOICE_DOCUMENT,
    tolerance: Decimal | None = None,
) -> list[ValidationIssue]:
    """One issue per candidate that looks like the same invoice, most severe first.

    Same supplier and same total: DUPLICATE_INVOICE (error). Same supplier but a different or unknown total:
    POSSIBLE_DUPLICATE (warning). Clearly different suppliers: nothing, invoice numbers repeat across suppliers.
    A missing supplier on either side does not rule a candidate out. ``source_kind`` is the kind of record ``data``
    comes from: a ledger row is never a duplicate (it is checked against invoice documents by cross-verification).
    ``tolerance`` is the absolute money tolerance (default: the ``AMOUNT_TOLERANCE`` setting).
    """
    if source_kind == SourceKind.LEDGER_ROW:
        return []
    if tolerance is None:
        tolerance = get_settings().amount_tolerance
    issues: list[ValidationIssue] = []
    for candidate in candidates:
        if candidate.source_kind == SourceKind.LEDGER_ROW or not _same_supplier(
            data.supplier_name, candidate.supplier_name
        ):
            continue
        same_total = (
            data.total is not None
            and candidate.total is not None
            and amounts_equal(data.total, candidate.total, tolerance)
        )
        code = IssueCode.DUPLICATE_INVOICE if same_total else IssueCode.POSSIBLE_DUPLICATE
        issues.append(
            ValidationIssue(
                code=code,
                severity=DEFAULT_SEVERITY[code],
                field="invoice_number",
                message=_MESSAGES[code],
                details={"other_invoice_id": str(candidate.id), "other_document_id": str(candidate.document_id)},
            )
        )
    return sorted(issues, key=lambda issue: _SEVERITY_ORDER[issue.severity])


def _same_supplier(left: str | None, right: str | None) -> bool:
    """The same supplier, or not provably different: an unknown supplier on either side counts as the same."""
    return _is_blank(left) or _is_blank(right) or names_match(left, right)


def _is_blank(value: str | None) -> bool:
    return value is None or not value.strip()
