"""Cross-verification: field-by-field comparison of two records of the same invoice.

Records that share a normalized invoice number across documents (a PDF and its scan, an invoice PDF and a ledger
row, ...) are compared on invoice number (normalized equality), invoice date (exact), supplier and customer
(fuzzy names), subtotal / tax / total (within tolerance) and currency (exact). Line items are compared as
multisets of their amounts in cents, but only differing items count against the record: a side without line
items (a spreadsheet ledger row, say) leaves the overall status to the fields. Values come back as JSON-friendly
strings so a result can be stored and served as is (``to_dict``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from functools import partial
from typing import Any, Literal
from uuid import UUID

from app.config import get_settings
from app.domain.schemas import DEFAULT_SEVERITY, InvoiceData, IssueCode, ValidationIssue
from app.extraction.normalize import (
    amounts_equal,
    names_match,
    normalize_invoice_number,
    normalize_name,
    quantize_money,
)
from app.ingestion.types import FileKind

__all__ = [
    "ComparisonResult",
    "FieldComparison",
    "LineItemComparison",
    "compare_invoices",
    "comparison_kind",
    "cross_verification_issue",
]

FieldStatus = Literal["match", "mismatch", "missing"]
ComparisonStatus = Literal["match", "mismatch", "partial"]
ComparisonKind = Literal["pdf_pdf", "pdf_excel", "other"]

_VISUAL_KINDS = frozenset({FileKind.PDF, FileKind.IMAGE})
_SPREADSHEET_KINDS = frozenset({FileKind.XLSX, FileKind.XLS, FileKind.CSV})

# A comparison returns whether the two values agree and an optional human-readable note.
_Comparison = Callable[[Any, Any], tuple[bool, str | None]]


@dataclass(frozen=True, slots=True)
class FieldComparison:
    """One compared field; ``left`` / ``right`` are display strings, ``None`` when that side has no value."""

    field: str
    left: str | None
    right: str | None
    status: FieldStatus
    note: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {"field": self.field, "left": self.left, "right": self.right, "status": self.status, "note": self.note}


@dataclass(frozen=True, slots=True)
class LineItemComparison:
    """Line items as multisets of amounts in cents; the counts cover the items that carry an amount."""

    left_count: int
    right_count: int
    matched: int
    status: FieldStatus

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_count": self.left_count,
            "right_count": self.right_count,
            "matched": self.matched,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """The outcome of comparing two records.

    ``mismatch`` if any field or the line items differ, else ``partial`` if a field is missing on a side, else
    ``match``. A line-item status of ``missing`` (a side has no items) does not make a result ``partial``.
    """

    status: ComparisonStatus
    fields: list[FieldComparison]
    line_items: LineItemComparison

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "fields": [comparison.to_dict() for comparison in self.fields],
            "line_items": self.line_items.to_dict(),
        }


def compare_invoices(left: InvoiceData, right: InvoiceData, *, tolerance: Decimal | None = None) -> ComparisonResult:
    """Compare two records field by field.

    ``tolerance`` is the absolute money tolerance (default: the ``AMOUNT_TOLERANCE`` setting).
    """
    if tolerance is None:
        tolerance = get_settings().amount_tolerance
    amount = partial(_same_amount, tolerance=tolerance)
    comparisons: tuple[tuple[str, _Comparison], ...] = (
        ("invoice_number", _same_number),
        ("invoice_date", _same_date),
        ("supplier_name", _same_name),
        ("customer_name", _same_name),
        ("subtotal", amount),
        ("tax", amount),
        ("total", amount),
        ("currency", _same_currency),
    )
    fields = [_compare_field(name, getattr(left, name), getattr(right, name), compare) for name, compare in comparisons]
    line_items = _compare_line_items(left, right)
    field_statuses = {comparison.status for comparison in fields}
    status: ComparisonStatus
    if "mismatch" in field_statuses or line_items.status == "mismatch":
        status = "mismatch"
    elif "missing" in field_statuses:
        status = "partial"  # line items that a side lacks are left out on purpose: ledger rows never have any
    else:
        status = "match"
    return ComparisonResult(status=status, fields=fields, line_items=line_items)


def comparison_kind(left_kind: FileKind, right_kind: FileKind) -> ComparisonKind:
    """``pdf_pdf`` (PDF/image on both sides), ``pdf_excel`` (one spreadsheet, one PDF/image), else ``other``."""
    kinds = (left_kind, right_kind)
    if all(kind in _VISUAL_KINDS for kind in kinds):
        return "pdf_pdf"
    if any(kind in _VISUAL_KINDS for kind in kinds) and any(kind in _SPREADSHEET_KINDS for kind in kinds):
        return "pdf_excel"
    return "other"


def cross_verification_issue(
    result: ComparisonResult, *, other_invoice_id: UUID, other_document_id: UUID
) -> ValidationIssue | None:
    """CROSS_VERIFICATION_MISMATCH for a ``mismatch`` result; ``match`` and ``partial`` raise no issue."""
    if result.status != "mismatch":
        return None
    mismatched = [comparison.field for comparison in result.fields if comparison.status == "mismatch"]
    if result.line_items.status == "mismatch":
        mismatched.append("line_items")
    return ValidationIssue(
        code=IssueCode.CROSS_VERIFICATION_MISMATCH,
        severity=DEFAULT_SEVERITY[IssueCode.CROSS_VERIFICATION_MISMATCH],
        message="The record differs from the matching record in another document",
        details={
            "other_invoice_id": str(other_invoice_id),
            "other_document_id": str(other_document_id),
            "mismatched_fields": mismatched,
        },
    )


# ------------------------------------------------------------------------------------------------ fields


def _compare_field(name: str, left: Any, right: Any, compare: _Comparison) -> FieldComparison:
    has_left, has_right = _has_value(left), _has_value(right)
    shown_left = _show(left) if has_left else None
    shown_right = _show(right) if has_right else None
    if not (has_left and has_right):
        return FieldComparison(name, shown_left, shown_right, "missing", _missing_note(has_left, has_right))
    agree, note = compare(left, right)
    return FieldComparison(name, shown_left, shown_right, "match" if agree else "mismatch", note)


def _has_value(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and not value.strip())


def _missing_note(has_left: bool, has_right: bool) -> str:
    if not has_left and not has_right:
        return "missing on both sides"
    return "missing on left" if not has_left else "missing on right"


def _show(value: str | date | Decimal) -> str:
    if isinstance(value, Decimal):
        return format(_cents(value), "f")
    if isinstance(value, date):
        return value.isoformat()
    return value


def _cents(value: Decimal) -> Decimal:
    """``value`` rounded to cents; a magnitude beyond the decimal precision is kept as it is."""
    try:
        return quantize_money(value)
    except InvalidOperation:
        return value


def _same_number(left: str, right: str) -> tuple[bool, str | None]:
    normalized_left, normalized_right = normalize_invoice_number(left), normalize_invoice_number(right)
    agree = normalized_left == normalized_right if normalized_left and normalized_right else left == right
    return agree, "equal after normalization" if agree and left != right else None


def _same_date(left: date, right: date) -> tuple[bool, str | None]:
    if left == right:
        return True, None
    days = abs((left - right).days)
    return False, f"{days} day{'' if days == 1 else 's'} apart"


def _same_name(left: str, right: str) -> tuple[bool, str | None]:
    if left == right:
        return True, None
    if not names_match(left, right):
        return False, None
    return True, "equal after normalization" if normalize_name(left) == normalize_name(right) else "similar names"


def _same_amount(left: Decimal, right: Decimal, *, tolerance: Decimal) -> tuple[bool, str | None]:
    if left == right:
        return True, None
    difference = format(_cents(abs(left - right)), "f")
    if amounts_equal(left, right, tolerance):
        return True, f"within tolerance (difference {difference})"
    return False, f"difference {difference}"


def _same_currency(left: str, right: str) -> tuple[bool, str | None]:
    return left == right, None


# --------------------------------------------------------------------------------------------- line items


def _amounts_in_cents(data: InvoiceData) -> Counter[Decimal]:
    return Counter(_cents(line.amount) for line in data.line_items if line.amount is not None)


def _compare_line_items(left: InvoiceData, right: InvoiceData) -> LineItemComparison:
    left_amounts, right_amounts = _amounts_in_cents(left), _amounts_in_cents(right)
    left_count, right_count = left_amounts.total(), right_amounts.total()
    status: FieldStatus
    if not left_count or not right_count:
        status = "missing"
    elif left_amounts == right_amounts:
        status = "match"
    else:
        status = "mismatch"
    return LineItemComparison(left_count, right_count, (left_amounts & right_amounts).total(), status)
