"""Field-level validation of one extracted invoice record.

``validate_invoice`` turns an ``InvoiceData`` and its per-field metadata into ``ValidationIssue`` objects: missing
fields, implausible amounts and dates, arithmetic consistency (line items -> subtotal -> total, tax rate) and the
signals extractors leave in ``FieldMeta`` (low confidence, disagreement, ambiguity, rejected values).
``document_level_issues`` covers what is known about the whole document rather than about one record.

Severities come from ``DEFAULT_SEVERITY`` (``MISSING_FIELD`` depends on the field). Messages are short and never
contain document text beyond field names; numbers travel in ``details`` as strings ("110.00").
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import date, timedelta
from decimal import MAX_EMAX, MIN_EMIN, Context, Decimal, DivisionByZero, InvalidOperation, localcontext
from typing import Any
from uuid import UUID

from app.config import get_settings
from app.domain.schemas import (
    CORE_FIELDS,
    DEFAULT_SEVERITY,
    INVOICE_FIELDS,
    MONEY_FIELDS,
    RECOMMENDED_FIELDS,
    REQUIRED_FIELDS,
    DocumentType,
    ExtractionWarning,
    FieldMeta,
    InvoiceData,
    IssueCode,
    MetaNote,
    Severity,
    SourceKind,
    ValidationIssue,
    field_is_present,
)
from app.extraction.normalize import amounts_equal

__all__ = ["document_level_issues", "validate_invoice"]

# Money arithmetic must never raise on absurd magnitudes (quantity x price overflowing the exponent range, ...):
# such values are reported as INVALID_AMOUNT, so the checks run with an effectively unbounded exponent range.
_ARITHMETIC = Context(prec=60, Emax=MAX_EMAX, Emin=MIN_EMIN, traps=[InvalidOperation, DivisionByZero])

_MAX_DECIMAL_PLACES = 4  # precision of the stored money columns
_UNIT_PRICE_ROUNDING = Decimal("0.005")  # unit prices are printed rounded to the cent: half a cent per unit
_TAX_RELATIVE_TOLERANCE = Decimal("0.005")  # tax is rounded per line or per document: 0.5 % of the tax amount
_FUTURE_GRACE = timedelta(days=1)  # time zones: tomorrow's date is not "the future"
# Only the discount is exempt from NEGATIVE_AMOUNT: invoices often print it negative ("-10.00").
_NEGATIVE_CHECKED = ("subtotal", "shipping", "tax", "total", "line_items")
_LLM_WARNINGS = frozenset(
    w.value for w in (ExtractionWarning.LLM_UNAVAILABLE, ExtractionWarning.LLM_FAILED, ExtractionWarning.LLM_REFUSED)
)

# FieldMeta note -> (issue code, message; ``{field}`` is replaced by the field name)
_NOTE_ISSUES: tuple[tuple[MetaNote, IssueCode, str], ...] = (
    (MetaNote.EXTRACTION_DISAGREEMENT, IssueCode.EXTRACTION_DISAGREEMENT, "Extraction sources disagreed on '{field}'"),
    (MetaNote.AMBIGUOUS_DATE_ORDER, IssueCode.AMBIGUOUS_DATE, "Day/month order of '{field}' is ambiguous"),
    (MetaNote.CURRENCY_INFERRED, IssueCode.CURRENCY_INFERRED, "Currency was inferred from a symbol"),
    (
        MetaNote.UNGROUNDED_VALUE_REJECTED,
        IssueCode.UNGROUNDED_VALUE_REJECTED,
        "A value for '{field}' was not found in the document and was rejected",
    ),
)


def validate_invoice(
    data: InvoiceData,
    meta: dict[str, FieldMeta],
    *,
    today: date | None = None,
    source_kind: SourceKind = SourceKind.INVOICE_DOCUMENT,
    tolerance: Decimal | None = None,
    confidence_threshold: float | None = None,
    max_amount: Decimal | None = None,
    max_tax_rate: Decimal | None = None,
) -> list[ValidationIssue]:
    """All issues for one record, in check order (missing fields, amounts, dates, arithmetic, tax, metadata).

    ``today`` (default: the current date) is injectable for tests. A ``LEDGER_ROW`` record never has line items,
    so it raises no MISSING_FIELD for them and skips the line-item rules. Every limit defaults to its setting:
    ``tolerance`` (absolute money tolerance) to ``AMOUNT_TOLERANCE``, ``confidence_threshold`` (the LOW_CONFIDENCE
    cut-off) to ``REVIEW_CONFIDENCE_THRESHOLD``, ``max_amount`` to ``MAX_REASONABLE_AMOUNT`` and ``max_tax_rate``
    (percent) to ``TAX_RATE_MAX_PERCENT``.
    """
    settings = get_settings()
    if today is None:
        today = date.today()
    if tolerance is None:
        tolerance = settings.amount_tolerance
    if confidence_threshold is None:
        confidence_threshold = settings.review_confidence_threshold
    if max_amount is None:
        max_amount = settings.max_reasonable_amount
    if max_tax_rate is None:
        max_tax_rate = settings.tax_rate_max_percent
    ledger_row = source_kind == SourceKind.LEDGER_ROW
    with localcontext(_ARITHMETIC):
        return [
            *_missing_fields(data, ledger_row),
            *_invalid_amounts(data, max_amount),
            *_negative_amounts(data),
            *_date_issues(data, today),
            *([] if ledger_row else _line_item_math(data, tolerance)),
            *([] if ledger_row else _line_items_sum(data, tolerance)),
            *_total_mismatch(data, tolerance),
            *_tax_issues(data, tolerance, max_tax_rate),
            *_meta_issues(data, meta, confidence_threshold),
        ]


def document_level_issues(
    *,
    ocr_confidence: float | None,
    duplicate_of_id: UUID | None,
    extraction_warnings: Sequence[str],
    ocr_threshold: float,
) -> list[ValidationIssue]:
    """Issues about the document as a whole: low mean OCR confidence, a re-upload, an unavailable LLM."""
    issues: list[ValidationIssue] = []
    if ocr_confidence is not None and ocr_confidence < ocr_threshold:
        issues.append(
            _issue(
                IssueCode.OCR_LOW_QUALITY,
                "OCR confidence is low; the extracted text may be unreliable",
                mean_confidence=ocr_confidence,
                threshold=ocr_threshold,
            )
        )
    if duplicate_of_id is not None:
        issues.append(
            _issue(IssueCode.DUPLICATE_FILE, "This file was already uploaded", duplicate_of_id=str(duplicate_of_id))
        )
    reasons = list(dict.fromkeys(w for w in extraction_warnings if w in _LLM_WARNINGS))
    if reasons:
        issues.append(
            _issue(
                IssueCode.LLM_UNAVAILABLE,
                "LLM extraction was unavailable; the rules-based result was used",
                reasons=reasons,
            )
        )
    return issues


# ---------------------------------------------------------------------------------------------- helpers


def _issue(code: IssueCode, message: str, *, field: str | None = None, **details: Any) -> ValidationIssue:
    """An issue with the severity ``DEFAULT_SEVERITY`` assigns to ``code``; ``details`` are keyword arguments."""
    return ValidationIssue(code=code, severity=DEFAULT_SEVERITY[code], field=field, message=message, details=details)


def _fmt(value: Decimal) -> str:
    """Plain-notation string with at least two decimals and no padding beyond them ("110" -> "110.00")."""
    whole, _, fraction = format(value, "f").partition(".")
    return f"{whole}.{fraction.rstrip('0').ljust(2, '0')}"


def _difference(expected: Decimal, actual: Decimal) -> dict[str, str]:
    """``expected`` is what the other numbers imply, ``actual`` what the document prints."""
    return {"expected": _fmt(expected), "actual": _fmt(actual), "difference": _fmt(abs(actual - expected))}


def _decimal_places(value: Decimal) -> int:
    """Significant decimal places: ``Decimal("1.2300")`` -> 2, ``Decimal("1E+2")`` -> 0."""
    _, digits, exponent = value.as_tuple()
    if not isinstance(exponent, int) or not any(digits):
        return 0
    trailing_zeros = len(digits) - len("".join(map(str, digits)).rstrip("0"))
    return max(0, -(exponent + trailing_zeros))


def _is_missing(data: InvoiceData, name: str) -> bool:
    if not field_is_present(data, name):
        return True
    value = getattr(data, name)
    return isinstance(value, str) and not value.strip()


def _discount(data: InvoiceData) -> Decimal:
    """The discount as a positive amount: invoices print it both as "10.00" and as "-10.00"."""
    return abs(data.discount) if data.discount is not None else Decimal(0)


def _total_is_beyond(total: Decimal, subtotal: Decimal) -> bool:
    """Whether ``total`` is at least as far from zero as ``subtotal``, on the same side (credit notes: lower)."""
    return (total - subtotal) * (1 if subtotal >= 0 else -1) >= 0


def _amounts(data: InvoiceData) -> Iterator[tuple[str, int | None, Decimal]]:
    """Every money value on the record as (field, line-item index or None, value)."""
    for name in MONEY_FIELDS:
        value = getattr(data, name)
        if value is not None:
            yield name, None, value
    for index, line in enumerate(data.line_items):
        if line.amount is not None:
            yield "line_items", index, line.amount


def _where(name: str, index: int | None) -> tuple[str, dict[str, int]]:
    """Human label and extra ``details`` locating a money value."""
    if index is None:
        return f"'{name}'", {}
    return f"line item {index + 1}", {"index": index}


# ------------------------------------------------------------------------------------------------ checks


def _missing_fields(data: InvoiceData, ledger_row: bool) -> Iterator[ValidationIssue]:
    # The one code whose severity depends on the field: an error for required fields, a warning for recommended.
    for names, severity, kind in (
        (REQUIRED_FIELDS, Severity.ERROR, "Required"),
        (RECOMMENDED_FIELDS, Severity.WARNING, "Recommended"),
    ):
        for name in names:
            if _is_missing(data, name) and not (ledger_row and name == "line_items"):
                message = f"{kind} field '{name}' is missing"
                yield ValidationIssue(code=IssueCode.MISSING_FIELD, severity=severity, field=name, message=message)


def _invalid_amounts(data: InvoiceData, max_amount: Decimal) -> Iterator[ValidationIssue]:
    for name, index, value in _amounts(data):
        label, where = _where(name, index)
        if abs(value) > max_amount:
            yield _issue(
                IssueCode.INVALID_AMOUNT,
                f"Amount in {label} exceeds the plausible maximum",
                field=name,
                **where,
                value=_fmt(value),
                reason="exceeds_maximum",
                maximum=_fmt(max_amount),
            )
        elif _decimal_places(value) > _MAX_DECIMAL_PLACES:
            yield _issue(
                IssueCode.INVALID_AMOUNT,
                f"Amount in {label} has more than {_MAX_DECIMAL_PLACES} decimal places",
                field=name,
                **where,
                value=_fmt(value),
                reason="too_many_decimals",
            )
    total, tax = data.total, data.tax
    if total is not None and total == 0 and data.document_type is not DocumentType.CREDIT_NOTE:
        yield _issue(IssueCode.INVALID_AMOUNT, "Total is zero", field="total", value=_fmt(total), reason="zero_total")
    if total is not None and tax is not None and 0 < total < tax:
        yield _issue(
            IssueCode.INVALID_AMOUNT,
            "Tax exceeds the total",
            field="tax",
            tax=_fmt(tax),
            total=_fmt(total),
            reason="tax_exceeds_total",
        )


def _negative_amounts(data: InvoiceData) -> Iterator[ValidationIssue]:
    if data.document_type is DocumentType.CREDIT_NOTE:
        return
    for name, index, value in _amounts(data):
        if name in _NEGATIVE_CHECKED and value < 0:
            label, where = _where(name, index)
            yield _issue(
                IssueCode.NEGATIVE_AMOUNT, f"Amount in {label} is negative", field=name, **where, value=_fmt(value)
            )


def _date_issues(data: InvoiceData, today: date) -> Iterator[ValidationIssue]:
    invoice_date, due_date = data.invoice_date, data.due_date
    if invoice_date is not None and due_date is not None and due_date < invoice_date:
        yield _issue(
            IssueCode.INVALID_DATE,
            "Due date is earlier than the invoice date",
            field="due_date",
            invoice_date=invoice_date.isoformat(),
            due_date=due_date.isoformat(),
        )
    if invoice_date is not None and invoice_date > today + _FUTURE_GRACE:
        yield _issue(
            IssueCode.FUTURE_DATE,
            "Invoice date is in the future",
            field="invoice_date",
            invoice_date=invoice_date.isoformat(),
            today=today.isoformat(),
        )


def _line_item_math(data: InvoiceData, tolerance: Decimal) -> Iterator[ValidationIssue]:
    for index, line in enumerate(data.line_items):
        if line.quantity is None or line.unit_price is None or line.amount is None:
            continue
        expected = line.quantity * line.unit_price
        allowed = max(tolerance, abs(line.quantity) * _UNIT_PRICE_ROUNDING)
        if abs(expected - line.amount) > allowed:
            yield _issue(
                IssueCode.LINE_ITEM_MATH,
                f"Quantity x unit price does not equal the amount of line item {index + 1}",
                field="line_items",
                index=index,
                **_difference(expected, line.amount),
            )


def _line_items_sum(data: InvoiceData, tolerance: Decimal) -> Iterator[ValidationIssue]:
    amounts = [line.amount for line in data.line_items if line.amount is not None]
    if not amounts or len(amounts) != len(data.line_items):
        return  # nothing to add up, or a partial sum that proves nothing
    items_sum = sum(amounts, Decimal(0))
    if data.subtotal is not None:
        compared_with, printed = "subtotal", data.subtotal
        # The items may be listed before a discount that the printed subtotal already includes.
        accepted = [printed, printed + _discount(data)]
    elif data.total is not None and data.tax == 0 and not any((data.discount, data.shipping)):
        # No subtotal: the items must add up to the total, but only if the document shows no tax (an explicit
        # zero, not a missing value), discount or shipping that could account for a difference.
        compared_with, printed = "total", data.total
        accepted = [printed]
    else:
        return
    if not any(amounts_equal(items_sum, target, tolerance) for target in accepted):
        yield _issue(
            IssueCode.LINE_ITEMS_SUM_MISMATCH,
            f"Line item amounts do not add up to the {compared_with}",
            field="line_items",
            compared_with=compared_with,
            **_difference(items_sum, printed),
        )


def _total_mismatch(data: InvoiceData, tolerance: Decimal) -> Iterator[ValidationIssue]:
    subtotal, total = data.subtotal, data.total
    if subtotal is None or total is None:
        return
    if data.tax is None and _total_is_beyond(total, subtotal):
        return  # the gap may be tax that was not extracted; MISSING_FIELD already reports that
    discount = _discount(data)
    charges = (data.shipping or 0) + (data.tax or 0)
    expected = subtotal - discount + charges
    # With a discount, the printed subtotal may already be net of it (the items add up to subtotal + discount).
    accepted = [expected, subtotal + charges] if discount else [expected]
    if not any(amounts_equal(candidate, total, tolerance) for candidate in accepted):
        yield _issue(
            IssueCode.TOTAL_MISMATCH,
            "Total does not equal subtotal - discount + shipping + tax",
            field="total",
            **_difference(expected, total),
        )


def _tax_issues(data: InvoiceData, tolerance: Decimal, max_rate: Decimal) -> Iterator[ValidationIssue]:
    subtotal, tax = data.subtotal, data.tax
    if subtotal is None or tax is None:
        return
    # What the tax may have been charged on: the subtotal, after the discount, after discount and shipping.
    after_discount = subtotal - _discount(data)
    bases = list(dict.fromkeys((subtotal, after_discount, after_discount + (data.shipping or 0))))
    if data.tax_rate is not None:
        expected = [data.tax_rate / 100 * base for base in bases]
        allowed = max(tolerance, abs(tax) * _TAX_RELATIVE_TOLERANCE)
        if all(abs(tax - candidate) > allowed for candidate in expected):
            closest = min(expected, key=lambda candidate: abs(tax - candidate))
            yield _issue(
                IssueCode.TAX_RATE_MISMATCH,
                "Tax amount does not match the stated tax rate",
                field="tax_rate",
                stated_rate=_fmt(data.tax_rate),
                **_difference(closest, tax),
            )
    # Unusual only if no plausible base brings the effective rate down to an ordinary level.
    rates = [abs(tax / base) * 100 for base in bases if base != 0]
    if rates and min(rates) > max_rate:
        yield _issue(
            IssueCode.TAX_RATE_UNUSUAL,
            "Effective tax rate is unusually high",
            field="tax",
            effective_rate=_fmt(min(rates)),
            max_rate=_fmt(max_rate),
        )


def _meta_issues(data: InvoiceData, meta: dict[str, FieldMeta], threshold: float) -> Iterator[ValidationIssue]:
    for name in INVOICE_FIELDS:
        field_meta = meta.get(name)
        if field_meta is None:
            continue
        if name in CORE_FIELDS and not _is_missing(data, name) and field_meta.confidence < threshold:
            yield _issue(
                IssueCode.LOW_CONFIDENCE,
                f"Low extraction confidence for '{name}'",
                field=name,
                confidence=field_meta.confidence,
                threshold=threshold,
            )
        for note, code, message in _NOTE_ISSUES:
            if note in field_meta.notes:
                yield _issue(code, message.format(field=name), field=name)
