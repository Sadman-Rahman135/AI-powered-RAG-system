"""The review decision."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import pytest

from app.domain.schemas import (
    CORE_FIELDS,
    ExtractionSource,
    FieldMeta,
    InvoiceData,
    IssueCode,
    MetaNote,
    ReviewStatus,
    Severity,
    ValidationIssue,
)
from app.validation.review import review_status_for
from app.validation.rules import validate_invoice

MakeInvoice = Callable[..., InvoiceData]


def issue(severity: Severity) -> ValidationIssue:
    return ValidationIssue(code=IssueCode.LOW_CONFIDENCE, severity=severity, message="m")


def test_no_issues_is_auto_approved() -> None:
    assert review_status_for([]) is ReviewStatus.AUTO_APPROVED


def test_info_only_is_auto_approved() -> None:
    assert review_status_for([issue(Severity.INFO), issue(Severity.INFO)]) is ReviewStatus.AUTO_APPROVED


@pytest.mark.parametrize(
    "severities",
    [
        [Severity.WARNING],
        [Severity.ERROR],
        [Severity.INFO, Severity.WARNING],
        [Severity.INFO, Severity.ERROR, Severity.WARNING],
    ],
)
def test_warning_or_error_needs_review(severities: list[Severity]) -> None:
    assert review_status_for([issue(s) for s in severities]) is ReviewStatus.NEEDS_REVIEW


def test_reviewed_is_never_assigned_automatically() -> None:
    every_combination = [[], [Severity.INFO], [Severity.WARNING], [Severity.ERROR]]
    results = {review_status_for([issue(s) for s in severities]) for severities in every_combination}
    assert ReviewStatus.REVIEWED not in results


def test_status_follows_validation_results(make_invoice: MakeInvoice) -> None:
    meta = {name: FieldMeta(confidence=0.95, source=ExtractionSource.RULES) for name in CORE_FIELDS}
    today = date(2024, 6, 1)

    clean = validate_invoice(make_invoice(), meta, today=today)
    assert review_status_for(clean) is ReviewStatus.AUTO_APPROVED

    inferred = FieldMeta(confidence=0.9, source=ExtractionSource.RULES, notes=[MetaNote.CURRENCY_INFERRED.value])
    info_only = validate_invoice(make_invoice(), {**meta, "currency": inferred}, today=today)
    assert [i.severity for i in info_only] == [Severity.INFO]
    assert review_status_for(info_only) is ReviewStatus.AUTO_APPROVED

    missing_customer = validate_invoice(make_invoice(customer_name=None), meta, today=today)
    assert review_status_for(missing_customer) is ReviewStatus.NEEDS_REVIEW
