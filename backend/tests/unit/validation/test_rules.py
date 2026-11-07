"""Field-level validation rules and document-level issues."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest

from app.config import reset_settings_cache
from app.domain.schemas import (
    CORE_FIELDS,
    DEFAULT_SEVERITY,
    RECOMMENDED_FIELDS,
    REQUIRED_FIELDS,
    DocumentType,
    ExtractionSource,
    ExtractionWarning,
    FieldMeta,
    InvoiceData,
    IssueCode,
    LineItem,
    MetaNote,
    Severity,
    SourceKind,
    ValidationIssue,
)
from app.validation.rules import document_level_issues, validate_invoice

TODAY = date(2024, 6, 1)
MakeInvoice = Callable[..., InvoiceData]

# Every code validate_invoice can emit for a single record (the others come from document, duplicate and
# cross-verification checks).
RECORD_CODES = {
    IssueCode.MISSING_FIELD,
    IssueCode.INVALID_AMOUNT,
    IssueCode.INVALID_DATE,
    IssueCode.TOTAL_MISMATCH,
    IssueCode.LINE_ITEMS_SUM_MISMATCH,
    IssueCode.LINE_ITEM_MATH,
    IssueCode.NEGATIVE_AMOUNT,
    IssueCode.FUTURE_DATE,
    IssueCode.TAX_RATE_MISMATCH,
    IssueCode.TAX_RATE_UNUSUAL,
    IssueCode.LOW_CONFIDENCE,
    IssueCode.EXTRACTION_DISAGREEMENT,
    IssueCode.AMBIGUOUS_DATE,
    IssueCode.CURRENCY_INFERRED,
    IssueCode.UNGROUNDED_VALUE_REJECTED,
}


def confident_meta(confidence: float = 0.95) -> dict[str, FieldMeta]:
    return {name: FieldMeta(confidence=confidence, source=ExtractionSource.RULES) for name in CORE_FIELDS}


def noted(confidence: float, *notes: MetaNote) -> FieldMeta:
    return FieldMeta(confidence=confidence, source=ExtractionSource.RULES, notes=[n.value for n in notes])


def validate(data: InvoiceData, meta: dict[str, FieldMeta] | None = None, **kwargs: Any) -> list[ValidationIssue]:
    return validate_invoice(data, confident_meta() if meta is None else meta, today=TODAY, **kwargs)


def of_code(issues: list[ValidationIssue], code: IssueCode) -> list[ValidationIssue]:
    return [issue for issue in issues if issue.code == code]


def codes(issues: list[ValidationIssue]) -> set[IssueCode]:
    return {issue.code for issue in issues}


def item(quantity: str | None, unit_price: str | None, amount: str | None) -> LineItem:
    def money(value: str | None) -> Decimal | None:
        return None if value is None else Decimal(value)

    return LineItem(description="x", quantity=money(quantity), unit_price=money(unit_price), amount=money(amount))


# ------------------------------------------------------------------------------------------------ baseline


def test_valid_invoice_has_no_issues(make_invoice: MakeInvoice) -> None:
    assert validate(make_invoice()) == []


def test_every_record_code_is_emitted_with_its_default_severity(make_invoice: MakeInvoice) -> None:
    data = make_invoice(
        supplier_name=None,
        customer_name=None,
        invoice_date=date(2024, 9, 1),
        due_date=date(2024, 8, 1),
        subtotal=Decimal("-100.00"),
        tax=Decimal("40.00"),
        tax_rate=Decimal("50"),
        total=Decimal("2000000000"),
        line_items=[item("2", "10.00", "5.00")],
    )
    meta = confident_meta(0.2)
    meta["invoice_date"] = noted(0.2, MetaNote.AMBIGUOUS_DATE_ORDER, MetaNote.EXTRACTION_DISAGREEMENT)
    meta["currency"] = noted(0.9, MetaNote.CURRENCY_INFERRED)
    meta["due_date"] = noted(0.9, MetaNote.UNGROUNDED_VALUE_REJECTED)

    issues = validate(data, meta)

    assert codes(issues) == RECORD_CODES
    for issue in issues:
        if issue.code is not IssueCode.MISSING_FIELD:
            assert issue.severity == DEFAULT_SEVERITY[issue.code], issue.code
    json.dumps([issue.model_dump(mode="json") for issue in issues])  # details are plain JSON values


def test_messages_and_details_do_not_echo_document_text(make_invoice: MakeInvoice) -> None:
    data = make_invoice(
        invoice_number="ZZ-SECRET-77",
        supplier_name="Secret Supplier GmbH",
        customer_name="Hidden Customer SA",
        total=Decimal("0"),
        due_date=date(2024, 1, 1),
    )
    issues = validate(data, confident_meta(0.1))
    assert issues
    for issue in issues:
        text = f"{issue.message} {issue.details}"
        for secret in ("ZZ-SECRET-77", "Secret Supplier", "Hidden Customer"):
            assert secret not in text


def test_absurd_magnitudes_are_reported_not_raised(make_invoice: MakeInvoice) -> None:
    huge = Decimal("1E+600000")
    data = make_invoice(
        subtotal=huge,
        total=huge,
        tax=None,
        tax_rate=None,
        line_items=[LineItem(quantity=huge, unit_price=huge, amount=Decimal("1"))],
    )
    issues = validate(data)
    assert {i.field for i in of_code(issues, IssueCode.INVALID_AMOUNT)} == {"subtotal", "total"}


def test_limits_default_to_settings_and_can_be_overridden(
    make_invoice: MakeInvoice, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAX_REASONABLE_AMOUNT", "100")
    monkeypatch.setenv("TAX_RATE_MAX_PERCENT", "5")
    reset_settings_cache()
    data = make_invoice()  # total 110 > 100; effective tax rate 10 % > 5 %

    assert {IssueCode.INVALID_AMOUNT, IssueCode.TAX_RATE_UNUSUAL} <= codes(validate(data))
    relaxed = validate(data, max_amount=Decimal("1000"), max_tax_rate=Decimal("30"))
    assert relaxed == []


def test_tolerance_and_confidence_threshold_default_to_settings(
    make_invoice: MakeInvoice, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = make_invoice(total=Decimal("110.50"))  # 0.50 off
    meta = confident_meta(0.85)
    assert codes(validate(data, meta)) == {IssueCode.TOTAL_MISMATCH}  # defaults: tolerance 0.02, threshold 0.6

    monkeypatch.setenv("AMOUNT_TOLERANCE", "0.50")
    monkeypatch.setenv("REVIEW_CONFIDENCE_THRESHOLD", "0.9")
    reset_settings_cache()

    assert codes(validate(data, meta)) == {IssueCode.LOW_CONFIDENCE}  # 0.50 is now within tolerance, 0.85 < 0.9
    explicit = validate(data, meta, tolerance=Decimal("0.02"), confidence_threshold=0.6)
    assert codes(explicit) == {IssueCode.TOTAL_MISMATCH}  # explicit arguments win over the settings


# ------------------------------------------------------------------------------------------ MISSING_FIELD


@pytest.mark.parametrize("name", [*REQUIRED_FIELDS, *RECOMMENDED_FIELDS])
def test_missing_field_flagged(make_invoice: MakeInvoice, name: str) -> None:
    data = make_invoice(**{name: [] if name == "line_items" else None})

    [issue] = validate(data)

    assert issue.code is IssueCode.MISSING_FIELD
    assert issue.field == name
    assert issue.severity is (Severity.ERROR if name in REQUIRED_FIELDS else Severity.WARNING)
    assert name in issue.message


def test_missing_field_not_flagged(make_invoice: MakeInvoice) -> None:
    assert IssueCode.MISSING_FIELD not in codes(validate(make_invoice()))
    optional_absent = make_invoice(
        due_date=None, discount=None, shipping=None, tax_rate=None, document_type=DocumentType.UNKNOWN
    )
    assert validate(optional_absent) == []


def test_ledger_row_without_line_items_has_no_issues(make_invoice: MakeInvoice) -> None:
    ledger_row = make_invoice(line_items=[])

    assert validate(ledger_row, source_kind=SourceKind.LEDGER_ROW) == []
    [issue] = validate(ledger_row, source_kind=SourceKind.INVOICE_DOCUMENT)
    assert (issue.code, issue.severity, issue.field) == (IssueCode.MISSING_FIELD, Severity.WARNING, "line_items")
    assert validate(ledger_row) == [issue]  # the default is an invoice document


def test_ledger_row_skips_the_line_item_rules(make_invoice: MakeInvoice) -> None:
    inconsistent = make_invoice(line_items=[item("2", "10.00", "5.00")])  # bad math, and the sum is not 100

    assert validate(inconsistent, source_kind=SourceKind.LEDGER_ROW) == []
    assert {IssueCode.LINE_ITEM_MATH, IssueCode.LINE_ITEMS_SUM_MISMATCH} <= codes(validate(inconsistent))


def test_blank_text_field_counts_as_missing(make_invoice: MakeInvoice) -> None:
    [issue] = validate(make_invoice(invoice_number="   "))
    assert (issue.code, issue.field) == (IssueCode.MISSING_FIELD, "invoice_number")


# ----------------------------------------------------------------------------------------- INVALID_AMOUNT


@pytest.mark.parametrize(
    ("overrides", "field", "reason"),
    [
        pytest.param({"total": Decimal("2000000000.00")}, "total", "exceeds_maximum", id="too-large-total"),
        pytest.param({"discount": Decimal("5000000000")}, "discount", "exceeds_maximum", id="too-large-discount"),
        pytest.param({"shipping": Decimal("1.23456")}, "shipping", "too_many_decimals", id="five-decimals"),
        pytest.param({"total": Decimal("0.00")}, "total", "zero_total", id="zero-total"),
        pytest.param({"tax": Decimal("150.00")}, "tax", "tax_exceeds_total", id="tax-above-total"),
        pytest.param(
            {"line_items": [LineItem(amount=Decimal("2000000000"))]},
            "line_items",
            "exceeds_maximum",
            id="too-large-item",
        ),
    ],
)
def test_invalid_amount_flagged(make_invoice: MakeInvoice, overrides: dict[str, Any], field: str, reason: str) -> None:
    [issue] = of_code(validate(make_invoice(**overrides)), IssueCode.INVALID_AMOUNT)

    assert issue.severity is Severity.ERROR
    assert issue.field == field
    assert issue.details["reason"] == reason
    if field == "line_items":
        assert issue.details["index"] == 0


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(
            {
                "subtotal": Decimal("999999990.00"),
                "total": Decimal("1000000000.00"),
                "tax_rate": None,
                "line_items": [],
            },
            id="exactly-the-maximum",
        ),
        pytest.param({"shipping": Decimal("1.2345")}, id="four-decimals"),
        pytest.param({"shipping": Decimal("1.230000")}, id="trailing-zeros"),
        pytest.param({"document_type": DocumentType.CREDIT_NOTE, "total": Decimal("0.00")}, id="credit-note-zero"),
        pytest.param({"tax": Decimal("110.00")}, id="tax-equals-total"),
        pytest.param({"tax": Decimal("-150.00")}, id="negative-tax"),
    ],
)
def test_invalid_amount_not_flagged(make_invoice: MakeInvoice, overrides: dict[str, Any]) -> None:
    assert IssueCode.INVALID_AMOUNT not in codes(validate(make_invoice(**overrides)))


# --------------------------------------------------------------------------------------- NEGATIVE_AMOUNT


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        pytest.param({"subtotal": Decimal("-100.00")}, "subtotal", id="subtotal"),
        pytest.param({"tax": Decimal("-10.00")}, "tax", id="tax"),
        pytest.param({"total": Decimal("-110.00")}, "total", id="total"),
        pytest.param({"line_items": [LineItem(amount=Decimal("-5.00"))]}, "line_items", id="item"),
        pytest.param({"shipping": Decimal("-3.00"), "total": Decimal("107.00")}, "shipping", id="shipping"),
    ],
)
def test_negative_amount_flagged(make_invoice: MakeInvoice, overrides: dict[str, Any], field: str) -> None:
    [issue] = of_code(validate(make_invoice(**overrides)), IssueCode.NEGATIVE_AMOUNT)

    assert issue.severity is Severity.WARNING
    assert issue.field == field
    if field == "line_items":
        assert issue.details["index"] == 0


def test_negative_shipping_is_flagged_like_any_other_amount(make_invoice: MakeInvoice) -> None:
    [issue] = validate(make_invoice(shipping=Decimal("-3.00"), total=Decimal("107.00")))  # 100 - 3 + 10

    assert (issue.code, issue.severity, issue.field) == (IssueCode.NEGATIVE_AMOUNT, Severity.WARNING, "shipping")


def test_negative_amount_not_flagged(make_invoice: MakeInvoice) -> None:
    # Only the discount is exempt: it is often printed negative ("-10.00"). 100 - 10 + 9 = 99, so nothing else fires.
    data = make_invoice(discount=Decimal("-10.00"), tax=Decimal("9.00"), total=Decimal("99.00"))
    assert validate(data) == []


def test_credit_note_negative_ok(make_invoice: MakeInvoice) -> None:
    credit_note = make_invoice(
        document_type=DocumentType.CREDIT_NOTE,
        subtotal=Decimal("-100.00"),
        shipping=Decimal("-5.00"),
        tax=Decimal("-10.00"),
        total=Decimal("-115.00"),
        line_items=[item("-2", "25.00", "-50.00"), item("-1", "50.00", "-50.00")],
    )
    assert validate(credit_note) == []


# ------------------------------------------------------------------------------------------------- discount


@pytest.mark.parametrize("discount", ["10.00", "-10.00"])
def test_discount_sign_does_not_matter(make_invoice: MakeInvoice, discount: str) -> None:
    # The discount is applied as a magnitude by the total check, the item sum check and the tax base:
    # 100 - 10 + 9 = 99, and 9 is 10 % of the discounted 90.
    data = make_invoice(discount=Decimal(discount), tax=Decimal("9.00"), total=Decimal("99.00"))
    assert validate(data) == []


# ------------------------------------------------------------------------------------------- INVALID_DATE


def test_invalid_date_flagged(make_invoice: MakeInvoice) -> None:
    [issue] = validate(make_invoice(due_date=date(2024, 3, 1)))

    assert issue.code is IssueCode.INVALID_DATE
    assert issue.severity is Severity.ERROR
    assert issue.field == "due_date"


@pytest.mark.parametrize("due", [date(2024, 3, 5), date(2024, 4, 4), None])
def test_invalid_date_not_flagged(make_invoice: MakeInvoice, due: date | None) -> None:
    assert validate(make_invoice(due_date=due)) == []


# ------------------------------------------------------------------------------------------- FUTURE_DATE


def test_future_date_flagged(make_invoice: MakeInvoice) -> None:
    [issue] = validate(make_invoice(invoice_date=date(2024, 6, 3), due_date=date(2024, 7, 3)))

    assert issue.code is IssueCode.FUTURE_DATE
    assert issue.severity is Severity.WARNING
    assert issue.field == "invoice_date"


@pytest.mark.parametrize("invoice_date", [date(2024, 6, 2), date(2024, 6, 1), date(2023, 1, 1)])
def test_future_date_not_flagged(make_invoice: MakeInvoice, invoice_date: date) -> None:
    assert validate(make_invoice(invoice_date=invoice_date, due_date=None)) == []


def test_today_defaults_to_the_current_date(make_invoice: MakeInvoice) -> None:
    far_future = make_invoice(invoice_date=date(2999, 1, 1), due_date=None)
    assert IssueCode.FUTURE_DATE in codes(validate_invoice(far_future, confident_meta()))


# ----------------------------------------------------------------------------------------- TOTAL_MISMATCH


def test_total_mismatch_flagged(make_invoice: MakeInvoice) -> None:
    [issue] = of_code(validate(make_invoice(total=Decimal("112.00"))), IssueCode.TOTAL_MISMATCH)

    assert issue.severity is Severity.ERROR
    assert issue.field == "total"
    assert issue.details == {"expected": "110.00", "actual": "112.00", "difference": "2.00"}


@pytest.mark.parametrize(
    ("total", "flagged"),
    [("110.00", False), ("110.02", False), ("109.98", False), ("110.03", True), ("109.97", True)],
)
def test_total_mismatch_tolerance_is_two_cents(make_invoice: MakeInvoice, total: str, flagged: bool) -> None:
    found = IssueCode.TOTAL_MISMATCH in codes(validate(make_invoice(total=Decimal(total))))
    assert found is flagged


def test_total_mismatch_tolerance_is_configurable(make_invoice: MakeInvoice) -> None:
    data = make_invoice(total=Decimal("110.50"))
    assert IssueCode.TOTAL_MISMATCH in codes(validate(data))
    assert IssueCode.TOTAL_MISMATCH not in codes(validate(data, tolerance=Decimal("0.50")))


def test_total_mismatch_not_flagged(make_invoice: MakeInvoice) -> None:
    assert validate(make_invoice()) == []
    # Without a subtotal or a total there is nothing to compare; only the missing field is reported.
    [no_subtotal] = validate(make_invoice(subtotal=None, total=Decimal("999.00")))
    assert (no_subtotal.code, no_subtotal.field) == (IssueCode.MISSING_FIELD, "subtotal")
    [no_total] = validate(make_invoice(total=None))
    assert (no_total.code, no_total.field) == (IssueCode.MISSING_FIELD, "total")


def test_total_with_discount_and_shipping_consistent(make_invoice: MakeInvoice) -> None:
    data = make_invoice(
        discount=Decimal("10.00"),
        shipping=Decimal("5.00"),
        tax=Decimal("9.00"),
        total=Decimal("104.00"),  # 100 - 10 + 5 + 9
    )
    assert validate(data) == []


def test_total_mismatch_flagged_with_a_discount(make_invoice: MakeInvoice) -> None:
    # 120 is neither 100 - 10 + 9 = 99 (subtotal before the discount) nor 100 + 9 = 109 (subtotal already net).
    data = make_invoice(discount=Decimal("10.00"), tax=Decimal("9.00"), total=Decimal("120.00"))

    [issue] = validate(data)

    assert issue.code is IssueCode.TOTAL_MISMATCH
    assert issue.details == {"expected": "99.00", "actual": "120.00", "difference": "21.00"}


@pytest.mark.parametrize("discount", ["10.00", "-10.00"])
def test_total_accepts_a_subtotal_that_is_already_net_of_the_discount(make_invoice: MakeInvoice, discount: str) -> None:
    # Items add up to 100 and the printed subtotal is 90, after the 10 discount: 90 + 9 = 99.
    data = make_invoice(
        subtotal=Decimal("90.00"), discount=Decimal(discount), tax=Decimal("9.00"), total=Decimal("99.00")
    )
    assert validate(data) == []


def test_total_mismatch_skipped_when_tax_is_missing_and_total_is_not_lower(make_invoice: MakeInvoice) -> None:
    # The gap may be tax that was not extracted; MISSING_FIELD already reports the absence.
    issues = validate(make_invoice(tax=None, tax_rate=None))
    assert codes(issues) == {IssueCode.MISSING_FIELD}


def test_total_mismatch_flagged_when_tax_is_missing_and_total_is_lower(make_invoice: MakeInvoice) -> None:
    issues = validate(make_invoice(tax=None, tax_rate=None, total=Decimal("90.00")))

    [issue] = of_code(issues, IssueCode.TOTAL_MISMATCH)
    assert issue.details == {"expected": "100.00", "actual": "90.00", "difference": "10.00"}


@pytest.mark.parametrize(
    ("subtotal", "total", "flagged"),
    [
        pytest.param("100.00", "110.00", False, id="gap-may-be-tax"),
        pytest.param("100.00", "100.00", False, id="equal"),
        pytest.param("100.00", "90.00", True, id="smaller"),
        pytest.param("-100.00", "-110.00", False, id="credit-note-gap-may-be-tax"),
        pytest.param("-100.00", "-100.00", False, id="credit-note-equal"),
        pytest.param("-100.00", "-90.00", True, id="credit-note-smaller-in-magnitude"),
        pytest.param("100.00", "-110.00", True, id="opposite-signs-are-never-a-missing-tax"),
    ],
)
def test_total_mismatch_without_tax_compares_magnitudes(
    make_invoice: MakeInvoice, subtotal: str, total: str, flagged: bool
) -> None:
    data = make_invoice(
        document_type=DocumentType.CREDIT_NOTE,  # either sign is fine for a credit note
        subtotal=Decimal(subtotal),
        total=Decimal(total),
        tax=None,
        tax_rate=None,
        line_items=[],
    )

    expected = {IssueCode.MISSING_FIELD} | ({IssueCode.TOTAL_MISMATCH} if flagged else set())
    assert codes(validate(data)) == expected


def test_credit_note_with_missing_tax_is_not_a_total_mismatch(make_invoice: MakeInvoice) -> None:
    credit_note = make_invoice(
        document_type=DocumentType.CREDIT_NOTE,
        subtotal=Decimal("-100.00"),
        total=Decimal("-110.00"),
        tax=None,
        tax_rate=None,
        line_items=[item("-2", "25.00", "-50.00"), item("-1", "50.00", "-50.00")],
    )

    [issue] = validate(credit_note)

    assert (issue.code, issue.field) == (IssueCode.MISSING_FIELD, "tax")


# ---------------------------------------------------------------------------------- LINE_ITEMS_SUM_MISMATCH


def test_line_items_sum_mismatch_flagged(make_invoice: MakeInvoice) -> None:
    data = make_invoice(subtotal=Decimal("105.00"), total=Decimal("115.00"), tax_rate=None)

    [issue] = validate(data)

    assert issue.code is IssueCode.LINE_ITEMS_SUM_MISMATCH
    assert issue.severity is Severity.ERROR
    assert issue.field == "line_items"
    assert issue.details == {
        "expected": "100.00",
        "actual": "105.00",
        "difference": "5.00",
        "compared_with": "subtotal",
    }


@pytest.mark.parametrize(("subtotal", "flagged"), [("100.02", False), ("99.98", False), ("100.03", True)])
def test_line_items_sum_tolerance(make_invoice: MakeInvoice, subtotal: str, flagged: bool) -> None:
    found = IssueCode.LINE_ITEMS_SUM_MISMATCH in codes(validate(make_invoice(subtotal=Decimal(subtotal))))
    assert found is flagged


@pytest.mark.parametrize("discount", ["10.00", "-10.00"])
def test_line_items_sum_accepts_pre_discount_amounts(make_invoice: MakeInvoice, discount: str) -> None:
    # Items add up to 100, the printed subtotal (90) is after the 10 discount: 90 + 9 = 99, 9 is 10 % of 90.
    data = make_invoice(
        subtotal=Decimal("90.00"), discount=Decimal(discount), tax=Decimal("9.00"), total=Decimal("99.00")
    )
    assert validate(data) == []


def test_line_items_sum_mismatch_not_flagged(make_invoice: MakeInvoice) -> None:
    assert validate(make_invoice()) == []
    [no_items] = validate(make_invoice(line_items=[]))
    assert (no_items.code, no_items.field) == (IssueCode.MISSING_FIELD, "line_items")
    assert validate(make_invoice(line_items=[LineItem(description="Consulting")])) == []


def test_line_items_sum_is_not_checked_when_an_item_has_no_amount(make_invoice: MakeInvoice) -> None:
    partial = make_invoice(line_items=[item(None, None, "50.00"), LineItem(description="Free sample")])
    assert validate(partial) == []


def test_line_items_sum_falls_back_to_total_when_the_tax_is_explicitly_zero(make_invoice: MakeInvoice) -> None:
    base = {"subtotal": None, "tax": Decimal("0.00"), "tax_rate": None}

    [only_missing] = validate(make_invoice(**base, total=Decimal("100.00")))
    assert (only_missing.code, only_missing.field) == (IssueCode.MISSING_FIELD, "subtotal")

    issues = validate(make_invoice(**base, total=Decimal("120.00")))
    assert codes(issues) == {IssueCode.MISSING_FIELD, IssueCode.LINE_ITEMS_SUM_MISMATCH}
    [issue] = of_code(issues, IssueCode.LINE_ITEMS_SUM_MISMATCH)
    assert issue.details == {
        "expected": "100.00",
        "actual": "120.00",
        "difference": "20.00",
        "compared_with": "total",
    }


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"tax": None}, id="tax-not-extracted"),
        pytest.param({}, id="tax-present"),
        pytest.param({"tax": Decimal("0.00"), "shipping": Decimal("10.00")}, id="shipping-present"),
        pytest.param({"tax": Decimal("0.00"), "discount": Decimal("10.00")}, id="discount-present"),
    ],
)
def test_line_items_sum_is_not_compared_with_the_total_otherwise(
    make_invoice: MakeInvoice, overrides: dict[str, Any]
) -> None:
    # Items add up to 100 and the total is 110: without a subtotal that is only a mismatch when the document
    # explicitly shows no tax, discount or shipping that could explain the difference.
    data = make_invoice(subtotal=None, tax_rate=None, total=Decimal("110.00"), **overrides)

    assert codes(validate(data)) == {IssueCode.MISSING_FIELD}


# ------------------------------------------------------------------------------------------ LINE_ITEM_MATH


def test_line_item_math_flagged(make_invoice: MakeInvoice) -> None:
    data = make_invoice(line_items=[item("2", "25.00", "50.00"), item("3", "10.00", "45.00")])

    [issue] = of_code(validate(data), IssueCode.LINE_ITEM_MATH)

    assert issue.severity is Severity.ERROR
    assert issue.field == "line_items"
    assert issue.details == {"index": 1, "expected": "30.00", "actual": "45.00", "difference": "15.00"}


def test_line_item_math_reports_every_bad_item(make_invoice: MakeInvoice) -> None:
    data = make_invoice(line_items=[item("1", "10.00", "11.00"), item("1", "10.00", "10.00"), item("2", "5", "11")])
    issues = of_code(validate(data), IssueCode.LINE_ITEM_MATH)
    assert [i.details["index"] for i in issues] == [0, 2]


@pytest.mark.parametrize(
    "line",
    [
        pytest.param(item("3", "3.33", "10.00"), id="a-cent-off"),
        pytest.param(item("100", "0.33", "33.33"), id="unit-price-rounding-scales-with-quantity"),
        pytest.param(item(None, "10.00", "99.00"), id="no-quantity"),
        pytest.param(item("2", None, "99.00"), id="no-unit-price"),
        pytest.param(item("2", "10.00", None), id="no-amount"),
    ],
)
def test_line_item_math_not_flagged(make_invoice: MakeInvoice, line: LineItem) -> None:
    assert IssueCode.LINE_ITEM_MATH not in codes(validate(make_invoice(line_items=[line])))


def test_line_item_math_flagged_beyond_the_quantity_scaled_allowance(make_invoice: MakeInvoice) -> None:
    # 100 x 0.33 = 33.00; allowance is 100 x 0.005 = 0.50, so 33.60 is out and 33.40 is in.
    out = validate(make_invoice(line_items=[item("100", "0.33", "33.60")]))
    inside = validate(make_invoice(line_items=[item("100", "0.33", "33.40")]))
    assert IssueCode.LINE_ITEM_MATH in codes(out)
    assert IssueCode.LINE_ITEM_MATH not in codes(inside)


# -------------------------------------------------------------------------------------- TAX_RATE_MISMATCH


def test_tax_rate_mismatch_flagged(make_invoice: MakeInvoice) -> None:
    [issue] = validate(make_invoice(tax_rate=Decimal("20")))

    assert issue.code is IssueCode.TAX_RATE_MISMATCH
    assert issue.severity is Severity.WARNING
    assert issue.field == "tax_rate"
    assert issue.details == {
        "stated_rate": "20.00",
        "expected": "20.00",
        "actual": "10.00",
        "difference": "10.00",
    }


@pytest.mark.parametrize(
    ("overrides", "other_codes"),
    [
        pytest.param({}, set(), id="rate-times-subtotal"),
        pytest.param(
            {"discount": Decimal("10.00"), "tax": Decimal("9.00"), "total": Decimal("99.00")},
            set(),
            id="rate-times-discounted-subtotal",
        ),
        pytest.param(
            {"discount": Decimal("-10.00"), "tax": Decimal("9.00"), "total": Decimal("99.00")},
            set(),
            id="negative-discount-is-a-magnitude",
        ),
        pytest.param(
            {"shipping": Decimal("20.00"), "tax": Decimal("12.00"), "total": Decimal("132.00")},
            set(),
            id="rate-times-subtotal-plus-shipping",
        ),
        pytest.param(
            {"tax_rate": Decimal("0"), "tax": Decimal("0"), "total": Decimal("100.00")}, set(), id="zero-rate"
        ),
        pytest.param(
            {"tax_rate": None, "tax": Decimal("20.00"), "total": Decimal("120.00")}, set(), id="no-stated-rate"
        ),
        pytest.param({"subtotal": None, "tax_rate": Decimal("20")}, {IssueCode.MISSING_FIELD}, id="no-subtotal"),
    ],
)
def test_tax_rate_mismatch_not_flagged(
    make_invoice: MakeInvoice, overrides: dict[str, Any], other_codes: set[IssueCode]
) -> None:
    assert codes(validate(make_invoice(**overrides))) == other_codes


@pytest.mark.parametrize(("tax", "flagged"), [("1004.00", False), ("1006.00", True)])
def test_tax_rate_tolerance_is_half_a_percent_of_the_tax(make_invoice: MakeInvoice, tax: str, flagged: bool) -> None:
    data = make_invoice(
        subtotal=Decimal("10000.00"),
        tax=Decimal(tax),
        total=Decimal("10000.00") + Decimal(tax),
        line_items=[],
    )
    expected = {IssueCode.MISSING_FIELD} | ({IssueCode.TAX_RATE_MISMATCH} if flagged else set())
    assert codes(validate(data)) == expected


# --------------------------------------------------------------------------------------- TAX_RATE_UNUSUAL


def test_tax_rate_unusual_flagged(make_invoice: MakeInvoice) -> None:
    data = make_invoice(tax=Decimal("40.00"), tax_rate=None, total=Decimal("140.00"))

    [issue] = of_code(validate(data), IssueCode.TAX_RATE_UNUSUAL)

    assert issue.severity is Severity.WARNING
    assert issue.field == "tax"
    assert issue.details == {"effective_rate": "40.00", "max_rate": "30.00"}


@pytest.mark.parametrize(
    ("overrides", "other_codes"),
    [
        pytest.param({"tax": Decimal("30.00"), "total": Decimal("130.00")}, set(), id="exactly-the-maximum"),
        pytest.param({}, set(), id="ten-percent"),
        pytest.param(
            {"tax": Decimal("40.00"), "shipping": Decimal("100.00"), "total": Decimal("240.00")},
            set(),
            id="rate-on-subtotal-plus-shipping-is-fine",
        ),
        pytest.param(
            {"subtotal": Decimal("0.00"), "tax": Decimal("5.00"), "total": Decimal("5.00"), "line_items": []},
            {IssueCode.MISSING_FIELD},
            id="zero-base",
        ),
    ],
)
def test_tax_rate_unusual_not_flagged(
    make_invoice: MakeInvoice, overrides: dict[str, Any], other_codes: set[IssueCode]
) -> None:
    assert codes(validate(make_invoice(tax_rate=None, **overrides))) == other_codes


def test_tax_rate_unusual_limit_is_configurable(make_invoice: MakeInvoice) -> None:
    assert IssueCode.TAX_RATE_UNUSUAL in codes(validate(make_invoice(), max_tax_rate=Decimal("5")))


# ---------------------------------------------------------------------------------------- LOW_CONFIDENCE


def test_low_confidence_flagged(make_invoice: MakeInvoice) -> None:
    meta = confident_meta()
    meta["total"] = FieldMeta(confidence=0.4, source=ExtractionSource.RULES)

    [issue] = validate(make_invoice(), meta)

    assert issue.code is IssueCode.LOW_CONFIDENCE
    assert issue.severity is Severity.WARNING
    assert issue.field == "total"
    assert issue.details == {"confidence": 0.4, "threshold": 0.6}


def test_low_confidence_not_flagged(make_invoice: MakeInvoice) -> None:
    assert validate(make_invoice(), confident_meta(0.6)) == []  # the threshold itself is acceptable
    assert validate(make_invoice(), {}) == []  # no metadata, no verdict

    not_core = confident_meta()
    not_core["due_date"] = FieldMeta(confidence=0.1, source=ExtractionSource.RULES)
    assert IssueCode.LOW_CONFIDENCE not in codes(validate(make_invoice(), not_core))

    absent = confident_meta()
    absent["customer_name"] = FieldMeta(confidence=0.1, source=ExtractionSource.RULES)
    issues = validate(make_invoice(customer_name=None), absent)
    assert IssueCode.LOW_CONFIDENCE not in codes(issues)
    assert IssueCode.MISSING_FIELD in codes(issues)


def test_low_confidence_threshold_is_configurable(make_invoice: MakeInvoice) -> None:
    meta = confident_meta(0.85)
    assert validate(make_invoice(), meta) == []
    assert codes(validate(make_invoice(), meta, confidence_threshold=0.9)) == {IssueCode.LOW_CONFIDENCE}


# ---------------------------------------------------------------------------------- metadata note issues


def test_extraction_disagreement_flagged(make_invoice: MakeInvoice) -> None:
    meta = confident_meta()
    meta["total"] = noted(0.9, MetaNote.EXTRACTION_DISAGREEMENT)
    meta["due_date"] = noted(0.9, MetaNote.EXTRACTION_DISAGREEMENT)  # any field, core or not

    issues = validate(make_invoice(), meta)

    assert [(i.code, i.severity, i.field) for i in issues] == [
        (IssueCode.EXTRACTION_DISAGREEMENT, Severity.WARNING, "due_date"),
        (IssueCode.EXTRACTION_DISAGREEMENT, Severity.WARNING, "total"),
    ]


def test_extraction_disagreement_not_flagged(make_invoice: MakeInvoice) -> None:
    meta = confident_meta()
    meta["total"] = noted(0.9, MetaNote.AMBIGUOUS_DATE_ORDER, MetaNote.LOW_OCR_CONFIDENCE)
    assert IssueCode.EXTRACTION_DISAGREEMENT not in codes(validate(make_invoice(), meta))


def test_ambiguous_date_flagged(make_invoice: MakeInvoice) -> None:
    meta = confident_meta()
    meta["invoice_date"] = noted(0.9, MetaNote.AMBIGUOUS_DATE_ORDER)

    [issue] = validate(make_invoice(), meta)

    assert (issue.code, issue.severity, issue.field) == (IssueCode.AMBIGUOUS_DATE, Severity.INFO, "invoice_date")


def test_ambiguous_date_not_flagged(make_invoice: MakeInvoice) -> None:
    assert IssueCode.AMBIGUOUS_DATE not in codes(validate(make_invoice()))


def test_currency_inferred_flagged(make_invoice: MakeInvoice) -> None:
    meta = confident_meta()
    meta["currency"] = noted(0.9, MetaNote.CURRENCY_INFERRED)

    [issue] = validate(make_invoice(), meta)

    assert (issue.code, issue.severity, issue.field) == (IssueCode.CURRENCY_INFERRED, Severity.INFO, "currency")


def test_currency_inferred_not_flagged(make_invoice: MakeInvoice) -> None:
    assert IssueCode.CURRENCY_INFERRED not in codes(validate(make_invoice()))


def test_ungrounded_value_rejected_flagged(make_invoice: MakeInvoice) -> None:
    # Grounding turned the supplier into None: it is missing (error) and the rejection is noted (info).
    meta = confident_meta()
    meta["supplier_name"] = noted(0.0, MetaNote.UNGROUNDED_VALUE_REJECTED)

    issues = validate(make_invoice(supplier_name=None), meta)

    assert [(i.code, i.severity, i.field) for i in issues] == [
        (IssueCode.MISSING_FIELD, Severity.ERROR, "supplier_name"),
        (IssueCode.UNGROUNDED_VALUE_REJECTED, Severity.INFO, "supplier_name"),
    ]


def test_ungrounded_value_rejected_not_flagged(make_invoice: MakeInvoice) -> None:
    assert IssueCode.UNGROUNDED_VALUE_REJECTED not in codes(validate(make_invoice(supplier_name=None)))


# ----------------------------------------------------------------------------------- document-level issues


def test_ocr_low_quality_flagged() -> None:
    [issue] = document_level_issues(
        ocr_confidence=0.6, duplicate_of_id=None, extraction_warnings=[], ocr_threshold=0.75
    )

    assert issue.code is IssueCode.OCR_LOW_QUALITY
    assert issue.severity is Severity.WARNING
    assert issue.field is None
    assert issue.details == {"mean_confidence": 0.6, "threshold": 0.75}


@pytest.mark.parametrize("confidence", [0.75, 0.9, None])
def test_ocr_low_quality_not_flagged(confidence: float | None) -> None:
    issues = document_level_issues(
        ocr_confidence=confidence, duplicate_of_id=None, extraction_warnings=[], ocr_threshold=0.75
    )
    assert issues == []


def test_duplicate_file_flagged() -> None:
    original = uuid4()

    [issue] = document_level_issues(
        ocr_confidence=None, duplicate_of_id=original, extraction_warnings=[], ocr_threshold=0.75
    )

    assert issue.code is IssueCode.DUPLICATE_FILE
    assert issue.severity is Severity.WARNING
    assert issue.details == {"duplicate_of_id": str(original)}


def test_duplicate_file_not_flagged() -> None:
    issues = document_level_issues(ocr_confidence=0.9, duplicate_of_id=None, extraction_warnings=[], ocr_threshold=0.75)
    assert issues == []


@pytest.mark.parametrize(
    "warning", [ExtractionWarning.LLM_UNAVAILABLE, ExtractionWarning.LLM_FAILED, ExtractionWarning.LLM_REFUSED]
)
def test_llm_unavailable_flagged(warning: ExtractionWarning) -> None:
    [issue] = document_level_issues(
        ocr_confidence=None, duplicate_of_id=None, extraction_warnings=[warning.value], ocr_threshold=0.75
    )

    assert issue.code is IssueCode.LLM_UNAVAILABLE
    assert issue.severity is Severity.INFO
    assert issue.details == {"reasons": [warning.value]}


def test_llm_unavailable_not_flagged() -> None:
    warnings = [ExtractionWarning.LLM_INPUT_TRUNCATED.value, ExtractionWarning.NO_LINE_ITEMS_TABLE.value]
    issues = document_level_issues(
        ocr_confidence=None, duplicate_of_id=None, extraction_warnings=warnings, ocr_threshold=0.75
    )
    assert issues == []


def test_several_llm_warnings_produce_one_issue() -> None:
    warnings = [ExtractionWarning.LLM_FAILED.value, ExtractionWarning.LLM_UNAVAILABLE.value, "llm_failed"]
    issues = document_level_issues(
        ocr_confidence=None, duplicate_of_id=None, extraction_warnings=warnings, ocr_threshold=0.75
    )
    assert [i.code for i in issues] == [IssueCode.LLM_UNAVAILABLE]
    assert issues[0].details == {"reasons": ["llm_failed", "llm_unavailable"]}


def test_document_level_issues_combine() -> None:
    issues = document_level_issues(
        ocr_confidence=0.1,
        duplicate_of_id=uuid4(),
        extraction_warnings=[ExtractionWarning.LLM_REFUSED.value],
        ocr_threshold=0.75,
    )
    assert [i.code for i in issues] == [
        IssueCode.OCR_LOW_QUALITY,
        IssueCode.DUPLICATE_FILE,
        IssueCode.LLM_UNAVAILABLE,
    ]
