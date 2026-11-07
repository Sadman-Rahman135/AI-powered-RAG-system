"""Cross-verification of two records of the same invoice (PDF vs PDF, PDF vs spreadsheet)."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest

from app.config import reset_settings_cache
from app.domain.schemas import InvoiceData, IssueCode, LineItem, Severity
from app.ingestion.types import FileKind
from app.validation.cross_verify import (
    ComparisonResult,
    FieldComparison,
    LineItemComparison,
    compare_invoices,
    comparison_kind,
    cross_verification_issue,
)

MakeInvoice = Callable[..., InvoiceData]

COMPARED_FIELDS = [
    "invoice_number",
    "invoice_date",
    "supplier_name",
    "customer_name",
    "subtotal",
    "tax",
    "total",
    "currency",
]


def by_field(result: ComparisonResult) -> dict[str, FieldComparison]:
    return {comparison.field: comparison for comparison in result.fields}


def items(*amounts: str) -> list[LineItem]:
    return [LineItem(description="x", amount=Decimal(amount)) for amount in amounts]


# ------------------------------------------------------------------------------------------ field status


def test_compare_identical_is_match(make_invoice: MakeInvoice) -> None:
    result = compare_invoices(make_invoice(), make_invoice())

    assert result.status == "match"
    assert [c.field for c in result.fields] == COMPARED_FIELDS
    assert all(c.status == "match" and c.note is None for c in result.fields)
    assert result.line_items == LineItemComparison(left_count=2, right_count=2, matched=2, status="match")


def test_compare_total_mismatch(make_invoice: MakeInvoice) -> None:
    result = compare_invoices(make_invoice(), make_invoice(total=Decimal("112.00")))

    assert result.status == "mismatch"
    total = by_field(result)["total"]
    assert (total.left, total.right, total.status, total.note) == ("110.00", "112.00", "mismatch", "difference 2.00")
    assert all(c.status == "match" for c in result.fields if c.field != "total")


@pytest.mark.parametrize(
    ("right_total", "status", "note"),
    [
        ("110.00", "match", None),
        ("110.02", "match", "within tolerance (difference 0.02)"),
        ("109.98", "match", "within tolerance (difference 0.02)"),
        ("110.03", "mismatch", "difference 0.03"),
    ],
)
def test_compare_amounts_use_the_tolerance(
    make_invoice: MakeInvoice, right_total: str, status: str, note: str | None
) -> None:
    total = by_field(compare_invoices(make_invoice(), make_invoice(total=Decimal(right_total))))["total"]
    assert (total.status, total.note) == (status, note)


def test_compare_tolerance_is_configurable(make_invoice: MakeInvoice) -> None:
    left, right = make_invoice(), make_invoice(total=Decimal("110.50"))
    assert compare_invoices(left, right).status == "mismatch"
    assert compare_invoices(left, right, tolerance=Decimal("0.50")).status == "match"


def test_compare_tolerance_defaults_to_the_setting(make_invoice: MakeInvoice, monkeypatch: pytest.MonkeyPatch) -> None:
    left, right = make_invoice(), make_invoice(total=Decimal("110.50"))
    assert compare_invoices(left, right).status == "mismatch"

    monkeypatch.setenv("AMOUNT_TOLERANCE", "0.50")
    reset_settings_cache()

    assert compare_invoices(left, right).status == "match"
    assert compare_invoices(left, right, tolerance=Decimal("0.02")).status == "mismatch"  # explicit argument wins


def test_compare_fuzzy_supplier_match(make_invoice: MakeInvoice) -> None:
    legal_form = compare_invoices(
        make_invoice(supplier_name="ACME Corp."), make_invoice(supplier_name="Acme Corporation")
    )
    typo = compare_invoices(
        make_invoice(customer_name="Globex Industries"), make_invoice(customer_name="Globex Industrie")
    )

    assert legal_form.status == "match"
    supplier = by_field(legal_form)["supplier_name"]
    assert (supplier.left, supplier.right, supplier.note) == (
        "ACME Corp.",
        "Acme Corporation",
        "equal after normalization",
    )
    assert typo.status == "match"
    assert by_field(typo)["customer_name"].note == "similar names"


def test_compare_identical_names_match_even_without_letters(make_invoice: MakeInvoice) -> None:
    # Name normalization strips punctuation, so "---" normalizes to nothing; identical strings must still agree.
    result = compare_invoices(make_invoice(supplier_name="---"), make_invoice(supplier_name="---"))

    assert (by_field(result)["supplier_name"].status, by_field(result)["supplier_name"].note) == ("match", None)


def test_compare_different_supplier_is_mismatch(make_invoice: MakeInvoice) -> None:
    result = compare_invoices(make_invoice(), make_invoice(supplier_name="Initech"))

    assert result.status == "mismatch"
    assert by_field(result)["supplier_name"].status == "mismatch"


def test_compare_invoice_number_uses_normalized_equality(make_invoice: MakeInvoice) -> None:
    same = compare_invoices(make_invoice(invoice_number="INV-2024/001"), make_invoice(invoice_number="inv 2024 001"))
    other = compare_invoices(make_invoice(), make_invoice(invoice_number="INV-2024-002"))

    assert by_field(same)["invoice_number"].status == "match"
    assert by_field(same)["invoice_number"].note == "equal after normalization"
    assert by_field(other)["invoice_number"].status == "mismatch"


def test_compare_dates_must_be_exact(make_invoice: MakeInvoice) -> None:
    result = compare_invoices(make_invoice(), make_invoice(invoice_date=date(2024, 3, 7)))

    invoice_date = by_field(result)["invoice_date"]
    assert (invoice_date.left, invoice_date.right) == ("2024-03-05", "2024-03-07")
    assert (invoice_date.status, invoice_date.note) == ("mismatch", "2 days apart")
    one_day = compare_invoices(make_invoice(), make_invoice(invoice_date=date(2024, 3, 6)))
    assert by_field(one_day)["invoice_date"].note == "1 day apart"


def test_compare_currency_must_be_exact(make_invoice: MakeInvoice) -> None:
    result = compare_invoices(make_invoice(), make_invoice(currency="EUR"))

    assert by_field(result)["currency"].status == "mismatch"
    assert result.status == "mismatch"


def test_compare_values_are_json_friendly_strings(make_invoice: MakeInvoice) -> None:
    result = compare_invoices(make_invoice(total=Decimal("110")), make_invoice(total=Decimal("110.00")))

    fields = by_field(result)
    assert fields["invoice_date"].left == "2024-03-05"
    assert (fields["total"].left, fields["total"].right) == ("110.00", "110.00")
    assert fields["total"].status == "match"
    assert all(c.left is None or isinstance(c.left, str) for c in result.fields)


# -------------------------------------------------------------------------------------- missing / partial


def test_compare_missing_side_is_partial(make_invoice: MakeInvoice) -> None:
    result = compare_invoices(make_invoice(), make_invoice(customer_name=None))

    assert result.status == "partial"
    customer = by_field(result)["customer_name"]
    assert (customer.left, customer.right, customer.status, customer.note) == (
        "Globex Ltd",
        None,
        "missing",
        "missing on right",
    )


def test_compare_missing_notes_name_the_side(make_invoice: MakeInvoice) -> None:
    result = compare_invoices(make_invoice(tax=None, subtotal=None), make_invoice(tax=None, currency=None))

    fields = by_field(result)
    assert fields["tax"].note == "missing on both sides"
    assert fields["subtotal"].note == "missing on left"
    assert fields["currency"].note == "missing on right"
    assert {c.status for c in result.fields if c.field in {"tax", "subtotal", "currency"}} == {"missing"}


def test_compare_blank_text_counts_as_missing(make_invoice: MakeInvoice) -> None:
    result = compare_invoices(make_invoice(), make_invoice(supplier_name="  "))
    assert by_field(result)["supplier_name"].status == "missing"


def test_compare_mismatch_outranks_missing(make_invoice: MakeInvoice) -> None:
    result = compare_invoices(make_invoice(), make_invoice(customer_name=None, total=Decimal("120.00")))

    assert result.status == "mismatch"
    assert by_field(result)["customer_name"].status == "missing"


# ----------------------------------------------------------------------------------------- line items


@pytest.mark.parametrize(
    ("left", "right", "expected", "overall"),
    [
        pytest.param(
            ["10.00", "20.00", "20.00"],
            ["20.00", "10.00", "20.00"],
            LineItemComparison(left_count=3, right_count=3, matched=3, status="match"),
            "match",
            id="same-multiset-any-order",
        ),
        pytest.param(
            ["10.00", "20.00", "30.00"],
            ["10.00", "20.00", "35.00"],
            LineItemComparison(left_count=3, right_count=3, matched=2, status="mismatch"),
            "mismatch",
            id="one-amount-differs",
        ),
        pytest.param(
            ["10.00", "20.00", "30.00"],
            ["10.00", "20.00"],
            LineItemComparison(left_count=3, right_count=2, matched=2, status="mismatch"),
            "mismatch",
            id="extra-item-on-left",
        ),
        pytest.param(
            ["10.00", "10.00"],
            ["10.00"],
            LineItemComparison(left_count=2, right_count=1, matched=1, status="mismatch"),
            "mismatch",
            id="multiplicity-matters",
        ),
        pytest.param(
            ["10.00", "20.00"],
            [],
            LineItemComparison(left_count=2, right_count=0, matched=0, status="missing"),
            "match",  # items that one side does not have do not make the comparison partial
            id="right-has-none",
        ),
        pytest.param(
            [],
            [],
            LineItemComparison(left_count=0, right_count=0, matched=0, status="missing"),
            "match",
            id="neither-has-any",
        ),
        pytest.param(
            ["10.001"],
            ["10.00"],
            LineItemComparison(left_count=1, right_count=1, matched=1, status="match"),
            "match",
            id="amounts-are-quantized-to-cents",
        ),
    ],
)
def test_compare_line_items_multiset(
    make_invoice: MakeInvoice, left: list[str], right: list[str], expected: LineItemComparison, overall: str
) -> None:
    result = compare_invoices(make_invoice(line_items=items(*left)), make_invoice(line_items=items(*right)))

    assert result.line_items == expected
    assert result.status == overall


def test_missing_line_items_do_not_make_the_result_partial(make_invoice: MakeInvoice) -> None:
    # A spreadsheet ledger row has no line items: PDF vs ledger row must be able to "match".
    result = compare_invoices(make_invoice(), make_invoice(line_items=[]))

    assert result.line_items.status == "missing"
    assert result.status == "match"
    # Only fields decide "partial", and differing items or fields still decide "mismatch".
    assert compare_invoices(make_invoice(), make_invoice(line_items=[], customer_name=None)).status == "partial"
    assert compare_invoices(make_invoice(), make_invoice(line_items=[], total=Decimal("112.00"))).status == "mismatch"
    assert compare_invoices(make_invoice(), make_invoice(line_items=items("1.00"))).status == "mismatch"


def test_compare_line_items_ignore_items_without_amount(make_invoice: MakeInvoice) -> None:
    left = make_invoice(line_items=[LineItem(description="Section heading"), *items("10.00")])
    right = make_invoice(line_items=items("10.00"))

    result = compare_invoices(left, right)

    assert result.line_items == LineItemComparison(left_count=1, right_count=1, matched=1, status="match")


def test_compare_survives_absurd_amounts(make_invoice: MakeInvoice) -> None:
    huge = Decimal("1E+30")  # too large to quantize to cents in the default decimal context
    result = compare_invoices(
        make_invoice(total=huge, line_items=items("1E+30")), make_invoice(total=huge, line_items=items("1E+30"))
    )

    assert result.status == "match"
    assert by_field(result)["total"].left == "1000000000000000000000000000000"


# ------------------------------------------------------------------------------------------------ JSON


def test_to_dict_is_json_ready(make_invoice: MakeInvoice) -> None:
    result = compare_invoices(make_invoice(), make_invoice(total=Decimal("112.00")))

    payload = result.to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert set(payload) == {"status", "fields", "line_items"}
    assert payload["status"] == "mismatch"
    fields = {entry["field"]: entry for entry in payload["fields"]}
    assert fields["total"] == {
        "field": "total",
        "left": "110.00",
        "right": "112.00",
        "status": "mismatch",
        "note": "difference 2.00",
    }
    assert payload["line_items"] == {"left_count": 2, "right_count": 2, "matched": 2, "status": "match"}


# ------------------------------------------------------------------------------------ comparison_kind


@pytest.mark.parametrize(
    ("left", "right", "kind"),
    [
        (FileKind.PDF, FileKind.PDF, "pdf_pdf"),
        (FileKind.PDF, FileKind.IMAGE, "pdf_pdf"),
        (FileKind.IMAGE, FileKind.IMAGE, "pdf_pdf"),
        (FileKind.PDF, FileKind.XLSX, "pdf_excel"),
        (FileKind.XLS, FileKind.PDF, "pdf_excel"),
        (FileKind.CSV, FileKind.IMAGE, "pdf_excel"),
        (FileKind.XLSX, FileKind.XLSX, "other"),
        (FileKind.XLSX, FileKind.CSV, "other"),
        (FileKind.XLS, FileKind.XLSX, "other"),
    ],
)
def test_comparison_kind(left: FileKind, right: FileKind, kind: str) -> None:
    assert comparison_kind(left, right) == kind


# ------------------------------------------------------------------------ CROSS_VERIFICATION_MISMATCH


def test_cross_verification_mismatch_flagged(make_invoice: MakeInvoice) -> None:
    result = compare_invoices(make_invoice(), make_invoice(total=Decimal("112.00"), line_items=items("1.00")))
    other_invoice, other_document = uuid4(), uuid4()

    issue = cross_verification_issue(result, other_invoice_id=other_invoice, other_document_id=other_document)

    assert issue is not None
    assert issue.code is IssueCode.CROSS_VERIFICATION_MISMATCH
    assert issue.severity is Severity.ERROR
    assert issue.field is None
    assert issue.details == {
        "other_invoice_id": str(other_invoice),
        "other_document_id": str(other_document),
        "mismatched_fields": ["total", "line_items"],
    }


def test_cross_verification_mismatch_not_flagged(make_invoice: MakeInvoice) -> None:
    ids = {"other_invoice_id": uuid4(), "other_document_id": uuid4()}
    matching = compare_invoices(make_invoice(), make_invoice())
    partial = compare_invoices(make_invoice(), make_invoice(customer_name=None))

    assert matching.status == "match" and cross_verification_issue(matching, **ids) is None
    assert partial.status == "partial" and cross_verification_issue(partial, **ids) is None
