"""Deterministic, layout-aware invoice field extraction (no LLM)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from app.domain.schemas import (
    INVOICE_FIELDS,
    DocumentType,
    ExtractionSource,
    ExtractionWarning,
    MetaNote,
    field_is_present,
)
from app.extraction.rules import extract_invoice_rules, lines_from_text
from tests.unit.extraction.rules_layouts import (
    CANADA_GST_QST,
    DOTTED_LEADERS,
    FOOTER_LETTERHEAD,
    HASH_NUMBER,
    ITEMS_ABOVE_TAX_LINE,
    LAYOUTS,
    MULTIPAGE_P1,
    MULTIPAGE_P2,
    OCR_TYPOS,
    SPACE_GROUPED_EURO,
    TALLY_STYLE,
    US,
    as_lines,
    pages_as_lines,
    span_line,
)

MODES = ["text", "pdf", "ocr"]

# Conventions found while probing layouts beyond the eight core ones; each entry checks the fields it is about.
CONVENTIONS: dict[str, tuple[str, dict[str, Any]]] = {
    "dotted_leaders_and_net_amount_qualifier": (
        DOTTED_LEADERS,
        {
            "invoice_number": "BO/7781", "customer_name": "Seaside Resorts Ltd", "subtotal": Decimal("1000.00"),
            "tax": Decimal("150.00"), "tax_rate": Decimal("15"), "total": Decimal("1150.00"), "currency": "ZAR",
        },
    ),
    "letterhead_in_footer": (
        FOOTER_LETTERHEAD,
        {"supplier_name": "Northwind Consulting LLC", "customer_name": "Woodgrove Bank", "total": Decimal("1500.00")},
    ),
    "tally_captions_above_values": (
        TALLY_STYLE,
        {
            "invoice_number": "SI/2024-25/0087", "invoice_date": date(2024, 3, 5),
            "supplier_name": "Ganesh Hardware Stores", "customer_name": "Lakshmi Builders",
            "tax": Decimal("864.00"), "tax_rate": None, "total": Decimal("5664.00"), "currency": "INR",
        },
    ),
    "hash_invoice_number_and_balance_below_caption": (
        HASH_NUMBER,
        {"invoice_number": "INV-000123", "total": Decimal("1180.00"), "customer_name": "Ms. Mary D. Dunton"},
    ),
    "ocr_letter_digit_confusions_in_labels": (
        OCR_TYPOS,
        {"invoice_number": "BB-4410", "subtotal": Decimal("20.00"), "tax": Decimal("4.00"), "total": Decimal("24.00")},
    ),
    "items_without_header_above_tax_line": (
        ITEMS_ABOVE_TAX_LINE,
        {"invoice_number": "INV-1001", "supplier_name": "Bright Star Media Ltd", "customer_name": "Orion Retail Plc",
         "subtotal": None, "tax": Decimal("480.00"), "total": Decimal("2880.00"), "currency": "GBP"},
    ),
    "canadian_gst_plus_qst": (
        CANADA_GST_QST,
        {"invoice_number": "NL-2024-55", "customer_name": "Caribou Lodge", "tax": Decimal("89.85"),
         "total": Decimal("689.85"), "currency": "CAD"},
    ),
    "space_grouped_euro_and_hyphenated_bill_to": (
        SPACE_GROUPED_EURO,
        {"customer_name": "Société Générale de Test SARL", "subtotal": Decimal("13734.56"),
         "tax": Decimal("2746.91"), "total": Decimal("16481.47"), "currency": "EUR"},
    ),
}


def _check_meta(result: Any) -> None:
    for name in INVOICE_FIELDS:
        if field_is_present(result.data, name):
            meta = result.meta.get(name)
            assert meta is not None, f"no meta for {name}"
            assert meta.source == ExtractionSource.RULES
            assert meta.evidence and len(meta.evidence) <= 200, name
            assert meta.page is not None and meta.page >= 1, name
            assert 0.0 < meta.confidence <= 1.0, name
        else:
            assert name not in result.meta, f"meta for missing {name}"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_rules_extracts_core_fields(layout: str, mode: str) -> None:
    pages, expected = LAYOUTS[layout]
    lines, numbers = pages_as_lines(pages, mode)

    result = extract_invoice_rules(lines, page_numbers=numbers)

    data = result.data
    for name, value in expected.items():
        if name != "line_items":
            assert getattr(data, name) == value, f"{layout}/{mode}: {name}={getattr(data, name)!r}"
    assert [li.model_dump() for li in data.line_items] == expected["line_items"], f"{layout}/{mode}"
    assert result.method == "rules"
    assert 0.0 < result.overall_confidence <= 1.0
    assert ExtractionWarning.NO_LINE_ITEMS_TABLE not in result.warnings
    _check_meta(result)


def test_lines_from_text_pseudo_geometry() -> None:
    lines = lines_from_text("Total:   110.00\n\nInvoice No: 7")
    assert len(lines) == 2  # blank lines produce no Line
    first, second = lines
    assert [s.text for s in first.spans] == ["Total:", "110.00"]
    assert first.spans[0].x0 == 0.0 and first.spans[1].x0 == 9 * 6.0
    assert first.spans[0].y0 == 0.0 and second.spans[0].y0 == 2 * 12.0  # the blank line keeps its row
    assert first.text == "Total:  110.00"  # a column gap survives as a double space
    assert second.text == "Invoice No: 7"


def test_rules_missing_customer_is_null() -> None:
    text = "Fresh Produce Co.\nInvoice No: FP-100\nDate: 2024-05-02\nTotal: 54.20 EUR\n"
    result = extract_invoice_rules(lines_from_text(text))
    assert result.data.customer_name is None
    assert "customer_name" not in result.meta
    assert result.data.invoice_number == "FP-100"
    assert result.data.total == Decimal("54.20")
    assert result.data.currency == "EUR"


@pytest.mark.parametrize(
    ("text", "subtotal", "tax", "total"),
    [
        ("Sub Total        100.00\nTotal Tax         10.00\nTotal            110.00\n", "100.00", "10.00", "110.00"),
        ("Total excl. VAT    100,00 €\nVAT 10%             10,00 €\nTotal incl. VAT    110,00 €\n", "100", "10", "110"),
        ("Total: 110.00   Subtotal: 100.00   Tax: 10.00\n", "100.00", "10.00", "110.00"),
        ("Total (excl. VAT)   200.00\nVAT 20%              40.00\nTotal (incl. VAT)   240.00\n", "200", "40", "240"),
        ("Net Total      50.00\nTax Total       5.00\nGrand Total    55.00\n", "50", "5", "55"),
    ],
)
def test_rules_does_not_confuse_subtotal_and_total(text: str, subtotal: str, tax: str, total: str) -> None:
    data = extract_invoice_rules(lines_from_text("Invoice No: 1\n" + text)).data
    assert data.subtotal == Decimal(subtotal)
    assert data.tax == Decimal(tax)
    assert data.total == Decimal(total)


def test_rules_due_date_not_used_as_invoice_date() -> None:
    only_due = "Northwind Ltd\nInvoice No: N-5\nDue Date: 2024-04-30\nDelivery Date: 2024-04-02\nTotal: 10.00\n"
    result = extract_invoice_rules(lines_from_text(only_due))
    assert result.data.due_date == date(2024, 4, 30)
    assert result.data.invoice_date is None

    date_due = "Invoice No: N-6\nDate Due: 2024-04-30\nDate: 2024-04-01\nPO Date: 2024-03-01\nTotal: 10.00\n"
    data = extract_invoice_rules(lines_from_text(date_due)).data
    assert data.invoice_date == date(2024, 4, 1)
    assert data.due_date == date(2024, 4, 30)


@pytest.mark.parametrize("other", ["Date of Supply", "Supply Date", "Tax Point Date", "Tax Point", "Service Date"])
def test_rules_supply_and_tax_point_dates_are_not_the_invoice_date(other: str) -> None:
    """UK/EU VAT invoices print the date of supply (tax point) next to the issue date; it is another date."""
    labelled = f"Invoice No: N-9\n{other}: 2024-02-28\nDate: 2024-03-05\nTotal: 10.00\n"
    assert extract_invoice_rules(lines_from_text(labelled)).data.invoice_date == date(2024, 3, 5)
    # without an invoice-date label the supply date is not taken by the fallback either
    unlabelled = f"Northwind Ltd\nInvoice No: N-9\n{other}: 2024-02-28\nTotal: 10.00\n"
    assert extract_invoice_rules(lines_from_text(unlabelled)).data.invoice_date is None


def test_rules_invoice_date_fallback_is_low_confidence() -> None:
    text = "Northwind Ltd                    2024-04-01\nInvoice N-7\nDue Date: 2024-04-30\nTotal: 10.00\n"
    result = extract_invoice_rules(lines_from_text(text))
    assert result.data.invoice_date == date(2024, 4, 1)
    assert result.meta["invoice_date"].confidence <= 0.6
    assert result.data.due_date == date(2024, 4, 30)


def test_rules_currency_inferred_flag() -> None:
    text = "Invoice No: 7\nTotal: $45.00\n"
    result = extract_invoice_rules(lines_from_text(text))
    assert result.data.currency == "USD"
    meta = result.meta["currency"]
    assert MetaNote.CURRENCY_INFERRED.value in meta.notes
    assert meta.confidence <= 0.5
    assert extract_invoice_rules(lines_from_text(text), default_dollar="CAD").data.currency == "CAD"

    explicit = extract_invoice_rules(lines_from_text("Invoice No: 7\nTotal: EUR 45.00\n"))
    assert explicit.data.currency == "EUR"
    assert MetaNote.CURRENCY_INFERRED.value not in explicit.meta["currency"].notes


def test_rules_ambiguous_date_note() -> None:
    text = "Invoice No: 8\nDate: 04/06/2024\nTotal: $10.00\n"
    result = extract_invoice_rules(lines_from_text(text))
    assert result.data.invoice_date == date(2024, 4, 6)
    assert MetaNote.AMBIGUOUS_DATE_ORDER.value in result.meta["invoice_date"].notes
    assert result.meta["invoice_date"].confidence <= 0.7

    explicit = extract_invoice_rules(lines_from_text(text), date_order="DMY")
    assert explicit.data.invoice_date == date(2024, 6, 4)
    assert MetaNote.AMBIGUOUS_DATE_ORDER.value not in explicit.meta["invoice_date"].notes

    inferred = extract_invoice_rules(lines_from_text(text + "Due Date: 25/06/2024\n"))
    assert inferred.data.invoice_date == date(2024, 6, 4)  # 25/06 proves the document is day-first
    assert MetaNote.AMBIGUOUS_DATE_ORDER.value not in inferred.meta["invoice_date"].notes


def test_rules_meta_evidence_and_pages() -> None:
    lines, numbers = pages_as_lines([MULTIPAGE_P1, MULTIPAGE_P2], "pdf")
    result = extract_invoice_rules(lines, page_numbers=numbers)
    assert result.meta["invoice_number"].page == 1
    assert "CS-55120" in (result.meta["invoice_number"].evidence or "")
    assert result.meta["total"].page == 2
    assert "1,003.75" in (result.meta["total"].evidence or "")
    assert result.meta["line_items"].page == 1


def test_rules_tax_lines_summed_with_evidence() -> None:
    text = "Invoice No: 42\nSubtotal   100.00\nGST 5%       5.00\nPST 7%       7.00\nTotal      112.00\n"
    result = extract_invoice_rules(lines_from_text(text))
    assert result.data.tax == Decimal("12.00")
    assert result.data.tax_rate is None  # two different rates
    evidence = result.meta["tax"].evidence or ""
    assert "GST" in evidence and "PST" in evidence
    assert result.meta["total"].confidence >= 0.95  # 100 + 12 = 112


def test_rules_never_computes_missing_values() -> None:
    text = "Invoice No: 43\nSubtotal   100.00\nTotal      110.00\n"
    result = extract_invoice_rules(lines_from_text(text))
    assert result.data.tax is None  # 110 - 100 is not printed: not invented
    assert result.data.subtotal == Decimal("100.00") and result.data.total == Decimal("110.00")
    assert result.meta["total"].confidence < 0.95  # the arithmetic does not hold without the tax


def test_rules_arithmetic_is_evidence_for_confidence() -> None:
    consistent = extract_invoice_rules(lines_from_text("Subtotal  100.00\nTax  10.00\nTotal  110.00\n"))
    assert all(consistent.meta[f].confidence >= 0.95 for f in ("subtotal", "tax", "total"))
    inconsistent = extract_invoice_rules(lines_from_text("Subtotal  100.00\nTax  10.00\nTotal  120.00\n"))
    assert inconsistent.data.total == Decimal("120.00")  # printed values are kept as printed
    assert all(inconsistent.meta[f].confidence < 0.95 for f in ("subtotal", "tax", "total"))


def test_rules_table_header_total_is_not_a_total() -> None:
    text = (
        "Invoice No: 44\n"
        "Description          Qty     Price     Total\n"
        "Widget                 2     10.00     20.00\n"
    )
    result = extract_invoice_rules(lines_from_text(text))
    assert result.data.total is None
    assert len(result.data.line_items) == 1


def test_rules_registration_numbers_are_not_amounts() -> None:
    text = (
        "Maple Supplies Inc.\n"
        "GST/HST Registration No: 123456789 RT0001\n"
        "VAT No: GB123456789\n"
        "Invoice No: 45\n"
        "Total: $20.00\n"
    )
    data = extract_invoice_rules(lines_from_text(text)).data
    assert data.tax is None
    assert data.total == Decimal("20.00")


def test_rules_balance_due_after_payment_is_not_the_total() -> None:
    text = (
        "Invoice No: 46\n"
        "Subtotal      100.00\n"
        "Tax            10.00\n"
        "Total         110.00\n"
        "Amount Paid   110.00\n"
        "Balance Due     0.00\n"
    )
    data = extract_invoice_rules(lines_from_text(text)).data
    assert data.total == Decimal("110.00")


def test_rules_document_types() -> None:
    credit = extract_invoice_rules(lines_from_text("CREDIT NOTE\nCredit Note No: CN-9\nTotal: -50.00\n")).data
    assert credit.document_type == DocumentType.CREDIT_NOTE
    assert credit.invoice_number == "CN-9"
    assert credit.total == Decimal("-50.00")
    unknown = extract_invoice_rules(lines_from_text("Order summary\nTotal: 5.00\n")).data
    assert unknown.document_type == DocumentType.UNKNOWN


def test_rules_supplier_fallback_confidence_is_capped() -> None:
    result = extract_invoice_rules(as_lines(US, "pdf"))
    assert result.data.supplier_name == "ACME Corporation"
    assert result.meta["supplier_name"].confidence <= 0.65
    assert result.meta["customer_name"].confidence >= 0.8  # label + block below


def test_rules_low_ocr_confidence_lowers_field_confidence() -> None:
    result = extract_invoice_rules(as_lines(US, "ocr", conf=0.6))
    meta = result.meta["invoice_number"]
    assert result.data.invoice_number == "10042"
    assert MetaNote.LOW_OCR_CONFIDENCE.value in meta.notes
    assert meta.confidence <= 0.6
    clean = extract_invoice_rules(as_lines(US, "ocr", conf=0.97)).meta["invoice_number"]
    assert MetaNote.LOW_OCR_CONFIDENCE.value not in clean.notes


def test_rules_values_printed_above_their_captions() -> None:
    """Modern two-column layout: each value sits above a small caption, nothing shares character columns."""
    lines = [
        span_line(40, ("Lumen Studio Pty Ltd", 40), ("INVOICE", 420)),
        span_line(56, ("7 Albert Rd, Brisbane QLD 4000", 40)),
        span_line(100, ("BILLED TO", 40), ("LS-2077", 420)),
        span_line(114, ("Harbour Cafe", 40), ("Invoice Number", 420)),
        span_line(128, ("2 Pier St, Sydney NSW 2000", 40)),
        span_line(150, ("18 March 2024", 420)),
        span_line(164, ("Date of Issue", 420)),
        span_line(186, ("1 April 2024", 420)),
        span_line(200, ("Due Date", 420)),
        span_line(250, ("Description", 40), ("Qty", 300), ("Price", 360), ("Amount", 460)),
        span_line(266, ("Logo design", 40), ("1", 306), ("900.00", 357), ("900.00", 463)),
        span_line(310, ("Subtotal", 360), ("900.00", 463)),
        span_line(326, ("GST 10%", 360), ("90.00", 468)),
        span_line(342, ("Total AUD", 360), ("990.00", 463)),
    ]
    data = extract_invoice_rules(lines).data
    assert data.invoice_number == "LS-2077"
    assert data.invoice_date == date(2024, 3, 18)
    assert data.due_date == date(2024, 4, 1)
    assert data.customer_name == "Harbour Cafe"
    assert data.supplier_name == "Lumen Studio Pty Ltd"
    assert data.total == Decimal("990.00") and data.tax == Decimal("90.00")
    assert data.currency == "AUD"
    assert [li.amount for li in data.line_items] == [Decimal("900.00")]


def test_rules_no_table_warns_and_empty_input_is_safe() -> None:
    result = extract_invoice_rules(lines_from_text("Invoice No: 50\nTotal: 10.00\n"))
    assert result.data.line_items == []
    assert ExtractionWarning.NO_LINE_ITEMS_TABLE in result.warnings
    empty = extract_invoice_rules([])
    assert empty.data.total is None and empty.data.invoice_number is None
    assert empty.meta == {}
    assert ExtractionWarning.NO_LINE_ITEMS_TABLE in empty.warnings


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", list(CONVENTIONS))
def test_rules_real_world_conventions(case: str, mode: str) -> None:
    text, expected = CONVENTIONS[case]
    result = extract_invoice_rules(as_lines(text, mode))
    for name, value in expected.items():
        assert getattr(result.data, name) == value, f"{case}/{mode}: {name}={getattr(result.data, name)!r}"
    _check_meta(result)


def test_rules_footer_letterhead_is_low_confidence() -> None:
    meta = extract_invoice_rules(lines_from_text(FOOTER_LETTERHEAD)).meta["supplier_name"]
    assert meta.confidence <= 0.5


def test_rules_headerless_items_and_ocr_typo_rows() -> None:
    items = extract_invoice_rules(lines_from_text(ITEMS_ABOVE_TAX_LINE)).data.line_items
    assert [(i.description, i.amount) for i in items] == [("Services rendered in January", Decimal("2400.00"))]
    typo_items = extract_invoice_rules(lines_from_text(OCR_TYPOS)).data.line_items
    assert [i.description for i in typo_items] == ["Paper reams"]  # "Subtota1" ends the table


def test_rules_label_with_its_own_key_word() -> None:
    text = "Invoice No: 1\nShipping charges: $12.00\nDiscount amount: 5.00\nTotal: 107.00   Paid: 50.00\n"
    data = extract_invoice_rules(lines_from_text(text)).data
    assert (data.shipping, data.discount, data.total) == (Decimal("12.00"), Decimal("5.00"), Decimal("107.00"))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Invoice No: 9\nShipping Address: Unit 5, Dock Road\nTotal: 50.00\n", {"shipping": None, "total": "50.00"}),
        ("Invoice No: 9\nShipping Method: Express 2\nTotal: 50.00\n", {"shipping": None}),
        ("Subtotal  40.00\nTotal  50.00\nTotal Savings:  5.00\n", {"total": "50.00"}),
        ("TOTAL  $15.00\nGST included in total  1.36\n", {"total": "15.00", "tax": "1.36"}),
        ("Total 120.00 (incl. VAT 20.00)\n", {"total": "120.00", "tax": "20.00"}),
        ("Total (incl. VAT)  120.00\n", {"total": "120.00", "tax": None}),
        ("Discount: 10%\nTotal: 90.00\n", {"discount": None, "total": "90.00"}),
    ],
)
def test_rules_does_not_invent_amounts_from_lookalike_labels(text: str, expected: dict[str, str | None]) -> None:
    data = extract_invoice_rules(lines_from_text(text)).data
    for name, value in expected.items():
        assert getattr(data, name) == (Decimal(value) if value is not None else None), (name, text)
