"""Amount / date / currency / name normalization."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.extraction.labels import (
    LEDGER_COLUMNS,
    LINE_ITEM_COLUMNS,
    OTHER_DATE,
    find_labels,
    match_column,
    match_label,
)
from app.extraction.normalize import (
    amounts_equal,
    date_order_prior,
    detect_currency,
    dollar_country,
    find_dates,
    find_money_values,
    infer_date_order,
    infer_decimal_separator,
    names_match,
    normalize_invoice_number,
    normalize_name,
    parse_date,
    parse_money,
    parse_number,
    parse_percentage,
    quantize_money,
)

# --------------------------------------------------------------------------------------------------- money


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$1,234.56", "1234.56"),
        ("1.234,56 €", "1234.56"),
        ("€1.234,56", "1234.56"),
        ("1 234,56", "1234.56"),
        ("1 234,56 EUR", "1234.56"),
        ("1,23,456.00", "123456.00"),
        ("₹ 1,23,456.78", "123456.78"),
        ("(12.00)", "-12.00"),
        ("($12.00)", "-12.00"),
        ("-12.00", "-12.00"),
        ("−12.00", "-12.00"),
        ("12.00-", "-12.00"),
        ("-$45.00", "-45.00"),
        ("$-45.00", "-45.00"),
        ("USD 1,234", "1234"),
        ("12.5", "12.5"),
        ("Total: 110.00", "110.00"),
        ("CHF 1'234.50", "1234.50"),
        ("£ 99.99", "99.99"),
        ("1234,56", "1234.56"),
        ("0.1250", "0.1250"),
        ("AUD 2,500.00", "2500.00"),
        ("1,234,567.89", "1234567.89"),
        ("1.234.567,89", "1234567.89"),
        ("Amount due 500", "500"),
        ("EUR1.000,00", "1000.00"),
        ("Tax (10%): 2.00", "2.00"),
        ("Total for 2024: $1,234.00", "1234.00"),
        ("$2024", "2024"),
        ("C$ 45.10", "45.10"),
        ("Rs. 1,250.00", "1250.00"),
    ],
)
def test_parse_money_locales(text: str, expected: str) -> None:
    assert parse_money(text) == Decimal(expected)


def test_parse_money_preserves_printed_scale() -> None:
    assert str(parse_money("12.50")) == "12.50"
    assert str(parse_money("1.234,50")) == "1234.50"


@pytest.mark.parametrize(
    "text",
    [
        "2024",
        "10%",
        "8.25 %",
        "+1 (555) 123-4567",
        "555-123-4567",
        "+91 98765 43210",
        "05/03/2024",
        "2024-03-05",
        "INV-2024-0042",
        "A1B2",
        "",
        "abc",
        "10:30",
        "1234567890123",
    ],
)
def test_parse_money_rejects_non_amounts(text: str) -> None:
    assert parse_money(text) is None


def test_decimal_hint_resolves_three_digit_groups() -> None:
    assert parse_money("1.234") == Decimal("1.234")
    assert parse_money("1.234", decimal_hint=",") == Decimal("1234")
    assert parse_money("1,234") == Decimal("1234")
    assert parse_money("1,234", decimal_hint=",") == Decimal("1.234")


def test_find_money_values_in_line_item_row() -> None:
    values = find_money_values("2 x Coffee beans   4.50   9.00")
    assert [v.value for v in values] == [Decimal("2"), Decimal("4.50"), Decimal("9.00")]
    totals = find_money_values("Subtotal 1,234.56 Tax 123.46 Total $1,358.02")
    assert [v.value for v in totals] == [Decimal("1234.56"), Decimal("123.46"), Decimal("1358.02")]
    last = totals[-1]
    assert last.raw.strip().startswith("$") and last.currency == "USD"
    assert "1,358.02" in "Subtotal 1,234.56 Tax 123.46 Total $1,358.02"[last.start : last.end]


def test_find_money_values_skips_dates_phones_percent_ids() -> None:
    text = "Date 05/03/2024 Tel +1 (555) 123-4567 INV-2024-0042 VAT 20% Total 99.00"
    assert [v.value for v in find_money_values(text)] == [Decimal("99.00")]


@pytest.mark.parametrize(
    ("text", "amount"),
    [
        ("VAT 20%  300.00", "300.00"),  # the masked percentage right before the amount
        ("05/03/2024 100.00", "100.00"),  # the masked date
        ("10:32  45.50", "45.50"),  # the masked time
        ("Total  (USD)  12.00", "12.00"),
    ],
)
def test_money_match_start_and_raw_point_at_the_amount_itself(text: str, amount: str) -> None:
    """Masked characters before an amount never become part of it (they used to: raw "0%  300.00")."""
    (match,) = find_money_values(text)
    assert match.raw == amount
    assert match.start == text.index(amount)
    assert text[match.start : match.end] == amount


def test_money_match_keeps_its_own_sign_and_currency_marker() -> None:
    (negative,) = find_money_values("Discount 10%  -5.00")
    assert (negative.value, negative.raw) == (Decimal("-5.00"), "-5.00")
    (marked,) = find_money_values("VAT 20%  € 300,00", ",")
    assert (marked.value, marked.raw, marked.currency) == (Decimal("300.00"), "€ 300,00", "EUR")


@pytest.mark.parametrize(
    ("text", "hint", "expected"),
    [
        ("Total 1 234,56 €", ",", "1234.56"),
        ("Total 12 345 678,90", ",", "12345678.90"),
        ("Total 1 234.56", ".", "1234.56"),
        ("Total 1 234.56", None, "1234.56"),
    ],
)
def test_regular_space_thousands_groups_are_one_amount(text: str, hint: str | None, expected: str) -> None:
    values = find_money_values(text, hint)  # type: ignore[arg-type]
    assert [v.value for v in values] == [Decimal(expected)]
    assert values[0].raw.startswith(expected[0])  # raw is the printed text, regular spaces included
    assert " " in values[0].raw
    assert parse_money(text, hint) == Decimal(expected)  # type: ignore[arg-type]
    assert parse_number(text.removeprefix("Total ").removesuffix(" €"), hint) == Decimal(expected)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "text",
    [
        "Qty 12 345",  # no decimals: not the grouped-amount shape
        "Ref 1234 567,89",  # a first group of four digits is not a thousands group
        "Items 1 23,45",  # a group of two digits is not a thousands group
    ],
)
def test_space_grouping_is_joined_only_in_the_thousands_shape(text: str) -> None:
    assert len(find_money_values(text, ",")) == 2


@pytest.mark.parametrize(
    ("text", "order", "expected"),
    [
        ("2024-03-0510:32", "auto", date(2024, 3, 5)),
        ("Date: 2024-03-0510:32:07", "auto", date(2024, 3, 5)),
        ("05/03/202410:32", "DMY", date(2024, 3, 5)),
        ("2024-03-059:05 AM", "auto", date(2024, 3, 5)),
    ],
)
def test_a_date_glued_to_a_time_is_found(text: str, order: str, expected: date) -> None:
    """OCR drops the space between a date and a time ("2024-03-0510:32")."""
    parsed = parse_date(text, order=order)  # type: ignore[arg-type]
    assert parsed is not None and parsed.value == expected
    (match,) = find_dates(text, order=order)  # type: ignore[arg-type]
    assert text[match.start : match.end] in ("2024-03-05", "05/03/2024")


def test_a_glued_time_is_not_an_amount_and_other_digit_runs_are_not_dates() -> None:
    assert [v.value for v in find_money_values("Date 2024-03-0510:32  Total 10.00")] == [Decimal("10.00")]
    assert parse_date("2024-03-0512345") is None  # digits that are not a time keep the date from matching


def test_date_order_prior_follows_regional_conventions() -> None:
    assert date_order_prior(["Total $10.00"], "USD", ".") == "MDY"
    assert date_order_prior(["Total 10,00 €"], "EUR", ",") == "DMY"
    assert date_order_prior(["Total 10.00"], "GBP", ".") == "DMY"
    assert date_order_prior(["Total $10.00"], "CAD", ".") is None  # Canada writes both
    assert date_order_prior(["Total 10,00"], None, ",") == "DMY"
    assert date_order_prior(["Issued 5 March 2024", "Total $1"], "USD", ".") == "DMY"  # the document's own style
    assert date_order_prior(["Total 10"], None, None) is None


def test_dollar_country_needs_one_unmistakable_marker() -> None:
    assert dollar_country("ABN 12 345 678 901\nTotal $10.00") == "AUD"
    assert dollar_country("GST/HST 123456789 RT0001") == "CAD"
    assert dollar_country("Total $10.00") is None
    assert dollar_country("ABN 1\nHST 2") is None


def test_infer_decimal_separator() -> None:
    assert infer_decimal_separator(["1.234,56", "12,00"]) == ","
    assert infer_decimal_separator(["1,234.56", "Total 9.99"]) == "."
    assert infer_decimal_separator(["100", "250"]) is None


def test_parse_number_and_percentage() -> None:
    assert parse_number("2") == Decimal("2")
    assert parse_number("1,5 h", decimal_hint=",") == Decimal("1.5")
    assert parse_number("qty: 12") == Decimal("12")
    assert parse_number("none") is None
    assert parse_percentage("VAT 20%") == Decimal("20")
    assert parse_percentage("Tax (8.25%)") == Decimal("8.25")
    assert parse_percentage("GST @ 18 %") == Decimal("18")
    assert parse_percentage("IVA 21,5%") == Decimal("21.5")
    assert parse_percentage("no rate here") is None


def test_amounts_equal_tolerance_and_quantize() -> None:
    assert amounts_equal(Decimal("10.00"), Decimal("10.01"))
    assert amounts_equal(Decimal("10.00"), Decimal("10.02"))
    assert not amounts_equal(Decimal("10.00"), Decimal("10.03"))
    assert amounts_equal(None, None)
    assert not amounts_equal(None, Decimal("1"))
    assert amounts_equal(Decimal("1"), Decimal("1.5"), tolerance=Decimal("0.5"))
    assert quantize_money(Decimal("2.345")) == Decimal("2.35")
    assert quantize_money(Decimal("2.344")) == Decimal("2.34")


# --------------------------------------------------------------------------------------------------- dates


@pytest.mark.parametrize(
    ("text", "order", "expected", "ambiguous"),
    [
        ("2024-03-05", "auto", date(2024, 3, 5), False),
        ("2024/03/05", "auto", date(2024, 3, 5), False),
        ("25/03/2024", "auto", date(2024, 3, 25), False),
        ("03/25/2024", "auto", date(2024, 3, 25), False),
        ("05.03.2024", "auto", date(2024, 3, 5), False),
        ("05/03/2024", "auto", date(2024, 5, 3), True),
        ("05/03/2024", "DMY", date(2024, 3, 5), False),
        ("05/03/2024", "MDY", date(2024, 5, 3), False),
        ("05/05/2024", "auto", date(2024, 5, 5), False),
        ("05-03-24", "auto", date(2024, 5, 3), True),
        ("5 March 2024", "auto", date(2024, 3, 5), False),
        ("5th March, 2024", "auto", date(2024, 3, 5), False),
        ("March 5, 2024", "auto", date(2024, 3, 5), False),
        ("Mar 5 2024", "auto", date(2024, 3, 5), False),
        ("05-Mar-2024", "auto", date(2024, 3, 5), False),
        ("Dated: 5-Mar-24", "auto", date(2024, 3, 5), False),
        ("Sept 30, 2024", "auto", date(2024, 9, 30), False),
        ("Invoice Date: 12 Jan 2025", "auto", date(2025, 1, 12), False),
    ],
)
def test_parse_date_formats(text: str, order: str, expected: date, ambiguous: bool) -> None:
    parsed = parse_date(text, order=order)  # type: ignore[arg-type]
    assert parsed is not None, text
    assert parsed.value == expected
    assert parsed.ambiguous is ambiguous


@pytest.mark.parametrize("text", ["31/02/2024", "13/13/2024", "Invoice 2024", "", "Total 12.50", "1800-01-01"])
def test_parse_date_rejects_invalid(text: str) -> None:
    assert parse_date(text) is None


def test_parse_date_ambiguity_flag() -> None:
    parsed = parse_date("Invoice date 04/06/2024")
    assert parsed is not None and parsed.ambiguous and parsed.raw == "04/06/2024"


def test_find_dates_returns_every_date_with_positions() -> None:
    text = "Invoice Date: 05/03/2024   Due Date: 04/04/2024"
    found = find_dates(text, order="DMY")
    assert [m.parsed.value for m in found] == [date(2024, 3, 5), date(2024, 4, 4)]
    assert text[found[0].start : found[0].end] == "05/03/2024"


def test_infer_date_order_from_unambiguous_dates() -> None:
    assert infer_date_order(["05/03/2024", "25/03/2024"]) == "DMY"
    assert infer_date_order(["Invoice 03/05/2024", "due 03/25/2024"]) == "MDY"
    assert infer_date_order(["05/03/2024"]) is None
    assert infer_date_order(["2024-03-05", "5 March 2024"]) is None


# ------------------------------------------------------------------------------------------------ currency


@pytest.mark.parametrize(
    ("text", "default_dollar", "code", "inferred"),
    [
        ("Total: USD 1,234.00", "USD", "USD", False),
        ("Total 1.234,56 €", "USD", "EUR", False),
        ("Total $ 45.00", "USD", "USD", True),
        ("Total $45.00", "CAD", "CAD", True),
        ("C$ 45.00", "USD", "CAD", False),
        ("A$ 120.00", "USD", "AUD", False),
        ("Rs. 1,234", "USD", "INR", True),
        ("Currency: GBP", "USD", "GBP", False),
        ("Grand Total ₹ 500", "USD", "INR", False),
        ("PHP development services 100.00 USD", "USD", "USD", False),
        ("All amounts in EUR. Total 10,00", "USD", "EUR", False),
        ("£12.00", "USD", "GBP", False),
    ],
)
def test_detect_currency_priority(text: str, default_dollar: str, code: str, inferred: bool) -> None:
    guess = detect_currency(text, default_dollar=default_dollar)
    assert guess is not None, text
    assert guess.code == code
    assert guess.inferred is inferred
    assert 0.0 < guess.confidence <= 1.0
    if inferred:
        assert guess.confidence <= 0.7


def test_detect_currency_none_and_confidence_order() -> None:
    assert detect_currency("Amount 500") is None
    assert detect_currency("") is None
    explicit = detect_currency("Total EUR 10.00")
    bare = detect_currency("Total $10.00")
    assert explicit is not None and bare is not None and explicit.confidence > bare.confidence


# --------------------------------------------------------------------------------------- names and numbers


def test_normalize_and_match_names() -> None:
    assert normalize_name("ACME Corp.") == "acme"
    assert normalize_name("Acme Corporation") == "acme"
    assert normalize_name("Café Déjà Vu GmbH") == "cafe deja vu"
    assert normalize_name("Smith & Sons Pvt. Ltd.") == "smith and sons"
    assert normalize_name("The Widget Company") == "widget"
    assert normalize_name("Ltd") == "ltd"
    assert names_match("ACME Corp.", "Acme Corporation")
    assert names_match("Northwind Traders Inc", "Northwind Traders")
    assert not names_match("Globex Ltd", "Initech LLC")
    assert not names_match(None, "Acme")
    assert not names_match("", "")


def test_normalize_invoice_number() -> None:
    assert normalize_invoice_number("inv-2024/0042") == "INV20240042"
    assert normalize_invoice_number(" INV 2024 0042 ") == "INV20240042"
    assert normalize_invoice_number("#12345") == "12345"


# ---------------------------------------------------------------------------------------------------- labels


@pytest.mark.parametrize(
    ("text", "field"),
    [
        ("Invoice No: INV-1", "invoice_number"),
        ("Invoice #12345", "invoice_number"),
        ("INVOICE NUMBER", "invoice_number"),
        ("Inv. No. 77", "invoice_number"),
        ("Receipt #: 0042", "invoice_number"),
        ("Invoice Date: 2024-01-01", "invoice_date"),
        ("Date of issue", "invoice_date"),
        ("Dated", "invoice_date"),
        ("Date: 05/03/2024", "invoice_date"),
        ("Due Date: 2024-02-01", "due_date"),
        ("Payment due", "due_date"),
        ("Bill To:", "customer_name"),
        ("Billed to", "customer_name"),
        ("Invoice To", "customer_name"),
        ("Customer: Globex", "customer_name"),
        ("Sold To", "customer_name"),
        ("From:", "supplier_name"),
        ("Vendor", "supplier_name"),
        ("Supplier Name", "supplier_name"),
        ("Sub Total", "subtotal"),
        ("Subtotal:", "subtotal"),
        ("Net Amount", "subtotal"),
        ("Total excl. VAT", "subtotal"),
        ("Taxable Amount", "subtotal"),
        ("Discount", "discount"),
        ("Shipping & Handling", "shipping"),
        ("Freight", "shipping"),
        ("Sales Tax (8.25%)", "tax"),
        ("VAT 20%", "tax"),
        ("Total Tax", "tax"),
        ("CGST 9%", "tax"),
        ("Grand Total", "total"),
        ("Amount Due", "total"),
        ("Balance Due", "total"),
        ("Total incl. VAT", "total"),
        ("TOTAL", "total"),
        ("Total (EUR)", "total"),
        ("Currency", "currency"),
    ],
)
def test_find_labels_classifies(text: str, field: str) -> None:
    labels = find_labels(text)
    assert labels, text
    assert labels[0].field == field, (text, labels)


@pytest.mark.parametrize(
    "text",
    ["Tax Invoice", "VAT No: GB123456789", "GSTIN: 29ABCDE1234F1Z5", "Tax ID 12-3456789", "Payment terms: Net 30",
     "Customer ID: 4471", "Total Items: 5", "Delivery Date 2024-03-10"],
)
def test_find_labels_ignores_identifier_and_title_lookalikes(text: str) -> None:
    fields = {label.field for label in find_labels(text)}
    assert "tax" not in fields and "subtotal" not in fields and "customer_name" not in fields
    assert "total" not in fields and "invoice_date" not in fields


@pytest.mark.parametrize(
    "text",
    [
        "Date of Supply: 05/03/2024",
        "Supply Date 05/03/2024",
        "Tax Point Date: 05/03/2024",
        "Tax Point 05/03/2024",
        "PO Date: 2024-03-01",
        "Service Date 2024-03-01",
        "Delivery Date: 2024-03-10",
        "Order Date 2024-03-01",
        "Date of delivery: 2024-03-10",
    ],
)
def test_other_dates_are_never_invoice_date_labels(text: str) -> None:
    labels = find_labels(text)
    assert [label.field for label in labels] == [OTHER_DATE], (text, labels)


@pytest.mark.parametrize(
    ("text", "field", "label"),
    [
        ("Date Due: 2024-04-04", "due_date", "Date Due"),
        ("Pay To: ACME Corp", "supplier_name", "Pay To"),
        ("Remit To: ACME Corp", "supplier_name", "Remit To"),
        ("Make cheques payable to ACME Corp", "supplier_name", "Make cheques payable to"),
        ("BILL-TO: Globex", "customer_name", "BILL-TO"),
        ("SHIP-TO: Globex", "customer_name", "SHIP-TO"),
        ("Tax Total 10.00", "tax", "Tax Total"),
        ("VAT Total: 38.57", "tax", "VAT Total"),
        ("Net Amount Payable 110.00", "total", "Net Amount Payable"),
        ("Total (incl. VAT) 120.00", "total", "Total (incl. VAT)"),
        ("Amount incl. VAT 120.00", "total", "Amount incl. VAT"),
        ("Amount (excl. VAT) 100.00", "subtotal", "Amount (excl. VAT)"),
        ("Total (excl. VAT) 100.00", "subtotal", "Total (excl. VAT)"),
    ],
)
def test_compound_labels_are_one_label(text: str, field: str, label: str) -> None:
    (only,) = find_labels(text)
    assert (only.field, only.text) == (field, label)


def test_a_qualifier_that_states_the_tax_amount_keeps_its_tax_label() -> None:
    fields = [label.field for label in find_labels("Total (incl. VAT 20.00) 120.00")]
    assert fields == ["total", "tax"]


@pytest.mark.parametrize(
    ("text", "absent"),
    [
        ("Shipping Address: 12 Main St", "shipping"),
        ("Shipping Method: Express", "shipping"),
        ("Total Savings 5.00", "total"),
        ("Total pieces 12", "total"),
        ("GST included in total 1.82", "total"),
    ],
)
def test_captions_that_only_share_a_money_word_are_not_money_labels(text: str, absent: str) -> None:
    assert absent not in {label.field for label in find_labels(text)}


def test_gst_included_in_the_total_is_still_a_tax_label() -> None:
    assert [label.field for label in find_labels("GST included in total 1.82")] == ["tax"]


def test_find_labels_multiple_on_one_line_in_order() -> None:
    labels = find_labels("Invoice No: INV-9   Date: 2024-03-05   Due Date: 2024-04-04")
    assert [lbl.field for lbl in labels] == ["invoice_number", "invoice_date", "due_date"]
    assert labels[0].end <= labels[1].start


def test_match_label_only_at_start() -> None:
    hit = match_label("  Invoice Number")
    assert hit is not None and hit[0] == "invoice_number"
    assert match_label("Please pay the invoice total") is None


def test_match_column_ledger_vs_line_items() -> None:
    assert match_column("Invoice No.", LEDGER_COLUMNS) == "invoice_number"
    assert match_column("Amount", LEDGER_COLUMNS) == "total"
    assert match_column("Amount", LINE_ITEM_COLUMNS) == "amount"
    assert match_column("Unit Price", LINE_ITEM_COLUMNS) == "unit_price"
    assert match_column("Qty", LINE_ITEM_COLUMNS) == "quantity"
    assert match_column("Item Description", LINE_ITEM_COLUMNS) == "description"
    assert match_column("Vendor Name", LEDGER_COLUMNS) == "supplier_name"
    assert match_column("Notes", LEDGER_COLUMNS) is None
