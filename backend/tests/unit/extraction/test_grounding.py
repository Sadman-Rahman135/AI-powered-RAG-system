"""Grounding. A value the source text does not support is rejected to ``None`` (never kept, never guessed)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from app.domain.schemas import ExtractionSource, LineItem, MetaNote
from app.extraction.grounding import SourceIndex, ground_result
from tests.unit.extraction.llm_support import make_result

REJECTED = MetaNote.UNGROUNDED_VALUE_REJECTED.value

US_INVOICE = """\
Northwind Traders Inc.
Invoice No: INV-7
Invoice Date: March 5, 2024
Bill To: Contoso Ltd
Description        Qty   Unit Price    Amount
Widget A             2        25.00     50.00
Widget B             1        50.00     50.00
Subtotal                              100.00
Tax 10%                                10.00
Total USD                             110.00
"""

EU_INVOICE = """\
ACME Industrial Supplies GmbH
Rechnungsnummer: INV-2024/0042
Rechnungsdatum: 5 March 2024
Bill To: Globex Handels AG
Zwischensumme    1.234,56 €
MwSt 19 %          234,57 €
Gesamtbetrag     1.469,13 €
"""


# ------------------------------------------------------------------------------------------ the core contract


def test_grounding_rejects_hallucinated_values() -> None:
    result = make_result(
        invoice_number="INV-7", supplier_name="Globex", total=Decimal("999.00"), subtotal=Decimal("100.00")
    )

    grounded = ground_result(result, US_INVOICE)

    assert grounded.data.supplier_name is None  # "Globex" is not in the source
    assert grounded.data.total is None  # 999.00 is not in the source
    assert grounded.rejected["supplier_name"] == "Globex"
    assert grounded.rejected["total"] == Decimal("999.00")
    for name in ("supplier_name", "total"):
        assert grounded.meta[name].confidence == 0.0
        assert REJECTED in grounded.meta[name].notes
    # supported values survive untouched
    assert grounded.data.invoice_number == "INV-7"
    assert grounded.data.subtotal == Decimal("100.00")
    assert grounded.meta["subtotal"].confidence == 0.85
    assert REJECTED not in grounded.meta["subtotal"].notes
    assert "subtotal" not in grounded.rejected


def test_grounding_accepts_reformatted_values() -> None:
    result = make_result(
        subtotal=Decimal("1234.56"), total=Decimal("1469.13"), tax=Decimal("234.57"), invoice_date=date(2024, 3, 5)
    )

    grounded = ground_result(result, EU_INVOICE)  # "1.234,56 €" vs 1234.56, "5 March 2024" vs 2024-03-05

    assert grounded.data == result.data
    assert grounded.rejected == {}
    assert all(meta.confidence == 0.85 and meta.notes == [] for meta in grounded.meta.values())


def test_grounding_does_not_mutate_its_input() -> None:
    result = make_result(supplier_name="Globex", total=Decimal("999.00"))
    snapshot = result.model_dump()
    ground_result(result, US_INVOICE)
    assert result.model_dump() == snapshot


def test_grounding_is_idempotent() -> None:
    result = make_result(
        supplier_name="Globex",
        total=Decimal("110.00"),
        line_items=[LineItem(description="Widget A", amount=Decimal("50.00")), LineItem(amount=Decimal("999.00"))],
    )
    once = ground_result(result, US_INVOICE)
    twice = ground_result(once, US_INVOICE)
    assert twice.model_dump() == once.model_dump()


def test_grounding_accepts_a_prebuilt_source_index() -> None:
    index = SourceIndex(US_INVOICE)
    grounded = ground_result(make_result(total=Decimal("110.00"), tax=Decimal("11.00")), index)
    assert grounded.data.total == Decimal("110.00")
    assert grounded.data.tax is None


def test_rejection_without_existing_meta_still_leaves_a_note() -> None:
    result = make_result(total=Decimal("110.00"))
    result.data.supplier_name = "Globex"  # no FieldMeta for it
    grounded = ground_result(result, US_INVOICE)
    meta = grounded.meta["supplier_name"]
    assert (meta.confidence, meta.source, REJECTED in meta.notes) == (0.0, ExtractionSource.LLM, True)


def test_rejected_field_keeps_evidence_only_when_it_is_really_in_the_source() -> None:
    result = make_result(
        subtotal=Decimal("101.00"),
        total=Decimal("999.00"),
        evidence={"subtotal": "Subtotal   100.00", "total": "Grand total due 999.00"},
    )
    grounded = ground_result(result, US_INVOICE)
    assert grounded.meta["subtotal"].evidence == "Subtotal   100.00"  # a real line of the document
    assert grounded.meta["total"].evidence is None  # invented evidence is not kept


def test_overall_confidence_is_recomputed_after_rejections() -> None:
    result = make_result(invoice_number="INV-7", supplier_name="Globex", total=Decimal("999.00"))
    grounded = ground_result(result, US_INVOICE)
    assert grounded.overall_confidence < result.overall_confidence
    assert grounded.overall_confidence == pytest.approx(grounded.compute_overall_confidence())


def test_rules_results_keep_their_confidence_and_source() -> None:
    result = make_result("rules", confidence=0.93, total=Decimal("110.00"), supplier_name="Northwind Traders Inc.")
    grounded = ground_result(result, US_INVOICE)
    assert grounded.meta["total"].confidence == 0.93
    assert grounded.meta["total"].source is ExtractionSource.RULES
    assert grounded.method == "rules"


def test_other_result_fields_are_preserved() -> None:
    result = make_result(total=Decimal("999.00"), warnings=("llm_input_truncated",))
    result.llm_provider, result.llm_model, result.prompt_version = "local", "qwen", "v"
    grounded = ground_result(result, US_INVOICE)
    assert grounded.warnings == ["llm_input_truncated"]
    assert (grounded.llm_provider, grounded.llm_model, grounded.prompt_version) == ("local", "qwen", "v")


def test_whitespace_differences_in_the_source_do_not_matter() -> None:
    nbsp, thin = "\N{NO-BREAK SPACE}", "\N{THIN SPACE}"
    source = f"Northwind{nbsp}Traders{thin}Inc.\f\nInvoice\tNo:\r\nINV-7\n\n\nTotal{nbsp}110.00"
    grounded = ground_result(make_result(supplier_name="Northwind Traders Inc.", total=Decimal("110.00")), source)
    assert grounded.data.supplier_name == "Northwind Traders Inc."
    assert grounded.data.total == Decimal("110.00")


# ------------------------------------------------------------------------------------------------------ money


def test_money_must_equal_an_amount_in_the_source_to_half_a_cent() -> None:
    source = "Subtotal 100.00\nTax 10.00\nTotal 110.00"
    ok = ground_result(make_result(total=Decimal("110.004"), subtotal=Decimal("100")), source)
    assert ok.data.total == Decimal("110.004")
    assert ok.data.subtotal == Decimal("100")
    off = ground_result(make_result(total=Decimal("110.01")), source)
    assert off.data.total is None


def test_every_money_field_is_checked() -> None:
    source = "Subtotal 100.00\nDiscount 5.00\nShipping 7.50\nTax 10.00\nTotal 112.50"
    good = make_result(
        subtotal=Decimal("100"),
        discount=Decimal("5"),
        shipping=Decimal("7.5"),
        tax=Decimal("10"),
        total=Decimal("112.5"),
    )
    assert ground_result(good, source).rejected == {}
    bad = make_result(
        subtotal=Decimal("101"),
        discount=Decimal("6"),
        shipping=Decimal("8.5"),
        tax=Decimal("11"),
        total=Decimal("113.5"),
    )
    grounded = ground_result(bad, source)
    assert set(grounded.rejected) == {"subtotal", "discount", "shipping", "tax", "total"}


def test_a_printed_negative_may_be_reported_as_a_magnitude_but_a_sign_may_not_be_invented() -> None:
    credit = "Credit note\nTotal (110.00)\nDiscount -5.00"
    assert ground_result(make_result(total=Decimal("-110.00")), credit).data.total == Decimal("-110.00")
    assert ground_result(make_result(total=Decimal("110.00")), credit).data.total == Decimal("110.00")
    assert ground_result(make_result(discount=Decimal("5.00")), credit).data.discount == Decimal("5.00")
    printed_positive = "Total 110.00"
    assert ground_result(make_result(total=Decimal("-110.00")), printed_positive).data.total is None


def test_percentages_dates_and_bare_years_are_not_amounts() -> None:
    source = "Invoice 2024\nDate 5 March 2024\nVAT 20%\nTotal 110.00"
    for value in (Decimal("20"), Decimal("2024"), Decimal("5")):
        assert ground_result(make_result(total=value), source).data.total is None


def test_decimal_separator_of_the_document_decides_ambiguous_amounts() -> None:
    european = "Netto 1.000,00\nSumme 1.234,56\nZoll 1.234"  # "," is the decimal separator here
    assert ground_result(make_result(tax=Decimal("1234")), european).data.tax == Decimal("1234")
    assert ground_result(make_result(tax=Decimal("1.234")), european).data.tax is None


def test_ambiguous_amount_is_grounded_under_either_reading_when_the_document_gives_no_hint() -> None:
    source = "Total 1.234"
    assert ground_result(make_result(total=Decimal("1234")), source).data.total == Decimal("1234")
    assert ground_result(make_result(total=Decimal("1.234")), source).data.total == Decimal("1.234")


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_amounts_are_never_grounded(value: str) -> None:
    index = SourceIndex("Total 110.00")
    assert not index.supports_money(Decimal(value))
    assert not index.supports_number(Decimal(value))
    assert not index.supports_percent(Decimal(value))


def test_money_formats_of_the_source_are_understood() -> None:
    source = "A $1,234.56\nB 1 234,56\nC 1,23,456.00\nD CHF 1'234.50\nE ₹ 99.99"
    for value in ("1234.56", "123456.00", "1234.50", "99.99"):
        assert ground_result(make_result(total=Decimal(value)), source).data.total == Decimal(value)


# ------------------------------------------------------------------------------------------------------ dates


def test_dates_are_grounded_in_any_printed_format() -> None:
    source = "Invoice Date: 5 March 2024\nDue: 2024-04-04\nPaid on Jan 9th, 2024\nOrder 12.02.2024"
    result = make_result(invoice_date=date(2024, 3, 5), due_date=date(2024, 4, 4))
    assert ground_result(result, source).rejected == {}
    for d in (date(2024, 1, 9), date(2024, 2, 12)):
        assert ground_result(make_result(invoice_date=d), source).data.invoice_date == d


def test_date_not_in_the_source_is_rejected() -> None:
    grounded = ground_result(make_result(invoice_date=date(2024, 6, 1), due_date=date(2024, 3, 6)), US_INVOICE)
    assert grounded.data.invoice_date is None
    assert grounded.data.due_date is None
    assert grounded.rejected["invoice_date"] == date(2024, 6, 1)


def test_ambiguous_numeric_date_is_grounded_under_either_order_and_flagged() -> None:
    source = "Date: 03/04/2024"
    for reading in (date(2024, 3, 4), date(2024, 4, 3)):
        grounded = ground_result(make_result(invoice_date=reading), source)
        assert grounded.data.invoice_date == reading
        assert MetaNote.AMBIGUOUS_DATE_ORDER.value in grounded.meta["invoice_date"].notes
    assert ground_result(make_result(invoice_date=date(2024, 5, 3)), source).data.invoice_date is None


def test_document_wide_date_order_settles_ambiguous_dates() -> None:
    source = "Invoice date 03/04/2024\nDue 25/12/2024"  # 25/12 proves day-first
    grounded = ground_result(make_result(invoice_date=date(2024, 4, 3)), source)
    assert grounded.data.invoice_date == date(2024, 4, 3)
    assert MetaNote.AMBIGUOUS_DATE_ORDER.value not in grounded.meta["invoice_date"].notes
    assert ground_result(make_result(invoice_date=date(2024, 3, 4)), source).data.invoice_date is None


def test_unambiguous_numeric_date_has_a_single_reading() -> None:
    source = "Date: 25/12/2023"
    assert ground_result(make_result(invoice_date=date(2023, 12, 25)), source).data.invoice_date == date(2023, 12, 25)
    assert ground_result(make_result(invoice_date=date(2023, 12, 5)), source).data.invoice_date is None


# ----------------------------------------------------------------------------------------------- invoice number


@pytest.mark.parametrize(
    "value",
    [
        "INV-2024/0042",  # verbatim
        "inv-2024-0042",  # other separators and case
        "INV 2024 0042",
        "20240042",  # label prefix dropped
    ],
)
def test_invoice_number_matches_the_source_token_after_normalisation(value: str) -> None:
    source = "Invoice No: INV-2024/0042\nPO: 778899"
    grounded = ground_result(make_result(invoice_number=value), source)
    assert grounded.data.invoice_number == value


def test_invoice_number_may_be_spread_over_several_words_in_the_source() -> None:
    grounded = ground_result(make_result(invoice_number="INV-2024-0042"), "Invoice No: INV 2024 0042")
    assert grounded.data.invoice_number == "INV-2024-0042"


@pytest.mark.parametrize(
    "value",
    [
        "INV-999",  # not printed at all
        "42",  # short: only a whole token counts, never a fragment of INV20240042 / 0042
        "INV-2024",  # a truncation that cuts through the digit run
        "12345",  # a fragment of the longer number 1234567
        "2024-004",  # ends inside a digit run
    ],
)
def test_invoice_number_not_in_the_source_is_rejected(value: str) -> None:
    source = "Invoice No: INV-2024/0042\nPhone +1 555 123 4567 ref 1234567"
    grounded = ground_result(make_result(invoice_number=value), source)
    assert grounded.data.invoice_number is None
    assert grounded.rejected["invoice_number"] == value


def test_a_label_glued_to_the_number_still_grounds_it() -> None:
    assert ground_result(make_result(invoice_number="12345"), "Invoice No.12345").data.invoice_number == "12345"


def test_short_invoice_number_matches_a_whole_token() -> None:
    assert ground_result(make_result(invoice_number="42"), "Invoice # 42").data.invoice_number == "42"


def test_short_invoice_number_never_matches_a_fragment_of_a_longer_token() -> None:
    # "12" would pass the digit/letter boundary rule inside "NO12"; only the five-character minimum stops it
    grounded = ground_result(make_result(invoice_number="12"), "Invoice No.12")
    assert grounded.data.invoice_number is None


@pytest.mark.parametrize(
    ("value", "source"),
    [
        ("12345", "Ref 9912345"),  # starts inside a digit run
        ("B12345", "Ref AB12345"),  # starts inside a letter run
    ],
)
def test_invoice_number_fragment_must_not_start_inside_a_run(value: str, source: str) -> None:
    assert ground_result(make_result(invoice_number=value), source).data.invoice_number is None


# ---------------------------------------------------------------------------------------------------- names


NAME_SOURCE = """\
ACME Industrial Supplies Ltd
12 Main Street
Bill To:
Globex
International Trading Co
Invoice
Café Roma
3M Company
"""


@pytest.mark.parametrize(
    "value",
    [
        "ACME Industrial Supplies Ltd",
        "acme industrial supplies",  # case, legal suffix
        "ACME  Industrial\tSupplies",  # whitespace
        "ACME Industrial Suppiies",  # OCR noise, caught by the fuzzy match
        "Globex International Trading Co",  # wrapped over two lines in the source
        "Cafe Roma",  # accents
        "3M",  # short names need a whole-word match
    ],
)
def test_names_found_in_the_source_are_kept(value: str) -> None:
    for field in ("supplier_name", "customer_name"):
        fields: dict[str, Any] = {field: value}
        grounded = ground_result(make_result(**fields), NAME_SOURCE)
        assert getattr(grounded.data, field) == value


@pytest.mark.parametrize(
    "value",
    [
        "Initech",  # absent
        "Globex Invoice Services",  # contains the real line "Invoice" but is not in the source
        "Contoso International Holdings",  # contains the real word "International"
        "ACME Industrial Supplies Ltd and Sons Holdings International",  # a real name extended with invented words
        "AB",  # short names never match inside other words (CABLE)
        "Roma",  # a near miss of the printed "Rome"
    ],
)
def test_names_not_in_the_source_are_rejected(value: str) -> None:
    source = NAME_SOURCE.replace("Café Roma", "Cable Rome")
    grounded = ground_result(make_result(supplier_name=value), source)
    assert grounded.data.supplier_name is None
    assert grounded.rejected["supplier_name"] == value


def test_a_hallucinated_name_cannot_hide_behind_a_short_source_line() -> None:
    source = "Invoice\nTotal 10.00"
    assert ground_result(make_result(supplier_name="Globex Invoice Services"), source).data.supplier_name is None


# ------------------------------------------------------------------------------------------------- currency


@pytest.mark.parametrize(
    ("source", "code"),
    [
        ("Total: 110.00 EUR", "EUR"),
        ("Gesamt 1.234,50 €", "EUR"),
        ("Total £99.99", "GBP"),
        ("Total ₹ 1,23,456.00", "INR"),
        ("Total US$ 10.00", "USD"),
        ("Total CA$ 10.00", "CAD"),
        ("Prices in USD, e.g. $100", "USD"),
        ("Currency: CAD\nTotal 10.00", "CAD"),
        ("Total CAD 100.00", "CAD"),
        ("Invoice\nAmounts in SEK\nTotal 100,00", "SEK"),
    ],
)
def test_currency_is_grounded_by_its_code_or_symbol(source: str, code: str) -> None:
    grounded = ground_result(make_result(currency=code), source)
    assert grounded.data.currency == code
    assert MetaNote.CURRENCY_INFERRED.value not in grounded.meta["currency"].notes


@pytest.mark.parametrize(
    ("source", "code"),
    [
        ("Total 110.00", "USD"),  # nothing printed
        ("Total 110.00", "EUR"),
        ("Total £99.99", "EUR"),  # a different symbol
        ("CAD drawings, 3 pcs\nTotal 110.00", "CAD"),  # "CAD" as a word, not next to an amount
        ("Currency: CAD\nTotal $110.00", "USD"),  # the document names CAD; a bare $ does not ground USD
    ],
)
def test_currency_not_supported_by_the_source_is_rejected(source: str, code: str) -> None:
    grounded = ground_result(make_result(currency=code), source)
    assert grounded.data.currency is None
    assert grounded.rejected["currency"] == code


def test_a_bare_dollar_grounds_only_the_currency_detected_for_the_document() -> None:
    source = "Total $110.00"
    usd = ground_result(make_result(currency="USD"), source)
    assert usd.data.currency == "USD"
    assert MetaNote.CURRENCY_INFERRED.value in usd.meta["currency"].notes
    for code in ("CAD", "AUD"):
        assert ground_result(make_result(currency=code), source).data.currency is None
    # a deployment whose dollar is the Canadian dollar
    assert ground_result(make_result(currency="CAD"), source, default_dollar="CAD").data.currency == "CAD"
    assert ground_result(make_result(currency="USD"), source, default_dollar="CAD").data.currency is None


# ------------------------------------------------------------------------------------------------- tax rate


def test_tax_rate_must_be_printed_as_a_percentage() -> None:
    source = "VAT 7,5 % and 20%\nTotal 110.00"
    for value in ("20", "7.5"):
        assert ground_result(make_result(tax_rate=Decimal(value)), source).data.tax_rate == Decimal(value)
    assert ground_result(make_result(tax_rate=Decimal("19.98")), source).data.tax_rate is None
    assert ground_result(make_result(tax_rate=Decimal("110")), source).data.tax_rate is None  # an amount, not a rate


# ------------------------------------------------------------------------------------------------ line items


def test_line_items_with_a_grounded_amount_are_kept_in_order() -> None:
    items = [
        LineItem(description="Widget A", quantity=Decimal("2"), unit_price=Decimal("25.00"), amount=Decimal("50.00")),
        LineItem(description="Widget B", quantity=Decimal("1"), unit_price=Decimal("50.00"), amount=Decimal("50.00")),
    ]
    grounded = ground_result(make_result(line_items=items), US_INVOICE)
    assert grounded.data.line_items == items
    assert "line_items" not in grounded.rejected
    assert grounded.meta["line_items"].confidence == 0.85
    assert grounded.meta["line_items"].notes == []


def test_line_item_with_an_ungrounded_amount_is_dropped_into_rejected() -> None:
    kept = LineItem(description="Widget A", amount=Decimal("50.00"))
    invented = LineItem(description="Widget C", amount=Decimal("999.00"))
    no_amount = LineItem(description="Widget D", quantity=Decimal("2"))

    grounded = ground_result(make_result(line_items=[kept, invented, no_amount]), US_INVOICE)

    assert grounded.data.line_items == [kept]
    assert grounded.rejected["line_items"] == [invented, no_amount]
    assert REJECTED in grounded.meta["line_items"].notes
    assert grounded.meta["line_items"].confidence <= 0.75


def test_all_line_items_rejected_leaves_an_empty_list_and_a_zero_confidence_note() -> None:
    grounded = ground_result(make_result(line_items=[LineItem(amount=Decimal("999.00"))]), US_INVOICE)
    assert grounded.data.line_items == []
    assert grounded.meta["line_items"].confidence == 0.0
    assert REJECTED in grounded.meta["line_items"].notes


def test_unsupported_secondary_values_of_a_kept_line_item_are_nulled() -> None:
    items = [
        LineItem(description="Widget A", quantity=Decimal("2"), unit_price=Decimal("25.00"), amount=Decimal("50.00")),
        LineItem(
            description="Gadget Pro Max", quantity=Decimal("37"), unit_price=Decimal("33.33"), amount=Decimal("50.00")
        ),
    ]
    grounded = ground_result(make_result(line_items=items), US_INVOICE)
    assert grounded.data.line_items[0] == items[0]
    assert grounded.data.line_items[1] == LineItem(amount=Decimal("50.00"))
    assert grounded.rejected["line_items[1].description"] == "Gadget Pro Max"
    assert grounded.rejected["line_items[1].quantity"] == Decimal("37")
    assert grounded.rejected["line_items[1].unit_price"] == Decimal("33.33")
    assert "line_items" not in grounded.rejected  # nothing was dropped
    assert REJECTED in grounded.meta["line_items"].notes


# ------------------------------------------------------------------------------------- verbatim occurrence


def test_numbers_must_occur_as_whole_numbers() -> None:
    index = SourceIndex("Total 1.579,00 € and 579,00 net, 12,5 kg")
    assert index.occurs_number("1.579,00")
    assert index.occurs_number("579,00")
    assert index.occurs_number("12,5")
    assert not index.occurs_number("79,00")  # the tail of 579,00
    assert not index.occurs_number("1579.00")  # same value, not as printed
    assert not index.occurs_number("2,5")


def test_literals_and_currency_markers_occur_on_word_boundaries() -> None:
    index = SourceIndex("Date: 5 March 2024\nAmounts in EUR\nEUROPE tour 12,00 €")
    assert index.occurs_literal("5 March 2024")
    assert index.occurs_literal("5  march\t2024")  # whitespace and case do not matter
    assert not SourceIndex("Date: 15 March 2024").occurs_literal("5 March 2024")  # tail of a longer date
    assert index.occurs_currency("EUR")
    assert index.occurs_currency("€")
    assert not SourceIndex("EUROPE tour").occurs_currency("EUR")  # inside a longer word
    assert not SourceIndex("Total 1,00").occurs_currency("$")


def test_text_and_id_support_tiers() -> None:
    index = SourceIndex("ACME Industrial Supplies Ltd\nInvoice No: INV-2024/0042")
    assert index.text_support("ACME Industrial Supplies Ltd") == "exact"
    assert index.text_support("ACME Industrial Suppiies Ltd") == "tolerant"
    assert index.text_support("Initech Corporation") is None
    assert index.id_support("INV-2024/0042") == "exact"
    assert index.id_support("INV 2024 0042") == "tolerant"
    assert index.id_support("INV-9999/0042") is None


@pytest.mark.parametrize(
    ("value", "source", "tier"),
    [
        ("2024-0042", "Invoice No: INV-2024-0042-A", "partial"),  # 8 of 12 characters: most of the number is missing
        ("20240042", "Invoice No: INV-2024/0042", "partial"),  # the prefix dropped: 8 of 11
        ("INV-2024-0042", "Invoice No: INV-2024-0042A", "tolerant"),  # 11 of 12: covers most of the token
        ("INV-2024-0042-A", "Invoice No: INV-2024-0042-A", "exact"),
    ],
)
def test_a_fragment_of_a_longer_id_token_must_cover_most_of_it(value: str, source: str, tier: str) -> None:
    """A fragment that leaves much of the printed number out is only a partial match (scored below review)."""
    assert SourceIndex(source).id_support(value) == tier
    assert ground_result(make_result(invoice_number=value), source).data.invoice_number == value  # kept, not invented


@pytest.mark.parametrize(
    ("source", "prior"),
    [
        ("Rechnung 05/03/2024\nGesamt 10,00 €", "DMY"),
        ("Invoice 05/03/2024\nTotal $10.00", "MDY"),
        ("Invoice 05/03/2024\nABN 51 824 753 556\nTotal $10.00", "DMY"),  # an Australian dollar document
        ("Invoice 05/03/2024\nTotal 10.00", None),
    ],
)
def test_source_index_knows_the_regional_date_convention(source: str, prior: str | None) -> None:
    assert SourceIndex(source).order_prior == prior
    assert SourceIndex("Invoice 05/03/2024\n25/03/2024\nTotal $10.00").order_prior is None  # the document decides


def test_locate_finds_a_snippet_through_whitespace_differences() -> None:
    index = SourceIndex("page one\f  Total:\n  110.00\nthe end")
    position = index.locate("total: 110.00")
    assert position is not None
    assert index.text[:position].count("\f") == 1
    assert index.locate("not there") is None
