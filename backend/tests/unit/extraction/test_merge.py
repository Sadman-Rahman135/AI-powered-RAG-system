"""Merging the rules result with the (grounded) LLM result, with arithmetic reconciliation."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.domain.schemas import (
    DocumentType,
    ExtractionResult,
    ExtractionSource,
    ExtractionWarning,
    FieldMeta,
    LineItem,
    MetaNote,
)
from app.extraction.merge import RULES_GUESS_OVERRULED, merge_results
from app.validation.rules import validate_invoice
from tests.unit.extraction.llm_support import make_result

DISAGREEMENT = MetaNote.EXTRACTION_DISAGREEMENT.value
REJECTED = MetaNote.UNGROUNDED_VALUE_REJECTED.value
D = Decimal


def items(*amounts: str) -> list[LineItem]:
    return [LineItem(description=f"Item {number}", amount=D(amount)) for number, amount in enumerate(amounts, 1)]


# ------------------------------------------------------------------------------------------------ agreement


def test_merge_agreement_boosts_confidence() -> None:
    rules = make_result(
        "rules",
        confidence=0.9,
        invoice_number="INV-7",
        total=D("110.00"),
        invoice_date=date(2024, 3, 5),
        currency="USD",
    )
    llm = make_result(
        "llm", confidence=0.85, invoice_number="INV-7", total=D("110.00"), invoice_date=date(2024, 3, 5), currency="USD"
    )

    merged = merge_results(rules, llm)

    assert merged.method == "hybrid"
    for name in ("invoice_number", "total", "invoice_date", "currency"):
        meta = merged.meta[name]
        assert meta.source is ExtractionSource.HYBRID
        assert meta.confidence == pytest.approx(0.99)  # min(0.99, max(0.9, 0.85) + 0.1)
        assert DISAGREEMENT not in meta.notes
    assert merged.data.total == D("110.00")


def test_agreement_boost_is_max_plus_a_tenth_capped_at_099() -> None:
    low = merge_results(
        make_result("rules", confidence=0.5, total=D("1")), make_result("llm", confidence=0.6, total=D("1"))
    )
    assert low.meta["total"].confidence == pytest.approx(0.7)
    high = merge_results(
        make_result("rules", confidence=0.95, total=D("1")), make_result("llm", confidence=0.99, total=D("1"))
    )
    assert high.meta["total"].confidence == pytest.approx(0.99)


def test_agreement_keeps_the_evidence_of_the_more_confident_side() -> None:
    rules = make_result("rules", confidence=0.9, total=D("110"), evidence={"total": "Total 110.00"})
    llm = make_result("llm", confidence=0.85, total=D("110"), evidence={"total": "110.00"})
    assert merge_results(rules, llm).meta["total"].evidence == "Total 110.00"
    assert merge_results(llm, rules).meta["total"].evidence == "Total 110.00"  # confidence decides, not position
    tied_rules = make_result("rules", confidence=0.8, total=D("110"), evidence={"total": "Total 110.00"})
    tied_llm = make_result("llm", confidence=0.8, total=D("110"), evidence={"total": "110.00"})
    assert merge_results(tied_rules, tied_llm).meta["total"].evidence == "Total 110.00"  # a tie goes to the rules


def test_agreement_unions_notes_and_keeps_the_page() -> None:
    rules = make_result("rules", invoice_date=date(2024, 3, 4))
    llm = make_result("llm", invoice_date=date(2024, 3, 4))
    rules.meta["invoice_date"] = FieldMeta(
        confidence=0.9, source=ExtractionSource.RULES, page=2, notes=[MetaNote.AMBIGUOUS_DATE_ORDER.value]
    )
    meta = merge_results(rules, llm).meta["invoice_date"]
    assert meta.notes == [MetaNote.AMBIGUOUS_DATE_ORDER.value]
    assert meta.page == 2


@pytest.mark.parametrize(
    ("rules_name", "llm_name"),
    [("ACME Corp.", "Acme Corporation"), ("The Bakery Ltd", "bakery"), ("Müller GmbH", "Muller GmbH")],
)
def test_names_that_differ_only_in_form_agree(rules_name: str, llm_name: str) -> None:
    merged = merge_results(make_result("rules", supplier_name=rules_name), make_result("llm", supplier_name=llm_name))
    assert merged.meta["supplier_name"].source is ExtractionSource.HYBRID
    assert DISAGREEMENT not in merged.meta["supplier_name"].notes


def test_invoice_numbers_agree_after_normalisation() -> None:
    merged = merge_results(
        make_result("rules", confidence=0.9, invoice_number="INV-2024/0042"),
        make_result("llm", confidence=0.85, invoice_number="inv 2024 0042"),
    )
    assert merged.meta["invoice_number"].source is ExtractionSource.HYBRID
    assert merged.data.invoice_number == "INV-2024/0042"  # the more confident side's spelling


def test_amounts_agree_to_half_a_cent_not_to_the_validation_tolerance() -> None:
    near = merge_results(make_result("rules", total=D("100.00")), make_result("llm", total=D("100.004")))
    assert near.meta["total"].source is ExtractionSource.HYBRID
    apart = merge_results(make_result("rules", total=D("100.00")), make_result("llm", total=D("100.01")))
    assert DISAGREEMENT in apart.meta["total"].notes


def test_discount_agrees_regardless_of_the_sign_convention() -> None:
    merged = merge_results(make_result("rules", discount=D("-10.00")), make_result("llm", discount=D("10.00")))
    assert merged.meta["discount"].source is ExtractionSource.HYBRID
    assert DISAGREEMENT not in merged.meta["discount"].notes


# ----------------------------------------------------------------------------------------- one side missing


def test_merge_fills_missing_from_other() -> None:
    rules = make_result("rules", confidence=0.9, invoice_number="INV-7", total=D("110.00"))
    llm = make_result("llm", confidence=0.85, customer_name="Contoso Ltd", tax_rate=D("10"), currency="USD")

    merged = merge_results(rules, llm)

    assert merged.data.invoice_number == "INV-7"
    assert merged.data.customer_name == "Contoso Ltd"
    assert merged.data.tax_rate == D("10")
    assert merged.data.currency == "USD"
    assert merged.meta["invoice_number"].source is ExtractionSource.RULES
    assert merged.meta["invoice_number"].confidence == 0.9
    assert merged.meta["customer_name"].source is ExtractionSource.LLM
    assert merged.meta["customer_name"].confidence == 0.85
    assert all(DISAGREEMENT not in meta.notes for meta in merged.meta.values())


def test_a_value_without_meta_gets_a_neutral_meta_from_its_side() -> None:
    rules = make_result("rules", total=D("110.00"))
    del rules.meta["total"]
    merged = merge_results(rules, make_result("llm", invoice_number="INV-7"))
    assert merged.meta["total"].source is ExtractionSource.RULES
    assert merged.meta["total"].confidence == 0.5


def test_a_rejected_llm_value_is_replaced_by_the_rules_value_without_a_note() -> None:
    rules = make_result("rules", confidence=0.9, supplier_name="ACME Corp")
    llm = make_result("llm", invoice_number="INV-7")
    llm.rejected["supplier_name"] = "Globex"
    llm.meta["supplier_name"] = FieldMeta(confidence=0.0, source=ExtractionSource.LLM, notes=[REJECTED])

    merged = merge_results(rules, llm)

    assert merged.data.supplier_name == "ACME Corp"
    assert merged.meta["supplier_name"].source is ExtractionSource.RULES
    assert REJECTED not in merged.meta["supplier_name"].notes
    assert merged.rejected["supplier_name"] == "Globex"  # still on record


def test_a_field_nobody_found_keeps_the_rejection_meta_so_validation_can_report_it() -> None:
    rules = make_result("rules", invoice_number="INV-7")
    llm = make_result("llm", invoice_number="INV-7")
    llm.rejected["total"] = D("999.00")
    llm.meta["total"] = FieldMeta(confidence=0.0, source=ExtractionSource.LLM, notes=[REJECTED])

    merged = merge_results(rules, llm)

    assert merged.data.total is None
    assert merged.meta["total"].confidence == 0.0
    assert REJECTED in merged.meta["total"].notes
    assert merged.rejected == {"total": D("999.00")}


# ----------------------------------------------------------------------------------------- disagreement


def test_merge_flags_disagreement() -> None:
    rules = make_result("rules", confidence=0.9, invoice_date=date(2024, 3, 5), supplier_name="ACME Corp")
    llm = make_result("llm", confidence=0.85, invoice_date=date(2024, 3, 6), supplier_name="Globex Inc")

    merged = merge_results(rules, llm)

    date_meta, name_meta = merged.meta["invoice_date"], merged.meta["supplier_name"]
    assert DISAGREEMENT in date_meta.notes and DISAGREEMENT in name_meta.notes
    assert date_meta.confidence == pytest.approx(0.9 * 0.7)  # the chosen (rules) value, times 0.7
    assert name_meta.confidence == pytest.approx(0.85 * 0.7)  # the chosen (llm) value, times 0.7
    assert date_meta.source is ExtractionSource.RULES
    assert name_meta.source is ExtractionSource.LLM


def test_names_prefer_the_llm() -> None:
    merged = merge_results(
        make_result("rules", confidence=0.95, customer_name="Bill To: Contoso"),
        make_result("llm", confidence=0.75, customer_name="Contoso Ltd"),
    )
    assert merged.data.customer_name == "Contoso Ltd"


def test_invoice_number_and_dates_prefer_confident_rules() -> None:
    rules = make_result(
        "rules", confidence=0.8, invoice_number="INV-7", invoice_date=date(2024, 3, 5), due_date=date(2024, 4, 5)
    )
    llm = make_result(
        "llm", confidence=0.99, invoice_number="INV-8", invoice_date=date(2024, 3, 6), due_date=date(2024, 4, 6)
    )
    merged = merge_results(rules, llm)
    assert merged.data.invoice_number == "INV-7"
    assert merged.data.invoice_date == date(2024, 3, 5)
    assert merged.data.due_date == date(2024, 4, 5)


def test_invoice_number_and_dates_follow_confidence_when_rules_are_unsure() -> None:
    rules = make_result("rules", confidence=0.79, invoice_number="INV-7", invoice_date=date(2024, 3, 5))
    llm = make_result("llm", confidence=0.85, invoice_number="INV-8", invoice_date=date(2024, 3, 6))
    merged = merge_results(rules, llm)
    assert merged.data.invoice_number == "INV-8"
    assert merged.data.invoice_date == date(2024, 3, 6)
    better_rules = merge_results(
        make_result("rules", confidence=0.7, invoice_number="INV-7"),
        make_result("llm", confidence=0.6, invoice_number="INV-8"),
    )
    assert better_rules.data.invoice_number == "INV-7"


def test_other_fields_follow_confidence_and_ties_go_to_rules() -> None:
    llm_wins = merge_results(
        make_result("rules", confidence=0.5, currency="USD", tax_rate=D("10")),
        make_result("llm", confidence=0.85, currency="CAD", tax_rate=D("5")),
    )
    assert (llm_wins.data.currency, llm_wins.data.tax_rate) == ("CAD", D("5"))
    tie = merge_results(
        make_result("rules", confidence=0.8, currency="USD"), make_result("llm", confidence=0.8, currency="CAD")
    )
    assert tie.data.currency == "USD"


def test_document_type_unknown_counts_as_missing() -> None:
    rules = make_result("rules", total=D("1"))
    llm = make_result("llm", confidence=0.75, document_type=DocumentType.CREDIT_NOTE)
    rules.meta["document_type"] = FieldMeta(confidence=0.9, source=ExtractionSource.RULES)  # a meta for an unknown type
    unknown_is_missing = merge_results(rules, llm)
    assert unknown_is_missing.data.document_type is DocumentType.CREDIT_NOTE
    assert unknown_is_missing.meta["document_type"].source is ExtractionSource.LLM
    assert unknown_is_missing.meta["document_type"].confidence == 0.75  # not flagged, not scaled
    assert DISAGREEMENT not in unknown_is_missing.meta["document_type"].notes
    both = merge_results(
        make_result("rules", document_type=DocumentType.INVOICE), make_result("llm", document_type=DocumentType.INVOICE)
    )
    assert both.meta["document_type"].source is ExtractionSource.HYBRID
    clash = merge_results(
        make_result("rules", confidence=0.6, document_type=DocumentType.INVOICE),
        make_result("llm", confidence=0.75, document_type=DocumentType.RECEIPT),
    )
    assert clash.data.document_type is DocumentType.RECEIPT
    assert DISAGREEMENT in clash.meta["document_type"].notes


# ----------------------------------------------------------------------------- arithmetic reconciliation


def test_merge_disagreement_prefers_arithmetically_consistent_set() -> None:
    rules = make_result("rules", confidence=0.9, subtotal=D("100.00"), tax=D("10.00"), total=D("150.00"))
    llm = make_result("llm", confidence=0.85, subtotal=D("100.00"), tax=D("10.00"), total=D("110.00"))

    merged = merge_results(rules, llm)

    assert merged.data.total == D("110.00")  # 100 + 10 = 110, although the rules were more confident
    assert merged.meta["total"].source is ExtractionSource.LLM
    assert DISAGREEMENT in merged.meta["total"].notes
    assert merged.meta["subtotal"].source is ExtractionSource.HYBRID


def test_reconciliation_may_take_different_fields_from_different_sides() -> None:
    rules = make_result("rules", confidence=0.9, subtotal=D("100.00"), tax=D("15.00"), total=D("110.00"))
    llm = make_result("llm", confidence=0.85, subtotal=D("100.00"), tax=D("10.00"), total=D("150.00"))

    merged = merge_results(rules, llm)

    assert (merged.data.subtotal, merged.data.tax, merged.data.total) == (D("100.00"), D("10.00"), D("110.00"))
    assert merged.meta["tax"].source is ExtractionSource.LLM
    assert merged.meta["total"].source is ExtractionSource.RULES
    assert DISAGREEMENT in merged.meta["tax"].notes and DISAGREEMENT in merged.meta["total"].notes


def test_reconciliation_counts_discount_and_shipping() -> None:
    rules = make_result(
        "rules", confidence=0.9, subtotal=D("100"), discount=D("10"), shipping=D("5"), tax=D("9.5"), total=D("104.50")
    )
    llm = make_result(
        "llm", confidence=0.85, subtotal=D("100"), discount=D("10"), shipping=D("5"), tax=D("9.5"), total=D("114.50")
    )
    assert merge_results(rules, llm).data.total == D("104.50")  # 100 - 10 + 5 + 9.5


def test_reconciliation_treats_a_negative_discount_as_its_magnitude() -> None:
    # the more confident side prints the discount as -10, and 100 - |-10| = 90 is the rules' total (not 100 + 10)
    rules = make_result("rules", confidence=0.9, subtotal=D("100"), discount=D("-10"), total=D("90"))
    llm = make_result("llm", confidence=0.85, subtotal=D("100"), discount=D("10"), total=D("110"))
    merged = merge_results(rules, llm)
    assert merged.data.total == D("90")
    assert merged.meta["discount"].source is ExtractionSource.HYBRID


def test_equally_good_combinations_prefer_the_one_with_more_values_from_the_rules() -> None:
    """Discount, shipping and tax are all disputed; two combinations balance 100 - discount + shipping + tax = 100."""
    rules = make_result(
        "rules", confidence=0.8, subtotal=D("100"), discount=D("10"), shipping=D("1"), tax=D("3"), total=D("100")
    )
    llm = make_result(
        "llm", confidence=0.8, subtotal=D("100"), discount=D("4"), shipping=D("5"), tax=D("5"), total=D("100")
    )
    # (rules discount, llm shipping, llm tax) is first in search order with one rules value;
    # (llm discount, rules shipping, rules tax) has two and must win the tie
    merged = merge_results(rules, llm)
    assert (merged.data.discount, merged.data.shipping, merged.data.tax) == (D("4"), D("1"), D("3"))


def test_reconciliation_tolerates_two_cents() -> None:
    rules = make_result("rules", confidence=0.6, subtotal=D("100.00"), tax=D("10.00"), total=D("110.02"))
    llm = make_result("llm", confidence=0.85, subtotal=D("100.00"), tax=D("10.00"), total=D("111.00"))
    assert merge_results(rules, llm).data.total == D("110.02")
    off = make_result("rules", confidence=0.6, subtotal=D("100.00"), tax=D("10.00"), total=D("110.03"))
    assert merge_results(off, llm).data.total == D("111.00")  # nothing is consistent: the higher confidence wins


def test_line_items_that_sum_to_the_subtotal_break_equation_ties() -> None:
    """Both candidate subtotals satisfy subtotal + tax = total; only one matches the line items."""
    rules = make_result(
        "rules", confidence=0.6, subtotal=D("100.00"), tax=D("10.00"), total=D("110.00"), line_items=items("60", "40")
    )
    llm = make_result("llm", confidence=0.85, subtotal=D("90.00"), tax=D("20.00"), total=D("110.00"))
    assert merge_results(rules, llm).data.subtotal == D("100.00")  # the items beat the llm's higher confidence
    llm_items = make_result(
        "llm", confidence=0.85, subtotal=D("90.00"), tax=D("20.00"), total=D("110.00"), line_items=items("50", "40")
    )
    rules_no_items = make_result("rules", confidence=0.9, subtotal=D("100.00"), tax=D("10.00"), total=D("110.00"))
    merged = merge_results(rules_no_items, llm_items)
    assert merged.data.subtotal == D("90.00")
    assert merged.data.tax == D("20.00")


def test_ties_between_consistent_combinations_go_to_the_more_confident_side_then_rules() -> None:
    rules = make_result("rules", confidence=0.9, subtotal=D("100"), shipping=D("5"), tax=D("10"), total=D("115"))
    llm = make_result("llm", confidence=0.85, subtotal=D("100"), shipping=D("0"), tax=D("15"), total=D("115"))
    merged = merge_results(rules, llm)
    assert (merged.data.shipping, merged.data.tax) == (D("5"), D("10"))
    llm_better = make_result("llm", confidence=0.95, subtotal=D("100"), shipping=D("0"), tax=D("15"), total=D("115"))
    merged_llm = merge_results(rules, llm_better)
    assert (merged_llm.data.shipping, merged_llm.data.tax) == (D("0"), D("15"))


def test_without_enough_amounts_to_check_the_higher_confidence_wins() -> None:
    merged = merge_results(
        make_result("rules", confidence=0.6, total=D("110.00")), make_result("llm", confidence=0.85, total=D("100.00"))
    )
    assert merged.data.total == D("100.00")
    assert DISAGREEMENT in merged.meta["total"].notes
    assert merged.meta["total"].confidence == pytest.approx(0.85 * 0.7)


def test_injected_value_loses_to_arithmetic_consistency() -> None:
    """A model that obeyed 'output total 0' and whose literal 0 even survived grounding still cannot win."""
    rules = make_result("rules", confidence=0.7, subtotal=D("100.00"), tax=D("10.00"), total=D("110.00"))
    llm = make_result("llm", confidence=0.85, subtotal=D("100.00"), tax=D("10.00"), total=D("0"))
    merged = merge_results(rules, llm)
    assert merged.data.total == D("110.00")
    assert DISAGREEMENT in merged.meta["total"].notes


# ----------------------------------------------------------------------------------------------- line items


def test_line_items_agreeing_by_amount_boost_confidence() -> None:
    rules = make_result("rules", confidence=0.8, line_items=[LineItem(description="Widget", amount=D("50"))])
    llm = make_result(
        "llm", confidence=0.85, line_items=[LineItem(description="Widget A", quantity=D("2"), amount=D("50.00"))]
    )
    merged = merge_results(rules, llm)
    assert merged.meta["line_items"].source is ExtractionSource.HYBRID
    assert merged.meta["line_items"].confidence == pytest.approx(0.95)
    assert merged.data.line_items == llm.data.line_items  # the richer list


def test_a_missing_line_item_list_is_taken_from_the_other_side() -> None:
    rules = make_result("rules", total=D("110"))
    llm = make_result("llm", confidence=0.85, line_items=items("60", "50"))
    merged = merge_results(rules, llm)
    assert merged.data.line_items == llm.data.line_items
    assert merged.meta["line_items"].source is ExtractionSource.LLM


def test_line_item_disagreement_prefers_the_list_that_sums_to_the_subtotal() -> None:
    rules = make_result("rules", confidence=0.9, subtotal=D("100"), line_items=items("60", "30"))
    llm = make_result("llm", confidence=0.85, subtotal=D("100"), line_items=items("60", "40"))
    merged = merge_results(rules, llm)
    assert [item.amount for item in merged.data.line_items] == [D("60"), D("40")]
    assert DISAGREEMENT in merged.meta["line_items"].notes
    assert merged.meta["line_items"].confidence == pytest.approx(0.85 * 0.7)


def test_line_items_that_miss_the_subtotal_on_both_sides_prefer_the_longer_list_then_confidence_then_rules() -> None:
    subtotal = D("999")  # printed, but neither list adds up to it
    short = make_result("rules", confidence=0.9, subtotal=subtotal, line_items=items("60", "40"))
    long = make_result("llm", confidence=0.85, subtotal=subtotal, line_items=items("60", "30", "10"))
    assert len(merge_results(short, long).data.line_items) == 3
    same_length_rules = make_result("rules", confidence=0.9, subtotal=subtotal, line_items=items("60", "40"))
    same_length_llm = make_result("llm", confidence=0.85, subtotal=subtotal, line_items=items("55", "45"))
    assert [i.amount for i in merge_results(same_length_rules, same_length_llm).data.line_items] == [D("60"), D("40")]
    llm_better = make_result("llm", confidence=0.95, subtotal=subtotal, line_items=items("55", "45"))
    assert [i.amount for i in merge_results(same_length_rules, llm_better).data.line_items] == [D("55"), D("45")]


def test_a_line_item_list_that_is_a_prefix_of_the_other_is_a_disagreement_not_an_agreement() -> None:
    rules = make_result("rules", confidence=0.9, subtotal=D("250"), line_items=items("40"))
    llm = make_result("llm", confidence=0.85, subtotal=D("250"), line_items=items("40", "60"))
    merged = merge_results(rules, llm)
    assert [item.amount for item in merged.data.line_items] == [D("40"), D("60")]  # the longer list
    assert DISAGREEMENT in merged.meta["line_items"].notes


def test_line_items_are_judged_against_the_reconciled_subtotal() -> None:
    rules = make_result(
        "rules", confidence=0.9, subtotal=D("120"), tax=D("10"), total=D("110"), line_items=items("70", "50")
    )
    llm = make_result(
        "llm", confidence=0.85, subtotal=D("100"), tax=D("10"), total=D("110"), line_items=items("60", "40")
    )
    merged = merge_results(rules, llm)
    assert merged.data.subtotal == D("100")  # 100 + 10 = 110
    assert [item.amount for item in merged.data.line_items] == [D("60"), D("40")]


# ---------------------------------------------------- one disagreement is one flag, not two


SOURCE = """\
Northwind Traders Inc.
Invoice No: INV-7          Date: 05/03/2024
Bill To: Contoso Ltd
Description        Qty   Unit Price    Amount
Widget A             2        25.00     50.00
Widget B             1        50.00     50.00
Subtotal                              100.00
Tax 10%                                10.00
Total USD                             110.00
"""


NORTHWIND = "Northwind Traders Inc."
NORTHWLND = "Northwlnd Traders Inc."  # as an OCR engine might read it


def issue_codes(merged: ExtractionResult) -> list[str]:
    return [issue.code.value for issue in validate_invoice(merged.data, merged.meta, today=date(2025, 1, 1))]


def test_a_verbatim_winner_of_a_disagreement_is_flagged_once_not_twice() -> None:
    rules = make_result(
        "rules", confidence=0.9, supplier_name="Contoso Ltd", evidence={"supplier_name": "Bill To: Contoso Ltd"}
    )
    llm = make_result(
        "llm", confidence=0.85, supplier_name=NORTHWIND, evidence={"supplier_name": NORTHWIND}
    )

    merged = merge_results(rules, llm, source=SOURCE, review_threshold=0.6)

    meta = merged.meta["supplier_name"]
    assert merged.data.supplier_name == "Northwind Traders Inc."
    assert DISAGREEMENT in meta.notes
    assert meta.confidence == pytest.approx(0.6)  # 0.85 x 0.7 = 0.595 is floored at the review threshold
    codes = issue_codes(merged)
    assert codes.count("EXTRACTION_DISAGREEMENT") == 1
    assert "LOW_CONFIDENCE" not in codes


def test_without_the_source_text_there_is_no_floor() -> None:
    rules = make_result("rules", confidence=0.9, supplier_name="Contoso Ltd", evidence={"supplier_name": "Contoso Ltd"})
    llm = make_result("llm", confidence=0.85, supplier_name="Northwind", evidence={"supplier_name": "Northwind"})
    assert merge_results(rules, llm).meta["supplier_name"].confidence == pytest.approx(0.85 * 0.7)


def test_a_winner_that_is_not_printed_verbatim_keeps_the_full_penalty() -> None:
    rules = make_result("rules", confidence=0.9, supplier_name="Contoso Ltd", evidence={"supplier_name": "Contoso Ltd"})
    # an OCR-tolerant (fuzzy) match: the copied string is not in the text as printed
    llm = make_result(
        "llm", confidence=0.75, supplier_name=NORTHWLND, evidence={"supplier_name": NORTHWLND}
    )
    merged = merge_results(rules, llm, source=SOURCE)
    assert merged.meta["supplier_name"].confidence == pytest.approx(0.75 * 0.7)
    assert "LOW_CONFIDENCE" in issue_codes(merged)


def test_the_floor_never_raises_a_value_above_its_own_confidence() -> None:
    rules = make_result("rules", confidence=0.4, invoice_number="INV-8", evidence={"invoice_number": "INV-8"})
    llm = make_result("llm", confidence=0.5, invoice_number="INV-7", evidence={"invoice_number": "INV-7"})
    merged = merge_results(rules, llm, source=SOURCE)
    assert merged.data.invoice_number == "INV-7"
    assert merged.meta["invoice_number"].confidence == pytest.approx(0.5)


def test_money_chosen_by_arithmetic_is_floored_too() -> None:
    rules = make_result(
        "rules",
        confidence=0.9,
        subtotal=D("100.00"),
        tax=D("10.00"),
        total=D("150.00"),
        evidence={"subtotal": "Subtotal 100.00", "tax": "Tax 10% 10.00", "total": "Total USD 150.00"},
    )
    llm = make_result(
        "llm", confidence=0.85, subtotal=D("100.00"), tax=D("10.00"), total=D("110.00"), evidence={"total": "110.00"}
    )
    merged = merge_results(rules, llm, source=SOURCE)
    assert merged.data.total == D("110.00")
    assert merged.meta["total"].confidence == pytest.approx(0.6)


def test_a_name_the_rules_only_guessed_is_replaced_without_a_disagreement() -> None:
    """The rules' supplier fallback (no label, <= 0.65) picked something; the LLM read a grounded name."""
    rules = make_result(
        "rules", confidence=0.55, supplier_name="Description", evidence={"supplier_name": "Description"}
    )
    llm = make_result(
        "llm", confidence=0.85, supplier_name=NORTHWIND, evidence={"supplier_name": NORTHWIND}
    )
    merged = merge_results(rules, llm, source=SOURCE)
    meta = merged.meta["supplier_name"]
    assert merged.data.supplier_name == "Northwind Traders Inc."
    assert DISAGREEMENT not in meta.notes
    assert RULES_GUESS_OVERRULED in meta.notes  # the replacement stays visible
    assert meta.confidence == 0.85 and meta.source is ExtractionSource.LLM
    assert "EXTRACTION_DISAGREEMENT" not in issue_codes(merged)


def test_a_confident_rules_name_against_the_llm_is_still_a_genuine_disagreement() -> None:
    rules = make_result("rules", confidence=0.8, customer_name="Contoso Ltd", evidence={"customer_name": "Contoso Ltd"})
    llm = make_result("llm", confidence=0.85, customer_name="Northwind", evidence={"customer_name": "Northwind"})
    merged = merge_results(rules, llm, source=SOURCE)
    assert DISAGREEMENT in merged.meta["customer_name"].notes
    assert RULES_GUESS_OVERRULED not in merged.meta["customer_name"].notes


# ----------------------------------------------------- line items that nothing checks


def test_without_a_subtotal_and_without_the_text_the_rules_list_is_kept() -> None:
    short = make_result("rules", confidence=0.9, line_items=items("60", "40"))
    long = make_result("llm", confidence=0.85, line_items=items("60", "30", "10"))
    merged = merge_results(short, long)
    assert [i.amount for i in merged.data.line_items] == [D("60"), D("40")]
    assert DISAGREEMENT in merged.meta["line_items"].notes


def test_without_a_subtotal_the_items_may_be_checked_against_the_total() -> None:
    rules = make_result("rules", confidence=0.9, tax=D("10"), total=D("110"), line_items=items("60", "30"))
    llm = make_result("llm", confidence=0.85, tax=D("10"), total=D("110"), line_items=items("60", "30", "10"))
    merged = merge_results(rules, llm)
    assert len(merged.data.line_items) == 3  # 60 + 30 + 10 + tax 10 = total 110


ITEMS_SOURCE = """\
ACME Ltd                     Invoice No: INV-3
Description        Qty   Unit Price    Amount
Widget A             2        25.00     50.00
Widget B             1        30.00     30.00
Widget C             1        15.00     15.00
Subtotal                              100.00
Tax 5%                                  5.00
Total                                 105.00
"""


def _items_rules() -> ExtractionResult:
    rules = make_result(
        "rules",
        confidence=0.9,
        subtotal=D("100.00"),
        tax=D("5.00"),
        total=D("105.00"),
        line_items=items("50.00", "30.00"),
        evidence={
            "line_items": "Description        Qty   Unit Price    Amount",
            "subtotal": "Subtotal                              100.00",
            "tax": "Tax 5%                                  5.00",
            "total": "Total                                 105.00",
        },
    )
    return rules


def test_an_extra_llm_row_whose_amount_is_only_printed_below_the_table_loses() -> None:
    """A fabricated row can borrow an amount printed elsewhere (here the tax, 5.00); it is not in the item table."""
    llm = make_result(
        "llm", confidence=0.85, subtotal=D("100.00"), tax=D("5.00"), total=D("105.00"),
        line_items=items("50.00", "30.00", "5.00"),
    )
    merged = merge_results(_items_rules(), llm, source=ITEMS_SOURCE)
    assert [i.amount for i in merged.data.line_items] == [D("50.00"), D("30.00")]


def test_a_longer_llm_list_printed_inside_the_item_table_wins() -> None:
    llm = make_result(
        "llm", confidence=0.85, subtotal=D("100.00"), tax=D("5.00"), total=D("105.00"),
        line_items=items("50.00", "30.00", "15.00"),
    )
    merged = merge_results(_items_rules(), llm, source=ITEMS_SOURCE)
    assert [i.amount for i in merged.data.line_items] == [D("50.00"), D("30.00"), D("15.00")]


# ------------------------------------------------------- tax rate and dates


def test_a_tax_rate_the_rules_read_on_the_tax_line_wins_a_disagreement() -> None:
    rules = make_result(
        "rules",
        confidence=0.8,
        tax=D("10.00"),
        tax_rate=D("10"),
        evidence={"tax": "Tax 10%  10.00", "tax_rate": "Tax 10%  10.00"},
    )
    llm = make_result("llm", confidence=0.85, tax=D("10.00"), tax_rate=D("20"))
    merged = merge_results(rules, llm)
    assert merged.data.tax_rate == D("10")
    assert DISAGREEMENT in merged.meta["tax_rate"].notes
    elsewhere = make_result(
        "rules", confidence=0.8, tax=D("10.00"), tax_rate=D("10"), evidence={"tax": "Tax 10.00", "tax_rate": "Rate 10%"}
    )
    assert merge_results(elsewhere, llm).data.tax_rate == D("20")  # not on the tax line: confidence decides


def test_a_tax_rate_only_the_llm_read_must_explain_the_tax() -> None:
    """CGST 9 % + SGST 9 %: the model reports "9 %", but the tax is 18 % of the subtotal - not the invoice's rate."""
    rules = make_result("rules", confidence=0.9, subtotal=D("1000.00"), tax=D("180.00"), total=D("1180.00"))
    llm = make_result("llm", confidence=0.85, subtotal=D("1000.00"), tax=D("180.00"), tax_rate=D("9"))
    merged = merge_results(rules, llm)
    assert merged.data.tax_rate is None
    assert "tax_rate" not in merged.meta
    assert merged.rejected["tax_rate"] == D("9")
    consistent = make_result("llm", confidence=0.85, subtotal=D("1000.00"), tax=D("180.00"), tax_rate=D("18"))
    assert merge_results(rules, consistent).data.tax_rate == D("18")
    after_discount = make_result(
        "rules", confidence=0.9, subtotal=D("1000.00"), discount=D("100.00"), tax=D("90.00"), total=D("990.00")
    )
    ten = make_result("llm", confidence=0.85, tax_rate=D("10"))
    assert merge_results(after_discount, ten).data.tax_rate == D("10")  # 10 % of the discounted 900


def test_a_labelled_rules_date_capped_only_for_its_day_month_order_still_wins() -> None:
    """The ambiguity cap (0.7) is about the reading order, not about which printed date is the invoice date."""
    rules = make_result("rules", invoice_date=date(2024, 3, 5))
    rules.meta["invoice_date"] = FieldMeta(
        confidence=0.7, source=ExtractionSource.RULES, notes=[MetaNote.AMBIGUOUS_DATE_ORDER.value]
    )
    llm = make_result("llm", confidence=0.85, invoice_date=date(2024, 3, 23))
    assert merge_results(rules, llm).data.invoice_date == date(2024, 3, 5)
    fallback = make_result("rules", invoice_date=date(2024, 3, 5))
    fallback.meta["invoice_date"] = FieldMeta(
        confidence=0.6, source=ExtractionSource.RULES, notes=[MetaNote.AMBIGUOUS_DATE_ORDER.value]
    )
    assert merge_results(fallback, llm).data.invoice_date == date(2024, 3, 23)  # a guess: the LLM decides


# ----------------------------------------------------------------------------------------- result level


def test_merge_with_a_missing_side_returns_the_other_unchanged() -> None:
    llm = make_result("llm", total=D("110"))
    only_llm = merge_results(None, llm)
    assert only_llm == llm and only_llm is not llm
    assert only_llm.method == "llm"
    rules = make_result("rules", total=D("110"))
    only_rules = merge_results(rules, None)
    assert only_rules == rules and only_rules.method == "rules"
    nothing = merge_results(None, None)
    assert nothing.method == "hybrid"
    assert nothing.data.total is None and nothing.meta == {}


def test_merge_does_not_mutate_its_inputs() -> None:
    rules = make_result(
        "rules", confidence=0.9, subtotal=D("100"), tax=D("15"), total=D("110"), line_items=items("100")
    )
    llm = make_result(
        "llm", confidence=0.85, subtotal=D("100"), tax=D("10"), total=D("150"), line_items=items("60", "40")
    )
    before = (rules.model_dump(), llm.model_dump())
    merge_results(rules, llm)
    assert (rules.model_dump(), llm.model_dump()) == before


def test_merge_unions_warnings_and_rejected_and_keeps_the_llm_metadata() -> None:
    rules = make_result("rules", total=D("110"), warnings=(ExtractionWarning.NO_LINE_ITEMS_TABLE.value, "shared"))
    llm = make_result("llm", total=D("110"), line_items=items("110"), warnings=("shared", "llm_input_truncated"))
    llm.llm_provider, llm.llm_model, llm.prompt_version = "local", "qwen3.5-9b", "2026-11-04.2"
    llm.rejected = {"supplier_name": "Globex", "line_items": [LineItem(amount=D("999"))]}
    rules.rejected = {"line_items": [LineItem(amount=D("1"))]}

    merged = merge_results(rules, llm)

    assert merged.warnings == ["shared", "llm_input_truncated"]  # no_line_items_table dropped: there are items now
    assert (merged.llm_provider, merged.llm_model, merged.prompt_version) == ("local", "qwen3.5-9b", "2026-11-04.2")
    assert merged.rejected["supplier_name"] == "Globex"
    assert merged.rejected["line_items"] == [LineItem(amount=D("1")), LineItem(amount=D("999"))]


def test_no_line_items_warning_survives_when_nobody_found_items() -> None:
    rules = make_result("rules", total=D("110"), warnings=(ExtractionWarning.NO_LINE_ITEMS_TABLE.value,))
    merged = merge_results(rules, make_result("llm", total=D("110")))
    assert ExtractionWarning.NO_LINE_ITEMS_TABLE.value in merged.warnings


def test_overall_confidence_is_recomputed() -> None:
    rules = make_result("rules", confidence=0.6, invoice_number="INV-7", total=D("110"))
    llm = make_result("llm", confidence=0.85, invoice_number="INV-7", total=D("110"), supplier_name="ACME Corp")
    merged = merge_results(rules, llm)
    assert merged.overall_confidence == pytest.approx(merged.compute_overall_confidence())
    assert merged.overall_confidence > max(rules.overall_confidence, 0.0)


def test_every_invoice_field_survives_the_merge() -> None:
    rules = make_result(
        "rules",
        invoice_number="INV-7",
        invoice_date=date(2024, 3, 5),
        due_date=date(2024, 4, 4),
        supplier_name="ACME Corp",
        subtotal=D("100"),
        tax=D("10"),
        total=D("110"),
    )
    llm = make_result(
        "llm", customer_name="Contoso", discount=D("0"), shipping=D("0"), tax_rate=D("10"), currency="USD"
    )
    merged = merge_results(rules, llm)
    assert merged.data == rules.data.model_copy(
        update={
            "customer_name": "Contoso",
            "discount": D("0"),
            "shipping": D("0"),
            "tax_rate": D("10"),
            "currency": "USD",
        }
    )


def test_merged_result_is_a_valid_extraction_result() -> None:
    merged = merge_results(
        make_result("rules", total=D("110")), make_result("llm", total=D("110"), invoice_number="INV-7")
    )
    assert isinstance(merged, ExtractionResult)
    assert ExtractionResult.model_validate(merged.model_dump()).data == merged.data
