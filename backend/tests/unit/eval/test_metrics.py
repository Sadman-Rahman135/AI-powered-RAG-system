"""Tests for the evaluation metrics (``eval.metrics``): hand-made truth/prediction pairs with known answers.

Every expected number below is worked out by hand from the cases, never from the implementation.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.domain.schemas import DocumentType, IssueCode, LineItem, SourceKind
from eval.labels import LabeledRecord
from eval.metrics import (
    FieldOutcome,
    compare_records,
    compare_value,
    compute_metrics,
    match_records,
    percentile,
    score_line_items,
)
from tests.unit.eval.builders import file_result as _result
from tests.unit.eval.builders import labeled_file as _file
from tests.unit.eval.builders import pred_record as _pred
from tests.unit.eval.builders import truth_record as _truth

D = Decimal


# --------------------------------------------------------------------------------------------------------------------
# field comparison
# --------------------------------------------------------------------------------------------------------------------
def test_field_accuracy_normalization() -> None:
    # names compare through normalize_name: case, punctuation and legal suffixes do not matter
    assert compare_value("supplier_name", "Acme Corp.", "ACME") is FieldOutcome.CORRECT
    assert compare_value("customer_name", "Edwards & Sons Ltd", "EDWARDS AND SONS") is FieldOutcome.CORRECT
    assert compare_value("supplier_name", "Acme Corp.", "Acme Trading") is FieldOutcome.WRONG
    # invoice numbers through normalize_invoice_number
    assert compare_value("invoice_number", "INV-2024/0042", "inv 20240042") is FieldOutcome.CORRECT
    assert compare_value("invoice_number", "INV-2024/0042", "INV-2024/0043") is FieldOutcome.WRONG
    # amounts within 0.01
    assert compare_value("total", D("120.00"), D("120.01")) is FieldOutcome.CORRECT
    assert compare_value("total", D("120.00"), D("120.02")) is FieldOutcome.WRONG
    assert compare_value("total", D("120.00"), D("120")) is FieldOutcome.CORRECT
    # dates exact, currency exact
    assert compare_value("invoice_date", date(2024, 3, 5), date(2024, 5, 3)) is FieldOutcome.WRONG
    assert compare_value("invoice_date", date(2024, 3, 5), date(2024, 3, 5)) is FieldOutcome.CORRECT
    assert compare_value("currency", "USD", "CAD") is FieldOutcome.WRONG
    # null handling: null truth is correct only when the prediction is null too
    assert compare_value("tax", None, None) is FieldOutcome.CORRECT
    assert compare_value("tax", None, D("0.00")) is FieldOutcome.HALLUCINATED
    assert compare_value("tax", D("20.00"), None) is FieldOutcome.MISSED
    # the fuzzy name variant accepts a near miss (OCR noise) that strict equality rejects
    assert compare_value("supplier_name", "Redwood Digital Ltd.", "Redwood Digita1") is FieldOutcome.WRONG
    assert compare_value("supplier_name", "Redwood Digital Ltd.", "Redwood Digita1", fuzzy=True) is FieldOutcome.CORRECT
    assert compare_value("supplier_name", "Redwood Digital", "Oakwood Systems", fuzzy=True) is FieldOutcome.WRONG


def test_document_type_unknown_counts_as_missing() -> None:
    assert compare_value("document_type", DocumentType.INVOICE, DocumentType.UNKNOWN) is FieldOutcome.MISSED
    assert compare_value("document_type", DocumentType.RECEIPT, DocumentType.INVOICE) is FieldOutcome.WRONG
    assert compare_value("document_type", DocumentType.INVOICE, DocumentType.INVOICE) is FieldOutcome.CORRECT


def test_compare_records_exact_match_and_line_items() -> None:
    truth = _truth()
    same = compare_records(truth, _pred(truth).data)
    assert same.exact_match
    assert all(o is FieldOutcome.CORRECT for o in same.core.values())
    assert set(same.core) == {
        "invoice_number",
        "invoice_date",
        "supplier_name",
        "customer_name",
        "subtotal",
        "tax",
        "total",
        "currency",
        "line_items",
    }
    assert "due_date" in same.secondary

    # one wrong amount breaks the exact match but nothing else
    off = compare_records(truth, _pred(truth, total=D("121.00")).data)
    assert not off.exact_match
    assert off.core["total"] is FieldOutcome.WRONG
    assert sum(o is FieldOutcome.CORRECT for o in off.core.values()) == 8

    # line items: the right amount with a wrong description is matched but not strictly correct
    wrong_desc = [LineItem(description="Gadget assembly", quantity=D("2"), unit_price=D("50.00"), amount=D("100.00"))]
    items = compare_records(truth, _pred(truth, line_items=wrong_desc).data)
    assert items.core["line_items"] is FieldOutcome.WRONG
    assert items.line_items.matched == 1 and items.line_items.strict_matched == 0

    # no items predicted for a record that has them
    none = compare_records(truth, _pred(truth, line_items=[]).data)
    assert none.core["line_items"] is FieldOutcome.MISSED

    # items predicted for a record without any (a ledger row)
    ledger = _truth(line_items=[], source_kind=SourceKind.LEDGER_ROW)
    invented = compare_records(ledger, _pred(ledger, line_items=truth.line_items).data)
    assert invented.core["line_items"] is FieldOutcome.HALLUCINATED


def test_line_items_precision_recall_greedy_by_amount() -> None:
    truth = [
        LineItem(description="Steel rebar 12 mm x 6 m", quantity=D("314"), unit_price=D("10.71"), amount=D("3362.94")),
        LineItem(description="Skip hire, 8 yard", quantity=D("5"), unit_price=D("190.99"), amount=D("954.95")),
        LineItem(description="Labour per hour", quantity=D("1"), unit_price=D("100.00"), amount=D("100.00")),
        LineItem(description="Delivery", quantity=D("1"), unit_price=D("100.00"), amount=D("100.00")),
    ]
    pred = [
        # wrapped description with different whitespace: still strictly correct
        LineItem(description="Steel rebar 12 mm x 6\nm", quantity=D("314"), unit_price=D("10.71"), amount=D("3362.94")),
        # same amount as two truth items: the description decides which one it is
        LineItem(description="Delivery", quantity=D("1"), unit_price=D("100.00"), amount=D("100.00")),
        # amount misread: no match
        LineItem(description="Skip hire, 8 yard", quantity=D("5"), unit_price=D("190.99"), amount=D("954.59")),
        # extra row (e.g. a subtotal line read as an item)
        LineItem(description="Subtotal", quantity=None, unit_price=None, amount=D("4517.89")),
    ]
    score = score_line_items(truth, pred)
    assert (score.truth, score.predicted, score.matched, score.strict_matched) == (4, 4, 2, 2)
    assert score.precision == pytest.approx(0.5)
    assert score.recall == pytest.approx(0.5)
    assert score.f1 == pytest.approx(0.5)
    # the "Delivery" prediction took the "Delivery" truth item, not the first 100.00 one
    assert score.pairs == [(0, 0), (3, 1)]


# --------------------------------------------------------------------------------------------------------------------
# record matching
# --------------------------------------------------------------------------------------------------------------------
def test_record_matching_by_invoice_number() -> None:
    truth = [
        _truth(invoice_number="A-1", page_start=1, page_end=1),
        _truth(invoice_number="B-2", page_start=2, page_end=3),
        _truth(invoice_number="C-3", page_start=4, page_end=4),
    ]
    # predictions out of order, number formatting differs, page ranges shifted by segmentation errors
    pred = [
        _pred(truth[2], invoice_number="c3", page_start=3, page_end=4),
        _pred(truth[0], invoice_number="A 1", page_start=1, page_end=2),
        _pred(truth[1], invoice_number="B/2", page_start=2, page_end=2),
    ]
    m = match_records(truth, pred)
    assert sorted((t, p) for t, p, _ in m.pairs) == [(0, 1), (1, 2), (2, 0)]
    assert {method for _, _, method in m.pairs} == {"invoice_number"}
    assert m.missed == [] and m.extra == []


def test_record_matching_falls_back_to_pages_then_order() -> None:
    truth = [
        _truth(invoice_number="A-1", page_start=1, page_end=1),
        _truth(invoice_number="B-2", page_start=2, page_end=2),
        _truth(invoice_number="C-3", page_start=3, page_end=3),
    ]
    pred = [
        _pred(truth[0]),  # number match
        _pred(truth[1], invoice_number="8-2"),  # OCR misread the number: matched by page overlap
        _pred(truth[2], invoice_number=None, page_start=5, page_end=5),  # contradicting pages: not matched
        _pred(truth[2], invoice_number="ZZZ", page_start=None, page_end=None),  # no location: matched by order
    ]
    m = match_records(truth, pred)
    assert sorted(m.pairs) == [(0, 0, "invoice_number"), (1, 1, "location"), (2, 3, "order")]
    assert m.missed == []
    assert m.extra == [2]


def test_record_matching_sheet_rows_misses_and_extras() -> None:
    def row(number: str, sheet: str, n: int) -> LabeledRecord:
        return _truth(
            invoice_number=number,
            page_start=None,
            page_end=None,
            sheet_name=sheet,
            row_number=n,
            line_items=[],
            source_kind=SourceKind.LEDGER_ROW,
        )

    truth = [row("R-1", "Sales", 2), row("R-2", "Sales", 3), row("R-3", "Sales", 4)]
    pred = [
        _pred(truth[1], invoice_number=None),  # number lost: matched by sheet + row
        _pred(truth[0], invoice_number="R-1"),
        _pred(truth[2], invoice_number=None, sheet_name="Notes", row_number=4),  # other sheet: contradicts
    ]
    m = match_records(truth, pred)
    assert sorted(m.pairs) == [(0, 1, "invoice_number"), (1, 0, "location")]
    assert m.missed == [2]
    assert m.extra == [2]


# --------------------------------------------------------------------------------------------------------------------
# aggregate metrics
# --------------------------------------------------------------------------------------------------------------------
def test_hallucination_rate() -> None:
    # truth has 3 null core fields (customer_name, tax, currency); the prediction fills two of them
    truth = _truth(customer_name=None, tax=None, currency=None, total=D("100.00"))
    pred = _pred(truth, customer_name="Globex", tax=D("0.00"))
    label = _file("a.pdf", [truth])
    metrics = compute_metrics([label], [_result("a.pdf", [pred])])["overall"]
    hall = metrics["hallucination"]
    assert hall["null_truth"] == 3
    assert hall["count"] == 2
    assert hall["rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert hall["per_field"]["customer_name"]["count"] == 1
    assert hall["per_field"]["currency"]["count"] == 0
    fields = metrics["fields"]
    assert fields["overall"]["correct"] == 7 and fields["overall"]["total"] == 9
    assert fields["per_field"]["tax"]["hallucinated"] == 1
    assert metrics["record_exact_match"]["count"] == 0


def test_missed_records_count_every_field_wrong_and_extras_are_reported() -> None:
    truth = [
        _truth(invoice_number="A-1", page_start=1, page_end=1),
        _truth(invoice_number="B-2", page_start=2, page_end=2),
    ]
    pred = [_pred(truth[0]), _pred(truth[0], invoice_number="Q-9", page_start=7, page_end=7)]
    metrics = compute_metrics([_file("m.pdf", truth, pages=2)], [_result("m.pdf", pred)])["overall"]
    rec = metrics["records"]
    assert (rec["truth"], rec["predicted"], rec["matched"], rec["missed"], rec["extra"]) == (2, 2, 1, 1, 1)
    assert rec["recall"] == pytest.approx(0.5) and rec["precision"] == pytest.approx(0.5)
    fields = metrics["fields"]
    assert fields["overall"]["total"] == 18
    assert fields["overall"]["correct"] == 9
    assert fields["per_field"]["total"]["missed_record"] == 1
    assert fields["matched_only"]["accuracy"] == pytest.approx(1.0)
    assert metrics["record_exact_match"]["rate"] == pytest.approx(0.5)


def test_issue_precision_recall() -> None:
    clean = _truth(invoice_number="C-1")
    bad_total = _truth(invoice_number="T-1", total=D("125.00"))
    missing = _truth(invoice_number="M-1", customer_name=None)
    labels = [
        _file("clean.pdf", [clean]),
        _file("bad.pdf", [bad_total], issues=[[IssueCode.TOTAL_MISMATCH]]),
        _file("missing.pdf", [missing], issues=[[IssueCode.MISSING_FIELD]]),
    ]
    results = [
        # false positive TOTAL_MISMATCH on a clean record; a non-evaluated code (LOW_CONFIDENCE) is ignored
        _result("clean.pdf", [_pred(clean, issues=["TOTAL_MISMATCH", "LOW_CONFIDENCE"])]),
        # true positive TOTAL_MISMATCH
        _result("bad.pdf", [_pred(bad_total, issues=["TOTAL_MISMATCH"])]),
        # MISSING_FIELD not raised: false negative
        _result("missing.pdf", [_pred(missing, issues=[])]),
    ]
    val = compute_metrics(labels, results)["overall"]["validation"]
    tm = val["per_code"]["TOTAL_MISMATCH"]
    assert (tm["tp"], tm["fp"], tm["fn"]) == (1, 1, 0)
    assert tm["precision"] == pytest.approx(0.5) and tm["recall"] == pytest.approx(1.0)
    mf = val["per_code"]["MISSING_FIELD"]
    assert (mf["tp"], mf["fp"], mf["fn"]) == (0, 0, 1)
    assert mf["precision"] is None and mf["recall"] == pytest.approx(0.0)
    assert "LOW_CONFIDENCE" not in val["per_code"]
    micro = val["micro"]
    assert (micro["tp"], micro["fp"], micro["fn"]) == (1, 1, 1)
    clean_stats = val["clean_records"]
    assert clean_stats["total"] == 1 and clean_stats["flagged"] == 1
    assert clean_stats["false_positive_rate"] == pytest.approx(1.0)
    assert clean_stats["sent_to_review"] == 1


def test_duplicate_pairs_detected_when_either_member_is_flagged() -> None:
    inv1, inv2 = _truth(invoice_number="D-1"), _truth(invoice_number="D-2")
    labels = [
        _file("dup_1_a.pdf", [inv1], group="dup-1"),
        _file(
            "dup_1_b.jpg",
            [inv1],
            group="dup-1",
            kind="jpg",
            quality="scan_light",
            issues=[[IssueCode.DUPLICATE_INVOICE]],
        ),
        _file("dup_2_a.pdf", [inv2], group="dup-2"),
        _file(
            "dup_2_b.jpg", [inv2], group="dup-2", kind="jpg", quality="photo", issues=[[IssueCode.DUPLICATE_INVOICE]]
        ),
    ]
    results = [
        _result("dup_1_a.pdf", [_pred(inv1, issues=["DUPLICATE_INVOICE"])]),  # flagged on the original only
        _result("dup_1_b.jpg", [_pred(inv1)]),
        _result("dup_2_a.pdf", [_pred(inv2)]),
        _result("dup_2_b.jpg", [_pred(inv2)]),
    ]
    metrics = compute_metrics(labels, results)
    pairs = metrics["overall"]["validation"]["duplicate_pairs"]
    assert (pairs["pairs"], pairs["detected"]) == (2, 1)
    assert pairs["rate"] == pytest.approx(0.5)
    # per-code scoring is strict about which record carries the code
    dup = metrics["overall"]["validation"]["per_code"]["DUPLICATE_INVOICE"]
    assert (dup["tp"], dup["fp"], dup["fn"]) == (0, 1, 2)
    # a breakdown that holds one member of a pair still knows whether the pair was detected
    assert metrics["by_kind"]["jpg"]["validation"]["duplicate_pairs"]["detected"] == 1


def test_success_failure_and_corrupt_handling_rates() -> None:
    ok = _truth(invoice_number="OK-1")
    labels = [
        _file("ok.pdf", [ok]),
        _file("review.pdf", [ok]),
        _file("empty_result.pdf", [ok]),
        _file("failed.pdf", [ok]),
        _file("crashed.pdf", [ok]),
        _file(
            "corrupt/trunc.pdf",
            [],
            kind="corrupt",
            template=None,
            split="na",
            expect="failure",
            expected_error_code="corrupt_file",
        ),
        _file(
            "corrupt/random.pdf",
            [],
            kind="corrupt",
            template=None,
            split="na",
            expect="failure",
            expected_error_code="corrupt_file",
        ),
        _file(
            "corrupt/empty.pdf",
            [],
            kind="corrupt",
            template=None,
            split="na",
            expect="failure",
            expected_error_code="empty_file",
        ),
    ]
    results = [
        _result("ok.pdf", [_pred(ok)]),
        _result("review.pdf", [_pred(ok, issues=["LOW_CONFIDENCE"])], status="needs_review"),
        _result("empty_result.pdf", []),
        _result("failed.pdf", [], status="failed", error_code="corrupt_file"),
        _result("crashed.pdf", [], status="crashed", seconds=None, harness_error="RuntimeError"),
        _result("corrupt/trunc.pdf", [], status="failed", error_code="corrupt_file"),
        # random bytes legitimately sniff as an unsupported type: handled, but not the exact code
        _result("corrupt/random.pdf", [], status="rejected", error_code="unsupported_file_type", seconds=None),
        # an expect=failure file that "succeeded" is not handled
        _result("corrupt/empty.pdf", [], status="completed"),
    ]
    metrics = compute_metrics(labels, results)
    out = metrics["overall"]["outcomes"]
    assert out["expect_success"] == 5
    assert out["succeeded"] == 2 and out["success_rate"] == pytest.approx(0.4)
    assert out["failed"] == 2 and out["failure_rate"] == pytest.approx(0.4)
    assert out["empty_result"] == 1
    assert out["expect_failure"] == 3
    assert out["handled"] == 2 and out["handling_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["exact_code"] == 1 and out["exact_code_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert metrics["by_split"]["na"]["outcomes"]["expect_failure"] == 3
    assert metrics["by_kind"]["corrupt"]["outcomes"]["expect_success"] == 0


def test_processing_time_statistics_and_breakdowns() -> None:
    labels = [
        _file("a.pdf", [_truth(invoice_number="A")], split="dev"),
        _file("b.pdf", [_truth(invoice_number="B")], split="heldout", template="T4", pages=2),
        _file("c.jpg", [_truth(invoice_number="C")], split="heldout", template="T5", kind="jpg", quality="photo"),
        _file("d.pdf", [_truth(invoice_number="D")], split="dev"),
    ]
    results = [
        _result("a.pdf", [_pred(_truth(invoice_number="A"))], seconds=1.0, page_count=1, processing_ms=900),
        _result("b.pdf", [_pred(_truth(invoice_number="B"))], seconds=3.0, page_count=2, processing_ms=2900),
        _result("c.jpg", [_pred(_truth(invoice_number="C"))], seconds=6.0, page_count=1, processing_ms=5900),
        _result("d.pdf", [], status="crashed", seconds=None, page_count=None, processing_ms=None),
    ]
    metrics = compute_metrics(labels, results)
    proc = metrics["overall"]["processing"]
    assert proc["files_timed"] == 3
    assert proc["seconds"]["mean"] == pytest.approx(10 / 3, abs=1e-3)
    assert proc["seconds"]["p50"] == pytest.approx(3.0)
    assert proc["seconds"]["p95"] == pytest.approx(5.7)
    assert proc["seconds"]["total"] == pytest.approx(10.0)
    assert proc["pages"] == 4
    assert proc["pages_per_s"] == pytest.approx(0.4)
    assert proc["files_per_min"] == pytest.approx(18.0)
    assert proc["processing_ms"]["p50"] == pytest.approx(2900)
    assert metrics["by_kind"]["jpg"]["processing"]["seconds"]["mean"] == pytest.approx(6.0)
    assert metrics["by_quality"]["photo"]["files"] == 1
    assert set(metrics["by_split"]) == {"dev", "heldout"}
    assert metrics["by_split"]["heldout"]["records"]["truth"] == 2
    assert metrics["by_split"]["dev"]["outcomes"]["failed"] == 1


def test_percentile_linear_interpolation() -> None:
    assert percentile([], 50) is None
    assert percentile([4.0], 95) == pytest.approx(4.0)
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
    assert percentile([10.0, 1.0, 5.0], 0) == pytest.approx(1.0)
    assert percentile([10.0, 1.0, 5.0], 100) == pytest.approx(10.0)


def test_results_without_label_or_label_without_result() -> None:
    label = _file("a.pdf", [_truth()])
    with pytest.raises(ValueError, match=r"a\.pdf"):
        compute_metrics([label], [])
    # results for files outside the labels are an error too (a mixed-up dataset)
    with pytest.raises(ValueError, match=r"z\.pdf"):
        compute_metrics([label], [_result("a.pdf", [_pred()]), _result("z.pdf", [])])


def test_outcome_buckets_sum_and_unfinished_files_count_as_failures() -> None:
    ok = _truth(invoice_number="OK-1")
    names = ["ok.pdf", "empty.pdf", "failed.pdf", "queued.pdf", "processing.pdf"]
    labels = [_file(name, [ok]) for name in names]
    results = [
        _result("ok.pdf", [_pred(ok)]),
        _result("empty.pdf", []),
        _result("failed.pdf", [], status="failed", error_code="corrupt_file"),
        # the pipeline never finished these: they are failures, not silently dropped
        _result("queued.pdf", [], status="queued", seconds=None),
        _result("processing.pdf", [], status="processing"),
    ]
    out = compute_metrics(labels, results)["overall"]["outcomes"]
    assert (out["succeeded"], out["empty_result"], out["failed"], out["other"]) == (1, 1, 1, 2)
    assert out["succeeded"] + out["empty_result"] + out["failed"] + out["other"] == out["expect_success"] == 5
    assert out["failure_rate"] == pytest.approx(0.6)
    assert out["success_rate"] + out["failure_rate"] + out["empty_rate"] == pytest.approx(1.0)


def test_record_issues_must_be_parallel_to_records() -> None:
    two = [_truth(invoice_number="A-1"), _truth(invoice_number="B-2", page_start=2, page_end=2)]
    bad = _file("bad.pdf", two, issues=[[IssueCode.TOTAL_MISMATCH]], pages=2)  # 1 entry for 2 records
    with pytest.raises(ValueError, match=r"bad\.pdf: record_issues has 1 entries for 2 records"):
        compute_metrics([bad], [_result("bad.pdf", [_pred(two[0]), _pred(two[1])])])
    stray = _file("stray.pdf", [], issues=[[IssueCode.MISSING_FIELD]], kind="corrupt", expect="failure")
    with pytest.raises(ValueError, match=r"stray\.pdf"):
        compute_metrics([stray], [_result("stray.pdf", [], status="failed", error_code="corrupt_file")])
    # a single record without record_issues falls back to expected_issue_codes (older label files)
    legacy = _file("legacy.pdf", [two[0]], issues=[], expected_issue_codes=[IssueCode.TOTAL_MISMATCH])
    val = compute_metrics([legacy], [_result("legacy.pdf", [_pred(two[0], issues=["TOTAL_MISMATCH"])])])
    assert val["overall"]["validation"]["per_code"]["TOTAL_MISMATCH"]["tp"] == 1
