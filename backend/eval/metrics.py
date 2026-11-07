"""Scoring of evaluation runs: pure functions from (labels, per-file results) to metrics.

Nothing here touches the database, the pipeline or the clock, so every number in the evaluation report can be
recomputed from a results JSON file and the dataset's ``labels.json``.

Definitions (they are also stated in ``docs/EVALUATION.md``):

Record matching (per file)
    1. **invoice number**: truth and prediction with the same ``normalize_invoice_number`` (non-empty) are paired;
       among several candidates the best location agreement wins, then the nearest position.
    2. **location**: remaining pairs whose page ranges overlap (score = intersection over union of the ranges), or
       that sit on the same sheet row (score 1; the same sheet without row numbers scores 0.5).
    3. **order**: remaining truth and predictions are paired in order where neither side contradicts the other
       (a contradiction = disjoint page ranges, different rows or different sheets).
    Each step assigns greedily by descending score. Unpaired truth records are **missed** (all their fields count as
    wrong), unpaired predictions are **extra** records (reported; they do not enter field accuracy).
Field comparison
    Names (supplier, customer) by ``normalize_name`` equality (a fuzzy variant accepts a ``rapidfuzz.fuzz.ratio`` of
    the normalised names >= 90); invoice numbers by ``normalize_invoice_number``; dates exactly; amounts and the tax
    rate within 0.01; currency exactly; a null truth is correct only when the prediction is null too, and a non-null
    prediction there is a **hallucination**. ``line_items`` is correct when every truth item is strictly matched and
    no item is extra (both empty is correct).
Line items
    Candidate pairs are items with equal amounts (within 0.01; two null amounts need similar descriptions); pairs are
    taken greedily by description similarity (casefolded, whitespace removed, ``fuzz.ratio``), so two items with the
    same amount go to the descriptions they resemble. A **strict** match also needs description similarity >= 90 and
    equal quantity and unit price. Precision/recall are micro-averaged; items of missed records count against recall,
    items of extra records against precision.
Validation detection
    Over the codes the labels have ground truth for (``EVALUATED_ISSUE_CODES``), per matched record: expected codes
    (``record_issues``) vs the codes on the predicted record. A missed record's expected codes are false negatives;
    extra records are not scored (their flags are counted separately). A duplicate pair (``dup-N``) is detected when
    either member carries ``DUPLICATE_INVOICE``. A **clean** record has no expected evaluated code.
Outcomes
    Success: an ``expect=success`` file that ended ``completed``/``needs_review`` with at least one record when its
    labels have records. Empty: it completed without a record although its labels have records. Failure: it ended
    ``failed``, was rejected at upload, crashed the pipeline, or was left in any other (non-terminal) status; the
    four buckets sum to the ``expect=success`` files. Corrupt handling: an ``expect=failure`` file that ended
    failed/rejected with a typed error code (the exact expected code is reported separately).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Any, NamedTuple

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from app.domain.schemas import (
    CORE_FIELDS,
    MONEY_FIELDS,
    DocumentType,
    InvoiceData,
    IssueCode,
    LineItem,
    SourceKind,
)
from app.extraction.normalize import normalize_invoice_number, normalize_name
from eval.labels import ERROR_CODES, EVALUATED_ISSUE_CODES, QUALITIES, LabeledFile, LabeledRecord

AMOUNT_TOLERANCE = Decimal("0.01")
NAME_FUZZY_THRESHOLD = 90.0  # rapidfuzz ratio of the normalised names
DESCRIPTION_THRESHOLD = 90.0  # rapidfuzz ratio of casefolded, whitespace-free descriptions
SECONDARY_FIELDS: tuple[str, ...] = ("due_date", "discount", "shipping", "tax_rate", "document_type")
NAME_FIELDS: tuple[str, ...] = ("supplier_name", "customer_name")
VALIDATION_CODES: tuple[str, ...] = tuple(code.value for code in EVALUATED_ISSUE_CODES)
SUCCESS_STATUSES = frozenset({"completed", "needs_review"})
FAILURE_STATUSES = frozenset({"failed", "rejected", "crashed"})
TYPED_ERROR_CODES = frozenset({*ERROR_CODES, "file_too_large", "ocr_unavailable"})
LLM_UNAVAILABLE = IssueCode.LLM_UNAVAILABLE.value

SPLIT_ORDER: tuple[str, ...] = ("dev", "heldout", "na")
KIND_ORDER: tuple[str, ...] = ("pdf", "png", "jpg", "tiff", "xlsx", "xls", "csv", "corrupt")
QUALITY_ORDER: tuple[str, ...] = QUALITIES

__all__ = [
    "AMOUNT_TOLERANCE",
    "SECONDARY_FIELDS",
    "VALIDATION_CODES",
    "FieldOutcome",
    "FileResult",
    "FileScore",
    "LineItemScore",
    "PredictedRecord",
    "RecordComparison",
    "RecordMatching",
    "RecordPair",
    "compare_records",
    "compare_value",
    "compute_metrics",
    "compute_section",
    "match_records",
    "percentile",
    "rate",
    "score_file",
    "score_line_items",
    "stage_rates",
    "timing_stats",
]


# --------------------------------------------------------------------------------------------------------------------
# what a run produces per file
# --------------------------------------------------------------------------------------------------------------------
class PredictedRecord(BaseModel):
    """One record the system extracted, read back from the database after the run."""

    data: InvoiceData
    source_kind: SourceKind = SourceKind.INVOICE_DOCUMENT
    page_start: int | None = None
    page_end: int | None = None
    sheet_name: str | None = None
    row_number: int | None = None
    issue_codes: list[str] = Field(default_factory=list)
    review_status: str | None = None
    extraction_method: str | None = None
    overall_confidence: float | None = None


class FileResult(BaseModel):
    """The outcome of one dataset file: final status, timings and the records read back."""

    file: str  # labels.json path
    status: str  # completed | needs_review | failed | rejected (at upload) | crashed (the pipeline raised) | ...
    error_code: str | None = None
    error_message: str | None = None
    seconds: float | None = None  # wall clock of process_job; None when the file was not processed
    processing_ms: int | None = None  # the pipeline's own measurement (documents.processing_ms)
    page_count: int | None = None
    ocr_page_count: int | None = None
    extraction_method: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_fallbacks: int = 0  # records whose LLM call failed/refused/was unavailable (rules result used)
    records: list[PredictedRecord] = Field(default_factory=list)
    document_issue_codes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)  # documents.warnings (codes only)
    attempts: int = 0  # process_job attempts (a transient error is retried like the queue would)
    harness_error: str | None = None  # exception type and message when the pipeline or the harness raised
    stage_seconds: dict[str, float] = Field(default_factory=dict)  # parsing/extracting/validating/indexing


# --------------------------------------------------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------------------------------------------------
def rate(numerator: float, denominator: float) -> float | None:
    """``numerator / denominator`` rounded to 4 places; ``None`` when there is nothing to divide by."""
    if not denominator:
        return None
    return round(numerator / denominator, 4)


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def percentile(values: Sequence[float], q: float) -> float | None:
    """The ``q``-th percentile (0-100) with linear interpolation between closest ranks; ``None`` for no values."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def timing_stats(values: Sequence[float]) -> dict[str, float | None]:
    """Mean, median, p95, max and total of ``values`` (all ``None`` when empty)."""
    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None, "total": None}
    p50, p95 = percentile(values, 50), percentile(values, 95)
    return {
        "mean": round(sum(values) / len(values), 4),
        "p50": round(p50, 4) if p50 is not None else None,
        "p95": round(p95, 4) if p95 is not None else None,
        "max": round(max(values), 4),
        "total": round(sum(values), 4),
    }


def stage_rates(stage_seconds: dict[str, float], pages: int) -> dict[str, dict[str, float | None]]:
    """Seconds and pages per second of every pipeline stage of one document."""
    return {
        stage: {"seconds": round(seconds, 4), "pages_per_s": rate(pages, seconds)}
        for stage, seconds in stage_seconds.items()
    }


# --------------------------------------------------------------------------------------------------------------------
# field comparison
# --------------------------------------------------------------------------------------------------------------------
class FieldOutcome(StrEnum):
    CORRECT = "correct"  # equal, or both null
    WRONG = "wrong"  # both present, different
    MISSED = "missed"  # printed value not extracted
    HALLUCINATED = "hallucinated"  # value extracted where nothing is printed


def _is_null(name: str, value: Any) -> bool:
    if name == "line_items":
        return not value
    if name == "document_type":
        return value is None or DocumentType(value) is DocumentType.UNKNOWN
    return value is None


def _description_key(text: str | None) -> str:
    return "".join((text or "").casefold().split())


def _description_similarity(a: str | None, b: str | None) -> float:
    ka, kb = _description_key(a), _description_key(b)
    if not ka and not kb:
        return 100.0
    if not ka or not kb:
        return 0.0
    return float(fuzz.ratio(ka, kb))


def _numbers_equal(a: Decimal | None, b: Decimal | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(Decimal(a) - Decimal(b)) <= AMOUNT_TOLERANCE


def _equal(name: str, truth: Any, pred: Any, fuzzy: bool) -> bool:
    if name == "invoice_number":
        nt, np_ = normalize_invoice_number(truth), normalize_invoice_number(pred)
        return nt == np_ if (nt or np_) else truth.strip() == pred.strip()
    if name in NAME_FIELDS:
        nt, np_ = normalize_name(truth), normalize_name(pred)
        return nt == np_ or (fuzzy and fuzz.ratio(nt, np_) >= NAME_FUZZY_THRESHOLD)
    if name in MONEY_FIELDS or name == "tax_rate":
        return _numbers_equal(truth, pred)
    if name == "currency":
        return str(truth).upper() == str(pred).upper()
    if name == "document_type":
        return DocumentType(truth) is DocumentType(pred)
    if name == "line_items":
        score = score_line_items(truth, pred)
        return score.truth == score.predicted == score.strict_matched
    if isinstance(truth, date) or isinstance(pred, date):
        return truth == pred
    raise ValueError(f"no comparison rule for field {name!r}")


def compare_value(name: str, truth: Any, pred: Any, *, fuzzy: bool = False) -> FieldOutcome:
    """The outcome of one predicted field value against its label (see the module docstring for the rules)."""
    truth_null, pred_null = _is_null(name, truth), _is_null(name, pred)
    if truth_null:
        return FieldOutcome.CORRECT if pred_null else FieldOutcome.HALLUCINATED
    if pred_null:
        return FieldOutcome.MISSED
    return FieldOutcome.CORRECT if _equal(name, truth, pred, fuzzy) else FieldOutcome.WRONG


@dataclass(slots=True)
class LineItemScore:
    """Line items of one record (or a sum over many): counts and the matched ``(truth, pred)`` index pairs."""

    truth: int = 0
    predicted: int = 0
    matched: int = 0
    strict_matched: int = 0
    pairs: list[tuple[int, int]] = field(default_factory=list)

    @property
    def precision(self) -> float | None:
        return rate(self.matched, self.predicted)

    @property
    def recall(self) -> float | None:
        return rate(self.matched, self.truth)

    @property
    def f1(self) -> float | None:
        return _f1(self.precision, self.recall)

    def add(self, other: LineItemScore) -> None:
        self.truth += other.truth
        self.predicted += other.predicted
        self.matched += other.matched
        self.strict_matched += other.strict_matched

    def as_dict(self) -> dict[str, Any]:
        strict_p, strict_r = rate(self.strict_matched, self.predicted), rate(self.strict_matched, self.truth)
        return {
            "truth": self.truth,
            "predicted": self.predicted,
            "matched": self.matched,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "strict": {
                "matched": self.strict_matched,
                "precision": strict_p,
                "recall": strict_r,
                "f1": _f1(strict_p, strict_r),
            },
        }


def _items_candidate(truth: LineItem, pred: LineItem, similarity: float) -> bool:
    if truth.amount is None or pred.amount is None:
        return truth.amount is None and pred.amount is None and similarity >= DESCRIPTION_THRESHOLD
    return abs(truth.amount - pred.amount) <= AMOUNT_TOLERANCE


def score_line_items(truth: Sequence[LineItem], pred: Sequence[LineItem]) -> LineItemScore:
    """Match predicted line items to truth items by amount, resolving ties by description similarity."""
    candidates: list[tuple[float, int, int, int]] = []
    for ti, t in enumerate(truth):
        for pi, p in enumerate(pred):
            similarity = _description_similarity(t.description, p.description)
            if _items_candidate(t, p, similarity):
                candidates.append((-similarity, abs(ti - pi), ti, pi))
    candidates.sort()
    used_t: set[int] = set()
    used_p: set[int] = set()
    score = LineItemScore(truth=len(truth), predicted=len(pred))
    for neg_similarity, _distance, ti, pi in candidates:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        score.matched += 1
        score.pairs.append((ti, pi))
        t, p = truth[ti], pred[pi]
        if (
            -neg_similarity >= DESCRIPTION_THRESHOLD
            and _numbers_equal(t.quantity, p.quantity)
            and _numbers_equal(t.unit_price, p.unit_price)
        ):
            score.strict_matched += 1
    score.pairs.sort()
    return score


@dataclass(slots=True)
class RecordComparison:
    """Field outcomes of one matched (truth, prediction) pair."""

    core: dict[str, FieldOutcome]
    core_fuzzy: dict[str, FieldOutcome]  # core outcomes with the fuzzy name rule
    secondary: dict[str, FieldOutcome]
    line_items: LineItemScore
    truth_null: frozenset[str]

    @property
    def exact_match(self) -> bool:
        """All nine core fields correct (strict name rule)."""
        return all(outcome is FieldOutcome.CORRECT for outcome in self.core.values())


def compare_records(truth: InvoiceData, pred: InvoiceData) -> RecordComparison:
    """Compare every core and secondary field of one predicted record with its label."""
    core: dict[str, FieldOutcome] = {}
    core_fuzzy: dict[str, FieldOutcome] = {}
    for name in CORE_FIELDS:
        t, p = getattr(truth, name), getattr(pred, name)
        core[name] = compare_value(name, t, p)
        core_fuzzy[name] = compare_value(name, t, p, fuzzy=True) if name in NAME_FIELDS else core[name]
    secondary = {name: compare_value(name, getattr(truth, name), getattr(pred, name)) for name in SECONDARY_FIELDS}
    truth_null = frozenset(name for name in (*CORE_FIELDS, *SECONDARY_FIELDS) if _is_null(name, getattr(truth, name)))
    return RecordComparison(
        core=core,
        core_fuzzy=core_fuzzy,
        secondary=secondary,
        line_items=score_line_items(truth.line_items, pred.line_items),
        truth_null=truth_null,
    )


# --------------------------------------------------------------------------------------------------------------------
# record matching
# --------------------------------------------------------------------------------------------------------------------
class RecordPair(NamedTuple):
    truth: int
    pred: int
    method: str  # invoice_number | location | order


@dataclass(slots=True)
class RecordMatching:
    pairs: list[RecordPair]
    missed: list[int]  # truth indices without a prediction
    extra: list[int]  # prediction indices without a truth record


def _norm_number(value: str | None) -> str | None:
    if not value:
        return None
    return normalize_invoice_number(value) or None


def _page_range(start: int | None, end: int | None) -> tuple[int, int] | None:
    if start is None:
        return None
    return start, end if end is not None else start


def _location_score(truth: LabeledRecord, pred: PredictedRecord) -> float | None:
    """> 0 when the locations agree, 0 when they contradict, ``None`` when either side has no location."""
    t_pages, p_pages = _page_range(truth.page_start, truth.page_end), _page_range(pred.page_start, pred.page_end)
    if t_pages and p_pages:
        overlap = min(t_pages[1], p_pages[1]) - max(t_pages[0], p_pages[0]) + 1
        if overlap <= 0:
            return 0.0
        union = max(t_pages[1], p_pages[1]) - min(t_pages[0], p_pages[0]) + 1
        return overlap / union
    t_sheet = (truth.sheet_name or "").strip().casefold()
    p_sheet = (pred.sheet_name or "").strip().casefold()
    if t_sheet and p_sheet and t_sheet != p_sheet:
        return 0.0
    if truth.row_number is not None and pred.row_number is not None:
        return 1.0 if truth.row_number == pred.row_number else 0.0
    if t_sheet and p_sheet:
        return 0.5
    return None


def _assign(
    candidates: Iterable[tuple[float, int, int]], used_t: set[int], used_p: set[int], method: str
) -> list[RecordPair]:
    """Greedy assignment by descending score, then by nearest position; marks the indices used."""
    pairs: list[RecordPair] = []
    for _score, ti, pi in sorted(candidates, key=lambda c: (-c[0], abs(c[1] - c[2]), c[1], c[2])):
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        pairs.append(RecordPair(ti, pi, method))
    return pairs


def match_records(truth: Sequence[LabeledRecord], predicted: Sequence[PredictedRecord]) -> RecordMatching:
    """Pair the predicted records of one file with its labels: invoice number, then location, then order."""
    used_t: set[int] = set()
    used_p: set[int] = set()
    by_number: dict[str, list[int]] = defaultdict(list)
    for pi, p in enumerate(predicted):
        if key := _norm_number(p.data.invoice_number):
            by_number[key].append(pi)
    number_candidates = [
        (_location_score(t, predicted[pi]) or 0.0, ti, pi)
        for ti, t in enumerate(truth)
        if (key := _norm_number(t.invoice_number))
        for pi in by_number.get(key, ())
    ]
    pairs = _assign(number_candidates, used_t, used_p, "invoice_number")

    location_candidates: list[tuple[float, int, int]] = []
    for ti, t in enumerate(truth):
        if ti in used_t:
            continue
        for pi, p in enumerate(predicted):
            if pi not in used_p and (score := _location_score(t, p)):
                location_candidates.append((score, ti, pi))
    pairs += _assign(location_candidates, used_t, used_p, "location")

    for ti, t in enumerate(truth):
        if ti in used_t:
            continue
        for pi, p in enumerate(predicted):
            if pi not in used_p and _location_score(t, p) is None:
                used_t.add(ti)
                used_p.add(pi)
                pairs.append(RecordPair(ti, pi, "order"))
                break
    pairs.sort()
    missed = [ti for ti in range(len(truth)) if ti not in used_t]
    extra = [pi for pi in range(len(predicted)) if pi not in used_p]
    return RecordMatching(pairs=pairs, missed=missed, extra=extra)


# --------------------------------------------------------------------------------------------------------------------
# per-file scoring
# --------------------------------------------------------------------------------------------------------------------
@dataclass(slots=True)
class CodeTally:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: CodeTally) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def as_dict(self) -> dict[str, Any]:
        precision, recall = rate(self.tp, self.tp + self.fp), rate(self.tp, self.tp + self.fn)
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "support": self.tp + self.fn,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        }


@dataclass(slots=True)
class FileScore:
    """Everything one file contributes to the aggregate metrics."""

    matching: RecordMatching
    comparisons: list[RecordComparison]  # parallel to ``matching.pairs``
    missed_line_items: int  # truth items of missed records
    extra_line_items: int  # predicted items of extra records
    validation: dict[str, CodeTally]
    clean_total: int = 0
    clean_flagged: int = 0
    clean_review: int = 0
    extra_flags: int = 0  # evaluated codes carried by extra records

    def summary(self) -> dict[str, Any]:
        """A compact per-file digest stored next to the file's result."""
        core_total = len(CORE_FIELDS) * (len(self.comparisons) + len(self.matching.missed))
        correct = sum(o is FieldOutcome.CORRECT for c in self.comparisons for o in c.core.values())
        items = LineItemScore()
        for c in self.comparisons:
            items.add(c.line_items)
        return {
            "matched": len(self.matching.pairs),
            "missed": len(self.matching.missed),
            "extra": len(self.matching.extra),
            "core_correct": correct,
            "core_total": core_total,
            "exact_records": sum(c.exact_match for c in self.comparisons),
            "wrong_fields": sorted(
                {name for c in self.comparisons for name, o in c.core.items() if o is not FieldOutcome.CORRECT}
            ),
            "line_items": {"truth": items.truth, "predicted": items.predicted, "matched": items.matched},
            "validation_fp": sorted(code for code, t in self.validation.items() if t.fp),
            "validation_fn": sorted(code for code, t in self.validation.items() if t.fn),
        }


def _expected_codes(label: LabeledFile, index: int) -> set[str]:
    """The evaluated codes record ``index`` of ``label`` must carry.

    ``record_issues`` must be parallel to ``records``; the only accepted exception is a single-record file without
    ``record_issues`` (its ``expected_issue_codes`` are the record's). Anything else is a malformed label and raises
    ``ValueError`` rather than silently scoring the file as expecting no codes."""
    _check_record_issues(label)
    if len(label.record_issues) == len(label.records):
        codes: Iterable[IssueCode] = label.record_issues[index]
    else:
        codes = label.expected_issue_codes
    return {IssueCode(c).value for c in codes} & set(VALIDATION_CODES)


def _check_record_issues(label: LabeledFile) -> None:
    if len(label.record_issues) == len(label.records) or (not label.record_issues and len(label.records) == 1):
        return
    raise ValueError(
        f"{label.file}: record_issues has {len(label.record_issues)} entries for {len(label.records)} records"
    )


def score_file(label: LabeledFile, result: FileResult) -> FileScore:
    """Match the file's records and score fields, line items and validation flags (``ValueError`` when the label's
    ``record_issues`` is not parallel to its ``records``)."""
    _check_record_issues(label)
    matching = match_records(label.records, result.records)
    comparisons = [compare_records(label.records[p.truth], result.records[p.pred].data) for p in matching.pairs]
    validation = {code: CodeTally() for code in VALIDATION_CODES}
    score = FileScore(
        matching=matching,
        comparisons=comparisons,
        missed_line_items=sum(len(label.records[ti].line_items) for ti in matching.missed),
        extra_line_items=sum(len(result.records[pi].data.line_items) for pi in matching.extra),
        validation=validation,
    )
    for pair in matching.pairs:
        expected = _expected_codes(label, pair.truth)
        record = result.records[pair.pred]
        predicted = set(record.issue_codes) & set(VALIDATION_CODES)
        for code in expected & predicted:
            validation[code].tp += 1
        for code in predicted - expected:
            validation[code].fp += 1
        for code in expected - predicted:
            validation[code].fn += 1
        if not expected:
            score.clean_total += 1
            score.clean_flagged += bool(predicted)
            score.clean_review += record.review_status == "needs_review"
    for ti in matching.missed:
        for code in _expected_codes(label, ti):
            validation[code].fn += 1
    score.extra_flags = sum(len(set(result.records[pi].issue_codes) & set(VALIDATION_CODES)) for pi in matching.extra)
    return score


# --------------------------------------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------------------------------------
FilePair = tuple[LabeledFile, FileResult]


@dataclass(slots=True)
class _FieldTally:
    total: int = 0
    correct: int = 0
    correct_fuzzy: int = 0
    wrong: int = 0
    missed_value: int = 0
    hallucinated: int = 0
    null_truth: int = 0
    missed_record: int = 0

    def as_dict(self, with_fuzzy: bool) -> dict[str, Any]:
        out: dict[str, Any] = {
            "accuracy": rate(self.correct, self.total),
            "correct": self.correct,
            "total": self.total,
            "wrong_value": self.wrong,
            "missed_value": self.missed_value,
            "hallucinated": self.hallucinated,
            "null_truth": self.null_truth,
            "missed_record": self.missed_record,
        }
        if with_fuzzy:
            out["accuracy_fuzzy"] = rate(self.correct_fuzzy, self.total)
        return out


def _tally_fields(
    names: Sequence[str], scores: Sequence[FileScore], pick: Callable[[RecordComparison], dict[str, FieldOutcome]]
) -> dict[str, _FieldTally]:
    tallies = {name: _FieldTally() for name in names}
    for score in scores:
        for comparison in score.comparisons:
            outcomes = pick(comparison)
            for name in names:
                tally, outcome = tallies[name], outcomes[name]
                tally.total += 1
                tally.correct += outcome is FieldOutcome.CORRECT
                fuzzy = comparison.core_fuzzy.get(name, outcome)
                tally.correct_fuzzy += fuzzy is FieldOutcome.CORRECT
                tally.wrong += outcome is FieldOutcome.WRONG
                tally.missed_value += outcome is FieldOutcome.MISSED
                tally.hallucinated += outcome is FieldOutcome.HALLUCINATED
                tally.null_truth += name in comparison.truth_null
        for name in names:
            tallies[name].total += len(score.matching.missed)
            tallies[name].missed_record += len(score.matching.missed)
    return tallies


def _fields_section(scores: Sequence[FileScore]) -> tuple[dict[str, Any], dict[str, Any]]:
    core = _tally_fields(CORE_FIELDS, scores, lambda c: c.core)
    secondary = _tally_fields(SECONDARY_FIELDS, scores, lambda c: c.secondary)
    total = sum(t.total for t in core.values())
    correct = sum(t.correct for t in core.values())
    matched_total = sum(t.total - t.missed_record for t in core.values())
    fields = {
        "overall": {
            "accuracy": rate(correct, total),
            "accuracy_fuzzy": rate(sum(t.correct_fuzzy for t in core.values()), total),
            "correct": correct,
            "total": total,
        },
        "matched_only": {"accuracy": rate(correct, matched_total), "correct": correct, "total": matched_total},
        "per_field": {name: core[name].as_dict(name in NAME_FIELDS) for name in CORE_FIELDS},
        "secondary": {name: secondary[name].as_dict(False) for name in SECONDARY_FIELDS},
    }
    null_truth = sum(t.null_truth for t in core.values())
    hallucinated = sum(t.hallucinated for t in core.values())
    hallucination = {
        "rate": rate(hallucinated, null_truth),
        "count": hallucinated,
        "null_truth": null_truth,
        "per_field": {
            name: {"count": t.hallucinated, "null_truth": t.null_truth, "rate": rate(t.hallucinated, t.null_truth)}
            for name, t in core.items()
        },
    }
    return fields, hallucination


def _outcomes(pairs: Sequence[FilePair]) -> dict[str, Any]:
    """Success/failure buckets. Every ``expect=success`` file lands in exactly one of succeeded / empty_result /
    failed / other (a non-terminal status such as ``queued`` or ``processing``: the pipeline did not finish it),
    so the four sum to ``expect_success``; ``failure_rate`` counts failed + other."""
    expect_success = [(lab, res) for lab, res in pairs if lab.expect == "success"]
    expect_failure = [(lab, res) for lab, res in pairs if lab.expect == "failure"]
    succeeded = failed = empty = other = 0
    for lab, res in expect_success:
        if res.status in SUCCESS_STATUSES:
            if lab.records and not res.records:
                empty += 1
            else:
                succeeded += 1
        elif res.status in FAILURE_STATUSES:
            failed += 1
        else:
            other += 1
    handled = [
        (lab, res)
        for lab, res in expect_failure
        if res.status in {"failed", "rejected"} and res.error_code in TYPED_ERROR_CODES
    ]
    exact = sum(res.error_code == lab.expected_error_code for lab, res in handled)
    return {
        "expect_success": len(expect_success),
        "succeeded": succeeded,
        "success_rate": rate(succeeded, len(expect_success)),
        "failed": failed,
        "other": other,
        "failure_rate": rate(failed + other, len(expect_success)),
        "empty_result": empty,
        "empty_rate": rate(empty, len(expect_success)),
        "status_counts": dict(sorted(Counter(res.status for _, res in pairs).items())),
        "expect_failure": len(expect_failure),
        "handled": len(handled),
        "handling_rate": rate(len(handled), len(expect_failure)),
        "exact_code": exact,
        "exact_code_rate": rate(exact, len(expect_failure)),
        "failure_codes": dict(sorted(Counter(str(res.error_code) for _, res in expect_failure).items())),
    }


def _processing(pairs: Sequence[FilePair]) -> dict[str, Any]:
    timed = [res for lab, res in pairs if lab.expect == "success" and res.seconds is not None]
    seconds = [res.seconds for res in timed if res.seconds is not None]
    pages = sum(res.page_count or 0 for res in timed)
    total = sum(seconds)
    return {
        "files_timed": len(timed),
        "seconds": timing_stats(seconds),
        "processing_ms": timing_stats([float(res.processing_ms) for res in timed if res.processing_ms is not None]),
        "pages": pages,
        "pages_per_s": rate(pages, total),
        "files_per_min": rate(len(timed) * 60, total),
    }


def _llm(pairs: Sequence[FilePair]) -> dict[str, Any]:
    records = [r for _, res in pairs for r in res.records]
    return {
        "llm_fallback_records": sum(res.llm_fallbacks for _, res in pairs),
        "records_llm_unavailable": sum(LLM_UNAVAILABLE in r.issue_codes for r in records),
        "files_llm_unavailable": sum(
            LLM_UNAVAILABLE in res.document_issue_codes or any(LLM_UNAVAILABLE in r.issue_codes for r in res.records)
            for _, res in pairs
        ),
        "record_methods": dict(sorted(Counter(str(r.extraction_method) for r in records).items())),
        "llm_models": dict(sorted(Counter(res.llm_model for _, res in pairs if res.llm_model).items())),
    }


def _duplicate_groups(pairs: Sequence[FilePair]) -> dict[str, tuple[int, bool]]:
    """``dup-N`` group -> (members in the run, detected)."""
    members: dict[str, list[FileResult]] = defaultdict(list)
    for lab, res in pairs:
        if lab.group and lab.group.startswith("dup-"):
            members[lab.group].append(res)
    dup = IssueCode.DUPLICATE_INVOICE.value
    return {
        group: (
            len(results),
            any(dup in res.document_issue_codes or any(dup in r.issue_codes for r in res.records) for res in results),
        )
        for group, results in members.items()
    }


def compute_section(
    pairs: Sequence[FilePair], scores: dict[str, FileScore], dup_groups: dict[str, tuple[int, bool]]
) -> dict[str, Any]:
    """Every metric over one subset of files (``scores`` and ``dup_groups`` cover the whole run)."""
    file_scores = [scores[lab.file] for lab, _ in pairs]
    fields, hallucination = _fields_section(file_scores)
    truth_records = sum(len(lab.records) for lab, _ in pairs)
    predicted_records = sum(len(res.records) for _, res in pairs)
    matched = sum(len(s.matching.pairs) for s in file_scores)
    methods = Counter(p.method for s in file_scores for p in s.matching.pairs)
    exact = sum(c.exact_match for s in file_scores for c in s.comparisons)

    items = LineItemScore()
    for s in file_scores:
        for c in s.comparisons:
            items.add(c.line_items)
        items.truth += s.missed_line_items
        items.predicted += s.extra_line_items

    tallies = {code: CodeTally() for code in VALIDATION_CODES}
    for s in file_scores:
        for code, tally in s.validation.items():
            tallies[code].add(tally)
    micro = CodeTally()
    for tally in tallies.values():
        micro.add(tally)
    groups = {lab.group for lab, _ in pairs if lab.group in dup_groups and dup_groups[lab.group][0] >= 2}
    detected = sum(dup_groups[g][1] for g in groups)
    clean_total = sum(s.clean_total for s in file_scores)
    clean_flagged = sum(s.clean_flagged for s in file_scores)
    clean_review = sum(s.clean_review for s in file_scores)

    return {
        "files": len(pairs),
        "records": {
            "truth": truth_records,
            "predicted": predicted_records,
            "matched": matched,
            "missed": truth_records - matched,
            "extra": predicted_records - matched,
            "recall": rate(matched, truth_records),
            "precision": rate(matched, predicted_records),
            "match_methods": {m: methods.get(m, 0) for m in ("invoice_number", "location", "order")},
        },
        "fields": fields,
        "record_exact_match": {"rate": rate(exact, truth_records), "count": exact, "total": truth_records},
        "hallucination": hallucination,
        "line_items": items.as_dict(),
        "validation": {
            "per_code": {code: tallies[code].as_dict() for code in VALIDATION_CODES},
            "micro": micro.as_dict(),
            "duplicate_pairs": {"pairs": len(groups), "detected": detected, "rate": rate(detected, len(groups))},
            "clean_records": {
                "total": clean_total,
                "flagged": clean_flagged,
                "false_positive_rate": rate(clean_flagged, clean_total),
                "sent_to_review": clean_review,
                "review_rate": rate(clean_review, clean_total),
            },
            "flags_on_extra_records": sum(s.extra_flags for s in file_scores),
        },
        "processing": _processing(pairs),
        "outcomes": _outcomes(pairs),
        "llm": _llm(pairs),
    }


def _breakdown(
    pairs: Sequence[FilePair],
    key: Callable[[LabeledFile], str],
    order: Sequence[str],
    scores: dict[str, FileScore],
    dup_groups: dict[str, tuple[int, bool]],
) -> dict[str, Any]:
    groups: dict[str, list[FilePair]] = defaultdict(list)
    for pair in pairs:
        groups[key(pair[0])].append(pair)
    ranked = sorted(groups, key=lambda k: (order.index(k) if k in order else len(order), k))
    return {k: compute_section(groups[k], scores, dup_groups) for k in ranked}


def _pair_up(labels: Sequence[LabeledFile], results: Sequence[FileResult]) -> list[FilePair]:
    by_file = {res.file: res for res in results}
    missing = [lab.file for lab in labels if lab.file not in by_file]
    if missing:
        raise ValueError(f"no result for labeled file(s): {', '.join(missing[:10])}")
    known = {lab.file for lab in labels}
    unknown = [res.file for res in results if res.file not in known]
    if unknown:
        raise ValueError(f"result(s) for file(s) not in the labels: {', '.join(unknown[:10])}")
    return [(lab, by_file[lab.file]) for lab in labels]


def compute_metrics(labels: Sequence[LabeledFile], results: Sequence[FileResult]) -> dict[str, Any]:
    """All metrics of a run: ``overall`` plus the same section per split, file kind, quality and template.

    ``labels`` are the files that were evaluated; every one needs exactly one result.
    """
    pairs = _pair_up(labels, results)
    scores = {lab.file: score_file(lab, res) for lab, res in pairs}
    dup_groups = _duplicate_groups(pairs)
    return {
        "overall": compute_section(pairs, scores, dup_groups),
        "by_split": _breakdown(pairs, lambda f: f.split, SPLIT_ORDER, scores, dup_groups),
        "by_kind": _breakdown(pairs, lambda f: f.kind, KIND_ORDER, scores, dup_groups),
        "by_quality": _breakdown(pairs, lambda f: f.quality, QUALITY_ORDER, scores, dup_groups),
        "by_template": _breakdown(pairs, lambda f: f.template or "none", (), scores, dup_groups),
    }
