"""Evaluation report: Markdown from the results JSON files that ``eval.run_eval`` writes.

Usage (from ``backend/``)::

    python -m eval.report eval/results/*.json --out ../docs/EVALUATION.md

Glob patterns are expanded here too (PowerShell does not expand them for native commands). Every number in the
report comes from a results file, and so does every statement about models, hardware, the dataset, the seed and
the embedder: what a results file does not record, the report does not claim. The methodology and limitations text
below is the hand-written part, kept in this generator so it is regenerated with the numbers and cannot drift from
them.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.domain.schemas import CORE_FIELDS
from eval.metrics import SECONDARY_FIELDS, VALIDATION_CODES
from eval.run_eval import CONFIGS

logger = logging.getLogger("eval.report")

CONFIG_ORDER: tuple[str, ...] = tuple(CONFIGS)
SUPPORTED_SCHEMA = 1
SOURCE_KEY = "_source"  # set by load_results: the results file a document came from
DASH = "–"
WORST_FILES = 10

Doc = dict[str, Any]


# --------------------------------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------------------------------
def _config_rank(doc: Doc) -> tuple[int, str, str]:
    name = doc["config"]["name"]
    return (CONFIG_ORDER.index(name) if name in CONFIG_ORDER else len(CONFIG_ORDER), name, _label(doc))


def load_results(patterns: Sequence[str]) -> list[Doc]:
    """Load the results files matching ``patterns`` (paths or globs), in canonical configuration order; each
    document remembers its file name under ``SOURCE_KEY``."""
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) if any(ch in pattern for ch in "*?[") else [pattern]
        paths.extend(Path(m) for m in matches if Path(m).is_file())
    docs = []
    for path in dict.fromkeys(paths):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("schema_version") != SUPPORTED_SCHEMA:
            raise ValueError(f"{path}: unsupported results schema {doc.get('schema_version')!r}")
        doc[SOURCE_KEY] = path.as_posix()
        docs.append(doc)
    return sorted(docs, key=_config_rank)


# --------------------------------------------------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------------------------------------------------
def _pct(value: float | None) -> str:
    return DASH if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 2) -> str:
    return DASH if value is None else f"{value:.{digits}f}"


def _get(doc: Doc | None, *path: str) -> Any:
    node: Any = doc
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _table(header: Sequence[str], rows: Sequence[Sequence[str]], align: str = "") -> str:
    """A Markdown table; ``align`` holds one letter per column (``l``/``r``), default: first left, rest right."""
    letters = align or "l" + "r" * (len(header) - 1)
    marks = [":---" if a == "l" else "---:" for a in letters]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(marks) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _label(doc: Doc) -> str:
    """A run's display name: its results file's stem when that differs from the configuration name (two runs of
    one configuration, e.g. ``rules`` and ``rules-postfix``), otherwise the configuration name."""
    name = doc["config"]["name"]
    source = doc.get(SOURCE_KEY)
    return Path(source).stem if source and Path(source).stem != name else name


def _names(docs: Sequence[Doc]) -> list[str]:
    return [_label(d) for d in docs]


def _source(doc: Doc) -> str:
    return doc.get(SOURCE_KEY) or f"eval/results/{doc['config']['name']}.json"


def _main_docs(docs: Sequence[Doc]) -> list[Doc]:
    return [d for d in docs if d.get("metrics")]


def _pr(tally: dict[str, Any] | None) -> str:
    if not tally:
        return DASH
    return f"{_pct(tally.get('precision'))} / {_pct(tally.get('recall'))}"


# --------------------------------------------------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------------------------------------------------
def _log_of(doc: Doc) -> str | None:
    """The run's log, ``logs/<results stem>.log`` next to its results file, when it exists."""
    source = doc.get(SOURCE_KEY)
    if not source:
        return None
    log = Path(source).parent / "logs" / f"{Path(source).stem}.log"
    return log.as_posix() if log.is_file() else None


def _header(docs: Sequence[Doc], generated_on: str, title: str) -> str:
    files = ", ".join(f"`{_source(d)}`" for d in docs)
    lines = [
        f"# {title}",
        "",
        f"Generated by `python -m eval.report` on {generated_on} from {len(docs)} results file(s): {files}.",
        "",
    ]
    logs = [f"`{_label(d)}`: `{log}`" for d in docs if (log := _log_of(d))]
    if logs:
        lines += ["Run logs (one per run, as the runner printed it): " + "; ".join(logs) + ".", ""]
    lines += [
        "Every number below was **measured** by `eval/run_eval.py`, which runs the real pipeline (upload sniffing, "
        "storage, parsing/OCR, extraction, validation, indexing) file by file over the labeled synthetic dataset and "
        "scores the stored results against the labels with `eval/metrics.py`. Nothing is estimated or extrapolated; "
        "a configuration without a results file is absent, not guessed. Methodology and limitations are at the end.",
    ]
    warnings: list[str] = []
    for doc in docs:
        run, config = doc["run"], doc["config"]
        reasons = []
        if run.get("limit") is not None:
            reasons.append(
                f"{run['files_evaluated']} of {run['labels_files']} labeled files (`--limit {run['limit']}`)"
            )
        if config.get("embedding_provider") != "fastembed":
            reasons.append(
                f"embedding: {config.get('embedding_provider')} (not the production embedder, so indexing time is "
                "not representative)"
            )
        if reasons:
            warnings.append(f"> **Partial run** — `{config['name']}`: " + "; ".join(reasons) + ".")
    for warning in warnings:  # one blockquote each (adjacent "> " lines would merge into one paragraph)
        lines += ["", warning]
    return "\n".join(lines)


def _summary(docs: Sequence[Doc]) -> str:
    main = _main_docs(docs)
    if not main:
        return "## Summary\n\nNo results file covers the main dataset (only scale runs)."

    def o(*path: str) -> Callable[[Doc], Any]:
        return lambda d: _get(d, "metrics", "overall", *path)

    def files(d: Doc) -> str:
        return f"{d['run']['files_evaluated']} / {d['run']['labels_files']}"

    def records(d: Doc) -> str:
        rec = _get(d, "metrics", "overall", "records") or {}
        return f"{rec.get('matched', 0)} / {rec.get('truth', 0)}"

    def dup(d: Doc) -> str:
        pairs = _get(d, "metrics", "overall", "validation", "duplicate_pairs") or {}
        return f"{pairs.get('detected', 0)} / {pairs.get('pairs', 0)}" if pairs.get("pairs") else DASH

    def runtime(d: Doc) -> str:
        return f"{d['run']['wall_seconds'] / 60:.1f} min"

    rows: list[tuple[str, Callable[[Doc], str]]] = [
        ("Files evaluated / labeled", files),
        ("Records matched / labeled", records),
        ("Field accuracy (9 core fields)", lambda d: _pct(o("fields", "overall", "accuracy")(d))),
        ("Field accuracy, fuzzy names (ratio ≥ 0.9)", lambda d: _pct(o("fields", "overall", "accuracy_fuzzy")(d))),
        (
            "Field accuracy on matched records only (records are matched by invoice number first)",
            lambda d: _pct(o("fields", "matched_only", "accuracy")(d)),
        ),
        ("Record exact match (all 9 core fields)", lambda d: _pct(o("record_exact_match", "rate")(d))),
        (
            "Hallucination rate (non-null where truth is null; 9 core fields, matched records)",
            lambda d: _pct(o("hallucination", "rate")(d)),
        ),
        ("Record recall (segmentation)", lambda d: _pct(o("records", "recall")(d))),
        ("Record precision (extra records)", lambda d: _pct(o("records", "precision")(d))),
        ("Line items F1 (amount match)", lambda d: _pct(o("line_items", "f1")(d))),
        ("Line items F1 (strict: description, qty, price)", lambda d: _pct(o("line_items", "strict", "f1")(d))),
        ("Validation flag recall (6 codes, micro)", lambda d: _pct(o("validation", "micro", "recall")(d))),
        (
            "Validation flag precision (6 codes, micro; flags on extra records reported separately)",
            lambda d: _pct(o("validation", "micro", "precision")(d)),
        ),
        ("Duplicate pairs detected", dup),
        (
            "Clean records with a false flag",
            lambda d: _pct(o("validation", "clean_records", "false_positive_rate")(d)),
        ),
        ("Clean records sent to review", lambda d: _pct(o("validation", "clean_records", "review_rate")(d))),
        ("Document success rate", lambda d: _pct(o("outcomes", "success_rate")(d))),
        ("Failure rate", lambda d: _pct(o("outcomes", "failure_rate")(d))),
        ("Corrupt files handled (typed error code)", lambda d: _pct(o("outcomes", "handling_rate")(d))),
        ("Corrupt files with the exact expected code", lambda d: _pct(o("outcomes", "exact_code_rate")(d))),
        ("Mean seconds per document", lambda d: _num(o("processing", "seconds", "mean")(d))),
        ("p50 seconds per document", lambda d: _num(o("processing", "seconds", "p50")(d))),
        ("p95 seconds per document", lambda d: _num(o("processing", "seconds", "p95")(d))),
        ("Pages per second", lambda d: _num(o("processing", "pages_per_s")(d))),
        ("Total runtime (wall clock)", runtime),
        ("LLM fallbacks to rules (records)", lambda d: str(o("llm", "llm_fallback_records")(d) or 0)),
    ]
    table = _table(["Metric", *_names(main)], [[label, *(fn(d) for d in main)] for label, fn in rows])
    return "\n\n".join(
        [
            "## Summary",
            *_fallback_warnings(main),
            "Rates are over the files each run evaluated (see the notes above, if any). Seconds are the "
            "wall clock of processing one document on the inline queue, after a one-time warm-up (model loading) "
            "that is reported in *Environment*.",
            table,
        ]
    )


def _fallback_warnings(docs: Sequence[Doc]) -> list[str]:
    """One prominent line per hybrid run in which the LLM failed for some records (those are rules-only results)."""
    lines = []
    for d in docs:
        llm = _get(d, "metrics", "overall", "llm") or {}
        fallbacks = llm.get("llm_fallback_records") or 0
        if d["config"].get("extraction_mode") != "rules" and fallbacks:
            records = _get(d, "metrics", "overall", "records", "predicted") or 0
            lines.append(
                f"> **LLM fallbacks** — `{d['config']['name']}`: {fallbacks} record(s) of {records} fell back to the "
                f"rules result because the LLM call failed, was refused or was unavailable "
                f"({llm.get('files_llm_unavailable', 0)} file(s) carry `LLM_UNAVAILABLE`); this configuration's "
                "numbers are partly rules-only."
            )
    return lines


def _per_field(docs: Sequence[Doc]) -> str:
    main = _main_docs(docs)
    if not main:
        return ""
    names = _names(main)
    rows = []
    for name in CORE_FIELDS:
        cells = []
        for d in main:
            stats = _get(d, "metrics", "overall", "fields", "per_field", name) or {}
            cell = _pct(stats.get("accuracy"))
            if "accuracy_fuzzy" in stats and stats.get("accuracy_fuzzy") != stats.get("accuracy"):
                cell += f" (fuzzy {_pct(stats.get('accuracy_fuzzy'))})"
            cells.append(cell)
        rows.append([name, *cells])
    secondary_rows = [
        [name, *(_pct(_get(d, "metrics", "overall", "fields", "secondary", name, "accuracy")) for d in main)]
        for name in SECONDARY_FIELDS
    ]

    def errors(name: str, d: Doc) -> str:
        s = _get(d, "metrics", "overall", "fields", "per_field", name) or {}
        return f"{s.get('wrong_value', 0)} / {s.get('missed_value', 0)} / {s.get('hallucinated', 0)}"

    error_rows = [[name, *(errors(name, d) for d in main)] for name in CORE_FIELDS]
    hall_rows = []
    for name in CORE_FIELDS:
        cells = []
        for d in main:
            h = _get(d, "metrics", "overall", "hallucination", "per_field", name) or {}
            cells.append(f"{h.get('count', 0)} / {h.get('null_truth', 0)}")
        hall_rows.append([name, *cells])
    return "\n\n".join(
        [
            "## Field accuracy per field",
            "Share of labeled records whose field is correct (missed records count as wrong). Names in parentheses: "
            "the fuzzy-name variant. Records are matched by invoice number first, so a misread `invoice_number` "
            "tends to turn into a location/order match or a missed record; its accuracy is not independent of "
            "the matching.",
            _table(["Field", *names], rows),
            "Secondary fields (not part of the overall accuracy):",
            _table(["Field", *names], secondary_rows),
            "Error types per core field — wrong value / printed value not extracted / value invented where nothing "
            "is printed:",
            _table(["Field", *names], error_rows),
            "Hallucinations per core field — non-null predictions / records whose truth is null:",
            _table(["Field", *names], hall_rows),
        ]
    )


def _section_cells(section: dict[str, Any] | None) -> list[str]:
    s = section or {}
    return [
        str(s.get("files", 0)),
        str(_get(s, "records", "truth") or 0),
        _pct(_get(s, "fields", "overall", "accuracy")),
        _pct(_get(s, "record_exact_match", "rate")),
        _pct(_get(s, "hallucination", "rate")),
        _pct(_get(s, "line_items", "f1")),
        _pct(_get(s, "outcomes", "success_rate")),
        _num(_get(s, "processing", "seconds", "mean")),
    ]


SECTION_HEADER = ["Files", "Records", "Field acc.", "Exact", "Halluc.", "Items F1", "Success", "Mean s"]


def _dev_vs_heldout(docs: Sequence[Doc]) -> str:
    main = _main_docs(docs)
    if not main:
        return ""
    rows = []
    for d in main:
        for split, section in (_get(d, "metrics", "by_split") or {}).items():
            rows.append([_label(d), split, *_section_cells(section)])
    return "\n\n".join(
        [
            "## Dev vs held-out",
            "`dev` = templates T1–T3 (used while building the rules), `heldout` = T4–T6 (never used for tuning; "
            "see Limitations for what the rules task knew about them), `na` = ledgers, corrupt files and other "
            "template-free files.",
            _table(["Config", "Split", *SECTION_HEADER], rows, align="ll" + "r" * len(SECTION_HEADER)),
        ]
    )


def _breakdown(docs: Sequence[Doc], key: str, title: str, label: str, note: str = "") -> str:
    """Rows = groups (kind / quality / template); per config: field accuracy / exact match, then success."""
    main = _main_docs(docs)
    if not main:
        return ""
    groups: list[str] = []
    for d in main:
        for group in _get(d, "metrics", key) or {}:
            if group not in groups:
                groups.append(group)
    first = _get(main[0], "metrics", key) or {}
    header = [
        label,
        "Files",
        "Records",
        *(f"{n} acc. / exact" for n in _names(main)),
        *(f"{n} success" for n in _names(main)),
    ]
    rows = []
    for group in groups:
        base = first.get(group) or next(
            (_get(d, "metrics", key, group) for d in main if _get(d, "metrics", key, group)), {}
        )
        accuracy = []
        success = []
        for d in main:
            s = _get(d, "metrics", key, group) or {}
            accuracy.append(
                f"{_pct(_get(s, 'fields', 'overall', 'accuracy'))} / {_pct(_get(s, 'record_exact_match', 'rate'))}"
            )
            success.append(_pct(_get(s, "outcomes", "success_rate")))
        rows.append([group, str(base.get("files", 0)), str(_get(base, "records", "truth") or 0), *accuracy, *success])
    parts = [f"## {title}"]
    if note:
        parts.append(note)
    parts.append(_table(header, rows))
    return "\n\n".join(parts)


def _validation(docs: Sequence[Doc]) -> str:
    main = _main_docs(docs)
    if not main:
        return ""
    names = _names(main)
    rows = []
    for code in VALIDATION_CODES:
        support = _get(main[0], "metrics", "overall", "validation", "per_code", code, "support") or 0
        rows.append(
            [code, str(support), *(_pr(_get(d, "metrics", "overall", "validation", "per_code", code)) for d in main)]
        )
    rows.append(["micro (all six)", DASH, *(_pr(_get(d, "metrics", "overall", "validation", "micro")) for d in main)])

    def extra(d: Doc) -> list[str]:
        v = _get(d, "metrics", "overall", "validation") or {}
        pairs, clean = v.get("duplicate_pairs") or {}, v.get("clean_records") or {}
        return [
            f"{pairs.get('detected', 0)} / {pairs.get('pairs', 0)}",
            f"{clean.get('flagged', 0)} / {clean.get('total', 0)} ({_pct(clean.get('false_positive_rate'))})",
            f"{clean.get('sent_to_review', 0)} / {clean.get('total', 0)} ({_pct(clean.get('review_rate'))})",
            str(v.get("flags_on_extra_records", 0)),
        ]

    extras = [extra(d) for d in main]
    extra_rows = [
        ["Duplicate pairs detected (either member flagged)", *(e[0] for e in extras)],
        ["Clean records carrying one of the six codes", *(e[1] for e in extras)],
        ["Clean records sent to review (any error/warning)", *(e[2] for e in extras)],
        ["Evaluated codes on extra (unmatched) records", *(e[3] for e in extras)],
    ]
    return "\n\n".join(
        [
            "## Validation flags",
            "Precision / recall per issue code over matched records (a missed record's expected codes are false "
            "negatives). Support = expected occurrences. Only these six codes have ground truth; quality-dependent "
            "codes (`LOW_CONFIDENCE`, `OCR_LOW_QUALITY`, `AMBIGUOUS_DATE`, ...) are not scored.",
            _table(["Code", "Support", *(f"{n} P / R" for n in names)], rows),
            _table(["", *names], extra_rows),
        ]
    )


def _processing(docs: Sequence[Doc]) -> str:
    main = _main_docs(docs)
    if not main:
        return ""
    names = _names(main)
    kinds: list[str] = []
    for d in main:
        for kind, section in (_get(d, "metrics", "by_kind") or {}).items():
            if kind not in kinds and _get(section, "processing", "files_timed"):
                kinds.append(kind)

    def cell(section: dict[str, Any] | None) -> str:
        sec = _get(section, "processing", "seconds") or {}
        return f"{_num(sec.get('mean'))} / {_num(sec.get('p50'))} / {_num(sec.get('p95'))}"

    rows = [[kind, *(cell(_get(d, "metrics", "by_kind", kind)) for d in main)] for kind in kinds]
    rows.append(["**all**", *(cell(_get(d, "metrics", "overall")) for d in main)])
    quality_rows = []
    for quality in _get(main[0], "metrics", "by_quality") or {}:
        quality_rows.append([quality, *(cell(_get(d, "metrics", "by_quality", quality)) for d in main)])

    def throughput(d: Doc) -> list[str]:
        p = _get(d, "metrics", "overall", "processing") or {}
        ms = p.get("processing_ms") or {}
        return [
            _num(p.get("pages_per_s")),
            _num(p.get("files_per_min"), 1),
            f"{_num(ms.get('mean'), 0)} / {_num(ms.get('p95'), 0)}",
        ]

    tp = [throughput(d) for d in main]
    stage_rows = []
    for stage in ("parsing", "extracting", "validating", "indexing"):
        cells = []
        for d in main:
            totals = [f.get("stage_seconds", {}).get(stage) for f in d.get("files", [])]
            values = [v for v in totals if v is not None]
            all_stages = sum(sum((f.get("stage_seconds") or {}).values()) for f in d.get("files", []))
            share = sum(values) / all_stages if all_stages else None
            cells.append(f"{_num(sum(values), 1)} s ({_pct(share)})" if values else DASH)
        stage_rows.append([stage, *cells])
    return "\n\n".join(
        [
            "## Processing time",
            "Seconds per document (mean / p50 / p95) by file kind and by image quality, for files expected to "
            "succeed. Measured sequentially on one machine (see Environment), SQLite, inline queue, "
            "OCR in-process unless noted; parsing includes OCR.",
            _table(["Kind", *(f"{n} mean / p50 / p95 s" for n in names)], rows),
            _table(["Quality", *(f"{n} mean / p50 / p95 s" for n in names)], quality_rows),
            _table(
                ["", *names],
                [
                    ["Pages per second (all pages / all seconds)", *(t[0] for t in tp)],
                    ["Documents per minute", *(t[1] for t in tp)],
                    ["Pipeline's own `processing_ms` (mean / p95)", *(t[2] for t in tp)],
                ],
            ),
            "Where the time goes — summed stage durations over all files (share of the staged time):",
            _table(["Stage", *names], stage_rows),
        ]
    )


def _outcomes(docs: Sequence[Doc]) -> str:
    main = _main_docs(docs)
    if not main:
        return ""
    rows = []
    problem_lines = []
    for d in main:
        out = _get(d, "metrics", "overall", "outcomes") or {}
        codes = ", ".join(f"`{code}` {n}" for code, n in (out.get("failure_codes") or {}).items()) or DASH
        rows.append(
            [
                _label(d),
                str(out.get("expect_success", 0)),
                str(out.get("succeeded", 0)),
                str(out.get("empty_result", 0)),
                str(out.get("failed", 0)),
                str(out.get("other", 0)),
                _pct(out.get("failure_rate")),
                str(out.get("expect_failure", 0)),
                str(out.get("handled", 0)),
                str(out.get("exact_code", 0)),
                codes,
            ]
        )
        for f in d.get("files", []):
            if f.get("expect", "success") != "success":
                continue
            score = f.get("score") or {}
            if f.get("status") in ("completed", "needs_review") and (score.get("matched") or not score.get("missed")):
                continue
            reason = (
                f.get("error_code")
                or f.get("harness_error")
                or ("no record found" if not f.get("records") else f.get("status"))
            )
            problem_lines.append(f"- `{d['config']['name']}`: `{f['file']}` → {f.get('status')} ({reason})")
    parts = [
        "## Success and failure",
        "Every `expect=success` file is in exactly one bucket: succeeded (ended `completed`/`needs_review` with at "
        "least one record), empty (processed but no record found), failed (ended `failed`, rejected at upload, or "
        "crashed the pipeline) or other (left unfinished in a non-terminal status); the failure rate counts failed "
        "+ other. Corrupt handling = an `expect=failure` file that failed with a typed error code (a random-bytes "
        "`.pdf` legitimately sniffs as `unsupported_file_type`, so the exact expected code is counted separately).",
        _table(
            [
                "Config",
                "Expect success",
                "Succeeded",
                "Empty",
                "Failed",
                "Other (unfinished)",
                "Failure rate",
                "Expect failure",
                "Handled",
                "Exact code",
                "Codes on expected failures",
            ],
            rows,
            align="l" + "r" * 9 + "l",
        ),
    ]
    if problem_lines:
        parts.append("Files expected to succeed that did not:\n\n" + "\n".join(problem_lines))
    return "\n\n".join(parts)


def _scale(docs: Sequence[Doc]) -> str:
    scaled = [d for d in docs if d.get("scale")]
    if not scaled:
        return (
            "## Scale suite\n\nThe scale suite (`scale/large_1000_pages.pdf`, `scale/scanned_100_pages.pdf`) was not "
            "run for any results file in this report (`--include-scale` / `--scale-only`)."
        )
    rows = []
    accuracy_rows = []
    for d in scaled:
        name = _label(d)
        for f in d["scale"]["files"]:
            stages = ", ".join(
                f"{stage} {_num(s.get('pages_per_s'), 1)}" for stage, s in (f.get("stages") or {}).items()
            )
            rows.append(
                [
                    name,
                    f["file"],
                    str(f.get("pages")),
                    str(f.get("ocr_pages")),
                    str(f.get("ocr_workers")),
                    _num(f.get("seconds"), 1),
                    _num(f.get("pages_per_s"), 1),
                    stages or DASH,
                    f"{f.get('records_predicted')} / {f.get('records_truth')}",
                ]
            )
        for res in d["scale"].get("results") or []:
            score = res.get("score") or {}
            total = score.get("core_total") or 0
            accuracy_rows.append(
                [
                    name,
                    res["file"],
                    f"{score.get('matched', 0)} / {score.get('missed', 0)} / {score.get('extra', 0)}",
                    _pct(score.get("core_correct", 0) / total) if total else DASH,
                    str(score.get("exact_records", 0)),
                ]
            )
    parts = [
        "## Scale suite",
        "Throughput on the scale files, each processed alone in its own fresh database and storage directory "
        "(application caches reset and the warm-up repeated before each; pages per second over the whole job and "
        "per stage; parsing includes OCR, and with OCR workers > 0 the worker processes start inside the timed job):",
        _table(
            [
                "Config",
                "File",
                "Pages",
                "OCR pages",
                "OCR workers",
                "Seconds",
                "Pages/s",
                "Pages/s per stage",
                "Records found / labeled",
            ],
            rows,
            align="ll" + "r" * 7,
        ),
    ]
    if accuracy_rows:
        parts += [
            "Record-level accuracy on the scale files' labels:",
            _table(
                ["Config", "File", "Matched / missed / extra", "Field acc.", "Exact records"],
                accuracy_rows,
                align="ll" + "r" * 3,
            ),
        ]
    return "\n\n".join(parts)


def _worst_files(docs: Sequence[Doc]) -> str:
    main = _main_docs(docs)
    if not main:
        return ""
    parts = [
        "## Files with the most field errors",
        f"Up to {WORST_FILES} files per configuration, ranked by wrong core fields (missed records count 9).",
    ]
    for d in main:
        ranked = []
        for f in d.get("files", []):
            score = f.get("score") or {}
            wrong = (score.get("core_total") or 0) - (score.get("core_correct") or 0)
            if wrong > 0:
                ranked.append((wrong, f))
        ranked.sort(key=lambda item: (-item[0], item[1]["file"]))
        if not ranked:
            parts.append(f"`{d['config']['name']}`: no field errors.")
            continue
        rows = []
        for wrong, f in ranked[:WORST_FILES]:
            score = f["score"]
            rows.append(
                [
                    f"`{f['file']}`",
                    f.get("status", ""),
                    f"{score.get('matched', 0)} / {score.get('missed', 0)} / {score.get('extra', 0)}",
                    f"{wrong} / {score.get('core_total', 0)}",
                    ", ".join(score.get("wrong_fields") or []) or DASH,
                ]
            )
        parts.append(f"`{d['config']['name']}`:")
        parts.append(
            _table(
                ["File", "Status", "Matched / missed / extra", "Wrong fields / total", "Fields wrong"],
                rows,
                align="ll" + "r" * 2 + "l",
            )
        )
    return "\n\n".join(parts)


def _tree_state(run: dict[str, Any]) -> str:
    dirty = run.get("dirty")
    if dirty is None:
        return "tree state unknown"
    if dirty:
        return f"uncommitted changes ({run.get('dirty_paths', '?')} paths)"
    return "clean tree"


def _server_facts(config: dict[str, Any]) -> str:
    """What the LLM server reported about itself at run time (``/v1/models`` and llama.cpp ``/props``), or ''."""
    parts = []
    for model in config.get("served_models") or []:
        if isinstance(model, str):  # results written before the server facts were structured
            parts.append(f"`{model}`")
            continue
        meta = model.get("meta") or {}
        details = []
        if isinstance(meta.get("n_params"), int | float):
            details.append(f"{meta['n_params'] / 1e9:.2f} B params")
        if isinstance(meta.get("size"), int | float):
            details.append(f"{meta['size'] / 1024**3:.2f} GiB")
        parts.append(f"`{model.get('id')}`" + (f" ({', '.join(details)})" if details else ""))
    props = config.get("server_props") or {}
    if props.get("model_file"):
        parts.append(f"model file `{props['model_file']}`")
    for key in ("n_ctx", "total_slots", "build_info"):
        if props.get(key) is not None:
            parts.append(f"{key} {props[key]}")
    return ", ".join(parts)


def _environment(docs: Sequence[Doc]) -> str:
    rows = []
    for d in docs:
        config, run, machine = d["config"], d["run"], d.get("machine") or {}
        model = config.get("llm_model") or DASH
        if config.get("llm_base_url"):
            model += f" @ {config['llm_base_url']}"
        facts = _server_facts(config)
        if facts:
            model += f" (server reported: {facts})"
        gpu = "; ".join(machine.get("gpu") or []) or DASH
        rows.append(
            [
                _label(d),
                f"`{(run.get('commit') or DASH)[:12]}` ({run.get('branch') or DASH}; {_tree_state(run)})",
                run.get("started_at", DASH),
                f"{machine.get('cpu', DASH)}, {machine.get('physical_cpus', DASH)} cores / "
                f"{machine.get('logical_cpus', DASH)} threads, {machine.get('ram_gb', DASH)} GB RAM",
                gpu,
                f"{config.get('extraction_mode')} / {config.get('llm_provider')}: {model}",
                f"{config.get('embedding_provider')} {config.get('embedding_model') or ''}".strip(),
                f"{config.get('ocr_engine')}, {config.get('ocr_workers')} workers",
                f"{run.get('warmup_seconds', DASH)} s",
            ]
        )
    commands = "\n".join(f"    {d['run'].get('command', '')}" for d in docs)
    return "\n\n".join(
        [
            "## Environment",
            _table(
                [
                    "Config",
                    "Commit",
                    "Started (UTC)",
                    "CPU / RAM",
                    "GPU",
                    "Extraction / LLM",
                    "Embedding",
                    "OCR",
                    "Warm-up",
                ],
                rows,
                align="l" * 9,
            ),
            f"OS / Python: {_get(docs[0], 'machine', 'os')} / {_get(docs[0], 'machine', 'python')}. Machine facts are "
            "read at run time (CPU name, core counts, RAM, `nvidia-smi` GPU list); the commit is HEAD when the run "
            "started and *uncommitted changes* means `git status --porcelain` was not empty then. Every effective "
            "setting of a run (secrets redacted) is in its results file under `config.settings`. Commands as run:",
            commands,
        ]
    )


def _summary_of(docs: Sequence[Doc]) -> dict[str, Any]:
    return next((s for d in docs if (s := _get(d, "run", "dataset_summary"))), {})


def _seed(docs: Sequence[Doc]) -> str:
    seeds = sorted({str(_get(d, "run", "labels_seed")) for d in docs if _get(d, "run", "labels_seed") is not None})
    return ", ".join(seeds) if seeds else "unknown"


def _split_templates(summary: dict[str, Any], split: str) -> str:
    """`` (T1, T2, T3)`` from the recorded dataset summary, or '' when it was not recorded."""
    templates = (summary.get("split_templates") or {}).get(split)
    return f" ({', '.join(templates)})" if templates else ""


def _counts(mapping: dict[str, Any] | None) -> str:
    return ", ".join(f"{key} {value}" for key, value in (mapping or {}).items()) or DASH


def _embedders(docs: Sequence[Doc]) -> str:
    names = []
    for d in docs:
        provider, model = d["config"].get("embedding_provider"), d["config"].get("embedding_model")
        name = f"{provider} `{model}`" if model else f"the `{provider}` embedder"
        if name not in names:
            names.append(name)
    return " / ".join(names) or "the configured embedder"


def _methodology(docs: Sequence[Doc]) -> str:
    summary = _summary_of(docs)
    seed = _seed(docs)
    if summary:
        templates = summary.get("templates") or []
        others = summary.get("other_templates") or []
        scale = summary.get("scale_files") or []
        dataset = (
            f"**Dataset.** `backend/eval/dataset/` is synthetic: generated by this project's own code "
            f"(`python -m eval.synth.build`, seed {seed}). {summary.get('files')} labeled files with "
            f"{summary.get('records')} records — by kind: {_counts(summary.get('by_kind'))}; by quality: "
            f"{_counts(summary.get('by_quality'))}; by split: {_counts(summary.get('by_split'))}. "
            f"{len(templates)} invoice templates ({', '.join(templates)})"
            + (f" plus {', '.join(f'`{t}`' for t in others)} files" if others else "")
            + f". {summary.get('expect_failure', 0)} files must fail (corrupt, empty, encrypted or mis-typed); "
            f"{summary.get('duplicate_pairs', 0)} duplicate pairs (an original and a degraded copy); a "
            f"cross-verification group of {summary.get('cross_verification_files', 0)} files (a ledger and the PDFs "
            f"its rows describe). Expected issue codes in the labels: {_counts(summary.get('expected_codes'))}."
        )
        if scale:
            files = ", ".join(f"`{s['file']}` ({s['pages']} pages, {s['records']} records)" for s in scale)
            dataset += f" Scale files (run only with `--include-scale` / `--scale-only`): {files}."
    else:
        dataset = (
            "**Dataset.** `backend/eval/dataset/` is synthetic: generated by this project's own code "
            f"(`python -m eval.synth.build`, seed {seed}); the results files do not record its composition."
        )
    dataset += (
        " Template styles as implemented in `eval/synth/templates.py`: T1 US letter, T2 EU VAT, T3 UK, T4 Indian "
        "GST, T5 80 mm receipt, T6 modern two-column. Labels (`labels.json`) are exactly what is printed, "
        "normalised; anything not printed is `null`; expected issue codes are derived from the labels."
    )
    database = (
        "SQLite"
        if all(str(_get(d, "config", "settings", "database_url") or "sqlite").startswith("sqlite") for d in docs)
        else "the configured"
    )
    ocr = " / ".join(sorted({str(d["config"].get("ocr_engine")) for d in docs}))
    return "\n\n".join(
        [
            "## Methodology",
            dataset,
            "**Splits.** `dev` = the templates the rules were developed against"
            f"{_split_templates(summary, 'dev')}; `heldout` = templates never used for tuning or debugging the "
            f"extractor{_split_templates(summary, 'heldout')}; `na` = template-free files"
            f"{_split_templates(summary, 'na')}. See Limitations for what the rules task knew about the held-out "
            "templates.",
            "**Pipeline under test.** For each file, in label order: read the bytes, sniff the type from magic bytes "
            "exactly as the upload endpoint does (an empty or unsupported file is rejected with its typed code), "
            "store it, create the document and job, commit, and process it on the inline queue — the same "
            f"`process_job` the Celery worker runs, with the same retries. The main set runs in a fresh {database} "
            "database and storage directory, each scale file in its own fresh one; `.env` is ignored so a "
            "developer's settings cannot leak in. Results are read back after the whole set, because duplicate and "
            "cross-verification flags on earlier documents depend on later uploads.",
            "**Record matching (per file).** 1) by `normalize_invoice_number` equality; 2) by location — overlapping "
            "page ranges (intersection over union) or the same sheet row; 3) by order where locations do not "
            "contradict. Assignment is greedy by score. Unmatched labeled records are *missed* and all their fields "
            "count as wrong; unmatched predictions are *extra* records (reported as record precision; they do not "
            "enter field accuracy, and their validation flags are counted separately).",
            "**Field comparison.** Supplier/customer names by `normalize_name` equality (case, punctuation, accents, "
            'leading "the" and legal suffixes such as Ltd/GmbH ignored); a fuzzy variant accepts a `rapidfuzz` ratio '
            "≥ 0.9 of the normalised names. Invoice numbers by `normalize_invoice_number` (alphanumerics, upper case). "
            "Dates exactly. Amounts and the tax rate within 0.01. Currency exactly. A null label is correct only if "
            "the prediction is null too; a non-null prediction there is a **hallucination** (hallucination rate = "
            "such predictions / null labels over the nine core fields of matched records). This counts any value put "
            "where the document prints none for that field — including a printed value assigned to the wrong field "
            "(e.g. a delivery date taken as the missing invoice date), not only invented values. `line_items` counts "
            "as correct only when every labeled item is strictly matched and none is extra. **Record exact match** = "
            "all nine core fields correct.",
            "**Line items.** Candidate pairs have equal amounts (within 0.01); pairs are taken greedily by "
            "description similarity, so two items with the same amount go to the descriptions they resemble. A "
            "strict match also needs a description ratio ≥ 0.9 (case and whitespace ignored) and equal quantity and "
            "unit price. Precision/recall are micro-averaged over all items; items of missed records count against "
            "recall, items of extra records against precision.",
            "**Validation flags.** For `TOTAL_MISMATCH`, `LINE_ITEMS_SUM_MISMATCH`, `LINE_ITEM_MATH`, "
            "`MISSING_FIELD`, `DUPLICATE_INVOICE` and `CROSS_VERIFICATION_MISMATCH`: expected codes per labeled "
            "record vs the codes on the matched record. A duplicate pair counts as detected when either member "
            "carries `DUPLICATE_INVOICE` (the per-code numbers are strict about which one: the later upload). A "
            "*clean* record has none of the six codes expected.",
            "**Timing.** Wall clock around `enqueue` (= processing incl. retries) per document, sequential, after a "
            f"warm-up that imports the pipeline and loads the OCR engine (`{ocr}`) and the embedding model once (a "
            "long-running worker pays that once; repeated for each fresh environment). `processing_ms` is the "
            "pipeline's own measurement from claim to commit. Pages per second = pages processed / seconds. "
            f"Indexing (chunking + embeddings with {_embedders(docs)}) is part of the time.",
        ]
    )


def _llm_statement(d: Doc) -> str:
    config = d["config"]
    name, model = _label(d), config.get("llm_model") or "unknown"
    text = f"`{name}` — configured model `{model}`"
    answered = _get(d, "metrics", "overall", "llm", "llm_models") or {}
    if answered:
        text += "; models recorded on the extracted documents: " + ", ".join(f"`{m}`" for m in answered)
    if config.get("llm_provider") == "local":
        facts = _server_facts(config)
        text += f"; the server reported {facts}" if facts else "; the server reported nothing about itself"
    return text


def _limitations(docs: Sequence[Doc]) -> str:
    measured = {d["config"]["name"] for d in docs}
    missing = [name for name in CONFIG_ORDER if name not in measured]
    llm_docs = [d for d in docs if d["config"].get("extraction_mode") != "rules"]
    templates = len(_summary_of(docs).get("templates") or [])
    template_text = f"{templates} templates" if templates else "a handful of templates"
    lines = [
        "## Limitations",
        "",
        f"- **The data is synthetic**, generated by this project's own code from {template_text}. It exercises the "
        "formats, degradations and error cases the system must handle, but layouts, vocabulary and noise are far "
        "less varied than real supplier documents. **Accuracy on real-world documents will differ** — probably "
        "lower for layouts unlike these templates and for real scanner and camera noise; these numbers measure the "
        "system on this dataset, not in production.",
        "- **What held-out means here.** The held-out templates T4–T6 were never used for tuning — no rule was "
        "written or adjusted while looking at them. However, the rules extractor was designed to support their *style "
        "categories* (an Indian GST invoice, a receipt, a two-column layout), so the held-out "
        "split is blind to the concrete layouts, not to the categories. It is a fairer test than the dev split, "
        "not a fully blind one.",
        "- **LLM configurations measured in this report:** "
        + ("; ".join(_llm_statement(d) for d in llm_docs) if llm_docs else "none")
        + ". Each hybrid result holds only for the model and server it was measured with; one model's numbers say "
        "nothing about another (local or cloud), and LLM latency depends on the server and, for cloud APIs, the "
        "network.",
    ]
    if missing:
        needs = "; ".join(f"`{name}` (needs {CONFIGS[name].needs})" for name in missing)
        lines += ["", f"Configurations not measured in this report (no results file): {needs}.", ""]
    else:
        lines.append(f"- All {len(CONFIG_ORDER)} configurations have a results file.")
    lines += [
        "- **Ground truth covers six issue codes.** Quality-dependent flags (`LOW_CONFIDENCE`, `OCR_LOW_QUALITY`, "
        "`AMBIGUOUS_DATE`, `CURRENCY_INFERRED`, `EXTRACTION_DISAGREEMENT`, ...) have no labels, so the review rate "
        "of clean records includes legitimate low-confidence flags on scans.",
        "- **One machine, sequential, SQLite.** Times are for one document at a time on the machine listed under "
        "Environment, with SQLite instead of PostgreSQL and no queue overhead; a deployment with several workers "
        "processes documents in parallel.",
        "- **Matching is lenient by design** (normalised names, invoice numbers without punctuation, amounts within "
        "0.01); the fuzzy-name and matched-only accuracies are reported next to the strict ones so the effect of the "
        "rules is visible.",
    ]
    return "\n".join(lines)


def _reproduce(docs: Sequence[Doc]) -> str:
    seed = _seed(docs).split(", ")[0]
    scale_flag = " --scale" if _summary_of(docs).get("scale_files") else ""
    servers = ", ".join(
        f"`{c.env.get('LLM_MODEL')}` at {c.llm_server} for `{c.name}`" for c in CONFIGS.values() if c.llm_server
    )
    secrets = ", ".join(f"{c.required_secret} for `{c.name}`" for c in CONFIGS.values() if c.required_secret)
    commands = [
        f"python -m eval.synth.build --out eval/dataset --seed {seed}{scale_flag}",
        *(
            f"python -m eval.run_eval --dataset eval/dataset --config {name} "
            + ("--include-scale " if name == "rules" and scale_flag else "")
            + f"--out eval/results/{name}.json"
            for name in CONFIGS
        ),
        "python -m eval.report eval/results/*.json --out ../docs/EVALUATION.md",
    ]
    return "\n\n".join(
        [
            "## How to reproduce",
            "From `backend/` with the virtual environment active (the commands each run recorded are under "
            "Environment):",
            "\n".join(f"    {command}" for command in commands),
            f"The hybrid configurations need an OpenAI-compatible LLM server ({servers}) or an API key in the "
            f"environment ({secrets}); without them the runner refuses to run. Options: `--limit N` (first N "
            "files), `--embedding hashing` (fast, not the production embedder), `--ocr-workers N`, `--scale-only`. "
            "The scorer is `eval/metrics.py` (unit-tested in `tests/unit/eval/test_metrics.py`); every metric can "
            "be recomputed from the committed results JSON and `labels.json`.",
        ]
    )


def render_report(
    docs: Sequence[Doc],
    *,
    generated_on: str | None = None,
    title: str = "Evaluation report",
    notes: str | None = None,
) -> str:
    """The full Markdown report for ``docs`` (results documents, any order). ``notes`` (Markdown written by hand, such
    as a "Run notes" section) is placed right after the summary, so regenerating the report keeps it."""
    if not docs:
        raise ValueError("no results to report")
    ordered = sorted(docs, key=_config_rank)
    generated_on = generated_on or datetime.now(UTC).strftime("%Y-%m-%d")
    sections = [
        _header(ordered, generated_on, title),
        _summary(ordered),
        notes.strip() if notes else "",
        _per_field(ordered),
        _dev_vs_heldout(ordered),
        _breakdown(
            ordered,
            "by_kind",
            "By file kind",
            "Kind",
            "`corrupt` = files that must fail; a PNG renamed `.pdf` counts as `png` (its true type).",
        ),
        _breakdown(
            ordered,
            "by_quality",
            "By image quality",
            "Quality",
            "`clean` includes digital PDFs and spreadsheets; the others are rendered-and-degraded images/scans.",
        ),
        _breakdown(
            ordered,
            "by_template",
            "By template",
            "Template",
            "`ledger` = spreadsheet ledgers (one record per row), `mixed` = scale files, `none` = corrupt files.",
        ),
        _validation(ordered),
        _processing(ordered),
        _outcomes(ordered),
        _scale(ordered),
        _worst_files(ordered),
        _environment(ordered),
        _methodology(ordered),
        _limitations(ordered),
        _reproduce(ordered),
    ]
    return "\n\n".join(s for s in sections if s) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.report", description="Write the Markdown evaluation report.")
    parser.add_argument("results", nargs="+", help="results JSON files or glob patterns")
    parser.add_argument("--out", type=Path, required=True, help="Markdown file to write")
    parser.add_argument("--title", default="Evaluation report", help="the report's top-level heading")
    parser.add_argument("--notes", type=Path, help="hand-written Markdown to place after the summary")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    docs = load_results(args.results)
    if not docs:
        logger.error("no results files match %s", " ".join(args.results))
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    notes = args.notes.read_text(encoding="utf-8") if args.notes else None
    args.out.write_text(render_report(docs, title=args.title, notes=notes), encoding="utf-8", newline="\n")
    logger.info("wrote %s (%d configuration(s))", args.out, len(docs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
