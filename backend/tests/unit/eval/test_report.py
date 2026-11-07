"""Tests for the Markdown evaluation report (``eval.report``), from small hand-made results documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.report import load_results, main, render_report
from tests.unit.eval.builders import results_document


def _row(markdown: str, first_cell: str) -> str:
    """The table row whose first cell is ``first_cell``."""
    for line in markdown.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and cells and cells[0] == first_cell:
            return line
    raise AssertionError(f"no table row starting with {first_cell!r}")


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def _two_configs() -> list[dict[str, Any]]:
    # rules gets the dev total wrong (16/18 core fields); the 9B config only misses the held-out customer (17/18)
    return [
        results_document("hybrid-local-9b", llm_model="qwen3.5-9b", seconds=40.0),
        results_document("rules", total_offset="5"),
    ]


def test_summary_table_compares_configs_in_canonical_order() -> None:
    report = render_report(_two_configs())
    header = _row(report, "Metric")
    assert _cells(header)[1:] == ["rules", "hybrid-local-9b"]
    assert _cells(_row(report, "Field accuracy (9 core fields)"))[1:] == ["88.9%", "94.4%"]
    assert _cells(_row(report, "Record exact match (all 9 core fields)"))[1:] == ["0.0%", "50.0%"]
    assert _cells(_row(report, "Document success rate"))[1:] == ["100.0%", "100.0%"]
    assert _cells(_row(report, "Corrupt files handled (typed error code)"))[1:] == ["100.0%", "100.0%"]
    assert _cells(_row(report, "Mean seconds per document"))[1:] == ["2.50", "100.00"]


def test_per_field_and_breakdown_tables() -> None:
    report = render_report(_two_configs())
    assert _cells(_row(report, "total"))[1:] == ["50.0%", "100.0%"]
    assert _cells(_row(report, "customer_name"))[1:] == ["50.0%", "50.0%"]
    assert "## Dev vs held-out" in report
    heldout = [line for line in report.splitlines() if line.startswith("| rules | heldout")]
    assert heldout, "dev vs held-out table has a held-out row for rules"
    assert _row(report, "jpg")  # by file kind
    assert _row(report, "photo")  # by quality
    assert _row(report, "TOTAL_MISMATCH")  # validation per code


def test_limitations_are_stated_and_unmeasured_providers_named() -> None:
    report = render_report(_two_configs())
    for phrase in ("synthetic", "T4–T6", "Indian GST", "receipt", "two-column"):
        assert phrase in report, phrase
    limitations = report.split("## Limitations", 1)[1]
    assert "hybrid-gemini" in limitations and "hybrid-claude" in limitations
    assert "not measured" in limitations
    assert "real-world" in limitations
    # what an unmeasured configuration needs comes from the runner's own definitions
    assert "`hybrid-claude` (needs ANTHROPIC_API_KEY)" in limitations
    assert "http://127.0.0.1:8081/v1" in limitations

    with_gemini = render_report([*_two_configs(), results_document("hybrid-gemini", llm_model="gemini-3")])
    not_measured = with_gemini.split("## Limitations", 1)[1].split("not measured", 1)[1].split("\n\n", 1)[0]
    assert "hybrid-gemini" not in not_measured
    assert "hybrid-claude" in not_measured


def test_partial_runs_are_flagged() -> None:
    smoke = results_document("rules", limit=25, embedding="hashing")
    report = render_report([smoke])
    assert "Partial run" in report
    assert "25 of 164" in report
    assert "hashing" in report
    assert "Partial run" not in render_report([results_document("rules")])


def test_scale_section() -> None:
    assert "not run" in render_report(_two_configs()).split("## Scale", 1)[1].split("\n## ", 1)[0]
    base = results_document("rules")
    base["scale"] = {
        "files": [
            {
                "file": "scale/large_1000_pages.pdf",
                "status": "needs_review",
                "pages": 1000,
                "ocr_pages": 0,
                "ocr_workers": 0,
                "seconds": 50.0,
                "processing_ms": 49000,
                "pages_per_s": 20.0,
                "records_truth": 900,
                "records_predicted": 897,
                "stages": {"parsing": {"seconds": 10.0, "pages_per_s": 100.0}},
            }
        ],
        "metrics": base["metrics"],
    }
    scale = render_report([base]).split("## Scale", 1)[1].split("\n## ", 1)[0]
    row = _cells(_row(scale, "rules"))
    assert "scale/large_1000_pages.pdf" in row
    assert "1000" in row and "20.0" in row
    assert "parsing 100.0" in scale


def test_cli_expands_globs_and_writes_markdown(tmp_path: Path) -> None:
    for doc in _two_configs():
        (tmp_path / f"{doc['config']['name']}.json").write_text(json.dumps(doc), encoding="utf-8")
    out = tmp_path / "EVALUATION.md"
    assert main([str(tmp_path / "*.json"), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Evaluation")
    assert [d["config"]["name"] for d in load_results([str(tmp_path / "*.json")])] == ["rules", "hybrid-local-9b"]


def test_cli_without_results_fails(tmp_path: Path) -> None:
    assert main([str(tmp_path / "nothing-*.json"), "--out", str(tmp_path / "x.md")]) != 0
    assert not (tmp_path / "x.md").exists()


SERVED_9B = {
    "served_models": [{"id": "qwen3.5-9b", "meta": {"n_params": 8950000000, "size": 5600000000}}],
    "server_props": {"model_file": "Qwen3.5-9B-Q4_K_M.gguf", "n_ctx": 16384, "total_slots": 2, "build_info": "b7001"},
}


def test_model_dataset_and_embedding_claims_come_from_the_results() -> None:
    rules = results_document("rules", seed=7, embedding="hashing")
    report = render_report([rules])
    assert "--seed 7" in report and "--seed 42" not in report
    assert "bge-small" not in report  # no run used fastembed, so the report does not name its model
    assert "2 invoice templates (T1, T4)" in report  # from the recorded dataset summary, not a fixed "6"
    assert "developed against (T1);" in report and "debugging the extractor (T4);" in report
    assert "plus 2 scale files" not in report
    for claim in ("offloaded", "fully on the GPU", "RTX", "laptop"):
        assert claim not in report, claim

    nine_b = results_document("hybrid-local-9b", llm_model="qwen3.5-9b", server=SERVED_9B)
    report = render_report([rules, nine_b])
    assert "BAAI/bge-small-en-v1.5" in report  # the 9B run used fastembed
    limitations = report.split("## Limitations", 1)[1]
    assert "Qwen3.5-9B-Q4_K_M.gguf" in limitations  # the model file the server reported
    environment = report.split("## Environment", 1)[1].split("\n## ", 1)[0]
    assert "Qwen3.5-9B-Q4_K_M.gguf" in environment and "n_ctx 16384" in environment
    no_props = results_document("hybrid-local-4b", llm_model="qwen3-4b")
    unreported = render_report([no_props]).split("## Limitations", 1)[1]
    assert "the server reported nothing about itself" in unreported


def test_dirty_working_tree_is_shown() -> None:
    def environment(doc: dict[str, Any]) -> str:
        return render_report([doc]).split("## Environment", 1)[1].split("\n## ", 1)[0]

    assert "uncommitted changes (3 paths)" in environment(results_document("rules", dirty=True))
    assert "clean tree" in environment(results_document("rules", dirty=False))
    assert "tree state unknown" in environment(results_document("rules", dirty=None))


def test_llm_fallbacks_are_flagged_in_the_summary() -> None:
    report = render_report([results_document("rules"), results_document("hybrid-local-4b", llm_fallbacks=2)])
    summary = report.split("## Summary", 1)[1].split("\n## ", 1)[0]
    warning = [line for line in summary.splitlines() if line.startswith("> **LLM fallbacks**")]
    assert len(warning) == 1
    assert "`hybrid-local-4b`" in warning[0] and "2 record(s)" in warning[0]
    assert "LLM fallbacks**" not in render_report([results_document("hybrid-local-4b")])


def test_summary_rows_state_their_scope() -> None:
    report = render_report(_two_configs())
    assert _row(report, "Validation flag precision (6 codes, micro; flags on extra records reported separately)")
    assert _row(report, "Hallucination rate (non-null where truth is null; 9 core fields, matched records)")
    assert _row(report, "Field accuracy on matched records only (records are matched by invoice number first)")
    per_field = report.split("## Field accuracy per field", 1)[1].split("\n## ", 1)[0]
    assert "matched by invoice number first" in per_field


def test_outcome_table_has_an_other_bucket() -> None:
    report = render_report(_two_configs())
    header = _cells(next(line for line in report.splitlines() if line.startswith("| Config | Expect success")))
    assert "Other (unfinished)" in header
    row = _cells(next(line for line in report.splitlines() if line.startswith("| rules | 2 |")))
    assert row[header.index("Other (unfinished)")] == "0"


def test_two_runs_of_one_config_are_labelled_by_their_results_file(tmp_path: Path) -> None:
    files = {
        "rules-postfix.json": results_document("rules"),
        "rules.json": results_document("rules", total_offset="5"),
        "hybrid-local-9b.json": results_document("hybrid-local-9b", llm_model="qwen3.5-9b", seconds=40.0),
    }
    for name, doc in files.items():
        (tmp_path / name).write_text(json.dumps(doc), encoding="utf-8")
    docs = load_results([str(tmp_path / n) for n in files])
    report = render_report(docs)
    assert _cells(_row(report, "Metric"))[1:] == ["rules", "rules-postfix", "hybrid-local-9b"]
    assert _cells(_row(report, "Field accuracy (9 core fields)"))[1:] == ["88.9%", "94.4%", "94.4%"]
    # coverage is by configuration, not by file: a renamed results file still counts as a ``rules`` run
    not_measured = next(line for line in report.splitlines() if line.startswith("Configurations not measured"))
    assert "`rules`" not in not_measured and "`hybrid-local-9b`" not in not_measured


def test_title_and_hand_written_notes_after_the_summary() -> None:
    report = render_report(_two_configs(), title="Evaluation: Qwen3-4B", notes="## Run notes\n\n- measured twice\n")
    assert report.startswith("# Evaluation: Qwen3-4B\n")
    sections = [line for line in report.splitlines() if line.startswith("## ")]
    assert sections[:3] == ["## Summary", "## Run notes", "## Field accuracy per field"]
    assert "- measured twice" in report


def test_cli_takes_notes_and_names_each_runs_log(tmp_path: Path) -> None:
    results = tmp_path / "results"
    (results / "logs").mkdir(parents=True)
    for doc in _two_configs():
        (results / f"{doc['config']['name']}.json").write_text(json.dumps(doc), encoding="utf-8")
    (results / "logs" / "rules.log").write_text("INFO done\n", encoding="utf-8")  # hybrid-local-9b has no log
    notes = tmp_path / "notes.md"
    notes.write_text("## Run notes\n\n- one machine\n", encoding="utf-8")
    out = tmp_path / "report.md"

    argv = [str(results / "*.json"), "--title", "Evaluation: rules", "--notes", str(notes), "--out", str(out)]
    assert main(argv) == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Evaluation: rules\n") and "- one machine" in text
    log_line = next(line for line in text.splitlines() if line.startswith("Run logs"))
    assert "logs/rules.log" in log_line and "hybrid-local-9b" not in log_line
