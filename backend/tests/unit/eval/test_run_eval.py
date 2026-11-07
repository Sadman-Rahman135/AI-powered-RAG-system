"""Tests for the evaluation runner (``eval.run_eval``): end-to-end runs over a tiny dataset through the real pipeline
(CSV ledgers, empty file, non-document), environment isolation, the scale suite's one-database-per-file rule,
config refusal and the facts the runner records (LLM server, git state, settings). No network, no git process."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.config import Settings, reset_settings_cache
from app.domain.schemas import DocumentType, SourceKind
from eval import run_eval
from eval.labels import LabeledFile, LabeledRecord, write_labels

LEDGER = (
    "Invoice No,Invoice Date,Due Date,Supplier,Customer,Currency,Subtotal,Tax,Total\n"
    "UK-1001,2023-01-24,2023-02-23,Kingsway Stationery Ltd,Patel Facilities Ltd,GBP,100.00,20.00,120.00\n"
    "UK-1002,2023-01-30,2023-02-13,Wright & Jones Ltd,Albion Associates,GBP,50.00,10.00,60.00\n"
)
CORRUPT: dict[str, Any] = {"kind": "corrupt", "template": None, "split": "na", "quality": "clean", "expect": "failure"}
# every variable the runner may write: the base environment of a run plus any configuration's own
RUNNER_KEYS = sorted(
    {
        "DOCINTEL_IGNORE_DOTENV", "APP_ENV", "QUEUE_BACKEND", "DATABASE_URL", "STORAGE_DIR", "EMBEDDING_PROVIDER",
        "OCR_WORKERS", "RATE_LIMIT_ENABLED", "LOG_JSON", "GEMINI_API_KEY", "ANTHROPIC_API_KEY",
        *(key for config in run_eval.CONFIGS.values() for key in config.env),
    }
)  # fmt: skip


@pytest.fixture
def runner_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Register every runner variable with monkeypatch (restored even if the runner's own restore failed), keep the
    runner off git and the OCR model, and reset the application caches afterwards."""
    for key in RUNNER_KEYS:
        if key in os.environ:
            monkeypatch.setenv(key, os.environ[key])
        else:  # register the absent key so that monkeypatch removes it again at teardown
            monkeypatch.setenv(key, "")
            monkeypatch.delenv(key)
    monkeypatch.setattr(run_eval, "warm_up", lambda: 0.0)
    monkeypatch.setattr(run_eval, "git_dirty", lambda *a, **k: {"dirty": False, "dirty_paths": 0})
    monkeypatch.setattr(run_eval, "_secret_from_dotenv", lambda name: None)
    yield
    run_eval.reset_app_caches()


def _ledger_rows() -> list[LabeledRecord]:
    rows = [
        ("UK-1001", date(2023, 1, 24), date(2023, 2, 23), "Kingsway Stationery Ltd", "Patel Facilities Ltd", "100.00",
         "20.00", "120.00", 2),
        ("UK-1002", date(2023, 1, 30), date(2023, 2, 13), "Wright & Jones Ltd", "Albion Associates", "50.00",
         "10.00", "60.00", 3),
    ]  # fmt: skip
    return [
        LabeledRecord(
            invoice_number=number,
            invoice_date=day,
            due_date=due,
            supplier_name=supplier,
            customer_name=customer,
            currency="GBP",
            subtotal=Decimal(sub),
            tax=Decimal(tax),
            total=Decimal(total),
            document_type=DocumentType.INVOICE,
            row_number=row,
            source_kind=SourceKind.LEDGER_ROW,
        )
        for number, day, due, supplier, customer, sub, tax, total, row in rows
    ]


def _ledger_label(name: str) -> LabeledFile:
    return LabeledFile(
        file=name,
        kind="csv",
        template="ledger",
        split="na",
        quality="clean",
        expect="success",
        records=_ledger_rows(),
        record_issues=[[], []],
        pages=1,
    )


def _dataset(root: Path, ledgers: list[str], *, corrupt: bool = True) -> Path:
    files = []
    for name in ledgers:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(LEDGER, encoding="utf-8")  # identical bytes: a duplicate *file* if one DB sees both
        files.append(_ledger_label(name))
    if corrupt:
        (root / "corrupt").mkdir(parents=True, exist_ok=True)
        (root / "corrupt" / "empty.pdf").write_bytes(b"")
        (root / "corrupt" / "junk.pdf").write_bytes(bytes(range(256)) * 8)  # binary, no known magic bytes
        files.append(LabeledFile(file="corrupt/empty.pdf", expected_error_code="empty_file", **CORRUPT))
        files.append(LabeledFile(file="corrupt/junk.pdf", expected_error_code="unsupported_file_type", **CORRUPT))
    write_labels(root / "labels.json", seed=7, files=files)
    return root


@pytest.mark.usefixtures("runner_env")
def test_run_over_a_tiny_dataset_writes_results_and_restores_the_environment(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset", ["ledgers/ledger.csv", "ledgers/ledger_copy.csv"])
    out = tmp_path / "results" / "rules.json"
    before = dict(os.environ)
    argv = ["--dataset", str(dataset), "--config", "rules", "--embedding", "hashing", "--out", str(out)]
    assert run_eval.main(argv) == 0
    assert dict(os.environ) == before  # nothing the run set is left behind

    doc = json.loads(out.read_text(encoding="utf-8"))
    files = {f["file"]: f for f in doc["files"]}
    empty = files["corrupt/empty.pdf"]
    assert (empty["status"], empty["error_code"], empty["expect"]) == ("rejected", "empty_file", "failure")
    assert files["corrupt/junk.pdf"]["error_code"] == "unsupported_file_type"
    ledger = files["ledgers/ledger.csv"]
    assert ledger["status"] in ("completed", "needs_review")
    assert ledger["seconds"] > 0 and ledger["attempts"] == 1
    assert [r["row_number"] for r in ledger["records"]] == [2, 3]
    assert ledger["score"]["matched"] == 2
    # the main set shares one database: the second identical file is a known duplicate file
    assert "DUPLICATE_FILE" in files["ledgers/ledger_copy.csv"]["document_issue_codes"]

    overall = doc["metrics"]["overall"]
    assert overall["records"]["matched"] == 4
    assert overall["fields"]["per_field"]["total"]["accuracy"] == pytest.approx(1.0)
    assert (overall["outcomes"]["handled"], overall["outcomes"]["exact_code"]) == (2, 2)
    assert overall["outcomes"]["success_rate"] == pytest.approx(1.0)
    config, run = doc["config"], doc["run"]
    assert config["name"] == "rules" and config["extraction_mode"] == "rules"
    assert "DATABASE_URL" not in config["env"]
    settings = config["settings"]
    assert set(settings) == set(Settings.model_fields)  # every effective setting is recorded
    assert settings["embedding_provider"] == "hashing" and settings["queue_backend"] == "inline"
    assert settings["database_url"].startswith("sqlite:///<workdir>/main/")
    assert run["files_evaluated"] == 4 and run["labels_seed"] == 7
    assert (run["dirty"], run["dirty_paths"]) == (False, 0)
    assert run["dataset_summary"]["files"] == 4 and run["dataset_summary"]["by_kind"] == {"corrupt": 2, "csv": 2}
    assert "--config rules" in run["command"]
    assert doc["machine"]["cpu"]

    # the run's own log sits next to its results, named after them, so each run (model) has a separate log
    log = tmp_path / "results" / "logs" / "rules.log"
    text = log.read_text(encoding="utf-8")
    assert "[1/4]" in text and "done: 4 files" in text
    assert not list(log.parent.glob("*.partial"))


@pytest.mark.usefixtures("runner_env")
def test_scale_files_each_get_their_own_fresh_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # two identical files: in a shared database the second would be flagged DUPLICATE_FILE
    monkeypatch.setattr(run_eval, "SCALE_FILES", {"scale/a.csv": None, "scale/b.csv": None})
    dataset = _dataset(tmp_path / "dataset", ["scale/a.csv", "scale/b.csv"], corrupt=False)
    out = tmp_path / "scale.json"
    argv = ["--dataset", str(dataset), "--config", "rules", "--embedding", "hashing", "--scale-only", "--out", str(out)]
    assert run_eval.main(argv) == 0

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["metrics"] is None and doc["files"] == []
    scale_files = doc["scale"]["files"]
    assert [f["database"] for f in scale_files] == ["scale-1", "scale-2"]
    for entry in scale_files:
        assert entry["status"] in ("completed", "needs_review")
        assert "DUPLICATE_FILE" not in entry["document_issue_codes"], entry["file"]
    assert doc["scale"]["metrics"]["overall"]["records"]["matched"] == 4


@pytest.mark.usefixtures("runner_env")
def test_cloud_config_refuses_to_run_without_its_key(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset", ["ledgers/ledger.csv"])
    out = tmp_path / "gemini.json"
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "gemini.log").write_text("an earlier, successful run\n", encoding="utf-8")
    argv = ["--dataset", str(dataset), "--config", "hybrid-gemini", "--out", str(out)]
    assert run_eval.main(argv) == run_eval.EXIT_CONFIG_ERROR
    assert not out.exists()
    # a refused run leaves an earlier run's log alone and no partial log behind
    assert (tmp_path / "logs" / "gemini.log").read_text(encoding="utf-8") == "an earlier, successful run\n"
    assert not list((tmp_path / "logs").glob("*.partial"))


def test_llm_server_facts_are_recorded_from_models_and_props() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"id": "qwen3-4b", "meta": {"n_params": 4022468096, "size": 2497281024, "x": [1]}}],
                },
            )
        if request.url.path == "/props":
            return httpx.Response(
                200,
                json={
                    "model_path": "/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                    "total_slots": 2,
                    "build_info": "b7001-abc",
                    "default_generation_settings": {"n_ctx": 8192},
                    "chat_template": "{{ a very long template }}",
                },
            )
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        facts = run_eval.check_llm_server("http://127.0.0.1:8081/v1", client=client)
    assert facts["served_models"] == [{"id": "qwen3-4b", "meta": {"n_params": 4022468096, "size": 2497281024}}]
    assert facts["server_props"] == {
        "model_file": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "n_ctx": 8192,
        "total_slots": 2,
        "build_info": "b7001-abc",
    }

    def no_props(request: httpx.Request) -> httpx.Response:
        return (
            httpx.Response(200, json={"data": [{"id": "m"}]})
            if request.url.path == "/v1/models"
            else (httpx.Response(404))
        )

    with httpx.Client(transport=httpx.MockTransport(no_props)) as client:
        assert run_eval.check_llm_server("http://h:1/v1", client=client)["server_props"] is None

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with httpx.Client(transport=httpx.MockTransport(down)) as client, pytest.raises(run_eval.ConfigError):
        run_eval.check_llm_server("http://h:1/v1", client=client)


def test_git_dirty_reads_porcelain_status_without_locks(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake(output: str) -> Any:
        def runner(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout=output, stderr="")

        return runner

    assert run_eval.git_dirty(tmp_path, git="git", runner=fake(" M backend/x.py\n?? notes.txt\n")) == {
        "dirty": True,
        "dirty_paths": 2,
    }
    assert run_eval.git_dirty(tmp_path, git="git", runner=fake("")) == {"dirty": False, "dirty_paths": 0}
    assert calls[0] == ["git", "--no-optional-locks", "status", "--porcelain"]

    def failing(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(128, cmd)

    assert run_eval.git_dirty(tmp_path, git="git", runner=failing) == {"dirty": None, "dirty_paths": None}


def test_settings_record_redacts_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gm-secret-value-123")
    monkeypatch.setenv("API_KEYS", "key-one-secret,key-two-secret")
    monkeypatch.setenv("REDIS_URL", "redis://user:hunter2@cache:6379/0")
    reset_settings_cache()
    record = run_eval.settings_record()
    text = json.dumps(record)
    for secret in ("gm-secret-value-123", "key-one-secret", "hunter2"):
        assert secret not in text
    assert record["gemini_api_key"] == run_eval.REDACTED and record["api_keys"] == run_eval.REDACTED
    assert record["anthropic_api_key"] is None  # unset stays visibly unset
    assert record["redis_url"] == f"redis://{run_eval.REDACTED}@cache:6379/0"
    assert set(record) == set(Settings.model_fields)


def test_select_labels_limits_the_main_set_and_finds_scale_files() -> None:
    labels = [LabeledFile(file=f"corrupt/{i}.pdf", **CORRUPT) for i in range(3)]
    labels += [LabeledFile(file=name, **CORRUPT) for name in run_eval.SCALE_FILES]
    main, scale = run_eval.select_labels(labels, 2)
    assert [lab.file for lab in main] == ["corrupt/0.pdf", "corrupt/1.pdf"]
    assert [lab.file for lab in scale] == list(run_eval.SCALE_FILES)


def test_git_identity_reads_loose_and_packed_refs(tmp_path: Path) -> None:
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/feature/x\n", encoding="utf-8")
    packed = "# pack-refs with: peeled\n" + "a" * 40 + " refs/heads/feature/x\n"
    (git / "packed-refs").write_text(packed, encoding="utf-8")
    assert run_eval.git_identity(tmp_path) == {"commit": "a" * 40, "branch": "feature/x"}
    (git / "refs" / "heads" / "feature").mkdir()
    (git / "refs" / "heads" / "feature" / "x").write_text("b" * 40 + "\n", encoding="utf-8")
    assert run_eval.git_identity(tmp_path)["commit"] == "b" * 40
    assert run_eval.git_identity(tmp_path / "nowhere") == {"commit": None, "branch": None}
