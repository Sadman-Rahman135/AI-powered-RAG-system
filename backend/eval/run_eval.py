"""Evaluation runner: drive the real pipeline over the labeled dataset and write one results JSON per configuration.

Usage (from ``backend/``)::

    python -m eval.run_eval --dataset eval/dataset --config rules --out eval/results/rules.json
    python -m eval.run_eval --dataset eval/dataset --config hybrid-local-9b --out eval/results/hybrid-local-9b.json
    python -m eval.run_eval --dataset eval/dataset --config rules --scale-only --out eval/results/rules-scale.json

Every file goes through the same path as an upload: read the bytes, sniff the type from magic bytes (an empty or
unsupported file is rejected with the API's typed error code), store it, ``create_document`` + commit, enqueue on the
``inline`` queue (= ``process_job`` with the queue's retries), timed wall-clock. Results (status, error code, pages,
records with their issue codes, review status, stage timings) are read back from the database **after** the whole
set is processed, because duplicate and cross-verification flags on earlier documents depend on later ones.

Isolation: the main set runs in one fresh SQLite database + storage directory, and **each** scale file in its own
fresh database + storage; before each of these environments every application cache that holds state (settings,
database engine, queue, storage, embedder) is reset and the warm-up repeated. The process environment
(``os.environ``) and those caches are restored when the run ends, so the runner can be called from other code (and
tests) without side effects. ``.env`` is ignored (``DOCINTEL_IGNORE_DOTENV``) so a developer's configuration cannot
leak in; API keys for the cloud configurations come from the environment (or, for that one key only, from a ``.env``
file). One-time start-up (pipeline imports, OCR engine, embedding model) happens in a warm-up before the first file
and is reported separately from per-file times.

The results document records what the report may claim: the effective settings (secrets redacted), what the LLM
server reported about itself, the machine, the commit and whether the working tree had uncommitted changes, and a
summary of the labeled dataset.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from collections import Counter
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID

from eval.labels import LabeledFile, load_labels
from eval.metrics import FileResult, PredictedRecord, compute_metrics, score_file, stage_rates

logger = logging.getLogger("eval.run_eval")

RESULTS_SCHEMA_VERSION = 1
BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent
SCALE_PREFIX = "scale/"
SCALE_FILES: dict[str, int | None] = {  # file -> OCR_WORKERS for it (None = the run's --ocr-workers)
    "scale/large_1000_pages.pdf": None,
    "scale/scanned_100_pages.pdf": 4,
}
TIMED_STAGES = ("parsing", "extracting", "validating", "indexing")
EXIT_CONFIG_ERROR = 2
REDACTED = "<redacted>"
WORKDIR_MARK = "<workdir>"
_URL_CREDENTIALS = re.compile(r"(?<=://)[^/@]+@")
_PROPS_KEYS = ("total_slots", "build_info", "n_ctx")


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """One extractor configuration: the environment it runs under and what it needs to run at all."""

    name: str
    description: str
    env: dict[str, str]
    required_secret: str | None = None  # env var that must be set (cloud providers)
    llm_server: str | None = None  # OpenAI-compatible base URL that must answer before the run starts

    @property
    def needs(self) -> str:
        """What this configuration needs to run, for the report's list of configurations without results."""
        if self.required_secret:
            return self.required_secret
        if self.llm_server:
            return f"an OpenAI-compatible LLM server at {self.llm_server} serving `{self.env.get('LLM_MODEL')}`"
        return "nothing beyond the dataset"


def _local(name: str, port: int, model: str) -> EvalConfig:
    base_url = f"http://127.0.0.1:{port}/v1"
    return EvalConfig(
        name=name,
        description=f"Rules + local LLM `{model}` (OpenAI-compatible server at {base_url})",
        env={
            "EXTRACTION_MODE": "hybrid",
            "LLM_PROVIDER": "local",
            "LLM_BASE_URL": base_url,
            "LLM_MODEL": model,
            "LLM_MAX_INPUT_CHARS": "20000",
        },
        llm_server=base_url,
    )


CONFIGS: dict[str, EvalConfig] = {
    "rules": EvalConfig("rules", "Rules only (no LLM)", {"EXTRACTION_MODE": "rules", "LLM_PROVIDER": "none"}),
    "hybrid-local-4b": _local("hybrid-local-4b", 8081, "qwen3-4b"),
    "hybrid-local-9b": _local("hybrid-local-9b", 8080, "qwen3.5-9b"),
    "hybrid-gemini": EvalConfig(
        "hybrid-gemini",
        "Rules + Google Gemini (GEMINI_MODEL)",
        {"EXTRACTION_MODE": "hybrid", "LLM_PROVIDER": "gemini"},
        required_secret="GEMINI_API_KEY",
    ),
    "hybrid-claude": EvalConfig(
        "hybrid-claude",
        "Rules + Anthropic Claude (ANTHROPIC_MODEL)",
        {"EXTRACTION_MODE": "hybrid", "LLM_PROVIDER": "anthropic"},
        required_secret="ANTHROPIC_API_KEY",
    ),
}


class ConfigError(Exception):
    """The configuration cannot be evaluated here (missing key, LLM server down)."""


# --------------------------------------------------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------------------------------------------------
def _secret_from_dotenv(name: str) -> str | None:
    """``name`` from the repo-root or backend ``.env`` (only that key is taken; the rest of the file is ignored)."""
    from dotenv import dotenv_values

    for path in (REPO_DIR / ".env", BACKEND_DIR / ".env"):
        if path.is_file():
            value = dotenv_values(path).get(name)
            if value:
                return value
    return None


def _props_url(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1") + "/props"


def check_llm_server(base_url: str, client: Any = None) -> dict[str, Any]:
    """What an OpenAI-compatible server reports about itself: ``/v1/models`` (ids + llama.cpp ``meta``) and, when
    the server has it, llama.cpp's ``/props`` (model file, context size, slots, build). Raises ``ConfigError`` when
    ``/models`` does not answer. ``client`` is an ``httpx.Client`` (tests pass one with a mock transport)."""
    import httpx

    owned = client is None
    http = client or httpx.Client(timeout=10.0)
    try:
        try:
            response = http.get(f"{base_url.rstrip('/')}/models")
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ConfigError(f"the LLM server at {base_url} does not answer ({type(exc).__name__})") from exc
        models = []
        for entry in payload.get("data", []):
            if isinstance(entry, dict):
                meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
                scalar_meta = {k: v for k, v in meta.items() if isinstance(v, int | float | str)}
                models.append({"id": str(entry.get("id")), "meta": scalar_meta})
        props: dict[str, Any] | None = None
        try:
            response = http.get(_props_url(base_url))
            if response.status_code == 200:
                raw = response.json()
                generation = raw.get("default_generation_settings") or {}
                props = {key: raw.get(key, generation.get(key)) for key in _PROPS_KEYS}
                model_path = raw.get("model_path") or generation.get("model")
                props["model_file"] = PurePosixPath(str(model_path).replace("\\", "/")).name if model_path else None
        except (httpx.HTTPError, ValueError, AttributeError):
            props = None
        return {"served_models": models, "server_props": props}
    finally:
        if owned:
            http.close()


def prepare_config(config: EvalConfig, *, skip_llm_check: bool) -> dict[str, Any]:
    """Check that ``config`` can run (key present, LLM server up); returns facts to record about it."""
    facts: dict[str, Any] = {}
    if config.required_secret:
        value = os.environ.get(config.required_secret) or _secret_from_dotenv(config.required_secret)
        if not value:
            raise ConfigError(f"config {config.name} needs {config.required_secret}; refusing to run without it")
        os.environ[config.required_secret] = value
        facts["secret_source"] = config.required_secret
    if config.llm_server and not skip_llm_check:
        facts.update(check_llm_server(config.llm_server))
    return facts


def reset_app_caches() -> None:
    """Drop every application cache that holds per-environment state: settings, database engine, queue, storage
    instances and embedders (the next use rebuilds them from the current environment). The OCR engine is kept: it
    holds only model weights, is process-wide and is not tied to a database or directory."""
    from app.config import reset_settings_cache
    from app.db.session import reset_engine
    from app.queue import reset_queue
    from app.search import embeddings
    from app.storage import _storage_at

    reset_queue()
    reset_engine()
    reset_settings_cache()
    _storage_at.cache_clear()
    embeddings._build_embedder.cache_clear()


@contextmanager
def preserved_environment() -> Generator[None, None, None]:
    """Restore ``os.environ`` and reset the application caches when the block ends (even on error)."""
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
        reset_app_caches()


def fresh_environment(config: EvalConfig, *, workdir: Path, embedding: str, ocr_workers: int) -> dict[str, str]:
    """Point the application at a new database + storage under ``workdir`` with ``config``'s settings, with every
    stateful cache rebuilt from scratch. Call inside ``preserved_environment``."""
    workdir.mkdir(parents=True, exist_ok=True)
    env = {
        "DOCINTEL_IGNORE_DOTENV": "1",
        "APP_ENV": "development",
        "QUEUE_BACKEND": "inline",
        "DATABASE_URL": f"sqlite:///{(workdir / 'eval.db').as_posix()}",
        "STORAGE_DIR": str(workdir / "storage"),
        "EMBEDDING_PROVIDER": embedding,
        "OCR_WORKERS": str(ocr_workers),
        "RATE_LIMIT_ENABLED": "false",
        "LOG_JSON": "false",
        **config.env,
    }
    os.environ.update(env)
    reset_app_caches()
    from app.db.session import init_db

    init_db()
    return env


def settings_record(workdir: Path | None = None) -> dict[str, Any]:
    """Every effective setting of the current environment, JSON-ready: secret values replaced by ``<redacted>``
    (``None`` stays ``None``), credentials removed from URLs, the run's temporary directory shown as ``<workdir>``."""
    from pydantic import SecretStr

    from app.config import get_settings

    settings = get_settings()
    dump = settings.model_dump(mode="json")
    for name in type(settings).model_fields:
        value = getattr(settings, name)
        if isinstance(value, SecretStr):
            dump[name] = REDACTED
        elif name == "api_keys":
            dump[name] = REDACTED if value else ""
        elif isinstance(dump.get(name), str) and "://" in dump[name]:
            dump[name] = _URL_CREDENTIALS.sub(f"{REDACTED}@", dump[name])
    if workdir is not None:
        roots = {workdir.as_posix(), str(workdir)}
        for name, value in dump.items():
            if isinstance(value, str):
                for root in roots:
                    value = value.replace(root, WORKDIR_MARK)
                dump[name] = value
    return dump


# --------------------------------------------------------------------------------------------------------------------
# machine, code and dataset identity
# --------------------------------------------------------------------------------------------------------------------
def _cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _gpu_names() -> list[str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def machine_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_name(),
        "logical_cpus": os.cpu_count(),
        "gpu": _gpu_names(),
    }
    try:
        import psutil

        info["physical_cpus"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        info["physical_cpus"] = info["ram_gb"] = None
    return info


def _git_dir(repo: Path) -> Path | None:
    dot_git = repo / ".git"
    if dot_git.is_dir():
        return dot_git
    if dot_git.is_file():  # a worktree: "gitdir: <path>"
        text = dot_git.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            return (repo / text.split(":", 1)[1].strip()).resolve()
    return None


def git_identity(repo: Path = REPO_DIR) -> dict[str, str | None]:
    """HEAD commit and branch, read from ``.git`` directly (see ``git_dirty`` for uncommitted changes)."""
    git_dir = _git_dir(repo)
    if git_dir is None:
        return {"commit": None, "branch": None}
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref:"):
        return {"commit": head, "branch": None}
    ref = head.split(":", 1)[1].strip()
    common = git_dir
    if (git_dir / "commondir").is_file():
        common = (git_dir / (git_dir / "commondir").read_text(encoding="utf-8").strip()).resolve()
    commit: str | None = None
    for base in dict.fromkeys((git_dir, common)):
        ref_file = base / ref
        if ref_file.is_file():
            commit = ref_file.read_text(encoding="utf-8").strip()
            break
        packed = base / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    commit = parts[0]
                    break
        if commit:
            break
    return {"commit": commit, "branch": ref.removeprefix("refs/heads/")}


def git_dirty(
    repo: Path = REPO_DIR,
    *,
    git: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Whether the working tree had uncommitted changes (``git status --porcelain`` non-empty) and how many paths.

    Runs with ``--no-optional-locks`` so it never takes the index lock (safe next to other git users). ``None``
    values mean unknown (no git, not a repository, or git failed)."""
    exe = git or shutil.which("git")
    if not exe:
        return {"dirty": None, "dirty_paths": None}
    try:
        out = runner(
            [exe, "--no-optional-locks", "status", "--porcelain"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"dirty": None, "dirty_paths": None}
    paths = [line for line in out.splitlines() if line.strip()]
    return {"dirty": bool(paths), "dirty_paths": len(paths)}


def dataset_summary(labels: Sequence[LabeledFile]) -> dict[str, Any]:
    """Composition of the labeled dataset (what the report's methodology may say about it)."""
    main = [lab for lab in labels if not lab.file.startswith(SCALE_PREFIX)]
    codes = Counter(str(code.value) for lab in main for issues in lab.record_issues for code in issues)
    groups = Counter(lab.group.split("-", 1)[0] for lab in main if lab.group)
    named = {lab.template for lab in main if lab.template}
    templates = sorted(t for t in named if re.fullmatch(r"T\d+", t))
    return {
        "files": len(main),
        "records": sum(len(lab.records) for lab in main),
        "by_kind": dict(sorted(Counter(lab.kind for lab in main).items())),
        "by_quality": dict(sorted(Counter(lab.quality for lab in main).items())),
        "by_split": dict(sorted(Counter(lab.split for lab in main).items())),
        "templates": templates,
        "other_templates": sorted(named - set(templates)),
        "split_templates": {
            split: sorted({lab.template for lab in main if lab.split == split and lab.template})
            for split in sorted({lab.split for lab in main})
        },
        "expect_failure": sum(lab.expect == "failure" for lab in main),
        "duplicate_pairs": len({lab.group for lab in main if lab.group and lab.group.startswith("dup-")}),
        "cross_verification_files": groups.get("xv", 0),
        "expected_codes": dict(sorted(codes.items())),
        "scale_files": [
            {"file": lab.file, "pages": lab.pages, "records": len(lab.records)}
            for lab in labels
            if lab.file.startswith(SCALE_PREFIX)
        ],
    }


# --------------------------------------------------------------------------------------------------------------------
# processing one file
# --------------------------------------------------------------------------------------------------------------------
@dataclass(slots=True)
class Submission:
    """A dataset file after upload + processing, before its results are read back."""

    label: LabeledFile
    document_id: UUID | None = None
    status: str | None = None  # set for files that never became documents (rejected) or crashed the harness
    error_code: str | None = None
    error_message: str | None = None
    seconds: float | None = None
    harness_error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def submit_file(label: LabeledFile, dataset: Path) -> Submission:
    """Upload ``label``'s file the way the API does and process it on the inline queue (timed)."""
    from app.config import get_settings
    from app.db.models import Document
    from app.db.session import SessionLocal
    from app.ingestion.sniff import HEAD_BYTES, detect_file_kind, detect_mime_type
    from app.ingestion.types import IngestionError
    from app.pipeline.intake import create_document
    from app.queue import get_queue
    from app.storage import get_storage, suffix_for

    sub = Submission(label=label)
    filename = PurePosixPath(label.file).name
    try:
        data = (dataset / label.file).read_bytes()
        head = data[:HEAD_BYTES]
        try:
            kind = detect_file_kind(head, filename)
            content_type = detect_mime_type(head, kind)
            stored = get_storage().save(
                io.BytesIO(data), suffix=suffix_for(kind, content_type), max_bytes=get_settings().max_upload_bytes
            )
        except IngestionError as exc:  # what the upload endpoint rejects (empty, unsupported, too large)
            sub.status, sub.error_code, sub.error_message = "rejected", exc.code, exc.message
            return sub
        with SessionLocal() as session:
            intake = create_document(session, filename=filename, stored=stored, kind=kind, content_type=content_type)
            session.commit()
            sub.document_id, job_id = intake.document.id, intake.job.id
        started = time.perf_counter()
        try:
            get_queue().enqueue(job_id)
        finally:
            sub.seconds = round(time.perf_counter() - started, 4)
        with SessionLocal() as session:
            document = session.get(Document, sub.document_id)
            sub.extra["status_now"] = document.status if document is not None else None
    except Exception as exc:  # the run survives anything one file does; the file counts as crashed
        sub.status = "crashed"
        sub.harness_error = f"{type(exc).__name__}: {exc}"[:300]
        logger.error("%s: harness error %s", label.file, type(exc).__name__)
    return sub


def _stage_seconds(events: Sequence[Any]) -> tuple[dict[str, float], int]:
    """Per-stage seconds of the last attempt (``duration_ms`` of the stage events) and the LLM fallback count."""
    stages: dict[str, float] = {}
    fallbacks = 0
    for event in events:
        data = event.data or {}
        if event.stage in TIMED_STAGES and "duration_ms" in data:
            stages[event.stage] = round(float(data["duration_ms"]) / 1000, 4)
        if event.stage == "extracting":
            fallbacks = int(data.get("llm_fallbacks") or 0)
    return stages, fallbacks


def read_back(sub: Submission) -> FileResult:
    """The file's final state from the database (call after every file of the run was processed)."""
    from sqlalchemy import select

    from app.db.models import Document, Invoice, ProcessingEvent, ProcessingJob, ValidationIssueRow
    from app.db.session import SessionLocal

    label = sub.label
    if sub.document_id is None:
        return FileResult(
            file=label.file,
            status=sub.status or "crashed",
            error_code=sub.error_code,
            error_message=sub.error_message,
            harness_error=sub.harness_error,
        )
    with SessionLocal() as session:
        document = session.get(Document, sub.document_id)
        if document is None:
            return FileResult(file=label.file, status="crashed", seconds=sub.seconds, harness_error="document missing")
        invoices = session.scalars(
            select(Invoice).where(Invoice.document_id == document.id).order_by(Invoice.record_index)
        ).all()
        issues = session.scalars(select(ValidationIssueRow).where(ValidationIssueRow.document_id == document.id)).all()
        job = session.scalars(
            select(ProcessingJob).where(ProcessingJob.document_id == document.id).order_by(ProcessingJob.created_at)
        ).first()
        events = session.scalars(
            select(ProcessingEvent).where(ProcessingEvent.document_id == document.id).order_by(ProcessingEvent.id)
        ).all()
        by_invoice: dict[UUID, list[str]] = {}
        for issue in issues:
            if issue.invoice_id is not None:
                by_invoice.setdefault(issue.invoice_id, []).append(issue.code)
        records = [
            PredictedRecord(
                data=invoice.to_invoice_data(),
                source_kind=invoice.source_kind,
                page_start=invoice.page_start,
                page_end=invoice.page_end,
                sheet_name=invoice.sheet_name,
                row_number=invoice.row_number,
                issue_codes=sorted(set(by_invoice.get(invoice.id, []))),
                review_status=invoice.review_status,
                extraction_method=invoice.extraction_method,
                overall_confidence=invoice.overall_confidence,
            )
            for invoice in invoices
        ]
        stages, fallbacks = _stage_seconds(events)
        return FileResult(
            file=label.file,
            status="crashed" if sub.status == "crashed" else document.status,
            error_code=document.error_code,
            error_message=document.error_message,
            seconds=sub.seconds,
            processing_ms=document.processing_ms,
            page_count=document.page_count,
            ocr_page_count=document.ocr_page_count,
            extraction_method=document.extraction_method,
            llm_provider=document.llm_provider,
            llm_model=document.llm_model,
            llm_fallbacks=fallbacks,
            records=records,
            document_issue_codes=sorted({i.code for i in issues if i.invoice_id is None}),
            warnings=list(document.warnings or []),
            attempts=job.attempt if job is not None else 0,
            harness_error=sub.harness_error,
            stage_seconds=stages,
        )


def warm_up() -> float:
    """Import the pipeline and load the OCR engine and the embedding model once, so the first file is not charged
    for one-time start-up (a long-running worker pays it once, not per document)."""
    started = time.perf_counter()
    from PIL import Image

    from app.ingestion.ocr import get_ocr_engine, ocr_image
    from app.queue import get_queue
    from app.search.embeddings import get_embedder

    get_queue()  # imports the processor, parsers and extractors
    ocr_image(Image.new("RGB", (320, 80), "white"), 1, engine=get_ocr_engine())
    get_embedder().embed_documents(["warm-up"])
    return round(time.perf_counter() - started, 3)


def run_files(labels: Sequence[LabeledFile], dataset: Path) -> list[FileResult]:
    """Process ``labels`` in order in the current environment, then read every result back."""
    submissions: list[Submission] = []
    total = len(labels)
    for index, label in enumerate(labels, start=1):
        sub = submit_file(label, dataset)
        submissions.append(sub)
        status = sub.status or sub.extra.get("status_now")
        seconds = f"{sub.seconds:.2f} s" if sub.seconds is not None else "-"
        logger.info("[%d/%d] %s -> %s (%s)", index, total, label.file, status, seconds)
    return [_safe_read_back(sub) for sub in submissions]


def _safe_read_back(sub: Submission) -> FileResult:
    try:
        return read_back(sub)
    except Exception as exc:  # one unreadable record must not lose the whole run
        logger.error("%s: read-back failed (%s)", sub.label.file, type(exc).__name__)
        return FileResult(
            file=sub.label.file,
            status="crashed",
            seconds=sub.seconds,
            harness_error=f"read-back {type(exc).__name__}: {exc}"[:300],
        )


def _with_scores(labels: Sequence[LabeledFile], results: Sequence[FileResult]) -> list[dict[str, Any]]:
    by_file = {lab.file: lab for lab in labels}
    out = []
    for res in results:
        label = by_file[res.file]
        entry = res.model_dump(mode="json")
        entry.update(kind=label.kind, quality=label.quality, split=label.split, expect=label.expect)
        entry["score"] = score_file(label, res).summary()
        out.append(entry)
    return out


def run_scale_suite(
    config: EvalConfig, labels: Sequence[LabeledFile], dataset: Path, root: Path, *, embedding: str, ocr_workers: int
) -> dict[str, Any]:
    """Each scale file alone in its own fresh database + storage (caches reset, warm-up repeated): throughput per
    file and stage, plus record-level accuracy on the scale files' labels."""
    from app.ingestion.ocr import shutdown_ocr_pool

    results: list[FileResult] = []
    files: list[dict[str, Any]] = []
    for index, label in enumerate(labels, start=1):
        override = SCALE_FILES.get(label.file)
        workers = ocr_workers if override is None else override
        shutdown_ocr_pool()  # no OCR worker processes carried over from the previous file
        workdir = root / f"scale-{index}"
        fresh_environment(config, workdir=workdir, embedding=embedding, ocr_workers=workers)
        warmup_s = warm_up()
        (res,) = run_files([label], dataset)
        shutdown_ocr_pool()
        results.append(res)
        pages = res.page_count or label.pages
        files.append(
            {
                "file": res.file,
                "status": res.status,
                "database": workdir.name,
                "warmup_seconds": warmup_s,
                "pages": pages,
                "ocr_pages": res.ocr_page_count,
                "ocr_workers": workers,
                "seconds": res.seconds,
                "processing_ms": res.processing_ms,
                "pages_per_s": round(pages / res.seconds, 3) if res.seconds else None,
                "records_truth": len(label.records),
                "records_predicted": len(res.records),
                "document_issue_codes": res.document_issue_codes,
                "stages": stage_rates(res.stage_seconds, pages),
            }
        )
    return {
        "files": files,
        "metrics": compute_metrics(labels, results),
        "results": _with_scores(labels, results),
    }


# --------------------------------------------------------------------------------------------------------------------
# the command
# --------------------------------------------------------------------------------------------------------------------
def select_labels(labels: Sequence[LabeledFile], limit: int | None) -> tuple[list[LabeledFile], list[LabeledFile]]:
    """(main files in label order, at most ``limit``; scale files)."""
    main = [lab for lab in labels if not lab.file.startswith(SCALE_PREFIX)]
    scale = [lab for lab in labels if lab.file in SCALE_FILES]
    if limit is not None:
        main = main[:limit]
    return main, scale


def _config_record(config: EvalConfig, env: dict[str, str], facts: dict[str, Any], workdir: Path) -> dict[str, Any]:
    from app.config import get_settings

    settings = get_settings()
    llm_model = None
    if settings.extraction_mode != "rules":
        llm_model = {
            "local": settings.llm_model,
            "gemini": settings.gemini_model,
            "anthropic": settings.anthropic_model,
        }.get(settings.llm_provider)
    hidden = {"DATABASE_URL", "STORAGE_DIR"}
    return {
        "name": config.name,
        "description": config.description,
        "extraction_mode": settings.extraction_mode,
        "llm_provider": settings.llm_provider,
        "llm_model": llm_model,
        "llm_base_url": settings.llm_base_url if settings.llm_provider == "local" else None,
        "llm_max_input_chars": settings.llm_max_input_chars,
        "llm_concurrency": settings.llm_concurrency,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model if settings.embedding_provider == "fastembed" else None,
        "ocr_engine": settings.ocr_engine,
        "ocr_workers": settings.ocr_workers,
        "ocr_dpi": settings.ocr_dpi,
        "amount_tolerance": str(settings.amount_tolerance),
        "env": {k: v for k, v in env.items() if k not in hidden},
        "settings": settings_record(workdir),
        **facts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.run_eval",
        description="Run the document pipeline over the labeled evaluation dataset and write a results JSON.",
    )
    parser.add_argument("--dataset", type=Path, default=Path("eval/dataset"), help="dataset root with labels.json")
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS), help="extractor configuration")
    parser.add_argument("--out", type=Path, required=True, help="results JSON to write")
    parser.add_argument("--limit", type=int, default=None, help="only the first N (non-scale) files, label order")
    parser.add_argument("--include-scale", action="store_true", help="also run the scale suite (1000 + 100 pages)")
    parser.add_argument("--scale-only", action="store_true", help="run only the scale suite")
    parser.add_argument("--embedding", choices=("fastembed", "hashing"), default="fastembed")
    parser.add_argument("--ocr-workers", type=int, default=0, help="OCR_WORKERS for the main set (default 0)")
    parser.add_argument("--workdir", type=Path, default=None, help="keep the databases/storage here (default: temp)")
    parser.add_argument("--skip-llm-check", action="store_true", help="do not require the local LLM server to answer")
    return parser


def run(args: argparse.Namespace, argv: Sequence[str]) -> dict[str, Any]:
    """Run one evaluation and return the results document (also written to ``args.out``); ``argv`` is recorded.

    ``os.environ`` and the application caches are restored afterwards."""
    with preserved_environment():
        return _run(args, argv)


def _run(args: argparse.Namespace, argv: Sequence[str]) -> dict[str, Any]:
    config = CONFIGS[args.config]
    facts = prepare_config(config, skip_llm_check=args.skip_llm_check)
    dataset = args.dataset.resolve()
    labels = load_labels(dataset / "labels.json")
    seed = json.loads((dataset / "labels.json").read_text(encoding="utf-8")).get("seed")
    main_labels, scale_labels = select_labels(labels, args.limit)
    if args.scale_only:
        main_labels = []
    run_scale = args.include_scale or args.scale_only
    if run_scale and len(scale_labels) != len(SCALE_FILES):
        raise ConfigError("the dataset has no scale files; build it with `python -m eval.synth.build --scale`")
    if args.workdir is not None and args.workdir.exists() and any(args.workdir.iterdir()):
        raise ConfigError(f"--workdir {args.workdir} is not empty; every run needs a fresh database")

    identity = {**git_identity(), **git_dirty()}
    started_at = datetime.now(UTC)
    wall_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="docintel-eval-", ignore_cleanup_errors=True) as tmp:
        root = args.workdir.resolve() if args.workdir else Path(tmp)
        env = fresh_environment(config, workdir=root / "main", embedding=args.embedding, ocr_workers=args.ocr_workers)
        config_record = _config_record(config, env, facts, root)
        warmup_s = warm_up()
        logger.info(
            "config %s: %d files (+%d scale), warm-up %.1f s",
            config.name,
            len(main_labels),
            len(scale_labels) if run_scale else 0,
            warmup_s,
        )
        results = run_files(main_labels, dataset)
        scale = None
        if run_scale:
            scale = run_scale_suite(
                config, scale_labels, dataset, root, embedding=args.embedding, ocr_workers=args.ocr_workers
            )
        from app.db.session import reset_engine
        from app.ingestion.ocr import shutdown_ocr_pool

        shutdown_ocr_pool()
        reset_engine()  # release the SQLite file before the temporary directory is removed
    wall_seconds = round(time.perf_counter() - wall_start, 2)

    document: dict[str, Any] = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "config": config_record,
        "run": {
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "wall_seconds": wall_seconds,
            "warmup_seconds": warmup_s,
            "processing_seconds": round(sum(r.seconds or 0 for r in results), 2),
            **identity,
            "dataset": dataset.relative_to(BACKEND_DIR).as_posix()
            if dataset.is_relative_to(BACKEND_DIR)
            else dataset.name,
            "labels_seed": seed,
            "labels_files": len([lab for lab in labels if not lab.file.startswith(SCALE_PREFIX)]),
            "dataset_summary": dataset_summary(labels),
            "files_evaluated": len(main_labels),
            "limit": args.limit,
            "include_scale": run_scale,
            "scale_only": bool(args.scale_only),
            "command": " ".join(["python -m eval.run_eval", *argv]),
        },
        "machine": machine_info(),
        "metrics": compute_metrics(main_labels, results) if main_labels else None,
        "files": _with_scores(main_labels, results),
        "scale": scale,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return document


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def run_log_path(out: Path) -> Path:
    """Where a run's log goes: ``logs/<results stem>.log`` next to its results file (one log per run)."""
    return out.parent / "logs" / f"{out.stem}.log"


def _configure_logging() -> None:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format=LOG_FORMAT)
    logger.setLevel(logging.INFO)
    warnings.filterwarnings("ignore", message=".*xlrd engine is deprecated.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="xlrd.*")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one evaluation. Everything it logs also goes to ``run_log_path(--out)``; the log is written aside and moved
    into place only when the run succeeds, so a refused or failed run never replaces an earlier run's log."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments)
    _configure_logging()
    log_path = run_log_path(args.out)
    partial = log_path.with_name(log_path.name + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(partial, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root = logging.getLogger()
    root.addHandler(handler)
    status = EXIT_CONFIG_ERROR
    try:
        status = _run_and_report(args, arguments)
        return status
    finally:
        root.removeHandler(handler)
        handler.close()
        if status == 0:
            partial.replace(log_path)
        else:
            partial.unlink(missing_ok=True)


def _run_and_report(args: argparse.Namespace, arguments: list[str]) -> int:
    try:
        document = run(args, arguments)
    except ConfigError as exc:
        logger.error("%s", exc)
        return EXIT_CONFIG_ERROR
    overall = (document["metrics"] or {}).get("overall")
    if overall:
        logger.info(
            "done: %d files, field accuracy %s, success rate %s, %.0f s -> %s",
            overall["files"],
            overall["fields"]["overall"]["accuracy"],
            overall["outcomes"]["success_rate"],
            document["run"]["wall_seconds"],
            args.out,
        )
    else:
        logger.info("done: scale suite, %.0f s -> %s", document["run"]["wall_seconds"], args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
