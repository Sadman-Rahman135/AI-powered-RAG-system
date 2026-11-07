"""Final review fixes in the pipeline: the attempt deadline (I2), attempt-owned progress writes (minor 2), the
compare-and-set sweep (minor 1), one cached LLM provider per configuration (minor 3) and the per-document count of
records that fell back from the LLM to the rules (minor 11)."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.config import reset_settings_cache
from app.db.models import Document, Invoice, JobStatus, ProcessingEvent, ProcessingJob, ValidationIssueRow
from app.db.session import SessionLocal, session_scope
from app.domain.schemas import ExtractionWarning
from app.pipeline import processor, progress
from app.pipeline.processor import process_job, reclaim_stale_jobs, sweep_jobs
from app.pipeline.progress import AttemptDeadline, Heartbeat, ProgressReporter
from tests.unit.pipeline.builders import canned_record, invoice_pages, upload, write_pdf
from tests.unit.pipeline.conftest import CannedExtraction


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    return write_pdf(tmp_path / "invoice.pdf", invoice_pages("INV-1001"))


@pytest.fixture(autouse=True)
def _no_cached_provider() -> Iterator[None]:
    processor.close_llm_provider()
    yield
    processor.close_llm_provider()


def _job(session: Session, job_id: UUID) -> ProcessingJob:
    job = session.get(ProcessingJob, job_id)
    assert job is not None
    return job


def _set_job(job_id: UUID, **values: Any) -> None:
    with session_scope() as session:
        job = _job(session, job_id)
        for name, value in values.items():
            setattr(job, name, value)


def _snapshot(job_id: UUID) -> tuple[str, int, str | None, datetime | None]:
    with SessionLocal() as session:
        job = _job(session, job_id)
        return job.status, job.attempt, job.stage, progress.as_utc(job.heartbeat_at)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


# ----------------------------------------------------------------------------------- deadline (I2)


def test_attempt_deadline() -> None:
    clock = _Clock()
    deadline = AttemptDeadline.after(10, clock=clock)
    assert not deadline.passed()
    clock.now = 9.9
    assert not deadline.passed()
    clock.now = 10.0
    assert deadline.passed()
    assert not AttemptDeadline.after(0).passed() and not AttemptDeadline.after(None).passed()  # no limit


def test_heartbeat_stops_beating_at_the_deadline(pdf: Path, caplog: pytest.LogCaptureFixture) -> None:
    _, job_id = upload(pdf)
    _set_job(job_id, status=JobStatus.RUNNING, attempt=1, heartbeat_at=datetime.now(UTC) - timedelta(hours=1))
    clock = _Clock()
    deadline = AttemptDeadline.after(100, clock=clock)
    with Heartbeat(job_id, 1, SessionLocal, interval_s=0.02, deadline=deadline) as heartbeat:
        time.sleep(0.15)
        beats_before = heartbeat.beats
        clock.now = 100  # the attempt has now run for JOB_MAX_RUNTIME_S
        time.sleep(0.15)
        assert heartbeat.expired
        last = _snapshot(job_id)[3]
        time.sleep(0.15)
        assert _snapshot(job_id)[3] == last  # nothing refreshes the job any more
    assert beats_before > 0 and heartbeat.beats >= beats_before
    assert "exceeded JOB_MAX_RUNTIME_S" in caplog.text


def test_progress_after_the_deadline_leaves_the_heartbeat_alone(pdf: Path) -> None:
    _, job_id = upload(pdf)
    old = datetime.now(UTC) - timedelta(hours=1)
    _set_job(job_id, status=JobStatus.RUNNING, attempt=1, heartbeat_at=old)
    clock = _Clock()
    reporter = ProgressReporter(job_id, 1, SessionLocal, deadline=AttemptDeadline.after(5, clock=clock))
    reporter.set_stage("parsing", current=1, total=9)
    assert _snapshot(job_id)[3] != old  # before the deadline progress also proves the attempt alive
    _set_job(job_id, heartbeat_at=old)
    clock.now = 5
    reporter.set_stage("extracting", current=2, total=9)
    status, attempt, stage, beat = _snapshot(job_id)
    assert (status, attempt, stage, beat) == ("running", 1, "extracting", old)  # progress yes, heartbeat no


# -------------------------------------------------------------------------- progress ownership (minor 2)


def test_progress_of_a_superseded_attempt_touches_nothing(pdf: Path) -> None:
    _, job_id = upload(pdf)
    old = datetime.now(UTC) - timedelta(hours=1)
    _set_job(job_id, status=JobStatus.RUNNING, attempt=2, stage="extracting", heartbeat_at=old)
    stale_reporter = ProgressReporter(job_id, 1, SessionLocal)  # attempt 1 was taken over by attempt 2

    stale_reporter.set_stage("indexing")
    stale_reporter.on_progress(5, 10, "parsing")

    assert _snapshot(job_id) == ("running", 2, "extracting", old)
    ProgressReporter(job_id, 2, SessionLocal).set_stage("validating")
    assert _snapshot(job_id)[2] == "validating"  # the owner still writes


# ------------------------------------------------------------- a hung attempt is reclaimed (I2, end to end)


@dataclass
class _Hang:
    """``extract_records`` that blocks the first call (a hung attempt) and answers every later one."""

    records: list[Any]
    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    calls: int = 0

    def __call__(self, doc: Any, **kwargs: Any) -> list[Any]:
        self.calls += 1
        if self.calls == 1:
            self.entered.set()
            assert self.release.wait(30)
        return list(self.records)


def test_hung_attempt_is_reclaimed_after_the_deadline_and_cannot_overwrite_the_takeover(
    pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JOB_MAX_RUNTIME_S", "1")
    reset_settings_cache()
    monkeypatch.setattr(progress, "HEARTBEAT_INTERVAL_S", 0.25)
    hang = _Hang([canned_record("INV-1001")])
    monkeypatch.setattr(processor, "extract_records", hang)
    document_id, job_id = upload(pdf)

    hung = threading.Thread(target=process_job, args=(job_id,), daemon=True)
    hung.start()
    assert hang.entered.wait(10)
    time.sleep(1.6)  # past the 1 s deadline (+ one 0.25 s beat): the heartbeat has stopped
    first = _snapshot(job_id)
    time.sleep(0.3)
    assert _snapshot(job_id) == first and first[:2] == ("running", 1)

    with session_scope() as session:  # the periodic sweep, run as if JOB_STALE_AFTER_S had gone by
        assert first[3] is not None
        assert sweep_jobs(session, now=first[3] + timedelta(seconds=301)) == [job_id]
    with SessionLocal() as session:
        job = _job(session, job_id)
        assert job.status == "queued" and (job.error or "").startswith("job_timeout:")

    process_job(job_id)  # the re-sent message: attempt 2 runs to the end while attempt 1 is still stuck
    assert _snapshot(job_id)[:2] == ("succeeded", 2)
    hang.release.set()  # attempt 1 comes back to life ...
    hung.join(10)
    assert not hung.is_alive()

    with SessionLocal() as session:  # ... and stops at its next stage boundary without touching anything
        assert _snapshot(job_id)[:3] == ("succeeded", 2, "done")
        assert session.get(Document, document_id).status == "completed"  # type: ignore[union-attr]
        assert len(session.scalars(select(Invoice).where(Invoice.document_id == document_id)).all()) == 1
        messages = [e.message for e in session.scalars(select(ProcessingEvent).order_by(ProcessingEvent.id))]
        assert "Attempt timed out during attempt 1 of 3; job re-queued" in messages
        assert sum(m.startswith("Extracted") for m in messages) == 1  # only attempt 2 recorded its extraction


def test_last_attempt_that_timed_out_fails_with_job_timeout(pdf: Path) -> None:
    document_id, job_id = upload(pdf)
    started = datetime.now(UTC) - timedelta(hours=5)  # longer than the 4 h JOB_MAX_RUNTIME_S default
    _set_job(job_id, status=JobStatus.RUNNING, attempt=3, max_attempts=3, started_at=started,
             heartbeat_at=started + timedelta(hours=4))  # fmt: skip
    with session_scope() as session:
        assert reclaim_stale_jobs(session).failed == [job_id]
    with SessionLocal() as session:
        document = session.get(Document, document_id)
        assert document is not None and (document.status, document.error_code) == ("failed", "job_timeout")
        assert "JOB_MAX_RUNTIME_S" in (document.error_message or "")


def test_redelivery_of_a_timed_out_last_attempt_fails_it_with_job_timeout(pdf: Path, canned: CannedExtraction) -> None:
    calls = canned([canned_record()])
    document_id, job_id = upload(pdf)
    started = datetime.now(UTC) - timedelta(hours=5)
    _set_job(job_id, status=JobStatus.RUNNING, attempt=3, max_attempts=3, started_at=started,
             heartbeat_at=started + timedelta(hours=4))  # fmt: skip
    process_job(job_id)
    assert calls == []
    with SessionLocal() as session:
        assert session.get(Document, document_id).error_code == "job_timeout"  # type: ignore[union-attr]


# ------------------------------------------------------------------------------ sweep CAS (minor 1)


@pytest.mark.parametrize(("race", "attempt"), [("takeover", 1), ("heartbeat", 1), ("heartbeat", 3)])
def test_sweep_never_undoes_a_change_made_after_it_read_the_job(
    pdf: Path, monkeypatch: pytest.MonkeyPatch, race: str, attempt: int
) -> None:
    document_id, job_id = upload(pdf)
    old = datetime.now(UTC) - timedelta(hours=1)
    _set_job(job_id, status=JobStatus.RUNNING, attempt=attempt, max_attempts=3, started_at=old, heartbeat_at=old)
    real_timed_out = processor._timed_out

    with session_scope() as session:

        def racing(job: ProcessingJob, settings: Any) -> bool:
            # between the sweep's read and its write, another worker takes the job over / the attempt beats
            values: dict[str, Any] = {"heartbeat_at": datetime.now(UTC)}
            if race == "takeover":
                values["attempt"] = ProcessingJob.attempt + 1
            session.execute(
                update(ProcessingJob)
                .where(ProcessingJob.id == job_id)
                .values(**values)
                .execution_options(synchronize_session=False)
            )
            return real_timed_out(job, settings)

        monkeypatch.setattr(processor, "_timed_out", racing)
        stale = reclaim_stale_jobs(session)

    assert (stale.requeued, stale.failed) == ([], [])
    status, now_attempt, _, beat = _snapshot(job_id)
    assert status == "running" and now_attempt == attempt + (race == "takeover")
    assert beat is not None and beat > old
    with SessionLocal() as session:
        assert session.get(Document, document_id).status == "queued"  # type: ignore[union-attr]  # untouched
        assert session.scalars(select(ProcessingEvent).where(ProcessingEvent.stage == "recovered")).all() == []


# ------------------------------------------------------------------------- provider cache (minor 3)


@dataclass
class _FakeClient:
    closed: bool = False

    def close(self) -> None:
        self.closed = True


@dataclass
class _FakeProvider:
    model: str
    name: str = "local"
    _client: _FakeClient = field(default_factory=_FakeClient)


def test_jobs_share_one_llm_provider_per_configuration(
    tmp_path: Path, canned: CannedExtraction, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = canned([canned_record()])
    built: list[_FakeProvider] = []

    def factory(settings: Any) -> _FakeProvider:
        built.append(_FakeProvider(model=settings.llm_model))
        return built[-1]

    monkeypatch.setattr(processor, "get_llm_provider", factory)
    for n in range(2):
        process_job(upload(write_pdf(tmp_path / f"{n}.pdf", invoice_pages(f"INV-{n}")))[1])
    assert len(built) == 1
    assert len(calls) == 2 and all(call[1]["provider"] is built[0] for call in calls)  # one client, both jobs

    monkeypatch.setenv("LLM_MODEL", "a-bigger-model")  # a changed configuration builds a new provider
    reset_settings_cache()
    process_job(upload(write_pdf(tmp_path / "2.pdf", invoice_pages("INV-2")))[1])
    assert [p.model for p in built] == ["qwen3.5-9b", "a-bigger-model"]
    assert not built[0]._client.closed  # a replaced provider may still serve a running job

    processor.close_llm_provider()  # worker / API shutdown
    assert built[1]._client.closed


def test_rules_mode_and_configuration_errors_build_no_provider(
    pdf: Path, canned: CannedExtraction, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.extraction.llm.base import LLMConfigurationError

    attempts: list[str] = []

    def broken(settings: Any) -> None:
        attempts.append(settings.llm_provider)
        raise LLMConfigurationError("LLM_PROVIDER=local requires LLM_BASE_URL")

    calls = canned([canned_record()])
    monkeypatch.setattr(processor, "get_llm_provider", broken)
    process_job(upload(pdf)[1])
    process_job(upload(pdf, filename="again.pdf")[1])
    assert len(attempts) == 2  # not cached: a fixed configuration is picked up by the next job
    monkeypatch.setenv("EXTRACTION_MODE", "rules")
    reset_settings_cache()
    process_job(upload(pdf, filename="rules.pdf")[1])
    assert len(attempts) == 2 and [c[1]["provider"] for c in calls] == [None, None, None]


# ------------------------------------------------------------------------- fallback count (minor 11)


def test_llm_note_counts_the_records_that_fell_back(tmp_path: Path, canned: CannedExtraction) -> None:
    pdf = write_pdf(tmp_path / "three.pdf", [*invoice_pages("INV-1"), *invoice_pages("INV-2"), *invoice_pages("INV-3")])
    canned([
        canned_record("INV-1", page_start=1, page_end=1, warnings=[ExtractionWarning.LLM_FAILED.value]),
        canned_record("INV-2", page_start=2, page_end=2),
        canned_record("INV-3", page_start=3, page_end=3, warnings=[ExtractionWarning.LLM_REFUSED.value]),
    ])  # fmt: skip
    document_id, job_id = upload(pdf)
    process_job(job_id)
    with SessionLocal() as session:
        notes = session.scalars(
            select(ValidationIssueRow).where(
                ValidationIssueRow.document_id == document_id, ValidationIssueRow.code == "LLM_UNAVAILABLE"
            )
        ).all()
        assert [(n.invoice_id, n.details) for n in notes] == [
            (None, {"reasons": ["llm_failed", "llm_refused"], "records": 2})
        ]
        extracted = session.scalars(
            select(ProcessingEvent).where(
                ProcessingEvent.document_id == document_id, ProcessingEvent.stage == "extracting"
            )
        ).one()
        assert extracted.data["llm_fallbacks"] == 2


# ------------------------------------------------------------------------ indexing speed (recommendation 3)


class _RecordingEmbedder:
    name = "recording"
    dim = 2

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: Any) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


def test_precomputed_embedder_embeds_each_text_once_shortest_first_with_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(processor, "_EMBED_BATCH", 2)
    base = _RecordingEmbedder()
    seen: list[tuple[int, int, str]] = []
    texts = ["dddd", "a", "ccc", "a", "bb"]

    embedder = processor._PrecomputedEmbedder(
        base, texts, lambda current, total, stage: seen.append((current, total, stage))
    )

    assert base.batches == [["a", "bb"], ["ccc", "dddd"]]  # distinct texts, globally ordered by length
    assert seen == [(2, 4, "indexing"), (4, 4, "indexing")]
    assert embedder.embed_documents(texts) == [[4.0, 1.0], [1.0, 1.0], [3.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    assert len(base.batches) == 2  # served from memory, in the caller's order
