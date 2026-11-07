"""Recovery: no reachable failure leaves a job stuck, and attempts are always bounded.

Covers: the retry of an attempt that could not record its failure (own-attempt takeover), the background heartbeat,
the periodic sweep (stale running jobs, queued jobs whose message was lost) for Celery workers and the local pool,
re-processing over a long-queued job, ``max_attempts`` on every takeover, attempt ownership of failure records and
the stage shown while validating.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import Document, JobStatus, ProcessingJob
from app.db.session import SessionLocal, session_scope
from app.pipeline import processor, progress
from app.pipeline.intake import AlreadyProcessingError, create_reprocess_job
from app.pipeline.processor import TransientError, process_job, sweep_jobs
from app.pipeline.progress import Heartbeat
from app.queue.base import JobSweeper
from tests.unit.pipeline.builders import canned_record, invoice_pages, upload, write_pdf
from tests.unit.pipeline.conftest import CannedExtraction


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    return write_pdf(tmp_path / "invoice.pdf", invoice_pages("INV-1001"))


def _job(session: Session, job_id: UUID) -> ProcessingJob:
    job = session.get(ProcessingJob, job_id)
    assert job is not None
    return job


def _doc(session: Session, document_id: UUID) -> Document:
    document = session.get(Document, document_id)
    assert document is not None
    return document


def _state(job_id: UUID) -> tuple[str, int]:
    with SessionLocal() as session:
        job = _job(session, job_id)
        return job.status, job.attempt


def _set_job(job_id: UUID, **values: Any) -> None:
    with session_scope() as session:
        job = _job(session, job_id)
        for name, value in values.items():
            setattr(job, name, value)


class _Outage:
    """A database that goes away: while ``down``, every commit of a session from ``factory`` fails."""

    def __init__(self) -> None:
        self.down = False

    def factory(self) -> Session:
        session = SessionLocal()
        real_commit = session.commit

        def commit() -> None:
            if self.down:
                raise OperationalError("COMMIT", {}, Exception("server closed the connection"))
            real_commit()

        session.commit = commit  # type: ignore[method-assign]
        return session


# ----------------------------------------------------------------------- (1a) own-attempt takeover


def test_retry_takes_over_an_attempt_that_could_not_record_its_failure(
    pdf: Path, canned: CannedExtraction, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned([canned_record()])
    document_id, job_id = upload(pdf)
    outage = _Outage()
    real_extract = processor.extract_records

    def failing_extract(doc: Any, **kwargs: Any) -> Any:
        outage.down = True  # the database goes away mid-job, so the retry bookkeeping cannot be written either
        raise OperationalError("SELECT 1", {}, Exception("server closed the connection"))

    monkeypatch.setattr(processor, "extract_records", failing_extract)
    with pytest.raises(TransientError) as raised:
        process_job(job_id, session_factory=outage.factory)
    assert raised.value.attempt == 1
    assert _state(job_id) == ("running", 1)  # nothing could be recorded

    outage.down = False
    monkeypatch.setattr(processor, "extract_records", real_extract)
    process_job(job_id, session_factory=outage.factory)  # an unrelated redelivery: the job looks alive, so no-op
    assert _state(job_id) == ("running", 1)
    process_job(job_id, session_factory=outage.factory, attempt=raised.value.attempt)  # the queue's retry
    assert _state(job_id) == ("succeeded", 2)
    with SessionLocal() as session:
        assert _doc(session, document_id).status == "completed"


def test_inline_queue_retry_passes_the_failed_attempt(
    pdf: Path, canned: CannedExtraction, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.queue.inline import InlineQueue

    canned([canned_record()])
    _, job_id = upload(pdf)
    real_commit = Session.commit
    failures = {"left": 0}
    real_extract = processor.extract_records

    def commit(self: Session) -> None:
        if failures["left"]:
            failures["left"] -= 1
            raise OperationalError("COMMIT", {}, Exception("server closed the connection"))
        real_commit(self)

    calls = {"n": 0}

    def flaky_extract(doc: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            failures["left"] = 1  # the retry bookkeeping commit fails once
            raise OperationalError("SELECT 1", {}, Exception("server closed the connection"))
        return real_extract(doc, **kwargs)

    monkeypatch.setattr(Session, "commit", commit)
    monkeypatch.setattr(processor, "extract_records", flaky_extract)
    InlineQueue().enqueue(job_id)
    assert _state(job_id) == ("succeeded", 2)


def test_celery_retry_carries_the_failed_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.queue import tasks

    seen: list[int | None] = []

    def flaky(job_id: UUID, attempt: int | None = None) -> None:
        seen.append(attempt)
        if len(seen) == 1:
            raise TransientError("database away", code="database_error", attempt=1)

    monkeypatch.setattr(tasks, "process_job", flaky)
    result = tasks.process_document_task.apply(args=["00000000-0000-0000-0000-000000000001"])
    assert result.successful()
    assert seen == [None, 1]


# ------------------------------------------------------------------------------ (1b, ii) heartbeat


def test_heartbeat_keeps_a_long_stage_alive_and_stops_with_the_attempt(
    pdf: Path, canned: CannedExtraction, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned([canned_record()])
    monkeypatch.setattr(progress, "HEARTBEAT_INTERVAL_S", 0.05)
    _, job_id = upload(pdf)
    real_extract = processor.extract_records
    seen: list[datetime] = []

    def slow_extract(doc: Any, **kwargs: Any) -> Any:
        for _ in range(2):  # a long LLM call that reports no progress
            with SessionLocal() as session:
                beat = _job(session, job_id).heartbeat_at
                assert beat is not None
                seen.append(beat)
            time.sleep(0.4)
        return real_extract(doc, **kwargs)

    monkeypatch.setattr(processor, "extract_records", slow_extract)
    process_job(job_id)
    assert seen[1] > seen[0]  # refreshed while nothing else wrote to the job
    assert not [t for t in threading.enumerate() if t.name.startswith("docintel-heartbeat")]


def test_heartbeat_only_touches_its_own_attempt(pdf: Path) -> None:
    _, job_id = upload(pdf)
    old = datetime.now(UTC) - timedelta(hours=1)
    _set_job(job_id, status=JobStatus.RUNNING, attempt=2, heartbeat_at=old)
    with Heartbeat(job_id, 1, SessionLocal, interval_s=0.02) as heartbeat:
        time.sleep(0.2)
    assert heartbeat.beats > 0
    with SessionLocal() as session:
        beat = _job(session, job_id).heartbeat_at
        assert beat is not None and progress.as_utc(beat) == old  # attempt 2 is not attempt 1's to keep alive


def test_default_stale_after_is_five_minutes(settings: Settings) -> None:
    assert settings.job_stale_after_s == 300
    assert settings.job_stale_after_s >= progress.HEARTBEAT_INTERVAL_S * 5  # many missed beats before "stale"


# ------------------------------------------------------------------------------------ (1b, iii) sweep


def test_sweep_resends_forgotten_queued_jobs_and_reclaims_stale_ones(pdf: Path) -> None:
    now = datetime.now(UTC)
    (_, forgotten), (_, fresh), (_, stale), (_, resent) = (upload(pdf, filename=f"{n}.pdf") for n in range(4))
    _set_job(forgotten, created_at=now - timedelta(minutes=11))  # its enqueue failed / message lost
    _set_job(stale, status=JobStatus.RUNNING, attempt=1, heartbeat_at=now - timedelta(minutes=6))
    _set_job(resent, created_at=now - timedelta(hours=1), heartbeat_at=now - timedelta(minutes=2))  # sent lately

    with session_scope() as session:
        picked = sweep_jobs(session, now=now)
    assert picked == [stale, forgotten]
    with session_scope() as session:
        assert sweep_jobs(session, now=now + timedelta(minutes=1)) == []  # just sent: not again for 10 minutes
        assert _job(session, stale).status == "queued"
        assert {_job(session, j).status for j in (forgotten, fresh, resent)} == {"queued"}
    with session_scope() as session:  # ten minutes on, everything still queued is due again (oldest first)
        later = sweep_jobs(session, now=now + timedelta(minutes=11))
        assert len(later) == 4 and set(later) == {forgotten, stale, fresh, resent}


def test_job_sweeper_publishes_on_a_timer_and_survives_errors(pdf: Path) -> None:
    _, job_id = upload(pdf)
    _set_job(job_id, created_at=datetime.now(UTC) - timedelta(minutes=30))
    published: list[UUID] = []
    done = threading.Event()

    def publish(sent: UUID) -> None:
        if not published:
            published.append(sent)
            raise ConnectionError("broker down")  # logged; the job is re-sent ten minutes later
        published.append(sent)
        done.set()

    assert JobSweeper(publish).run_once() == []  # the publish failed
    _set_job(job_id, heartbeat_at=datetime.now(UTC) - timedelta(minutes=11))
    sweeper = JobSweeper(publish, interval_s=0.05)
    sweeper.start()
    try:
        assert done.wait(5)
    finally:
        sweeper.stop()
    assert published == [job_id, job_id]


def test_worker_ready_runs_a_sweep_and_starts_the_periodic_one(pdf: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.queue import celery_app as module

    _, job_id = upload(pdf)
    _set_job(job_id, created_at=datetime.now(UTC) - timedelta(minutes=15))
    sent: list[Any] = []
    monkeypatch.setattr(module.celery_app, "send_task", lambda name, args=None, queue=None, **kw: sent.append(args))
    try:
        module._on_worker_ready()
        assert sent == [[str(job_id)]]
        assert module._sweeper is not None
    finally:
        module._on_worker_shutdown()
    assert module._sweeper is None


def test_local_queue_sweep_picks_up_a_forgotten_job(
    pdf: Path, canned: CannedExtraction, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.queue.local import LocalQueue

    canned([canned_record()])
    _, job_id = upload(pdf)
    _set_job(job_id, created_at=datetime.now(UTC) - timedelta(minutes=20))  # never enqueued (API crashed)
    queue = LocalQueue(1, sweep_interval_s=0.05)
    try:
        deadline = time.monotonic() + 10
        while _state(job_id)[0] != "succeeded" and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        queue.shutdown(wait=True)
    assert _state(job_id) == ("succeeded", 1)


# ------------------------------------------------------------------------------ (1b, iv) reprocess


def test_reprocess_replaces_a_job_that_waited_too_long_in_the_queue(pdf: Path) -> None:
    document_id, job_id = upload(pdf)
    with session_scope() as session, pytest.raises(AlreadyProcessingError):
        create_reprocess_job(session, _doc(session, document_id))  # just queued: give it a chance
    _set_job(job_id, created_at=datetime.now(UTC) - timedelta(minutes=11))
    with session_scope() as session:
        new_job = create_reprocess_job(session, _doc(session, document_id))
        assert new_job.status == "queued"
    with SessionLocal() as session:
        old = _job(session, job_id)
        assert old.status == "failed" and (old.error or "").startswith("superseded")


# ------------------------------------------------------------------------------ (2) attempts bounded


def test_stale_takeover_without_attempts_left_fails_worker_lost(pdf: Path, canned: CannedExtraction) -> None:
    calls = canned([canned_record()])
    document_id, job_id = upload(pdf)
    old = datetime.now(UTC) - timedelta(hours=7)
    _set_job(job_id, status=JobStatus.RUNNING, attempt=3, max_attempts=3, heartbeat_at=old, started_at=old)
    process_job(job_id)  # the broker redelivers the unacked message of a worker that crashed on attempt 3
    assert calls == []  # not run a fourth time
    with SessionLocal() as session:
        job, document = _job(session, job_id), _doc(session, document_id)
        assert (job.status, job.attempt) == ("failed", 3)
        assert (document.status, document.error_code) == ("failed", "worker_lost")


def test_own_takeover_of_the_last_attempt_fails_database_error(pdf: Path, canned: CannedExtraction) -> None:
    calls = canned([canned_record()])
    document_id, job_id = upload(pdf)
    _set_job(job_id, status=JobStatus.RUNNING, attempt=3, max_attempts=3, heartbeat_at=datetime.now(UTC))
    process_job(job_id, attempt=3)  # the last attempt's failure could not be recorded
    assert calls == []
    with SessionLocal() as session:
        assert _job(session, job_id).status == "failed"
        assert _doc(session, document_id).error_code == "database_error"


# --------------------------------------------------------------------------------- (m1) ownership


def test_a_late_failure_does_not_overwrite_the_attempt_that_took_over(
    pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_id, job_id = upload(pdf)

    def taken_over_then_broken(doc: Any, **kwargs: Any) -> Any:
        _set_job(job_id, attempt=2, heartbeat_at=datetime.now(UTC))  # another worker took the job over meanwhile
        raise ValueError("bug")

    monkeypatch.setattr(processor, "extract_records", taken_over_then_broken)
    process_job(job_id)
    with SessionLocal() as session:
        assert (_job(session, job_id).status, _job(session, job_id).attempt) == ("running", 2)
        assert _doc(session, document_id).status == "processing"  # attempt 1 had no right to fail it


# ----------------------------------------------------------------------------------------- (m5) stage


def test_stage_shows_validating_while_records_are_validated(
    pdf: Path, canned: CannedExtraction, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned([canned_record()])
    _, job_id = upload(pdf)
    stages: dict[str, str | None] = {}
    real_own, real_embedder = processor.own_record_issues, processor.get_embedder

    def watching_validation(invoice: Any) -> Any:
        with SessionLocal() as session:
            stages["validating"] = _job(session, job_id).stage
        return real_own(invoice)

    def watching_embedder() -> Any:
        with SessionLocal() as session:
            stages["indexing"] = _job(session, job_id).stage
        return real_embedder()

    monkeypatch.setattr(processor, "own_record_issues", watching_validation)
    monkeypatch.setattr(processor, "get_embedder", watching_embedder)
    process_job(job_id)
    assert stages == {"validating": "validating", "indexing": "indexing"}


@pytest.fixture(autouse=True)
def _no_sweeper_left_behind() -> Iterator[None]:
    yield
    from app.queue import celery_app as module

    module.stop_sweeper()
