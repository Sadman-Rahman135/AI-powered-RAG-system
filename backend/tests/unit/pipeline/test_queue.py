"""Queue backends (inline, local thread pool, Celery) and startup recovery."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from app.config import BACKEND_DIR, get_settings, reset_settings_cache
from app.db.models import JobStatus, ProcessingJob
from app.db.session import SessionLocal, session_scope
from app.pipeline.processor import TransientError
from app.queue import get_queue, requeue_pending_jobs, reset_queue
from app.queue.base import JobQueue, retry_delay
from tests.unit.pipeline.builders import canned_record, invoice_pages, upload, write_pdf
from tests.unit.pipeline.conftest import CannedExtraction


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    return write_pdf(tmp_path / "invoice.pdf", invoice_pages("INV-1001"))


def _status(job_id: UUID) -> tuple[str, int]:
    with SessionLocal() as session:
        job = session.get(ProcessingJob, job_id)
        assert job is not None
        return job.status, job.attempt


def _backend(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv("QUEUE_BACKEND", name)
    reset_settings_cache()


# ------------------------------------------------------------------------------------------- selection


def test_get_queue_follows_the_backend_setting_and_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    inline = get_queue()
    assert inline.name == "inline" and get_queue() is inline and isinstance(inline, JobQueue)
    assert inline.ping()
    _backend(monkeypatch, "local")
    local = get_queue()
    assert local.name == "local" and get_queue() is local
    reset_queue()
    assert get_queue() is not local


def test_importing_app_queue_does_not_need_celery(tmp_path: Path) -> None:
    code = (
        "import sys, app.queue\n"
        "queue = app.queue.get_queue()\n"
        "print(queue.name, 'celery' in sys.modules, 'redis' in sys.modules)\n"
        "app.queue.reset_queue()\n"
    )
    env = {
        **os.environ,
        "DOCINTEL_IGNORE_DOTENV": "1",
        "QUEUE_BACKEND": "local",
        "DATABASE_URL": f"sqlite:///{(tmp_path / 'sub.db').as_posix()}",
        "REDIS_URL": "",
    }
    out = subprocess.run(  # a fresh interpreter: this test process has imported celery already
        [sys.executable, "-c", code], cwd=BACKEND_DIR, env=env, capture_output=True, text=True, timeout=120, check=True
    )
    assert out.stdout.split() == ["local", "False", "False"]


def test_retry_delay_is_exponential_capped_and_jittered() -> None:
    delays = [retry_delay(n) for n in range(8)]
    assert 10 <= delays[0] <= 12 and 20 <= delays[1] <= 24 and 40 <= delays[2] <= 48
    assert all(600 <= d <= 720 for d in delays[6:])


# ---------------------------------------------------------------------------------------------- inline


def test_inline_queue_processes_synchronously(pdf: Path, canned: CannedExtraction) -> None:
    canned([canned_record()])
    _, job_id = upload(pdf)
    get_queue().enqueue(job_id)
    assert _status(job_id) == ("succeeded", 1)


def test_inline_queue_retries_transient_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.queue import inline

    calls: list[UUID] = []
    abandoned: list[UUID] = []

    def flaky(job_id: UUID, attempt: int | None = None) -> None:
        calls.append(job_id)
        if len(calls) < 3:
            raise TransientError("database busy", code="database_error")

    monkeypatch.setattr(inline, "process_job", flaky)
    monkeypatch.setattr(inline, "abandon_job", abandoned.append)
    job_id = uuid4()
    inline.InlineQueue().enqueue(job_id)
    assert calls == [job_id] * 3 and abandoned == []

    calls.clear()

    def always(job_id: UUID, attempt: int | None = None) -> None:
        calls.append(job_id)
        raise TransientError("database down", code="database_error")

    monkeypatch.setattr(inline, "process_job", always)
    inline.InlineQueue().enqueue(job_id)
    assert len(calls) == 3 and abandoned == [job_id]  # out of attempts: failed, not raised


# ----------------------------------------------------------------------------------------------- local


def test_local_queue_runs_jobs_in_background(
    pdf: Path, canned: CannedExtraction, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned([canned_record()])
    _backend(monkeypatch, "local")
    _, job_id = upload(pdf)
    from app.pipeline import processor

    threads: list[str] = []
    real = processor.process_job
    def recording(job_id: UUID, attempt: int | None = None) -> None:
        threads.append(threading.current_thread().name)
        real(job_id, attempt=attempt)

    monkeypatch.setattr("app.queue.local.process_job", recording)
    get_queue().enqueue(job_id)
    reset_queue()  # waits for the running job
    assert _status(job_id) == ("succeeded", 1)
    assert threads and threads[0].startswith("docintel-job") and threads[0] != threading.current_thread().name


def test_local_queue_retries_after_a_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.queue import local

    calls: list[UUID] = []
    done = threading.Event()

    def flaky(job_id: UUID, attempt: int | None = None) -> None:
        calls.append(job_id)
        if len(calls) == 1:
            raise TransientError("database busy")
        done.set()

    monkeypatch.setattr(local, "process_job", flaky)
    queue = local.LocalQueue(1, retry_base_s=0.01)
    job_id = uuid4()
    queue.enqueue(job_id)
    assert done.wait(5)
    queue.shutdown(wait=True)
    assert calls == [job_id, job_id]
    assert not queue.ping()


def test_requeue_pending_jobs_local_enqueues_every_queued_job(
    pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, first = upload(pdf, filename="1.pdf")
    _, second = upload(pdf, filename="2.pdf")
    _, stale = upload(pdf, filename="3.pdf")
    with session_scope() as session:
        job = session.get(ProcessingJob, stale)
        assert job is not None
        job.status, job.attempt = JobStatus.RUNNING, 1
        job.heartbeat_at = datetime.now(UTC) - timedelta(hours=3)

    class Recorder:
        def __init__(self, name: str) -> None:
            self.name = name
            self.enqueued: list[UUID] = []

        def enqueue(self, job_id: UUID) -> None:
            self.enqueued.append(job_id)

        def ping(self) -> bool:
            return True

    import app.queue as queue_module

    local_queue = Recorder("local")
    monkeypatch.setattr(queue_module, "get_queue", lambda: local_queue)
    assert requeue_pending_jobs() == 3
    assert local_queue.enqueued == [first, second, stale]  # creation order; the stale one was recovered first

    with session_scope() as session:
        job = session.get(ProcessingJob, stale)
        assert job is not None
        job.status, job.heartbeat_at = JobStatus.RUNNING, datetime.now(UTC) - timedelta(hours=3)
    celery_queue = Recorder("celery")
    monkeypatch.setattr(queue_module, "get_queue", lambda: celery_queue)
    assert requeue_pending_jobs() == 1
    assert celery_queue.enqueued == [stale]  # the broker still holds the others


# ---------------------------------------------------------------------------------------------- celery


def test_celery_app_configuration() -> None:
    from app.queue.celery_app import QUEUE_NAME, celery_app

    conf = celery_app.conf
    assert conf.task_acks_late is True and conf.task_reject_on_worker_lost is True
    assert conf.worker_prefetch_multiplier == 1
    assert conf.broker_transport_options["visibility_timeout"] == 6 * 3600
    assert conf.task_default_queue == QUEUE_NAME == "documents"
    assert conf.broker_connection_retry_on_startup is True
    assert not conf.result_backend and conf.task_ignore_result is True
    # The thread pool ignores Celery time limits; JOB_MAX_RUNTIME_S (the heartbeat deadline) is the limit instead,
    # and the visibility timeout must outlast it so a running job's message is not redelivered meanwhile.
    assert conf.task_time_limit is None and conf.task_soft_time_limit is None
    assert conf.broker_transport_options["visibility_timeout"] > get_settings().job_max_runtime_s + 600


def test_celery_worker_command_finds_the_app_and_its_task() -> None:
    from celery.app.utils import find_app

    from app.queue.celery_app import TASK_NAME, celery_app

    assert find_app("app.queue.celery_app") is celery_app  # what `celery -A app.queue.celery_app worker` loads
    celery_app.loader.import_default_modules()  # the worker imports ``include`` modules at start
    task = celery_app.tasks[TASK_NAME]
    assert task.acks_late is True and task.max_retries == 2


def test_celery_queue_sends_the_job_to_the_documents_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.queue import celery_app as module

    sent: list[tuple[str, Any, Any]] = []
    monkeypatch.setattr(
        module.celery_app, "send_task", lambda name, args=None, queue=None, **kw: sent.append((name, args, queue))
    )
    _backend(monkeypatch, "celery")
    queue = get_queue()
    assert queue.name == "celery"
    job_id = uuid4()
    queue.enqueue(job_id)
    assert sent == [("app.queue.tasks.process_document_task", [str(job_id)], "documents")]


def test_celery_ping_reports_the_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    import fakeredis
    import redis

    from app.queue.celery_app import CeleryQueue

    assert CeleryQueue("redis://127.0.0.1:1/0").ping() is False  # nothing listens on port 1
    monkeypatch.setattr(redis.Redis, "from_url", staticmethod(lambda *a, **k: fakeredis.FakeRedis()))
    assert CeleryQueue("redis://broker:6379/0").ping() is True


def test_celery_task_retries_transient_errors_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.queue import tasks

    calls: list[UUID] = []

    def flaky(job_id: UUID, attempt: int | None = None) -> None:
        calls.append(job_id)
        if len(calls) < 3:
            raise TransientError("database busy", code="database_error")

    monkeypatch.setattr(tasks, "process_job", flaky)
    job_id = uuid4()
    result = tasks.process_document_task.apply(args=[str(job_id)])  # eager: retries run in place
    assert result.successful()
    assert calls == [job_id] * 3


def test_celery_task_gives_up_after_its_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.queue import tasks

    calls: list[UUID] = []
    abandoned: list[UUID] = []

    def always(job_id: UUID, attempt: int | None = None) -> None:
        calls.append(job_id)
        raise TransientError("database down", code="database_error")

    monkeypatch.setattr(tasks, "process_job", always)
    monkeypatch.setattr(tasks, "abandon_job", abandoned.append)
    job_id = uuid4()
    result = tasks.process_document_task.apply(args=[str(job_id)])
    assert result.successful()  # the failure is recorded on the job, not raised into the broker
    assert len(calls) == 3 and abandoned == [job_id]


def test_worker_ready_requeues_stale_jobs(pdf: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.queue import celery_app as module

    _, job_id = upload(pdf)
    with session_scope() as session:
        job = session.get(ProcessingJob, job_id)
        assert job is not None
        job.status, job.attempt = JobStatus.RUNNING, 1
        job.heartbeat_at = datetime.now(UTC) - timedelta(hours=2)
    sent: list[Any] = []
    monkeypatch.setattr(module.celery_app, "send_task", lambda name, args=None, queue=None, **kw: sent.append(args))
    module._on_worker_ready()
    assert sent == [[str(job_id)]]
    assert _status(job_id) == ("queued", 1)
    with SessionLocal() as session:
        assert (session.scalars(select(ProcessingJob.error)).one() or "").startswith("worker_lost")
