"""Celery application and the Celery queue backend.

Worker (docker compose)::

    celery -A app.queue.celery_app worker --pool=threads --concurrency=2 -Q documents

The thread pool is deliberate: OCR runs in its own spawn-based process pool, which a prefork child cannot host.
Redis is only the broker (no result backend): job state lives in the database, so a lost broker message is
recoverable. Messages are acknowledged after the task (``acks_late``) and redelivered if the worker dies; the
visibility timeout outlasts the longest job. Every worker runs a ``JobSweeper`` (at ``worker_ready``, then every
minute): jobs left ``running`` by a dead worker are re-queued and ``queued`` jobs whose message was lost are
re-sent, so nothing waits for the visibility timeout or a restart. On shutdown the sweep, the OCR process pool and
the cached LLM client stop.

No Celery ``task_time_limit`` / ``task_soft_time_limit`` is set: the thread pool does not implement them (they
would be silently inert). The attempt time limit is ``JOB_MAX_RUNTIME_S``, enforced by the job's own heartbeat
(``pipeline.progress.Heartbeat`` stops beating at the deadline, so the sweep re-queues or fails the attempt), and
OCR has its own per-page timeout (``OCR_PAGE_TIMEOUT_S``). A thread stuck in native code keeps its pool slot until
the worker restarts; its job is reclaimed and finished elsewhere meanwhile.

Importing this module needs no running Redis (connections are lazy); ``app.queue`` imports it only for
``QUEUE_BACKEND=celery``.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

import redis
from celery import Celery
from celery.signals import worker_ready, worker_shutdown

from app.config import Settings, get_settings
from app.queue.base import SWEEP_INTERVAL_S, JobSweeper

logger = logging.getLogger(__name__)

QUEUE_NAME = "documents"
TASK_NAME = "app.queue.tasks.process_document_task"
DEFAULT_BROKER_URL = "redis://localhost:6379/0"
# Longer than the default JOB_MAX_RUNTIME_S (4 h) plus the stale period, so Redis does not hand a running job's
# unacknowledged message to another worker (which would be a harmless no-op claim, but wasted work).
VISIBILITY_TIMEOUT_S = 6 * 3600
PING_TIMEOUT_S = 2.0

__all__ = [
    "QUEUE_NAME",
    "TASK_NAME",
    "CeleryQueue",
    "celery_app",
    "create_celery_app",
    "recover_and_enqueue",
    "start_sweeper",
    "stop_sweeper",
]


def broker_url(settings: Settings | None = None) -> str:
    return (settings or get_settings()).redis_url or DEFAULT_BROKER_URL


def create_celery_app(settings: Settings | None = None) -> Celery:
    """A configured Celery app (the module-level ``celery_app`` is the one workers and the API use)."""
    app = Celery("docintel", broker=broker_url(settings), include=["app.queue.tasks"])
    app.conf.update(
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        broker_transport_options={"visibility_timeout": VISIBILITY_TIMEOUT_S},
        task_default_queue=QUEUE_NAME,
        broker_connection_retry_on_startup=True,
        result_backend=None,
        task_ignore_result=True,
        task_serializer="json",
        accept_content=["json"],
        enable_utc=True,
        timezone="UTC",
        worker_send_task_events=False,
    )
    return app


celery_app = create_celery_app()


def send_job(job_id: UUID) -> None:
    """Publish one job to the ``documents`` queue."""
    celery_app.send_task(TASK_NAME, args=[str(job_id)], queue=QUEUE_NAME)


class CeleryQueue:
    """Sends jobs to the Celery workers through Redis."""

    name = "celery"

    def __init__(self, url: str | None = None) -> None:
        self._url = url or broker_url()

    def enqueue(self, job_id: UUID) -> None:
        send_job(job_id)

    def ping(self) -> bool:
        """The broker answers a Redis PING within two seconds."""
        client = redis.Redis.from_url(self._url, socket_connect_timeout=PING_TIMEOUT_S, socket_timeout=PING_TIMEOUT_S)
        try:
            return bool(client.ping())
        except (redis.RedisError, OSError):
            return False
        finally:
            client.close()


_sweeper: JobSweeper | None = None


def recover_and_enqueue() -> list[UUID]:
    """One sweep now: re-queue jobs a dead worker left ``running`` and re-send ``queued`` jobs whose message was
    lost; returns the ids published."""
    return JobSweeper(send_job).run_once()


def start_sweeper(interval_s: float = SWEEP_INTERVAL_S) -> JobSweeper:
    """Start this worker's periodic sweep (idempotent)."""
    global _sweeper
    if _sweeper is None:
        _sweeper = JobSweeper(send_job, interval_s=interval_s)
        _sweeper.start()
    return _sweeper


def stop_sweeper() -> None:
    global _sweeper
    if _sweeper is not None:
        _sweeper.stop()
        _sweeper = None


@worker_ready.connect
def _on_worker_ready(**_: Any) -> None:
    """Recover at once (a crashed predecessor's jobs), then every minute while the worker runs."""
    requeued = recover_and_enqueue()
    if requeued:
        logger.warning("worker start: re-sent %d stale or forgotten job(s)", len(requeued))
    start_sweeper()


@worker_shutdown.connect
def _on_worker_shutdown(**_: Any) -> None:
    from app.ingestion.ocr import shutdown_ocr_pool
    from app.pipeline.processor import close_llm_provider

    stop_sweeper()
    shutdown_ocr_pool(wait=False)
    close_llm_provider()
