"""Job queue backends, chosen by ``QUEUE_BACKEND``.

* ``inline`` - ``enqueue`` runs the job synchronously (tests, the evaluation runner);
* ``local`` - a thread pool in the API process (development without Redis);
* ``celery`` - ``process_document_task`` on the ``documents`` queue in Redis (production).

Celery is imported only for the ``celery`` backend, so ``import app.queue`` works without Celery configured.
"""

from __future__ import annotations

import logging
import threading

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.config import Settings, get_settings
from app.queue.base import JobQueue

logger = logging.getLogger(__name__)

__all__ = ["JobQueue", "get_queue", "requeue_pending_jobs", "reset_queue"]

_lock = threading.Lock()
_cached: tuple[tuple[object, ...], JobQueue] | None = None


def _key(settings: Settings) -> tuple[object, ...]:
    return settings.queue_backend, settings.local_queue_workers, settings.redis_url


def _build(settings: Settings) -> JobQueue:
    if settings.queue_backend == "inline":
        from app.queue.inline import InlineQueue

        return InlineQueue()
    if settings.queue_backend == "local":
        from app.queue.local import LocalQueue

        return LocalQueue(settings.local_queue_workers)
    from app.queue.celery_app import CeleryQueue

    return CeleryQueue()


def _close(queue: JobQueue) -> None:
    shutdown = getattr(queue, "shutdown", None)
    if callable(shutdown):
        shutdown(wait=True)


def get_queue() -> JobQueue:
    """The configured queue; one instance per configuration (a changed ``QUEUE_BACKEND`` builds a new one)."""
    global _cached
    settings = get_settings()
    key = _key(settings)
    with _lock:
        if _cached is not None and _cached[0] == key:
            return _cached[1]
        if _cached is not None:
            _close(_cached[1])
        queue = _build(settings)
        _cached = (key, queue)
        return queue


def reset_queue() -> None:
    """Drop the cached queue (a local pool finishes its running jobs first). For tests."""
    global _cached
    with _lock:
        cached, _cached = _cached, None
    if cached is not None:
        _close(cached[1])


def requeue_pending_jobs() -> int:
    """Startup hook for the API: recover stale ``running`` jobs and enqueue the work a restart lost.

    The local backend keeps jobs in memory only, so every ``queued`` job is enqueued again; for Celery (the broker
    keeps its messages) and inline, what ``sweep_jobs`` picks: jobs recovered from dead workers and queued jobs not
    handed to the queue for ten minutes. Returns how many jobs were enqueued. Database errors are logged, not
    raised: the API must start (``/ready`` reports the database).
    """
    from app.db.models import JobStatus, ProcessingJob
    from app.db.session import session_scope
    from app.pipeline.processor import sweep_jobs

    queue = get_queue()
    try:
        with session_scope() as session:
            swept = sweep_jobs(session)
            if queue.name == "local":
                job_ids = list(
                    session.scalars(
                        select(ProcessingJob.id)
                        .where(ProcessingJob.status == JobStatus.QUEUED)
                        .order_by(ProcessingJob.created_at)
                    )
                )
            else:
                job_ids = swept
    except SQLAlchemyError as exc:
        logger.error("startup job recovery failed (%s)", type(exc).__name__)
        return 0
    for job_id in job_ids:
        queue.enqueue(job_id)
    if job_ids:
        logger.info("startup: enqueued %d pending job(s) on the %s queue", len(job_ids), queue.name)
    return len(job_ids)
