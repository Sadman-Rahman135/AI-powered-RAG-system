"""The job-queue contract shared by every backend, the retry backoff and the periodic job sweep."""

from __future__ import annotations

import logging
import random
import threading
from collections.abc import Callable
from typing import Protocol, runtime_checkable
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

RETRY_BASE_S = 10.0
RETRY_CAP_S = 600.0
RETRY_JITTER = 0.2
SWEEP_INTERVAL_S = 60.0


@runtime_checkable
class JobQueue(Protocol):
    """Runs ``process_job`` for enqueued job ids - now, in a thread, or on a Celery worker."""

    name: str  # "inline" | "local" | "celery"

    def enqueue(self, job_id: UUID) -> None:
        """Schedule a job whose rows are already committed."""
        ...

    def ping(self) -> bool:
        """Whether the backend can accept work (Celery: the broker is reachable)."""
        ...


def retry_delay(retries: int, *, base_s: float = RETRY_BASE_S, cap_s: float = RETRY_CAP_S) -> float:
    """Exponential backoff before retry number ``retries + 1``: base, 2 x base, 4 x base ... capped, plus up to
    20 % jitter so jobs that failed together do not retry together."""
    delay = min(cap_s, base_s * 2 ** max(0, retries))
    return delay * (1 + random.uniform(0, RETRY_JITTER))


class JobSweeper:
    """Every ``interval_s`` (daemon thread): ``processor.sweep_jobs`` in a short transaction, then ``publish``
    each returned id - stale running jobs re-queued, and queued jobs whose message was lost re-sent.

    This is what makes a lost broker message, a failed enqueue or a crashed worker recover on its own within
    minutes instead of waiting for a restart. Errors are logged and the next round tries again.
    """

    def __init__(self, publish: Callable[[UUID], None], *, interval_s: float = SWEEP_INTERVAL_S) -> None:
        self._publish = publish
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="docintel-job-sweeper", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()

    def run_once(self) -> list[UUID]:
        """One sweep; returns the ids it published."""
        from app.db.session import session_scope
        from app.pipeline.processor import sweep_jobs

        try:
            with session_scope() as session:
                job_ids = sweep_jobs(session)
        except SQLAlchemyError as exc:
            logger.warning("job sweep failed (%s); retrying at the next round", type(exc).__name__)
            return []
        published: list[UUID] = []
        for job_id in job_ids:
            try:
                self._publish(job_id)
                published.append(job_id)
            except Exception as exc:  # the broker may be down too; the next round re-sends (stamp is 10 min)
                logger.warning("job sweep could not send job %s (%s)", job_id, type(exc).__name__)
        return published

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self.run_once()
