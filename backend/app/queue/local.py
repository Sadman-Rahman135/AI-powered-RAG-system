"""Local backend: a thread pool inside the API process (development without Redis).

Jobs are lost from memory when the process stops, but not from the database: ``requeue_pending_jobs`` enqueues the
still-``queued`` jobs again at startup, and a ``JobSweeper`` re-queues stale or forgotten jobs every minute (jobs
already waiting in this pool are not submitted twice). Transient failures are retried after an exponential backoff
(a timer, so no pool thread sleeps), handing the failed attempt number to the retry.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

from app.config import get_settings
from app.pipeline.processor import TransientError, abandon_job, process_job
from app.queue.base import RETRY_BASE_S, SWEEP_INTERVAL_S, JobSweeper, retry_delay

logger = logging.getLogger(__name__)


class LocalQueue:
    name = "local"

    def __init__(
        self,
        workers: int,
        *,
        retry_base_s: float = RETRY_BASE_S,
        sweep_interval_s: float | None = SWEEP_INTERVAL_S,
    ) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="docintel-job")
        self._retry_base_s = retry_base_s
        self._timers: set[threading.Timer] = set()
        self._pending: set[UUID] = set()  # submitted and not started yet
        self._lock = threading.Lock()
        self._closed = False
        self._sweeper: JobSweeper | None = None
        if sweep_interval_s is not None:
            self._sweeper = JobSweeper(self._sweep_publish, interval_s=sweep_interval_s)
            self._sweeper.start()

    def enqueue(self, job_id: UUID) -> None:
        self._submit(job_id, 0, None)

    def ping(self) -> bool:
        return not self._closed

    def shutdown(self, *, wait: bool = True) -> None:
        """Stop accepting work, the sweeper and pending retries; (``wait``) let running jobs finish."""
        with self._lock:
            self._closed = True
            timers, self._timers = self._timers, set()
        for timer in timers:
            timer.cancel()
        if self._sweeper is not None:
            self._sweeper.stop()
        self._executor.shutdown(wait=wait)

    def _sweep_publish(self, job_id: UUID) -> None:
        with self._lock:
            if job_id in self._pending:
                return  # already waiting in this pool
        self._submit(job_id, 0, None)

    def _submit(self, job_id: UUID, retries: int, attempt: int | None) -> None:
        with self._lock:
            if self._closed:
                logger.info("job %s: local queue is shut down; the job stays queued", job_id)
                return
            self._pending.add(job_id)
            self._executor.submit(self._run, job_id, retries, attempt)

    def _run(self, job_id: UUID, retries: int, attempt: int | None) -> None:
        with self._lock:
            self._pending.discard(job_id)
        try:
            process_job(job_id, attempt=attempt)
        except TransientError as exc:
            if retries + 1 >= max(1, get_settings().job_max_attempts):
                abandon_job(job_id)
                return
            delay = retry_delay(retries, base_s=self._retry_base_s)
            logger.info("job %s: transient failure (%s); retry in %.1f s", job_id, exc.code, delay)
            self._schedule(job_id, retries + 1, exc.attempt, delay)
        except Exception as exc:  # never let a pool thread die silently
            logger.error("job %s: local worker error (%s)", job_id, type(exc).__name__)

    def _schedule(self, job_id: UUID, retries: int, attempt: int | None, delay: float) -> None:
        if delay <= 0:
            self._submit(job_id, retries, attempt)
            return

        def fire() -> None:
            with self._lock:
                self._timers.discard(timer)
            self._submit(job_id, retries, attempt)

        timer = threading.Timer(delay, fire)
        timer.daemon = True
        with self._lock:
            if self._closed:
                return
            self._timers.add(timer)
        timer.start()
