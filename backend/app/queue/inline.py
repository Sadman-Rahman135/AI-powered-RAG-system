"""Inline backend: ``enqueue`` processes the job synchronously (tests, the evaluation runner)."""

from __future__ import annotations

import logging
from uuid import UUID

from app.config import get_settings
from app.pipeline.processor import TransientError, abandon_job, process_job

logger = logging.getLogger(__name__)


class InlineQueue:
    """Runs the job in the caller's thread; transient failures are retried at once, up to ``JOB_MAX_ATTEMPTS``."""

    name = "inline"

    def enqueue(self, job_id: UUID) -> None:
        attempts = max(1, get_settings().job_max_attempts)
        attempt: int | None = None  # the failed attempt a retry may take over (TransientError.attempt)
        for _ in range(attempts):
            try:
                process_job(job_id, attempt=attempt)
                return
            except TransientError as exc:
                attempt = exc.attempt
                logger.info("job %s: transient failure (%s); retrying inline", job_id, exc.code)
        abandon_job(job_id)

    def ping(self) -> bool:
        return True
