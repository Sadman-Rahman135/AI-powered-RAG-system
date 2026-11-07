"""Celery tasks."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from app.config import get_settings
from app.pipeline.processor import TransientError, abandon_job, process_job
from app.queue.base import retry_delay
from app.queue.celery_app import TASK_NAME, celery_app


def _max_retries() -> int:
    return max(0, get_settings().job_max_attempts - 1)


@celery_app.task(bind=True, name=TASK_NAME, acks_late=True, ignore_result=True, max_retries=_max_retries())
def process_document_task(self: Any, job_id: str, attempt: int | None = None) -> None:
    """Process one job; a ``TransientError`` is retried with exponential backoff, up to ``JOB_MAX_ATTEMPTS``.

    The retry carries the failed attempt's number (``attempt``), so it can take over a job that attempt left
    ``running`` because it could not record its own failure. ``process_job`` itself fails the job after its last
    attempt; if attempts keep failing before they can even be recorded (database down), the job is failed here once
    Celery's retries are used up.
    """
    job = UUID(job_id)
    try:
        process_job(job, attempt=attempt)
    except TransientError as exc:
        max_retries = _max_retries()
        if self.request.retries >= max_retries:
            abandon_job(job)
            return
        raise self.retry(
            args=[job_id],
            kwargs={"attempt": exc.attempt},
            exc=exc,
            countdown=retry_delay(self.request.retries),
            max_retries=max_retries,
        ) from exc
