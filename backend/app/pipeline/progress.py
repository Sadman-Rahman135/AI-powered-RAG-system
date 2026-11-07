"""Job progress, heartbeats and processing history.

``Heartbeat`` is a daemon thread that refreshes ``heartbeat_at`` every ``HEARTBEAT_INTERVAL_S`` for as long as an
attempt runs, whatever the attempt is doing (a long OCR page, a slow LLM call, embedding), so a job is only ever
"stale" (no heartbeat for ``JOB_STALE_AFTER_S``) when its worker is gone - or when the attempt has outlived its
``AttemptDeadline`` (``JOB_MAX_RUNTIME_S``): past the deadline nothing refreshes the heartbeat any more, so the
sweep re-queues (or, after the last attempt, fails) a hung attempt like a dead one. This is the attempt time limit:
Celery's ``task_time_limit`` cannot do it, because the worker's thread pool does not enforce time limits, and a
thread stuck in native code cannot be killed anyway. For a ``queued`` job the same column records when it was last
handed to the queue (``queued_since``), which the periodic sweep uses to re-send jobs whose message was lost.

``ProgressReporter`` writes a job's stage, ``progress_current/total`` and heartbeat through a *short, separate*
session per write, so a 1000-page job never holds a transaction open while it parses and the UI sees progress
as it happens. Page-level progress is throttled to at most one write per ``min_interval_s`` (plus the final
page), so the database is not flooded either. Progress writes are best effort: a failed write is logged and the
job continues (the final transaction decides the outcome). Like the heartbeat, they only touch the job while it
is still running under the reporter's attempt, and stop refreshing the heartbeat after the deadline.

Events (``processing_events``) are the history: one row per stage, written when the stage finishes, with counts
and timings in ``data`` - never document text or extracted values.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.base import utcnow
from app.db.models import JobStatus, ProcessingEvent, ProcessingJob

logger = logging.getLogger(__name__)

# Job stages (``processing_jobs.stage``) and the stages events are recorded under.
STAGE_UPLOAD = "upload"
STAGE_START = "start"
STAGE_PARSING = "parsing"
STAGE_OCR = "ocr"
STAGE_EXTRACTING = "extracting"
STAGE_VALIDATING = "validating"
STAGE_INDEXING = "indexing"
STAGE_DONE = "done"
STAGE_RETRY = "retry"
STAGE_ERROR = "error"
STAGE_REPROCESS = "reprocess"
STAGE_RECOVERED = "recovered"

HEARTBEAT_INTERVAL_S = 30.0
QUEUED_RESEND_AFTER_S = 600
"""A ``queued`` job not handed to the queue for this long may have lost its message: the sweep re-sends it and a
re-processing request may replace it."""

SessionFactory = Callable[[], Session]

__all__ = [
    "HEARTBEAT_INTERVAL_S",
    "NO_DEADLINE",
    "QUEUED_RESEND_AFTER_S",
    "STAGE_DONE",
    "STAGE_ERROR",
    "STAGE_EXTRACTING",
    "STAGE_INDEXING",
    "STAGE_OCR",
    "STAGE_PARSING",
    "STAGE_RECOVERED",
    "STAGE_REPROCESS",
    "STAGE_RETRY",
    "STAGE_START",
    "STAGE_UPLOAD",
    "STAGE_VALIDATING",
    "AttemptDeadline",
    "Heartbeat",
    "ProgressReporter",
    "SessionFactory",
    "add_event",
    "as_utc",
    "heartbeat_is_stale",
    "queued_since",
    "worker_identity",
]


def as_utc(value: datetime | None) -> datetime | None:
    """``value`` as an aware UTC datetime (SQLite hands back naive UTC values)."""
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def heartbeat_is_stale(job: ProcessingJob, now: datetime, stale_after_s: int) -> bool:
    """Whether a running job's last sign of life (heartbeat, else start) is older than ``stale_after_s``."""
    last = as_utc(job.heartbeat_at) or as_utc(job.started_at)
    return last is None or last < now - timedelta(seconds=stale_after_s)


def queued_since(job: ProcessingJob) -> datetime | None:
    """When a ``queued`` job was last handed to the queue (re-queues stamp ``heartbeat_at``), else its creation."""
    return as_utc(job.heartbeat_at) or as_utc(job.created_at)


def worker_identity() -> str:
    """``host:pid:thread`` of the current worker thread (fits ``processing_jobs.worker_id``)."""
    return f"{socket.gethostname()}:{os.getpid()}:{threading.get_ident()}"[:100]


@dataclass(frozen=True, slots=True)
class AttemptDeadline:
    """The moment (``clock`` seconds) after which an attempt has run too long; ``at=None`` means no limit."""

    at: float | None
    clock: Callable[[], float] = field(default=time.monotonic, compare=False)

    @classmethod
    def after(cls, seconds: float | None, *, clock: Callable[[], float] = time.monotonic) -> AttemptDeadline:
        """A deadline ``seconds`` from now (``None`` or ``<= 0``: no limit)."""
        return cls(clock() + seconds if seconds and seconds > 0 else None, clock)

    def passed(self) -> bool:
        return self.at is not None and self.clock() >= self.at


NO_DEADLINE = AttemptDeadline(None)


def add_event(
    session: Session,
    *,
    document_id: UUID,
    job_id: UUID | None,
    stage: str,
    message: str,
    level: str = "info",
    data: dict[str, Any] | None = None,
) -> ProcessingEvent:
    """Add a history row to ``session`` (not flushed or committed)."""
    event = ProcessingEvent(
        document_id=document_id, job_id=job_id, stage=stage, level=level, message=message, data=data or {}
    )
    session.add(event)
    return event


class ProgressReporter:
    """Stage, progress and heartbeat updates of one attempt of a job, each in its own short transaction.

    Writes apply only while the job is ``running`` under ``attempt`` (an attempt that was taken over cannot touch
    the new attempt's stage, progress or heartbeat), and leave ``heartbeat_at`` alone once ``deadline`` has passed.
    ``on_progress`` has the ``ProgressCallback`` signature, so it can be handed to the parsers and the extraction
    service directly. ``clock`` (monotonic seconds) is injectable for tests; ``writes`` counts successful writes.
    """

    def __init__(
        self,
        job_id: UUID,
        attempt: int,
        session_factory: SessionFactory,
        *,
        deadline: AttemptDeadline = NO_DEADLINE,
        min_interval_s: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.job_id = job_id
        self.attempt = attempt
        self._session_factory = session_factory
        self._deadline = deadline
        self._interval = min_interval_s
        self._clock = clock
        self._last_write: float | None = None
        self._pending: dict[str, Any] | None = None
        self._lock = threading.Lock()  # the LLM pool reports from several threads
        self.writes = 0

    def set_stage(self, stage: str, *, current: int = 0, total: int = 0) -> None:
        """Enter ``stage`` now (never throttled)."""
        with self._lock:
            self._pending = None
            self._write({"stage": stage, "progress_current": current, "progress_total": total})

    def on_progress(self, current: int, total: int, stage: str) -> None:
        """Page/segment progress: written at most once per interval, and always for the last unit."""
        values = {"stage": stage, "progress_current": current, "progress_total": total}
        with self._lock:
            now = self._clock()
            due = self._last_write is None or now - self._last_write >= self._interval
            if due or (total and current >= total):
                self._pending = None
                self._write(values)
            else:
                self._pending = values

    def flush(self) -> None:
        """Write progress that the throttle held back."""
        with self._lock:
            if self._pending is not None:
                self._write(self._pending)
                self._pending = None

    def event(
        self, document_id: UUID, stage: str, message: str, *, level: str = "info", data: dict[str, Any] | None = None
    ) -> None:
        """Record a history row right away (its own transaction; database errors propagate)."""
        with self._session_factory() as session:
            add_event(session, document_id=document_id, job_id=self.job_id, stage=stage, message=message,
                      level=level, data=data)
            session.commit()

    def _write(self, values: dict[str, Any]) -> None:
        self._last_write = self._clock()
        if not self._deadline.passed():
            values = {**values, "heartbeat_at": utcnow()}
        try:
            with self._session_factory() as session:
                session.execute(
                    update(ProcessingJob)
                    .where(
                        ProcessingJob.id == self.job_id,
                        ProcessingJob.status == JobStatus.RUNNING,
                        ProcessingJob.attempt == self.attempt,
                    )
                    .values(**values)
                    .execution_options(synchronize_session=False)
                )
                session.commit()
            self.writes += 1
        except SQLAlchemyError as exc:
            logger.warning("progress update of job %s failed (%s); continuing", self.job_id, type(exc).__name__)


class Heartbeat:
    """Refreshes ``heartbeat_at`` of one attempt every ``interval_s`` from a daemon thread (short sessions).

    Only the attempt it was started for is touched (``status = running AND attempt = <attempt>``), so a heartbeat
    can never keep a job alive that another attempt has taken over. Once ``deadline`` has passed the thread logs a
    warning and stops beating (``expired``): the attempt then looks like a dead worker's and the sweep reclaims it.
    Failed writes are logged and retried at the next beat. ``beats`` counts successful writes. Use as a context
    manager, or ``start()``/``stop()``.
    """

    def __init__(
        self,
        job_id: UUID,
        attempt: int,
        session_factory: SessionFactory,
        *,
        interval_s: float | None = None,
        deadline: AttemptDeadline = NO_DEADLINE,
    ) -> None:
        self.job_id = job_id
        self.attempt = attempt
        self._session_factory = session_factory
        self._interval = HEARTBEAT_INTERVAL_S if interval_s is None else interval_s
        self._deadline = deadline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=f"docintel-heartbeat-{job_id}", daemon=True)
        self.beats = 0
        self.expired = False

    def __enter__(self) -> Heartbeat:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            if self._deadline.passed():
                self.expired = True
                logger.warning(
                    "job %s: attempt %d exceeded JOB_MAX_RUNTIME_S; heartbeat stopped so the job sweep reclaims it",
                    self.job_id,
                    self.attempt,
                )
                return
            self._beat()

    def _beat(self) -> None:
        try:
            with self._session_factory() as session:
                session.execute(
                    update(ProcessingJob)
                    .where(
                        ProcessingJob.id == self.job_id,
                        ProcessingJob.status == JobStatus.RUNNING,
                        ProcessingJob.attempt == self.attempt,
                    )
                    .values(heartbeat_at=utcnow())
                    .execution_options(synchronize_session=False)
                )
                session.commit()
            self.beats += 1
        except SQLAlchemyError as exc:
            logger.warning("heartbeat of job %s failed (%s); retrying at the next beat", self.job_id,
                           type(exc).__name__)
