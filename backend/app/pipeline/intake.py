"""Intake: the database side of an upload (document + queued job + history) and re-processing requests.

The caller (the upload endpoint, the evaluation runner) stores the file first (``app.storage``), calls
``create_document``, commits, and only then enqueues ``job.id`` - so a worker never sees a job whose rows are not
committed yet.
"""

from __future__ import annotations

import os
import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.base import utcnow
from app.db.models import Document, DocumentStatus, JobStatus, ProcessingJob
from app.ingestion.sniff import extension_mismatch
from app.ingestion.types import FileKind
from app.pipeline.progress import (
    QUEUED_RESEND_AFTER_S,
    STAGE_REPROCESS,
    STAGE_UPLOAD,
    add_event,
    heartbeat_is_stale,
    queued_since,
)
from app.storage import StoredFile

MAX_FILENAME_CHARS = 255
DEFAULT_FILENAME = "document"
WARNING_EXTENSION_MISMATCH = "extension_mismatch"
INTAKE_WARNINGS = frozenset({WARNING_EXTENSION_MISMATCH})
"""Document warnings set at intake; processing keeps them when it rewrites ``documents.warnings``."""

_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_MAX_EXTENSION_CHARS = 16
_REMOVED_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co", "Cn"})  # control, format (bidi overrides), unassigned

__all__ = [
    "INTAKE_WARNINGS",
    "AlreadyProcessingError",
    "IntakeResult",
    "create_document",
    "create_reprocess_job",
    "sanitize_filename",
]


def sanitize_filename(name: str) -> str:
    """A safe display name: the basename (any path, drive or ``..`` stripped, both separators), NFC-normalised,
    control and bidi-format characters removed, at most 255 characters keeping the extension; ``"document"``
    when nothing is left. The result is shown to people only - storage keys never come from it."""
    text = unicodedata.normalize("NFC", name or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) not in _REMOVED_CATEGORIES)
    base = re.split(r"[\\/]", text)[-1]
    base = _DRIVE_RE.sub("", base).strip()
    if not base.strip("."):
        return DEFAULT_FILENAME
    if len(base) > MAX_FILENAME_CHARS:
        stem, extension = os.path.splitext(base)
        if len(extension) > _MAX_EXTENSION_CHARS:
            stem, extension = base, ""
        base = stem[: MAX_FILENAME_CHARS - len(extension)].rstrip() + extension
    return base


@dataclass
class IntakeResult:
    document: Document
    job: ProcessingJob
    duplicate_of_id: UUID | None  # oldest earlier document with the same sha256


class AlreadyProcessingError(Exception):
    """The document already has a queued or running job."""


def create_document(
    session: Session, *, filename: str, stored: StoredFile, kind: FileKind, content_type: str
) -> IntakeResult:
    """Add the document (``queued``), its first job and the ``upload`` event; flushes, does not commit.

    A document whose bytes (sha256) were uploaded before points at the oldest such document
    (``duplicate_of_id``), which processing reports as ``DUPLICATE_FILE``.
    """
    settings = get_settings()
    display_name = sanitize_filename(filename)
    duplicate_of_id = session.scalar(
        select(Document.id).where(Document.sha256 == stored.sha256).order_by(Document.created_at, Document.id).limit(1)
    )
    warnings = [WARNING_EXTENSION_MISMATCH] if extension_mismatch(kind, display_name) else []
    document = Document(
        id=uuid.uuid4(),
        filename=display_name,
        content_type=content_type[:100],
        file_kind=FileKind(kind).value,
        size_bytes=stored.size,
        sha256=stored.sha256,
        storage_key=stored.key,
        status=DocumentStatus.QUEUED,
        duplicate_of_id=duplicate_of_id,
        warnings=warnings,
    )
    job = _new_job(document.id, settings.job_max_attempts)
    session.add_all([document, job])
    session.flush()  # the event references the job, which the ORM does not know to insert first
    add_event(
        session,
        document_id=document.id,
        job_id=job.id,
        stage=STAGE_UPLOAD,
        message="File received and queued for processing",
        data={
            "size_bytes": stored.size,
            "file_kind": document.file_kind,
            "duplicate_of_id": str(duplicate_of_id) if duplicate_of_id else None,
            "warnings": warnings,
        },
    )
    session.flush()
    return IntakeResult(document=document, job=job, duplicate_of_id=duplicate_of_id)


def create_reprocess_job(session: Session, document: Document, *, now: datetime | None = None) -> ProcessingJob:
    """Queue a new job for ``document`` and set it ``queued``; flushes, does not commit.

    Raises ``AlreadyProcessingError`` while a job is queued or running - except that a job which can no longer
    be making progress does not block, it is marked failed and superseded by the new job: a running job whose worker
    stopped sending heartbeats (``JOB_STALE_AFTER_S``), or a queued job that has not been handed to the queue for
    ``QUEUED_RESEND_AFTER_S`` (its message was probably lost).
    """
    settings = get_settings()
    now = now or utcnow()
    active = session.scalars(
        select(ProcessingJob).where(
            ProcessingJob.document_id == document.id,
            ProcessingJob.status.in_((JobStatus.QUEUED, JobStatus.RUNNING)),
        )
    ).all()
    waited_too_long = now - timedelta(seconds=QUEUED_RESEND_AFTER_S)
    for job in active:
        if job.status == JobStatus.RUNNING and heartbeat_is_stale(job, now, settings.job_stale_after_s):
            reason = "superseded: the worker stopped sending heartbeats"
        elif job.status == JobStatus.QUEUED and (since := queued_since(job)) is not None and since < waited_too_long:
            reason = "superseded: the job never started"
        else:
            raise AlreadyProcessingError(f"document {document.id} already has a {job.status} job")
        job.status = JobStatus.FAILED
        job.finished_at = now
        job.error = reason
    job = _new_job(document.id, settings.job_max_attempts)
    session.add(job)
    document.status = DocumentStatus.QUEUED
    session.flush()
    add_event(
        session,
        document_id=document.id,
        job_id=job.id,
        stage=STAGE_REPROCESS,
        message="Reprocessing requested",
        data={"max_attempts": job.max_attempts},
    )
    session.flush()
    return job


def _new_job(document_id: UUID, max_attempts: int) -> ProcessingJob:
    return ProcessingJob(
        id=uuid.uuid4(),
        document_id=document_id,
        status=JobStatus.QUEUED,
        attempt=0,
        max_attempts=max_attempts,
        progress_current=0,
        progress_total=0,
    )
