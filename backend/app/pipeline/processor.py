"""Background job processing: one stored file -> validated, searchable invoice records.

``process_job(job_id)`` runs one attempt of a job:

1. **claim** (short transaction): a compare-and-set moves the job ``queued -> running`` - or takes over a running job
   whose heartbeat is stale (dead worker), or whose attempt is the one the queue is retrying (that attempt failed
   and could not even record it). A job that is already ``succeeded``/``failed``, missing, or owned by a live
   worker is left alone - duplicate deliveries are no-ops. A takeover with no attempts left fails the job instead
   (``worker_lost`` / ``job_timeout`` / ``database_error``), so ``max_attempts`` always holds. While the attempt
   runs, a background ``Heartbeat`` refreshes ``heartbeat_at`` every 30 s - until the attempt has run for
   ``JOB_MAX_RUNTIME_S``: then it stops, so a hung attempt (a stuck OCR child, a parser that never returns) goes
   stale like a dead worker's. The queue's ``JobSweeper`` re-queues jobs whose heartbeat stopped
   (``JOB_STALE_AFTER_S``, 5 min) and re-sends queued jobs whose message was lost. An attempt that lost the job
   (re-queued by the sweep, taken over) stops at its next stage boundary; if it gets as far as the persist step,
   its results are discarded there.
2. **parsing** -> **extracting** -> **validating** -> **indexing**: parsing (progress = pages, stage ``ocr`` while
   pages are OCR'd), extraction (progress = segments), then chunking and embedding - all outside any long
   transaction; ``ProgressReporter`` writes stage, progress and heartbeat in short throttled sessions.
3. **persist** (one transaction): the document's previous pages, records, issues, chunks and cross-verifications
   are replaced (``Correction`` rows survive, their invoice link set NULL), records are validated, duplicates and
   cross-verification are reconciled against other documents (``review.reconcile_invoices``), document-level issues
   are added, chunks are stored, and the document and job get their final status. A crash before the commit
   changes nothing; a redelivery repeats the attempt.

Errors: a permanent ``IngestionError`` fails the document with its code and safe message (no retry). Database
trouble (``OperationalError``, pool timeouts, invalidated connections, integrity conflicts with a concurrent
writer) and non-permanent ingestion errors re-queue the job and raise ``TransientError`` so the queue retries it
with backoff, until ``max_attempts`` - then the job fails. Anything else fails the document as ``internal_error``
with a generic message; the log gets the stack frames and ids, never document text or exception messages.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import and_, delete, insert, or_, select, update
from sqlalchemy import exc as sa_exc
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.base import utcnow
from app.db.models import (
    Chunk,
    Document,
    DocumentPage,
    DocumentStatus,
    Invoice,
    JobStatus,
    ProcessingJob,
    ValidationIssueRow,
)
from app.db.session import SessionLocal
from app.domain.schemas import (
    ExtractionResult,
    ExtractionWarning,
    FieldMeta,
    InvoiceData,
    IssueCode,
    Severity,
    ValidationIssue,
)
from app.extraction.llm.base import LLMConfigurationError, LLMProvider
from app.extraction.llm.factory import get_llm_provider
from app.extraction.service import ExtractedRecord, extract_records
from app.ingestion.router import parse_document
from app.ingestion.types import FileKind, IngestionError, ParsedDocument, ProgressCallback
from app.pipeline import progress
from app.pipeline.intake import INTAKE_WARNINGS
from app.pipeline.progress import (
    QUEUED_RESEND_AFTER_S,
    STAGE_DONE,
    STAGE_ERROR,
    STAGE_EXTRACTING,
    STAGE_INDEXING,
    STAGE_PARSING,
    STAGE_RECOVERED,
    STAGE_RETRY,
    STAGE_START,
    STAGE_VALIDATING,
    AttemptDeadline,
    Heartbeat,
    ProgressReporter,
    SessionFactory,
    add_event,
    as_utc,
    heartbeat_is_stale,
    queued_since,
    worker_identity,
)
from app.pipeline.review import (
    own_record_issues,
    reconcile_invoices,
    record_issue_rows,
    refresh_document_status,
    serialize_reconciliation,
)
from app.search.chunking import ChunkDraft, build_chunks
from app.search.embeddings import Embedder, get_embedder
from app.search.service import index_document
from app.storage import get_storage
from app.validation.rules import document_level_issues

logger = logging.getLogger(__name__)

INTERNAL_ERROR = "internal_error"
INTERNAL_ERROR_MESSAGE = "An unexpected error occurred while processing the document"
DATABASE_ERROR = "database_error"
DATABASE_ERROR_MESSAGE = "The database was unavailable or busy"
WORKER_LOST = "worker_lost"
JOB_TIMEOUT = "job_timeout"
FILE_MISSING = "file_missing"
LLM_FALLBACK_WARNINGS = frozenset(
    w.value for w in (ExtractionWarning.LLM_UNAVAILABLE, ExtractionWarning.LLM_FAILED, ExtractionWarning.LLM_REFUSED)
)
MAX_DOCUMENT_WARNINGS = 200
_EMBED_BATCH = 128
_INSERT_BATCH = 500
_TEXT_LIMITS = {"invoice_number": 100, "supplier_name": 255, "customer_name": 255}

__all__ = [
    "StaleJobs",
    "TransientError",
    "abandon_job",
    "close_llm_provider",
    "process_job",
    "reclaim_stale_jobs",
    "recover_stale_jobs",
    "sweep_jobs",
]


class TransientError(Exception):
    """A failure that may go away if the job runs again later (the queue retries it with backoff).

    ``attempt`` is the attempt that failed (``None`` when the job could not even be claimed). The queue passes it
    back to ``process_job(..., attempt=...)``: if the failed attempt could not record its failure (the database
    was away), the job is still ``running`` under that attempt and the retry takes it over.
    """

    def __init__(self, message: str, *, code: str = "transient_error", attempt: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.attempt = attempt


@dataclass(frozen=True, slots=True)
class _Claim:
    """What one attempt needs to know about its job and document (read when the job was claimed)."""

    job_id: UUID
    document_id: UUID
    attempt: int
    max_attempts: int
    storage_key: str
    file_kind: str
    filename: str
    duplicate_of_id: UUID | None
    intake_warnings: tuple[str, ...]


@dataclass(slots=True)
class _Outcome:
    """Everything the stages before the persist step produced for one document."""

    doc: ParsedDocument
    records: list[ExtractedRecord]
    diagnostics: list[str]
    pages: list[dict[str, Any]]
    invoices: list[Invoice]
    own_issues: dict[UUID, list[ValidationIssue]]
    drafts: list[ChunkDraft]
    embedder: Embedder


@dataclass(slots=True)
class _Timings:
    started: float = field(default_factory=time.perf_counter)
    marks: dict[str, int] = field(default_factory=dict)

    def mark(self, stage: str, since: float) -> int:
        self.marks[stage] = int((time.perf_counter() - since) * 1000)
        return self.marks[stage]

    @property
    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)


# ------------------------------------------------------------------------------------------- the entry


def process_job(job_id: UUID, session_factory: SessionFactory = SessionLocal, *, attempt: int | None = None) -> None:
    """Run one attempt of job ``job_id`` (see the module docstring). Raises only ``TransientError``.

    ``attempt`` is set by a queue retrying a failed attempt (``TransientError.attempt``): a job still ``running``
    under that attempt number is taken over even though its heartbeat looks fresh.
    """
    settings = get_settings()
    timings = _Timings()
    try:
        claim = _claim(job_id, session_factory, settings, expected_attempt=attempt)
    except Exception as exc:
        if _is_transient_db_error(exc):
            raise TransientError(DATABASE_ERROR_MESSAGE, code=DATABASE_ERROR, attempt=attempt) from exc
        raise
    if claim is None:
        return
    logger.info("job %s document %s: attempt %d of %d started", job_id, claim.document_id, claim.attempt,
                claim.max_attempts)
    deadline = AttemptDeadline.after(settings.job_max_runtime_s)
    try:
        with Heartbeat(claim.job_id, claim.attempt, session_factory, deadline=deadline):
            _attempt(claim, session_factory, settings, timings, deadline)
    except TransientError as exc:
        exc.attempt = claim.attempt
        raise


class _Superseded(Exception):
    """The job is no longer running under this attempt (re-queued by the sweep, taken over, or deleted)."""


def _attempt(
    claim: _Claim, session_factory: SessionFactory, settings: Settings, timings: _Timings, deadline: AttemptDeadline
) -> None:
    """Run the claimed attempt and record how it ended (see "Errors" in the module docstring)."""
    try:
        _run(claim, session_factory, settings, timings, deadline)
    except _Superseded:
        logger.warning("job %s: attempt %d no longer owns the job; it stops here", claim.job_id, claim.attempt)
    except IngestionError as exc:
        if exc.permanent:
            _fail(claim, session_factory, code=exc.code, message=exc.message, timings=timings)
        else:
            _retry_or_fail(claim, session_factory, code=exc.code, message=exc.message, cause=exc, timings=timings)
    except TransientError as exc:
        _retry_or_fail(claim, session_factory, code=exc.code, message=str(exc), cause=exc, timings=timings)
    except Exception as exc:
        if _is_transient_db_error(exc):
            _retry_or_fail(
                claim, session_factory, code=DATABASE_ERROR, message=DATABASE_ERROR_MESSAGE, cause=exc, timings=timings
            )
            return
        logger.error(
            "job %s document %s: unexpected %s at\n%s",
            claim.job_id,
            claim.document_id,
            type(exc).__name__,
            _frames(exc),
        )
        _fail(claim, session_factory, code=INTERNAL_ERROR, message=INTERNAL_ERROR_MESSAGE, timings=timings)


def _frames(exc: BaseException) -> str:
    """The stack of ``exc`` as ``file:line in function`` - no exception message, no source lines (either may
    quote document text)."""
    return "\n".join(
        f"  {frame.filename}:{frame.lineno} in {frame.name}" for frame in traceback.extract_tb(exc.__traceback__)
    )


def _is_transient_db_error(exc: BaseException) -> bool:
    if isinstance(exc, sa_exc.OperationalError | sa_exc.TimeoutError | sa_exc.IntegrityError):
        return True
    return isinstance(exc, sa_exc.DBAPIError) and exc.connection_invalidated


# ------------------------------------------------------------------------------------------- the claim


def _claim(
    job_id: UUID, session_factory: SessionFactory, settings: Settings, *, expected_attempt: int | None = None
) -> _Claim | None:
    """Take the job for a new attempt, or return ``None`` when there is nothing to run.

    Runnable: a ``queued`` job; a ``running`` job whose heartbeat is stale (its worker is gone); a ``running`` job
    whose attempt is ``expected_attempt`` (the retry of an attempt that could not record its own failure). The take
    is a compare-and-set on (status, attempt), so two workers can never both win. A job with no attempts left is
    failed instead of run: ``worker_lost`` after a dead worker, ``database_error`` when the last attempt's outcome
    could not be recorded.
    """
    now = utcnow()
    stale_before = now - timedelta(seconds=settings.job_stale_after_s)
    with session_factory() as session:
        job = session.get(ProcessingJob, job_id)
        if job is None:
            logger.warning("job %s: not found; nothing to do", job_id)
            return None
        if job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
            logger.info("job %s: already %s; duplicate delivery ignored", job_id, job.status)
            return None
        own = job.status == JobStatus.RUNNING and expected_attempt is not None and job.attempt == expected_attempt
        stale = job.status == JobStatus.RUNNING and heartbeat_is_stale(job, now, settings.job_stale_after_s)
        if job.status == JobStatus.RUNNING and not (own or stale):
            logger.info("job %s: running on another worker; duplicate delivery ignored", job_id)
            return None
        observed = and_(
            ProcessingJob.id == job_id,
            ProcessingJob.status == job.status,
            ProcessingJob.attempt == job.attempt,
        )
        if job.status == JobStatus.RUNNING and not own:  # the heartbeat must still be stale when we take it
            observed = and_(
                observed, or_(ProcessingJob.heartbeat_at.is_(None), ProcessingJob.heartbeat_at < stale_before)
            )
        if job.attempt >= job.max_attempts:
            lost = JOB_TIMEOUT if _timed_out(job, settings) else WORKER_LOST
            _fail_exhausted(session, job, observed, code=DATABASE_ERROR if own else lost, settings=settings)
            return None
        taken = session.execute(
            update(ProcessingJob)
            .where(observed)
            .values(
                status=JobStatus.RUNNING,
                attempt=ProcessingJob.attempt + 1,
                stage=STAGE_PARSING,
                progress_current=0,
                progress_total=0,
                started_at=now,
                heartbeat_at=now,
                finished_at=None,
                error=None,
                worker_id=worker_identity(),
            )
            .execution_options(synchronize_session=False)
        )
        if taken.rowcount != 1:  # type: ignore[attr-defined]
            session.rollback()
            logger.info("job %s: claimed by another worker first", job_id)
            return None
        session.refresh(job)
        document = session.get(Document, job.document_id)
        if document is None:  # deleted while queued (the cascade normally removes the job too)
            session.rollback()
            return None
        document.status = DocumentStatus.PROCESSING
        add_event(
            session,
            document_id=document.id,
            job_id=job.id,
            stage=STAGE_START,
            message=f"Processing started (attempt {job.attempt} of {job.max_attempts})",
            data={"attempt": job.attempt, "max_attempts": job.max_attempts, "takeover": own or stale},
        )
        claim = _Claim(
            job_id=job.id,
            document_id=document.id,
            attempt=job.attempt,
            max_attempts=job.max_attempts,
            storage_key=document.storage_key,
            file_kind=document.file_kind,
            filename=document.filename,
            duplicate_of_id=document.duplicate_of_id,
            intake_warnings=tuple(w for w in (document.warnings or []) if w in INTAKE_WARNINGS),
        )
        session.commit()
    return claim


def _fail_exhausted(session: Session, job: ProcessingJob, observed: Any, *, code: str, settings: Settings) -> None:
    """Fail a job that would need more attempts than it has (compare-and-set on what ``_claim`` observed)."""
    claim = _Claim(job.id, job.document_id, job.attempt, job.max_attempts, "", "", "", None, ())
    locked = session.execute(
        update(ProcessingJob)
        .where(observed)
        .values(status=JobStatus.RUNNING)  # a no-op write that only succeeds if nobody changed the job meanwhile
        .execution_options(synchronize_session=False)
    )
    if locked.rowcount != 1:  # type: ignore[attr-defined]
        session.rollback()
        return None
    message = _exhausted_message(code, job.attempt, job.max_attempts, settings)
    session.expire(job)
    if _mark_failed(session, claim, code=code, message=message, processing_ms=None):
        session.commit()
        logger.warning("job %s: no attempts left; failed (%s)", job.id, code)
    else:
        session.rollback()
    return None


# ----------------------------------------------------------------------------------------- the stages


_LLM_SETTING_PREFIXES = ("llm_", "anthropic_", "gemini_")
_provider_lock = threading.Lock()
_provider_cache: tuple[tuple[Any, ...], LLMProvider | None] | None = None


def _llm_settings_key(settings: Settings) -> tuple[Any, ...]:
    """Every setting a provider can be built from (``LLM_*``, ``ANTHROPIC_*``, ``GEMINI_*``)."""
    return tuple(
        (name, getattr(settings, name))
        for name in type(settings).model_fields
        if name.startswith(_LLM_SETTING_PREFIXES)
    )


def _llm_provider(settings: Settings) -> LLMProvider | None:
    """The LLM provider for ``settings``: built once per effective configuration and shared by every job of the
    process (one HTTP client with keep-alive instead of a new client per job). A configuration error is logged and
    not cached, so a fixed configuration is picked up by the next job."""
    global _provider_cache
    if settings.extraction_mode == "rules":
        return None
    key = _llm_settings_key(settings)
    with _provider_lock:
        if _provider_cache is not None and _provider_cache[0] == key:
            return _provider_cache[1]
        try:
            provider = get_llm_provider(settings)
        except (LLMConfigurationError, ImportError) as exc:
            logger.error("LLM provider unavailable (%s); extraction falls back to rules", type(exc).__name__)
            return None
        # A replaced provider is not closed here: a job that is still running may be using it.
        _provider_cache = (key, provider)
        return provider


def close_llm_provider() -> None:
    """Close the cached provider's HTTP client (worker / API shutdown). Best effort; the next job builds a new one."""
    global _provider_cache
    with _provider_lock:
        cached, _provider_cache = _provider_cache, None
    provider = cached[1] if cached is not None else None
    if provider is None:
        return
    closer = getattr(provider, "close", None) or getattr(getattr(provider, "_client", None), "close", None)
    if callable(closer):
        try:
            closer()
        except Exception as exc:  # shutting down: never let a client's close raise
            logger.warning("closing the LLM client failed (%s)", type(exc).__name__)


def _ensure_owner(claim: _Claim, session_factory: SessionFactory) -> None:
    """Raise ``_Superseded`` when the job is no longer running under this attempt (checked between stages, so an
    attempt that the sweep reclaimed after its deadline - or that another worker took over - stops early instead of
    running to the end only to have its results discarded)."""
    with session_factory() as session:
        owner = session.execute(
            select(ProcessingJob.status, ProcessingJob.attempt).where(ProcessingJob.id == claim.job_id)
        ).one_or_none()
    if owner is None or owner.status != JobStatus.RUNNING or owner.attempt != claim.attempt:
        raise _Superseded()


def _run(
    claim: _Claim, session_factory: SessionFactory, settings: Settings, timings: _Timings, deadline: AttemptDeadline
) -> None:
    """Parse, extract, validate and embed outside any transaction, then persist in one.

    Memory: the parsed document (every page's text and every positioned span of its layout), the records and the
    chunk vectors are all held until the persist step commits. Measured on the 1000-page evaluation PDF: the parsed
    document (28,587 lines, 190,695 spans) holds about 45 MiB, and the vectors of its 1,506 chunks (Python float
    lists) about 18 MiB more. Both grow linearly with pages (``MAX_PDF_PAGES`` bounds them) and a worker runs
    ``--concurrency`` such jobs at once. OCR page images are rendered inside each OCR job and dropped after it; only
    the recognised text and boxes come back.
    """
    reporter = ProgressReporter(claim.job_id, claim.attempt, session_factory, deadline=deadline)

    path = get_storage().path(claim.storage_key)
    if not path.is_file():
        raise IngestionError("the stored file is missing", code=FILE_MISSING)
    since = time.perf_counter()
    doc = parse_document(path, FileKind(claim.file_kind), on_progress=reporter.on_progress)
    reporter.flush()
    _ensure_owner(claim, session_factory)
    failed_pages = sum(1 for page in doc.pages if page.method == "none")
    reporter.event(
        claim.document_id,
        STAGE_PARSING,
        f"Parsed {doc.page_count} {'sheet' if doc.sheets else 'page'}(s), {doc.ocr_page_count} by OCR",
        data={
            "pages": len(doc.pages),
            "sheets": len(doc.sheets),
            "ocr_pages": doc.ocr_page_count,
            "failed_pages": failed_pages,
            "warnings": len(doc.warnings),
            "duration_ms": timings.mark(STAGE_PARSING, since),
        },
    )

    reporter.set_stage(STAGE_EXTRACTING)
    since = time.perf_counter()
    provider = _llm_provider(settings)
    diagnostics: list[str] = []
    records = extract_records(
        doc,
        mode=settings.extraction_mode,
        provider=provider,
        settings=settings,
        on_progress=reporter.on_progress,
        diagnostics=diagnostics,
    )
    reporter.flush()
    _ensure_owner(claim, session_factory)
    methods = Counter(record.result.method for record in records)
    reporter.event(
        claim.document_id,
        STAGE_EXTRACTING,
        f"Extracted {len(records)} record(s)",
        data={
            "records": len(records),
            "methods": dict(methods),
            "mode": settings.extraction_mode,
            "llm_provider": provider.name if provider is not None else None,
            "llm_fallbacks": _fallback_records(records),
            "duration_ms": timings.mark(STAGE_EXTRACTING, since),
        },
    )

    # Validation of each record's own fields runs here; the checks against other documents (duplicates,
    # cross-verification) need the database and run in the persist transaction.
    reporter.set_stage(STAGE_VALIDATING, current=0, total=len(records))
    since = time.perf_counter()
    pages = _page_rows(claim.document_id, doc)
    invoices = [_invoice_row(claim.document_id, index, record) for index, record in enumerate(records)]
    own_issues = {invoice.id: own_record_issues(invoice) for invoice in invoices}
    timings.mark(STAGE_VALIDATING, since)

    reporter.set_stage(STAGE_INDEXING)
    since = time.perf_counter()
    drafts = _chunk_drafts(pages, invoices, claim.filename)
    embedder = _PrecomputedEmbedder(get_embedder(), [draft.text for draft in drafts], reporter.on_progress)
    reporter.flush()
    timings.mark(STAGE_INDEXING, since)

    outcome = _Outcome(doc, records, diagnostics, pages, invoices, own_issues, drafts, embedder)
    with session_factory() as session:
        if _persist(session, claim, outcome, settings, timings):
            session.commit()
        else:
            session.rollback()


def _page_rows(document_id: UUID, doc: ParsedDocument) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in doc.pages:
        text = page.text.replace("\x00", "")
        rows.append(
            {
                "document_id": document_id,
                "page_number": page.number,
                "label": None,
                "method": page.method,
                "text": text,
                "ocr_confidence": page.ocr_confidence,
                "char_count": len(text),
            }
        )
    for sheet in doc.sheets:
        text = sheet.text.replace("\x00", "")
        rows.append(
            {
                "document_id": document_id,
                "page_number": sheet.index + 1,
                "label": sheet.name[:255],
                "method": "sheet",
                "text": text,
                "ocr_confidence": None,
                "char_count": len(text),
            }
        )
    return rows


def _clean_text(value: str | None, limit: int | None = None) -> str | None:
    """No NUL (PostgreSQL rejects it in text and JSON) and no more characters than the column holds."""
    if value is None:
        return None
    value = value.replace("\x00", "")
    return value[:limit] if limit is not None else value


def _clean_data(data: InvoiceData) -> InvoiceData:
    update_values: dict[str, Any] = {
        name: _clean_text(getattr(data, name), limit) for name, limit in _TEXT_LIMITS.items()
    }
    update_values["line_items"] = [
        item.model_copy(update={"description": _clean_text(item.description)}) for item in data.line_items
    ]
    return data.model_copy(update=update_values)


def _clean_meta(meta: dict[str, FieldMeta]) -> dict[str, FieldMeta]:
    return {name: m.model_copy(update={"evidence": _clean_text(m.evidence)}) for name, m in meta.items()}


def _invoice_row(document_id: UUID, index: int, record: ExtractedRecord) -> Invoice:
    """The ``Invoice`` of one record, with its id assigned up front (chunk drafts refer to it before the flush)."""
    result = record.result
    invoice = Invoice(
        id=uuid.uuid4(),
        document_id=document_id,
        record_index=index,
        source_kind=record.source_kind.value,
        page_start=record.page_start,
        page_end=record.page_end,
        sheet_name=_clean_text(record.sheet_name, 255),
        row_number=record.row_number,
        extraction_method=result.method[:20],
        overall_confidence=result.overall_confidence,
    )
    invoice.apply_invoice_data(_clean_data(result.data))
    invoice.set_field_meta(_clean_meta(result.meta))
    return invoice


def _chunk_drafts(pages: Sequence[dict[str, Any]], invoices: Sequence[Invoice], filename: str) -> list[ChunkDraft]:
    pairs = [(invoice.id, invoice.to_invoice_data()) for invoice in invoices]
    if any(invoice_id is None for invoice_id, _ in pairs):  # a None id would turn a summary into a document hit
        raise RuntimeError("invoice ids must be assigned before chunking")
    return build_chunks([(page["page_number"], page["text"]) for page in pages], pairs, filename)


class _PrecomputedEmbedder:
    """Embeds the chunk texts up front (outside the database transaction); ``index_document`` then gets the
    vectors from memory. Unknown texts fall through to the wrapped embedder.

    Each distinct text is embedded once, shortest first, so every batch holds texts of similar length (an embedding
    batch costs as much as its longest text, see ``FastEmbedEmbedder``); ``on_progress`` hears about every batch."""

    def __init__(self, base: Embedder, texts: Sequence[str], on_progress: ProgressCallback | None = None) -> None:
        self.name = base.name
        self.dim = base.dim
        self._base = base
        self._vectors: dict[str, list[float]] = {}
        unique = sorted(dict.fromkeys(texts), key=len)
        for start in range(0, len(unique), _EMBED_BATCH):
            batch = unique[start : start + _EMBED_BATCH]
            self._vectors.update(zip(batch, base.embed_documents(batch), strict=True))
            if on_progress is not None:
                on_progress(start + len(batch), len(unique), STAGE_INDEXING)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        missing = [text for text in dict.fromkeys(texts) if text not in self._vectors]
        if missing:
            self._vectors.update(zip(missing, self._base.embed_documents(missing), strict=True))
        return [self._vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._base.embed_query(text)


# ------------------------------------------------------------------------------------------ the persist


def _fallback_records(records: Sequence[ExtractedRecord]) -> int:
    """Records whose LLM read was unavailable, failed or refused, so the rules result was used."""
    return sum(1 for record in records if LLM_FALLBACK_WARNINGS.intersection(record.result.warnings))


def _document_issues(
    claim: _Claim, doc: ParsedDocument, records: Sequence[ExtractedRecord], settings: Settings
) -> list[ValidationIssue]:
    """Document-level issues; the ``LLM_UNAVAILABLE`` note also carries how many records fell back to the rules
    (``details.records``), which ``GET /stats`` sums."""
    confidences = [p.ocr_confidence for p in doc.pages if p.method == "ocr" and p.ocr_confidence is not None]
    issues = document_level_issues(
        ocr_confidence=round(sum(confidences) / len(confidences), 4) if confidences else None,
        duplicate_of_id=claim.duplicate_of_id,
        extraction_warnings=[w for record in records for w in record.result.warnings],
        ocr_threshold=settings.ocr_low_confidence,
    )
    issues = [
        issue.model_copy(update={"details": {**issue.details, "records": _fallback_records(records)}})
        if issue.code == IssueCode.LLM_UNAVAILABLE
        else issue
        for issue in issues
    ]
    if not records:
        issues.append(
            ValidationIssue(
                code=IssueCode.MISSING_FIELD,
                severity=Severity.WARNING,
                message="No invoice data was found in the document",
                details={"reason": "no_records"},
            )
        )
    return issues


def _document_warnings(claim: _Claim, outcome: _Outcome) -> list[str]:
    codes = [
        *claim.intake_warnings,
        *outcome.doc.warnings,
        *outcome.diagnostics,
        *(w for record in outcome.records for w in record.result.warnings),
    ]
    unique = list(dict.fromkeys(codes))
    if len(unique) > MAX_DOCUMENT_WARNINGS:
        unique = [*unique[: MAX_DOCUMENT_WARNINGS - 1], "more_warnings_truncated"]
    return unique


def _llm_identity(results: Sequence[ExtractionResult]) -> tuple[str | None, str | None, str | None]:
    for result in results:
        if result.llm_provider:
            model = (result.llm_model or "")[:100] or None
            return result.llm_provider[:50], model, (result.prompt_version or "")[:40] or None
    return None, None, None


def _persist(session: Session, claim: _Claim, outcome: _Outcome, settings: Settings, timings: _Timings) -> bool:
    """Replace the document's results and finish the job in ``session``'s transaction; False = abandon (the
    document was deleted, or another attempt took the job over)."""
    doc, records, pages, invoices = outcome.doc, outcome.records, outcome.pages, outcome.invoices
    serialize_reconciliation(session)
    document = session.get(Document, claim.document_id)
    job = session.get(ProcessingJob, claim.job_id)
    if document is None or job is None:
        logger.info("job %s: document %s was deleted during processing", claim.job_id, claim.document_id)
        return False
    if job.status != JobStatus.RUNNING or job.attempt != claim.attempt:
        logger.warning("job %s: attempt %d was taken over; its results are discarded", claim.job_id, claim.attempt)
        return False
    since = time.perf_counter()
    document_id = claim.document_id

    old_numbers = {
        number
        for number in session.scalars(select(Invoice.invoice_number_norm).where(Invoice.document_id == document_id))
        if number
    }
    for model in (ValidationIssueRow, Chunk, Invoice, DocumentPage):  # cascades: issues, chunks, cross-verifications
        session.execute(
            delete(model).where(model.document_id == document_id).execution_options(synchronize_session=False)
        )
    for start in range(0, len(pages), _INSERT_BATCH):
        session.execute(insert(DocumentPage), list(pages[start : start + _INSERT_BATCH]))
    session.add_all(invoices)
    session.flush()

    summary = reconcile_invoices(session, invoices, extra_numbers=old_numbers, own_issues=outcome.own_issues)
    document_issues = _document_issues(claim, doc, records, settings)
    issue_rows = record_issue_rows(document_issues, document_id=document_id, invoice_id=None)
    if issue_rows:
        session.execute(insert(ValidationIssueRow), issue_rows)
    validating_ms = timings.marks.get(STAGE_VALIDATING, 0) + int((time.perf_counter() - since) * 1000)
    timings.marks[STAGE_VALIDATING] = validating_ms

    since = time.perf_counter()
    chunk_count = index_document(session, document_id, outcome.drafts, outcome.embedder)
    indexing_ms = timings.marks.get(STAGE_INDEXING, 0) + int((time.perf_counter() - since) * 1000)

    now = utcnow()
    provider, llm_model, prompt_version = _llm_identity([record.result for record in records])
    methods = Counter(record.result.method for record in records)
    document.page_count = doc.page_count
    document.ocr_page_count = doc.ocr_page_count
    document.extraction_method = methods.most_common(1)[0][0][:20] if methods else None
    document.llm_provider, document.llm_model, document.prompt_version = provider, llm_model, prompt_version
    document.warnings = _document_warnings(claim, outcome)
    document.error_code = None
    document.error_message = None
    document.processed_at = now
    document.status = DocumentStatus.COMPLETED
    refresh_document_status(session, document)
    for other_id in sorted(summary.document_ids - {document_id}, key=str):
        other = session.get(Document, other_id)
        if other is not None:
            refresh_document_status(session, other)

    counts = dict(summary.issues)
    for issue in document_issues:
        counts[issue.severity.value] += 1
    add_event(
        session,
        document_id=document_id,
        job_id=claim.job_id,
        stage=STAGE_VALIDATING,
        message=f"Validation: {counts['error']} error(s), {counts['warning']} warning(s), {counts['info']} note(s)",
        data={
            "issues": counts,
            "document_issues": len(document_issues),
            "records_revalidated": summary.records,
            "comparisons": summary.comparisons,
            "mismatches": summary.mismatches,
            "duplicates": summary.duplicates,
            "other_documents": len(summary.document_ids - {document_id}),
            "duration_ms": validating_ms,
        },
    )
    add_event(
        session,
        document_id=document_id,
        job_id=claim.job_id,
        stage=STAGE_INDEXING,
        message=f"Indexed {chunk_count} search chunk(s)",
        data={"chunks": chunk_count, "duration_ms": indexing_ms},
    )
    job = session.scalars(
        select(ProcessingJob)
        .where(ProcessingJob.id == claim.job_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    ).one_or_none()
    if job is None or job.status != JobStatus.RUNNING or job.attempt != claim.attempt:
        logger.warning(
            "job %s: attempt %d was taken over while it ran; its results are discarded", claim.job_id, claim.attempt
        )
        return False
    document.processing_ms = timings.elapsed_ms
    job.status = JobStatus.SUCCEEDED
    job.stage = STAGE_DONE
    job.finished_at = now
    job.heartbeat_at = now
    job.error = None
    add_event(
        session,
        document_id=document_id,
        job_id=claim.job_id,
        stage=STAGE_DONE,
        message=f"Processing finished: {document.status}",
        data={
            "status": document.status,
            "records": len(records),
            "processing_ms": document.processing_ms,
            "stage_ms": dict(timings.marks),
        },
    )
    logger.info(
        "job %s document %s: %s, %d record(s), %d chunk(s), %d ms",
        claim.job_id,
        document_id,
        document.status,
        len(records),
        chunk_count,
        document.processing_ms,
    )
    return True


# ------------------------------------------------------------------------------------------ the failures


def _fail(claim: _Claim, session_factory: SessionFactory, *, code: str, message: str, timings: _Timings) -> None:
    """Record a final failure of the document and the job; database trouble here becomes ``TransientError``."""
    try:
        with session_factory() as session:
            _mark_failed(session, claim, code=code, message=message, processing_ms=timings.elapsed_ms)
            session.commit()
    except Exception as exc:
        if _is_transient_db_error(exc):
            raise TransientError(DATABASE_ERROR_MESSAGE, code=DATABASE_ERROR) from exc
        raise
    logger.warning("job %s document %s: failed (%s)", claim.job_id, claim.document_id, code)


def _mark_failed(session: Session, claim: _Claim, *, code: str, message: str, processing_ms: int | None) -> bool:
    """Fail the job and its document - only if the job is still unfinished under ``claim``'s attempt (a job taken
    over by a later attempt is not ours to fail). Returns whether anything was marked."""
    now = utcnow()
    job = session.get(ProcessingJob, claim.job_id, with_for_update=True)
    if job is None or job.attempt != claim.attempt or job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
        return False
    document = session.get(Document, claim.document_id)
    job.status = JobStatus.FAILED
    job.finished_at = now
    job.heartbeat_at = now
    job.error = f"{code}: {message}"
    if document is not None:
        document.status = DocumentStatus.FAILED
        document.error_code = code[:50]
        document.error_message = message
        document.processing_ms = processing_ms
        add_event(
            session,
            document_id=document.id,
            job_id=claim.job_id,
            stage=STAGE_ERROR,
            level="error",
            message=f"Processing failed: {message}",
            data={"error_code": code, "attempt": claim.attempt, "max_attempts": claim.max_attempts},
        )
    return True


def _retry_or_fail(
    claim: _Claim,
    session_factory: SessionFactory,
    *,
    code: str,
    message: str,
    cause: BaseException,
    timings: _Timings,
) -> None:
    """Re-queue the job and raise ``TransientError`` while attempts remain; fail it after the last one."""
    final = claim.attempt >= claim.max_attempts
    try:
        with session_factory() as session:
            job = session.get(ProcessingJob, claim.job_id)
            document = session.get(Document, claim.document_id)
            if job is None or document is None or job.status != JobStatus.RUNNING or job.attempt != claim.attempt:
                return  # deleted, or another attempt owns the job now
            if final:
                _mark_failed(
                    session,
                    claim,
                    code=code,
                    message=f"{message} (gave up after {claim.attempt} attempts)",
                    processing_ms=timings.elapsed_ms,
                )
            else:
                job.status = JobStatus.QUEUED
                job.error = f"{code}: {message}"
                job.heartbeat_at = utcnow()
                document.status = DocumentStatus.QUEUED
                add_event(
                    session,
                    document_id=document.id,
                    job_id=job.id,
                    stage=STAGE_RETRY,
                    level="warning",
                    message=f"Attempt {claim.attempt} of {claim.max_attempts} failed ({code}); it will be retried",
                    data={"error_code": code, "attempt": claim.attempt, "max_attempts": claim.max_attempts},
                )
            session.commit()
    except Exception as exc:
        if _is_transient_db_error(exc):
            raise TransientError(DATABASE_ERROR_MESSAGE, code=DATABASE_ERROR) from exc
        raise
    logger.warning(
        "job %s document %s: attempt %d failed (%s, %s)%s",
        claim.job_id,
        claim.document_id,
        claim.attempt,
        code,
        type(cause).__name__,
        "; giving up" if final else "; will retry",
    )
    if not final:
        raise TransientError(message, code=code) from cause


# ------------------------------------------------------------------------------------- stale recovery


@dataclass(slots=True)
class StaleJobs:
    requeued: list[UUID] = field(default_factory=list)  # back to ``queued``: enqueue these
    failed: list[UUID] = field(default_factory=list)  # out of attempts


def _stale_running(cutoff: datetime) -> Any:
    """SQL: a ``running`` job whose last sign of life (heartbeat, else start) is older than ``cutoff``."""
    return and_(
        ProcessingJob.status == JobStatus.RUNNING,
        or_(
            ProcessingJob.heartbeat_at < cutoff,
            and_(
                ProcessingJob.heartbeat_at.is_(None),
                or_(ProcessingJob.started_at.is_(None), ProcessingJob.started_at < cutoff),
            ),
        ),
    )


def _timed_out(job: ProcessingJob, settings: Settings) -> bool:
    """Whether a stale attempt went silent because it outlived ``JOB_MAX_RUNTIME_S`` rather than because its worker
    died: the heartbeat stops at the deadline, so the last one lands about ``JOB_MAX_RUNTIME_S`` after the start
    (within a beat or two), while a dead worker's last heartbeat is whenever it died."""
    started, last = as_utc(job.started_at), as_utc(job.heartbeat_at)
    if started is None or last is None:
        return False
    margin = timedelta(seconds=2 * progress.HEARTBEAT_INTERVAL_S)
    return last - started >= timedelta(seconds=settings.job_max_runtime_s) - margin


def reclaim_stale_jobs(session: Session, *, now: datetime | None = None) -> StaleJobs:
    """Running jobs without a heartbeat for ``JOB_STALE_AFTER_S`` - their worker died, or the attempt outlived
    ``JOB_MAX_RUNTIME_S`` and its heartbeat stopped (``job_timeout``) - are re-queued while attempts remain, else
    failed (``worker_lost`` / ``job_timeout``). Each change is a compare-and-set on what was read (still running,
    same attempt, still stale), so a takeover or a heartbeat landing in between is never undone and no attempt is
    burnt. Does not commit or enqueue; the caller enqueues ``requeued``."""
    settings = get_settings()
    now = now or utcnow()
    cutoff = now - timedelta(seconds=settings.job_stale_after_s)
    jobs = session.scalars(select(ProcessingJob).where(_stale_running(cutoff)).order_by(ProcessingJob.created_at)).all()
    stale = StaleJobs()
    for job in jobs:
        job_id, document_id, attempt, max_attempts = job.id, job.document_id, job.attempt, job.max_attempts
        timed_out = _timed_out(job, settings)
        code = JOB_TIMEOUT if timed_out else WORKER_LOST
        observed = and_(ProcessingJob.id == job_id, ProcessingJob.attempt == attempt, _stale_running(cutoff))
        if attempt < max_attempts:
            why = (
                f"no heartbeat after the attempt exceeded JOB_MAX_RUNTIME_S ({settings.job_max_runtime_s} s)"
                if timed_out
                else "no heartbeat since the worker stopped"
            )
            values: dict[str, Any] = {
                "status": JobStatus.QUEUED,
                "heartbeat_at": now,  # for a queued job: when it was last handed to the queue
                "error": f"{code}: {why}; re-queued",
            }
        else:
            values = {"status": JobStatus.RUNNING}  # a no-op write that only succeeds if nobody changed the job
        changed = session.execute(
            update(ProcessingJob).where(observed).values(**values).execution_options(synchronize_session=False)
        )
        session.expire(job)
        if changed.rowcount != 1:  # type: ignore[attr-defined]
            continue  # taken over, finished or alive again since we read it
        if attempt < max_attempts:
            document = session.get(Document, document_id)
            if document is not None:
                document.status = DocumentStatus.QUEUED
                lost = "Attempt timed out" if timed_out else "Worker lost"
                add_event(
                    session,
                    document_id=document.id,
                    job_id=job_id,
                    stage=STAGE_RECOVERED,
                    level="warning",
                    message=f"{lost} during attempt {attempt} of {max_attempts}; job re-queued",
                    data={"attempt": attempt, "max_attempts": max_attempts, "error_code": code},
                )
            stale.requeued.append(job_id)
        else:
            claim = _Claim(job_id, document_id, attempt, max_attempts, "", "", "", None, ())
            _mark_failed(
                session,
                claim,
                code=code,
                message=_exhausted_message(code, attempt, max_attempts, settings),
                processing_ms=None,
            )
            stale.failed.append(job_id)
    session.flush()
    if stale.requeued or stale.failed:
        logger.warning("stale jobs: %d re-queued, %d failed", len(stale.requeued), len(stale.failed))
    return stale


def _exhausted_message(code: str, attempt: int, max_attempts: int, settings: Settings) -> str:
    reason = {
        DATABASE_ERROR: "The outcome of the last attempt could not be recorded (database unavailable)",
        JOB_TIMEOUT: f"The last attempt ran longer than JOB_MAX_RUNTIME_S ({settings.job_max_runtime_s} s)",
    }.get(code, "The worker stopped during the last attempt")
    return f"{reason} ({attempt} of {max_attempts} attempts used)"


def recover_stale_jobs(session: Session, *, now: datetime | None = None) -> int:
    """``reclaim_stale_jobs`` returning how many stale jobs were handled (re-queued + failed)."""
    stale = reclaim_stale_jobs(session, now=now)
    return len(stale.requeued) + len(stale.failed)


def sweep_jobs(session: Session, *, now: datetime | None = None) -> list[UUID]:
    """The periodic safety net: reclaim stale running jobs, and pick ``queued`` jobs that have not been handed to
    the queue for ``QUEUED_RESEND_AFTER_S`` (a lost broker message, a failed enqueue). Returns the ids to (re)send,
    stamped as sent now; does not commit. Sending a job twice is harmless - the claim lets only one delivery run."""
    now = now or utcnow()
    ids = list(reclaim_stale_jobs(session, now=now).requeued)
    cutoff = now - timedelta(seconds=QUEUED_RESEND_AFTER_S)
    waiting = session.scalars(
        select(ProcessingJob)
        .where(
            ProcessingJob.status == JobStatus.QUEUED,
            or_(
                ProcessingJob.heartbeat_at < cutoff,
                and_(ProcessingJob.heartbeat_at.is_(None), ProcessingJob.created_at < cutoff),
            ),
        )
        .order_by(ProcessingJob.created_at)
    ).all()
    for job in waiting:
        since = queued_since(job)
        if job.id not in ids and since is not None and since < cutoff:
            job.heartbeat_at = now
            ids.append(job.id)
    session.flush()
    if waiting:
        logger.warning("sweep: re-sending %d queued job(s) that waited over %d s", len(waiting), QUEUED_RESEND_AFTER_S)
    return ids


def abandon_job(job_id: UUID, *, session_factory: SessionFactory = SessionLocal) -> None:
    """Fail a job the queue has stopped retrying (its attempts kept failing before they could even be recorded,
    e.g. the database was down). Best effort: if the database is still unreachable this only logs."""
    try:
        with session_factory() as session:
            job = session.get(ProcessingJob, job_id)
            if job is None or job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
                return
            claim = _Claim(job.id, job.document_id, job.attempt, job.max_attempts, "", "", "", None, ())
            _mark_failed(session, claim, code=DATABASE_ERROR, message=f"{DATABASE_ERROR_MESSAGE}; retries exhausted",
                         processing_ms=None)
            session.commit()
    except Exception as exc:
        logger.error("job %s: retries exhausted and the failure could not be recorded (%s)", job_id,
                     type(exc).__name__)

