"""Documents: multi-file upload, list/filter, detail, status, results, pages, events, original file, reprocess and
delete."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID

import structlog
from fastapi import APIRouter, Query, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session
from starlette.datastructures import UploadFile

from app.api.deps import (
    API_PREFIX,
    JobQueue,
    Limit,
    ListLimit,
    Offset,
    QueueDep,
    SessionDep,
    SettingsDep,
    StorageDep,
    enqueue_job,
    guard,
)
from app.api.errors import AlreadyProcessingError, NotFoundError, ValidationFailedError
from app.api.queries import contains, sort_order
from app.api.schemas import (
    DocumentDetailOut,
    DocumentList,
    DocumentResultsOut,
    DocumentStatusOut,
    DocumentStatusValue,
    ErrorResponse,
    EventList,
    EventOut,
    FileKindValue,
    PageList,
    PageOut,
    QueryText,
    ReprocessResponse,
    UploadError,
    UploadItem,
    UploadResponse,
)
from app.api.serializers import document_detail_out, documents_out, invoices_out, job_out, latest_jobs
from app.config import Settings
from app.db.base import utcnow
from app.db.models import (
    Document,
    DocumentPage,
    DocumentStatus,
    Invoice,
    JobStatus,
    ProcessingEvent,
    ProcessingJob,
    ValidationIssueRow,
)
from app.domain.schemas import IssueCode
from app.ingestion.types import FileTooLargeError
from app.logging_setup import exception_summary
from app.pipeline import intake
from app.pipeline.progress import heartbeat_is_stale
from app.pipeline.review import reconcile_invoices, refresh_document_status, serialize_reconciliation
from app.security.uploads import (
    UPLOAD_FIELD,
    UploadRejectedError,
    content_disposition,
    inspect_upload,
    read_upload_form,
    size_limit_message,
    too_many_files,
)
from app.storage import Storage

logger = structlog.get_logger(__name__)

DOCUMENTS = f"{API_PREFIX}/documents"
upload_router = APIRouter(prefix=DOCUMENTS, tags=["documents"], dependencies=guard("upload"))
router = APIRouter(prefix=DOCUMENTS, tags=["documents"], dependencies=guard("default"))

DocumentSort = Literal[
    "created_at", "-created_at", "updated_at", "-updated_at", "processed_at", "-processed_at",
    "filename", "-filename", "status", "-status", "size_bytes", "-size_bytes", "processing_ms", "-processing_ms",
]  # fmt: skip
_NOT_FOUND: dict[int | str, dict[str, Any]] = {404: {"model": ErrorResponse, "description": "No such document"}}
_UPLOAD_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": [UPLOAD_FIELD],
                    "properties": {
                        UPLOAD_FIELD: {"type": "array", "items": {"type": "string", "format": "binary"}},
                    },
                }
            }
        },
    }
}


def get_document_or_404(session: Session, document_id: UUID) -> Document:
    document = session.get(Document, document_id)
    if document is None:
        raise NotFoundError("Document not found")
    return document


# ------------------------------------------------------------------------------------------------ upload


@upload_router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload documents",
    responses={
        413: {"model": ErrorResponse, "description": "The request is larger than the upload limits"},
        415: {"model": ErrorResponse, "description": "The body is not multipart/form-data"},
        422: {"model": ErrorResponse, "description": "No files, or more than MAX_FILES_PER_REQUEST"},
    },
    openapi_extra=_UPLOAD_BODY,
)
async def upload_documents(
    request: Request, session: SessionDep, settings: SettingsDep, storage: StorageDep, queue: QueueDep
) -> UploadResponse:
    """Accept 1..``MAX_FILES_PER_REQUEST`` files in the multipart field ``files``.

    Each file is identified by its content (PDF, PNG/JPEG/TIFF, XLSX/XLS, CSV), stored under a random name and
    queued for processing. The response has one item per file, in request order: ``queued`` with the new document
    and job ids (``duplicate_of_id`` when the same bytes were uploaded before), or ``rejected`` with an error code
    (``unsupported_file_type``, ``empty_file``, ``file_too_large``). Rejected files create nothing.
    """
    form = await read_upload_form(request, settings)
    try:
        uploads = form.getlist(UPLOAD_FIELD)
        if not uploads or not all(isinstance(upload, UploadFile) for upload in uploads):
            raise ValidationFailedError(
                f"Send one or more files in the multipart field '{UPLOAD_FIELD}'",
                details={"fields": {UPLOAD_FIELD: "at least one file is required"}},
            )
        if len(uploads) > settings.max_files_per_request:
            raise too_many_files(settings, received=len(uploads))
        files = [upload for upload in uploads if isinstance(upload, UploadFile)]
        items = await run_in_threadpool(_ingest_all, files, session, storage, queue, settings)
    finally:
        await form.close()
    return UploadResponse(items=items)


def _ingest_all(
    uploads: list[UploadFile], session: Session, storage: Storage, queue: JobQueue, settings: Settings
) -> list[UploadItem]:
    return [_ingest(upload, session, storage, queue, settings) for upload in uploads]


def _rejected(filename: str, code: str, message: str) -> UploadItem:
    return UploadItem(filename=filename, status="rejected", error=UploadError(code=code, message=message))


def _ingest(upload: UploadFile, session: Session, storage: Storage, queue: JobQueue, settings: Settings) -> UploadItem:
    original_name = upload.filename or ""
    filename = intake.sanitize_filename(original_name)
    try:
        inspection = inspect_upload(upload.file, original_name, size=upload.size, max_bytes=settings.max_upload_bytes)
        stored = storage.save(upload.file, suffix=inspection.storage_suffix, max_bytes=settings.max_upload_bytes)
    except UploadRejectedError as exc:
        return _rejected(filename, exc.code, exc.message)
    except FileTooLargeError:
        return _rejected(filename, "file_too_large", size_limit_message(settings.max_upload_bytes))
    except OSError as exc:
        logger.error("upload_not_stored", **exception_summary(exc))
        return _rejected(filename, "internal_error", "The file could not be stored; try again")

    try:
        result = intake.create_document(
            session, filename=filename, stored=stored, kind=inspection.kind, content_type=inspection.content_type
        )
        session.commit()
    except Exception as exc:
        session.rollback()
        storage.delete(stored.key)
        logger.error("upload_not_saved", stored_key=stored.key, **exception_summary(exc))
        return _rejected(filename, "internal_error", "The file could not be saved; try again")

    enqueue_job(queue, result.job.id)
    return UploadItem(
        filename=filename,
        status="queued",
        document_id=result.document.id,
        job_id=result.job.id,
        duplicate_of_id=result.duplicate_of_id,
    )


# ------------------------------------------------------------------------------------------------ reads


@router.get("", summary="List documents")
def list_documents(
    session: SessionDep,
    status_: Annotated[list[DocumentStatusValue] | None, Query(alias="status")] = None,
    file_kind: FileKindValue | None = None,
    q: Annotated[QueryText, Query(max_length=200, description="Filename, invoice number or supplier contains")] = None,
    needs_review: bool | None = None,
    limit: Limit = 20,
    offset: Offset = 0,
    sort: DocumentSort = "-created_at",
) -> DocumentList:
    statement = select(Document)
    if status_:
        statement = statement.where(Document.status.in_(status_))
    if file_kind:
        statement = statement.where(Document.file_kind == file_kind)
    if needs_review is not None:
        is_review = Document.status == DocumentStatus.NEEDS_REVIEW
        statement = statement.where(is_review if needs_review else ~is_review)
    if q and q.strip():
        term = q.strip()
        record_matches = (
            select(Invoice.id)
            .where(
                Invoice.document_id == Document.id,
                or_(contains(Invoice.invoice_number, term), contains(Invoice.supplier_name, term)),
            )
            .exists()
        )
        statement = statement.where(or_(contains(Document.filename, term), record_matches))
    total = session.scalar(select(func.count()).select_from(statement.subquery())) or 0
    documents = session.scalars(
        statement.order_by(sort_order(Document, sort), Document.id).limit(limit).offset(offset)
    ).all()
    return DocumentList(items=documents_out(session, documents), total=total, limit=limit, offset=offset)


@router.get("/{document_id}", summary="Get a document with its records and issues", responses=_NOT_FOUND)
def get_document(document_id: UUID, session: SessionDep) -> DocumentDetailOut:
    return document_detail_out(session, get_document_or_404(session, document_id))


@router.get("/{document_id}/status", summary="Processing status (for polling)", responses=_NOT_FOUND)
def get_document_status(document_id: UUID, session: SessionDep) -> DocumentStatusOut:
    document = get_document_or_404(session, document_id)
    job = latest_jobs(session, [document.id]).get(document.id)
    return DocumentStatusOut(id=document.id, status=document.status, job=job_out(job), updated_at=document.updated_at)


@router.get("/{document_id}/results", summary="Extracted records", responses=_NOT_FOUND)
def get_document_results(document_id: UUID, session: SessionDep) -> DocumentResultsOut:
    document = get_document_or_404(session, document_id)
    invoices = session.scalars(
        select(Invoice).where(Invoice.document_id == document.id).order_by(Invoice.record_index, Invoice.id)
    ).all()
    return DocumentResultsOut(
        document_id=document.id,
        status=document.status,
        invoices=invoices_out(session, [(invoice, document.filename) for invoice in invoices]),
    )


@router.get("/{document_id}/pages", summary="Extracted text per page or sheet", responses=_NOT_FOUND)
def get_document_pages(document_id: UUID, session: SessionDep, limit: Limit = 20, offset: Offset = 0) -> PageList:
    document = get_document_or_404(session, document_id)
    condition = DocumentPage.document_id == document.id
    total = session.scalar(select(func.count()).select_from(DocumentPage).where(condition)) or 0
    pages = session.scalars(
        select(DocumentPage).where(condition).order_by(DocumentPage.page_number).limit(limit).offset(offset)
    ).all()
    return PageList(items=[PageOut.model_validate(page) for page in pages], total=total)


@router.get("/{document_id}/events", summary="Processing history", responses=_NOT_FOUND)
def get_document_events(
    document_id: UUID, session: SessionDep, limit: ListLimit = 500, offset: Offset = 0
) -> EventList:
    """Events oldest first (default: the first 500)."""
    document = get_document_or_404(session, document_id)
    events = session.scalars(
        select(ProcessingEvent)
        .where(ProcessingEvent.document_id == document.id)
        .order_by(ProcessingEvent.id)
        .limit(limit)
        .offset(offset)
    ).all()
    return EventList(items=[EventOut.model_validate(event) for event in events])


@router.get(
    "/{document_id}/file",
    summary="Download the original file",
    response_class=FileResponse,
    responses={200: {"content": {"application/octet-stream": {}}, "description": "The uploaded bytes"}, **_NOT_FOUND},
)
def download_document_file(document_id: UUID, session: SessionDep, storage: StorageDep) -> FileResponse:
    document = get_document_or_404(session, document_id)
    try:
        path: Path = storage.path(document.storage_key)
    except ValueError:
        logger.error("invalid_storage_key", document_id=str(document.id))
        raise NotFoundError("The original file is not available") from None
    if not path.is_file():
        raise NotFoundError("The original file is not available")
    return FileResponse(
        path,
        media_type=document.content_type,
        headers={"Content-Disposition": content_disposition(document.filename), "X-Content-Type-Options": "nosniff"},
    )


# ------------------------------------------------------------------------------------------------ writes


@router.post(
    "/{document_id}/reprocess",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Process the document again",
    responses={**_NOT_FOUND, 409: {"model": ErrorResponse, "description": "Already queued or processing"}},
)
def reprocess_document(document_id: UUID, session: SessionDep, queue: QueueDep) -> ReprocessResponse:
    document = get_document_or_404(session, document_id)
    try:
        job = intake.create_reprocess_job(session, document)
    except intake.AlreadyProcessingError:
        raise AlreadyProcessingError() from None
    session.commit()
    enqueue_job(queue, job.id)
    return ReprocessResponse(document_id=document.id, job_id=job.id)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document, its results and its file",
    responses={**_NOT_FOUND, 409: {"model": ErrorResponse, "description": "A job is running"}},
)
def delete_document(document_id: UUID, session: SessionDep, settings: SettingsDep, storage: StorageDep) -> Response:
    """Deletes the document with its pages, records, issues, jobs, events and stored file. Refused (409) while a
    worker is running a job for it (a job whose worker stopped sending heartbeats does not count); a queued job is
    removed with the document and its worker finds nothing to do.

    Other documents' records that shared an invoice number with the deleted records are reconciled again in the
    same transaction, so their duplicate and cross-verification issues (and review / document status) no longer
    point at the deleted document; re-uploads of the deleted file lose their ``DUPLICATE_FILE`` issue."""
    document = get_document_or_404(session, document_id)
    now = utcnow()
    running = session.scalars(
        select(ProcessingJob).where(ProcessingJob.document_id == document.id, ProcessingJob.status == JobStatus.RUNNING)
    ).all()
    if any(not heartbeat_is_stale(job, now, settings.job_stale_after_s) for job in running):
        raise AlreadyProcessingError("The document is being processed; delete it when processing has finished")
    storage_key = document.storage_key
    serialize_reconciliation(session)  # before the first write (lock order, see its docstring)
    numbers = {
        number
        for number in session.scalars(select(Invoice.invoice_number_norm).where(Invoice.document_id == document.id))
        if number
    }
    reuploads = set(session.scalars(select(Document.id).where(Document.duplicate_of_id == document.id)))
    session.delete(document)
    session.flush()  # cascades: records, their cross-verifications and issues; re-uploads' duplicate_of_id -> NULL
    affected = set(reuploads)
    if numbers:
        affected |= reconcile_invoices(session, [], extra_numbers=numbers).document_ids
    if reuploads:
        session.execute(
            delete(ValidationIssueRow)
            .where(
                ValidationIssueRow.document_id.in_(reuploads),
                ValidationIssueRow.invoice_id.is_(None),
                ValidationIssueRow.code == IssueCode.DUPLICATE_FILE.value,
            )
            .execution_options(synchronize_session=False)
        )
    for other_id in sorted(affected, key=str):
        other = session.get(Document, other_id)
        if other is not None:
            refresh_document_status(session, other)
    session.commit()
    try:
        storage.delete(storage_key)
    except (OSError, ValueError) as exc:
        logger.warning("stored_file_not_deleted", document_id=str(document_id), **exception_summary(exc))
    return Response(status_code=status.HTTP_204_NO_CONTENT)
