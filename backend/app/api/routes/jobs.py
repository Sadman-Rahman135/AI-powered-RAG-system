"""Processing jobs: list, and re-queue every failed document in one call."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, status
from sqlalchemy import func, select

from app.api.deps import API_PREFIX, Limit, Offset, QueueDep, SessionDep, enqueue_job, guard
from app.api.schemas import JobList, JobOut, JobStatusValue, ReprocessFailedResponse
from app.db.models import Document, DocumentStatus, ProcessingJob
from app.pipeline import intake

router = APIRouter(prefix=f"{API_PREFIX}/jobs", tags=["jobs"], dependencies=guard("default"))


@router.get("", summary="List processing jobs, newest first")
def list_jobs(
    session: SessionDep,
    status_filter: Annotated[JobStatusValue | None, Query(alias="status")] = None,
    limit: Limit = 20,
    offset: Offset = 0,
) -> JobList:
    statement = select(ProcessingJob)
    if status_filter is not None:
        statement = statement.where(ProcessingJob.status == status_filter)
    total = session.scalar(select(func.count()).select_from(statement.subquery())) or 0
    jobs = session.scalars(
        statement.order_by(ProcessingJob.created_at.desc(), ProcessingJob.id).limit(limit).offset(offset)
    ).all()
    return JobList(items=[JobOut.model_validate(job) for job in jobs], total=total)


@router.post("/reprocess-failed", status_code=status.HTTP_202_ACCEPTED, summary="Re-queue every failed document")
def reprocess_failed(session: SessionDep, queue: QueueDep) -> ReprocessFailedResponse:
    """Creates one new job per document whose status is ``failed`` (documents that already have a queued or running
    job are skipped) and queues them after one commit."""
    failed = session.scalars(
        select(Document).where(Document.status == DocumentStatus.FAILED).order_by(Document.created_at, Document.id)
    ).all()
    job_ids: list[UUID] = []
    for document in failed:
        try:
            job_ids.append(intake.create_reprocess_job(session, document).id)
        except intake.AlreadyProcessingError:
            continue
    session.commit()
    for job_id in job_ids:
        enqueue_job(queue, job_id)
    return ReprocessFailedResponse(requeued=len(job_ids), job_ids=job_ids)
