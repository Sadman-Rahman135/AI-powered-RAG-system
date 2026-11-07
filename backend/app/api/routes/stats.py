"""Dashboard statistics: counts by status, issues by code and severity, processing times, the LLM in use and how
often extraction fell back from it to the rules."""

from __future__ import annotations

import math
from uuid import UUID

from fastapi import APIRouter
from sqlalchemy import func, select
from sqlalchemy.orm import QueryableAttribute, Session

from app.api.deps import API_PREFIX, SessionDep, SettingsDep, guard
from app.api.llm_info import describe_llm
from app.api.schemas import (
    DocumentStats,
    InvoiceStats,
    IssueStats,
    LlmInfo,
    ProcessingStats,
    StatsOut,
)
from app.db.models import Document, DocumentStatus, Invoice, ValidationIssueRow
from app.domain.schemas import IssueCode, ReviewStatus, Severity

router = APIRouter(prefix=f"{API_PREFIX}/stats", tags=["stats"], dependencies=guard("default"))


def _counts(session: Session, column: QueryableAttribute[str], known: tuple[str, ...] | None = None) -> dict[str, int]:
    """``{value: rows}`` for ``column``; every ``known`` value is present (0 when absent)."""
    found = {value: count for value, count in session.execute(select(column, func.count()).group_by(column))}
    if known is None:
        return dict(sorted(found.items(), key=lambda item: (-item[1], item[0])))
    return {**dict.fromkeys(known, 0), **found}


def _processing(session: Session) -> ProcessingStats:
    """Mean and 95th percentile (nearest rank) of ``processing_ms`` over processed documents, and pages."""
    timed = Document.processing_ms.is_not(None)
    count, average = session.execute(select(func.count(), func.avg(Document.processing_ms)).where(timed)).one()
    pages = session.scalar(select(func.coalesce(func.sum(Document.page_count), 0))) or 0
    if not count:
        return ProcessingStats(avg_ms=None, p95_ms=None, pages_total=int(pages))
    rank = math.ceil(0.95 * count)
    p95 = session.scalar(
        select(Document.processing_ms).where(timed).order_by(Document.processing_ms).offset(rank - 1).limit(1)
    )
    return ProcessingStats(avg_ms=round(float(average), 1), p95_ms=p95, pages_total=int(pages))


def _llm_fallbacks(session: Session) -> tuple[int, int]:
    """(documents, records) that used the rules result because the LLM was unavailable, failed or refused.

    Processing records this as the document-level ``LLM_UNAVAILABLE`` note, whose ``details.records`` counts the
    records that fell back; a note written before that count existed counts the document's rules-extracted records.
    """
    notes = session.execute(
        select(ValidationIssueRow.document_id, ValidationIssueRow.details).where(
            ValidationIssueRow.code == IssueCode.LLM_UNAVAILABLE.value, ValidationIssueRow.invoice_id.is_(None)
        )
    ).all()
    documents = {document_id for document_id, _ in notes}
    records = 0
    uncounted: set[UUID] = set()
    for document_id, details in notes:
        count = (details or {}).get("records")
        if isinstance(count, int) and not isinstance(count, bool):
            records += count
        else:
            uncounted.add(document_id)
    for batch in _batches(sorted(uncounted, key=str)):
        records += (
            session.scalar(
                select(func.count())
                .select_from(Invoice)
                .where(Invoice.document_id.in_(batch), Invoice.extraction_method == "rules")
            )
            or 0
        )
    return len(documents), records


def _batches(values: list[UUID], size: int = 500) -> list[list[UUID]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


@router.get("", summary="Counts, issue breakdown, processing times and the LLM in use")
def get_stats(session: SessionDep, settings: SettingsDep) -> StatsOut:
    """``llm.fallback_documents`` / ``llm.fallback_records``: how much of the processed data did not get the LLM
    it was configured for (no provider, provider errors, refusals) and used the rules result instead."""
    by_status = _counts(session, Document.status, DocumentStatus.ALL)
    by_review = _counts(session, Invoice.review_status, tuple(status.value for status in ReviewStatus))
    llm = describe_llm(settings)
    fallback_documents, fallback_records = _llm_fallbacks(session)
    return StatsOut(
        documents=DocumentStats(total=sum(by_status.values()), by_status=by_status),
        invoices=InvoiceStats(total=sum(by_review.values()), by_review_status=by_review),
        issues=IssueStats(
            by_code=_counts(session, ValidationIssueRow.code),
            by_severity=_counts(session, ValidationIssueRow.severity, tuple(severity.value for severity in Severity)),
        ),
        processing=_processing(session),
        llm=LlmInfo(
            provider=llm.provider,
            model=llm.model,
            fallback_documents=fallback_documents,
            fallback_records=fallback_records,
        ),
    )
