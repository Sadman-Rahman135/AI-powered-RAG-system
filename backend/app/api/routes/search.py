"""Search: keyword (full text), semantic (embeddings) and hybrid (rank fusion), with structured invoice filters."""

from __future__ import annotations

from datetime import date
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query
from pydantic import ValidationError

from app.api.deps import API_PREFIX, EmbedderDep, Limit, Offset, SessionDep, guard
from app.api.errors import ValidationFailedError, validation_details
from app.api.schemas import QueryAmount, QueryText, SearchHitOut, SearchResponse
from app.domain.schemas import ReviewStatus
from app.ingestion.types import FileKind
from app.search.service import MAX_QUERY_CHARS, SearchFilters, SearchMode, search

router = APIRouter(prefix=f"{API_PREFIX}/search", tags=["search"], dependencies=guard("search"))
_TEXT = Annotated[QueryText, Query(max_length=200)]


@router.get("", summary="Search documents and records")
def search_documents(
    session: SessionDep,
    embedder: EmbedderDep,
    q: Annotated[QueryText, Query(max_length=MAX_QUERY_CHARS, description='Words, "phrases", -exclusions, or')] = "",
    mode: SearchMode = "hybrid",
    supplier: _TEXT = None,
    customer: _TEXT = None,
    invoice_number: Annotated[QueryText, Query(max_length=100)] = None,
    currency: Annotated[QueryText, Query(max_length=3)] = None,
    date_from: date | None = None,
    date_to: date | None = None,
    total_min: QueryAmount = None,
    total_max: QueryAmount = None,
    review_status: ReviewStatus | None = None,
    file_kind: FileKind | None = None,
    document_id: UUID | None = None,
    limit: Limit = 20,
    offset: Offset = 0,
) -> SearchResponse:
    """One hit per document or record, best first; ``snippet`` marks matches with « and ».

    ``match_type`` says which ranking found a hit: ``keyword``, ``semantic``, ``hybrid`` (both) or ``filter`` - a
    blank ``q`` with filters lists the matching records newest first (score 0). Scores are comparable only within
    one result list.
    """
    try:
        filters = SearchFilters(
            supplier=supplier,
            customer=customer,
            invoice_number=invoice_number,
            currency=currency,
            date_from=date_from,
            date_to=date_to,
            total_min=total_min,
            total_max=total_max,
            review_status=review_status,
            file_kind=file_kind,
            document_id=document_id,
        )
    except ValidationError as exc:
        raise ValidationFailedError(details=validation_details(exc.errors())) from None
    page = search(session, q or "", mode=mode, filters=filters, limit=limit, offset=offset, embedder=embedder)
    return SearchResponse(
        items=[SearchHitOut.model_validate(hit.model_dump()) for hit in page.hits],
        total=page.total,
        took_ms=page.took_ms,
        mode=page.mode,
    )
