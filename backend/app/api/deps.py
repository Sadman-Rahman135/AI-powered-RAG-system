"""Shared dependencies: per-request DB session, settings, queue / storage / embedder, caller identity, pagination,
and the guard (rate limit, then API key) that every protected router declares.

The queue, storage and embedder come through dependencies so tests can swap them with
``app.dependency_overrides`` instead of patching modules.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated, Any
from uuid import UUID

import structlog
from fastapi import Depends, Query, Request
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.session import SessionLocal
from app.logging_setup import exception_summary
from app.queue import JobQueue, get_queue
from app.search.embeddings import Embedder, get_embedder
from app.security.auth import actor_for, require_api_key
from app.security.ratelimit import RateScope, rate_limit
from app.storage import Storage, get_storage

API_PREFIX = "/api/v1"
logger = structlog.get_logger(__name__)


def get_db() -> Iterator[Session]:
    """One session per request; routes commit explicitly, anything uncommitted is rolled back on close."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_job_queue() -> JobQueue:
    """The configured queue backend."""
    return get_queue()


def enqueue_job(queue: JobQueue, job_id: UUID) -> None:
    """Hand a committed job to the queue. A failure is logged, not raised: the job stays ``queued`` in the database
    (the source of truth) and is picked up by queue recovery."""
    try:
        queue.enqueue(job_id)
    except Exception as exc:
        logger.error("enqueue_failed", job_id=str(job_id), queue=getattr(queue, "name", None), **exception_summary(exc))


def get_file_storage() -> Storage:
    return get_storage()


def get_search_embedder() -> Embedder:
    return get_embedder()


def get_actor(request: Request) -> str:
    return actor_for(request)


def guard(scope: RateScope = "default") -> list[Any]:
    """Router dependencies of a protected route group: the ``scope`` rate limit, then the API-key check.

    The limit runs first so failed authentication attempts are counted (per client IP).
    """
    return [Depends(rate_limit(scope)), Depends(require_api_key)]


SessionDep = Annotated[Session, Depends(get_db)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
QueueDep = Annotated[JobQueue, Depends(get_job_queue)]
StorageDep = Annotated[Storage, Depends(get_file_storage)]
EmbedderDep = Annotated[Embedder, Depends(get_search_embedder)]
ActorDep = Annotated[str, Depends(get_actor)]
Limit = Annotated[int, Query(ge=1, le=100, description="Page size (1-100)")]
MAX_OFFSET = 1_000_000
Offset = Annotated[int, Query(ge=0, le=MAX_OFFSET, description="Items to skip (0-1000000)")]
ListLimit = Annotated[int, Query(ge=1, le=1000, description="At most this many items (1-1000)")]
