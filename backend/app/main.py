"""FastAPI application: the REST API under ``/api/v1`` with its security layer.

Run with ``uvicorn app.main:app`` (OpenAPI UI at ``/docs``, schema at ``/openapi.json``). Middleware, outermost
first: security headers -> request id + access log -> CORS -> request size limit -> last-resort 500 handler.
Startup refuses to run with an unsafe production configuration or a malformed ``RATE_LIMIT_*`` value, creates the
tables on SQLite (PostgreSQL uses Alembic migrations) and, with the in-process ``local`` queue, re-queues jobs left
``queued`` by a previous process.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.errors import install_error_handlers
from app.api.middleware import RequestContextMiddleware, UnhandledErrorMiddleware
from app.api.routes import cross_verify, documents, health, invoices, jobs, search, stats
from app.config import get_settings
from app.db.session import init_db
from app.logging_setup import configure_logging
from app.queue import requeue_pending_jobs
from app.security.headers import SecurityHeadersMiddleware
from app.security.ratelimit import RateLimiter, validate_rate_limits
from app.security.uploads import RequestSizeLimitMiddleware

logger = structlog.get_logger(__name__)

_ROUTERS = (
    health.public_router,
    health.router,
    documents.upload_router,
    documents.router,
    invoices.router,
    jobs.router,
    search.router,
    cross_verify.router,
    stats.router,
)
_CORS_HEADERS = ["Authorization", "Content-Type", "X-API-Key", "X-Actor", "X-Request-ID"]
_CORS_EXPOSED = ["X-Request-ID", "Retry-After", "Content-Disposition"]


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    settings = get_settings()
    problems = settings.production_problems()
    if problems:
        raise RuntimeError("refusing to start with an unsafe production configuration: " + "; ".join(problems))
    validate_rate_limits(settings)
    if settings.is_sqlite:
        init_db()
    if settings.queue_backend == "local":
        requeued = requeue_pending_jobs()
        logger.info("pending_jobs_requeued", count=requeued)
    logger.info("api_started", env=settings.app_env, queue=settings.queue_backend, auth=bool(settings.api_key_list))
    yield
    from app.pipeline.processor import close_llm_provider  # imported late: only the local queue runs jobs here

    close_llm_provider()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)
    app = FastAPI(
        title=f"{settings.app_name} API",
        version=__version__,
        description=(
            "Upload invoices and receipts (PDF, images, Excel, CSV), follow their processing, review and correct the "
            "extracted records, search them and cross-verify them. Errors use the envelope "
            '`{"error": {"code", "message", "details"}}`. When `API_KEYS` is set every route except '
            "`/api/v1/health` needs `X-API-Key: <key>` or `Authorization: Bearer <key>`."
        ),
        lifespan=lifespan,
    )
    app.state.rate_limiter = RateLimiter.from_settings(settings)
    install_error_handlers(app)
    for router in _ROUTERS:
        app.include_router(router)

    # add_middleware wraps: the last one added is the outermost
    app.add_middleware(UnhandledErrorMiddleware)
    app.add_middleware(RequestSizeLimitMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,  # keys travel in headers, never cookies
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=_CORS_HEADERS,
        expose_headers=_CORS_EXPOSED,
        max_age=600,
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    return app


app = create_app()
