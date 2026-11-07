"""Liveness (``/health``, never authenticated or rate limited), readiness (``/ready``) and public configuration."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app.api.deps import API_PREFIX, QueueDep, SessionDep, SettingsDep, guard
from app.api.llm_info import describe_llm
from app.api.schemas import ConfigOut, HealthOut, ReadyOut
from app.ingestion.sniff import EXTENSIONS
from app.security.uploads import max_upload_request_bytes

public_router = APIRouter(prefix=API_PREFIX, tags=["health"])
router = APIRouter(prefix=API_PREFIX, tags=["health"], dependencies=guard("default"))


@public_router.get("/health", summary="Liveness probe")
def health() -> HealthOut:
    return HealthOut()


@router.get(
    "/ready",
    summary="Readiness: database and queue reachable",
    responses={503: {"model": ReadyOut, "description": "The database or the queue is not reachable"}},
)
def ready(session: SessionDep, queue: QueueDep, settings: SettingsDep, response: Response) -> ReadyOut:
    """``status`` is ``ok`` (HTTP 200) when the database answers and the queue backend responds, else
    ``unavailable`` (HTTP 503). ``llm`` reports the configuration only (``none`` / ``configured`` / ``misconfigured``);
    the model server is not called."""
    database: Literal["ok", "error"] = "ok"
    queue_state: Literal["ok", "error"] = "ok"
    try:
        session.execute(text("SELECT 1"))
    except Exception:
        database = "error"
    try:
        if not queue.ping():
            queue_state = "error"
    except Exception:
        queue_state = "error"
    report = ReadyOut(
        status="ok" if database == queue_state == "ok" else "unavailable",
        database=database,
        queue=queue_state,
        llm=describe_llm(settings).state,
    )
    if report.status != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return report


@router.get("/config", summary="Upload limits, accepted file types and the LLM in use")
def config(settings: SettingsDep) -> ConfigOut:
    llm = describe_llm(settings)
    return ConfigOut(
        max_upload_mb=settings.max_upload_mb,
        max_files_per_request=settings.max_files_per_request,
        max_request_mb=max_upload_request_bytes(settings) // (1024 * 1024),
        allowed_extensions=sorted(set().union(*EXTENSIONS.values())),
        llm_provider=llm.provider,
        llm_model=llm.model,
        auth_required=bool(settings.api_key_list),
    )
