"""Request context and last-resort error handling (pure ASGI middleware).

``RequestContextMiddleware`` gives every request an id - the incoming ``X-Request-ID`` when it is a short token of
safe characters, else a new UUID4 - binds it to the log context, returns it in ``X-Request-ID`` and writes one access
line per request: method, route template (``/api/v1/documents/{document_id}``, never the raw path or query string),
status, duration in ms.

``UnhandledErrorMiddleware`` turns any exception that escaped the routes into a 500 ``internal_error`` envelope with
a generic message. The exception text may contain document data or SQL parameters, so it goes nowhere: the log line
carries the request id, the exception classes and the code location (``exception_summary``).
"""

from __future__ import annotations

import re
import time
import uuid

import structlog
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.errors import error_response
from app.logging_setup import exception_summary

_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
access_log = structlog.get_logger("docintel.access")
error_log = structlog.get_logger("docintel.errors")


def route_template(scope: Scope) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else "<unmatched>"


def _incoming_request_id(scope: Scope) -> str | None:
    for name, value in scope.get("headers", ()):
        if name == b"x-request-id":
            candidate = value.decode("latin-1")
            return candidate if _REQUEST_ID.fullmatch(candidate) else None
    return None


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = _incoming_request_id(scope) or str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        status = 500
        started = time.perf_counter()

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", [])
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        with structlog.contextvars.bound_contextvars(request_id=request_id):
            try:
                await self.app(scope, receive, send_with_id)
            finally:
                access_log.info(
                    "request",
                    method=scope.get("method"),
                    path=route_template(scope),
                    status=status,
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                )


class UnhandledErrorMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        response_started = False

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception as exc:
            request_id = scope.get("state", {}).get("request_id")
            error_log.error(
                "unhandled_error", path=route_template(scope), request_id=request_id, **exception_summary(exc)
            )
            if response_started:
                raise
            response = error_response(
                500, "internal_error", "An unexpected error occurred", details={"request_id": request_id}
            )
            await response(scope, receive, send)
