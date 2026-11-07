"""The error envelope and the exception handlers that produce it.

Every error response is ``{"error": {"code", "message", "details"}}``. Routes raise an ``ApiError`` subclass; the
handlers below also translate framework errors (unknown route, wrong method, request validation). ``ApiError``
derives from Starlette's ``HTTPException`` so that one raised while FastAPI reads a request body (the size limit)
reaches its handler unchanged. Unexpected exceptions are turned into ``internal_error`` by
``app.api.middleware.UnhandledErrorMiddleware``.

Validation details are ``{"fields": {name: message}}``. A name is the parameter or body path without its location
prefix, dotted for nesting (``limit``, ``line_items.0.amount``); the ``fields.`` wrapper of the invoice PATCH body is
dropped, so a bad corrected total is reported as ``total``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

_CODES_BY_STATUS = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    406: "not_acceptable",
    409: "conflict",
    413: "file_too_large",
    415: "unsupported_file_type",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    503: "service_unavailable",
}
_LOCATIONS = frozenset({"body", "query", "path", "header", "cookie"})


class ApiError(StarletteHTTPException):
    """An error with a stable machine-readable ``code`` (part of the API contract)."""

    status_code_default = 500
    code = "internal_error"
    default_message = "An unexpected error occurred"

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.details = details
        super().__init__(status_code=self.status_code_default, detail=self.message, headers=headers)


class UnauthorizedError(ApiError):
    status_code_default = 401
    code = "unauthorized"
    default_message = "A valid API key is required (X-API-Key header or Authorization: Bearer)"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message, headers={"WWW-Authenticate": "Bearer"})


class NotFoundError(ApiError):
    status_code_default = 404
    code = "not_found"
    default_message = "The requested resource was not found"


class AlreadyProcessingError(ApiError):
    status_code_default = 409
    code = "already_processing"
    default_message = "The document is already queued or being processed"


class PayloadTooLargeError(ApiError):
    status_code_default = 413
    code = "file_too_large"
    default_message = "The request body is too large"


class UnsupportedMediaTypeError(ApiError):
    status_code_default = 415
    code = "unsupported_file_type"
    default_message = "Unsupported request content type"


class ValidationFailedError(ApiError):
    status_code_default = 422
    code = "validation_error"
    default_message = "The request is invalid"


class RateLimitedError(ApiError):
    status_code_default = 429
    code = "rate_limited"
    default_message = "Too many requests; retry later"


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """The envelope as a response."""
    return JSONResponse(
        {"error": {"code": code, "message": message, "details": details}}, status_code=status_code, headers=headers
    )


def field_key(location: Sequence[str | int]) -> str:
    """``("body", "fields", "total")`` -> ``"total"``; ``("query", "limit")`` -> ``"limit"`` (see module doc)."""
    parts = list(location)
    if parts and parts[0] in _LOCATIONS:
        source, parts = parts[0], parts[1:]
        if source != "body":
            parts = parts[:1]  # a parameter; ("query", "status", 0) is the "status" list parameter
    if len(parts) >= 2 and parts[0] == "fields":
        parts = parts[1:]
    return ".".join(str(part) for part in parts) or "body"


def validation_details(errors: Sequence[Any]) -> dict[str, Any]:
    """Pydantic errors as ``{"fields": {name: message}}``; input values are never echoed back."""
    fields: dict[str, str] = {}
    for error in errors:
        # a JSON syntax error's location is a character offset, not a field
        key = "body" if error.get("type") == "json_invalid" else field_key(error.get("loc", ()))
        message = str(error.get("msg", "Invalid value")).removeprefix("Value error, ")
        fields[key] = f"{fields[key]}; {message}" if key in fields else message
    return {"fields": fields}


async def _api_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return error_response(exc.status_code, exc.code, exc.message, exc.details, exc.headers)


async def _http_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    status = exc.status_code
    code = _CODES_BY_STATUS.get(status, "http_error")
    message = exc.detail if isinstance(exc.detail, str) and exc.detail else HTTPStatus(status).phrase
    return error_response(status, code, message, None, exc.headers)


async def _validation_handler(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    return error_response(422, "validation_error", "The request is invalid", validation_details(exc.errors()))


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiError, _api_error_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
