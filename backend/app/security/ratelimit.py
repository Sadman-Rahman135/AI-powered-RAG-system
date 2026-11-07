"""Request rate limits: fixed-window counters from the ``limits`` library (the engine behind slowapi).

Three budgets: ``RATE_LIMIT_UPLOAD`` for uploads, ``RATE_LIMIT_SEARCH`` for search and ``RATE_LIMIT_DEFAULT`` for
every other ``/api/v1`` route (the liveness probe is exempt). A caller is counted by its API-key fingerprint when it
presents a valid key (plus its ``X-Actor``, if any: the web UI's users share the proxy's key but not its budget),
else by client IP - so failed authentication attempts are limited per IP, and an ``X-Actor`` without a valid key
changes nothing. A key holder can split its traffic over several actor names; key holders are trusted callers (one
trust domain), and the web proxy sets ``X-Actor`` itself from the signed-in user, never from the browser. Behind a
reverse proxy, run uvicorn with ``--proxy-headers --forwarded-allow-ips`` so the client IP is the real one.

Counters live in Redis when ``REDIS_URL`` is set (shared by every API process), in memory otherwise. A storage
failure never blocks traffic: the request is let through and a warning is logged. ``RATE_LIMIT_ENABLED=false``
switches limiting off (the test default). Over the limit -> 429 ``rate_limited`` with ``Retry-After``. The limit
strings are validated at startup (``validate_rate_limits``).
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from functools import lru_cache
from typing import Literal

import structlog
from fastapi import Request
from limits import RateLimitItem, parse
from limits.storage import storage_from_string
from limits.strategies import FixedWindowRateLimiter

from app.api.errors import RateLimitedError
from app.config import Settings, get_settings
from app.logging_setup import exception_summary
from app.security.auth import get_principal, sanitize_actor

logger = structlog.get_logger(__name__)

RateScope = Literal["upload", "search", "default"]
SCOPES: tuple[RateScope, ...] = ("upload", "search", "default")
_NAMESPACE = "docintel"


@lru_cache(maxsize=32)
def _parse(limit: str) -> RateLimitItem:
    return parse(limit)


class RateLimiter:
    """One limiter per app; its storage is created on first use."""

    def __init__(self, storage_uri: str) -> None:
        self._storage_uri = storage_uri
        self._strategy: FixedWindowRateLimiter | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> RateLimiter:
        return cls(settings.redis_url or "memory://")

    def _limiter(self) -> FixedWindowRateLimiter:
        if self._strategy is None:
            self._strategy = FixedWindowRateLimiter(storage_from_string(self._storage_uri))
        return self._strategy

    def check(self, scope: str, client_key: str, limit: str) -> None:
        """Count one request; raise ``RateLimitedError`` when ``client_key`` is over ``limit`` in ``scope``."""
        item = _parse(limit)
        try:
            limiter = self._limiter()
            if limiter.hit(item, _NAMESPACE, scope, client_key):
                return
            reset_at = limiter.get_window_stats(item, _NAMESPACE, scope, client_key).reset_time
        except Exception as exc:  # the limiter must never take the API down with it
            logger.warning("rate_limiter_unavailable", **exception_summary(exc))
            return
        retry_after = max(1, math.ceil(reset_at - time.time()))
        raise RateLimitedError(
            details={"limit": str(item), "retry_after_s": retry_after},
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(item.amount),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(math.ceil(reset_at)),
            },
        )


def client_key(request: Request) -> str:
    """``key:<fingerprint>`` for a valid API key - ``key:<fingerprint>:actor:<sha256(actor)>`` when the request also
    names an ``X-Actor`` (the web proxy sends its signed-in user's name with the shared key, so every UI user gets a
    budget of their own) - else ``ip:<client address>``. ``X-Actor`` is never trusted without a valid key."""
    fingerprint = get_principal(request).key_fingerprint
    if fingerprint:
        header = request.headers.get("x-actor")
        actor = sanitize_actor(header) if header is not None else None
        if actor:
            return f"key:{fingerprint[:32]}:actor:{hashlib.sha256(actor.encode('utf-8')).hexdigest()[:32]}"
        return f"key:{fingerprint[:32]}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


def validate_rate_limits(settings: Settings) -> None:
    """Fail fast (at startup) on a malformed ``RATE_LIMIT_*`` value, which would otherwise make every request a 500."""
    for scope in SCOPES:
        value = _configured_limit(settings, scope)
        try:
            _parse(value)
        except ValueError as exc:
            name = f"RATE_LIMIT_{scope.upper()}"
            raise RuntimeError(f"{name}={value!r} is not a valid rate limit (use e.g. '60/minute')") from exc


def _configured_limit(settings: Settings, scope: RateScope) -> str:
    return {
        "upload": settings.rate_limit_upload,
        "search": settings.rate_limit_search,
        "default": settings.rate_limit_default,
    }[scope]


def rate_limit(scope: RateScope) -> Callable[[Request], None]:
    """A dependency that applies the ``scope`` budget (the limiter is ``app.state.rate_limiter``)."""

    def enforce(request: Request) -> None:
        settings = get_settings()
        if not settings.rate_limit_enabled:
            return
        limiter: RateLimiter = request.app.state.rate_limiter
        limiter.check(scope, client_key(request), _configured_limit(settings, scope))

    enforce.__name__ = f"rate_limit_{scope}"
    return enforce
