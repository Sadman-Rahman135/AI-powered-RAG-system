"""API-key authentication and the caller identity used for audit fields.

When ``API_KEYS`` is set, every ``/api/v1`` route except the liveness probe requires one of the keys, sent as
``X-API-Key: <key>`` or ``Authorization: Bearer <key>``. Keys are compared in constant time against every configured
key, and are never logged or stored: the caller is known by a fingerprint (``sha256(key)``) only.

``actor_for`` decides the audit identity: ``corrected_by`` / ``reviewed_by`` is the ``X-Actor`` header the web
proxy forwards (trimmed, control characters removed, at most 100 characters), else ``"api-key:" + sha256(key)[:8]``
when a valid key was used, else ``"anonymous"``.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass

from fastapi import Request

from app.api.errors import UnauthorizedError
from app.config import get_settings

ACTOR_MAX_CHARS = 100
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")


@dataclass(frozen=True, slots=True)
class Principal:
    """Who is calling: ``key_fingerprint`` is the full sha256 hex of a valid configured key, else ``None``."""

    key_fingerprint: str | None

    @property
    def authenticated(self) -> bool:
        return self.key_fingerprint is not None


def presented_key(request: Request) -> str | None:
    """The key the request carries: ``X-API-Key`` wins over an ``Authorization: Bearer`` token."""
    key = request.headers.get("x-api-key")
    if key is not None:
        return key.strip() or None
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    return None


def _matches_any(candidate: str, keys: list[str]) -> bool:
    presented = candidate.encode("utf-8")
    matched = False
    for key in keys:  # no early exit: the time taken does not depend on which key matched
        matched |= hmac.compare_digest(presented, key.encode("utf-8"))
    return matched


def get_principal(request: Request) -> Principal:
    """Identify the caller once per request (cached on ``request.state``)."""
    cached = getattr(request.state, "principal", None)
    if isinstance(cached, Principal):
        return cached
    keys = get_settings().api_key_list
    key = presented_key(request)
    fingerprint = hashlib.sha256(key.encode("utf-8")).hexdigest() if key and keys and _matches_any(key, keys) else None
    principal = Principal(key_fingerprint=fingerprint)
    request.state.principal = principal
    return principal


def require_api_key(request: Request) -> None:
    """Dependency: 401 ``unauthorized`` unless auth is disabled (no ``API_KEYS``) or a valid key is presented."""
    if get_settings().api_key_list and not get_principal(request).authenticated:
        raise UnauthorizedError()


def sanitize_actor(value: str) -> str | None:
    cleaned = _CONTROL.sub("", value).strip()[:ACTOR_MAX_CHARS].strip()
    return cleaned or None


def actor_for(request: Request) -> str:
    """The audit identity of the caller."""
    header = request.headers.get("x-actor")
    if header is not None and (actor := sanitize_actor(header)):
        return actor
    fingerprint = get_principal(request).key_fingerprint
    return f"api-key:{fingerprint[:8]}" if fingerprint else "anonymous"
