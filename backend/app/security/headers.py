"""Security headers on every response (pure ASGI middleware, so streamed and file responses get them too).

- ``X-Content-Type-Options: nosniff``, ``X-Frame-Options: DENY``, ``Referrer-Policy: no-referrer`` everywhere;
- ``Content-Security-Policy: default-src 'none'; frame-ancestors 'none'`` - the API serves data, never pages - except
  on the interactive docs (``/docs``, ``/redoc``), which load their scripts and styles from a CDN and only get
  ``frame-ancestors 'none'``;
- ``Cache-Control: no-store`` on ``/api/`` responses (invoice data and original files must not be cached), unless
  the route set its own.
"""

from __future__ import annotations

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

STRICT_CSP = "default-src 'none'; frame-ancestors 'none'"
DOCS_CSP = "frame-ancestors 'none'"
DOCS_PATHS = ("/docs", "/redoc")
_ALWAYS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path: str = scope.get("path", "")
        is_docs = path.startswith(DOCS_PATHS)
        is_api = path.startswith("/api/")

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in _ALWAYS.items():
                    headers[name] = value
                headers["Content-Security-Policy"] = DOCS_CSP if is_docs else STRICT_CSP
                if is_api and "cache-control" not in headers:
                    headers["Cache-Control"] = "no-store"
            await send(message)

        await self.app(scope, receive, send_with_headers)
