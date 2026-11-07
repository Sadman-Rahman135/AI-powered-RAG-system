"""Provider-neutral LLM contract: the ``LLMProvider`` protocol, its errors and the JSON reply parser.

Error messages carry status codes, error types and request ids only. They never contain document text, model
output or provider free text (that stays in the chained ``__cause__``), because callers log and persist them.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable


class LLMError(Exception):
    """A provider call failed. ``transient`` says whether retrying the same request later may succeed."""

    def __init__(self, message: str, transient: bool = False) -> None:
        super().__init__(message)
        self.transient = transient


class LLMRefusalError(LLMError):
    """The model declined to answer (policy refusal or blocked prompt). Never transient."""

    def __init__(self, message: str) -> None:
        super().__init__(message, transient=False)


class LLMOutputTruncatedError(LLMError):
    """The model ran out of output tokens before it finished the JSON (``finish_reason: length``, ``max_tokens``).

    Never transient - the same request truncates again - but a smaller request (without line items) may fit.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, transient=False)


class LLMUnavailableError(LLMError):
    """The provider could not be reached in time (connection refused, DNS failure, timeout). Transient by default."""

    def __init__(self, message: str, transient: bool = True) -> None:
        super().__init__(message, transient=transient)


class LLMConfigurationError(ValueError):
    """An explicitly selected provider is missing its configuration (API key, base URL).

    Deliberately not an ``LLMError``: a broad ``except LLMError`` that falls back to rules must not hide a
    deployment mistake.
    """


@runtime_checkable
class LLMProvider(Protocol):
    """Something that turns a system prompt, a user prompt and a JSON schema into a JSON object."""

    name: str
    model: str

    def complete_json(
        self, *, system: str, user: str, schema: dict[str, Any], max_tokens: int = 4096
    ) -> dict[str, Any]:
        """Return the model's reply as a JSON object that follows ``schema``; raise ``LLMError`` on any failure."""
        ...


_FENCE_RE = re.compile(r"^```[A-Za-z0-9_-]*[ \t]*\r?\n?(?P<body>.*?)\s*```$", re.DOTALL)


def parse_json_object(text: str | None, source: str) -> dict[str, Any]:
    """Parse a model reply into a JSON object (surrounding ```json fences are tolerated).

    ``source`` labels the reply in error messages (for example ``"anthropic response req_123"``).
    """
    if text is None or not text.strip():
        raise LLMError(f"{source} contained no text")
    body = text.strip()
    fenced = _FENCE_RE.match(body)
    if fenced:
        body = fenced.group("body").strip()
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise LLMError(f"{source} was not valid JSON ({exc.msg} at character {exc.pos})") from exc
    if not isinstance(value, dict):
        raise LLMError(f"{source} was a JSON {type(value).__name__}, expected an object")
    return value
