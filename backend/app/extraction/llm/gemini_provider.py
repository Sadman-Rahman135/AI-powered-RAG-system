"""Gemini (google-genai) provider with schema-constrained JSON output.

The JSON schema goes through ``response_json_schema`` (the full-JSON-Schema field; ``response_schema`` is the older
OpenAPI subset). Gemini 3 models take a ``thinking_level``; it is set to LOW because extraction needs no reasoning
budget. Other models get no thinking configuration (some cannot disable thinking, so a budget of 0 is not safe).
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from typing import Any

import httpx
from google import genai
from google.genai import errors, types

from app.extraction.llm.base import (
    LLMError,
    LLMOutputTruncatedError,
    LLMRefusalError,
    LLMUnavailableError,
    parse_json_object,
)

# The SDK retries timeouts and server errors: 3 attempts, waiting about 2 and 4 s.
_RETRY = types.HttpRetryOptions(
    attempts=3, initial_delay=2.0, exp_base=2.0, max_delay=30.0, http_status_codes=[408, 500, 502, 503, 504]
)
# Rate limits (429) are retried here, after the delay the server asks for (RetryInfo): up to 4 tries in total. A delay
# over a minute means the quota is spent for now (a free-tier key gets a small daily quota per model), so the segment
# falls back to rules at once instead of holding its document; a 429 that names no delay waits 15 s.
_RATE_LIMIT_TRIES = 4
_MAX_RATE_LIMIT_WAIT_S = 60.0
_DEFAULT_RATE_LIMIT_WAIT_S = 15.0

logger = logging.getLogger(__name__)

# Stops where the model withheld its answer. RECITATION counts: copying document text verbatim can trip it.
_REFUSAL_FINISH_REASONS = frozenset({"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"})


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: float = 120,
        *,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self._sleep = sleep
        self._client = (
            client
            if client is not None
            else genai.Client(
                api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout * 1000), retry_options=_RETRY)
            )
        )

    def complete_json(
        self, *, system: str, user: str, schema: dict[str, Any], max_tokens: int = 4096
    ) -> dict[str, Any]:
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema,
            max_output_tokens=max_tokens,
            thinking_config=self._thinking_config(),
        )
        for attempt in range(1, _RATE_LIMIT_TRIES + 1):
            try:
                response = self._client.models.generate_content(model=self.model, contents=user, config=config)
                break
            except errors.APIError as exc:
                wait = _rate_limit_wait(exc)
                if wait is None or attempt == _RATE_LIMIT_TRIES:
                    raise _map_api_error(exc) from exc
                if wait > _MAX_RATE_LIMIT_WAIT_S:
                    raise _map_api_error(exc, note=f"quota exhausted, retry after {wait / 3600:.1f} h") from exc
                logger.info("gemini rate limit (HTTP 429); retrying in %.1f s (try %d of %d)", wait, attempt + 1,
                            _RATE_LIMIT_TRIES)  # fmt: skip
                self._sleep(wait)
            except (httpx.TransportError, TimeoutError, ConnectionError) as exc:
                raise LLMUnavailableError(f"gemini request failed: {type(exc).__name__}") from exc
        return _read_json(response)

    def _thinking_config(self) -> types.ThinkingConfig | None:
        if self.model.startswith("gemini-3"):
            return types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
        return None


def _name(value: Any) -> str:
    return getattr(value, "name", None) or str(value)


def _read_json(response: Any) -> dict[str, Any]:
    feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(feedback, "block_reason", None)
    if block_reason:
        raise LLMRefusalError(f"gemini blocked the prompt ({_name(block_reason)})")
    candidates = getattr(response, "candidates", None) or []
    finish_reason = _name(candidates[0].finish_reason) if candidates and candidates[0].finish_reason else None
    if finish_reason in _REFUSAL_FINISH_REASONS:
        raise LLMRefusalError(f"gemini withheld the answer ({finish_reason})")
    if finish_reason == "MAX_TOKENS":
        raise LLMOutputTruncatedError("gemini stopped before finishing the JSON: MAX_TOKENS")
    return parse_json_object(response.text, "gemini response")


def _rate_limit_wait(exc: errors.APIError) -> float | None:
    """Seconds to wait before retrying a 429: the server's RetryInfo delay, else the default; ``None`` for other
    errors."""
    if exc.code != 429:
        return None
    body = exc.details if isinstance(exc.details, dict) else {}
    for detail in (body.get("error") or {}).get("details") or []:
        if isinstance(detail, dict) and str(detail.get("@type", "")).endswith("RetryInfo"):
            match = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(detail.get("retryDelay", "")).strip())
            if match:
                return float(match.group(1))
    return _DEFAULT_RATE_LIMIT_WAIT_S


def _map_api_error(exc: errors.APIError, *, note: str | None = None) -> LLMError:
    """Translate an SDK error (message: HTTP status and status name only; the provider's text stays in the cause)."""
    code = exc.code if isinstance(exc.code, int) else 0
    transient = code in (408, 429) or code >= 500
    status = exc.status or "unknown"
    return LLMError(f"gemini API error: HTTP {code} ({status}{'; ' + note if note else ''})", transient=transient)
