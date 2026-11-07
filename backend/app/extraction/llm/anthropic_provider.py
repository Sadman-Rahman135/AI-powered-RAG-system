"""Claude (Anthropic Messages API) provider with schema-constrained JSON output.

Request shape (verified against the installed SDK): ``client.beta.messages.create`` with the server-side
refusal-fallback beta, ``fallbacks="default"`` and ``output_config`` carrying the effort level and the JSON schema.
Current models reject ``temperature``/``top_p``/``top_k`` and an explicit ``thinking: disabled``, so none is sent.
"""

from __future__ import annotations

from typing import Any, Literal

import anthropic

from app.extraction.llm.base import (
    LLMError,
    LLMOutputTruncatedError,
    LLMRefusalError,
    LLMUnavailableError,
    parse_json_object,
)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
_SDK_RETRIES = 2  # the SDK retries 408/409/429/5xx and connection errors itself before raising
Effort = Literal["low", "medium", "high", "xhigh", "max"]


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-5-5",
        effort: Effort = "medium",
        timeout: float = 120,
        *,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self._client = (
            client
            if client is not None
            else anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=_SDK_RETRIES)
        )

    def complete_json(
        self, *, system: str, user: str, schema: dict[str, Any], max_tokens: int = 4096
    ) -> dict[str, Any]:
        try:
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                betas=[FALLBACK_BETA],
                fallbacks="default",
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
            )
        except anthropic.APIError as exc:
            raise _map_error(exc) from exc
        return _read_json(response)


def _read_json(response: Any) -> dict[str, Any]:
    request_id = getattr(response, "_request_id", None)
    reference = f" (request {request_id})" if request_id else ""
    stop_reason = response.stop_reason
    # The stop reason decides whether the content may be read at all: a refusal can carry partial text.
    if stop_reason == "refusal":
        category = getattr(getattr(response, "stop_details", None), "category", None)
        detail = f", category {category}" if category else ""
        raise LLMRefusalError(f"anthropic declined the request{detail}{reference}")
    if stop_reason in ("max_tokens", "model_context_window_exceeded"):
        raise LLMOutputTruncatedError(f"anthropic stopped before finishing the JSON: {stop_reason}{reference}")
    text = next((block.text for block in response.content if block.type == "text"), None)
    return parse_json_object(text, f"anthropic response{reference}")


def _map_error(exc: anthropic.APIError) -> LLMError:
    """Translate an SDK exception into an ``LLMError`` (message: status, error type, request id; no provider text)."""
    if isinstance(exc, anthropic.APITimeoutError):
        return LLMUnavailableError("anthropic request timed out")
    if isinstance(exc, anthropic.APIConnectionError):
        return LLMUnavailableError("anthropic connection failed")
    if isinstance(exc, anthropic.APIStatusError):
        status = exc.status_code
        error_type = exc.type or "unknown"
        request_id = f", request {exc.request_id}" if exc.request_id else ""
        # 503 / 504 / 529 have their own classes in this SDK (not InternalServerError), so the status decides too.
        transient = (
            isinstance(exc, anthropic.RateLimitError | anthropic.InternalServerError)
            or status in (408, 429)
            or status >= 500
        )
        return LLMError(f"anthropic API error: HTTP {status} ({error_type}){request_id}", transient=transient)
    return LLMError(f"anthropic client error: {type(exc).__name__}", transient=False)
