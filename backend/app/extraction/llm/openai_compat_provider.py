"""OpenAI-compatible chat-completions provider (llama.cpp server, vLLM, LM Studio, Ollama, gateways, ...).

The schema goes through ``response_format`` (llama.cpp turns it into a grammar, so the output is valid JSON by
construction) and sampling is greedy (``temperature: 0``). Nothing else is sent unless asked for:

* ``extra_body`` (``LLM_EXTRA_BODY``, a JSON object) is merged into every request body; its keys override the
  defaults above (``model`` and ``messages`` cannot be set). An empty object ``{}`` sends nothing extra.
* Without it, Qwen-style thinking is switched off through ``chat_template_kwargs`` - but only when the server
  identifies itself as llama.cpp, because strict OpenAI endpoints and several gateways reject unknown fields.

What the server is and how much context one request gets is asked once, on first use (never at construction):
llama.cpp ``GET /props`` next to the ``/v1`` API (per-slot ``default_generation_settings.n_ctx``; older builds
report the total ``n_ctx`` and ``total_slots``), else ``GET /v1/models`` (vLLM ``max_model_len``, llama.cpp
``meta.n_ctx``). A server without either is fine: the context stays unknown. ``context_tokens``
(``LLM_CONTEXT_TOKENS``) overrides whatever the server says; set it for a llama.cpp server with a unified KV cache
(``--kv-unified`` / ``-np auto``), whose slots share one context that /props reports whole.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from app.extraction.llm.base import (
    LLMConfigurationError,
    LLMError,
    LLMOutputTruncatedError,
    LLMRefusalError,
    LLMUnavailableError,
    parse_json_object,
)

logger = logging.getLogger(__name__)

PROBE_TIMEOUT_S = 5.0
LLAMA_CPP_EXTRA_BODY: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}
_RESERVED_KEYS = frozenset({"model", "messages"})
_LLAMA_CPP_PROPS_KEYS = ("default_generation_settings", "build_info", "total_slots")

ServerKind = Literal["llama.cpp", "vllm", "unknown"]


@dataclass(frozen=True, slots=True)
class ServerInfo:
    """What the server said about itself; ``context_tokens`` is the context one request may use."""

    kind: ServerKind
    context_tokens: int | None = None
    slots: int | None = None


_UNKNOWN = ServerInfo("unknown")


def parse_extra_body(raw: str | None) -> dict[str, Any] | None:
    """``LLM_EXTRA_BODY`` as a dict; ``None`` (automatic) for an unset or blank value.

    Raises ``LLMConfigurationError`` for anything but a JSON object or for an object that sets ``model`` /
    ``messages``. The message carries the parser's position only, never the value (it may hold credentials).
    """
    if raw is None or not raw.strip():
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMConfigurationError(
            f"LLM_EXTRA_BODY is not valid JSON ({exc.msg} at character {exc.pos})"
        ) from None
    if not isinstance(value, dict):
        raise LLMConfigurationError(f"LLM_EXTRA_BODY must be a JSON object, not a JSON {type(value).__name__}")
    reserved = sorted(_RESERVED_KEYS & value.keys())
    if reserved:
        raise LLMConfigurationError(f"LLM_EXTRA_BODY must not set {', '.join(reserved)}")
    return value


def _positive_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _llama_cpp_info(props: dict[str, Any]) -> ServerInfo:
    """Per-request context from llama.cpp ``/props``: the per-slot ``n_ctx``, or the total over the slots."""
    slots = _positive_int(props.get("total_slots"))
    generation = props.get("default_generation_settings")
    per_slot = _positive_int(generation.get("n_ctx")) if isinstance(generation, dict) else None
    total = _positive_int(props.get("n_ctx"))
    candidates = [n for n in (per_slot, total // (slots or 1) if total else None) if n]
    return ServerInfo("llama.cpp", min(candidates) if candidates else None, slots)


def _models_info(body: object, model: str) -> ServerInfo | None:
    """Context from an OpenAI-style model list: vLLM ``max_model_len`` or llama.cpp ``meta.n_ctx``."""
    data = body.get("data") if isinstance(body, dict) else None
    entries = [entry for entry in data if isinstance(entry, dict)] if isinstance(data, list) else []
    entry = next((e for e in entries if e.get("id") == model), entries[0] if len(entries) == 1 else None)
    if entry is None:
        return None
    meta = entry.get("meta")
    if entry.get("owned_by") == "llamacpp" or (isinstance(meta, dict) and "n_ctx" in meta):
        return ServerInfo("llama.cpp", _positive_int(meta.get("n_ctx")) if isinstance(meta, dict) else None)
    context = _positive_int(entry.get("max_model_len"))
    return ServerInfo("vllm", context) if context else None


def _transient(status: int) -> bool:
    return status in (408, 429) or status >= 500


class OpenAICompatProvider:
    name = "local"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 180,
        *,
        extra_body: str | None = None,
        context_tokens: int | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if context_tokens is not None and context_tokens <= 0:
            raise LLMConfigurationError("LLM_CONTEXT_TOKENS must be a positive number of tokens")
        self.model = model
        self.extra_body = parse_extra_body(extra_body)
        base = base_url.rstrip("/")
        root = base.removesuffix("/v1")
        self._url = f"{base}/chat/completions"
        self._props_url = f"{root}/props"
        self._models_url = f"{base}/models"
        self._api_key = api_key
        self._context_setting = context_tokens
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._server: ServerInfo | None = None
        self._server_lock = threading.Lock()

    # ------------------------------------------------------------------------------------------- the server

    def _headers(self) -> dict[str, str] | None:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None

    def _get_json(self, url: str) -> tuple[object, bool]:
        """(JSON body or ``None``, whether the answer is final). Transport errors and 5xx are not final."""
        try:
            response = self._client.get(url, headers=self._headers(), timeout=PROBE_TIMEOUT_S)
        except httpx.TransportError:
            return None, False
        if response.status_code >= 400:
            return None, not _transient(response.status_code)
        try:
            return response.json(), True
        except ValueError:
            return None, True

    def _probe(self) -> tuple[ServerInfo, bool]:
        props, props_final = self._get_json(self._props_url)
        if isinstance(props, dict) and any(key in props for key in _LLAMA_CPP_PROPS_KEYS):
            return _llama_cpp_info(props), True
        models, models_final = self._get_json(self._models_url)
        info = _models_info(models, self.model)
        if info is not None:
            return info, True
        return _UNKNOWN, props_final and models_final

    def server_info(self) -> ServerInfo:
        """What the server says about itself, asked once (a failed attempt is retried on the next call)."""
        if self._server is not None:
            return self._server
        with self._server_lock:
            if self._server is None:
                info, final = self._probe()
                logger.debug(
                    "llm server: kind=%s context_tokens=%s slots=%s final=%s",
                    info.kind,
                    info.context_tokens,
                    info.slots,
                    final,
                )
                if not final:
                    return info
                self._server = info
        return self._server

    def context_tokens(self) -> int | None:
        """Tokens one request may use: ``LLM_CONTEXT_TOKENS`` if set, else what the server reports, else ``None``."""
        if self._context_setting is not None:
            return self._context_setting
        return self.server_info().context_tokens

    def _extra(self) -> dict[str, Any]:
        if self.extra_body is not None:
            return self.extra_body
        return LLAMA_CPP_EXTRA_BODY if self.server_info().kind == "llama.cpp" else {}

    # ------------------------------------------------------------------------------------------- the request

    def complete_json(
        self, *, system: str, user: str, schema: dict[str, Any], max_tokens: int = 4096
    ) -> dict[str, Any]:
        response_format = {"type": "json_schema", "json_schema": {"name": "invoice", "strict": True, "schema": schema}}
        payload: dict[str, Any] = {
            "model": self.model,
            "response_format": response_format,
            "temperature": 0,
            "max_tokens": max_tokens,
            **self._extra(),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        try:
            response = self._client.post(self._url, json=payload, headers=self._headers())
        except httpx.TransportError as exc:  # connection refused, DNS, timeouts, dropped connection
            raise LLMUnavailableError(f"local LLM request failed: {type(exc).__name__}") from exc
        status = response.status_code
        if status >= 400:
            raise LLMError(f"local LLM returned HTTP {status}", transient=_transient(status))
        return _read_json(response)


def _read_json(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError as exc:
        raise LLMError("local LLM returned a body that is not JSON") from exc
    choices = body.get("choices") if isinstance(body, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices else None
    if not isinstance(choice, dict):
        raise LLMError("local LLM response had no choices")
    message = choice.get("message")
    message = message if isinstance(message, dict) else {}
    if message.get("refusal"):
        raise LLMRefusalError("local LLM declined the request")
    if choice.get("finish_reason") == "length":
        raise LLMOutputTruncatedError("local LLM stopped before finishing the JSON: length")
    content = message.get("content")
    return parse_json_object(content if isinstance(content, str) else None, "local LLM response")
