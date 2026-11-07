"""Builders shared by the LLM tests (grounding, merge, extractor) and a fake OpenAI-compatible server."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx

from app.domain.schemas import (
    INVOICE_FIELDS,
    ExtractionResult,
    ExtractionSource,
    FieldMeta,
    InvoiceData,
    field_is_present,
)

SOURCES = {
    "rules": ExtractionSource.RULES,
    "llm": ExtractionSource.LLM,
    "hybrid": ExtractionSource.HYBRID,
}


def make_result(
    method: str = "llm",
    *,
    confidence: float = 0.85,
    confidences: Mapping[str, float] | None = None,
    evidence: Mapping[str, str] | None = None,
    warnings: tuple[str, ...] = (),
    **fields: Any,
) -> ExtractionResult:
    """An ``ExtractionResult`` whose every present field carries a ``FieldMeta`` from ``method``."""
    data = InvoiceData(**fields)
    meta = {
        name: FieldMeta(
            confidence=(confidences or {}).get(name, confidence),
            source=SOURCES[method],
            evidence=(evidence or {}).get(name),
        )
        for name in INVOICE_FIELDS
        if field_is_present(data, name)
    }
    result = ExtractionResult(data=data, meta=meta, method=method, warnings=list(warnings))
    result.compute_overall_confidence()
    return result


# ------------------------------------------------------------------ a fake OpenAI-compatible server


def completion(content: str | None, finish_reason: str = "stop", **message: Any) -> dict[str, Any]:
    return {
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content, **message},
            }
        ]
    }


LLAMA_CPP_PROPS: dict[str, Any] = {  # the shape of llama.cpp b11580 GET /props (-c 16384 --parallel 2)
    "default_generation_settings": {"params": {"temperature": 0.8}, "n_ctx": 8192},
    "total_slots": 2,
    "model_alias": "qwen3-4b",
    "build_info": "b11580-4413a1924",
}


class FakeServer:
    """An OpenAI-compatible server: ``chat`` answers POST .../chat/completions; ``props`` answers GET /props and
    ``models`` GET /v1/models (404 by default, like a server that has neither). Records every request."""

    def __init__(self, chat: Any = None, *, props: Any = None, models: Any = None) -> None:
        self.chat = chat or (lambda request: httpx.Response(200, json=completion('{"a": "x"}')))
        self.props = props
        self.models = models
        self.requests: list[httpx.Request] = []

    @staticmethod
    def _answer(spec: Any, request: httpx.Request) -> httpx.Response:
        if spec is None:
            return httpx.Response(404, json={"error": {"message": "File Not Found"}})
        if callable(spec):
            return spec(request)
        return httpx.Response(200, json=spec)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.method == "GET" and path.endswith("/props"):
            return self._answer(self.props, request)
        if request.method == "GET" and path.endswith("/models"):
            return self._answer(self.models, request)
        return self.chat(request)

    def paths(self) -> list[str]:
        return [f"{request.method} {request.url.path}" for request in self.requests]

    def chat_bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.url.path.endswith("/chat/completions")]
