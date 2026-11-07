"""Which LLM the extraction uses, as reported by stats / ready / config - read from the provider factory without any
network call. Building a provider creates an HTTP client, so the answer is kept until the settings change."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal

from app.config import Settings
from app.extraction.llm.base import LLMConfigurationError
from app.extraction.llm.factory import get_llm_provider

LlmState = Literal["none", "configured", "misconfigured"]


@dataclass(frozen=True, slots=True)
class LlmDescription:
    provider: str  # "none" when extraction runs on rules alone
    model: str | None
    state: LlmState


_lock = threading.Lock()
_cached: tuple[Settings, LlmDescription] | None = None


def describe_llm(settings: Settings) -> LlmDescription:
    global _cached
    with _lock:
        if _cached is not None and _cached[0] is settings:
            return _cached[1]
        try:
            provider = get_llm_provider(settings)
        except LLMConfigurationError:
            description = LlmDescription(provider=settings.llm_provider, model=None, state="misconfigured")
        else:
            description = (
                LlmDescription(provider="none", model=None, state="none")
                if provider is None
                else LlmDescription(provider=provider.name, model=provider.model, state="configured")
            )
        _cached = (settings, description)
        return description
