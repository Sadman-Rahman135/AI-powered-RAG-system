"""Pick the configured LLM provider.

``LLM_PROVIDER=auto`` uses the first provider that is configured: Claude (``ANTHROPIC_API_KEY``), then Gemini
(``GEMINI_API_KEY``), then a local OpenAI-compatible server (``LLM_BASE_URL``); with none of them there is no LLM
and extraction runs on rules alone. Naming a provider explicitly without its configuration is a deployment mistake
and raises ``LLMConfigurationError``, as does an invalid ``LLM_EXTRA_BODY`` / ``LLM_CONTEXT_TOKENS``. Secrets are
unwrapped only at the moment a client is constructed. Building a provider makes no network call.
"""

from __future__ import annotations

from pydantic import SecretStr

from app.config import Settings
from app.extraction.llm.base import LLMConfigurationError, LLMProvider

# The provider modules import their SDKs, so they are imported only when that provider is actually used.


def _anthropic(settings: Settings, api_key: SecretStr) -> LLMProvider:
    from app.extraction.llm.anthropic_provider import AnthropicProvider

    return AnthropicProvider(
        api_key=api_key.get_secret_value(),
        model=settings.anthropic_model,
        effort=settings.anthropic_effort,
        timeout=settings.llm_timeout_s,
    )


def _gemini(settings: Settings, api_key: SecretStr) -> LLMProvider:
    from app.extraction.llm.gemini_provider import GeminiProvider

    return GeminiProvider(
        api_key=api_key.get_secret_value(), model=settings.gemini_model, timeout=settings.llm_timeout_s
    )


def _local(settings: Settings, base_url: str) -> LLMProvider:
    from app.extraction.llm.openai_compat_provider import OpenAICompatProvider

    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    return OpenAICompatProvider(
        base_url=base_url,
        model=settings.llm_model,
        api_key=api_key,
        timeout=settings.llm_timeout_s,
        extra_body=settings.llm_extra_body,
        context_tokens=settings.llm_context_tokens,
    )


def get_llm_provider(settings: Settings) -> LLMProvider | None:
    """The provider selected by ``settings.llm_provider``, or ``None`` when extraction should run without an LLM."""
    choice = settings.llm_provider
    if choice == "none":
        return None
    anthropic_key, gemini_key, base_url = settings.anthropic_api_key, settings.gemini_api_key, settings.llm_base_url
    if choice == "auto":
        if anthropic_key:
            return _anthropic(settings, anthropic_key)
        if gemini_key:
            return _gemini(settings, gemini_key)
        if base_url:
            return _local(settings, base_url)
        return None
    if choice == "anthropic":
        if not anthropic_key:
            raise LLMConfigurationError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY to be set")
        return _anthropic(settings, anthropic_key)
    if choice == "gemini":
        if not gemini_key:
            raise LLMConfigurationError("LLM_PROVIDER=gemini requires GEMINI_API_KEY to be set")
        return _gemini(settings, gemini_key)
    if not base_url:
        raise LLMConfigurationError(
            "LLM_PROVIDER=local requires LLM_BASE_URL to be set (e.g. http://localhost:8080/v1)"
        )
    return _local(settings, base_url)
