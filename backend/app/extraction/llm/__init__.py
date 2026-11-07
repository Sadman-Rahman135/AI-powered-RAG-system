"""Pluggable LLM extraction layer: providers (Claude, Gemini, local OpenAI-compatible), prompt, extractor.

Typical use: ``provider = get_llm_provider(settings)``; if it is not ``None``,
``ground_result``-checked records come from ``LLMExtractor(provider).extract(text)``. How much document text one
request carries follows the model's context window: ``describe_input_budget(provider, settings)``.
"""

from app.extraction.llm.base import (
    LLMConfigurationError,
    LLMError,
    LLMOutputTruncatedError,
    LLMProvider,
    LLMRefusalError,
    LLMUnavailableError,
)
from app.extraction.llm.budget import InputBudget, describe_input_budget
from app.extraction.llm.extractor import LLMExtractor
from app.extraction.llm.factory import get_llm_provider
from app.extraction.llm.prompt import PROMPT_VERSION

__all__ = [
    "PROMPT_VERSION",
    "InputBudget",
    "LLMConfigurationError",
    "LLMError",
    "LLMExtractor",
    "LLMOutputTruncatedError",
    "LLMProvider",
    "LLMRefusalError",
    "LLMUnavailableError",
    "describe_input_budget",
    "get_llm_provider",
]
