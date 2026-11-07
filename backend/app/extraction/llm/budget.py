"""How much document text one LLM request may carry: it follows the model's context window.

For a provider that knows its context window per request (the local OpenAI-compatible server: ``LLM_CONTEXT_TOKENS``
or what llama.cpp / vLLM report), the budget is

    (context tokens - LLM_MAX_OUTPUT_TOKENS - prompt overhead) x 3.0 characters per token

capped by ``LLM_MAX_INPUT_CHARS`` and never below ``MIN_INPUT_CHARS``. Three characters per token is deliberately
conservative (English invoice text averages about four; digits, symbols and OCR noise tokenise worse), and the whole
output budget is reserved, so a document that fits is never cut off by the server. Claude and Gemini (and a server
that says nothing about itself) use ``LLM_MAX_INPUT_CHARS`` as is.

The effective budget is logged once per provider/model/budget at INFO (sizes only, never document text).
``describe_input_budget`` returns the same figures for status endpoints.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from app.config import Settings
from app.extraction.llm.base import LLMProvider
from app.extraction.llm.prompt import SYSTEM_PROMPT, wrap_document

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 3.0
TEMPLATE_MARGIN_TOKENS = 128  # chat-template tokens, page markers and the "characters omitted" marker
MIN_INPUT_CHARS = 2_000  # below this the model sees too little to be useful; a warning says why

BudgetSource = Literal["model", "setting"]


@runtime_checkable
class ContextWindowAware(Protocol):
    """A provider that can say how many tokens one request may use (``None`` when it does not know)."""

    def context_tokens(self) -> int | None: ...


@dataclass(frozen=True, slots=True)
class InputBudget:
    """The document characters one request carries and where the figure comes from.

    ``source`` is ``"model"`` when it was derived from ``context_tokens`` (capped by the setting) and ``"setting"``
    when ``LLM_MAX_INPUT_CHARS`` was used because the context window is not known.
    """

    chars: int
    source: BudgetSource
    context_tokens: int | None
    output_tokens: int
    overhead_tokens: int


def prompt_overhead_tokens() -> int:
    """Tokens the prompt takes besides the document: the system prompt and the user-prompt wrapper (the longer,
    header-only variant), plus a margin for the chat template and page markers."""
    fixed = len(SYSTEM_PROMPT) + len(wrap_document("", header_only=True))
    return math.ceil(fixed / CHARS_PER_TOKEN) + TEMPLATE_MARGIN_TOKENS


_reported: set[tuple[str, str, int, int | None, int]] = set()
_reported_lock = threading.Lock()


def reset_reported() -> None:
    """Forget which budgets were logged (tests)."""
    with _reported_lock:
        _reported.clear()


def _report(provider: LLMProvider, budget: InputBudget, cramped: bool) -> None:
    key = (provider.name, provider.model, budget.chars, budget.context_tokens, budget.output_tokens)
    with _reported_lock:
        if key in _reported:
            return
        _reported.add(key)
    logger.info(
        "llm input budget: provider=%s model=%s context_tokens=%s output_tokens=%d overhead_tokens=%d "
        "input_chars=%d source=%s",
        provider.name,
        provider.model,
        budget.context_tokens if budget.context_tokens is not None else "unknown",
        budget.output_tokens,
        budget.overhead_tokens,
        budget.chars,
        budget.source,
    )
    if cramped:
        logger.warning(
            "llm context of %d tokens leaves almost no room for the document next to %d output tokens; "
            "lower LLM_MAX_OUTPUT_TOKENS or give the model a larger context (LLM_CTX / LLM_CONTEXT_TOKENS)",
            budget.context_tokens,
            budget.output_tokens,
        )


def input_budget(provider: LLMProvider, *, max_input_chars: int, max_output_tokens: int) -> InputBudget:
    """The input budget for ``provider`` (see the module docstring). May ask a local server about itself once."""
    overhead = prompt_overhead_tokens()
    context = provider.context_tokens() if isinstance(provider, ContextWindowAware) else None
    cramped = False
    if context is None:
        budget = InputBudget(max_input_chars, "setting", None, max_output_tokens, overhead)
    else:
        room = math.floor((context - max_output_tokens - overhead) * CHARS_PER_TOKEN)
        cramped = room < MIN_INPUT_CHARS
        chars = min(max_input_chars, max(MIN_INPUT_CHARS, room))
        budget = InputBudget(chars, "model", context, max_output_tokens, overhead)
    _report(provider, budget, cramped)
    return budget


def describe_input_budget(provider: LLMProvider | None, settings: Settings) -> InputBudget | None:
    """The budget extraction would use with ``provider`` now (``None`` without an LLM), for ``/stats`` or
    ``/config``. For a local server this may make the one-off ``/props`` request."""
    if provider is None:
        return None
    return input_budget(
        provider, max_input_chars=settings.llm_max_input_chars, max_output_tokens=settings.llm_max_output_tokens
    )
