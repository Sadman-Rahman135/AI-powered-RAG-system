"""Final-review I3: the LLM input budget follows the model's context window (per request), capped by the setting."""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.extraction.llm import budget as budget_module
from app.extraction.llm.budget import (
    CHARS_PER_TOKEN,
    MIN_INPUT_CHARS,
    describe_input_budget,
    input_budget,
    prompt_overhead_tokens,
)
from app.extraction.llm.extractor import LLMExtractor
from app.extraction.llm.openai_compat_provider import OpenAICompatProvider
from app.extraction.llm.prompt import SYSTEM_PROMPT, wrap_document
from tests.unit.extraction.llm_support import LLAMA_CPP_PROPS, FakeServer, completion


class ContextProvider:
    """A provider that knows its context window (like the local one) and records what it is sent."""

    name = "local"
    model = "ctx-model"

    def __init__(self, context: int | None) -> None:
        self.context = context
        self.calls: list[dict[str, Any]] = []

    def context_tokens(self) -> int | None:
        return self.context

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any], max_tokens: int = 4096) -> Any:
        self.calls.append({"user": user, "max_tokens": max_tokens})
        return {}


class CloudProvider:
    name = "anthropic"
    model = "claude-x"

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any], max_tokens: int = 4096) -> Any:
        return {}


def local(server: FakeServer) -> OpenAICompatProvider:
    return OpenAICompatProvider("http://llm:8080/v1", "m", client=httpx.Client(transport=httpx.MockTransport(server)))


@pytest.fixture(autouse=True)
def _fresh_log_memory() -> None:
    budget_module.reset_reported()


def expected_chars(context: int, output: int, cap: int) -> int:
    return min(cap, max(MIN_INPUT_CHARS, math.floor((context - output - prompt_overhead_tokens()) * CHARS_PER_TOKEN)))


def test_prompt_overhead_covers_the_system_prompt_and_the_wrapper() -> None:
    overhead = prompt_overhead_tokens()
    assert overhead * CHARS_PER_TOKEN >= len(SYSTEM_PROMPT) + len(wrap_document("", header_only=True))
    assert overhead < 2000  # a small local model must still have room for the document


def test_the_budget_follows_the_context_window_of_the_model() -> None:
    small = input_budget(ContextProvider(8192), max_input_chars=60_000, max_output_tokens=4096)
    assert small.source == "model"
    assert small.context_tokens == 8192
    assert small.chars == expected_chars(8192, 4096, 60_000)
    assert 8_000 < small.chars < 12_500  # an 8k slot with 4k reserved for the answer: about 9-10 k characters

    bigger = input_budget(ContextProvider(32768), max_input_chars=60_000, max_output_tokens=4096)
    assert bigger.chars == expected_chars(32768, 4096, 60_000) > small.chars


def test_the_setting_caps_a_huge_context() -> None:
    huge = input_budget(ContextProvider(262_144), max_input_chars=60_000, max_output_tokens=4096)
    assert huge.chars == 60_000


def test_output_tokens_are_reserved() -> None:
    more_room = input_budget(ContextProvider(8192), max_input_chars=60_000, max_output_tokens=2048)
    assert more_room.chars == expected_chars(8192, 2048, 60_000)
    assert more_room.chars - input_budget(
        ContextProvider(8192), max_input_chars=60_000, max_output_tokens=4096
    ).chars == pytest.approx(2048 * CHARS_PER_TOKEN, abs=1)


def test_a_context_too_small_for_the_answer_keeps_a_minimum_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="app.extraction.llm.budget"):
        tiny = input_budget(ContextProvider(4096), max_input_chars=60_000, max_output_tokens=4096)
    assert tiny.chars == MIN_INPUT_CHARS
    assert any(record.levelno == logging.WARNING for record in caplog.records)


@pytest.mark.parametrize("provider", [CloudProvider(), ContextProvider(None)], ids=["cloud", "unknown-context"])
def test_without_a_known_context_the_setting_is_the_budget(provider: Any) -> None:
    budget = input_budget(provider, max_input_chars=20_000, max_output_tokens=4096)
    assert (budget.chars, budget.source, budget.context_tokens) == (20_000, "setting", None)


def test_the_effective_budget_is_logged_once_without_document_text(caplog: pytest.LogCaptureFixture) -> None:
    provider = ContextProvider(8192)
    with caplog.at_level(logging.INFO, logger="app.extraction.llm.budget"):
        for _ in range(3):
            input_budget(provider, max_input_chars=60_000, max_output_tokens=4096)
    records = [r for r in caplog.records if "llm input budget" in r.getMessage()]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "context_tokens=8192" in message and f"input_chars={expected_chars(8192, 4096, 60_000)}" in message


def test_the_live_llama_cpp_props_shape_gives_the_per_slot_budget() -> None:
    server = FakeServer(props=LLAMA_CPP_PROPS)
    provider = local(server)
    budget = input_budget(provider, max_input_chars=60_000, max_output_tokens=4096)
    assert budget.context_tokens == 8192
    assert budget.chars == expected_chars(8192, 4096, 60_000)


def test_describe_input_budget_reports_what_extraction_would_use() -> None:
    settings = Settings(_env_file=None, llm_max_input_chars=50_000, llm_max_output_tokens=2048)
    assert describe_input_budget(None, settings) is None
    described = describe_input_budget(ContextProvider(16384), settings)
    assert described is not None
    assert described.chars == expected_chars(16384, 2048, 50_000)
    assert described.output_tokens == 2048
    assert describe_input_budget(CloudProvider(), settings).chars == 50_000  # type: ignore[union-attr]


# ------------------------------------------------------------------------------------------- in the extractor


def test_the_extractor_cuts_long_documents_to_the_model_budget() -> None:
    provider = ContextProvider(8192)
    text = "".join(f"line {i:05d} of a long multi-page invoice\n" for i in range(1500))  # ~57 000 characters
    result = LLMExtractor(provider, max_output_tokens=4096).extract(text)
    assert "llm_input_truncated" in result.warnings
    body = provider.calls[0]["user"].split("<document>\n", 1)[1].split("\n</document>", 1)[0]
    budget = expected_chars(8192, 4096, Settings(_env_file=None).llm_max_input_chars)
    assert budget <= len(body) < budget + 100  # the budget plus the omission marker


def test_an_explicit_budget_still_wins() -> None:
    provider = ContextProvider(8192)
    LLMExtractor(provider, max_input_chars=500).extract("x" * 400)
    assert "characters omitted" not in provider.calls[0]["user"]
    LLMExtractor(provider, max_input_chars=300).extract("x" * 400)
    assert "characters omitted" in provider.calls[1]["user"]


def test_a_cloud_provider_keeps_the_configured_budget() -> None:
    calls: list[str] = []

    class Recording(CloudProvider):
        def complete_json(self, *, system: str, user: str, schema: dict[str, Any], max_tokens: int = 4096) -> Any:
            calls.append(user)
            return {}

    text = "y" * 30_000
    LLMExtractor(Recording(), max_output_tokens=4096).extract(text)  # LLM_MAX_INPUT_CHARS default is 60 000
    assert "characters omitted" not in calls[0]


def test_the_local_provider_end_to_end_against_a_llama_cpp_shaped_server() -> None:
    server = FakeServer(chat=lambda request: httpx.Response(200, json=completion("{}")), props=LLAMA_CPP_PROPS)
    provider = local(server)
    result = LLMExtractor(provider, max_output_tokens=4096).extract("z" * 20_000)
    assert "llm_input_truncated" in result.warnings  # 20 000 characters do not fit an 8k slot next to 4k of output
    assert server.paths().count("GET /props") == 1


@pytest.mark.live_llm
def test_live_local_server_reports_its_context_and_yields_a_model_budget(caplog: pytest.LogCaptureFixture) -> None:
    base_url = os.environ.get("LIVE_LLM_BASE_URL")
    if not base_url:
        pytest.skip("set LIVE_LLM_BASE_URL (for example http://localhost:8080/v1) to run against a real local server")
    provider = OpenAICompatProvider(
        base_url, os.environ.get("LIVE_LLM_MODEL", "qwen3.5-9b"), api_key=os.environ.get("LIVE_LLM_API_KEY") or None
    )
    info = provider.server_info()
    with caplog.at_level(logging.INFO, logger="app.extraction.llm.budget"):
        budget = input_budget(provider, max_input_chars=60_000, max_output_tokens=4096)
    assert info.context_tokens is not None, info
    assert budget.source == "model"
    assert budget.chars == expected_chars(info.context_tokens, 4096, 60_000)
    assert any("llm input budget" in record.getMessage() for record in caplog.records)
