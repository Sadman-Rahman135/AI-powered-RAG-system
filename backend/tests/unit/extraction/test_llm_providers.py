"""LLM provider adapters (request shape, response parsing, error mapping) and the provider factory.

No network: the Anthropic and Gemini SDK clients are replaced by recording fakes, the local OpenAI-compatible server
by ``httpx.MockTransport``.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import httpx2  # the HTTP library the installed anthropic SDK builds its exceptions on
import pytest
from anthropic.resources.beta.messages import Messages as BetaMessages
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import SecretStr

from app.config import Settings
from app.extraction.llm.anthropic_provider import AnthropicProvider
from app.extraction.llm.base import (
    LLMConfigurationError,
    LLMError,
    LLMOutputTruncatedError,
    LLMProvider,
    LLMRefusalError,
    LLMUnavailableError,
)
from app.extraction.llm.factory import get_llm_provider
from app.extraction.llm.gemini_provider import GeminiProvider
from app.extraction.llm.openai_compat_provider import OpenAICompatProvider
from tests.unit.extraction.llm_support import LLAMA_CPP_PROPS, FakeServer, completion

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["a"],
    "properties": {"a": {"type": "string"}},
}


# ------------------------------------------------------------------------------------------------ Anthropic fakes


class FakeAnthropic:
    """Records ``client.beta.messages.create`` calls; returns ``response`` or raises ``error``."""

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._response = response
        self._error = error
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


def anthropic_message(
    text: str | None = '{"a": "x"}', stop_reason: str = "end_turn", request_id: str | None = "req_test_1"
) -> SimpleNamespace:
    content: list[SimpleNamespace] = []
    if text is not None:
        content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=content,
        stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
        _request_id=request_id,
    )


_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error(cls: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    response = httpx2.Response(status, request=_REQUEST, headers={"request-id": "req_err_9"})
    body = {"type": "error", "error": {"type": "some_error", "message": "SECRET-DOCUMENT-TEXT"}}
    return cls("provider said: SECRET-DOCUMENT-TEXT", response=response, body=body)


def anthropic_provider(client: FakeAnthropic, **kwargs: Any) -> AnthropicProvider:
    return AnthropicProvider(api_key="sk-ant-test", client=client, **kwargs)


# ---------------------------------------------------------------------------------------------------- Anthropic


def test_anthropic_request_shape() -> None:
    client = FakeAnthropic(anthropic_message())
    provider = anthropic_provider(client, model="claude-opus-5-5", effort="medium")

    out = provider.complete_json(system="SYS", user="USER", schema=SCHEMA, max_tokens=1234)

    assert out == {"a": "x"}
    assert provider.name == "anthropic"
    assert provider.model == "claude-opus-5-5"
    (call,) = client.calls
    assert call["model"] == "claude-opus-5-5"
    assert call["max_tokens"] == 1234
    assert call["system"] == "SYS"
    assert call["messages"] == [{"role": "user", "content": "USER"}]
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["fallbacks"] == "default"
    assert call["output_config"] == {"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}}
    # current models reject sampling parameters and an explicit "disabled" thinking block
    for forbidden in ("temperature", "top_p", "top_k", "thinking"):
        assert forbidden not in call


def test_anthropic_request_arguments_exist_in_the_installed_sdk() -> None:
    """The fake client accepts anything, so check the keyword names against the real SDK signature."""
    client = FakeAnthropic(anthropic_message())
    anthropic_provider(client).complete_json(system="s", user="u", schema=SCHEMA)
    parameters = inspect.signature(BetaMessages.create).parameters
    assert set(client.calls[0]) <= set(parameters)


def test_anthropic_default_max_tokens_and_effort() -> None:
    client = FakeAnthropic(anthropic_message())
    anthropic_provider(client).complete_json(system="s", user="u", schema=SCHEMA)
    assert client.calls[0]["max_tokens"] == 4096
    assert client.calls[0]["output_config"]["effort"] == "medium"


def test_anthropic_client_is_built_with_timeout_and_two_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class Recorder:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

    monkeypatch.setattr(anthropic, "Anthropic", Recorder)
    AnthropicProvider(api_key="sk-ant-test", timeout=77)
    assert seen == {"api_key": "sk-ant-test", "timeout": 77, "max_retries": 2}


def test_anthropic_reads_the_first_text_block() -> None:
    message = anthropic_message()
    message.content = [
        SimpleNamespace(type="thinking", thinking="..."),
        SimpleNamespace(type="text", text='{"a": "y"}'),
    ]
    out = anthropic_provider(FakeAnthropic(message)).complete_json(system="s", user="u", schema=SCHEMA)
    assert out == {"a": "y"}


def test_anthropic_refusal_raises() -> None:
    client = FakeAnthropic(anthropic_message(text=None, stop_reason="refusal"))
    with pytest.raises(LLMRefusalError) as info:
        anthropic_provider(client).complete_json(system="s", user="u", schema=SCHEMA)
    assert info.value.transient is False
    assert "req_test_1" in str(info.value)


def test_anthropic_stop_reason_is_checked_before_content() -> None:
    """A refusal can carry partial text; it must still be a refusal, never parsed."""
    client = FakeAnthropic(anthropic_message(text='{"a": "partial', stop_reason="refusal"))
    with pytest.raises(LLMRefusalError):
        anthropic_provider(client).complete_json(system="s", user="u", schema=SCHEMA)


@pytest.mark.parametrize("stop_reason", ["max_tokens", "model_context_window_exceeded"])
def test_anthropic_truncated_output_is_a_permanent_error(stop_reason: str) -> None:
    # The text is complete, valid JSON: only the stop reason can make this an error.
    client = FakeAnthropic(anthropic_message(text='{"a": "x"}', stop_reason=stop_reason))
    with pytest.raises(LLMOutputTruncatedError) as info:  # the extractor retries these without line items
        anthropic_provider(client).complete_json(system="s", user="u", schema=SCHEMA)
    assert info.value.transient is False
    assert not isinstance(info.value, LLMRefusalError)
    assert stop_reason in str(info.value)


@pytest.mark.parametrize("text", ["not json", "[1, 2]", "", None])
def test_anthropic_unusable_content_raises_permanent_llm_error(text: str | None) -> None:
    client = FakeAnthropic(anthropic_message(text=text))
    with pytest.raises(LLMError) as info:
        anthropic_provider(client).complete_json(system="s", user="u", schema=SCHEMA)
    assert info.value.transient is False


@pytest.mark.parametrize(
    "error",
    [
        status_error(anthropic.RateLimitError, 429),
        status_error(anthropic.InternalServerError, 500),
        status_error(anthropic.InternalServerError, 502),
        status_error(anthropic.ServiceUnavailableError, 503),
        status_error(anthropic.DeadlineExceededError, 504),
        status_error(anthropic.OverloadedError, 529),
        anthropic.APIConnectionError(request=_REQUEST),
        anthropic.APITimeoutError(request=_REQUEST),
    ],
    ids=lambda e: type(e).__name__,
)
def test_anthropic_transient_errors(error: Exception) -> None:
    with pytest.raises(LLMError) as info:
        anthropic_provider(FakeAnthropic(error=error)).complete_json(system="s", user="u", schema=SCHEMA)
    assert info.value.transient is True
    assert info.value.__cause__ is error


@pytest.mark.parametrize(
    "error",
    [
        status_error(anthropic.AuthenticationError, 401),
        status_error(anthropic.PermissionDeniedError, 403),
        status_error(anthropic.BadRequestError, 400),
        status_error(anthropic.NotFoundError, 404),
        status_error(anthropic.UnprocessableEntityError, 422),
        status_error(anthropic.RequestTooLargeError, 413),
    ],
    ids=lambda e: type(e).__name__,
)
def test_anthropic_permanent_errors(error: Exception) -> None:
    with pytest.raises(LLMError) as info:
        anthropic_provider(FakeAnthropic(error=error)).complete_json(system="s", user="u", schema=SCHEMA)
    assert info.value.transient is False
    assert not isinstance(info.value, LLMRefusalError)


def test_anthropic_connection_failures_are_unavailable_errors() -> None:
    for error in (anthropic.APIConnectionError(request=_REQUEST), anthropic.APITimeoutError(request=_REQUEST)):
        with pytest.raises(LLMUnavailableError):
            anthropic_provider(FakeAnthropic(error=error)).complete_json(system="s", user="u", schema=SCHEMA)


def test_anthropic_error_messages_carry_ids_but_never_provider_text() -> None:
    error = status_error(anthropic.BadRequestError, 400)
    with pytest.raises(LLMError) as info:
        anthropic_provider(FakeAnthropic(error=error)).complete_json(system="s", user="u", schema=SCHEMA)
    message = str(info.value)
    assert "400" in message
    assert "req_err_9" in message
    assert "SECRET-DOCUMENT-TEXT" not in message
    assert "sk-ant-test" not in message


# ------------------------------------------------------------------------------------------- OpenAI-compatible


def local_provider(handler: Any, **kwargs: Any) -> OpenAICompatProvider:
    server = handler if isinstance(handler, FakeServer) else FakeServer(handler)
    client = httpx.Client(transport=httpx.MockTransport(server))
    return OpenAICompatProvider("http://llm.local:8080/v1/", "qwen3.5-9b", client=client, **kwargs)


def test_openai_compat_request_and_parse() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion('{"a": "x"}'))

    provider = local_provider(handler, api_key="sekret-token")
    out = provider.complete_json(system="SYS", user="USER", schema=SCHEMA, max_tokens=900)

    assert out == {"a": "x"}
    assert provider.name == "local"
    assert provider.model == "qwen3.5-9b"
    assert seen["method"] == "POST"
    assert seen["url"] == "http://llm.local:8080/v1/chat/completions"
    assert seen["headers"]["authorization"] == "Bearer sekret-token"
    # a server that does not identify as llama.cpp gets the plain OpenAI request, no extra fields
    assert seen["body"] == {
        "model": "qwen3.5-9b",
        "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "invoice", "strict": True, "schema": SCHEMA},
        },
        "temperature": 0,
        "max_tokens": 900,
    }


# ------------------------------------------------------------------- extra request fields and server identity (I6)


def test_the_thinking_switch_is_sent_only_to_a_server_that_identifies_as_llama_cpp() -> None:
    llama = FakeServer(props=LLAMA_CPP_PROPS)
    local_provider(llama).complete_json(system="s", user="u", schema=SCHEMA)
    assert llama.chat_bodies()[0]["chat_template_kwargs"] == {"enable_thinking": False}

    strict_openai = FakeServer()  # no /props, no /models
    local_provider(strict_openai).complete_json(system="s", user="u", schema=SCHEMA)
    assert "chat_template_kwargs" not in strict_openai.chat_bodies()[0]
    assert strict_openai.paths()[-1] == "POST /v1/chat/completions"


def test_llama_cpp_is_also_recognised_from_its_model_list_behind_a_v1_only_proxy() -> None:
    models = {"object": "list", "data": [{"id": "qwen3.5-9b", "owned_by": "llamacpp", "meta": {"n_ctx": 8192}}]}
    server = FakeServer(models=models)
    provider = local_provider(server)
    provider.complete_json(system="s", user="u", schema=SCHEMA)
    assert server.chat_bodies()[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert provider.context_tokens() == 8192


def test_extra_body_setting_is_merged_into_every_request_and_replaces_the_automatic_choice() -> None:
    server = FakeServer(props=LLAMA_CPP_PROPS)
    provider = local_provider(server, extra_body='{"top_k": 1, "temperature": 0.1, "reasoning": {"effort": "low"}}')
    provider.complete_json(system="s", user="u", schema=SCHEMA, max_tokens=50)
    body = server.chat_bodies()[0]
    assert (body["top_k"], body["temperature"], body["reasoning"]) == (1, 0.1, {"effort": "low"})  # set keys win
    assert body["max_tokens"] == 50 and body["model"] == "qwen3.5-9b"
    assert "chat_template_kwargs" not in body  # an explicit extra body replaces the llama.cpp default
    assert not any(path.endswith("/props") for path in server.paths())  # nothing to decide, so no probe


def test_an_empty_extra_body_object_sends_nothing_extra_even_to_llama_cpp() -> None:
    server = FakeServer(props=LLAMA_CPP_PROPS)
    local_provider(server, extra_body="{}").complete_json(system="s", user="u", schema=SCHEMA)
    assert "chat_template_kwargs" not in server.chat_bodies()[0]


def test_a_blank_extra_body_means_automatic() -> None:
    server = FakeServer(props=LLAMA_CPP_PROPS)
    local_provider(server, extra_body="  ").complete_json(system="s", user="u", schema=SCHEMA)
    assert server.chat_bodies()[0]["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ('{"chat_template_kwargs": {"enable_thinking": false}', "not valid JSON"),
        ("[1, 2]", "JSON object"),
        ('"text"', "JSON object"),
        ('{"messages": [], "model": "x"}', "messages, model"),
    ],
)
def test_an_invalid_extra_body_is_a_configuration_error_at_construction(raw: str, message: str) -> None:
    with pytest.raises(LLMConfigurationError) as info:
        OpenAICompatProvider("http://x/v1", "m", client=httpx.Client(), extra_body=raw)
    assert "LLM_EXTRA_BODY" in str(info.value)
    assert message in str(info.value)


def test_the_server_is_asked_about_itself_only_once() -> None:
    server = FakeServer(props=LLAMA_CPP_PROPS)
    provider = local_provider(server)
    for _ in range(3):
        provider.complete_json(system="s", user="u", schema=SCHEMA)
    assert provider.context_tokens() == 8192
    assert server.paths().count("GET /props") == 1
    assert server.paths().count("POST /v1/chat/completions") == 3


def test_a_server_that_cannot_be_reached_for_props_is_asked_again_later() -> None:
    calls = {"props": 0}

    def flaky_props(request: httpx.Request) -> httpx.Response:
        calls["props"] += 1
        if calls["props"] == 1:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json=LLAMA_CPP_PROPS)

    server = FakeServer(props=flaky_props)
    provider = local_provider(server)
    provider.complete_json(system="s", user="u", schema=SCHEMA)  # the probe failed: the request still goes out
    assert "chat_template_kwargs" not in server.chat_bodies()[0]
    assert provider.context_tokens() == 8192  # asked again, now answered
    provider.complete_json(system="s", user="u", schema=SCHEMA)
    assert server.chat_bodies()[1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert calls["props"] == 2


@pytest.mark.parametrize(
    ("props", "expected"),
    [
        (LLAMA_CPP_PROPS, 8192),  # per-slot context as current builds report it
        ({"n_ctx": 16384, "total_slots": 2, "build_info": "old"}, 8192),  # older builds: total context / slots
        ({"n_ctx": 16384, "total_slots": 4, "default_generation_settings": {"n_ctx": 8192}}, 4096),  # the smaller
        ({"default_generation_settings": {"n_ctx": 0}, "total_slots": 1}, None),  # nonsense is ignored
        ({"build_info": "b1"}, None),
    ],
)
def test_context_window_per_request_from_llama_cpp_props(props: dict[str, Any], expected: int | None) -> None:
    assert local_provider(FakeServer(props=props)).context_tokens() == expected


def test_context_window_from_a_vllm_model_list() -> None:
    models = {
        "object": "list",
        "data": [{"id": "other", "max_model_len": 4096}, {"id": "qwen3.5-9b", "max_model_len": 32768}],
    }
    server = FakeServer(models=models)
    provider = local_provider(server)
    assert provider.context_tokens() == 32768
    provider.complete_json(system="s", user="u", schema=SCHEMA)
    assert "chat_template_kwargs" not in server.chat_bodies()[0]  # vLLM is not llama.cpp


def test_the_context_setting_wins_and_needs_no_request() -> None:
    server = FakeServer(props=LLAMA_CPP_PROPS)
    provider = local_provider(server, context_tokens=32768)
    assert provider.context_tokens() == 32768
    assert server.requests == []
    with pytest.raises(LLMConfigurationError):
        OpenAICompatProvider("http://x/v1", "m", client=httpx.Client(), context_tokens=0)


@pytest.mark.parametrize(
    ("base_url", "props_url"),
    [
        ("http://llm:8080/v1", "http://llm:8080/props"),
        ("http://llm:8080/v1/", "http://llm:8080/props"),
        ("http://llm:8080", "http://llm:8080/props"),
        ("https://gateway.example/llm/v1", "https://gateway.example/llm/props"),
    ],
)
def test_props_are_read_next_to_the_v1_api(base_url: str, props_url: str) -> None:
    server = FakeServer(props=LLAMA_CPP_PROPS)
    provider = OpenAICompatProvider(base_url, "m", client=httpx.Client(transport=httpx.MockTransport(server)))
    provider.context_tokens()
    assert str(server.requests[0].url) == props_url


def test_the_probe_carries_the_api_key_and_never_logs_it(caplog: pytest.LogCaptureFixture) -> None:
    server = FakeServer(props=LLAMA_CPP_PROPS)
    with caplog.at_level("DEBUG"):
        local_provider(server, api_key="sekret-token").context_tokens()
    assert server.requests[0].headers["authorization"] == "Bearer sekret-token"
    assert "sekret-token" not in caplog.text


def test_openai_compat_sends_no_authorization_header_without_a_key() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json=completion('{"a": "x"}'))

    local_provider(handler).complete_json(system="s", user="u", schema=SCHEMA)
    assert "authorization" not in seen["headers"]


@pytest.mark.parametrize("fence", ["```json\n{payload}\n```", "```\n{payload}\n```", "  ```JSON\n{payload}\n```  "])
def test_openai_compat_strips_code_fences(fence: str) -> None:
    content = fence.format(payload='{"a": "fenced"}')
    provider = local_provider(lambda request: httpx.Response(200, json=completion(content)))
    assert provider.complete_json(system="s", user="u", schema=SCHEMA) == {"a": "fenced"}


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_openai_compat_http_error_transient(status: int) -> None:
    provider = local_provider(lambda request: httpx.Response(status, json={"error": {"message": "SECRET-DOC"}}))
    with pytest.raises(LLMError) as info:
        provider.complete_json(system="s", user="u", schema=SCHEMA)
    assert info.value.transient is True
    assert str(status) in str(info.value)
    assert "SECRET-DOC" not in str(info.value)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_openai_compat_http_error_permanent(status: int) -> None:
    provider = local_provider(lambda request: httpx.Response(status, json={"error": {"message": "nope"}}))
    with pytest.raises(LLMError) as info:
        provider.complete_json(system="s", user="u", schema=SCHEMA)
    assert info.value.transient is False


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("slow"),
        httpx.ConnectTimeout("slow"),
        httpx.RemoteProtocolError("eof"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_openai_compat_connection_failures_are_transient_unavailable_errors(failure: Exception) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise failure

    with pytest.raises(LLMUnavailableError) as info:
        local_provider(handler).complete_json(system="s", user="u", schema=SCHEMA)
    assert info.value.transient is True


def test_openai_compat_truncated_output_is_a_permanent_error() -> None:
    # Complete, valid JSON: only the finish reason can make this an error.
    provider = local_provider(
        lambda request: httpx.Response(200, json=completion('{"a": "x"}', finish_reason="length"))
    )
    with pytest.raises(LLMOutputTruncatedError) as info:
        provider.complete_json(system="s", user="u", schema=SCHEMA)
    assert info.value.transient is False
    assert "length" in str(info.value)


def test_openai_compat_refusal_field_raises_refusal() -> None:
    provider = local_provider(
        lambda request: httpx.Response(200, json=completion(None, refusal="I cannot help with that"))
    )
    with pytest.raises(LLMRefusalError):
        provider.complete_json(system="s", user="u", schema=SCHEMA)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>gateway</html>"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json=completion("")),
        httpx.Response(200, json=completion("not json at all")),
        httpx.Response(200, json=completion("[1, 2, 3]")),
        httpx.Response(200, json={"unexpected": True}),
    ],
    ids=["html", "no-choices", "empty-content", "bad-json", "json-array", "no-choices-key"],
)
def test_openai_compat_unusable_response_raises_permanent_llm_error(response: httpx.Response) -> None:
    provider = local_provider(lambda request: response)
    with pytest.raises(LLMError) as info:
        provider.complete_json(system="s", user="u", schema=SCHEMA)
    assert info.value.transient is False


def test_openai_compat_builds_its_own_client_with_the_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    real_client = httpx.Client

    def recording_client(**kwargs: Any) -> httpx.Client:
        seen.update(kwargs)
        return real_client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=completion("{}"))))

    monkeypatch.setattr(httpx, "Client", recording_client)
    provider = OpenAICompatProvider("http://llm.local/v1", "m", timeout=42)
    assert seen == {"timeout": 42}
    assert provider.complete_json(system="s", user="u", schema=SCHEMA) == {}


# ------------------------------------------------------------------------------------------------------ Gemini


class FakeGenai:
    """Records ``client.models.generate_content`` calls."""

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._response = response
        self._error = error
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


def gemini_response(
    text: str | None = '{"a": "x"}',
    finish_reason: genai_types.FinishReason = genai_types.FinishReason.STOP,
    block_reason: genai_types.BlockedReason | None = None,
) -> SimpleNamespace:
    feedback = SimpleNamespace(block_reason=block_reason) if block_reason else None
    return SimpleNamespace(
        text=text, candidates=[SimpleNamespace(finish_reason=finish_reason)], prompt_feedback=feedback
    )


def api_error(cls: type[genai_errors.APIError], code: int) -> genai_errors.APIError:
    return cls(code, {"error": {"code": code, "message": "SECRET-DOCUMENT-TEXT", "status": "SOME_STATUS"}})


def test_gemini_request_and_parse() -> None:
    client = FakeGenai(gemini_response())
    provider = GeminiProvider(api_key="g-key", model="gemini-3-flash-preview", client=client)

    out = provider.complete_json(system="SYS", user="USER", schema=SCHEMA, max_tokens=777)

    assert out == {"a": "x"}
    assert provider.name == "gemini"
    assert provider.model == "gemini-3-flash-preview"
    (call,) = client.calls
    assert call["model"] == "gemini-3-flash-preview"
    assert call["contents"] == "USER"
    config = call["config"]
    assert isinstance(config, genai_types.GenerateContentConfig)
    assert config.system_instruction == "SYS"
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == SCHEMA
    assert config.max_output_tokens == 777
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level == genai_types.ThinkingLevel.LOW


def test_gemini_sets_no_thinking_level_for_models_that_do_not_support_it() -> None:
    client = FakeGenai(gemini_response())
    GeminiProvider(api_key="g-key", model="gemini-2.5-flash", client=client).complete_json(
        system="s", user="u", schema=SCHEMA
    )
    assert client.calls[0]["config"].thinking_config is None


def test_gemini_client_is_built_with_a_millisecond_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class Recorder:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

    monkeypatch.setattr("app.extraction.llm.gemini_provider.genai.Client", Recorder)
    GeminiProvider(api_key="g-key", model="gemini-3-flash-preview", timeout=90)
    assert seen["api_key"] == "g-key"
    assert seen["http_options"].timeout == 90_000
    # The SDK retries timeouts and server errors with backoff; rate limits (429) are left to the provider, which
    # reads the delay the server asks for (see the rate-limit tests below).
    retry = seen["http_options"].retry_options
    assert retry is not None and retry.attempts == 3
    assert {408, 500, 502, 503, 504} <= set(retry.http_status_codes)
    assert not {400, 403, 429} & set(retry.http_status_codes)


@pytest.mark.parametrize(
    "error",
    [
        api_error(genai_errors.ClientError, 429),
        api_error(genai_errors.ClientError, 408),
        api_error(genai_errors.ServerError, 500),
        api_error(genai_errors.ServerError, 503),
    ],
    ids=lambda e: f"{type(e).__name__}-{e.code}",
)
def test_gemini_transient_errors(error: Exception) -> None:
    provider = GeminiProvider(api_key="k", model="m", client=FakeGenai(error=error), sleep=lambda _s: None)
    with pytest.raises(LLMError) as info:
        provider.complete_json(system="s", user="u", schema=SCHEMA)
    assert info.value.transient is True
    assert "SECRET-DOCUMENT-TEXT" not in str(info.value)


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_gemini_permanent_errors(code: int) -> None:
    provider = GeminiProvider(api_key="k", model="m", client=FakeGenai(error=api_error(genai_errors.ClientError, code)))
    with pytest.raises(LLMError) as info:
        provider.complete_json(system="s", user="u", schema=SCHEMA)
    assert info.value.transient is False


def test_gemini_connection_failure_is_an_unavailable_error() -> None:
    provider = GeminiProvider(api_key="k", model="m", client=FakeGenai(error=httpx.ConnectError("refused")))
    with pytest.raises(LLMUnavailableError) as info:
        provider.complete_json(system="s", user="u", schema=SCHEMA)
    assert info.value.transient is True


def test_gemini_blocked_prompt_is_a_refusal() -> None:
    client = FakeGenai(gemini_response(text=None, block_reason=genai_types.BlockedReason.PROHIBITED_CONTENT))
    with pytest.raises(LLMRefusalError):
        GeminiProvider(api_key="k", model="m", client=client).complete_json(system="s", user="u", schema=SCHEMA)


@pytest.mark.parametrize(
    "reason",
    [
        genai_types.FinishReason.SAFETY,
        genai_types.FinishReason.PROHIBITED_CONTENT,
        genai_types.FinishReason.BLOCKLIST,
        genai_types.FinishReason.SPII,
        genai_types.FinishReason.RECITATION,
    ],
    ids=lambda r: r.name,
)
def test_gemini_withheld_answer_is_a_refusal(reason: genai_types.FinishReason) -> None:
    # Valid JSON text is present, so only the finish reason can make this a refusal.
    client = FakeGenai(gemini_response(text='{"a": "x"}', finish_reason=reason))
    with pytest.raises(LLMRefusalError) as info:
        GeminiProvider(api_key="k", model="m", client=client).complete_json(system="s", user="u", schema=SCHEMA)
    assert reason.name in str(info.value)


def test_gemini_max_tokens_is_a_permanent_error() -> None:
    client = FakeGenai(gemini_response(text='{"a": "x"}', finish_reason=genai_types.FinishReason.MAX_TOKENS))
    with pytest.raises(LLMOutputTruncatedError) as info:
        GeminiProvider(api_key="k", model="m", client=client).complete_json(system="s", user="u", schema=SCHEMA)
    assert info.value.transient is False
    assert not isinstance(info.value, LLMRefusalError)
    assert "MAX_TOKENS" in str(info.value)


@pytest.mark.parametrize("text", [None, "", "not json", "[1]"])
def test_gemini_unusable_text_raises_permanent_llm_error(text: str | None) -> None:
    provider = GeminiProvider(api_key="k", model="m", client=FakeGenai(gemini_response(text=text)))
    with pytest.raises(LLMError) as info:
        provider.complete_json(system="s", user="u", schema=SCHEMA)
    assert info.value.transient is False


# ---------------------------------------------------------------------------------------------- shared contract


def test_error_hierarchy_and_defaults() -> None:
    assert issubclass(LLMRefusalError, LLMError)
    assert issubclass(LLMUnavailableError, LLMError)
    assert issubclass(LLMOutputTruncatedError, LLMError)
    assert LLMOutputTruncatedError("cut").transient is False
    assert not issubclass(LLMConfigurationError, LLMError)  # misconfiguration must not be swallowed as "LLM failed"
    assert LLMError("boom").transient is False
    assert LLMError("boom", transient=True).transient is True
    assert LLMError("boom", True).transient is True
    assert str(LLMError("boom")) == "boom"
    assert LLMRefusalError("declined").transient is False


def test_every_provider_satisfies_the_protocol() -> None:
    providers = [
        AnthropicProvider(api_key="k", client=FakeAnthropic()),
        GeminiProvider(api_key="k", model="m", client=FakeGenai()),
        OpenAICompatProvider("http://x/v1", "m", client=httpx.Client()),
    ]
    for provider in providers:
        assert isinstance(provider, LLMProvider)


# --------------------------------------------------------------------------------------------------- factory


def make_settings(**values: Any) -> Settings:
    return Settings(_env_file=None, **values)


def test_factory_auto_resolution() -> None:
    assert get_llm_provider(make_settings(llm_provider="auto")) is None
    assert get_llm_provider(make_settings(llm_provider="none", anthropic_api_key=SecretStr("k"))) is None

    local_only = get_llm_provider(make_settings(llm_provider="auto", llm_base_url="http://localhost:8080/v1"))
    assert isinstance(local_only, OpenAICompatProvider)
    assert (local_only.name, local_only.model) == ("local", "qwen3.5-9b")

    gemini_wins_over_local = get_llm_provider(
        make_settings(llm_provider="auto", gemini_api_key=SecretStr("g"), llm_base_url="http://localhost:8080/v1")
    )
    assert isinstance(gemini_wins_over_local, GeminiProvider)
    assert gemini_wins_over_local.model == "gemini-3-flash-preview"

    anthropic_wins = get_llm_provider(
        make_settings(
            llm_provider="auto",
            anthropic_api_key=SecretStr("a"),
            gemini_api_key=SecretStr("g"),
            llm_base_url="http://localhost:8080/v1",
        )
    )
    assert isinstance(anthropic_wins, AnthropicProvider)
    assert anthropic_wins.model == "claude-opus-5-5"
    assert anthropic_wins.effort == "medium"


def test_factory_explicit_choice_overrides_auto_priority() -> None:
    settings = make_settings(
        llm_provider="local",
        anthropic_api_key=SecretStr("a"),
        gemini_api_key=SecretStr("g"),
        llm_base_url="http://localhost:8080/v1",
        llm_model="qwen3-4b",
        llm_api_key=SecretStr("local-key"),
    )
    provider = get_llm_provider(settings)
    assert isinstance(provider, OpenAICompatProvider)
    assert provider.model == "qwen3-4b"

    chosen = get_llm_provider(
        make_settings(llm_provider="gemini", gemini_api_key=SecretStr("g"), gemini_model="gemini-x")
    )
    assert isinstance(chosen, GeminiProvider)
    assert chosen.model == "gemini-x"

    claude = get_llm_provider(
        make_settings(llm_provider="anthropic", anthropic_api_key=SecretStr("a"), anthropic_effort="high")
    )
    assert isinstance(claude, AnthropicProvider)
    assert claude.effort == "high"


@pytest.mark.parametrize(
    ("values", "missing"),
    [
        ({"llm_provider": "anthropic"}, "ANTHROPIC_API_KEY"),
        ({"llm_provider": "gemini"}, "GEMINI_API_KEY"),
        ({"llm_provider": "local"}, "LLM_BASE_URL"),
    ],
)
def test_factory_explicit_provider_with_missing_config_raises(values: dict[str, str], missing: str) -> None:
    with pytest.raises(LLMConfigurationError) as info:
        get_llm_provider(make_settings(**values))
    assert missing in str(info.value)


def test_factory_does_not_leak_secrets() -> None:
    settings = make_settings(
        llm_provider="auto", anthropic_api_key=SecretStr("sk-ant-very-secret"), llm_api_key=SecretStr("local-secret")
    )
    provider = get_llm_provider(settings)
    assert provider is not None
    assert "very-secret" not in repr(provider)
    local = get_llm_provider(
        make_settings(llm_provider="local", llm_base_url="http://x/v1", llm_api_key=SecretStr("local-secret"))
    )
    assert local is not None
    assert "local-secret" not in repr(local)


def test_factory_passes_the_context_and_extra_body_settings_to_the_local_provider() -> None:
    provider = get_llm_provider(
        make_settings(
            llm_provider="local",
            llm_base_url="http://localhost:8080/v1",
            llm_context_tokens=32768,
            llm_extra_body='{"top_k": 20}',
        )
    )
    assert isinstance(provider, OpenAICompatProvider)
    assert provider.context_tokens() == 32768  # from the setting: no request is made
    assert provider.extra_body == {"top_k": 20}


def test_factory_reports_an_invalid_extra_body_as_a_configuration_error() -> None:
    settings = make_settings(llm_provider="auto", llm_base_url="http://localhost:8080/v1", llm_extra_body="{oops")
    with pytest.raises(LLMConfigurationError) as info:
        get_llm_provider(settings)
    assert "LLM_EXTRA_BODY" in str(info.value)


def rate_limited(retry_delay: str | None) -> genai_errors.ClientError:
    details: list[dict[str, Any]] = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": []}]
    if retry_delay is not None:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay})
    body = {
        "error": {"code": 429, "message": "SECRET-DOCUMENT-TEXT", "status": "RESOURCE_EXHAUSTED", "details": details}
    }
    return genai_errors.ClientError(429, body)


class SequenceGenai(FakeGenai):
    """Raises the given errors in turn, then answers."""

    def __init__(self, *errors: Exception) -> None:
        super().__init__(gemini_response())
        self._errors = list(errors)

    def _generate(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._errors:
            raise self._errors.pop(0)
        return self._response


def test_gemini_waits_the_delay_a_rate_limit_asks_for() -> None:
    slept: list[float] = []
    client = SequenceGenai(rate_limited("12.5s"))
    provider = GeminiProvider(api_key="k", model="m", client=client, sleep=slept.append)

    assert provider.complete_json(system="s", user="u", schema=SCHEMA) == {"a": "x"}
    assert slept == [12.5] and len(client.calls) == 2


def test_gemini_falls_back_at_once_when_the_quota_is_spent() -> None:
    # A free-tier daily quota answers 429 with a retry delay of many hours: waiting cannot help, so the segment
    # falls back to rules immediately instead of holding the document for a minute.
    slept: list[float] = []
    client = SequenceGenai(rate_limited("71842s"))
    provider = GeminiProvider(api_key="k", model="m", client=client, sleep=slept.append)

    with pytest.raises(LLMError) as info:
        provider.complete_json(system="s", user="u", schema=SCHEMA)
    assert info.value.transient is True
    assert "quota" in str(info.value) and "SECRET-DOCUMENT-TEXT" not in str(info.value)
    assert slept == [] and len(client.calls) == 1


def test_gemini_gives_up_after_repeated_rate_limits() -> None:
    slept: list[float] = []
    client = SequenceGenai(*(rate_limited(None) for _ in range(10)))
    provider = GeminiProvider(api_key="k", model="m", client=client, sleep=slept.append)

    with pytest.raises(LLMError) as info:
        provider.complete_json(system="s", user="u", schema=SCHEMA)
    assert info.value.transient is True
    assert len(client.calls) == 4 and slept == [15.0, 15.0, 15.0]
