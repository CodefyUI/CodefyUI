"""Tests for the LLMChat workflow node."""

from __future__ import annotations

import json

import httpx
import pytest
import torch

from app.core.execution_context import INTERRUPTED_KEY, ExecutionContext
from app.core.llm_proxy import codex_auth
from app.core.node_base import DataType, ParamType
from app.nodes.llm import llm_chat_node
from app.nodes.llm.llm_chat_node import LLMChatNode, _build_request, _normalize_provider


def params(**overrides):
    base = {
        "provider": "ChatGPT API",
        "model": "gpt-5.2",
        "prompt": "Prompt",
        "system_prompt": "System",
        "openai_api_key": "sk-test",
        "anthropic_api_key": "sk-ant-test",
        "ollama_base_url": "http://127.0.0.1:11434/v1",
        "max_tokens": 128,
        "temperature": 0.2,
    }
    base.update(overrides)
    return base


def test_node_metadata():
    assert LLMChatNode.NODE_NAME == "LLMChat"
    assert LLMChatNode.CATEGORY == "LLM"
    assert LLMChatNode.cacheable is False

    inputs = {p.name: p for p in LLMChatNode.define_inputs()}
    assert inputs["text"].data_type == DataType.STRING
    assert inputs["image"].data_type == DataType.IMAGE
    assert inputs["array"].data_type == DataType.LIST
    assert all(p.optional for p in inputs.values())

    outputs = LLMChatNode.define_outputs()
    assert len(outputs) == 1
    assert outputs[0].name == "text"
    assert outputs[0].data_type == DataType.STRING

    provider_param = next(p for p in LLMChatNode.define_params() if p.name == "provider")
    assert provider_param.param_type == ParamType.SELECT
    assert provider_param.options == ["ChatGPT API", "Codex", "Claude API", "Ollama"]

    # Both API-key params are SECRET so their values are never persisted to a
    # saved graph (scrubbed by the save endpoint and the publish pre-flight).
    key_params = {p.name: p for p in LLMChatNode.define_params()
                  if p.name.endswith("_api_key")}
    assert key_params["openai_api_key"].param_type == ParamType.SECRET
    assert key_params["anthropic_api_key"].param_type == ParamType.SECRET


def test_provider_aliases():
    assert _normalize_provider("ChatGPT API") == "openai"
    assert _normalize_provider("Codex") == "openai-codex"
    assert _normalize_provider("Claude API") == "anthropic"
    assert _normalize_provider("Ollama") == "custom"
    with pytest.raises(ValueError, match="Unsupported"):
        _normalize_provider("Bedrock")


def test_execute_builds_openai_request(monkeypatch):
    seen = {}

    async def fake_collect(req, adapter, progress_callback=None, *, context=None):
        seen["req"] = req
        return "assistant text", {"input_tokens": 3, "output_tokens": 4}, None

    monkeypatch.setattr(llm_chat_node, "_collect_chat", fake_collect)
    res = LLMChatNode().execute(
        {"text": "hello", "array": [{"x": 1}]},
        params(),
    )

    assert res == {
        "text": "assistant text",
        "__usage__": {"input_tokens": 3, "output_tokens": 4},
    }
    req = seen["req"]
    assert req.provider == "openai"
    assert req.api_key == "sk-test"
    assert req.model == "gpt-5.2"
    assert req.max_tokens == 128
    assert req.temperature == 0.2
    assert req.messages[0].content == "System"
    user_content = req.messages[1].content
    assert isinstance(user_content, str)
    assert "Prompt" in user_content
    assert "hello" in user_content
    assert '"x": 1' in user_content


def test_ollama_maps_to_custom_provider(monkeypatch):
    seen = {}

    async def fake_collect(req, adapter, progress_callback=None, *, context=None):
        seen["req"] = req
        return "local", {}, None

    monkeypatch.setattr(llm_chat_node, "_collect_chat", fake_collect)
    res = LLMChatNode().execute(
        {},
        params(
            provider="Ollama",
            model="llama3.2",
            prompt="hi",
            system_prompt="",
            ollama_base_url="http://localhost:11434/v1/",
        ),
    )

    assert res == {"text": "local"}
    req = seen["req"]
    assert req.provider == "custom"
    assert req.base_url == "http://localhost:11434/v1/"
    assert req.api_key is None


def test_image_tensor_becomes_multimodal_content():
    req = _build_request(
        "openai",
        {"image": torch.zeros(3, 2, 2)},
        params(prompt="Describe it"),
    )

    content = req.messages[-1].content
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "Describe it"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_codex_image_input_falls_back_to_text():
    req = _build_request(
        "openai-codex",
        {"image": torch.zeros(3, 2, 2)},
        params(provider="Codex", prompt="Look"),
    )

    content = req.messages[-1].content
    assert isinstance(content, str)
    assert "Look" in content
    assert "text only" in content


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("CODEFYUI_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="requires openai_api_key"):
        _build_request("openai", {}, params(openai_api_key="", prompt="hi"))


# ── #155: Stop lands mid-generation ───────────────────────────────────────
#
# Before this, ``context`` was a dead parameter on the streaming path: the
# five long-loop nodes treated in #122 (PR #154) grew ``should_stop()``
# checks and LLMChat was missed, so Stop was acknowledged and the node kept
# streaming until the provider decided to finish -- minutes, with
# ``max_tokens`` up to 200_000.


class _EndlessSSE(httpx.AsyncByteStream):
    """A response body that keeps emitting *frame*, and records its teardown.

    ``closed`` alone proves nothing: an abandoned adapter generator IS
    eventually finalized -- by refcounting, or by ``asyncio.run``'s closing
    ``shutdown_asyncgens`` -- so the body ends up closed either way. What
    separates a released connection from a leaked one is WHEN, so
    ``client_was_open_at_close`` records whether the ``AsyncClient`` that
    owns the connection was still alive at teardown. True means the
    adapter's ``async with client.stream(...)`` exit ran on the way out of
    the loop, as it should; False means it ran during or after the client's
    own teardown, i.e. the response outlived the client it belonged to.

    ``limit`` keeps a regression to a failed assertion instead of a hang.
    """

    def __init__(self, frame: bytes, limit: int = 500) -> None:
        self._frame = frame
        self._limit = limit
        self.sent = 0
        self.closed = False
        #: Set by the client factory before the request goes out.
        self.client: httpx.AsyncClient | None = None
        self.client_was_open_at_close: bool | None = None

    def _record_close(self) -> None:
        self.closed = True
        if self.client_was_open_at_close is None:
            self.client_was_open_at_close = (
                self.client is not None and not self.client.is_closed)

    async def __aiter__(self):
        try:
            while self.sent < self._limit:
                self.sent += 1
                yield self._frame
        except GeneratorExit:
            self._record_close()
            raise

    async def aclose(self) -> None:
        self._record_close()


def _openai_frame(text: str) -> bytes:
    return f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n".encode()


def _anthropic_frame(text: str) -> bytes:
    payload = {"type": "content_block_delta",
               "delta": {"type": "text_delta", "text": text}}
    return f"data: {json.dumps(payload)}\n\n".encode()


def _codex_frame(text: str) -> bytes:
    payload = {"type": "response.output_text.delta", "delta": text}
    return f"data: {json.dumps(payload)}\n\n".encode()


# Every provider the node offers, each with the SSE dialect its adapter
# parses. All four go through the same two-line stop, because all three
# adapter modules are async generators over ``client.stream`` -- there is no
# provider-specific cancel call to get wrong.
STREAMING_PROVIDERS = [
    pytest.param("ChatGPT API", _openai_frame, id="openai"),
    pytest.param("Claude API", _anthropic_frame, id="anthropic"),
    pytest.param("Ollama", _openai_frame, id="ollama"),
    pytest.param("Codex", _codex_frame, id="codex"),
]


def _install_mock_transport(monkeypatch, body: _EndlessSSE) -> None:
    """Point every ``AsyncClient`` this node builds at *body*."""
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=body)

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        client = real_client(*args, **kwargs)
        body.client = client
        return client

    monkeypatch.setattr(llm_chat_node.httpx, "AsyncClient", fake_client)

    async def fake_access(client, force_refresh=False):
        return "access-token", "account-id"

    monkeypatch.setattr(codex_auth, "get_valid_access", fake_access)


@pytest.mark.parametrize("provider,frame", STREAMING_PROVIDERS)
def test_stop_mid_stream_keeps_the_partial_text_and_closes_the_stream(
    monkeypatch, provider, frame
):
    """Stop lands on the next chunk; partial text survives; socket released."""
    body = _EndlessSSE(frame("tok "))
    _install_mock_transport(monkeypatch, body)

    ctx = ExecutionContext()
    seen: list[str] = []

    def on_progress(payload):
        # Drive the stop off the node's own stream, the way a user's click
        # lands: partway through, with plenty of generation left to do.
        seen.append(payload["text"])
        if len(seen) == 2:
            ctx.cancel()

    result = LLMChatNode().execute(
        {}, params(provider=provider, prompt="write an essay"),
        on_progress, context=ctx,
    )

    assert result["text"] == "tok tok ", "the partial completion must survive"
    assert seen[-1] == "tok tok "

    marker = result[INTERRUPTED_KEY]
    assert marker["batch"] == 2, "stopped after the chunk the click landed on"
    assert marker["chars"] == len(result["text"])
    assert "__usage__" not in result, "an interrupted generation reports no usage"

    assert body.closed
    assert body.client_was_open_at_close, (
        "the provider response outlived its client -- breaking out of the "
        "`async for` does not run the adapter's `async with client.stream()` "
        "exit, so the connection is only reclaimed when the abandoned "
        "generator is finalized, after the client it belonged to is gone"
    )
    assert body.sent < 10, (
        f"kept reading after the stop ({body.sent} frames pulled)"
    )


@pytest.mark.parametrize("provider,frame", STREAMING_PROVIDERS)
def test_an_uninterrupted_generation_is_unchanged(monkeypatch, provider, frame):
    """The stop plumbing must not alter a run nobody stopped."""
    body = _EndlessSSE(frame("hi"), limit=1)
    _install_mock_transport(monkeypatch, body)

    result = LLMChatNode().execute({}, params(provider=provider, prompt="hi"))

    assert result["text"] == "hi"
    assert INTERRUPTED_KEY not in result
    assert body.closed, "the normal path should hang up promptly too"


def test_a_stop_that_landed_before_the_call_skips_the_provider_entirely(
    monkeypatch,
):
    """A node dequeued after Stop must not pay for a whole generation."""
    requests: list[httpx.Request] = []
    body = _EndlessSSE(_openai_frame("tok "))
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, stream=body)

    monkeypatch.setattr(
        llm_chat_node.httpx, "AsyncClient",
        lambda *a, **kw: real_client(*a, transport=httpx.MockTransport(handler), **kw),
    )

    ctx = ExecutionContext()
    ctx.cancel()
    result = LLMChatNode().execute({}, params(prompt="write an essay"), context=ctx)

    assert result["text"] == ""
    assert result[INTERRUPTED_KEY]["batch"] == 0
    assert requests == [], "the request went out after the run was stopped"


async def test_a_context_free_call_still_streams():
    """``context=None`` -- the export runner, most unit tests -- never stops."""
    async def adapter(req, client):
        yield {"type": "text_delta", "text": "a"}
        yield {"type": "done", "message": {"content": "ab"},
               "usage": {"input_tokens": 1, "output_tokens": 2}}

    text, usage, stopped = await llm_chat_node._collect_chat(
        _build_request("openai", {}, params(prompt="hi")), adapter)

    assert (text, usage, stopped) == (
        "ab", {"input_tokens": 1, "output_tokens": 2}, None)


async def test_close_stream_tolerates_an_iterator_without_aclose():
    """``ProviderAdapter`` promises an AsyncIterator, which need not close."""
    class _Plain:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    await llm_chat_node._close_stream(_Plain())  # must not raise


async def test_close_stream_swallows_a_failing_teardown(caplog):
    """A hiccup while hanging up must not fail a node that has its result."""
    class _Boom:
        async def aclose(self):
            raise RuntimeError("connection reset")

    await llm_chat_node._close_stream(_Boom())
    assert "could not close the LLM provider stream" in caplog.text