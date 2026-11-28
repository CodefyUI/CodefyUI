"""The LLM proxy after the real-model round (round 6, H2).

Measured with real models: a local reasoning model streamed `reasoning`
for 114 seconds before its first word, the proxy dropped all of it, the
120-second read timeout fired on a cold start, and the user read
"upstream request failed: " with nothing after it (``str(httpx.
ReadTimeout())`` is empty). The same empty message once came from Codex.

Every adapter is driven the same way here: OpenAI-compatible (custom),
Anthropic and Codex.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from app.core.llm_proxy import _common, anthropic, codex, codex_auth, openai_like
from app.core.llm_proxy.schema import ChatMessage, ChatRequest

PROVIDERS = {
    "custom": openai_like,
    "openai": openai_like,
    "anthropic": anthropic,
    "openai-codex": codex,
}


def make_req(provider: str) -> ChatRequest:
    extra = {"base_url": "http://127.0.0.1:11434/v1"} if provider == "custom" else {}
    return ChatRequest(provider=provider, model="m-1", api_key="sk-secret-key",
                       messages=[ChatMessage(role="user", content="hi")], **extra)


@pytest.fixture(autouse=True)
def fake_login(monkeypatch):
    async def fake_access(client, *, force_refresh=False):
        return ("at-1", "acc-1")
    monkeypatch.setattr(codex_auth, "get_valid_access", fake_access)


async def collect(provider: str, handler) -> list[dict]:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return [e async for e in PROVIDERS[provider].stream_chat(make_req(provider), client)]


def sse(*events: dict, done: bool = False) -> bytes:
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    return (body + ("data: [DONE]\n\n" if done else "")).encode()


#: A short answer in each upstream's dialect, reasoning first.
def reasoning_stream(provider: str) -> bytes:
    if provider in ("custom", "openai"):
        return sse(
            {"choices": [{"delta": {"reasoning": "Let me "}}]},
            {"choices": [{"delta": {"reasoning_content": "think."}}]},
            {"choices": [{"delta": {"content": "Hello"}, "finish_reason": "stop"}]},
            done=True)
    if provider == "anthropic":
        return sse(
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "thinking_delta", "thinking": "Let me "}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "thinking_delta", "thinking": "think."}},
            {"type": "content_block_delta", "index": 1,
             "delta": {"type": "text_delta", "text": "Hello"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"})
    return sse(
        {"type": "response.reasoning_summary_text.delta", "delta": "Let me "},
        {"type": "response.reasoning_text.delta", "delta": "think."},
        {"type": "response.output_text.delta", "delta": "Hello"},
        {"type": "response.completed", "response": {"usage": {}}})


# ── reasoning_delta ──────────────────────────────────────────────────────

@pytest.mark.parametrize("provider", list(PROVIDERS))
async def test_reasoning_is_relayed_and_kept_out_of_the_content(provider):
    events = await collect(provider, lambda r: httpx.Response(200, content=reasoning_stream(provider)))
    reasoning = [e["text"] for e in events if e["type"] == "reasoning_delta"]
    assert reasoning == ["Let me ", "think."]
    assert [e["text"] for e in events if e["type"] == "text_delta"] == ["Hello"]
    assert events[-1]["type"] == "done"
    assert events[-1]["message"]["content"] == "Hello"


# ── errors always say something ──────────────────────────────────────────

def raising(exc_type):
    def handler(request):
        raise exc_type("", request=request)
    return handler


@pytest.mark.parametrize("provider", list(PROVIDERS))
@pytest.mark.parametrize("exc_type,expected", [
    (httpx.ReadTimeout, "ReadTimeout: the endpoint sent no data for "),
    (httpx.ConnectError, "ConnectError: cannot reach "),
    (httpx.ConnectTimeout, "ConnectTimeout: could not connect to "),
    (httpx.RemoteProtocolError, "RemoteProtocolError: the endpoint closed"),
])
async def test_an_upstream_failure_names_its_class_and_says_what_happened(
        provider, exc_type, expected, caplog):
    with caplog.at_level(logging.WARNING, logger="app.llm_proxy"):
        events = await collect(provider, raising(exc_type))
    assert events[-1]["type"] == "error"
    message = events[-1]["message"]
    assert message.startswith("upstream request failed: " + expected), message
    assert not message.endswith(": ")
    # One line in the server log: class, provider and model; never the key.
    logged = [r.getMessage() for r in caplog.records if r.name == "app.llm_proxy"]
    assert logged == [f"LLM upstream request failed: {exc_type.__name__} "
                      f"(provider={provider}, model=m-1)"]
    assert "sk-secret-key" not in caplog.text


async def test_a_refused_connect_names_the_host_and_port():
    events = await collect("custom", raising(httpx.ConnectError))
    assert "cannot reach 127.0.0.1:11434" in events[-1]["message"]


async def test_a_read_timeout_says_how_long_it_waited(monkeypatch):
    events = await collect("custom", raising(httpx.ReadTimeout))
    assert "sent no data for 600 seconds" in events[-1]["message"]
    assert "CODEFYUI_LLM_READ_TIMEOUT" in events[-1]["message"]
    events = await collect("anthropic", raising(httpx.ReadTimeout))
    assert "sent no data for 120 seconds" in events[-1]["message"]


# ── read timeout ─────────────────────────────────────────────────────────

def test_read_timeouts(monkeypatch):
    monkeypatch.delenv(_common.READ_TIMEOUT_ENV, raising=False)
    assert _common.read_timeout("custom") == 600
    for provider in ("openai", "openrouter", "anthropic", "openai-codex"):
        assert _common.read_timeout(provider) == 120
    monkeypatch.setenv(_common.READ_TIMEOUT_ENV, "900")
    assert _common.read_timeout("custom") == 900
    assert _common.read_timeout("anthropic") == 900
    for bad in ("0", "-5", "soon"):
        monkeypatch.setenv(_common.READ_TIMEOUT_ENV, bad)
        assert _common.read_timeout("custom") == 600
    timeout = _common.timeout_for("openai")
    assert (timeout.connect, timeout.read) == (10.0, 120.0)


@pytest.mark.parametrize("provider", list(PROVIDERS))
async def test_the_request_carries_the_providers_timeout(provider, monkeypatch):
    monkeypatch.setenv(_common.READ_TIMEOUT_ENV, "321")
    seen = []

    def handler(request):
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, content=reasoning_stream(provider))
    await collect(provider, handler)
    assert seen[0]["read"] == 321 and seen[0]["connect"] == 10


# ── one retry, only before any byte ──────────────────────────────────────

@pytest.mark.parametrize("provider", list(PROVIDERS))
async def test_a_transport_error_before_any_byte_is_retried_once(provider):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, content=reasoning_stream(provider))
    events = await collect(provider, handler)
    assert len(calls) == 2
    assert events[-1]["type"] == "done"


@pytest.mark.parametrize("provider", list(PROVIDERS))
async def test_it_is_retried_only_once(provider):
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ConnectError("refused", request=request)
    events = await collect(provider, handler)
    assert len(calls) == 2
    assert events[-1]["type"] == "error"


@pytest.mark.parametrize("provider", list(PROVIDERS))
async def test_a_timeout_is_not_retried(provider):
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ReadTimeout("", request=request)
    await collect(provider, handler)
    assert len(calls) == 1


class _BreaksMidway(httpx.AsyncByteStream):
    def __init__(self, first: bytes):
        self.first = first

    async def __aiter__(self):
        yield self.first
        raise httpx.ReadError("connection reset")


@pytest.mark.parametrize("provider", list(PROVIDERS))
async def test_an_error_after_data_arrived_is_not_retried(provider):
    calls = []
    first = reasoning_stream(provider).split(b"\n\n")[0] + b"\n\n"

    def handler(request):
        calls.append(1)
        return httpx.Response(200, stream=_BreaksMidway(first))
    events = await collect(provider, handler)
    assert len(calls) == 1
    assert events[-1]["type"] == "error"
    assert "ReadError: the connection to the endpoint broke" in events[-1]["message"]
