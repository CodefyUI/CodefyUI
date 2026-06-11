"""Codex (ChatGPT backend) adapter tests."""

from __future__ import annotations

import json

import httpx
import pytest

from app.core.llm_proxy import codex, codex_auth
from app.core.llm_proxy.schema import ChatMessage, ChatRequest, ToolCall, ToolSpec


def make_req(**overrides):
    base = dict(
        provider="openai-codex",
        model="gpt-5.5",
        messages=[ChatMessage(role="system", content="be terse"),
                  ChatMessage(role="user", content="hi")],
    )
    base.update(overrides)
    return ChatRequest(**base)


@pytest.fixture(autouse=True)
def fake_login(monkeypatch):
    async def fake_access(client, *, force_refresh=False):
        return ("at-1" if not force_refresh else "at-2", "acc-1")
    monkeypatch.setattr(codex_auth, "get_valid_access", fake_access)


def sse_body(*events: dict) -> bytes:
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()


async def collect(req, handler):
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        return [e async for e in codex.stream_chat(req, client)]


def test_payload_shapes():
    req = make_req(messages=[
        ChatMessage(role="system", content="be terse"),
        ChatMessage(role="user", content="go"),
        ChatMessage(role="assistant", content="ok",
                    tool_calls=[ToolCall(id="c1", name="apply", arguments={"x": 1})]),
        ChatMessage(role="tool", tool_call_id="c1", content="done"),
    ], tools=[ToolSpec(name="apply", description="d",
                       input_schema={"type": "object", "properties": {}})])
    p = codex.build_payload(req)
    assert p["model"] == "gpt-5.5"
    assert p["instructions"] == "be terse"
    assert p["store"] is False
    assert p["stream"] is True
    items = p["input"]
    assert items[0] == {"type": "message", "role": "user",
                        "content": [{"type": "input_text", "text": "go"}]}
    assert items[1] == {"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}]}
    assert items[2] == {"type": "function_call", "call_id": "c1",
                        "name": "apply", "arguments": json.dumps({"x": 1})}
    assert items[3] == {"type": "function_call_output", "call_id": "c1",
                        "output": "done"}
    assert p["tools"][0] == {"type": "function", "name": "apply", "description": "d",
                             "parameters": {"type": "object", "properties": {}},
                             "strict": False}


@pytest.mark.asyncio
async def test_streams_text_and_function_calls():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["account"] = request.headers.get("chatgpt-account-id")
        seen["originator"] = request.headers.get("originator")
        return httpx.Response(200, content=sse_body(
            {"type": "response.output_text.delta", "delta": "He"},
            {"type": "response.output_text.delta", "delta": "y"},
            {"type": "response.output_item.done",
             "item": {"type": "function_call", "call_id": "fc1",
                      "name": "apply", "arguments": "{\"n\": 2}"}},
            {"type": "response.completed",
             "response": {"usage": {"input_tokens": 4, "output_tokens": 6}}},
        ), headers={"content-type": "text/event-stream"})

    events = await collect(make_req(), handler)
    assert seen["url"] == "https://chatgpt.com/backend-api/codex/responses"
    assert seen["auth"] == "Bearer at-1"
    assert seen["account"] == "acc-1"
    assert seen["originator"] == "codex_cli_rs"
    assert events[0] == {"type": "text_delta", "text": "He"}
    done = events[-1]
    assert done["message"]["content"] == "Hey"
    assert done["message"]["tool_calls"] == [
        {"id": "fc1", "name": "apply", "arguments": {"n": 2}}]
    assert done["stop_reason"] == "tool_use"
    assert done["usage"] == {"input_tokens": 4, "output_tokens": 6}


@pytest.mark.asyncio
async def test_retries_once_on_401_with_forced_refresh():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            assert request.headers["authorization"] == "Bearer at-1"
            return httpx.Response(401, json={"detail": "expired"})
        assert request.headers["authorization"] == "Bearer at-2"
        return httpx.Response(200, content=sse_body(
            {"type": "response.output_text.delta", "delta": "ok"},
            {"type": "response.completed", "response": {"usage": {}}},
        ), headers={"content-type": "text/event-stream"})

    events = await collect(make_req(), handler)
    assert calls["n"] == 2
    assert events[-1]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_not_logged_in_yields_error_event(monkeypatch):
    async def raising(client, *, force_refresh=False):
        raise codex_auth.CodexNotLoggedIn("nope")
    monkeypatch.setattr(codex_auth, "get_valid_access", raising)

    events = await collect(make_req(), lambda r: httpx.Response(500))
    assert events == [{"type": "error",
                       "message": "Not signed in to ChatGPT - open Settings to sign in."}]


@pytest.mark.asyncio
async def test_response_failed_becomes_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse_body(
            {"type": "response.failed",
             "response": {"error": {"message": "usage limit reached"}}},
        ), headers={"content-type": "text/event-stream"})

    events = await collect(make_req(), handler)
    assert events[-1]["type"] == "error"
    assert "usage limit" in events[-1]["message"]


def test_static_models_list():
    assert "gpt-5.5" in codex.STATIC_MODELS
