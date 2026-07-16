"""OpenAI-compatible adapter tests (drives the adapter with MockTransport)."""

from __future__ import annotations

import json

import httpx
import pytest

from app.core.llm_proxy.openai_like import build_payload, resolve_base_url, stream_chat
from app.core.llm_proxy.schema import ChatMessage, ChatRequest, ToolCall, ToolSpec


def make_req(**overrides):
    base = dict(
        provider="openai",
        model="gpt-5.2",
        messages=[ChatMessage(role="system", content="sys"),
                  ChatMessage(role="user", content="hi")],
        api_key="sk-test",
    )
    base.update(overrides)
    return ChatRequest(**base)


def sse_body(*chunks: dict) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


async def collect(req, handler):
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        return [e async for e in stream_chat(req, client)]


# -- request building --------------------------------------------------------

def test_base_url_per_provider():
    assert resolve_base_url(make_req()) == "https://api.openai.com/v1"
    assert resolve_base_url(make_req(provider="openrouter")) == "https://openrouter.ai/api/v1"
    assert resolve_base_url(make_req(provider="custom", base_url="http://127.0.0.1:11434/v1/")) == "http://127.0.0.1:11434/v1"


def test_custom_requires_http_base_url():
    with pytest.raises(ValueError):
        resolve_base_url(make_req(provider="custom", base_url="ftp://x"))
    with pytest.raises(ValueError):
        resolve_base_url(make_req(provider="custom", base_url=None))


def test_payload_maps_messages_tools_and_sampling():
    req = make_req(
        tools=[ToolSpec(name="apply", description="d",
                        input_schema={"type": "object", "properties": {"x": {"type": "integer"}}})],
        messages=[
            ChatMessage(role="user", content="go"),
            ChatMessage(role="assistant", content="",
                        tool_calls=[ToolCall(id="c1", name="apply", arguments={"x": 1})]),
            ChatMessage(role="tool", tool_call_id="c1", content="done"),
        ],
        temperature=0.3,
        reasoning_effort="high",
    )
    p = build_payload(req)
    assert p["model"] == "gpt-5.2"
    assert p["stream"] is True
    assert p["stream_options"] == {"include_usage": True}
    assert p["max_completion_tokens"] == 4096
    assert "max_tokens" not in p
    assert p["reasoning_effort"] == "high"
    assert p["temperature"] == 0.3
    assert p["tools"][0]["function"]["name"] == "apply"
    assert p["tools"][0]["function"]["parameters"]["properties"]["x"]["type"] == "integer"
    asst = p["messages"][1]
    assert asst["tool_calls"][0]["function"]["arguments"] == json.dumps({"x": 1})
    tool = p["messages"][2]
    assert tool == {"role": "tool", "tool_call_id": "c1", "content": "done"}


# -- streaming ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_streams_text_and_done_with_usage():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, content=sse_body(
            {"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2}},
        ), headers={"content-type": "text/event-stream"})

    events = await collect(make_req(), handler)
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert events[0] == {"type": "text_delta", "text": "Hel"}
    assert events[1] == {"type": "text_delta", "text": "lo"}
    done = events[-1]
    assert done["type"] == "done"
    assert done["message"]["content"] == "Hello"
    assert done["stop_reason"] == "end"
    assert done["usage"] == {"input_tokens": 7, "output_tokens": 2}


@pytest.mark.asyncio
async def test_accumulates_streamed_tool_calls():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse_body(
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c1", "function": {"name": "apply", "arguments": "{\"a\""}}]},
                "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": ": 1}"}}]},
                "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ), headers={"content-type": "text/event-stream"})

    events = await collect(make_req(), handler)
    done = events[-1]
    assert done["stop_reason"] == "tool_use"
    assert done["message"]["tool_calls"] == [
        {"id": "c1", "name": "apply", "arguments": {"a": 1}}]


@pytest.mark.asyncio
async def test_unparseable_tool_arguments_marked():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse_body(
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c1", "function": {"name": "apply", "arguments": "{broken"}}]},
                "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ), headers={"content-type": "text/event-stream"})

    events = await collect(make_req(), handler)
    args = events[-1]["message"]["tool_calls"][0]["arguments"]
    assert args["__parse_error__"] == "{broken"


@pytest.mark.asyncio
async def test_upstream_error_becomes_error_event():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    events = await collect(make_req(), handler)
    assert events[-1]["type"] == "error"
    assert "401" in events[-1]["message"]
    assert "bad key" in events[-1]["message"]


@pytest.mark.asyncio
async def test_custom_provider_hits_custom_base():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, content=sse_body(
            {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
        ), headers={"content-type": "text/event-stream"})

    req = make_req(provider="custom", base_url="http://127.0.0.1:11434/v1", api_key=None)
    events = await collect(req, handler)
    assert seen["url"] == "http://127.0.0.1:11434/v1/chat/completions"
    assert events[-1]["type"] == "done"


def test_custom_payload_has_no_stream_options():
    req = make_req(
        provider="custom",
        base_url="http://127.0.0.1:11434/v1",
        api_key=None,
        reasoning_effort="high",
    )
    p = build_payload(req)
    assert "stream_options" not in p
    assert p["max_tokens"] == 4096
    assert "max_completion_tokens" not in p
    assert "reasoning_effort" not in p


def test_openrouter_keeps_compatible_token_field_and_omits_effort():
    req = make_req(provider="openrouter", reasoning_effort="high")
    p = build_payload(req)
    assert p["max_tokens"] == 4096
    assert "max_completion_tokens" not in p
    assert "reasoning_effort" not in p


@pytest.mark.asyncio
async def test_bad_base_url_yields_error_event():
    req = make_req(provider="custom", base_url="ftp://nope", api_key=None)
    events = await collect(req, lambda r: httpx.Response(500))
    assert events == [{"type": "error",
                       "message": "custom provider requires an http(s) base_url"}]

def test_payload_preserves_user_multimodal_content():
    content = [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]
    req = make_req(messages=[ChatMessage(role="user", content=content)])
    p = build_payload(req)
    assert p["messages"][0]["content"] == content
