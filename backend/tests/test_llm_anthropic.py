"""Anthropic Messages API adapter tests."""

from __future__ import annotations

import json

import httpx
import pytest

from app.core.llm_proxy.anthropic import build_payload, stream_chat
from app.core.llm_proxy.schema import ChatMessage, ChatRequest, ToolCall, ToolSpec


def make_req(**overrides):
    base = dict(
        provider="anthropic",
        model="claude-sonnet-4-6",
        messages=[ChatMessage(role="system", content="be brief"),
                  ChatMessage(role="user", content="hi")],
        api_key="sk-ant-test",
    )
    base.update(overrides)
    return ChatRequest(**base)


def sse_events(*events: tuple[str, dict]) -> bytes:
    out = []
    for name, payload in events:
        out.append(f"event: {name}\ndata: {json.dumps(payload)}\n\n")
    return "".join(out).encode()


async def collect(req, handler):
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        return [e async for e in stream_chat(req, client)]


def test_payload_extracts_system_and_maps_tools():
    req = make_req(
        tools=[ToolSpec(name="apply", description="d",
                        input_schema={"type": "object", "properties": {}})],
        messages=[
            ChatMessage(role="system", content="be brief"),
            ChatMessage(role="user", content="go"),
            ChatMessage(role="assistant", content="thinking",
                        tool_calls=[ToolCall(id="t1", name="apply", arguments={"x": 2})]),
            ChatMessage(role="tool", tool_call_id="t1", content="ok"),
        ],
        reasoning_effort="high",
    )
    p = build_payload(req)
    assert p["system"] == "be brief"
    assert p["model"] == "claude-sonnet-4-6"
    assert p["stream"] is True
    assert "reasoning" not in p
    assert "reasoning_effort" not in p
    assert p["tools"] == [{"name": "apply", "description": "d",
                           "input_schema": {"type": "object", "properties": {}}}]
    assert all(m["role"] != "system" for m in p["messages"])
    asst = p["messages"][1]
    assert asst["content"][0] == {"type": "text", "text": "thinking"}
    assert asst["content"][1]["type"] == "tool_use"
    assert asst["content"][1]["input"] == {"x": 2}
    tool_result = p["messages"][2]
    assert tool_result["role"] == "user"
    assert tool_result["content"][0]["type"] == "tool_result"
    assert tool_result["content"][0]["tool_use_id"] == "t1"


@pytest.mark.asyncio
async def test_streams_text_then_done():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-api-key")
        seen["version"] = request.headers.get("anthropic-version")
        return httpx.Response(200, content=sse_events(
            ("message_start", {"type": "message_start",
                               "message": {"usage": {"input_tokens": 9}}}),
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": "Hey"}}),
            ("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": "end_turn"},
                               "usage": {"output_tokens": 3}}),
            ("message_stop", {"type": "message_stop"}),
        ), headers={"content-type": "text/event-stream"})

    events = await collect(make_req(), handler)
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["key"] == "sk-ant-test"
    assert seen["version"] == "2023-06-01"
    assert events[0] == {"type": "text_delta", "text": "Hey"}
    done = events[-1]
    assert done["message"]["content"] == "Hey"
    assert done["stop_reason"] == "end"
    assert done["usage"] == {"input_tokens": 9, "output_tokens": 3}


@pytest.mark.asyncio
async def test_accumulates_tool_use_blocks():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse_events(
            ("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 1}}}),
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "tool_use", "id": "t9", "name": "apply"}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "input_json_delta", "partial_json": "{\"n\""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "input_json_delta", "partial_json": ": 5}"}}),
            ("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": "tool_use"},
                               "usage": {"output_tokens": 2}}),
            ("message_stop", {"type": "message_stop"}),
        ), headers={"content-type": "text/event-stream"})

    events = await collect(make_req(), handler)
    done = events[-1]
    assert done["stop_reason"] == "tool_use"
    assert done["message"]["tool_calls"] == [
        {"id": "t9", "name": "apply", "arguments": {"n": 5}}]


@pytest.mark.asyncio
async def test_upstream_error_becomes_error_event():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(529, json={"error": {"message": "overloaded"}})

    events = await collect(make_req(), handler)
    assert events[-1]["type"] == "error"
    assert "529" in events[-1]["message"]

def test_payload_maps_multimodal_content_blocks():
    content = [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]
    req = make_req(messages=[ChatMessage(role="user", content=content)])
    p = build_payload(req)
    blocks = p["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "look"}
    assert blocks[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "abc"},
    }
