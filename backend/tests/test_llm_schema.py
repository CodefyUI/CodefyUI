"""Schema + SSE event-shape tests for the LLM proxy."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.core.llm_proxy.events import done_event, error_event, sse_format, text_delta
from app.core.llm_proxy.schema import ChatMessage, ChatRequest, ToolCall, ToolSpec


def test_chat_request_minimal():
    req = ChatRequest(provider="openai", model="gpt-5.2",
                      messages=[ChatMessage(role="user", content="hi")])
    assert req.max_tokens == 4096
    assert req.tools == []
    assert req.api_key is None
    assert req.reasoning_effort is None


@pytest.mark.parametrize("effort", ["none", "low", "max", "future_effort"])
def test_chat_request_accepts_safe_reasoning_effort_slugs(effort):
    req = ChatRequest(
        provider="openai",
        model="gpt-5.6-sol",
        messages=[ChatMessage(role="user", content="hi")],
        reasoning_effort=effort,
    )
    assert req.reasoning_effort == effort


@pytest.mark.parametrize("effort", ["", "High", "very high", "ultra\n", "a" * 65])
def test_chat_request_rejects_unsafe_reasoning_effort_values(effort):
    with pytest.raises(ValidationError):
        ChatRequest(
            provider="openai",
            model="gpt-5.6-sol",
            messages=[ChatMessage(role="user", content="hi")],
            reasoning_effort=effort,
        )


def test_chat_request_rejects_unknown_provider():
    with pytest.raises(ValidationError):
        ChatRequest(provider="bedrock", model="x",
                    messages=[ChatMessage(role="user", content="hi")])


def test_tool_message_roundtrip():
    msg = ChatMessage(role="tool", tool_call_id="call_1", content="{\"ok\": true}")
    assert msg.tool_call_id == "call_1"
    asst = ChatMessage(role="assistant", content="",
                       tool_calls=[ToolCall(id="call_1", name="apply", arguments={"a": 1})])
    assert asst.tool_calls[0].arguments == {"a": 1}


def test_tool_spec_defaults():
    t = ToolSpec(name="get_graph")
    assert t.input_schema["type"] == "object"


def test_event_shapes():
    assert text_delta("ab") == {"type": "text_delta", "text": "ab"}
    err = error_event("boom")
    assert err == {"type": "error", "message": "boom"}
    done = done_event(content="hi", tool_calls=[ToolCall(id="1", name="f", arguments={})],
                      stop_reason="tool_use", usage={"input_tokens": 1, "output_tokens": 2})
    assert done["type"] == "done"
    assert done["message"]["role"] == "assistant"
    assert done["message"]["tool_calls"][0]["name"] == "f"
    assert done["stop_reason"] == "tool_use"


def test_sse_format_is_data_line_json():
    line = sse_format({"type": "text_delta", "text": "x"})
    assert line.startswith("data: ")
    assert line.endswith("\n\n")
    assert json.loads(line[len("data: "):]) == {"type": "text_delta", "text": "x"}

def test_chat_message_accepts_multimodal_content():
    content = [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]
    msg = ChatMessage(role="user", content=content)
    assert msg.content == content
