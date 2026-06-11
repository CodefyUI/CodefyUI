"""Unified stream-event shapes relayed to the browser as SSE data lines.

Contract (spec Part B): text streams as it arrives; tool calls are delivered
complete in the terminal "done" event so the four providers' incompatible
tool-call streaming dialects never leak past this module.
"""

from __future__ import annotations

import json
from typing import Any

from .schema import ToolCall


def text_delta(text: str) -> dict[str, Any]:
    return {"type": "text_delta", "text": text}


def error_event(message: str) -> dict[str, Any]:
    return {"type": "error", "message": message}


def done_event(
    *,
    content: str,
    tool_calls: list[ToolCall],
    stop_reason: str,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "type": "done",
        "message": {
            "role": "assistant",
            "content": content,
            "tool_calls": [tc.model_dump() for tc in tool_calls],
        },
        "stop_reason": stop_reason,
        "usage": usage or {},
    }


def sse_format(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
