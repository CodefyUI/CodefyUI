"""Adapter for the Anthropic Messages API (api.anthropic.com)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ._common import TIMEOUT, parse_tool_args
from .events import done_event, error_event, text_delta
from .schema import ChatRequest, ToolCall

_BASE = "https://api.anthropic.com/v1"
_VERSION = "2023-06-01"


def build_payload(req: ChatRequest) -> dict[str, Any]:
    system_parts = [m.content for m in req.messages if m.role == "system" and m.content]
    messages: list[dict[str, Any]] = []
    for m in req.messages:
        if m.role == "system":
            continue
        if m.role == "tool":
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": m.tool_call_id or "",
                "content": m.content,
            }]})
        elif m.role == "assistant" and m.tool_calls:
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            blocks.extend({
                "type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments,
            } for tc in m.tool_calls)
            messages.append({"role": "assistant", "content": blocks})
        else:
            messages.append({"role": m.role, "content": m.content})

    payload: dict[str, Any] = {
        "model": req.model,
        "messages": messages,
        "max_tokens": req.max_tokens,
        "stream": True,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.tools:
        payload["tools"] = [{
            "name": t.name, "description": t.description, "input_schema": t.input_schema,
        } for t in req.tools]
    return payload


async def stream_chat(
    req: ChatRequest, client: httpx.AsyncClient
) -> AsyncIterator[dict[str, Any]]:
    headers = {
        "content-type": "application/json",
        "x-api-key": req.api_key or "",
        "anthropic-version": _VERSION,
    }
    content_parts: list[str] = []
    # index -> partially-built tool_use block
    blocks: dict[int, dict[str, Any]] = {}
    stop_reason: str | None = None
    usage: dict[str, int] = {}

    try:
        async with client.stream("POST", f"{_BASE}/messages", json=build_payload(req),
                                 headers=headers, timeout=TIMEOUT) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")[:500]
                yield error_event(f"upstream {resp.status_code}: {body}")
                return
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    ev = json.loads(line[len("data: "):])
                except json.JSONDecodeError:
                    continue
                etype = ev.get("type")
                if etype == "message_start":
                    u = (ev.get("message") or {}).get("usage") or {}
                    if "input_tokens" in u:
                        usage["input_tokens"] = int(u["input_tokens"])
                elif etype == "content_block_start":
                    block = ev.get("content_block") or {}
                    if block.get("type") == "tool_use":
                        blocks[int(ev.get("index", 0))] = {
                            "id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "partial_json": "",
                        }
                elif etype == "content_block_delta":
                    delta = ev.get("delta") or {}
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        content_parts.append(delta["text"])
                        yield text_delta(delta["text"])
                    elif delta.get("type") == "input_json_delta":
                        idx = int(ev.get("index", 0))
                        if idx in blocks:
                            blocks[idx]["partial_json"] += delta.get("partial_json", "")
                elif etype == "message_delta":
                    d = ev.get("delta") or {}
                    if d.get("stop_reason"):
                        stop_reason = d["stop_reason"]
                    u = ev.get("usage") or {}
                    if "output_tokens" in u:
                        usage["output_tokens"] = int(u["output_tokens"])
                elif etype == "message_stop":
                    break
    except httpx.HTTPError as exc:
        yield error_event(f"upstream request failed: {exc}")
        return

    tool_calls = [
        ToolCall(id=b["id"] or f"toolu_{i}", name=b["name"],
                 arguments=parse_tool_args(b["partial_json"]) if b["partial_json"] else {})
        for i, b in sorted(blocks.items())
    ]
    yield done_event(
        content="".join(content_parts),
        tool_calls=tool_calls,
        stop_reason="tool_use" if stop_reason == "tool_use" else "end",
        usage=usage,
    )
