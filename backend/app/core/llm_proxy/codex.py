"""Adapter for the ChatGPT Codex backend (subscription-quota chat).

Speaks the Responses API dialect against chatgpt.com/backend-api/codex.
Constraints verified against the open-source Codex CLI (2026-06): the
"originator" header is allowlisted server-side; "instructions" is required;
"store" must be false; user content items are "input_text" while assistant
history items are "output_text".
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from . import codex_auth, codex_models
from ._common import TIMEOUT, content_to_text, parse_tool_args
from .events import done_event, error_event, text_delta
from .schema import ChatRequest, ToolCall

_URL = "https://chatgpt.com/backend-api/codex/responses"

# Backward-compatible id-only view for older in-process callers.  The HTTP
# route uses the dynamic rich catalog in ``codex_models``.
STATIC_MODELS = [model["id"] for model in codex_models.fallback_models()]


def build_payload(req: ChatRequest) -> dict[str, Any]:
    system_parts = [
        content_to_text(m.content)
        for m in req.messages
        if m.role == "system" and content_to_text(m.content)
    ]
    items: list[dict[str, Any]] = []
    for m in req.messages:
        if m.role == "system":
            continue
        if m.role == "tool":
            items.append({"type": "function_call_output",
                          "call_id": m.tool_call_id or "",
                          "output": content_to_text(m.content)})
            continue
        if m.role == "assistant":
            text_content = content_to_text(m.content)
            if text_content:
                items.append({"type": "message", "role": "assistant",
                              "content": [{"type": "output_text", "text": text_content}]})
            for tc in m.tool_calls or []:
                items.append({"type": "function_call", "call_id": tc.id,
                              "name": tc.name, "arguments": json.dumps(tc.arguments)})
            continue
        items.append({"type": "message", "role": m.role,
                      "content": [{"type": "input_text", "text": content_to_text(m.content)}]})

    payload: dict[str, Any] = {
        "model": req.model,
        "instructions": "\n\n".join(system_parts) or "You are a helpful assistant.",
        "input": items,
        "tools": [{"type": "function", "name": t.name, "description": t.description,
                   "parameters": t.input_schema, "strict": False} for t in req.tools],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "store": False,
        "stream": True,
    }
    if req.reasoning_effort is not None:
        payload["reasoning"] = {"effort": req.reasoning_effort}
    return payload


async def stream_chat(
    req: ChatRequest, client: httpx.AsyncClient
) -> AsyncIterator[dict[str, Any]]:
    try:
        access, account = await codex_auth.get_valid_access(client)
    except codex_auth.CodexNotLoggedIn:
        yield error_event("Not signed in to ChatGPT - open Settings to sign in.")
        return
    except httpx.HTTPError as exc:
        yield error_event(f"token refresh failed: {exc}")
        return

    payload = build_payload(req)
    content_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    usage: dict[str, int] = {}
    failed: str | None = None

    for attempt in (0, 1):
        headers = {
            "authorization": f"Bearer {access}",
            "chatgpt-account-id": account,
            "originator": codex_auth.ORIGINATOR,
            "content-type": "application/json",
            "accept": "text/event-stream",
        }
        try:
            async with client.stream("POST", _URL, json=payload, headers=headers,
                                     timeout=TIMEOUT) as resp:
                if resp.status_code == 401 and attempt == 0:
                    try:
                        access, account = await codex_auth.get_valid_access(
                            client, force_refresh=True)
                    except (codex_auth.CodexNotLoggedIn, httpx.HTTPError) as exc:
                        yield error_event(f"ChatGPT session expired: {exc}")
                        return
                    continue
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
                    if etype == "response.output_text.delta":
                        delta = ev.get("delta", "")
                        if delta:
                            content_parts.append(delta)
                            yield text_delta(delta)
                    elif etype == "response.output_item.done":
                        item = ev.get("item") or {}
                        if item.get("type") == "function_call":
                            tool_calls.append(ToolCall(
                                id=item.get("call_id", f"fc_{len(tool_calls)}"),
                                name=item.get("name", ""),
                                arguments=parse_tool_args(item.get("arguments", "")),
                            ))
                    elif etype == "response.completed":
                        u = (ev.get("response") or {}).get("usage") or {}
                        usage = {k: int(u[k]) for k in ("input_tokens", "output_tokens")
                                 if isinstance(u.get(k), (int, float))}
                    elif etype == "response.failed":
                        err = ((ev.get("response") or {}).get("error") or {})
                        failed = err.get("message") or "response.failed"
            break
        except httpx.HTTPError as exc:
            yield error_event(f"upstream request failed: {exc}")
            return

    if failed:
        yield error_event(failed)
        return
    yield done_event(
        content="".join(content_parts),
        tool_calls=tool_calls,
        stop_reason="tool_use" if tool_calls else "end",
        usage=usage,
    )
