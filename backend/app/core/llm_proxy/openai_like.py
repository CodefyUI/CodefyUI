"""Adapter for OpenAI-compatible chat-completions APIs.

Covers three providers: "openai" (api.openai.com), "openrouter"
(openrouter.ai), and "custom" (user-supplied base URL -- Ollama, LM Studio,
vLLM, test mocks). Hosts other than the user-configured custom base are
hard-coded -- this proxy is not an open relay.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ._common import TIMEOUT, content_to_text, parse_tool_args
from .events import done_event, error_event, text_delta
from .schema import ChatRequest, ToolCall

_BASES = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

# Back-compat alias -- internal callers that imported _parse_args directly
# (e.g. anthropic.py before its own refactor) continue to work.
_parse_args = parse_tool_args


def resolve_base_url(req: ChatRequest) -> str:
    if req.provider in _BASES:
        return _BASES[req.provider]
    if req.provider == "custom":
        base = (req.base_url or "").strip().rstrip("/")
        if not base.startswith(("http://", "https://")):
            raise ValueError("custom provider requires an http(s) base_url")
        return base
    raise ValueError(f"openai_like cannot serve provider {req.provider!r}")


def build_payload(req: ChatRequest) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for m in req.messages:
        if m.role == "tool":
            messages.append({"role": "tool", "tool_call_id": m.tool_call_id or "",
                             "content": content_to_text(m.content)})
        elif m.role == "assistant" and m.tool_calls:
            text_content = content_to_text(m.content)
            messages.append({
                "role": "assistant",
                "content": text_content or None,
                "tool_calls": [{
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                } for tc in m.tool_calls],
            })
        elif m.role in {"system", "assistant"}:
            messages.append({"role": m.role, "content": content_to_text(m.content)})
        else:
            messages.append({"role": m.role, "content": m.content})

    payload: dict[str, Any] = {
        "model": req.model,
        "messages": messages,
        "stream": True,
        "max_tokens": req.max_tokens,
    }
    # Older OpenAI-compatible servers (Ollama, LM Studio, etc.) reject unknown
    # fields -- only send stream_options for first-party providers.
    if req.provider != "custom":
        payload["stream_options"] = {"include_usage": True}
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.tools:
        payload["tools"] = [{
            "type": "function",
            "function": {"name": t.name, "description": t.description,
                         "parameters": t.input_schema},
        } for t in req.tools]
    return payload


async def stream_chat(
    req: ChatRequest, client: httpx.AsyncClient
) -> AsyncIterator[dict[str, Any]]:
    headers = {"content-type": "application/json"}
    if req.api_key:
        headers["authorization"] = f"Bearer {req.api_key}"

    content_parts: list[str] = []
    # OpenAI streams tool calls as index-keyed fragments.
    pending: dict[int, dict[str, str]] = {}
    finish_reason: str | None = None
    usage: dict[str, int] = {}

    try:
        # Resolve inside the try so an invalid custom base_url surfaces as an
        # error_event rather than raising synchronously to the caller.
        try:
            url = f"{resolve_base_url(req)}/chat/completions"
        except ValueError as exc:
            yield error_event(str(exc))
            return

        async with client.stream("POST", url, json=build_payload(req),
                                 headers=headers, timeout=TIMEOUT) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")[:500]
                yield error_event(f"upstream {resp.status_code}: {body}")
                return
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk.get("usage"), dict):
                    u = chunk["usage"]
                    usage = {"input_tokens": int(u.get("prompt_tokens", 0)),
                             "output_tokens": int(u.get("completion_tokens", 0))}
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
                    yield text_delta(delta["content"])
                for frag in delta.get("tool_calls") or []:
                    slot = pending.setdefault(int(frag.get("index", 0)),
                                              {"id": "", "name": "", "arguments": ""})
                    if frag.get("id"):
                        slot["id"] = frag["id"]
                    fn = frag.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
    except httpx.HTTPError as exc:
        yield error_event(f"upstream request failed: {exc}")
        return

    tool_calls = [
        ToolCall(id=slot["id"] or f"call_{i}", name=slot["name"],
                 arguments=parse_tool_args(slot["arguments"]))
        for i, slot in sorted(pending.items())
    ]
    stop_reason = "tool_use" if (tool_calls or finish_reason == "tool_calls") else "end"
    yield done_event(content="".join(content_parts), tool_calls=tool_calls,
                     stop_reason=stop_reason, usage=usage)
