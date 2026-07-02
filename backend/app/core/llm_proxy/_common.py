"""Shared helpers for provider adapters.

This module centralises constants and utilities used by more than one
adapter so callers import from a single authoritative location.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

# Upstream connect/read timeout.  Reads are long (model thinking time between
# SSE chunks) -- 120 s idle gap tolerated, 10 s to connect.
TIMEOUT = httpx.Timeout(10.0, read=120.0)


def parse_tool_args(raw: str) -> dict[str, Any]:
    """Parse a JSON string of tool-call arguments into a dict.

    Returns an empty dict for empty input and a ``{"__parse_error__": raw}``
    sentinel when the string is not valid JSON or not a JSON object -- so
    callers always receive a dict without raising.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"__parse_error__": raw}
    except json.JSONDecodeError:
        return {"__parse_error__": raw}


def content_to_text(content: Any) -> str:
    """Flatten string or multimodal message content into provider-safe text."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            ptype = part.get("type")
            if ptype == "text":
                text = part.get("text", "")
                if text:
                    parts.append(str(text))
            elif ptype == "image_url":
                parts.append("[image]")
            else:
                parts.append(json.dumps(part, ensure_ascii=False, default=str))
        else:
            parts.append(str(part))
    return "\n".join(p for p in parts if p)


def data_url_to_anthropic_source(url: str) -> dict[str, str] | None:
    """Convert a data:image/... URL into Anthropic's base64 image source."""
    if not url.startswith("data:") or ";base64," not in url:
        return None
    header, data = url[5:].split(",", 1)
    media_type = header.split(";", 1)[0] or "image/png"
    if not media_type.startswith("image/") or not data:
        return None
    return {"type": "base64", "media_type": media_type, "data": data}


def content_to_anthropic_blocks(content: Any) -> str | list[dict[str, Any]]:
    """Map OpenAI-style content parts to Anthropic Messages content blocks."""
    if not isinstance(content, list):
        return content_to_text(content)

    blocks: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            text = str(part)
            if text:
                blocks.append({"type": "text", "text": text})
            continue

        ptype = part.get("type")
        if ptype == "text":
            text = str(part.get("text", ""))
            if text:
                blocks.append({"type": "text", "text": text})
        elif ptype == "image_url":
            image_url = part.get("image_url") or {}
            url = image_url.get("url") if isinstance(image_url, dict) else None
            source = data_url_to_anthropic_source(str(url or ""))
            if source is not None:
                blocks.append({"type": "image", "source": source})
            else:
                blocks.append({"type": "text", "text": "[unsupported image URL]"})
        else:
            blocks.append({
                "type": "text",
                "text": json.dumps(part, ensure_ascii=False, default=str),
            })
    return blocks or ""
