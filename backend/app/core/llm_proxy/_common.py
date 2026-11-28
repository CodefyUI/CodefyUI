"""Shared helpers for provider adapters.

This module centralises constants and utilities used by more than one
adapter so callers import from a single authoritative location.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger("app.llm_proxy")

#: Seconds to connect to an upstream.
CONNECT_TIMEOUT_S = 10.0
#: Seconds an upstream may go without sending anything, by provider. A local
#: model that thinks before it answers (Ollama with a reasoning model) can be
#: silent for minutes on a cold start, so ``custom`` waits longer.
DEFAULT_READ_TIMEOUT_S = 120.0
CUSTOM_READ_TIMEOUT_S = 600.0
#: Overrides the read timeout for every provider, in seconds.
READ_TIMEOUT_ENV = "CODEFYUI_LLM_READ_TIMEOUT"

# Kept for callers that imported it: the default provider's timeout.
TIMEOUT = httpx.Timeout(CONNECT_TIMEOUT_S, read=DEFAULT_READ_TIMEOUT_S)


def read_timeout(provider: str) -> float:
    """The read timeout for *provider*: ``CODEFYUI_LLM_READ_TIMEOUT`` when it
    is a positive number of seconds, else 600 for ``custom`` and 120 for the
    rest. Read on every request, so a changed setting needs no restart."""
    raw = os.environ.get(READ_TIMEOUT_ENV, "").strip()
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = 0.0
        if value > 0:
            return value
        logger.warning("%s=%r is not a positive number of seconds; using the default",
                       READ_TIMEOUT_ENV, raw)
    return CUSTOM_READ_TIMEOUT_S if provider == "custom" else DEFAULT_READ_TIMEOUT_S


def timeout_for(provider: str) -> httpx.Timeout:
    return httpx.Timeout(CONNECT_TIMEOUT_S, read=read_timeout(provider))


def _host_of(exc: httpx.HTTPError) -> str:
    try:
        url = exc.request.url
    except RuntimeError:
        return "the endpoint"
    parts = urlsplit(str(url))
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return f"{parts.hostname}:{port}" if parts.hostname else "the endpoint"


def describe_failure(exc: httpx.HTTPError, provider: str) -> str:
    """``<ExceptionClass>: <a plain sentence>`` -- never empty.

    ``str(httpx.ReadTimeout())`` is the empty string, which is how a user
    once read "upstream request failed: " and nothing after it.
    """
    name = type(exc).__name__
    detail = str(exc).strip()
    if isinstance(exc, httpx.ReadTimeout):
        sentence = (f"the endpoint sent no data for {read_timeout(provider):g} seconds "
                    f"(set {READ_TIMEOUT_ENV} to wait longer)")
    elif isinstance(exc, httpx.ConnectTimeout):
        sentence = f"could not connect to {_host_of(exc)} within {CONNECT_TIMEOUT_S:g} seconds"
    elif isinstance(exc, httpx.ConnectError):
        sentence = f"cannot reach {_host_of(exc)}"
    elif isinstance(exc, httpx.WriteTimeout):
        sentence = "sending the request to the endpoint timed out"
    elif isinstance(exc, httpx.PoolTimeout):
        sentence = "no connection to the endpoint became free in time"
    elif isinstance(exc, httpx.RemoteProtocolError):
        sentence = "the endpoint closed the connection in the middle of the response"
    elif isinstance(exc, httpx.ReadError):
        sentence = "the connection to the endpoint broke while reading the response"
    else:
        sentence = "the request to the endpoint failed"
    if detail and detail not in sentence:
        sentence = f"{sentence} ({detail})"
    return f"{name}: {sentence}"


def upstream_failure(exc: httpx.HTTPError, provider: str, model: str) -> str:
    """The ``error`` event message for *exc*, logged once with the provider
    and model -- never the key, the URL's query or the conversation."""
    logger.warning("LLM upstream request failed: %s (provider=%s, model=%s)",
                   type(exc).__name__, provider, model)
    return f"upstream request failed: {describe_failure(exc, provider)}"


def should_retry(exc: httpx.HTTPError, *, received: bool, attempt: int) -> bool:
    """Retry once on a transport error before ANY byte of the response
    arrived (a dropped keep-alive connection, a refused connect while a
    local server restarts). Never after data arrived, which would repeat
    output, and never on a timeout, which would double a long wait."""
    return (attempt == 0 and not received
            and isinstance(exc, httpx.TransportError)
            and not isinstance(exc, httpx.TimeoutException))


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
