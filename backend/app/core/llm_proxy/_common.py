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
