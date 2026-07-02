"""Pure helpers for the graph-as-a-function I/O contract (GraphInput / GraphOutput).

A saved graph declares its transport-agnostic function signature with
``GraphInput`` / ``GraphOutput`` nodes on the canvas. This module derives
that contract from raw graph JSON, validates and injects caller-supplied
inputs, and collects/serializes declared outputs. Everything here is pure
JSON manipulation — no FastAPI imports (mirrors the CodefyUI-OJ patcher's
pure-JSON split); the endpoints in ``app.api.routes_graph_run`` stay thin.
"""

from __future__ import annotations

import json
from typing import Any

# Input `type` values (v1, frozen). `json` stays `json` — it describes
# exactly what a caller can send; no new DataType is needed.
INPUT_TYPES = ("string", "number", "integer", "boolean", "json", "image")


class InputCoercionError(Exception):
    """A caller-supplied value cannot be coerced to its declared input type."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def json_type_name(value: Any) -> str:
    """Name *value*'s JSON type for error messages (bool checked before int)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def coerce_input(value: Any, declared_type: str, *, from_string: bool = False) -> Any:
    """Coerce a value to *declared_type*; raise :class:`InputCoercionError`.

    ``from_string=False`` (the API body path) applies the strict JSON table:
    no string-to-number coercion, bools are never numbers, and ``integer``
    accepts integral floats (``3.0 -> 3``, JSON Schema / OpenAPI 3.1
    alignment — JS clients cannot control whether 3 serializes as 3.0).

    ``from_string=True`` (the canvas ``default`` param path / an omitted
    optional input) additionally parses string values per type — the one
    deliberate asymmetry, documented on the node's ``default`` param.
    Primitive values re-coerce idempotently under either mode.
    """
    if (
        from_string
        and isinstance(value, str)
        and declared_type in ("number", "integer", "boolean", "json")
    ):
        # Parsed value falls through to the strict table below (idempotent).
        value = _parse_default_string(value, declared_type)
    if declared_type == "string":
        if isinstance(value, str):
            return value
        raise InputCoercionError(f"expected string, got {json_type_name(value)}")
    if declared_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InputCoercionError(f"expected number, got {json_type_name(value)}")
        return float(value)
    if declared_type == "integer":
        if isinstance(value, bool):
            raise InputCoercionError("expected integer, got boolean")
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, float):
            raise InputCoercionError(f"expected integer, got non-integral number {value}")
        raise InputCoercionError(f"expected integer, got {json_type_name(value)}")
    if declared_type == "boolean":
        if isinstance(value, bool):
            return value
        raise InputCoercionError(f"expected boolean, got {json_type_name(value)}")
    if declared_type == "json":
        return value
    if declared_type == "image":
        return _decode_image(value)
    raise InputCoercionError(f"unknown input type '{declared_type}'")


def _parse_default_string(raw: str, declared_type: str) -> Any:
    """Parse a string ``default`` param per type (canvas / omitted-optional path)."""
    text = raw.strip()
    if declared_type == "number":
        try:
            return float(text)
        except ValueError:
            raise InputCoercionError(f"default {raw!r} does not parse as number")
    if declared_type == "integer":
        try:
            return int(text)
        except ValueError:
            pass
        try:
            # Integral float strings like "3.0" pass the strict table below.
            return float(text)
        except ValueError:
            raise InputCoercionError(f"default {raw!r} does not parse as integer")
    if declared_type == "boolean":
        lowered = text.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        raise InputCoercionError(
            f"default {raw!r} does not parse as boolean (use 'true'/'false')"
        )
    if declared_type == "json":
        try:
            return json.loads(raw)
        except ValueError:
            raise InputCoercionError(f"default {raw!r} does not parse as JSON")
    return raw


def _decode_image(value: Any) -> Any:
    """Decode a base64 image string to a ``(C, H, W)`` float32 tensor in [0, 1].

    Matches what image consumers already receive from ``ImageReader``.
    Accepts an optional ``data:image/...;base64,`` prefix and tolerates
    whitespace/newline-wrapped base64. Torch/PIL imports are lazy so
    torch-free callers of the other helpers never pay for them.
    """
    if not isinstance(value, str):
        raise InputCoercionError(
            f"expected base64 image string, got {json_type_name(value)}"
        )
    import base64
    import binascii
    import io

    payload = value
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    payload = "".join(payload.split())  # tolerate wrapped base64
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise InputCoercionError("image value is not valid base64")
    from PIL import Image, UnidentifiedImageError
    from torchvision import transforms

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise InputCoercionError("base64 payload does not decode to an image")
    img = img.convert("RGB")
    return transforms.ToTensor()(img)
