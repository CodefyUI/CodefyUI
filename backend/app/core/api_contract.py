"""Pure helpers for the graph-as-a-function I/O contract (GraphInput / GraphOutput).

A saved graph declares its transport-agnostic function signature with
``GraphInput`` / ``GraphOutput`` nodes on the canvas. This module derives
that contract from raw graph JSON, validates and injects caller-supplied
inputs, and collects/serializes declared outputs. Everything here is pure
JSON manipulation — no FastAPI imports (mirrors the CodefyUI-OJ patcher's
pure-JSON split); the endpoints in ``app.api.routes_graph_run`` stay thin.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .graph_engine import find_entry_points, reachable_from_entry_points

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


# Registry type strings of the contract nodes (frozen — they serialize into
# saved graphs forever).
GRAPH_INPUT_TYPE = "GraphInput"
GRAPH_OUTPUT_TYPE = "GraphOutput"

# Frozen contract-name charset: tightening later breaks saved graphs,
# loosening never does. Stage 2 OpenAPI and the future `cdui call
# --input k=v` need identifier-safe names.
NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,63}$")


@dataclass
class Contract:
    """The graph-level I/O contract derived from GraphInput / GraphOutput nodes.

    ``problems`` is reported non-fatally by ``GET /api/graph/contract`` and
    blocks ``POST /api/graph/run`` with 409 ``invalid_contract``.
    """

    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def derive_contract(nodes: list[dict]) -> Contract:
    """Scan top-level nodes for GraphInput / GraphOutput declarations.

    Presets are NOT expanded — contract derivation scans top-level nodes
    only (GraphInput/GraphOutput inside presets are a non-goal).
    """
    contract = Contract()
    for node in nodes:
        node_type = node.get("type", "")
        if node_type not in (GRAPH_INPUT_TYPE, GRAPH_OUTPUT_TYPE):
            continue
        data = node.get("data")
        params = data.get("params", {}) if isinstance(data, dict) else {}
        if node_type == GRAPH_INPUT_TYPE:
            contract.inputs.append({
                "name": str(params.get("name", "input")),
                "type": str(params.get("type", "string")),
                "required": bool(params.get("required", True)),
                "default": params.get("default", ""),
                "description": str(params.get("description", "")),
                "node_id": node.get("id", ""),
            })
        else:
            contract.outputs.append({
                "name": str(params.get("name", "output")),
                "description": str(params.get("description", "")),
                "node_id": node.get("id", ""),
            })

    if not contract.outputs:
        contract.problems.append(
            "graph has no GraphOutput node — declare at least one output"
        )

    _check_names(contract.inputs, "input", contract.problems)
    _check_names(contract.outputs, "output", contract.problems)

    for inp in contract.inputs:
        if inp["type"] == "image":
            if not inp["required"]:
                contract.problems.append(
                    f"image input '{inp['name']}' must be required=true "
                    "(base64 has no sensible API-side default)"
                )
            # The image default is a canvas-only file path, validated at
            # canvas run time — exempt from default parsing.
            continue
        if not inp["required"]:
            # The API applies only optional inputs' defaults; a required
            # input's default is a canvas-only test value and must NOT
            # 409-block API calls.
            try:
                coerce_input(inp["default"], inp["type"], from_string=True)
            except InputCoercionError as exc:
                contract.problems.append(
                    f"optional input '{inp['name']}': default does not parse "
                    f"as {inp['type']} ({exc.reason})"
                )
    return contract


def _check_names(
    entries: list[dict[str, Any]], kind: str, problems: list[str]
) -> None:
    """Empty-name, charset, and duplicate checks shared by inputs and outputs."""
    seen: set[str] = set()
    for entry in entries:
        name = entry["name"]
        if name == "":
            problems.append(f"{kind} node '{entry['node_id']}' has an empty name")
        elif not NAME_PATTERN.match(name):
            problems.append(
                f"{kind} name '{name}' is invalid — must match "
                "^[a-zA-Z_][a-zA-Z0-9_]{0,63}$"
            )
        if name in seen:
            problems.append(f"duplicate {kind} name '{name}'")
        seen.add(name)


@dataclass
class WiringReport:
    """Static pre-flight wiring findings on the raw (unexpanded) graph."""

    untriggered: list[str] = field(default_factory=list)  # GraphInput names
    unreachable: list[str] = field(default_factory=list)  # GraphOutput names


def check_wiring(
    nodes: list[dict], edges: list[dict], contract: Contract
) -> WiringReport:
    """Flag untriggered GraphInputs and unreachable GraphOutputs.

    Uses the exact traversal the engine prunes with (``find_entry_points``
    + ``reachable_from_entry_points``), applied statically to the raw
    graph: verified engine behaviour silently prunes untriggered data
    roots, so a clean report is what makes a run's declared I/O real.
    """
    report = WiringReport()
    triggered = {
        e["target"] for e in edges if e.get("type", "data") == "trigger"
    }
    for inp in contract.inputs:
        if inp["node_id"] not in triggered:
            report.untriggered.append(inp["name"])

    entry_ids = find_entry_points(nodes, edges)
    reachable = reachable_from_entry_points(entry_ids, edges)
    for out in contract.outputs:
        if out["node_id"] not in reachable:
            report.unreachable.append(out["name"])
    return report


def inject_inputs(
    nodes: list[dict],
    contract: Contract,
    request_inputs: dict[str, Any],
) -> tuple[list[dict], list[dict[str, str]]]:
    """Deep-copy *nodes* and write each request value into its GraphInput.

    The RAW JSON value — not the coerced result — lands in
    ``data.params["value"]``: the node's ``execute()`` performs the actual
    coercion, which avoids double coercion (an endpoint-coerced image
    tensor would fail the node's own base64 check), keeps injected params
    JSON-serializable, and decodes images exactly once (in the node);
    primitive types re-coerce idempotently. ``coerce_input`` is still
    called here — result discarded — to validate coercibility up front.
    Unknown names are rejected by case-sensitive exact match. All
    per-input errors (unknown name, missing required, coercion failure)
    are aggregated so the 422 reports everything at once.
    """
    errors: list[dict[str, str]] = []
    by_name = {inp["name"]: inp for inp in contract.inputs}

    for key in request_inputs:
        if key not in by_name:  # case-sensitive exact match
            errors.append({"input": key, "reason": "unknown input name"})

    to_inject: dict[str, Any] = {}  # node_id -> raw value
    for inp in contract.inputs:
        name = inp["name"]
        if name in request_inputs:
            raw = request_inputs[name]
            try:
                coerce_input(raw, inp["type"])  # validate only; inject raw
            except InputCoercionError as exc:
                errors.append({"input": name, "reason": exc.reason})
                continue
            to_inject[inp["node_id"]] = raw
        elif inp["required"]:
            errors.append({"input": name, "reason": "missing required input"})
        # Optional + omitted: no injection — the node's execute() falls
        # back to its parsed `default` param, identical to a canvas run.

    patched = copy.deepcopy(nodes)
    for node in patched:
        if node.get("id") in to_inject:
            data = node.setdefault("data", {})
            params = data.setdefault("params", {})
            params["value"] = to_inject[node["id"]]
    return patched, errors


def collect_outputs(
    contract: Contract,
    engine_result: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Read each declared output's ``value`` port from the engine result.

    ``missing`` survives only as a safety net behind the
    ``unreachable_output`` pre-flight check.
    """
    outputs: dict[str, Any] = {}
    missing: list[str] = []
    for out in contract.outputs:
        node_result = engine_result.get(out["node_id"])
        if isinstance(node_result, dict) and "value" in node_result:
            outputs[out["name"]] = node_result["value"]
        else:
            missing.append(out["name"])
    return outputs, missing


# A single tensor/ndarray output above this many elements fails
# serialization with ``output_too_large`` — callers should use
# ``record_outputs=true`` + the slicing outputs API instead.
MAX_TENSOR_ELEMENTS = 65536


class OutputSerializationError(Exception):
    """A value reaching a GraphOutput cannot be returned as JSON."""

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code  # "output_too_large" | "unserializable_output"
        self.reason = reason


def serialize_output(value: Any) -> Any:
    """Convert a value reaching a GraphOutput into a JSON-compatible form.

    Full values, not summaries (contrast the inspector wire shapes in
    ``routes_execution_outputs`` / ``ws_execution``). Tagged objects use
    the ``__type__`` key — intentionally diverging from the internal
    ``"type"``-keyed shapes so they can never collide with a user dict
    output that happens to contain a ``"type"`` key.
    """
    try:  # numpy scalars can subclass Python builtins (np.float64 -> float)
        import numpy as np
    except ImportError:
        np = None
    if np is not None and isinstance(value, np.generic):
        # numpy scalar (sklearn nodes emit these)
        return serialize_output(value.item())

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {key: serialize_output(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_output(item) for item in value]

    if np is not None and isinstance(value, np.ndarray):
        if value.size > MAX_TENSOR_ELEMENTS:
            raise OutputSerializationError(
                "output_too_large",
                f"array has {value.size} elements (max {MAX_TENSOR_ELEMENTS}); "
                "use record_outputs=true and the slicing outputs API "
                "(GET /api/execution/outputs/{run_id}/...) instead",
            )
        return {
            "__type__": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "values": value.tolist(),
        }

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        if isinstance(value, torch.Tensor):
            if value.numel() > MAX_TENSOR_ELEMENTS:
                raise OutputSerializationError(
                    "output_too_large",
                    f"tensor has {value.numel()} elements "
                    f"(max {MAX_TENSOR_ELEMENTS}); use record_outputs=true and "
                    "the slicing outputs API "
                    "(GET /api/execution/outputs/{run_id}/...) instead",
                )
            # 0-dim tensors keep shape [] rather than unwrapping — one rule,
            # no surprises.
            return {
                "__type__": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "values": value.detach().cpu().tolist(),
            }
        if isinstance(value, torch.nn.Module):
            raise OutputSerializationError(
                "unserializable_output",
                "torch.nn.Module cannot be returned as JSON — save it with a "
                "ModelSaver node in-graph and return its path string instead",
            )

    try:
        from PIL import Image
    except ImportError:
        Image = None
    if Image is not None and isinstance(value, Image.Image):
        import base64
        import io

        buf = io.BytesIO()
        value.save(buf, format="PNG")
        return {
            "__type__": "image",
            "format": "png",
            "base64": base64.b64encode(buf.getvalue()).decode("ascii"),
        }

    raise OutputSerializationError(
        "unserializable_output",
        f"value of type {type(value).__name__} is not JSON-serializable",
    )
