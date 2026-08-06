"""Tests for the GraphInput node — declares a named input of the graph."""

from __future__ import annotations

import pytest

from app.core.api_contract import INPUT_TYPES, InputCoercionError
from app.core.cache import ExecutionCache
from app.core.graph_engine import execute_graph
from app.core.node_base import DataType, ParamType
from app.nodes.io.graph_input_node import GraphInputNode


def test_metadata():
    assert GraphInputNode.NODE_NAME == "GraphInput"
    assert GraphInputNode.CATEGORY == "IO"
    assert "API" in GraphInputNode.DESCRIPTION  # palette search finds it
    assert GraphInputNode.cacheable is True


def test_ports():
    assert GraphInputNode.define_inputs() == []
    outputs = GraphInputNode.define_outputs()
    assert len(outputs) == 1
    assert outputs[0].name == "value"
    assert outputs[0].data_type == DataType.ANY


def test_declared_params_exclude_value():
    params = GraphInputNode.define_params()
    names = [p.name for p in params]
    assert names == ["name", "type", "required", "default", "description"]
    assert "value" not in names  # the injected param must never render in the UI
    by_name = {p.name: p for p in params}
    assert by_name["name"].default == "input"
    assert by_name["type"].param_type == ParamType.SELECT
    # Options must track INPUT_TYPES exactly — no hand-maintained duplicate
    # list that can drift from the contract's source of truth.
    assert by_name["type"].options == list(INPUT_TYPES)
    assert by_name["type"].options == [
        "string", "number", "integer", "boolean", "json", "image",
    ]
    assert by_name["type"].default == "string"
    assert by_name["required"].param_type == ParamType.BOOL
    assert by_name["required"].default is True
    assert by_name["default"].default == ""
    assert by_name["description"].default == ""


def test_injected_value_takes_precedence_over_default():
    res = GraphInputNode().execute(
        {}, {"type": "string", "value": "from-api", "default": "from-canvas"}
    )
    assert res == {"value": "from-api"}


def test_canvas_default_string_parsing_number():
    res = GraphInputNode().execute({}, {"type": "number", "default": "2.5"})
    assert res == {"value": 2.5}


def test_canvas_default_boolean_and_json():
    assert GraphInputNode().execute({}, {"type": "boolean", "default": "true"}) == {
        "value": True
    }
    assert GraphInputNode().execute({}, {"type": "json", "default": '{"a": 1}'}) == {
        "value": {"a": 1}
    }


def test_injected_integral_float_integer():
    assert GraphInputNode().execute({}, {"type": "integer", "value": 3.0}) == {"value": 3}
    with pytest.raises(InputCoercionError):
        GraphInputNode().execute({}, {"type": "integer", "value": 3.5})


def test_injected_value_strict_no_string_parsing():
    with pytest.raises(InputCoercionError):
        GraphInputNode().execute({}, {"type": "number", "value": "2.5"})


def test_missing_type_defaults_to_string():
    assert GraphInputNode().execute({}, {"default": "plain"}) == {"value": "plain"}


def test_image_default_empty_raises_clear_canvas_error():
    with pytest.raises(ValueError, match="server-local image path"):
        GraphInputNode().execute({}, {"type": "image", "default": ""})


def test_image_default_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        GraphInputNode().execute({}, {"type": "image", "default": "no_such_file_12345.png"})


def test_image_default_loads_file(tmp_path):
    import torch
    from PIL import Image

    p = tmp_path / "tiny.png"
    Image.new("RGB", (4, 2), color=(0, 255, 0)).save(p)
    res = GraphInputNode().execute({}, {"type": "image", "default": str(p)})
    assert isinstance(res["value"], torch.Tensor)
    assert res["value"].shape == (3, 2, 4)


def test_image_injected_value_is_decoded_base64():
    import base64
    import io

    import torch
    from PIL import Image

    img = Image.new("RGB", (4, 2), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    res = GraphInputNode().execute({}, {"type": "image", "value": b64})
    assert isinstance(res["value"], torch.Tensor)
    assert res["value"].shape == (3, 2, 4)


def test_registry_discovers_graph_input():
    from app.core.node_registry import registry

    assert registry.get("GraphInput") is GraphInputNode


# ── cache_fingerprint (#145) ─────────────────────────────────────────────
#
# GraphInputNode stays cacheable=True unconditionally (the class-level
# comment on it is load-bearing: on the API path, `value` is already in
# `params`, so the ordinary cache key is already complete). The gap #145
# closes is narrower: on a CANVAS run with type=image, `default` is a PATH,
# not the pixels, so only the fingerprint hook can see a replaced file.


def test_cache_fingerprint_is_none_on_the_api_path():
    """`value` already reaches the cache key via params; no extra
    fingerprint is needed, and computing one from `default` would be
    actively wrong (an API caller need not set `default` at all)."""
    assert GraphInputNode.cache_fingerprint(
        {"type": "image", "value": "c29tZS1iYXNlNjQ=", "default": ""}
    ) is None


def test_cache_fingerprint_is_none_for_non_image_types_on_canvas():
    """For every other type, `default` (a plain string) IS the value --
    already fully captured by params, so no extra fingerprint applies."""
    assert GraphInputNode.cache_fingerprint({"type": "string", "default": "hello"}) is None
    assert GraphInputNode.cache_fingerprint({"type": "number", "default": "2.5"}) is None
    assert GraphInputNode.cache_fingerprint({"type": "boolean", "default": "true"}) is None


def test_cache_fingerprint_is_none_when_default_is_empty():
    assert GraphInputNode.cache_fingerprint({"type": "image", "default": ""}) is None


def test_cache_fingerprint_changes_when_the_canvas_image_file_changes(tmp_path):
    from PIL import Image

    p = tmp_path / "pic.png"
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(p)
    fp1 = GraphInputNode.cache_fingerprint({"type": "image", "default": str(p)})

    Image.new("RGB", (2, 2), color=(0, 255, 0)).save(p)  # replaced
    fp2 = GraphInputNode.cache_fingerprint({"type": "image", "default": str(p)})

    assert fp1 is not None
    assert fp1 != fp2


# ── end-to-end through the real engine (#145) ───────────────────────────
#
# The mechanism itself (does the engine actually consult cache_fingerprint
# and fold it into the key) is proven generically, via a synthetic node,
# in test_cache_content_fingerprint.py. These two compose that already-
# proven wiring with the node-level fix above against the real node, for
# direct coverage of the issue's own repro steps and acceptance bar.


async def _run(nodes, edges, cache) -> dict[str, tuple[str, dict | None]]:
    """Execute once; return ``{node_id: (status, outputs)}``."""
    seen: dict[str, tuple[str, dict | None]] = {}

    async def track(node_id, status, data):
        if status in ("completed", "cached"):
            seen[node_id] = (status, data)

    await execute_graph(nodes, edges, on_progress=track, cache=cache)
    return seen


@pytest.mark.asyncio
async def test_canvas_image_run_is_cached_when_unchanged_and_fresh_when_replaced(tmp_path):
    """The #145 repro, verbatim: a canvas graph with GraphInput(type=image,
    default=<path>). Run it; replace the image on disk; re-run. The pixels
    must be the new ones, not the first run's stale tensor -- and, since
    GraphInput stays cacheable=True throughout (the class-level contract
    #145 is explicit must not regress), an UNCHANGED image must still hit
    the cache rather than paying a re-decode on every run.
    """
    from PIL import Image

    img_path = tmp_path / "canvas_pic.png"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(img_path)  # red
    cache = ExecutionCache()
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {
            "id": "gi", "type": "GraphInput",
            "data": {"params": {
                "name": "photo", "type": "image", "required": True,
                "default": str(img_path), "description": "",
            }},
        },
    ]
    edges = [
        {"id": "et", "source": "start", "target": "gi", "sourceHandle": "trigger", "type": "trigger"},
    ]

    first = await _run(nodes, edges, cache)
    assert first["gi"][0] == "completed"
    assert first["gi"][1]["value"][0, 0, 0].item() == pytest.approx(1.0)  # red channel

    second = await _run(nodes, edges, cache)
    assert second["gi"][0] == "cached", (
        "an untouched canvas image must hit the cache -- GraphInput's "
        "cacheable=True contract must not regress"
    )

    Image.new("RGB", (4, 4), color=(0, 255, 0)).save(img_path)  # replaced: green
    third = await _run(nodes, edges, cache)
    assert third["gi"][0] == "completed", (
        "a replaced canvas image must bust the cache -- it was served from "
        f"cache instead (status={third['gi'][0]!r}), i.e. the #145 stale-pixels bug"
    )
    assert third["gi"][1]["value"][1, 0, 0].item() == pytest.approx(1.0)  # green channel


@pytest.mark.asyncio
async def test_api_path_still_caches_on_repeat_identical_value():
    """#145's other acceptance bar: 'API runs keep their current cache
    behaviour'. On the API path `value` (not `default`) carries the
    pixels, already fully in params -- unaffected by cache_fingerprint,
    which returns None there. Same value twice must still hit the cache.
    """
    import base64
    import io

    from PIL import Image

    img = Image.new("RGB", (4, 4), color=(123, 45, 67))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    cache = ExecutionCache()
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {
            "id": "gi", "type": "GraphInput",
            "data": {"params": {
                "name": "photo", "type": "image", "required": True,
                "default": "", "description": "", "value": b64,
            }},
        },
    ]
    edges = [
        {"id": "et", "source": "start", "target": "gi", "sourceHandle": "trigger", "type": "trigger"},
    ]

    first = await _run(nodes, edges, cache)
    assert first["gi"][0] == "completed"

    second = await _run(nodes, edges, cache)
    assert second["gi"][0] == "cached", (
        "repeating the same API-injected image value must still hit the "
        "cache -- #145 must not regress published-app performance"
    )
