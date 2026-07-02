"""Tests for the GraphInput node — declares a named input of the graph."""

from __future__ import annotations

import pytest

from app.core.api_contract import INPUT_TYPES, InputCoercionError
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
