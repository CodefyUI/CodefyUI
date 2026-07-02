"""Tests for app.core.api_contract — pure graph-as-a-function contract helpers."""

from __future__ import annotations

import pytest

from app.core.api_contract import InputCoercionError, coerce_input, json_type_name


def _tiny_png_base64() -> str:
    import base64
    import io

    from PIL import Image

    img = Image.new("RGB", (4, 2), color=(255, 0, 0))  # width=4, height=2
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── coerce_input: strict API table (from_string=False) ──────────────────


def test_string_passes_through():
    assert coerce_input("hi", "string") == "hi"


def test_string_rejects_non_strings():
    for bad in (3, 2.5, True, None, {"a": 1}, [1]):
        with pytest.raises(InputCoercionError):
            coerce_input(bad, "string")


def test_rejection_names_expected_vs_received():
    with pytest.raises(InputCoercionError, match="expected string, got number"):
        coerce_input(3, "string")
    with pytest.raises(InputCoercionError, match="expected number, got string"):
        coerce_input("3", "number")
    with pytest.raises(InputCoercionError, match="expected boolean, got number"):
        coerce_input(1, "boolean")


def test_number_accepts_int_and_float_as_float():
    assert coerce_input(3, "number") == 3.0
    assert isinstance(coerce_input(3, "number"), float)
    assert coerce_input(2.5, "number") == 2.5


def test_number_rejects_string_bool_null():
    for bad in ("3", True, False, None, [1]):
        with pytest.raises(InputCoercionError):
            coerce_input(bad, "number")


def test_integer_accepts_int_and_integral_float():
    assert coerce_input(3, "integer") == 3
    assert coerce_input(3.0, "integer") == 3
    assert isinstance(coerce_input(3.0, "integer"), int)


def test_integer_rejects_fractional_string_bool():
    for bad in (3.5, "3", True, None):
        with pytest.raises(InputCoercionError):
            coerce_input(bad, "integer")


def test_boolean_accepts_only_bool():
    assert coerce_input(True, "boolean") is True
    assert coerce_input(False, "boolean") is False
    for bad in (0, 1, "true", None):
        with pytest.raises(InputCoercionError):
            coerce_input(bad, "boolean")


def test_json_accepts_any_json_value():
    for val in ({"a": 1}, [1, 2], "s", 3, 2.5, True, None):
        assert coerce_input(val, "json") == val


def test_unknown_declared_type_raises():
    with pytest.raises(InputCoercionError, match="unknown input type"):
        coerce_input("x", "tensor")


# ── coerce_input: from_string=True (canvas `default` path) ──────────────


def test_from_string_parses_number_integer_boolean_json():
    assert coerce_input("2.5", "number", from_string=True) == 2.5
    assert coerce_input("3", "integer", from_string=True) == 3
    assert coerce_input("3.0", "integer", from_string=True) == 3
    assert coerce_input("true", "boolean", from_string=True) is True
    assert coerce_input("False", "boolean", from_string=True) is False
    assert coerce_input('{"k": [1, 2]}', "json", from_string=True) == {"k": [1, 2]}


def test_from_string_string_type_stays_verbatim():
    assert coerce_input("3.5", "string", from_string=True) == "3.5"


def test_from_string_bad_parses_raise():
    with pytest.raises(InputCoercionError):
        coerce_input("abc", "number", from_string=True)
    with pytest.raises(InputCoercionError):
        coerce_input("3.5", "integer", from_string=True)
    with pytest.raises(InputCoercionError):
        coerce_input("yes", "boolean", from_string=True)
    with pytest.raises(InputCoercionError):
        coerce_input("{not json", "json", from_string=True)


def test_from_string_false_never_parses_strings():
    with pytest.raises(InputCoercionError):
        coerce_input("2.5", "number")


def test_from_string_non_string_value_uses_strict_table():
    # A hand-edited graph JSON may hold a non-string default; strict rules apply.
    assert coerce_input(5, "number", from_string=True) == 5.0
    with pytest.raises(InputCoercionError):
        coerce_input(True, "number", from_string=True)


# ── coerce_input: image ─────────────────────────────────────────────────


def test_image_decodes_to_chw_float_tensor():
    import torch

    tensor = coerce_input(_tiny_png_base64(), "image")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 2, 4)  # (C, H, W)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0


def test_image_accepts_data_uri_prefix():
    import torch

    tensor = coerce_input("data:image/png;base64," + _tiny_png_base64(), "image")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 2, 4)


def test_image_rejects_non_string_and_garbage():
    with pytest.raises(InputCoercionError, match="expected base64 image string"):
        coerce_input(123, "image")
    with pytest.raises(InputCoercionError, match="not valid base64"):
        coerce_input("!!!not-base64!!!", "image")
    with pytest.raises(InputCoercionError, match="does not decode to an image"):
        coerce_input("aGVsbG8gd29ybGQ=", "image")  # base64("hello world")


# ── json_type_name ───────────────────────────────────────────────────────


def test_json_type_name_labels():
    assert json_type_name(None) == "null"
    assert json_type_name(True) == "boolean"
    assert json_type_name(3) == "number"
    assert json_type_name(2.5) == "number"
    assert json_type_name("s") == "string"
    assert json_type_name([1]) == "array"
    assert json_type_name({}) == "object"


# ── derive_contract ──────────────────────────────────────────────────────

from app.core.api_contract import derive_contract  # noqa: E402


def _gi(node_id: str, **params) -> dict:
    merged = {
        "name": "input", "type": "string", "required": True,
        "default": "", "description": "",
    }
    merged.update(params)
    return {
        "id": node_id, "type": "GraphInput",
        "position": {"x": 0, "y": 0}, "data": {"params": merged},
    }


def _go(node_id: str, **params) -> dict:
    merged = {"name": "output", "description": ""}
    merged.update(params)
    return {
        "id": node_id, "type": "GraphOutput",
        "position": {"x": 0, "y": 0}, "data": {"params": merged},
    }


def _other(node_id: str, node_type: str = "_TestSource") -> dict:
    return {
        "id": node_id, "type": node_type,
        "position": {"x": 0, "y": 0}, "data": {"params": {}},
    }


def test_derive_contract_collects_inputs_and_outputs():
    nodes = [
        _other("s", "Start"),
        _gi("i1", name="prompt", type="string"),
        _go("o1", name="answer", description="the answer"),
    ]
    contract = derive_contract(nodes)
    assert contract.problems == []
    assert contract.inputs == [{
        "name": "prompt", "type": "string", "required": True,
        "default": "", "description": "", "node_id": "i1",
    }]
    assert contract.outputs == [
        {"name": "answer", "description": "the answer", "node_id": "o1"},
    ]


def test_derive_contract_no_graph_output_is_problem():
    contract = derive_contract([_gi("i1", name="x")])
    assert any("GraphOutput" in p for p in contract.problems)


def test_derive_contract_empty_and_bad_charset_names():
    contract = derive_contract([
        _gi("i1", name=""),
        _gi("i2", name="9lives"),
        _gi("i3", name="has space"),
        _go("o1", name="ok_name"),
    ])
    assert any("empty name" in p for p in contract.problems)
    assert sum("is invalid" in p for p in contract.problems) == 2


def test_derive_contract_name_at_charset_limits():
    contract = derive_contract([
        _gi("i1", name="_x" + "a" * 62),   # 64 chars total: OK
        _gi("i2", name="b" * 65),          # 65 chars: too long
        _go("o1", name="y"),
    ])
    assert sum("is invalid" in p for p in contract.problems) == 1


def test_derive_contract_duplicate_names():
    contract = derive_contract([
        _gi("i1", name="x"), _gi("i2", name="x"),
        _go("o1", name="y"), _go("o2", name="y"),
    ])
    assert any(p == "duplicate input name 'x'" for p in contract.problems)
    assert any(p == "duplicate output name 'y'" for p in contract.problems)


def test_derive_contract_optional_default_must_parse():
    contract = derive_contract([
        _gi("i1", name="n", type="number", required=False, default="abc"),
        _go("o1", name="y"),
    ])
    assert any("default does not parse" in p for p in contract.problems)


def test_derive_contract_required_bad_default_is_not_a_problem():
    # A required input's default is a canvas-only test value: its failure
    # surfaces on canvas runs and must NOT 409-block API calls.
    contract = derive_contract([
        _gi("i1", name="n", type="number", required=True, default="abc"),
        _go("o1", name="y"),
    ])
    assert contract.problems == []


def test_derive_contract_image_default_exempt_from_parsing():
    # An image default is a canvas-only file path, validated at canvas run time.
    contract = derive_contract([
        _gi("i1", name="img", type="image", required=True, default="photo.png"),
        _go("o1", name="y"),
    ])
    assert contract.problems == []


def test_derive_contract_optional_image_is_problem():
    contract = derive_contract([
        _gi("i1", name="img", type="image", required=False),
        _go("o1", name="y"),
    ])
    assert any("must be required" in p for p in contract.problems)


def test_derive_contract_ignores_preset_and_other_nodes():
    nodes = [_other("p1", "preset:TrainLoop"), _other("t1"), _go("o1", name="y")]
    contract = derive_contract(nodes)
    assert contract.inputs == []
    assert len(contract.outputs) == 1


# ── check_wiring ─────────────────────────────────────────────────────────

from app.core.api_contract import check_wiring  # noqa: E402


def _trigger_edge(eid: str, src: str, tgt: str) -> dict:
    return {
        "id": eid, "source": src, "target": tgt,
        "sourceHandle": "trigger", "targetHandle": "", "type": "trigger",
    }


def _data_edge(eid: str, src: str, tgt: str,
               src_handle: str = "value", tgt_handle: str = "value") -> dict:
    return {
        "id": eid, "source": src, "target": tgt,
        "sourceHandle": src_handle, "targetHandle": tgt_handle, "type": "data",
    }


def test_check_wiring_clean_graph():
    nodes = [_other("s", "Start"), _gi("i1", name="x"), _go("o1", name="y")]
    edges = [_trigger_edge("t1", "s", "i1"), _data_edge("d1", "i1", "o1")]
    report = check_wiring(nodes, edges, derive_contract(nodes))
    assert report.untriggered == []
    assert report.unreachable == []


def test_check_wiring_untriggered_input():
    # i1 feeds o1 but has no trigger; a *different* node is triggered, so the
    # engine would silently prune i1 and run "successfully" without the
    # caller's input — exactly what the 409 pre-flight must prevent.
    nodes = [_other("s", "Start"), _other("src"), _gi("i1", name="x"), _go("o1", name="y")]
    edges = [_trigger_edge("t1", "s", "src"), _data_edge("d1", "i1", "o1")]
    report = check_wiring(nodes, edges, derive_contract(nodes))
    assert report.untriggered == ["x"]
    # o1 is fed only through the untriggered i1, so it is also unreachable.
    assert report.unreachable == ["y"]


def test_check_wiring_unreachable_output():
    nodes = [
        _other("s", "Start"), _gi("i1", name="x"),
        _go("o1", name="y1"), _go("o2", name="y2"), _other("src2"),
    ]
    edges = [
        _trigger_edge("t1", "s", "i1"),
        _data_edge("d1", "i1", "o1"),
        _data_edge("d2", "src2", "o2"),  # src2 untriggered -> pruned at run time
    ]
    report = check_wiring(nodes, edges, derive_contract(nodes))
    assert report.untriggered == []
    assert report.unreachable == ["y2"]


def test_check_wiring_no_entry_points_marks_everything():
    nodes = [_gi("i1", name="x"), _go("o1", name="y")]
    edges = [_data_edge("d1", "i1", "o1")]
    report = check_wiring(nodes, edges, derive_contract(nodes))
    assert report.untriggered == ["x"]
    assert report.unreachable == ["y"]


# ── inject_inputs / collect_outputs ──────────────────────────────────────

from app.core.api_contract import collect_outputs, inject_inputs  # noqa: E402


def test_inject_inputs_writes_raw_value_and_preserves_original():
    nodes = [_other("s", "Start"), _gi("i1", name="x", type="integer"), _go("o1", name="y")]
    contract = derive_contract(nodes)
    patched, errors = inject_inputs(nodes, contract, {"x": 3.0})
    assert errors == []
    patched_params = patched[1]["data"]["params"]
    assert patched_params["value"] == 3.0            # RAW value, not int(3)
    assert isinstance(patched_params["value"], float)
    assert "value" not in nodes[1]["data"]["params"]  # deep-copy: original untouched


def test_inject_inputs_image_stays_base64_string():
    import json as _json

    nodes = [_gi("i1", name="img", type="image"), _go("o1", name="y")]
    contract = derive_contract(nodes)
    b64 = _tiny_png_base64()
    patched, errors = inject_inputs(nodes, contract, {"img": b64})
    assert errors == []
    injected = patched[0]["data"]["params"]["value"]
    assert injected == b64        # decoded once, in the node — not here
    _json.dumps(injected)         # injected params stay JSON-serializable


def test_inject_inputs_aggregates_all_errors():
    nodes = [
        _gi("i1", name="a", type="number"),
        _gi("i2", name="b", type="string"),
        _go("o1", name="y"),
    ]
    contract = derive_contract(nodes)
    _, errors = inject_inputs(nodes, contract, {"a": "not-a-number", "typo": 1})
    reasons = {e["input"]: e["reason"] for e in errors}
    assert set(reasons) == {"a", "typo", "b"}
    assert reasons["typo"] == "unknown input name"
    assert reasons["b"] == "missing required input"
    assert "expected number" in reasons["a"]


def test_inject_inputs_is_case_sensitive_exact_match():
    nodes = [_gi("i1", name="x"), _go("o1", name="y")]
    contract = derive_contract(nodes)
    _, errors = inject_inputs(nodes, contract, {"X": "hi"})
    reasons = {e["input"]: e["reason"] for e in errors}
    assert reasons == {"X": "unknown input name", "x": "missing required input"}


def test_inject_inputs_optional_omitted_not_injected():
    nodes = [_gi("i1", name="x", required=False, default="fallback"), _go("o1", name="y")]
    contract = derive_contract(nodes)
    patched, errors = inject_inputs(nodes, contract, {})
    assert errors == []
    # No injection: the node's execute() falls back to `default`, identical
    # to a canvas run.
    assert "value" not in patched[0]["data"]["params"]


def test_collect_outputs_reads_value_ports_and_reports_missing():
    nodes = [_go("o1", name="y1"), _go("o2", name="y2")]
    contract = derive_contract(nodes)
    outputs, missing = collect_outputs(contract, {"o1": {"value": 42}})
    assert outputs == {"y1": 42}
    assert missing == ["y2"]


def test_collect_outputs_none_value_is_present_not_missing():
    nodes = [_go("o1", name="y")]
    contract = derive_contract(nodes)
    outputs, missing = collect_outputs(contract, {"o1": {"value": None}})
    assert outputs == {"y": None}
    assert missing == []


# ── serialize_output ─────────────────────────────────────────────────────

from app.core.api_contract import (  # noqa: E402
    MAX_TENSOR_ELEMENTS,
    OutputSerializationError,
    serialize_output,
)


def test_serialize_primitives_pass_through():
    for val in (None, True, 3, 2.5, "s"):
        assert serialize_output(val) == val


def test_serialize_containers_recurse_tuple_becomes_list():
    assert serialize_output({"a": (1, 2), "b": [3.5, "x"]}) == {
        "a": [1, 2], "b": [3.5, "x"],
    }


def test_serialize_numpy_scalar_item():
    import numpy as np

    assert serialize_output(np.float32(2.5)) == 2.5
    assert isinstance(serialize_output(np.int64(7)), int)


def test_serialize_ndarray_tagged():
    import numpy as np

    out = serialize_output(np.zeros((2, 3), dtype=np.float32))
    assert out["__type__"] == "tensor"
    assert out["shape"] == [2, 3]
    assert out["dtype"] == "float32"
    assert out["values"] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_serialize_tensor_tagged_and_zero_dim():
    import torch

    out = serialize_output(torch.tensor([[1.0, 2.0]]))
    assert out == {
        "__type__": "tensor", "shape": [1, 2],
        "dtype": "torch.float32", "values": [[1.0, 2.0]],
    }
    # 0-dim tensors serialize with shape [] rather than unwrapping.
    zero_dim = serialize_output(torch.tensor(5.0))
    assert zero_dim["shape"] == []
    assert zero_dim["values"] == 5.0


def test_serialize_tensor_cap_65536():
    import torch

    assert MAX_TENSOR_ELEMENTS == 65536
    ok = serialize_output(torch.zeros(65536))
    assert len(ok["values"]) == 65536
    with pytest.raises(OutputSerializationError) as exc_info:
        serialize_output(torch.zeros(65537))
    assert exc_info.value.code == "output_too_large"
    assert "record_outputs" in exc_info.value.reason


def test_serialize_ndarray_cap_65536():
    import numpy as np

    with pytest.raises(OutputSerializationError) as exc_info:
        serialize_output(np.zeros(65537))
    assert exc_info.value.code == "output_too_large"


def test_serialize_module_rejected_with_modelsaver_hint():
    import torch

    with pytest.raises(OutputSerializationError) as exc_info:
        serialize_output(torch.nn.Linear(2, 2))
    assert exc_info.value.code == "unserializable_output"
    assert "ModelSaver" in exc_info.value.reason


def test_serialize_pil_image_base64_roundtrip():
    import base64
    import io

    from PIL import Image

    out = serialize_output(Image.new("RGB", (4, 2), color=(0, 0, 255)))
    assert out["__type__"] == "image"
    assert out["format"] == "png"
    round_tripped = Image.open(io.BytesIO(base64.b64decode(out["base64"])))
    assert round_tripped.size == (4, 2)


def test_serialize_unknown_type_rejected_with_type_name():
    class Widget:
        pass

    with pytest.raises(OutputSerializationError) as exc_info:
        serialize_output(Widget())
    assert exc_info.value.code == "unserializable_output"
    assert "Widget" in exc_info.value.reason


def test_serialize_base64_plot_string_passes_through():
    # Base64-string plots (e.g. Visualize node output) are plain strings.
    assert serialize_output("iVBORw0KGgo=") == "iVBORw0KGgo="


def test_image_base64_roundtrip_to_tensor():
    # Input side of the same story: base64 -> coerce_input -> tensor.
    import torch

    tensor = coerce_input(_tiny_png_base64(), "image")
    assert tensor.shape == (3, 2, 4)
    assert tensor.dtype == torch.float32


def test_serialize_numpy_builtin_subclass_scalars_normalize():
    import numpy as np

    out = serialize_output(np.float64(2.5))
    assert out == 2.5 and type(out) is float
    s = serialize_output(np.str_("hi"))
    assert s == "hi" and type(s) is str
    b = serialize_output(np.bool_(True))
    assert b is True


def test_serialize_numpy_complex_hits_catch_all():
    import numpy as np

    with pytest.raises(OutputSerializationError):
        serialize_output(np.complex128(1 + 2j))


def test_serialize_ndarray_zero_dim_keeps_empty_shape():
    import numpy as np

    tagged = serialize_output(np.array(7.5))
    assert tagged["__type__"] == "tensor"
    assert tagged["shape"] == []
    assert tagged["values"] == 7.5
