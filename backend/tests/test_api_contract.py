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


def test_image_decompression_bomb_becomes_input_coercion_error(monkeypatch):
    # PIL.Image.DecompressionBombError subclasses Exception directly (not
    # OSError), so it must still be caught and enveloped — never escape as
    # a raw, unenveloped exception through the API layer.
    import PIL.Image

    def _boom(*_args, **_kwargs):
        raise PIL.Image.DecompressionBombError("boom")

    monkeypatch.setattr(PIL.Image, "open", _boom)
    with pytest.raises(InputCoercionError, match="does not decode to an image") as exc_info:
        coerce_input(_tiny_png_base64(), "image")
    assert not isinstance(exc_info.value, PIL.Image.DecompressionBombError)


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


def test_derive_contract_unknown_type_required_is_problem():
    # A hand-edited graph with a bogus `type` must 409 (fix the graph) at
    # GET /contract time, not surface as a per-caller 422 at run time.
    contract = derive_contract([
        _gi("i1", name="n", type="tensor", required=True),
        _go("o1", name="y"),
    ])
    assert any("unknown type" in p and "tensor" in p for p in contract.problems)


def test_derive_contract_unknown_type_optional_is_problem():
    contract = derive_contract([
        _gi("i1", name="n", type="tensor", required=False, default="x"),
        _go("o1", name="y"),
    ])
    assert any("unknown type" in p and "tensor" in p for p in contract.problems)


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


# ── node_status output contract (#117) ───────────────────────────────────
# The node_status wire shape is a contract in its own right: nodes DECLARE
# what a port carries (PortDefinition.media) and the run's progress bridge
# turns declared outputs into typed `{"output_kind": K, K: payload}`
# entries. Nothing in this path is allowed to sniff a value's length or
# character class. (Lived in api/ws_execution until #121 made the socket a
# view; the builders belong to the RUN, not to one transport.)

from app.core.output_entries import (  # noqa: E402
    OUTPUT_KIND_IMAGE,
    OUTPUT_KIND_PROGRESS,
    OUTPUT_KIND_TENSOR_SUMMARY,
    OUTPUT_KIND_TEXT,
    OUTPUT_KIND_CHART,
    build_node_output_entries,
    declared_media_ports,
)
from app.core.node_base import (  # noqa: E402
    MEDIA_CHART,
    MEDIA_IMAGE,
    BaseNode,
    DataType,
    PortDefinition,
)


class _PlotLikeNode(BaseNode):
    """Declares one image port and one plain-string port."""

    NODE_NAME = "_ContractPlotLike"
    CATEGORY = "Test"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="image", data_type=DataType.STRING, media=MEDIA_IMAGE),
            PortDefinition(name="caption", data_type=DataType.STRING),
        ]

    def execute(self, inputs, params, progress_callback=None, *, context=None):
        return {"image": "", "caption": ""}


def _kinds(entries: list[dict]) -> list[str]:
    return [e["output_kind"] for e in entries]


@pytest.fixture
def plot_like_node_type(monkeypatch, registry_with_nodes) -> str:
    """Register _PlotLikeNode for one test only.

    monkeypatch.setitem restores the registry afterwards so the synthetic
    node never shows up in `GET /api/nodes` assertions elsewhere.
    """
    monkeypatch.setitem(registry_with_nodes._nodes, _PlotLikeNode.NODE_NAME, _PlotLikeNode)
    return _PlotLikeNode.NODE_NAME


def test_port_definition_media_defaults_to_none():
    # Ports are plain data unless a node explicitly opts in.
    assert PortDefinition(name="x", data_type=DataType.STRING).media is None
    assert (
        PortDefinition(name="x", data_type=DataType.STRING, media=MEDIA_IMAGE).media
        == "image"
    )


def test_declared_media_ports_lists_only_declared_ports(plot_like_node_type):
    ports = declared_media_ports([
        {"id": "a", "type": plot_like_node_type, "data": {"params": {}}},
        {"id": "b", "type": "Print", "data": {"params": {}}},
    ])
    # Only the node that declares media is listed, and only that port.
    assert ports == {"a": {MEDIA_IMAGE: ["image"]}}


def test_declared_media_ports_ignores_unknown_and_malformed_nodes():
    ports = declared_media_ports([
        {"id": "x", "type": "NoSuchNodeType"},
        {"id": "y"},              # no type
        {"type": "Print"},        # no id
        "not-a-dict",
        {"id": "z", "type": "preset:Whatever"},
    ])
    assert ports == {}


def test_declared_media_ports_keys_on_whatever_kind_the_port_declared(
    monkeypatch, registry_with_nodes,
):
    """The open-vocabulary promise, exercised by a kind core does not know.

    #117 documented that a node pack could add its own kind without a core
    release, but the resolver only ever looked for MEDIA_IMAGE — so until #130
    a new kind never reached the wire. Nothing here mentions "image" or
    "chart": a port declaring "waveform" must arrive as a waveform entry.
    """
    class _MultiMediaNode(BaseNode):
        NODE_NAME = "_ContractMultiMedia"
        CATEGORY = "Test"

        @classmethod
        def define_inputs(cls) -> list[PortDefinition]:
            return []

        @classmethod
        def define_outputs(cls) -> list[PortDefinition]:
            return [
                PortDefinition(name="pic", data_type=DataType.STRING, media=MEDIA_IMAGE),
                PortDefinition(name="plot", data_type=DataType.ANY, media=MEDIA_CHART),
                PortDefinition(name="sound", data_type=DataType.ANY, media="waveform"),
                PortDefinition(name="plain", data_type=DataType.ANY),
            ]

        def execute(self, inputs, params, progress_callback=None, *, context=None):
            return {}

    monkeypatch.setitem(
        registry_with_nodes._nodes, _MultiMediaNode.NODE_NAME, _MultiMediaNode
    )
    declared = declared_media_ports(
        [{"id": "n", "type": _MultiMediaNode.NODE_NAME, "data": {"params": {}}}]
    )
    assert declared == {"n": {
        MEDIA_IMAGE: ["pic"], MEDIA_CHART: ["plot"], "waveform": ["sound"],
    }}

    entries = build_node_output_entries(
        "completed",
        {"sound": {"hz": 440}, "plain": {"not": "media"}},
        declared["n"],
    )
    waveform = next(e for e in entries if e["output_kind"] == "waveform")
    assert waveform == {"output_kind": "waveform", "port": "sound", "waveform": {"hz": 440}}
    assert "plain" not in _kinds(entries)


def test_build_output_entries_declared_chart_port_ships_the_spec_verbatim():
    spec = {"kind": "heatmap", "matrix": [[0.0, 1.0], [1.0, 0.0]], "title": "cm"}
    entries = build_node_output_entries(
        "completed", {"chart": spec}, {MEDIA_CHART: ["chart"]}
    )
    assert _kinds(entries) == [OUTPUT_KIND_CHART, OUTPUT_KIND_TENSOR_SUMMARY]
    chart = entries[0]
    assert chart["port"] == "chart"
    # Unlike an image, a chart spec needs no container — it IS the payload.
    assert chart[OUTPUT_KIND_CHART] == spec


@pytest.mark.parametrize("value", [None, "", {}, "a string", 42, [1, 2]])
def test_build_output_entries_chart_port_without_a_spec_is_skipped(value):
    entries = build_node_output_entries(
        "completed", {"chart": value}, {MEDIA_CHART: ["chart"]}
    )
    assert _kinds(entries) == [OUTPUT_KIND_TENSOR_SUMMARY]


@pytest.mark.parametrize("reserved", ["output_kind", "port", "elided", "bytes", "cap_bytes"])
def test_build_output_entries_refuses_a_kind_that_collides_with_a_reserved_key(reserved):
    """A payload is stored under a key NAMED BY THE KIND.

    So ``media="port"`` would overwrite the port name and ``media="elided"``
    would forge the marker run_service writes when a payload exceeds the size
    cap — making an intact entry read as truncated. Refused, not renamed: a
    collision is a bug in the declaring pack and must surface to its author.
    """
    entries = build_node_output_entries(
        "completed", {"p": {"a": 1}}, {reserved: ["p"]},
    )
    assert _kinds(entries) == [OUTPUT_KIND_TENSOR_SUMMARY]


def test_build_output_entries_rejects_the_pre_130_sequence_signature():
    """A bare list was the old third argument.

    Left unguarded it iterates as media kinds (and a string iterates as
    characters), so an out-of-repo caller would get plausible nonsense rather
    than an error.
    """
    for legacy in (["image"], "image", ("image",)):
        with pytest.raises(TypeError, match="mapping of media kind"):
            build_node_output_entries("completed", {"image": "x"}, legacy)


def test_build_output_entries_emits_one_entry_per_declared_kind():
    entries = build_node_output_entries(
        "completed",
        {"image": _tiny_png_base64(), "chart": {"kind": "bar", "bars": []},
         "__log__": "done"},
        {MEDIA_IMAGE: ["image"], MEDIA_CHART: ["chart"]},
    )
    assert _kinds(entries) == [
        OUTPUT_KIND_TEXT,
        OUTPUT_KIND_IMAGE,
        OUTPUT_KIND_CHART,
        OUTPUT_KIND_TENSOR_SUMMARY,
    ]


def test_build_output_entries_progress_carries_the_event():
    event = {"event": "epoch", "epoch": 1, "total_epochs": 3, "loss": 0.5}
    entries = build_node_output_entries("progress", event)
    assert entries == [{"output_kind": OUTPUT_KIND_PROGRESS, "progress": event}]


def test_build_output_entries_text_comes_from_the_log_channel():
    entries = build_node_output_entries("completed", {"value": 1, "__log__": "hi"})
    assert _kinds(entries) == [OUTPUT_KIND_TEXT, OUTPUT_KIND_TENSOR_SUMMARY]
    assert entries[0]["text"] == "hi"
    # Dunder keys never leak into the summary.
    assert set(entries[1][OUTPUT_KIND_TENSOR_SUMMARY]) == {"value"}


def test_build_output_entries_declared_image_port_produces_an_image_entry():
    b64 = _tiny_png_base64()
    entries = build_node_output_entries("completed", {"image": b64}, {MEDIA_IMAGE: ["image"]})
    assert _kinds(entries) == [OUTPUT_KIND_IMAGE, OUTPUT_KIND_TENSOR_SUMMARY]
    img = entries[0]
    assert img["port"] == "image"
    assert img["image"] == {"format": "png", "encoding": "base64", "data": b64}


@pytest.mark.parametrize(
    "long_text",
    [
        # A token-id / hash dump: plain ASCII alphanumerics.
        ("TokenIds0123456789abcdef" * 21)[:500],
        # Traditional Chinese prose from an LLM node: no spaces at all, and
        # ``str.isalnum()`` is True for CJK, so this always tripped the sniff.
        ("注意力機制讓模型在每一步都能回頭看整個序列" * 25)[:500],
    ],
    ids=["ascii-token-dump", "cjk-prose"],
)
def test_build_output_entries_long_alphanumeric_text_is_never_an_image(long_text):
    """Headline regression for #117.

    A 500-char alphanumeric string (an LLM node's answer, a tokenizer dump)
    used to trip ``len(val) > 200 and val[:20].isalnum()`` and be shipped to
    the browser as a base64 PNG, rendering as a broken image. With declared
    media there is no port to attach it to, so it stays a string output.
    """
    # Exactly the shape the pre-#117 sniff claimed was a base64 PNG.
    assert len(long_text) == 500 and long_text[:20].isalnum()

    entries = build_node_output_entries("completed", {"answer": long_text})
    assert OUTPUT_KIND_IMAGE not in _kinds(entries)
    summary = entries[0][OUTPUT_KIND_TENSOR_SUMMARY]
    assert summary["answer"]["type"] == "string"


def test_build_output_entries_undeclared_port_never_becomes_an_image():
    # Even a genuinely base64-looking value stays plain data without a
    # declaration — declaring is the node's job, not the transport's.
    entries = build_node_output_entries("completed", {"blob": _tiny_png_base64()})
    assert _kinds(entries) == [OUTPUT_KIND_TENSOR_SUMMARY]


def test_build_output_entries_declared_port_with_no_value_is_skipped():
    entries = build_node_output_entries("completed", {"image": ""}, {MEDIA_IMAGE: ["image"]})
    assert _kinds(entries) == [OUTPUT_KIND_TENSOR_SUMMARY]
    entries = build_node_output_entries("completed", {"other": "x"}, {MEDIA_IMAGE: ["image"]})
    assert _kinds(entries) == [OUTPUT_KIND_TENSOR_SUMMARY]
    entries = build_node_output_entries("completed", {"image": 42}, {MEDIA_IMAGE: ["image"]})
    assert _kinds(entries) == [OUTPUT_KIND_TENSOR_SUMMARY]


def test_build_output_entries_cached_status_matches_completed():
    result = {"image": _tiny_png_base64(), "__log__": "from cache"}
    assert _kinds(build_node_output_entries("cached", result, {MEDIA_IMAGE: ["image"]})) == _kinds(
        build_node_output_entries("completed", result, {MEDIA_IMAGE: ["image"]})
    )


def test_build_output_entries_non_terminal_statuses_are_empty():
    for status in ("running", "skipped", "error"):
        assert build_node_output_entries(status, {"value": 1}, {MEDIA_IMAGE: ["image"]}) == []
    assert build_node_output_entries("completed", None, {MEDIA_IMAGE: ["image"]}) == []
    assert build_node_output_entries("progress", None) == []


def test_every_output_entry_stores_its_payload_under_the_matching_key():
    result = {"image": _tiny_png_base64(), "__log__": "hi", "value": 1}
    entries = build_node_output_entries("completed", result, {MEDIA_IMAGE: ["image"]})
    entries += build_node_output_entries("progress", {"event": "epoch"})
    for entry in entries:
        assert entry["output_kind"] in {
            OUTPUT_KIND_TEXT,
            OUTPUT_KIND_IMAGE,
            OUTPUT_KIND_CHART,
            OUTPUT_KIND_PROGRESS,
            OUTPUT_KIND_TENSOR_SUMMARY,
        }
        assert entry["output_kind"] in entry


# ── port summaries: the per-element cut a LIST port gets ─────────────────
#
# A STRING port has been cut at 200 characters since #117. A LIST[str] port
# was not, so `DocumentLoader.texts` put whole documents into every
# node_status frame -- 26 KB for the five bundled RAG samples, megabytes for
# a real folder, at which point the whole entry is elided against the 128 KB
# cap and the learner sees nothing at all.


def test_summarize_cuts_each_list_element_like_a_string_port():
    from app.core.output_entries import _STRING_PREVIEW_CHARS, _summarize_single

    out = _summarize_single(["x" * 5000, "short"])

    assert len(out["values"][0]) == _STRING_PREVIEW_CHARS == 200
    assert out["values"][1] == "short"
    # The LENGTH is the port's, not the preview's: a learner reading "2" has
    # to be reading how many documents there are.
    assert out["length"] == 2


def test_summarize_leaves_a_non_string_list_alone():
    from app.core.output_entries import _summarize_single

    assert _summarize_single([1, 2, 3])["values"] == [1, 2, 3]
    assert _summarize_single([[3, 7]])["values"] == [[3, 7]]
