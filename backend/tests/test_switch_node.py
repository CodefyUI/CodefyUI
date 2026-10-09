"""Tests for SwitchNode (#655): a MUX picked by its param or its port."""

from __future__ import annotations

import pytest
import torch

from app.core.graph_engine import execute_graph, resolve_bypass, validate_graph
from app.nodes.dataflow.switch_node import (
    DEFAULT_INPUTS,
    MAX_INPUTS,
    SwitchNode,
    coerce_selector,
)


def _run(params=None, **inputs):
    return SwitchNode().execute(inputs, params or {})


# -- The node on its own ------------------------------------------------------


def test_node_metadata():
    assert SwitchNode.NODE_NAME == "Switch"
    assert SwitchNode.CATEGORY == "Data Flow"


def test_the_param_picks_the_input_when_no_selector_is_wired():
    assert _run({"selector": 1}, input_0="a", input_1="b")["output"] == "b"


def test_a_missing_selector_param_picks_input_0():
    assert _run({}, input_0="a", input_1="b")["output"] == "a"


def test_a_wired_selector_overrides_the_param():
    res = _run({"selector": 0}, selector=2, input_0="a", input_1="b", input_2="c")
    assert res["output"] == "c"


def test_a_tensor_and_an_integral_float_are_indices():
    assert _run({}, selector=torch.tensor(1), input_0="a", input_1="b")["output"] == "b"
    assert _run({}, selector=1.0, input_0="a", input_1="b")["output"] == "b"


def test_the_selected_value_passes_through_even_when_it_is_none():
    assert _run({"selector": 1}, input_0="a", input_1=None)["output"] is None


@pytest.mark.parametrize(
    "selector",
    [1.7, float("nan"), True, None, "1", torch.tensor([0, 1])],
    ids=["fraction", "nan", "bool", "none", "string", "two-element-tensor"],
)
def test_a_selector_that_is_not_an_integer_is_an_error(selector):
    with pytest.raises(ValueError, match="selector must be"):
        _run({}, selector=selector, input_0="a", input_1="b")


@pytest.mark.parametrize("selector", [-1, DEFAULT_INPUTS])
def test_an_out_of_range_selector_is_an_error(selector):
    # The old Switch fell back to input_0 here, without a word.
    with pytest.raises(ValueError, match=f"is {selector}, but this Switch has inputs 0 to 3"):
        _run({"selector": selector}, input_0="a")


def test_a_selector_naming_an_unwired_input_is_an_error():
    with pytest.raises(ValueError, match="input_2 has no value"):
        _run({}, selector=2, input_0="a", input_1="b")


def test_the_error_says_where_the_selector_came_from():
    with pytest.raises(ValueError, match="selector param"):
        _run({"selector": 3}, input_0="a")
    with pytest.raises(ValueError, match="selector port"):
        _run({"selector": 0}, selector=3, input_0="a")


def test_coerce_selector_accepts_plain_integers():
    assert coerce_selector(0) == 0
    assert coerce_selector(torch.tensor(3.0)) == 3


# -- Ports ---------------------------------------------------------------------


def _input_names(params):
    return [p.name for p in SwitchNode.define_inputs_dynamic(params)]


def test_a_graph_without_the_inputs_param_keeps_the_four_ports_it_was_wired_to():
    assert _input_names({}) == ["input_0", "input_1", "input_2", "input_3", "selector"]
    assert _input_names(None) == _input_names({})


def test_the_inputs_param_sets_the_port_count_and_clamps():
    assert _input_names({"inputs": 6})[:-1] == [f"input_{i}" for i in range(6)]
    assert len(_input_names({"inputs": 1})) == 2 + 1
    assert len(_input_names({"inputs": 999})) == MAX_INPUTS + 1
    assert len(_input_names({"inputs": "garbage"})) == DEFAULT_INPUTS + 1


def test_every_port_is_optional():
    assert all(p.optional for p in SwitchNode.define_inputs_dynamic({}))


# -- In a graph ----------------------------------------------------------------


def _node(nid, ntype, **params):
    return {"id": nid, "type": ntype, "data": {"params": params}}


def _edge(eid, src, src_handle, tgt, tgt_handle):
    return {
        "id": eid, "source": src, "target": tgt,
        "sourceHandle": src_handle, "targetHandle": tgt_handle, "type": "data",
    }


def _trigger(eid, src, tgt):
    return {"id": eid, "source": src, "target": tgt, "sourceHandle": "trigger", "type": "trigger"}


def _script(nid, value, out_type="TENSOR"):
    """A PythonScript whose one output is *value*, typed *out_type*."""
    return _node(
        nid, "PythonScript",
        code=f"def run(inputs, params):\n    return {{'out1': {value}}}\n",
        input_ports=1, output_ports=1, output_types=out_type,
    )


def _graph(switch_params, *, wire=("a", "b"), selector_source=None):
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        _node("a", "TensorCreate", shape="2", fill="zeros"),
        _node("b", "TensorCreate", shape="2", fill="ones"),
        _node("sw", "Switch", **switch_params),
        _node("out", "Print", label="picked"),
    ]
    edges = [
        _trigger("t1", "start", "a"),
        _trigger("t2", "start", "b"),
        _edge("e3", "sw", "output", "out", "value"),
    ]
    for index, src in enumerate(wire):
        edges.append(_edge(f"in{index}", src, "tensor", "sw", f"input_{index}"))
    if selector_source is not None:
        nodes.append(selector_source)
        edges.append(_trigger("t3", "start", selector_source["id"]))
        edges.append(_edge("sel", selector_source["id"], "out1", "sw", "selector"))
    return nodes, edges


def _codes(errors):
    return [getattr(e, "code", None) for e in errors]


def test_a_param_selector_validates_and_runs_without_a_selector_wire():
    nodes, edges = _graph({"selector": 1})
    assert validate_graph(nodes, edges) == []


async def test_a_param_selector_forwards_the_input_it_names():
    nodes, edges = _graph({"selector": 1})
    results = await execute_graph(nodes, edges)
    assert torch.equal(results["sw"]["output"], torch.ones(2))


async def test_a_wired_selector_overrides_the_param_in_a_run():
    nodes, edges = _graph({"selector": 1}, selector_source=_script("s", 0, "SCALAR"))
    assert validate_graph(nodes, edges) == []
    results = await execute_graph(nodes, edges)
    assert torch.equal(results["sw"]["output"], torch.zeros(2))


def test_a_param_selector_out_of_range_is_refused_before_the_run():
    nodes, edges = _graph({"selector": 4, "inputs": 4})
    assert "switch_selector_out_of_range" in _codes(validate_graph(nodes, edges))


def test_a_param_selector_naming_an_unwired_input_is_refused_before_the_run():
    nodes, edges = _graph({"selector": 2})
    errors = validate_graph(nodes, edges)
    assert "switch_selected_unwired" in _codes(errors)
    issue = next(e for e in errors if getattr(e, "code", None) == "switch_selected_unwired")
    assert issue.params["port"] == "input_2"


def test_a_wired_selector_is_left_to_the_run():
    # Its value is only known during the run, which checks it then.
    nodes, edges = _graph({"selector": 9}, selector_source=_script("s", 0, "SCALAR"))
    assert validate_graph(nodes, edges) == []


def test_inputs_of_different_types_are_refused():
    nodes, edges = _graph({"selector": 0}, wire=("a",))
    nodes.append(_script("text", "'hello'", "STRING"))
    edges.append(_trigger("t4", "start", "text"))
    edges.append(_edge("in1", "text", "out1", "sw", "input_1"))
    errors = validate_graph(nodes, edges)
    assert "switch_input_types_differ" in _codes(errors)
    issue = next(e for e in errors if getattr(e, "code", None) == "switch_input_types_differ")
    assert issue.params["types"] == "STRING, TENSOR"


def test_the_output_carries_the_input_type_downstream():
    # A TENSOR Switch cannot feed a STRING port, as a TENSOR wire cannot.
    nodes, edges = _graph({"selector": 0})
    nodes.append(_node("tok", "PythonScript",
                       code="def run(inputs, params):\n    return {'out1': 1}\n",
                       input_ports=1, input_types="STRING", output_ports=1))
    edges.append(_edge("e9", "sw", "output", "tok", "in1"))
    errors = validate_graph(nodes, edges)
    mismatch = [e for e in errors if getattr(e, "code", None) == "type_mismatch"]
    assert mismatch and mismatch[0].params["source_type"] == "TENSOR"


def test_a_switch_fed_by_a_switch_takes_its_type():
    nodes, edges = _graph({"selector": 0})
    nodes.append(_node("sw2", "Switch", selector=0))
    nodes.append(_node("tok", "PythonScript",
                       code="def run(inputs, params):\n    return {'out1': 1}\n",
                       input_ports=1, input_types="STRING", output_ports=1))
    edges.append(_edge("e10", "sw", "output", "sw2", "input_0"))
    edges.append(_edge("e11", "sw2", "output", "tok", "in1"))
    mismatch = [e for e in validate_graph(nodes, edges)
                if getattr(e, "code", None) == "type_mismatch"]
    assert [m.params["source"] for m in mismatch] == ["sw2"]


def test_a_bypassed_switch_forwards_input_0():
    nodes, edges = _graph({"selector": 1})
    nodes[3]["data"]["bypassed"] = True
    resolution = resolve_bypass(nodes, edges)
    assert resolution.errors == []
    assert [(link.input, link.source) for link in resolution.links] == [("input_0", "a")]


# -- Export Python runs the same Switch -----------------------------------------


def _run_script_here(script: str) -> dict:
    """The exported file's ``run_graph``, in this process (as in
    ``test_codegen_container_triggers``): the session's registry is already
    discovered, so the file's own runtime reset is skipped."""
    from app.core.execution_context import ExecutionContext

    module: dict = {"__name__": "switch_exported_graph"}
    exec(compile(script, "<exported graph>", "exec"), module)  # noqa: S102
    module["_RT"] = module["_load_runtime"](None)
    context = ExecutionContext(device="cpu", weights_persistent=False, graph_id="switch-script")
    return module["run_graph"](context, {})


@pytest.mark.parametrize(
    "switch_params,selector_source,expected",
    [
        ({"selector": 1}, None, torch.ones(2)),
        ({"selector": 1}, _script("s", 0, "SCALAR"), torch.zeros(2)),
    ],
    ids=["param", "port overrides param"],
)
async def test_export_python_forwards_what_the_run_forwards(
    test_client, switch_params, selector_source, expected,
):
    nodes, edges = _graph(switch_params, selector_source=selector_source)
    engine = await execute_graph(nodes, edges)
    assert torch.equal(engine["sw"]["output"], expected)

    response = await test_client.post(
        "/api/graph/export", json={"name": "switch", "nodes": nodes, "edges": edges},
    )
    assert response.status_code == 200, response.text
    script = _run_script_here(response.json()["script"])
    assert torch.equal(script["sw"]["output"], expected)


async def test_export_refuses_a_param_selector_naming_an_unwired_input(test_client):
    nodes, edges = _graph({"selector": 3})
    response = await test_client.post(
        "/api/graph/export", json={"name": "switch", "nodes": nodes, "edges": edges},
    )
    assert response.status_code != 200
    assert "input_3 is not connected" in response.text
