"""A Switch set by its param skips the branches it does not select (#656).

The run and Export Python both build on ``prepare_executable_graph``, which
leaves those nodes out after validating them; a Map body leaves them out of
every item. A Switch set by its port decides during the run, so every input
still runs.
"""

from __future__ import annotations

import pytest
import torch

from app.core.execution_context import ExecutionContext
from app.core.graph_engine import GraphValidationError, execute_graph, validate_graph
from app.core.node_registry import registry
from app.core.preset_registry import preset_registry
from app.nodes.dataflow.map_node import MapNode
from app.nodes.dataflow.switch_node import unselected_node_ids
from app.schemas.models import (
    ExposedPortSchema,
    InternalEdgeSchema,
    InternalNodeSchema,
    PresetDefinition,
)


def _node(nid, ntype, **params):
    return {"id": nid, "type": ntype, "data": {"params": params}}


def _edge(src, src_handle, tgt, tgt_handle):
    return {
        "id": f"{src}.{src_handle}->{tgt}.{tgt_handle}", "source": src, "target": tgt,
        "sourceHandle": src_handle, "targetHandle": tgt_handle, "type": "data",
    }


def _trigger(src, tgt):
    return {"id": f"t-{tgt}", "source": src, "target": tgt, "sourceHandle": "trigger", "type": "trigger"}


def _tensor(nid, fill):
    return _node(nid, "TensorCreate", shape="2", fill=fill)


def _boom(nid):
    """A node that fails if it runs: its input is a TENSOR, its output too."""
    return _node(
        nid, "PythonScript",
        code="def run(inputs, params):\n    raise ValueError('boom ran')\n",
        input_ports=1, input_types="TENSOR", output_ports=1, output_types="TENSOR",
    )


def _scalar(nid, value):
    return _node(
        nid, "PythonScript",
        code=f"def run(inputs, params):\n    return {{'out1': {value}}}\n",
        input_ports=1, output_ports=1, output_types="SCALAR",
    )


def _branching(switch_params, *, selector=None):
    """a (zeros) -> Switch.input_0; b (ones) -> boom -> Switch.input_1."""
    nodes = [
        _node("start", "Start"),
        _tensor("a", "zeros"), _tensor("b", "ones"), _boom("boom"),
        _node("sw", "Switch", **switch_params),
        _node("out", "Print", label="picked"),
    ]
    edges = [
        _trigger("start", "a"), _trigger("start", "b"),
        _edge("a", "tensor", "sw", "input_0"),
        _edge("b", "tensor", "boom", "in1"),
        _edge("boom", "out1", "sw", "input_1"),
        _edge("sw", "output", "out", "value"),
    ]
    if selector is not None:
        nodes.append(_scalar("sel", selector))
        edges += [_trigger("start", "sel"), _edge("sel", "out1", "sw", "selector")]
    return nodes, edges


async def _run(nodes, edges, **kwargs):
    statuses: list[tuple[str, str]] = []

    async def on_progress(node_id, status, data):
        statuses.append((node_id, status))

    outputs = await execute_graph(nodes, edges, on_progress=on_progress, **kwargs)
    return outputs, statuses


# -- Which nodes are left out ----------------------------------------------------


def test_the_nodes_only_feeding_an_unselected_input_are_left_out():
    nodes, edges = _branching({"selector": 0})
    assert unselected_node_ids(nodes, edges, registry) == {"b", "boom"}


def test_nothing_is_left_out_when_the_selector_is_wired_or_invalid():
    nodes, edges = _branching({"selector": 0}, selector=0)
    assert unselected_node_ids(nodes, edges, registry) == set()
    nodes, edges = _branching({"selector": "x"})
    assert unselected_node_ids(nodes, edges, registry) == set()


def test_a_node_that_also_feeds_something_else_runs():
    nodes, edges = _branching({"selector": 0})
    nodes.append(_node("peek", "Print", label="peek"))
    edges.append(_edge("b", "tensor", "peek", "value"))
    assert unselected_node_ids(nodes, edges, registry) == {"boom"}


def test_a_node_feeding_the_selected_input_too_runs():
    nodes, edges = _branching({"selector": 0})
    edges.append(_edge("b", "tensor", "sw", "input_2"))
    nodes[4]["data"]["params"]["selector"] = 2
    assert unselected_node_ids(nodes, edges, registry) == {"a", "boom"}


def test_a_switch_in_an_unselected_branch_takes_its_branches_with_it():
    nodes = [
        _tensor("a", "zeros"), _tensor("b", "ones"), _tensor("c", "ones"),
        _node("inner", "Switch", selector=0), _node("outer", "Switch", selector=0),
        _node("out", "Print"),
    ]
    edges = [
        _edge("a", "tensor", "outer", "input_0"),
        _edge("b", "tensor", "inner", "input_0"),
        _edge("c", "tensor", "inner", "input_1"),
        _edge("inner", "output", "outer", "input_1"),
        _edge("outer", "output", "out", "value"),
    ]
    assert unselected_node_ids(nodes, edges, registry) == {"b", "c", "inner"}


# -- A run --------------------------------------------------------------------------


async def test_a_param_selector_does_not_run_the_branch_it_does_not_pick():
    nodes, edges = _branching({"selector": 0})
    outputs, statuses = await _run(nodes, edges)
    assert torch.equal(outputs["sw"]["output"], torch.zeros(2))
    assert "boom" not in outputs and "b" not in outputs
    assert ("boom", "unselected") in statuses and ("b", "unselected") in statuses
    # Said before anything runs, so no view waits on them.
    first_running = statuses.index(next(s for s in statuses if s[1] == "running"))
    assert statuses.index(("boom", "unselected")) < first_running


async def test_a_port_selector_runs_every_input():
    nodes, edges = _branching({"selector": 1}, selector=0)
    with pytest.raises(Exception, match="boom ran"):
        await execute_graph(nodes, edges)


async def test_an_unselected_branch_is_still_validated():
    nodes, edges = _branching({"selector": 0})
    # A STRING into boom's TENSOR input: wrong in a branch that will not run.
    nodes.append(_node("text", "PythonScript",
                       code="def run(inputs, params):\n    return {'out1': 'x'}\n",
                       input_ports=1, output_ports=1, output_types="STRING"))
    edges = [e for e in edges if e["target"] != "boom"]
    edges += [_trigger("start", "text"), _edge("text", "out1", "boom", "in1")]
    with pytest.raises(GraphValidationError, match="Type mismatch"):
        await execute_graph(nodes, edges)
    assert any(getattr(e, "code", None) == "type_mismatch" for e in validate_graph(nodes, edges))


async def test_a_block_with_nothing_selected_inside_says_unselected():
    block = {
        "id": "pair", "name": "pair",
        "nodes": [_tensor("x", "ones"), _boom("y")],
        "edges": [_edge("x", "tensor", "y", "in1")],
        "interface": {
            "inputs": [],
            "outputs": [{"port": "out", "innerNode": "y", "innerPort": "out1"}],
            "triggerTargets": [],
        },
    }
    nodes = [
        _node("start", "Start"), _tensor("a", "zeros"), _node("blk", "subgraph:pair"),
        _node("sw", "Switch", selector=0), _node("out", "Print"),
    ]
    edges = [
        _trigger("start", "a"), _trigger("start", "blk"),
        _edge("a", "tensor", "sw", "input_0"),
        _edge("blk", "out", "sw", "input_1"),
        _edge("sw", "output", "out", "value"),
    ]
    outputs, statuses = await _run(nodes, edges, subgraphs=[block])
    assert torch.equal(outputs["sw"]["output"], torch.zeros(2))
    assert ("blk", "unselected") in statuses


async def test_a_switch_inside_a_block_skips_its_unselected_branch():
    nodes, edges = _branching({"selector": 0})
    inner_nodes = [n for n in nodes if n["id"] != "start"]
    inner_edges = [e for e in edges if e["type"] == "data"]
    block = {
        "id": "branchy", "name": "branchy", "nodes": inner_nodes, "edges": inner_edges,
        "interface": {
            "inputs": [],
            "outputs": [{"port": "out", "innerNode": "sw", "innerPort": "output"}],
            "triggerTargets": ["a", "b"],
        },
    }
    outer_nodes = [_node("start", "Start"), _node("blk", "subgraph:branchy")]
    outer_edges = [_trigger("start", "blk")]
    outputs, _ = await _run(outer_nodes, outer_edges, subgraphs=[block])
    assert torch.equal(outputs["blk/sw"]["output"], torch.zeros(2))
    assert "blk/boom" not in outputs


# -- A Map body ---------------------------------------------------------------------


@pytest.fixture()
def switch_body():
    """A preset: item -> Switch.input_0; a failing node -> Switch.input_1."""
    saved = dict(preset_registry._presets)
    preset_registry._presets["switch-body"] = PresetDefinition(
        preset_name="switch-body", category="Testing", description="",
        nodes=[
            InternalNodeSchema(id="b", type="TensorCreate", params={"shape": "2", "fill": "ones"}),
            InternalNodeSchema(id="boom", type="PythonScript", params=_boom("boom")["data"]["params"]),
            InternalNodeSchema(id="sw", type="Switch", params={"selector": 0}),
        ],
        edges=[
            InternalEdgeSchema(source="b", sourceHandle="tensor", target="boom", targetHandle="in1"),
            InternalEdgeSchema(source="boom", sourceHandle="out1", target="sw", targetHandle="input_1"),
        ],
        exposed_inputs=[ExposedPortSchema(name="in", internal_node="sw", internal_port="input_0")],
        exposed_outputs=[ExposedPortSchema(name="out", internal_node="sw", internal_port="output")],
        exposed_params=[],
    )
    try:
        yield "switch-body"
    finally:
        preset_registry._presets.clear()
        preset_registry._presets.update(saved)


def test_a_map_body_skips_the_branch_its_switch_does_not_pick(switch_body):
    context = ExecutionContext()
    context.current_node_id = "map1"
    result = MapNode().execute({"items": ["a", "b"]}, {"subgraph": switch_body}, context=context)
    assert result["results"] == ["a", "b"]


# -- Export Python --------------------------------------------------------------------


async def test_export_python_leaves_out_what_the_run_leaves_out(test_client):
    nodes, edges = _branching({"selector": 0})
    response = await test_client.post(
        "/api/graph/export", json={"name": "prune", "nodes": nodes, "edges": edges},
    )
    assert response.status_code == 200, response.text
    script = response.json()["script"]
    assert "'boom'" not in script and "'b'" not in script

    module: dict = {"__name__": "prune_exported_graph"}
    exec(compile(script, "<exported graph>", "exec"), module)  # noqa: S102
    module["_RT"] = module["_load_runtime"](None)
    ran = module["run_graph"](ExecutionContext(device="cpu", weights_persistent=False), {})
    assert torch.equal(ran["sw"]["output"], torch.zeros(2))
