"""A data input takes one wire (#562, #658).

Several wires into one input used to merge by edge order: the run, and the
exported script after it, read the last one. Every door a graph comes
through now refuses that -- validation, a run, Export (``test_codegen``), a
preset saved from it, and a Map running an old preset -- and a trigger,
which is control flow, is left alone.
"""

from __future__ import annotations

import pytest

from app.core.execution_context import ExecutionContext
from app.core.graph_engine import (
    GraphValidationError,
    execute_graph,
    multiple_source_errors,
    validate_graph,
)
from app.core.preset_registry import preset_registry
from app.nodes.dataflow.map_node import MapNode
from app.schemas.models import (
    ExposedPortSchema,
    InternalEdgeSchema,
    InternalNodeSchema,
    PresetDefinition,
)


def _node(nid, ntype, **params):
    return {"id": nid, "type": ntype, "data": {"params": params}}


def _edge(eid, src, src_handle, tgt, tgt_handle, kind="data"):
    return {"id": eid, "source": src, "target": tgt,
            "sourceHandle": src_handle, "targetHandle": tgt_handle, "type": kind}


def _codes(errors):
    return [getattr(e, "code", None) for e in errors]


def test_one_line_per_input_naming_the_node_port_and_sources():
    nodes = [_node("a", "TensorCreate"), _node("b", "TensorCreate"), _node("p", "Print")]
    edges = [_edge("1", "a", "tensor", "p", "value"), _edge("2", "b", "tensor", "p", "value")]
    [issue] = multiple_source_errors(nodes, edges)
    assert issue.code == "multiple_sources"
    assert issue.node_id == "p"
    assert issue.params == {"port": "value", "sources": ["a", "b"], "count": 2}


def test_triggers_and_an_edge_to_a_missing_node_do_not_count():
    nodes = [_node("s1", "Start"), _node("s2", "Start"), _node("p", "Print")]
    edges = [
        _edge("t1", "s1", "trigger", "p", "__trigger", kind="trigger"),
        _edge("t2", "s2", "trigger", "p", "__trigger", kind="trigger"),
        # A trigger saved without its type still names its handles.
        _edge("t3", "s1", "trigger", "p", "", kind="data"),
        _edge("t4", "s2", "trigger", "p", "", kind="data"),
        _edge("g1", "s1", "x", "gone", "value"),
        _edge("g2", "s2", "x", "gone", "value"),
    ]
    assert multiple_source_errors(nodes, edges) == []


def test_one_output_feeding_several_inputs_is_fine():
    nodes = [_node("a", "TensorCreate"), _node("p", "Print"), _node("q", "Print")]
    edges = [_edge("1", "a", "tensor", "p", "value"), _edge("2", "a", "tensor", "q", "value")]
    assert multiple_source_errors(nodes, edges) == []


def _block_with_shared_input():
    """Two outside wires into one block input: they meet at one inner port."""
    block = {
        "id": "blk", "name": "blk",
        "nodes": [_node("p", "Print")],
        "edges": [],
        "interface": {
            "inputs": [{"port": "value", "innerNode": "p", "innerPort": "value"}],
            "outputs": [],
            "triggerTargets": [],
        },
    }
    nodes = [
        _node("start", "Start"),
        _node("a", "TensorCreate", shape="2", fill="ones"),
        _node("b", "TensorCreate", shape="2", fill="zeros"),
        _node("inst", "subgraph:blk"),
    ]
    edges = [
        _edge("t1", "start", "trigger", "a", "", kind="trigger"),
        _edge("t2", "start", "trigger", "b", "", kind="trigger"),
        _edge("1", "a", "tensor", "inst", "value"),
        _edge("2", "b", "tensor", "inst", "value"),
    ]
    return nodes, edges, [block]


def test_validation_finds_several_wires_meeting_inside_a_block():
    nodes, edges, subgraphs = _block_with_shared_input()
    errors = validate_graph(nodes, edges, subgraphs=subgraphs)
    issues = [e for e in errors if getattr(e, "code", None) == "multiple_sources"]
    assert [(i.node_id, i.params["port"]) for i in issues] == [("inst/p", "value")]


async def test_a_run_refuses_several_wires_meeting_inside_a_block():
    nodes, edges, subgraphs = _block_with_shared_input()
    with pytest.raises(GraphValidationError, match="has 2 wires"):
        await execute_graph(nodes, edges, subgraphs=subgraphs)


def test_a_bypassed_node_cannot_hide_its_extra_wire():
    # Bypass would keep one wire into the muted node and drop the other.
    nodes = [
        _node("a", "TensorCreate"), _node("b", "TensorCreate"),
        {"id": "mid", "type": "Print", "data": {"params": {}, "bypassed": True}},
        _node("p", "Print"),
    ]
    edges = [
        _edge("1", "a", "tensor", "mid", "value"),
        _edge("2", "b", "tensor", "mid", "value"),
        _edge("3", "mid", "value", "p", "value"),
    ]
    assert "multiple_sources" in _codes(validate_graph(nodes, edges))


@pytest.fixture()
def _isolated_presets(tmp_path, monkeypatch):
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    monkeypatch.setattr("app.config.settings.USER_PRESETS_DIR", user_dir)
    monkeypatch.setattr("app.config.settings.PRESETS_DIR", tmp_path / "builtin")
    saved = dict(preset_registry._presets)
    try:
        yield user_dir
    finally:
        preset_registry._presets.clear()
        preset_registry._presets.update(saved)


async def test_a_preset_with_several_wires_into_one_input_is_not_saved(
    test_client, _isolated_presets,
):
    response = await test_client.post("/api/presets/create", json={
        "name": "Merged",
        "nodes": [
            {**_node("a", "TensorCreate"), "position": {"x": 0, "y": 0}},
            {**_node("b", "TensorCreate"), "position": {"x": 0, "y": 0}},
            {**_node("p", "Print"), "position": {"x": 0, "y": 0}},
        ],
        "edges": [_edge("1", "a", "tensor", "p", "value"), _edge("2", "b", "tensor", "p", "value")],
    })
    assert response.status_code == 400
    assert "has 2 wires" in response.text
    assert preset_registry.get("Merged") is None


@pytest.fixture()
def merged_body():
    """A preset stored before the rule: two wires into its Print."""
    saved = dict(preset_registry._presets)
    preset_registry._presets["merged-body"] = PresetDefinition(
        preset_name="merged-body", category="Testing", description="",
        nodes=[
            InternalNodeSchema(id="a", type="TensorCreate", params={"shape": "2", "fill": "ones"}),
            InternalNodeSchema(id="p", type="Print", params={}),
        ],
        edges=[
            InternalEdgeSchema(source="a", sourceHandle="tensor", target="p", targetHandle="value"),
        ],
        exposed_inputs=[ExposedPortSchema(name="in", internal_node="p", internal_port="value")],
        exposed_outputs=[ExposedPortSchema(name="out", internal_node="p", internal_port="value")],
        exposed_params=[],
    )
    try:
        yield "merged-body"
    finally:
        preset_registry._presets.clear()
        preset_registry._presets.update(saved)


def test_a_map_refuses_an_old_preset_with_several_wires_into_one_input(merged_body):
    preset = preset_registry.get(merged_body)
    preset.edges.append(
        InternalEdgeSchema(source="a", sourceHandle="tensor", target="p", targetHandle="value"),
    )
    context = ExecutionContext()
    context.current_node_id = "map1"
    with pytest.raises(ValueError, match="Map: preset 'merged-body'.*has 2 wires"):
        MapNode().execute({"items": [1]}, {"subgraph": merged_body}, context=context)
