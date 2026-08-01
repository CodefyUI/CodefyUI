"""Tests for the graph engine: topological sort, validation, execution."""

import asyncio

import pytest

from app.core.graph_engine import (
    GraphValidationError,
    execute_graph,
    topological_levels,
    topological_sort,
    validate_graph,
)


def test_topological_sort_linear():
    nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    edges = [
        {"source": "a", "target": "b"},
        {"source": "b", "target": "c"},
    ]
    order = topological_sort(nodes, edges)
    assert order == ["a", "b", "c"]


def test_topological_sort_diamond():
    nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}]
    edges = [
        {"source": "a", "target": "b"},
        {"source": "a", "target": "c"},
        {"source": "b", "target": "d"},
        {"source": "c", "target": "d"},
    ]
    order = topological_sort(nodes, edges)
    assert order[0] == "a"
    assert order[-1] == "d"


def test_cycle_detection():
    nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    edges = [
        {"source": "a", "target": "b"},
        {"source": "b", "target": "c"},
        {"source": "c", "target": "a"},
    ]
    with pytest.raises(GraphValidationError):
        topological_sort(nodes, edges)


def _start_node(nid="start"):
    return {"id": nid, "type": "Start", "data": {"params": {}}}


def _trigger(eid, src, tgt):
    return {"id": eid, "source": src, "target": tgt, "sourceHandle": "trigger", "type": "trigger"}


def test_validate_graph_valid():
    """_TestSource has no required inputs, Print's required input is satisfied by the edge."""
    nodes = [
        _start_node(),
        {"id": "1", "type": "_TestSource", "data": {"params": {}}},
        {"id": "2", "type": "Print", "data": {"params": {}}},
    ]
    edges = [
        _trigger("et", "start", "1"),
        {"source": "1", "target": "2", "sourceHandle": "value", "targetHandle": "value"},
    ]
    errors = validate_graph(nodes, edges)
    assert errors == [], f"Unexpected errors: {errors}"


def test_validate_graph_unknown_node():
    nodes = [
        _start_node(),
        {"id": "1", "type": "NonExistentNode", "data": {}},
    ]
    edges = [_trigger("et", "start", "1")]
    errors = validate_graph(nodes, edges)
    assert any("Unknown node type" in e and "NonExistentNode" in e and "node 1" in e for e in errors)


def test_validate_graph_type_mismatch():
    nodes = [
        {"id": "1", "type": "Loss"},
        {"id": "2", "type": "Conv2d"},
    ]
    edges = [
        {"source": "1", "target": "2", "sourceHandle": "loss_fn", "targetHandle": "tensor"},
    ]
    errors = validate_graph(nodes, edges)
    assert len(errors) > 0
    assert "mismatch" in errors[0].lower() or "Type" in errors[0]


@pytest.mark.asyncio
async def test_execute_print_nodes():
    """Use _TestSource (registered in conftest, no torch) to feed Print."""
    nodes = [
        _start_node(),
        {"id": "1", "type": "_TestSource", "data": {"params": {}}},
        {"id": "2", "type": "Print", "data": {"params": {"label": "second"}}},
    ]
    edges = [
        _trigger("et", "start", "1"),
        {"source": "1", "target": "2", "sourceHandle": "value", "targetHandle": "value"},
    ]
    results = await execute_graph(nodes, edges)
    assert "1" in results
    assert "2" in results


def test_validate_graph_missing_required_input():
    """Conv2d has a required 'tensor' input; without an edge it should error."""
    nodes = [
        {"id": "1", "type": "Conv2d", "data": {"params": {}}},
    ]
    edges = []
    errors = validate_graph(nodes, edges)
    assert any("Missing required input" in e and "'tensor'" in e and "node 1" in e for e in errors)


def test_validate_graph_optional_input_no_error():
    """TrainingLoop has optional inputs (val_dataloader, lr_scheduler); leaving them
    unconnected should not produce errors for those ports."""
    nodes = [
        {"id": "1", "type": "TrainingLoop", "data": {"params": {}}},
    ]
    edges = []
    errors = validate_graph(nodes, edges)
    # Should have errors for the required inputs but NOT for the optional ones
    optional_names = {"val_dataloader", "lr_scheduler"}
    for e in errors:
        for name in optional_names:
            assert name not in e, f"Optional input '{name}' should not cause an error"


def test_validate_graph_param_below_min():
    """Dropout 'p' has min_value=0.0; supplying -0.5 should error."""
    nodes = [
        {"id": "1", "type": "Dropout", "data": {"params": {"p": -0.5}}},
    ]
    edges = []
    errors = validate_graph(nodes, edges)
    assert any("below minimum" in e and "'p'" in e for e in errors)


def test_validate_graph_param_above_max():
    """Dropout 'p' has max_value=1.0; supplying 1.5 should error."""
    nodes = [
        {"id": "1", "type": "Dropout", "data": {"params": {"p": 1.5}}},
    ]
    edges = []
    errors = validate_graph(nodes, edges)
    assert any("above maximum" in e and "'p'" in e for e in errors)


def test_validate_graph_param_within_range_no_error():
    """Dropout 'p' within [0.0, 1.0] should not produce a range error."""
    nodes = [
        {"id": "1", "type": "Dropout", "data": {"params": {"p": 0.5}}},
    ]
    edges = []
    errors = validate_graph(nodes, edges)
    range_errors = [e for e in errors if "below minimum" in e or "above maximum" in e]
    assert range_errors == []


def test_validate_graph_multiple_unknown_nodes():
    """Multiple unknown node types should each produce their own error."""
    nodes = [
        {"id": "1", "type": "FakeNodeA"},
        {"id": "2", "type": "FakeNodeB"},
    ]
    edges = []
    errors = validate_graph(nodes, edges)
    assert len([e for e in errors if "Unknown node type" in e]) == 2
    assert any("FakeNodeA" in e for e in errors)
    assert any("FakeNodeB" in e for e in errors)


def test_validate_graph_required_input_satisfied_by_edge():
    """Conv2d's required 'tensor' input connected via an edge should pass the check."""
    nodes = [
        {"id": "1", "type": "Conv2d", "data": {"params": {}}},
        {"id": "2", "type": "Conv2d", "data": {"params": {}}},
    ]
    edges = [
        {"source": "1", "target": "2", "sourceHandle": "tensor", "targetHandle": "tensor"},
    ]
    errors = validate_graph(nodes, edges)
    # Node 1 still has a missing required input, but node 2's 'tensor' is satisfied
    node2_missing = [e for e in errors if "node 2" in e and "Missing required input" in e]
    assert node2_missing == [], f"Node 2's input should be satisfied: {node2_missing}"


from app.core.graph_engine import find_entry_points, reachable_from_entry_points


def test_execute_graph_untriggered_producer_fails_validation():
    """Every path must start from a Start node. A pure producer (no inputs,
    no trigger) that feeds a required input is NOT auto-included — it gets
    pruned, and validate_graph then flags the missing required input.

    To run such a producer the user must wire a Start node into it (see
    test_execute_graph_triggered_producer_runs below).
    """
    nodes = [
        _start_node(),
        # Reachable via trigger from Start.
        {"id": "trig_src", "type": "TensorInput",
         "data": {"params": {"shape": "1,1,5,5", "value_mode": "zeros"}}},
        # Pure producer — no inputs, no trigger → not reachable from Start →
        # pruned from the executable subgraph.
        {"id": "kernel", "type": "Conv2dKernel",
         "data": {"params": {"preset": "EdgeDetection3x3"}}},
        # Downstream node with TWO required inputs, one fed by the
        # trigger chain and one by the (pruned) pure producer.
        {"id": "conv", "type": "Conv2dExplicit",
         "data": {"params": {"stride": 1, "padding": 1}}},
        {"id": "sink", "type": "Print",
         "data": {"params": {"label": "out"}}},
    ]
    edges = [
        _trigger("et", "start", "trig_src"),
        {"id": "e_src_conv", "source": "trig_src", "target": "conv",
         "sourceHandle": "tensor", "targetHandle": "tensor"},
        {"id": "e_kernel_conv", "source": "kernel", "target": "conv",
         "sourceHandle": "tensor", "targetHandle": "kernel"},
        {"id": "e_conv_sink", "source": "conv", "target": "sink",
         "sourceHandle": "tensor", "targetHandle": "value"},
    ]

    with pytest.raises(GraphValidationError, match="kernel"):
        asyncio.run(execute_graph(nodes, edges))


def test_execute_graph_triggered_producer_runs():
    """A producer wired to a Start node (via its own trigger edge) IS an
    entry point and runs — feeding its output downstream as expected."""
    nodes = [
        _start_node(),
        {"id": "trig_src", "type": "TensorInput",
         "data": {"params": {"shape": "1,1,5,5", "value_mode": "zeros"}}},
        # Producer now has its own trigger from Start → it's an entry point.
        {"id": "kernel", "type": "Conv2dKernel",
         "data": {"params": {"preset": "EdgeDetection3x3"}}},
        {"id": "conv", "type": "Conv2dExplicit",
         "data": {"params": {"stride": 1, "padding": 1}}},
        {"id": "sink", "type": "Print",
         "data": {"params": {"label": "out"}}},
    ]
    edges = [
        _trigger("et", "start", "trig_src"),
        _trigger("ek", "start", "kernel"),
        {"id": "e_src_conv", "source": "trig_src", "target": "conv",
         "sourceHandle": "tensor", "targetHandle": "tensor"},
        {"id": "e_kernel_conv", "source": "kernel", "target": "conv",
         "sourceHandle": "tensor", "targetHandle": "kernel"},
        {"id": "e_conv_sink", "source": "conv", "target": "sink",
         "sourceHandle": "tensor", "targetHandle": "value"},
    ]

    results = asyncio.run(execute_graph(nodes, edges))
    assert "kernel" in results, "Triggered producer Conv2dKernel must be executed"
    assert "conv" in results, "Conv2dExplicit downstream must be executed"
    assert "sink" in results


def test_execute_graph_still_prunes_draft_pure_producer():
    """A pure producer that does NOT feed into anything reachable should
    still be pruned (the fix doesn't relax the draft-component rule)."""
    nodes = [
        _start_node(),
        {"id": "trig_src", "type": "_TestSource", "data": {"params": {"val": 1}}},
        # Draft producer: not wired to anything reachable.
        {"id": "draft_producer", "type": "Conv2dKernel",
         "data": {"params": {"preset": "EdgeDetection3x3"}}},
    ]
    edges = [
        _trigger("et", "start", "trig_src"),
    ]
    results = asyncio.run(execute_graph(nodes, edges))
    assert "trig_src" in results
    assert "draft_producer" not in results, (
        "Producer with no executable consumer should still be pruned"
    )


def test_find_entry_points_only_trigger_targets():
    """Only nodes with incoming trigger edges are entry points."""
    nodes = [
        _start_node(),
        {"id": "ds", "type": "Dataset", "data": {}},
        {"id": "other", "type": "Dataset", "data": {}},
    ]
    edges = [_trigger("e1", "start", "ds")]
    result = find_entry_points(nodes, edges)
    assert "ds" in result
    assert "start" not in result  # Start itself is NOT an entry point
    assert "other" not in result


def test_find_entry_points_via_trigger_edge():
    nodes = [
        _start_node(),
        {"id": "ds", "type": "Dataset", "data": {}},
    ]
    edges = [_trigger("e1", "start", "ds")]
    result = find_entry_points(nodes, edges)
    assert result == ["ds"]
    assert "start" not in result


def test_find_entry_points_multiple_targets():
    nodes = [
        _start_node(),
        {"id": "a", "type": "Dataset", "data": {}},
        {"id": "b", "type": "Dataset", "data": {}},
    ]
    edges = [
        _trigger("e1", "start", "a"),
        _trigger("e2", "start", "b"),
    ]
    result = set(find_entry_points(nodes, edges))
    assert result == {"a", "b"}


def test_find_entry_points_none():
    nodes = [{"id": "a"}, {"id": "b"}]
    edges = [{"id": "e1", "source": "a", "target": "b", "type": "data"}]
    assert find_entry_points(nodes, edges) == []


def test_reachable_traverses_data_edges_only():
    nodes = [{"id": n} for n in ["start", "ds", "dl", "model"]]
    edges = [
        {"id": "e1", "source": "start", "target": "ds", "type": "trigger"},
        {"id": "e2", "source": "ds", "target": "dl", "type": "data"},
        {"id": "e3", "source": "dl", "target": "model", "type": "data"},
    ]
    reachable = reachable_from_entry_points(["ds"], edges)
    assert reachable == {"ds", "dl", "model"}


def test_reachable_handles_disconnected_components():
    nodes = [{"id": n} for n in ["a", "b", "x", "y"]]
    edges = [
        {"id": "e1", "source": "a", "target": "b", "type": "data"},
        {"id": "e2", "source": "x", "target": "y", "type": "data"},
    ]
    assert reachable_from_entry_points(["a"], edges) == {"a", "b"}


def _make_node(nid, ntype="Dataset", is_entry=False):
    return {"id": nid, "type": ntype, "data": {"params": {}}}


def _make_edge(eid, src, tgt, etype="data"):
    return {
        "id": eid,
        "source": src,
        "target": tgt,
        "sourceHandle": "out",
        "targetHandle": "in",
        "type": etype,
    }


def test_validate_rejects_no_entry_points():
    nodes = [_make_node("a"), _make_node("b")]
    edges = [_make_edge("e1", "a", "b")]
    errors = validate_graph(nodes, edges)
    assert any("entry point" in err.lower() for err in errors)


def test_validate_accepts_start_with_trigger():
    nodes = [_start_node(), _make_node("a"), _make_node("b")]
    edges = [
        _trigger("et", "start", "a"),
        _make_edge("e1", "a", "b"),
    ]
    errors = validate_graph(nodes, edges)
    assert not any("entry point" in err.lower() for err in errors)


def test_validate_allows_trigger_to_any_node():
    """Start trigger can connect to any node — it's a control-flow marker, not data."""
    nodes = [
        _start_node(),
        {"id": "conv", "type": "Conv2d", "data": {"params": {}}},
    ]
    edges = [_trigger("et", "start", "conv")]
    errors = validate_graph(nodes, edges)
    # Conv2d will have a "missing required input" error (its tensor input),
    # but NOT a "trigger cannot connect" error.
    assert not any("trigger" in err.lower() for err in errors)


def test_validate_allows_cycle_in_draft_component():
    """A cycle inside a non-entry-pointed (draft) component should NOT
    fail validation, because the draft is skipped at execution."""
    nodes = [
        _start_node(),
        _make_node("ep"),
        _make_node("a"),
        _make_node("b"),
        _make_node("c"),
    ]
    edges = [
        _trigger("et", "start", "ep"),
        # Cycle in the draft component a->b->c->a
        _make_edge("e1", "a", "b"),
        _make_edge("e2", "b", "c"),
        _make_edge("e3", "c", "a"),
    ]
    errors = validate_graph(nodes, edges)
    assert not any("cycle" in err.lower() for err in errors)


def test_validate_rejects_cycle_in_entry_pointed_component():
    """A cycle in an entry-pointed component fails."""
    nodes = [
        _start_node(),
        _make_node("a"),
        _make_node("b"),
    ]
    edges = [
        _trigger("et", "start", "a"),
        _make_edge("e1", "a", "b"),
        _make_edge("e2", "b", "a"),
    ]
    errors = validate_graph(nodes, edges)
    assert any("cycle" in err.lower() for err in errors)


def test_topological_levels_excludes_trigger_from_in_degree():
    """A Dataset receiving a trigger from a Start should still be at level 0."""
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "ds", "type": "Dataset", "data": {"params": {}}},
        {"id": "dl", "type": "DataLoader", "data": {"params": {}}},
    ]
    edges = [
        {"id": "e1", "source": "start", "target": "ds", "sourceHandle": "trigger", "type": "trigger"},
        {"id": "e2", "source": "ds", "target": "dl", "sourceHandle": "dataset", "targetHandle": "dataset", "type": "data"},
    ]
    levels = topological_levels(nodes, edges)
    # Both start and ds should be in level 0 (start has no inputs, ds's
    # only incoming edge is a trigger which is excluded).
    assert "start" in levels[0]
    assert "ds" in levels[0]
    assert "dl" in levels[1]


@pytest.mark.asyncio
async def test_execute_graph_skips_draft_components():
    """Draft components (no Start trigger) should be skipped silently."""
    nodes = [
        _start_node(),
        {"id": "live", "type": "_TestSource", "data": {"params": {"val": 42}}},
        {"id": "draft", "type": "_TestSource", "data": {"params": {"val": 99}}},
    ]
    edges = [_trigger("et", "start", "live")]
    results = await execute_graph(nodes, edges)
    assert "live" in results
    assert "draft" not in results


# ── Bypass / mute (core#128) ─────────────────────────────────────────────
#
# A bypassed node is removed from the executable graph and each of its
# outputs forwards the first type-compatible input, ComfyUI-style. The tests
# below cover the shapes that matter: a straight chain, a fan-out, a chain of
# two bypassed nodes, and the failure modes (no compatible input, nothing
# wired into the matched input).


def _bypassed(node):
    """Mark a node dict as bypassed, mirroring what the canvas serializes."""
    node.setdefault("data", {})["bypassed"] = True
    return node


def _node(nid, ntype, **params):
    return {"id": nid, "type": ntype, "data": {"params": params}}


def _data_edge(eid, src, src_handle, tgt, tgt_handle):
    return {
        "id": eid,
        "source": src,
        "target": tgt,
        "sourceHandle": src_handle,
        "targetHandle": tgt_handle,
        "type": "data",
    }


def _chain_with_bypassed_dropout(bypass=True):
    """Start -> TensorCreate -> Dropout(bypassed?) -> Print."""
    dropout = _node("drop", "Dropout", p=0.5)
    if bypass:
        _bypassed(dropout)
    nodes = [
        _start_node(),
        _node("make", "TensorCreate", shape="2,2", fill="ones"),
        dropout,
        _node("out", "Print", label="tail"),
    ]
    edges = [
        _trigger("et", "start", "make"),
        _data_edge("e1", "make", "tensor", "drop", "tensor"),
        _data_edge("e2", "drop", "tensor", "out", "value"),
    ]
    return nodes, edges


def test_bypass_removes_the_node_and_rewires_downstream():
    from app.core.graph_engine import prepare_executable_graph

    nodes, edges = _chain_with_bypassed_dropout()
    exec_nodes, exec_edges, _ = prepare_executable_graph(nodes, edges)

    assert "drop" not in {n["id"] for n in exec_nodes}
    rewired = [e for e in exec_edges if e["target"] == "out"]
    assert len(rewired) == 1
    assert rewired[0]["source"] == "make"
    assert rewired[0]["sourceHandle"] == "tensor"
    assert rewired[0]["targetHandle"] == "value"


def test_bypass_leaves_a_clean_graph_untouched_by_identity():
    """No bypassed node -> the very same list objects come back out."""
    from app.core.graph_engine import resolve_bypass

    nodes, edges = _chain_with_bypassed_dropout(bypass=False)
    resolution = resolve_bypass(nodes, edges)
    assert resolution.nodes is nodes
    assert resolution.edges is edges
    assert resolution.errors == []


@pytest.mark.asyncio
async def test_bypassed_mid_chain_node_runs_as_if_absent():
    nodes, edges = _chain_with_bypassed_dropout()
    results = await execute_graph(nodes, edges)

    assert "drop" not in results
    # Pass-through is by reference: Print received exactly what TensorCreate
    # produced, with no Dropout in between.
    assert results["out"]["value"] is results["make"]["tensor"]


@pytest.mark.asyncio
async def test_un_bypassing_restores_the_node():
    nodes, edges = _chain_with_bypassed_dropout(bypass=False)
    results = await execute_graph(nodes, edges)

    assert "drop" in results
    assert results["out"]["value"] is not results["make"]["tensor"]


@pytest.mark.asyncio
async def test_bypass_fans_out_to_every_consumer():
    nodes, edges = _chain_with_bypassed_dropout()
    nodes.append(_node("out2", "Print", label="second"))
    edges.append(_data_edge("e3", "drop", "tensor", "out2", "value"))

    results = await execute_graph(nodes, edges)
    assert results["out"]["value"] is results["make"]["tensor"]
    assert results["out2"]["value"] is results["make"]["tensor"]


@pytest.mark.asyncio
async def test_a_chain_of_bypassed_nodes_resolves_to_the_original_source():
    nodes = [
        _start_node(),
        _node("make", "TensorCreate", shape="2,2", fill="ones"),
        _bypassed(_node("d1", "Dropout", p=0.5)),
        _bypassed(_node("d2", "Dropout", p=0.5)),
        _node("out", "Print", label="tail"),
    ]
    edges = [
        _trigger("et", "start", "make"),
        _data_edge("e1", "make", "tensor", "d1", "tensor"),
        _data_edge("e2", "d1", "tensor", "d2", "tensor"),
        _data_edge("e3", "d2", "tensor", "out", "value"),
    ]
    results = await execute_graph(nodes, edges)

    assert "d1" not in results and "d2" not in results
    assert results["out"]["value"] is results["make"]["tensor"]


def test_bypass_with_no_type_compatible_input_is_a_validation_error():
    """DataLoader takes DATASET and emits DATALOADER -- nothing to forward."""
    from app.core.graph_engine import prepare_executable_graph

    nodes = [
        _start_node(),
        _node("ds", "Dataset"),
        _bypassed(_node("dl", "DataLoader")),
        _node("loop", "TrainingLoop"),
    ]
    edges = [
        _trigger("et", "start", "ds"),
        _data_edge("e1", "ds", "dataset", "dl", "dataset"),
        _data_edge("e2", "dl", "dataloader", "loop", "dataloader"),
    ]

    errors = validate_graph(nodes, edges)
    assert any(
        "dl" in e and "no type-compatible input" in e and "dataloader" in e
        for e in errors
    ), errors

    with pytest.raises(GraphValidationError, match="no type-compatible input"):
        prepare_executable_graph(nodes, edges)


def test_bypass_whose_matched_input_is_unconnected_leaves_downstream_unwired():
    """Nothing flows into the bypassed node, so its consumer loses its input."""
    nodes = [
        _start_node(),
        _bypassed(_node("drop", "Dropout", p=0.5)),
        _node("out", "Print", label="tail"),
    ]
    edges = [
        _trigger("et", "start", "drop"),
        _data_edge("e1", "drop", "tensor", "out", "value"),
    ]
    errors = validate_graph(nodes, edges)
    assert any("Missing required input 'value'" in e and "out" in e for e in errors), errors


def test_bypassing_a_trigger_target_repoints_the_trigger_downstream():
    """A bypassed entry point hands its trigger to what it fed."""
    from app.core.graph_engine import find_entry_points, resolve_bypass

    nodes = [
        _start_node(),
        _bypassed(_node("drop", "Dropout", p=0.5)),
        _node("out", "Print", label="tail"),
    ]
    edges = [
        _trigger("et", "start", "drop"),
        _data_edge("e1", "drop", "tensor", "out", "value"),
    ]
    resolution = resolve_bypass(nodes, edges)

    triggers = [e for e in resolution.edges if e.get("type") == "trigger"]
    assert [e["target"] for e in triggers] == ["out"]
    assert find_entry_points(resolution.nodes, resolution.edges) == ["out"]


def test_bypass_is_refused_on_a_preset_node():
    from app.core.graph_engine import prepare_executable_graph

    nodes = [
        _start_node(),
        _bypassed({"id": "p1", "type": "preset:Whatever", "data": {"params": {}}}),
    ]
    edges = [_trigger("et", "start", "p1")]

    assert any("not supported on preset node" in e for e in validate_graph(nodes, edges))
    with pytest.raises(GraphValidationError, match="not supported on preset node"):
        prepare_executable_graph(nodes, edges)


def test_bypass_on_an_unknown_node_type_reports_the_unknown_type():
    """The real problem is the missing node class, not the pass-through."""
    nodes = [
        _start_node(),
        _bypassed(_node("ghost", "NotARealNode")),
        _node("out", "Print", label="tail"),
    ]
    edges = [
        _trigger("et", "start", "ghost"),
        _data_edge("e1", "ghost", "value", "out", "value"),
    ]
    errors = validate_graph(nodes, edges)
    assert any("Unknown node type: NotARealNode" in e for e in errors), errors
    assert not any("no type-compatible input" in e for e in errors), errors


def test_bypass_records_the_pass_through_links_it_applied():
    """The link list is what the Python exporter comments the bypass with."""
    from app.core.graph_engine import resolve_bypass

    nodes, edges = _chain_with_bypassed_dropout()
    resolution = resolve_bypass(nodes, edges)

    assert len(resolution.links) == 1
    link = resolution.links[0]
    assert (link.node_id, link.node_type) == ("drop", "Dropout")
    assert (link.output, link.input) == ("tensor", "tensor")
    assert (link.source, link.source_handle) == ("make", "tensor")


def test_bypass_is_refused_on_the_graph_io_contract_nodes():
    """GraphInput/GraphOutput ARE the signature; muting one cannot be silent.

    GraphOutput is the sharp case: it declares no output ports, so nothing
    would ever ask what it forwards and no pass-through error would fire --
    it would simply vanish, leaving a published contract advertising an
    output the run cannot produce.
    """
    from app.core.graph_engine import prepare_executable_graph, resolve_bypass

    nodes = [
        _start_node(),
        _node("make", "TensorCreate", shape="2,2", fill="ones"),
        _bypassed(_node("out", "GraphOutput", name="result")),
    ]
    edges = [
        _trigger("et", "start", "make"),
        _data_edge("e1", "make", "tensor", "out", "value"),
    ]

    assert any(
        "Bypass is not supported on GraphOutput node out" in e
        for e in validate_graph(nodes, edges)
    )
    # Refused means LEFT IN PLACE: the contract layer still sees the node.
    assert "out" in {n["id"] for n in resolve_bypass(nodes, edges).nodes}

    with pytest.raises(GraphValidationError, match="not supported on GraphOutput"):
        prepare_executable_graph(nodes, edges)


def test_bypass_is_refused_on_a_graph_input_node():
    from app.core.graph_engine import prepare_executable_graph

    nodes = [
        _start_node(),
        _bypassed(_node("in", "GraphInput", name="x", type="string")),
        _node("show", "Print", label="tail"),
    ]
    edges = [
        _trigger("et", "start", "in"),
        _data_edge("e1", "in", "value", "show", "value"),
    ]
    assert any(
        "Bypass is not supported on GraphInput node in" in e
        for e in validate_graph(nodes, edges)
    )
    with pytest.raises(GraphValidationError, match="not supported on GraphInput"):
        prepare_executable_graph(nodes, edges)


def test_bypass_forwards_the_first_matching_input_of_several():
    """Lerp takes tensor_a, tensor_b, alpha and emits one TENSOR.

    All three inputs are equally compatible, so this pins WHICH one
    positional first-match picks: the earliest declared, tensor_a.
    """
    from app.core.graph_engine import resolve_bypass

    nodes = [
        _start_node(),
        _node("a", "TensorCreate", shape="2,2", fill="ones"),
        _node("b", "TensorCreate", shape="2,2", fill="zeros"),
        _bypassed(_node("mix", "Lerp", alpha=0.5)),
        _node("out", "Print", label="tail"),
    ]
    edges = [
        _trigger("et", "start", "a"),
        _data_edge("e1", "a", "tensor", "mix", "tensor_a"),
        _data_edge("e2", "b", "tensor", "mix", "tensor_b"),
        _data_edge("e3", "mix", "tensor", "out", "value"),
    ]
    resolution = resolve_bypass(nodes, edges)

    assert resolution.errors == []
    assert [(link.input, link.source) for link in resolution.links] == [("tensor_a", "a")]
    rewired = [e for e in resolution.edges if e["target"] == "out"]
    assert (rewired[0]["source"], rewired[0]["sourceHandle"]) == ("a", "tensor")


def test_bypass_resolves_each_output_of_a_multi_output_node_by_type():
    """TrainTestSplit emits x_train/x_test (TENSOR) and y_train/y_test (LIST).

    Each output picks the first input its OWN type is compatible with, so the
    two consumed here resolve to different inputs -- and to different source
    ports of the same upstream node.
    """
    from app.core.graph_engine import resolve_bypass

    nodes = [
        _start_node(),
        _node("csv", "CSVReader", path="data/samples/iris.csv", target_column="species"),
        _bypassed(_node("split", "TrainTestSplit")),
        _node("o1", "Print", label="a"),
        _node("o2", "Print", label="b"),
    ]
    edges = [
        _trigger("et", "start", "csv"),
        _data_edge("e1", "csv", "tensor", "split", "features"),
        _data_edge("e2", "csv", "labels", "split", "labels"),
        _data_edge("e3", "split", "x_train", "o1", "value"),
        _data_edge("e4", "split", "y_test", "o2", "value"),
    ]
    resolution = resolve_bypass(nodes, edges)

    assert resolution.errors == []
    assert sorted(
        (link.output, link.input, link.source_handle) for link in resolution.links
    ) == [
        ("x_train", "features", "tensor"),
        ("y_test", "labels", "labels"),
    ]


def test_bypass_matching_is_wider_than_equality_when_a_port_is_any():
    """Switch: selector (SCALAR), input_0..3 (ANY) -> output (ANY).

    ``is_compatible`` -- the edge validator's own predicate -- accepts SCALAR
    into ANY, so positional first-match takes `selector`, not `input_0`.
    Surprising at a glance and deliberately pinned: the rule is positional
    over a WIDE compatibility test, exactly as ComfyUI's is, and a strict
    same-DataType rule would refuse pass-throughs the graph could legally
    have been wired with.
    """
    from app.core.graph_engine import resolve_bypass

    nodes = [
        _start_node(),
        _node("sel", "TensorCreate", shape="1", fill="ones"),
        _node("a", "TensorCreate", shape="2,2", fill="ones"),
        _bypassed(_node("sw", "Switch")),
        _node("out", "Print", label="tail"),
    ]
    edges = [
        _trigger("et", "start", "sel"),
        _data_edge("e1", "sel", "tensor", "sw", "selector"),
        _data_edge("e2", "a", "tensor", "sw", "input_0"),
        _data_edge("e3", "sw", "output", "out", "value"),
    ]
    resolution = resolve_bypass(nodes, edges)

    assert resolution.errors == []
    assert [(link.input, link.source) for link in resolution.links] == [("selector", "sel")]


def test_a_missing_input_caused_by_a_bypass_names_the_bypass():
    """Otherwise the error reads as a port the user forgot to wire."""
    nodes = [
        _start_node(),
        _bypassed(_node("drop", "Dropout", p=0.5)),
        _node("out", "Print", label="tail"),
    ]
    edges = [
        _trigger("et", "start", "drop"),
        _data_edge("e1", "drop", "tensor", "out", "value"),
    ]
    errors = validate_graph(nodes, edges)
    assert any(
        "Missing required input 'value' on node out" in e
        and "input dropped because 'drop' is bypassed" in e
        for e in errors
    ), errors


def test_an_ordinary_missing_input_carries_no_bypass_attribution():
    nodes = [_start_node(), _node("out", "Print", label="tail")]
    edges = [_trigger("et", "start", "out")]
    errors = validate_graph(nodes, edges)
    assert any("Missing required input 'value' on node out" in e for e in errors)
    assert not any("is bypassed" in e for e in errors), errors
