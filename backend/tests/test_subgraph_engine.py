"""core#137 -- subgraph expansion, boundary rules, cycles, equivalence.

The load-bearing claim of a subgraph is that collapsing a block of graph
changes nothing about what the graph computes. Structural assertions cannot
carry that claim, so the equivalence test here EXECUTES both graphs and
compares the numbers that come out.
"""

from __future__ import annotations

import copy

import pytest

from app.core.graph_engine import (
    GraphValidationError,
    build_preset_fallback,
    build_subgraph_index,
    describe_cycle,
    execute_graph,
    expand_subgraphs,
    expand_subgraphs_deep,
    find_cycle,
    outermost_container,
    prepare_executable_graph,
    validate_graph,
)


# ── Fixtures: a tiny arithmetic graph and its collapsed twin ─────────────
#
# Start triggers two TensorCreate roots; each feeds a ScalarMultiply; the two
# products are added and averaged. Every value is exact in float32, so the
# equivalence check is an equality, not a tolerance.


def _flat_graph() -> tuple[list[dict], list[dict]]:
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {"params": {}}},
        {"id": "a", "type": "TensorCreate", "position": {"x": 1, "y": 0},
         "data": {"params": {"shape": "2,2", "fill": "full", "value": 2.0}}},
        {"id": "b", "type": "TensorCreate", "position": {"x": 1, "y": 1},
         "data": {"params": {"shape": "2,2", "fill": "full", "value": 3.0}}},
        {"id": "m1", "type": "ScalarMultiply", "position": {"x": 2, "y": 0},
         "data": {"params": {"scalar": 5.0}}},
        {"id": "m2", "type": "ScalarMultiply", "position": {"x": 2, "y": 1},
         "data": {"params": {"scalar": 7.0}}},
        {"id": "sum", "type": "Add", "position": {"x": 3, "y": 0},
         "data": {"params": {"alpha": 1.0}}},
        {"id": "avg", "type": "Mean", "position": {"x": 4, "y": 0},
         "data": {"params": {"dim": "-1", "keepdim": False}}},
    ]
    edges = [
        {"id": "t1", "source": "start", "target": "a",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        {"id": "t2", "source": "start", "target": "b",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        {"id": "e1", "source": "a", "target": "m1",
         "sourceHandle": "tensor", "targetHandle": "tensor", "type": "data"},
        {"id": "e2", "source": "b", "target": "m2",
         "sourceHandle": "tensor", "targetHandle": "tensor", "type": "data"},
        {"id": "e3", "source": "m1", "target": "sum",
         "sourceHandle": "tensor", "targetHandle": "tensor_a", "type": "data"},
        {"id": "e4", "source": "m2", "target": "sum",
         "sourceHandle": "tensor", "targetHandle": "tensor_b", "type": "data"},
        {"id": "e5", "source": "sum", "target": "avg",
         "sourceHandle": "tensor", "targetHandle": "tensor", "type": "data"},
    ]
    return nodes, edges


def _collapsed_graph() -> tuple[list[dict], list[dict], list[dict]]:
    """The same computation with {m1, m2, sum} collapsed into one instance.

    Hand-written on purpose: this fixture is the CONTRACT the frontend's
    collapse must produce, so writing it by hand is what lets the backend
    test fail if the contract drifts.
    """
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {"params": {}}},
        {"id": "a", "type": "TensorCreate", "position": {"x": 1, "y": 0},
         "data": {"params": {"shape": "2,2", "fill": "full", "value": 2.0}}},
        {"id": "b", "type": "TensorCreate", "position": {"x": 1, "y": 1},
         "data": {"params": {"shape": "2,2", "fill": "full", "value": 3.0}}},
        {"id": "blk", "type": "subgraph:combine", "position": {"x": 2, "y": 0},
         "data": {"params": {}}},
        {"id": "avg", "type": "Mean", "position": {"x": 4, "y": 0},
         "data": {"params": {"dim": "-1", "keepdim": False}}},
    ]
    edges = [
        {"id": "t1", "source": "start", "target": "a",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        {"id": "t2", "source": "start", "target": "b",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        {"id": "e1", "source": "a", "target": "blk",
         "sourceHandle": "tensor", "targetHandle": "tensor", "type": "data"},
        {"id": "e2", "source": "b", "target": "blk",
         "sourceHandle": "tensor", "targetHandle": "tensor_2", "type": "data"},
        {"id": "e5", "source": "blk", "target": "avg",
         "sourceHandle": "tensor", "targetHandle": "tensor", "type": "data"},
    ]
    subgraphs = [{
        "id": "combine",
        "name": "Combine",
        "nodes": [
            {"id": "m1", "type": "ScalarMultiply", "position": {"x": 0, "y": 0},
             "data": {"params": {"scalar": 5.0}}},
            {"id": "m2", "type": "ScalarMultiply", "position": {"x": 0, "y": 1},
             "data": {"params": {"scalar": 7.0}}},
            {"id": "sum", "type": "Add", "position": {"x": 1, "y": 0},
             "data": {"params": {"alpha": 1.0}}},
        ],
        "edges": [
            {"id": "i1", "source": "m1", "target": "sum",
             "sourceHandle": "tensor", "targetHandle": "tensor_a",
             "type": "data"},
            {"id": "i2", "source": "m2", "target": "sum",
             "sourceHandle": "tensor", "targetHandle": "tensor_b",
             "type": "data"},
        ],
        "interface": {
            "inputs": [
                {"port": "tensor", "innerNode": "m1", "innerPort": "tensor"},
                {"port": "tensor_2", "innerNode": "m2", "innerPort": "tensor"},
            ],
            "outputs": [
                {"port": "tensor", "innerNode": "sum", "innerPort": "tensor"},
            ],
            "triggerTargets": [],
        },
    }]
    return nodes, edges, subgraphs


def _passthrough_subgraph(sid: str = "pass") -> dict:
    """One ScalarMultiply behind a boundary: one input, one output."""
    return {
        "id": sid,
        "name": sid,
        "nodes": [
            {"id": "mul", "type": "ScalarMultiply",
             "position": {"x": 0, "y": 0}, "data": {"params": {"scalar": 2.0}}},
        ],
        "edges": [],
        "interface": {
            "inputs": [
                {"port": "in", "innerNode": "mul", "innerPort": "tensor"},
            ],
            "outputs": [
                {"port": "out", "innerNode": "mul", "innerPort": "tensor"},
            ],
            "triggerTargets": ["mul"],
        },
    }


# ── Expansion mechanics ─────────────────────────────────────────────────


def test_expansion_namespaces_inner_nodes_under_the_instance():
    nodes, edges, subgraphs = _collapsed_graph()
    out_nodes, _out_edges, mapping = expand_subgraphs(
        nodes, edges, build_subgraph_index(subgraphs)
    )
    ids = {n["id"] for n in out_nodes}
    assert {"blk/m1", "blk/m2", "blk/sum"} <= ids
    assert "blk" not in ids  # the instance itself is gone
    assert mapping == {"blk/m1": "blk", "blk/m2": "blk", "blk/sum": "blk"}


def test_expansion_rewires_boundary_edges_to_inner_ports():
    nodes, edges, subgraphs = _collapsed_graph()
    _out_nodes, out_edges, _ = expand_subgraphs(
        nodes, edges, build_subgraph_index(subgraphs)
    )
    wiring = {
        (e["source"], e.get("sourceHandle"), e["target"], e.get("targetHandle"))
        for e in out_edges
    }
    assert ("a", "tensor", "blk/m1", "tensor") in wiring
    assert ("b", "tensor", "blk/m2", "tensor") in wiring
    assert ("blk/sum", "tensor", "avg", "tensor") in wiring
    # Nothing may still name the instance node.
    assert not any(
        e["source"] == "blk" or e["target"] == "blk" for e in out_edges
    )


def test_expansion_deep_copies_params_so_instances_cannot_alias():
    """Two instances must not share one params dict.

    Aliasing would be invisible in v1 (no per-instance overrides) and would
    become a data-corruption bug the moment one is added.
    """
    subgraphs = [_passthrough_subgraph()]
    nodes = [
        {"id": "p1", "type": "subgraph:pass", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "p2", "type": "subgraph:pass", "position": {"x": 1, "y": 0},
         "data": {}},
    ]
    out_nodes, _, _ = expand_subgraphs(
        nodes, [], build_subgraph_index(subgraphs)
    )
    by_id = {n["id"]: n for n in out_nodes}
    first = by_id["p1/mul"]["data"]["params"]
    second = by_id["p2/mul"]["data"]["params"]
    assert first == second
    assert first is not second


def test_unknown_subgraph_is_named_with_the_instance():
    nodes = [{"id": "blk", "type": "subgraph:ghost",
              "position": {"x": 0, "y": 0}, "data": {}}]
    with pytest.raises(GraphValidationError) as excinfo:
        expand_subgraphs(nodes, [], {})
    assert "Unknown subgraph: ghost" in str(excinfo.value)
    assert "blk" in str(excinfo.value)


def test_boundary_port_the_interface_does_not_declare_is_refused():
    """An undeclared port must be an error, not a silently dropped edge."""
    subgraphs = [_passthrough_subgraph()]
    nodes = [
        {"id": "src", "type": "TensorCreate", "position": {"x": 0, "y": 0},
         "data": {"params": {}}},
        {"id": "p1", "type": "subgraph:pass", "position": {"x": 1, "y": 0},
         "data": {}},
    ]
    edges = [{"id": "e", "source": "src", "target": "p1",
              "sourceHandle": "tensor", "targetHandle": "nope",
              "type": "data"}]
    with pytest.raises(GraphValidationError) as excinfo:
        expand_subgraphs(nodes, edges, build_subgraph_index(subgraphs))
    assert "'nope'" in str(excinfo.value)
    assert "pass" in str(excinfo.value)


def test_trigger_edge_fans_out_to_every_declared_target():
    """One Start edge into an instance must reach every inner root.

    A data-only dependency does not pull an untriggered root into a run, so
    fanning out to only the first inner node would silently drop half the
    block.
    """
    definition = _passthrough_subgraph("two")
    definition["nodes"].append(
        {"id": "mul2", "type": "ScalarMultiply", "position": {"x": 0, "y": 1},
         "data": {"params": {"scalar": 3.0}}}
    )
    definition["interface"]["triggerTargets"] = ["mul", "mul2"]
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "blk", "type": "subgraph:two", "position": {"x": 1, "y": 0},
         "data": {}},
    ]
    edges = [{"id": "t", "source": "start", "target": "blk",
              "sourceHandle": "trigger", "targetHandle": "__trigger",
              "type": "trigger"}]
    _, out_edges, _ = expand_subgraphs(
        nodes, edges, build_subgraph_index([definition])
    )
    triggered = {
        e["target"] for e in out_edges if e.get("type") == "trigger"
    }
    assert triggered == {"blk/mul", "blk/mul2"}
    # Fanned-out edges must not collide on id -- React Flow and the run store
    # both key on it.
    trigger_ids = [e["id"] for e in out_edges if e.get("type") == "trigger"]
    assert len(trigger_ids) == len(set(trigger_ids))


def test_trigger_falls_back_to_inner_roots_when_none_declared():
    """A hand-wired Start -> instance edge still starts the block."""
    definition = _passthrough_subgraph("roots")
    definition["interface"]["triggerTargets"] = []
    definition["nodes"].append(
        {"id": "after", "type": "Mean", "position": {"x": 1, "y": 0},
         "data": {"params": {}}}
    )
    definition["edges"].append(
        {"id": "i", "source": "mul", "target": "after",
         "sourceHandle": "tensor", "targetHandle": "tensor", "type": "data"}
    )
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "blk", "type": "subgraph:roots", "position": {"x": 1, "y": 0},
         "data": {}},
    ]
    edges = [{"id": "t", "source": "start", "target": "blk",
              "sourceHandle": "trigger", "targetHandle": "__trigger",
              "type": "trigger"}]
    _, out_edges, _ = expand_subgraphs(
        nodes, edges, build_subgraph_index([definition])
    )
    triggered = {e["target"] for e in out_edges if e.get("type") == "trigger"}
    assert triggered == {"blk/mul"}  # 'after' is fed from inside, not a root


def test_trigger_target_that_is_not_in_the_definition_is_refused():
    definition = _passthrough_subgraph("bad")
    definition["interface"]["triggerTargets"] = ["nosuch"]
    nodes = [{"id": "blk", "type": "subgraph:bad",
              "position": {"x": 0, "y": 0}, "data": {}}]
    with pytest.raises(GraphValidationError) as excinfo:
        expand_subgraphs(nodes, [], build_subgraph_index([definition]))
    assert "nosuch" in str(excinfo.value)


# ── Nesting, recursion, depth ───────────────────────────────────────────


def test_nested_subgraphs_expand_to_the_bottom():
    inner = _passthrough_subgraph("inner")
    outer = {
        "id": "outer",
        "name": "outer",
        "nodes": [
            {"id": "nested", "type": "subgraph:inner",
             "position": {"x": 0, "y": 0}, "data": {}},
        ],
        "edges": [],
        "interface": {
            "inputs": [
                {"port": "in", "innerNode": "nested", "innerPort": "in"},
            ],
            "outputs": [
                {"port": "out", "innerNode": "nested", "innerPort": "out"},
            ],
            "triggerTargets": ["nested"],
        },
    }
    nodes = [{"id": "blk", "type": "subgraph:outer",
              "position": {"x": 0, "y": 0}, "data": {}}]
    out_nodes, _, mapping = expand_subgraphs_deep(
        nodes, [], build_subgraph_index([inner, outer])
    )
    assert [n["id"] for n in out_nodes] == ["blk/nested/mul"]
    assert mapping["blk/nested/mul"] == "blk/nested"
    assert mapping["blk/nested"] == "blk"


def test_recursive_definition_is_named_not_left_to_the_depth_budget():
    a = {
        "id": "a", "name": "a", "edges": [],
        "nodes": [{"id": "x", "type": "subgraph:b",
                   "position": {"x": 0, "y": 0}, "data": {}}],
        "interface": {"inputs": [], "outputs": [], "triggerTargets": []},
    }
    b = {
        "id": "b", "name": "b", "edges": [],
        "nodes": [{"id": "y", "type": "subgraph:a",
                   "position": {"x": 0, "y": 0}, "data": {}}],
        "interface": {"inputs": [], "outputs": [], "triggerTargets": []},
    }
    nodes = [{"id": "blk", "type": "subgraph:a",
              "position": {"x": 0, "y": 0}, "data": {}}]
    with pytest.raises(GraphValidationError) as excinfo:
        expand_subgraphs_deep(nodes, [], build_subgraph_index([a, b]))
    message = str(excinfo.value)
    assert "Subgraph recursion" in message
    # Both definitions must be named, in the order they close the loop.
    assert "a -> b -> a" in message


def test_nesting_beyond_the_depth_budget_is_refused():
    chain = []
    for level in range(12):
        chain.append({
            "id": f"s{level}", "name": f"s{level}", "edges": [],
            "nodes": [{"id": "n", "type": f"subgraph:s{level + 1}",
                       "position": {"x": 0, "y": 0}, "data": {}}],
            "interface": {"inputs": [], "outputs": [], "triggerTargets": []},
        })
    chain.append({
        "id": "s12", "name": "s12", "edges": [],
        "nodes": [{"id": "leaf", "type": "ScalarMultiply",
                   "position": {"x": 0, "y": 0}, "data": {"params": {}}}],
        "interface": {"inputs": [], "outputs": [], "triggerTargets": []},
    })
    nodes = [{"id": "blk", "type": "subgraph:s0",
              "position": {"x": 0, "y": 0}, "data": {}}]
    with pytest.raises(GraphValidationError) as excinfo:
        expand_subgraphs_deep(nodes, [], build_subgraph_index(chain))
    assert "maximum depth of 10" in str(excinfo.value)


def test_a_definition_nobody_instantiates_cannot_break_a_run():
    """Stale definitions are data, not behaviour."""
    broken = {"id": "self", "name": "self", "edges": [],
              "nodes": [{"id": "me", "type": "subgraph:self",
                         "position": {"x": 0, "y": 0}, "data": {}}],
              "interface": {"inputs": [], "outputs": [], "triggerTargets": []}}
    nodes, edges = _flat_graph()
    out_nodes, _, mapping = expand_subgraphs_deep(
        nodes, edges, build_subgraph_index([broken])
    )
    assert out_nodes is nodes and mapping == {}


# ── Cycles across the boundary ──────────────────────────────────────────


def test_find_cycle_returns_a_closed_path():
    nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    edges = [
        {"source": "a", "target": "b"},
        {"source": "b", "target": "c"},
        {"source": "c", "target": "a"},
    ]
    cycle = find_cycle(nodes, edges)
    assert cycle is not None
    assert cycle[0] == cycle[-1]
    assert set(cycle) == {"a", "b", "c"}


def test_find_cycle_ignores_trigger_edges():
    nodes = [{"id": "s"}, {"id": "a"}]
    edges = [
        {"source": "s", "target": "a", "type": "trigger"},
        {"source": "a", "target": "s", "type": "data"},
    ]
    assert find_cycle(nodes, edges) is None


def test_describe_cycle_names_both_sides_of_a_boundary():
    text = describe_cycle(["outer", "blk/inner", "outer"])
    assert "outer -> blk/inner -> outer" in text
    # The path alone already contains "blk"; the clause is what tells a user
    # the loop involves a BOUNDARY, so pin the clause, not the substring.
    assert "crosses subgraph instance(s): blk" in text


def test_a_cycle_that_exists_only_inside_a_definition_is_rejected():
    """The interesting case: the canvas graph is acyclic, the run is not.

    The two instances are wired in a line, so nothing on the canvas looks
    like a loop. The loop is entirely inside the shared definition, which is
    invisible until expansion -- exactly the failure a boundary introduces.
    """
    definition = {
        "id": "loopy", "name": "loopy",
        "nodes": [
            {"id": "p", "type": "ScalarMultiply", "position": {"x": 0, "y": 0},
             "data": {"params": {}}},
            {"id": "q", "type": "ScalarMultiply", "position": {"x": 1, "y": 0},
             "data": {"params": {}}},
        ],
        "edges": [
            {"id": "f", "source": "p", "target": "q",
             "sourceHandle": "tensor", "targetHandle": "tensor",
             "type": "data"},
            {"id": "g", "source": "q", "target": "p",
             "sourceHandle": "tensor", "targetHandle": "tensor",
             "type": "data"},
        ],
        "interface": {
            "inputs": [{"port": "in", "innerNode": "p", "innerPort": "tensor"}],
            "outputs": [{"port": "out", "innerNode": "q",
                         "innerPort": "tensor"}],
            "triggerTargets": ["p"],
        },
    }
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "src", "type": "TensorCreate", "position": {"x": 1, "y": 0},
         "data": {"params": {"shape": "2,2"}}},
        {"id": "blk", "type": "subgraph:loopy", "position": {"x": 2, "y": 0},
         "data": {}},
    ]
    edges = [
        {"id": "t", "source": "start", "target": "src",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        {"id": "e", "source": "src", "target": "blk",
         "sourceHandle": "tensor", "targetHandle": "in", "type": "data"},
    ]

    # Pre-condition: at instance granularity the graph is a straight line.
    assert find_cycle(nodes, edges) is None

    errors = validate_graph(nodes, edges, subgraphs=[definition])
    cycle_errors = [e for e in errors if "cycle" in e]
    assert len(cycle_errors) == 1, errors
    message = cycle_errors[0]
    assert "blk/p" in message and "blk/q" in message
    assert "crosses subgraph instance(s): blk" in message


def test_a_cycle_between_two_instances_names_both():
    definition = _passthrough_subgraph("pass")
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "one", "type": "subgraph:pass", "position": {"x": 1, "y": 0},
         "data": {}},
        {"id": "two", "type": "subgraph:pass", "position": {"x": 2, "y": 0},
         "data": {}},
    ]
    edges = [
        {"id": "t", "source": "start", "target": "one",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        {"id": "e1", "source": "one", "target": "two",
         "sourceHandle": "out", "targetHandle": "in", "type": "data"},
        {"id": "e2", "source": "two", "target": "one",
         "sourceHandle": "out", "targetHandle": "in", "type": "data"},
    ]
    errors = validate_graph(nodes, edges, subgraphs=[definition])
    cycle_errors = [e for e in errors if "cycle" in e]
    assert len(cycle_errors) == 1, errors
    assert "one/mul" in cycle_errors[0]
    assert "two/mul" in cycle_errors[0]
    assert "crosses subgraph instance(s): one, two" in cycle_errors[0]


def test_an_ordinary_cycle_still_reports_its_path():
    """The message gained a path for every graph, not only subgraph ones."""
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "m1", "type": "ScalarMultiply", "position": {"x": 1, "y": 0},
         "data": {"params": {}}},
        {"id": "m2", "type": "ScalarMultiply", "position": {"x": 2, "y": 0},
         "data": {"params": {}}},
    ]
    edges = [
        {"id": "t", "source": "start", "target": "m1",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        {"id": "e1", "source": "m1", "target": "m2",
         "sourceHandle": "tensor", "targetHandle": "tensor", "type": "data"},
        {"id": "e2", "source": "m2", "target": "m1",
         "sourceHandle": "tensor", "targetHandle": "tensor", "type": "data"},
    ]
    errors = validate_graph(nodes, edges)
    assert any("m1 -> m2 -> m1" in e or "m2 -> m1 -> m2" in e for e in errors)
    assert not any("subgraph instance" in e for e in errors)


# ── Validation surface ──────────────────────────────────────────────────


def test_validate_reports_an_unknown_subgraph_once_not_twice():
    """The instance must not ALSO be reported as an unknown node type."""
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "blk", "type": "subgraph:ghost", "position": {"x": 1, "y": 0},
         "data": {}},
    ]
    edges = [{"id": "t", "source": "start", "target": "blk",
              "sourceHandle": "trigger", "targetHandle": "__trigger",
              "type": "trigger"}]
    errors = validate_graph(nodes, edges, subgraphs=[])
    assert any("Unknown subgraph: ghost" in e for e in errors)
    assert not any("Unknown node type" in e for e in errors)


def test_validate_type_checks_the_inner_port_a_boundary_stands_for():
    """A boundary port is only a rename: the inner port's type still rules.

    TextInput.text is STRING and ScalarMultiply.tensor is TENSOR. The
    instance's own ``in`` port declares no type at all, so the only way this
    can be caught is by looking through the boundary.
    """
    definition = _passthrough_subgraph()
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "txt", "type": "TextInput", "position": {"x": 1, "y": 0},
         "data": {"params": {"text": "hi"}}},
        {"id": "blk", "type": "subgraph:pass", "position": {"x": 2, "y": 0},
         "data": {}},
    ]
    edges = [
        {"id": "t", "source": "start", "target": "txt",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        {"id": "e", "source": "txt", "target": "blk",
         "sourceHandle": "text", "targetHandle": "in", "type": "data"},
    ]
    errors = validate_graph(nodes, edges, subgraphs=[definition])
    assert any("Type mismatch" in e and "ScalarMultiply" in e for e in errors), (
        errors
    )
    # Without the definitions the instance is opaque and nothing is checked --
    # which is exactly the weaker behaviour a preset node gets.
    assert not any(
        "Type mismatch" in e for e in validate_graph(nodes, edges)
    )


def test_validate_sees_a_required_input_left_unconnected_inside_a_definition():
    """An inner node missing an input is reported by its namespaced id."""
    definition = _passthrough_subgraph()
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "blk", "type": "subgraph:pass", "position": {"x": 1, "y": 0},
         "data": {}},
    ]
    edges = [{"id": "t", "source": "start", "target": "blk",
              "sourceHandle": "trigger", "targetHandle": "__trigger",
              "type": "trigger"}]
    errors = validate_graph(nodes, edges, subgraphs=[definition])
    assert any(
        "Missing required input 'tensor' on node blk/mul" in e for e in errors
    ), errors


def test_bypassing_a_subgraph_instance_is_refused_by_name():
    definition = _passthrough_subgraph()
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "blk", "type": "subgraph:pass", "position": {"x": 1, "y": 0},
         "data": {"bypassed": True}},
    ]
    with pytest.raises(GraphValidationError) as excinfo:
        prepare_executable_graph(nodes, [], subgraphs=[definition])
    assert "Bypass is not supported on subgraph instance(s): blk" in str(
        excinfo.value
    )


# ── Shared definition, two instances ────────────────────────────────────


def test_two_instances_expand_from_one_definition():
    """Editing the definition changes both instances -- the reuse win.

    The two instances are built from ONE definition object and then the
    definition is edited, so what is being pinned is that expansion reads the
    definition at expansion time rather than a copy taken per instance.
    """
    definition = _passthrough_subgraph()
    nodes = [
        {"id": "one", "type": "subgraph:pass", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "two", "type": "subgraph:pass", "position": {"x": 1, "y": 0},
         "data": {}},
    ]
    index = build_subgraph_index([definition])
    out_nodes, _, _ = expand_subgraphs(nodes, [], index)
    scalars = {n["id"]: n["data"]["params"]["scalar"] for n in out_nodes}
    assert scalars == {"one/mul": 2.0, "two/mul": 2.0}

    edited = copy.deepcopy(definition)
    edited["nodes"][0]["data"]["params"]["scalar"] = 9.0
    out_nodes, _, _ = expand_subgraphs(
        nodes, [], build_subgraph_index([edited])
    )
    scalars = {n["id"]: n["data"]["params"]["scalar"] for n in out_nodes}
    assert scalars == {"one/mul": 9.0, "two/mul": 9.0}


# ── The acceptance criterion: identical execution results ───────────────


def _mean_value(results: dict) -> float:
    tensor = results["avg"]["tensor"]
    return float(tensor.reshape(-1)[0])


async def test_collapsed_graph_computes_exactly_what_the_flat_one_does():
    """Criterion 1: collapse -> run -> expand round-trips identically.

    Proved by RUNNING both graphs. A structural comparison would only show
    that the two shapes match, which is not the claim being made.
    """
    flat_nodes, flat_edges = _flat_graph()
    flat = await execute_graph(flat_nodes, flat_edges)

    col_nodes, col_edges, subgraphs = _collapsed_graph()
    collapsed = await execute_graph(col_nodes, col_edges, subgraphs=subgraphs)

    # 2*5 + 3*7 = 31, averaged over a row of 2 equal values -> 31.
    assert _mean_value(flat) == pytest.approx(31.0)
    assert _mean_value(collapsed) == _mean_value(flat)

    # And the collapsed run really did go through the boundary: its inner
    # nodes are present under namespaced ids, the flat ones are not.
    assert "blk/sum" in collapsed
    assert "sum" not in collapsed
    assert "sum" in flat


async def test_editing_the_shared_definition_changes_every_instance_run():
    """Criterion 2, at the level that matters: the numbers change."""
    definition = _passthrough_subgraph()
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "src", "type": "TensorCreate", "position": {"x": 1, "y": 0},
         "data": {"params": {"shape": "1,2", "fill": "full", "value": 1.0}}},
        {"id": "one", "type": "subgraph:pass", "position": {"x": 2, "y": 0},
         "data": {}},
        {"id": "two", "type": "subgraph:pass", "position": {"x": 3, "y": 0},
         "data": {}},
    ]
    edges = [
        {"id": "t", "source": "start", "target": "src",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        {"id": "e1", "source": "src", "target": "one",
         "sourceHandle": "tensor", "targetHandle": "in", "type": "data"},
        {"id": "e2", "source": "one", "target": "two",
         "sourceHandle": "out", "targetHandle": "in", "type": "data"},
    ]
    before = await execute_graph(nodes, edges, subgraphs=[definition])
    assert float(before["two/mul"]["tensor"].reshape(-1)[0]) == pytest.approx(
        4.0  # 1 * 2 * 2
    )

    edited = copy.deepcopy(definition)
    edited["nodes"][0]["data"]["params"]["scalar"] = 3.0
    after = await execute_graph(nodes, edges, subgraphs=[edited])
    assert float(after["two/mul"]["tensor"].reshape(-1)[0]) == pytest.approx(
        9.0  # 1 * 3 * 3 -- BOTH instances picked the edit up
    )


async def test_status_events_roll_up_to_the_instance_node():
    """Spec item 4: the canvas sees the instance, never its internals."""
    col_nodes, col_edges, subgraphs = _collapsed_graph()
    seen: list[tuple[str, str]] = []

    async def on_progress(node_id, status, data):
        seen.append((node_id, status))

    await execute_graph(
        col_nodes, col_edges, on_progress=on_progress, subgraphs=subgraphs
    )
    reported = {node_id for node_id, _ in seen}
    assert not any("/" in node_id for node_id in reported), reported
    assert ("blk", "running") in seen
    assert ("blk", "completed") in seen
    # One running and one completed for the whole block, not three of each.
    assert [s for n, s in seen if n == "blk"].count("completed") == 1


# ── Review follow-ups (PR #198) ─────────────────────────────────────────
#
# Everything below pins a claim the first cut of this feature MADE but did
# not honour: that nesting changes nothing for the status roll-up, that the
# separator cannot collide, that an unreferenced definition is inert, and
# that validation and execution agree about what is legal.


def _nested_definitions() -> list[dict]:
    """``outer`` holds one instance of ``inner``; ``inner`` is one node.

    Two boundaries deep is the smallest shape that tells a single-level
    lookup apart from a transitive one: the container map becomes a CHAIN
    (``blk/nest/mul -> blk/nest -> blk``) whose only link naming a node the
    canvas actually shows is the last one.
    """
    inner = _passthrough_subgraph("inner")
    outer = {
        "id": "outer",
        "name": "outer",
        "nodes": [
            {"id": "nest", "type": "subgraph:inner",
             "position": {"x": 0, "y": 0}, "data": {}},
        ],
        "edges": [],
        "interface": {
            "inputs": [{"port": "in", "innerNode": "nest", "innerPort": "in"}],
            "outputs": [{"port": "out", "innerNode": "nest",
                         "innerPort": "out"}],
            "triggerTargets": ["nest"],
        },
    }
    return [inner, outer]


def _nested_instance_graph() -> tuple[list[dict], list[dict], list[dict]]:
    """start -> src -> [ blk = outer( nest = inner( mul ) ) ]."""
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "src", "type": "TensorCreate", "position": {"x": 1, "y": 0},
         "data": {"params": {"shape": "1,2", "fill": "full", "value": 1.0}}},
        {"id": "blk", "type": "subgraph:outer", "position": {"x": 2, "y": 0},
         "data": {}},
    ]
    edges = [
        {"id": "t1", "source": "start", "target": "src",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        {"id": "t2", "source": "start", "target": "blk",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        {"id": "e", "source": "src", "target": "blk",
         "sourceHandle": "tensor", "targetHandle": "in", "type": "data"},
    ]
    return nodes, edges, _nested_definitions()


def test_outermost_container_walks_the_whole_chain():
    """The container map is a CHAIN, not a lookup table.

    ``blk`` is the only id in this map that the canvas has ever seen, so it
    is the only answer a status event may carry -- from any depth.
    """
    mapping = {"blk/nest": "blk", "blk/nest/mul": "blk/nest"}
    assert outermost_container("blk/nest/mul", mapping) == "blk"
    assert outermost_container("blk/nest", mapping) == "blk"
    assert outermost_container("start", mapping) is None
    # An instance whose OWN id contains the separator resolves through the
    # map, never by splitting the string.
    assert outermost_container(
        "we/ird/mul", {"we/ird/mul": "we/ird"}
    ) == "we/ird"


def test_outermost_container_cannot_loop_forever():
    """A malformed map must not hang the run that reads it."""
    assert outermost_container("a", {"a": "a"}) == "a"
    assert outermost_container("a", {"a": "b", "b": "a"}) in {"a", "b"}


async def test_nested_instance_status_rolls_up_to_the_outermost_container():
    """A two-deep instance must report as the box the canvas actually shows.

    ``blk/nest`` is an id no client has ever seen: expansion invented it and
    expansion consumed it again, so a status keyed to it updates nothing and
    the box the user is watching sits at ``idle`` for the whole run.
    """
    nodes, edges, subgraphs = _nested_instance_graph()
    seen: list[tuple[str, str]] = []

    async def on_progress(node_id, status, data):
        seen.append((node_id, status))

    await execute_graph(
        nodes, edges, on_progress=on_progress, subgraphs=subgraphs
    )
    reported = sorted({node_id for node_id, _ in seen})
    assert not any("/" in node_id for node_id in reported), reported
    assert ("blk", "running") in seen, seen
    assert ("blk", "completed") in seen, seen
    assert [s for n, s in seen if n == "blk"].count("completed") == 1, seen


async def test_roll_up_still_works_for_an_instance_whose_id_has_a_separator():
    """Single level, but the INSTANCE id contains ``/``.

    The resolver walks the container map, never the id string, so an id that
    happens to look namespaced is not mistaken for one.
    """
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "src", "type": "TensorCreate", "position": {"x": 1, "y": 0},
         "data": {"params": {"shape": "1,2", "fill": "full", "value": 1.0}}},
        {"id": "we/ird", "type": "subgraph:pass", "position": {"x": 2, "y": 0},
         "data": {}},
    ]
    edges = [
        {"id": "t", "source": "start", "target": "src",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        {"id": "e", "source": "src", "target": "we/ird",
         "sourceHandle": "tensor", "targetHandle": "in", "type": "data"},
    ]
    seen: list[tuple[str, str]] = []

    async def on_progress(node_id, status, data):
        seen.append((node_id, status))

    await execute_graph(
        nodes, edges, on_progress=on_progress,
        subgraphs=[_passthrough_subgraph()],
    )
    assert ("we/ird", "running") in seen, seen
    assert ("we/ird", "completed") in seen, seen
    assert not any(n == "we/ird/mul" for n, _ in seen), seen


def _block_with_an_untriggered_sibling(nested: bool) -> list[dict]:
    """A block holding a triggered node and a root nothing triggers.

    ``side`` is exactly the shape the retention rule exists for -- the Dataset
    sitting beside the Loss in a training preset: a root with no incoming
    edge, pulled into the run only because something else in its block is.
    """
    inner = _passthrough_subgraph("inner")
    if nested:
        body = [
            {"id": "nest", "type": "subgraph:inner",
             "position": {"x": 0, "y": 0}, "data": {}},
            {"id": "side", "type": "TensorCreate", "position": {"x": 0, "y": 1},
             "data": {"params": {"shape": "1,2"}}},
        ]
        ports = ("nest", "in", "out")
    else:
        body = [
            {"id": "mul", "type": "ScalarMultiply", "position": {"x": 0, "y": 0},
             "data": {"params": {"scalar": 2.0}}},
            {"id": "side", "type": "TensorCreate", "position": {"x": 0, "y": 1},
             "data": {"params": {"shape": "1,2"}}},
        ]
        ports = ("mul", "tensor", "tensor")
    outer = {
        "id": "outer", "name": "outer", "nodes": body, "edges": [],
        "interface": {
            "inputs": [{"port": "in", "innerNode": ports[0],
                        "innerPort": ports[1]}],
            "outputs": [{"port": "out", "innerNode": ports[0],
                         "innerPort": ports[2]}],
            "triggerTargets": [ports[0]],
        },
    }
    return [inner, outer] if nested else [outer]


def _block_instance_graph() -> tuple[list[dict], list[dict]]:
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "src", "type": "TensorCreate", "position": {"x": 1, "y": 0},
         "data": {"params": {"shape": "1,2", "fill": "full", "value": 1.0}}},
        {"id": "blk", "type": "subgraph:outer", "position": {"x": 2, "y": 0},
         "data": {}},
    ]
    edges = [
        {"id": "t1", "source": "start", "target": "src",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        {"id": "t2", "source": "start", "target": "blk",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        {"id": "e", "source": "src", "target": "blk",
         "sourceHandle": "tensor", "targetHandle": "in", "type": "data"},
    ]
    return nodes, edges


def test_one_level_block_retains_a_sibling_nothing_triggers():
    """The control: this is the behaviour nesting has to match."""
    nodes, edges = _block_instance_graph()
    exec_nodes, _e, _m = prepare_executable_graph(
        nodes, edges, subgraphs=_block_with_an_untriggered_sibling(False)
    )
    assert "blk/side" in {n["id"] for n in exec_nodes}


def test_a_nested_block_retains_its_sibling_the_same_way():
    """Same block, one boundary deeper -- and it must not change the run.

    Retention groups by container, and the container map is a chain: the only
    reachable node here is ``blk/nest/mul``, whose IMMEDIATE container is
    ``blk/nest``. Grouping there retains ``nest``'s own contents and prunes
    ``blk/side``, which sits directly inside ``blk`` -- so how deeply the user
    happened to nest a block decides whether its roots survive.
    """
    nodes, edges = _block_instance_graph()
    exec_nodes, _e, _m = prepare_executable_graph(
        nodes, edges, subgraphs=_block_with_an_untriggered_sibling(True)
    )
    assert "blk/side" in {n["id"] for n in exec_nodes}


# ── Id collisions across the separator ──────────────────────────────────


def test_two_instances_whose_namespaced_ids_collide_are_refused():
    """``x`` + inner ``a/b`` and ``x/a`` + inner ``b`` both make ``x/a/b``.

    Silently keeping one of the two is the worst outcome available: the run
    then dies inside ``topological_levels`` naming a cycle that does not
    exist, because its node count no longer matches the graph's.
    """
    wrap = {
        "id": "wrap", "name": "wrap", "edges": [],
        "nodes": [{"id": "a/b", "type": "ScalarMultiply",
                   "position": {"x": 0, "y": 0}, "data": {"params": {}}}],
        "interface": {"inputs": [], "outputs": [], "triggerTargets": []},
    }
    plain = {
        "id": "plain", "name": "plain", "edges": [],
        "nodes": [{"id": "b", "type": "ScalarMultiply",
                   "position": {"x": 0, "y": 0}, "data": {"params": {}}}],
        "interface": {"inputs": [], "outputs": [], "triggerTargets": []},
    }
    nodes = [
        {"id": "x", "type": "subgraph:wrap", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "x/a", "type": "subgraph:plain", "position": {"x": 1, "y": 0},
         "data": {}},
    ]
    with pytest.raises(GraphValidationError) as excinfo:
        expand_subgraphs(nodes, [], build_subgraph_index([wrap, plain]))
    message = str(excinfo.value)
    assert "x/a/b" in message
    # BOTH contributors, or the user cannot tell which of the two to rename.
    assert "'x'" in message and "'x/a'" in message, message


def test_an_inner_id_colliding_with_a_top_level_node_is_refused():
    nodes = [
        {"id": "blk", "type": "subgraph:pass", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "blk/mul", "type": "ScalarMultiply",
         "position": {"x": 1, "y": 0}, "data": {"params": {}}},
    ]
    with pytest.raises(GraphValidationError) as excinfo:
        expand_subgraphs(
            nodes, [], build_subgraph_index([_passthrough_subgraph()])
        )
    message = str(excinfo.value)
    assert "blk/mul" in message
    assert "'blk'" in message, message


def test_a_graph_that_carries_one_id_twice_is_not_blamed_on_the_boundary():
    """Claiming every expanded id also catches a plain duplicate.

    It must not be described as a flattening collision -- there is no
    boundary involved and the user would go looking at the wrong thing.
    """
    nodes = [
        {"id": "dup", "type": "ScalarMultiply", "position": {"x": 0, "y": 0},
         "data": {"params": {}}},
        {"id": "dup", "type": "ScalarMultiply", "position": {"x": 1, "y": 0},
         "data": {"params": {}}},
        {"id": "blk", "type": "subgraph:pass", "position": {"x": 2, "y": 0},
         "data": {}},
    ]
    with pytest.raises(GraphValidationError) as excinfo:
        expand_subgraphs(
            nodes, [], build_subgraph_index([_passthrough_subgraph()])
        )
    message = str(excinfo.value)
    assert "'dup'" in message
    assert "after subgraph expansion" not in message, message


def test_validate_reports_an_id_collision_instead_of_a_phantom_cycle():
    """The route must name the real fault, not the symptom it becomes."""
    wrap = {
        "id": "wrap", "name": "wrap", "edges": [],
        "nodes": [{"id": "a/b", "type": "ScalarMultiply",
                   "position": {"x": 0, "y": 0}, "data": {"params": {}}}],
        "interface": {"inputs": [], "outputs": [], "triggerTargets": []},
    }
    plain = {
        "id": "plain", "name": "plain", "edges": [],
        "nodes": [{"id": "b", "type": "ScalarMultiply",
                   "position": {"x": 0, "y": 0}, "data": {"params": {}}}],
        "interface": {"inputs": [], "outputs": [], "triggerTargets": []},
    }
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "x", "type": "subgraph:wrap", "position": {"x": 1, "y": 0},
         "data": {}},
        {"id": "x/a", "type": "subgraph:plain", "position": {"x": 2, "y": 0},
         "data": {}},
    ]
    edges = [{"id": "t", "source": "start", "target": "x",
              "sourceHandle": "trigger", "targetHandle": "__trigger",
              "type": "trigger"}]
    errors = validate_graph(nodes, edges, subgraphs=[wrap, plain])
    # Not merely "some error mentions x/a/b" -- the collapsed graph produces a
    # "Missing required input" on that id either way. The collision itself has
    # to be named, with both instances.
    assert any(
        "x/a/b" in e and "'x'" in e and "'x/a'" in e for e in errors
    ), errors
    assert not any("cycle" in e for e in errors), errors


# ── Unreferenced definitions really are inert ───────────────────────────


def test_a_stale_recursive_definition_cannot_break_a_graph_that_has_instances():
    """The no-instance early return is not the interesting case.

    A graph that uses ONE subgraph and still carries a definition nobody
    instantiates must run. Scanning every definition in the index for
    recursion turns dead data into a refused run.
    """
    nodes, edges, subgraphs = _collapsed_graph()
    stale = {"id": "selfy", "name": "selfy", "edges": [],
             "nodes": [{"id": "me", "type": "subgraph:selfy",
                        "position": {"x": 0, "y": 0}, "data": {}}],
             "interface": {"inputs": [], "outputs": [], "triggerTargets": []}}
    out_nodes, _, mapping = expand_subgraphs_deep(
        nodes, edges, build_subgraph_index(subgraphs + [stale])
    )
    assert {"blk/m1", "blk/m2", "blk/sum"} <= {n["id"] for n in out_nodes}
    assert mapping == {"blk/m1": "blk", "blk/m2": "blk", "blk/sum": "blk"}


def test_a_self_recursive_definition_that_is_used_is_still_refused_by_name():
    """Scoping the scan must not stop it catching the real thing."""
    selfy = {"id": "selfy", "name": "selfy", "edges": [],
             "nodes": [{"id": "me", "type": "subgraph:selfy",
                        "position": {"x": 0, "y": 0}, "data": {}}],
             "interface": {"inputs": [], "outputs": [], "triggerTargets": []}}
    nodes = [{"id": "blk", "type": "subgraph:selfy",
              "position": {"x": 0, "y": 0}, "data": {}}]
    with pytest.raises(GraphValidationError) as excinfo:
        expand_subgraphs_deep(nodes, [], build_subgraph_index([selfy]))
    assert "selfy -> selfy" in str(excinfo.value)


# ── Honest messages for degenerate definitions ──────────────────────────


def test_a_subgraph_type_with_an_empty_id_is_reported():
    """``subgraph:`` is not a legitimate node.

    ``subgraph_id_of`` returns ``""`` for it, which is falsey -- so a
    truthiness test skips expansion while an ``is not None`` test treats the
    node as a valid opaque container. The graph then validates clean and
    fails at run time.
    """
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "x", "type": "subgraph:", "position": {"x": 1, "y": 0},
         "data": {}},
    ]
    edges = [{"id": "t", "source": "start", "target": "x",
              "sourceHandle": "trigger", "targetHandle": "__trigger",
              "type": "trigger"}]
    errors = validate_graph(nodes, edges, subgraphs=[])
    assert errors, "an empty subgraph id validated clean"
    assert any("x" in e for e in errors), errors
    # "Unknown subgraph:  (node x)" -- a blank where the id should be -- is
    # not a message anyone can act on.
    assert not any("Unknown subgraph:  " in e for e in errors), errors


def _malformed_definition(sid: str = "d") -> dict:
    """One inner node whose position is not a point. Fails model validation."""
    return {
        "id": sid, "name": sid, "edges": [],
        "nodes": [{"id": "mul", "type": "ScalarMultiply",
                   "position": {"x": "nope", "y": 0},
                   "data": {"params": {}}}],
        "interface": {"inputs": [], "outputs": [], "triggerTargets": []},
    }


def test_a_referenced_malformed_definition_says_so_instead_of_unknown():
    """"Unknown subgraph: d" is a lie when ``d`` is right there in the file.

    The user goes looking for a missing definition, finds it, and has no way
    to learn that the reason it was dropped was one bad field inside it.
    """
    nodes = [{"id": "blk", "type": "subgraph:d",
              "position": {"x": 0, "y": 0}, "data": {}}]
    index = build_subgraph_index([_malformed_definition()])
    with pytest.raises(GraphValidationError) as excinfo:
        expand_subgraphs(nodes, [], index)
    message = str(excinfo.value)
    assert "Unknown subgraph" not in message, message
    assert "blk" in message and "d" in message
    assert "position" in message, message  # the reason, quoted


def test_a_malformed_definition_nobody_uses_still_cannot_break_a_run():
    """The skip is kept; only the diagnosis changes."""
    nodes, edges = _flat_graph()
    out_nodes, _, mapping = expand_subgraphs_deep(
        nodes, edges, build_subgraph_index([_malformed_definition()])
    )
    assert out_nodes is nodes and mapping == {}


# ── Validation and execution must agree ─────────────────────────────────


def test_validate_reports_bypass_on_a_subgraph_instance():
    """The route must refuse what the runner refuses.

    Expansion runs before bypass resolution, so by the time bypass is
    considered the instance node -- and its ``bypassed`` flag -- are gone.
    The graph validates clean and then the run dies.
    """
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "blk", "type": "subgraph:pass", "position": {"x": 1, "y": 0},
         "data": {"bypassed": True}},
    ]
    edges = [{"id": "t", "source": "start", "target": "blk",
              "sourceHandle": "trigger", "targetHandle": "__trigger",
              "type": "trigger"}]
    errors = validate_graph(
        nodes, edges, subgraphs=[_passthrough_subgraph()]
    )
    assert any(
        "Bypass is not supported on subgraph instance(s): blk" in e
        for e in errors
    ), errors


def test_validate_reports_bypass_on_a_preset_node_exactly_once():
    """The preset half of this hole never existed -- and must not be opened.

    ``validate_graph`` does NOT expand preset nodes, so a bypassed one is
    still standing when ``resolve_bypass`` runs and is named there. That is
    the whole reason ``container_bypass_errors`` takes an
    ``include_presets`` switch: adding the pre-expansion check unconditionally
    would report the same fault twice on this path.
    """
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "p", "type": "preset:whatever", "position": {"x": 1, "y": 0},
         "data": {"bypassed": True}},
    ]
    edges = [{"id": "t", "source": "start", "target": "p",
              "sourceHandle": "trigger", "targetHandle": "__trigger",
              "type": "trigger"}]
    errors = validate_graph(nodes, edges)
    bypass_errors = [e for e in errors if "Bypass is not supported" in e]
    assert bypass_errors == ["Bypass is not supported on preset node p"], errors


def test_expanded_graph_does_not_re_report_a_container_bypass():
    """``prepare_executable_graph`` validates AFTER expansion.

    By then no container node is left, so the shared check must contribute
    nothing rather than repeating itself.
    """
    col_nodes, col_edges, subgraphs = _collapsed_graph()
    _nodes, _edges, _map = prepare_executable_graph(
        col_nodes, col_edges, subgraphs=subgraphs
    )
    assert not any("Bypass is not supported" in e
                   for e in validate_graph(_nodes, _edges))


# ── Cycles below more than one boundary ─────────────────────────────────


def test_describe_cycle_names_every_enclosing_instance():
    text = describe_cycle(["blk/nest/p", "blk/nest/q", "blk/nest/p"])
    assert "crosses subgraph instance(s): blk, blk/nest" in text, text


def test_a_cycle_inside_a_nested_definition_names_both_boundaries():
    loopy = {
        "id": "loopy", "name": "loopy",
        "nodes": [
            {"id": "p", "type": "ScalarMultiply", "position": {"x": 0, "y": 0},
             "data": {"params": {}}},
            {"id": "q", "type": "ScalarMultiply", "position": {"x": 1, "y": 0},
             "data": {"params": {}}},
        ],
        "edges": [
            {"id": "f", "source": "p", "target": "q", "sourceHandle": "tensor",
             "targetHandle": "tensor", "type": "data"},
            {"id": "g", "source": "q", "target": "p", "sourceHandle": "tensor",
             "targetHandle": "tensor", "type": "data"},
        ],
        "interface": {
            "inputs": [{"port": "in", "innerNode": "p", "innerPort": "tensor"}],
            "outputs": [{"port": "out", "innerNode": "q",
                         "innerPort": "tensor"}],
            "triggerTargets": ["p"],
        },
    }
    wrap = {
        "id": "wrap", "name": "wrap", "edges": [],
        "nodes": [{"id": "nest", "type": "subgraph:loopy",
                   "position": {"x": 0, "y": 0}, "data": {}}],
        "interface": {
            "inputs": [{"port": "in", "innerNode": "nest", "innerPort": "in"}],
            "outputs": [{"port": "out", "innerNode": "nest",
                         "innerPort": "out"}],
            "triggerTargets": ["nest"],
        },
    }
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "src", "type": "TensorCreate", "position": {"x": 1, "y": 0},
         "data": {"params": {"shape": "2,2"}}},
        {"id": "blk", "type": "subgraph:wrap", "position": {"x": 2, "y": 0},
         "data": {}},
    ]
    edges = [
        {"id": "t", "source": "start", "target": "src",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        {"id": "e", "source": "src", "target": "blk",
         "sourceHandle": "tensor", "targetHandle": "in", "type": "data"},
    ]
    errors = validate_graph(nodes, edges, subgraphs=[loopy, wrap])
    cycle_errors = [e for e in errors if "cycle" in e]
    assert len(cycle_errors) == 1, errors
    assert "crosses subgraph instance(s): blk, blk/nest" in cycle_errors[0], (
        cycle_errors[0]
    )


# ── Round-2 review findings ─────────────────────────────────────────────


def test_two_instances_sharing_an_id_are_refused():
    """The guard claimed inner ids and outer ids, never the INSTANCE id.

    Two instances wearing one id whose definitions have disjoint inner ids
    therefore slipped straight through: the run exited clean with the second
    block's boundary edges consumed by the first, so it contributed nothing
    and nothing said so. A silently wrong answer is strictly worse than the
    phantom cycle the collision guard was written for.
    """
    first = {
        "id": "first", "name": "first", "edges": [],
        "nodes": [{"id": "m", "type": "ScalarMultiply",
                   "position": {"x": 0, "y": 0}, "data": {"params": {}}}],
        "interface": {
            "inputs": [{"port": "in", "innerNode": "m", "innerPort": "tensor"}],
            "outputs": [{"port": "out", "innerNode": "m",
                         "innerPort": "tensor"}],
            "triggerTargets": ["m"],
        },
    }
    second = {
        "id": "second", "name": "second", "edges": [],
        "nodes": [{"id": "n", "type": "ScalarMultiply",
                   "position": {"x": 0, "y": 0}, "data": {"params": {}}}],
        "interface": {
            "inputs": [{"port": "in", "innerNode": "n", "innerPort": "tensor"}],
            "outputs": [{"port": "out", "innerNode": "n",
                         "innerPort": "tensor"}],
            "triggerTargets": ["n"],
        },
    }
    nodes = [
        {"id": "p", "type": "subgraph:first", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "p", "type": "subgraph:second", "position": {"x": 1, "y": 0},
         "data": {}},
    ]
    with pytest.raises(GraphValidationError) as excinfo:
        expand_subgraphs(nodes, [], build_subgraph_index([first, second]))
    message = str(excinfo.value)
    assert "'p'" in message
    # A plain duplicate, not a flattening collision -- no boundary involved.
    assert "after subgraph expansion" not in message, message


def test_an_instance_id_colliding_with_a_plain_node_is_refused():
    nodes = [
        {"id": "p", "type": "ScalarMultiply", "position": {"x": 0, "y": 0},
         "data": {"params": {}}},
        {"id": "p", "type": "subgraph:pass", "position": {"x": 1, "y": 0},
         "data": {}},
    ]
    with pytest.raises(GraphValidationError) as excinfo:
        expand_subgraphs(
            nodes, [], build_subgraph_index([_passthrough_subgraph()])
        )
    assert "'p'" in str(excinfo.value)


def test_a_duplicate_id_is_refused_even_with_no_subgraph_in_the_graph():
    """The claim guard lived inside expansion, which returns early when the
    graph has no instances -- so the identical duplicate was reported with a
    block present and accepted without one."""
    nodes = [
        {"id": "dup", "type": "ScalarMultiply", "position": {"x": 0, "y": 0},
         "data": {"params": {}}},
        {"id": "dup", "type": "ScalarMultiply", "position": {"x": 1, "y": 0},
         "data": {"params": {}}},
    ]
    with pytest.raises(GraphValidationError) as excinfo:
        expand_subgraphs_deep(nodes, [], {})
    assert "'dup'" in str(excinfo.value)


def test_validate_reports_a_duplicate_id_with_no_subgraph_present():
    """Validate and the run have to agree about it, and say it once."""
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "dup", "type": "TensorCreate", "position": {"x": 1, "y": 0},
         "data": {"params": {"shape": "2,2"}}},
        {"id": "dup", "type": "TensorCreate", "position": {"x": 2, "y": 0},
         "data": {"params": {"shape": "2,2"}}},
    ]
    edges = [{"id": "t", "source": "start", "target": "dup",
              "sourceHandle": "trigger", "targetHandle": "__trigger",
              "type": "trigger"}]
    errors = validate_graph(nodes, edges)
    duplicates = [e for e in errors if "Duplicate node id" in e]
    assert len(duplicates) == 1, errors
    assert "'dup'" in duplicates[0]


def test_a_duplicate_id_is_reported_exactly_once_with_a_subgraph_present():
    """The same fault must not be listed twice just because the graph also
    happens to contain a block."""
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "dup", "type": "TensorCreate", "position": {"x": 1, "y": 0},
         "data": {"params": {"shape": "2,2"}}},
        {"id": "dup", "type": "TensorCreate", "position": {"x": 2, "y": 0},
         "data": {"params": {"shape": "2,2"}}},
        {"id": "blk", "type": "subgraph:pass", "position": {"x": 3, "y": 0},
         "data": {}},
    ]
    errors = validate_graph(
        nodes, [], subgraphs=[_passthrough_subgraph()],
    )
    duplicates = [e for e in errors if "Duplicate node id" in e]
    assert len(duplicates) == 1, errors


def _subgraph_holding_preset() -> dict:
    """A portable preset whose internal nodes name a graph-local subgraph."""
    return {
        "preset_name": "HoldsBlock",
        "category": "Test",
        "description": "",
        "tags": [],
        "nodes": [
            {"id": "si", "type": "subgraph:leaf", "params": {}},
        ],
        "edges": [],
        "exposed_inputs": [],
        "exposed_outputs": [],
        "exposed_params": [],
    }


def test_a_preset_naming_a_subgraph_is_refused_by_validate_and_by_the_run():
    """A preset's internals were assumed to be unable to name a subgraph.

    They can: `build_preset_fallback` reads the graph's OWN client-supplied
    `presets[]`. Subgraphs are expanded before presets and never re-expanded,
    so the instance the preset carries reached the executor unexpanded --
    validate said clean, the run died `Unknown subgraph: leaf` for a
    definition sitting right there in the graph. Both surfaces must refuse,
    and by name.
    """
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {}},
        {"id": "pnode", "type": "preset:HoldsBlock",
         "position": {"x": 1, "y": 0}, "data": {"params": {}}},
    ]
    edges = [{"id": "t", "source": "start", "target": "pnode",
              "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"}]
    leaf = _passthrough_subgraph("leaf")
    fallback = build_preset_fallback([_subgraph_holding_preset()])

    errors = validate_graph(
        nodes, edges, preset_fallback=fallback, subgraphs=[leaf],
    )
    named = [e for e in errors if "HoldsBlock" in e]
    assert named, errors

    with pytest.raises(GraphValidationError) as excinfo:
        prepare_executable_graph(
            nodes, edges, preset_fallback=fallback, subgraphs=[leaf],
        )
    assert "HoldsBlock" in str(excinfo.value)
