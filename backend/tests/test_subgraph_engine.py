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
    build_subgraph_index,
    describe_cycle,
    execute_graph,
    expand_subgraphs,
    expand_subgraphs_deep,
    find_cycle,
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
