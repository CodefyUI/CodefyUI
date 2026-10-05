"""A trigger wired into a preset card starts the card (#561).

Preset expansion remapped an edge into a card by its handle alone. A trigger
edge carries ``__trigger`` (the canvas) or ``""``, never an exposed input, so
after expansion it still named the card, which no longer existed. Start into
a card alone was refused with "Graph has no entry points"; beside another
trigger, the card silently did not run. Validation never expands a card, so
it called both graphs clean.

A trigger into a card now starts the card's inner roots, the nodes nothing
inside it feeds, the way a trigger into a block that names no trigger targets
does. What expansion cannot do is refused, in the same sentence by
validation, a run and an export: a triggered card with no node to start, an
edge on a port the card does not expose, and a trigger edge whose node is not
in the graph.

Graphs and presets are synthetic and written by the tests.
"""

from __future__ import annotations

import pytest

from app.core.api_contract import check_wiring, collect_outputs, derive_contract
from app.core.execution_context import ExecutionContext
from app.core.graph_engine import (
    GraphValidationError,
    build_preset_fallback,
    execute_graph,
    expand_presets,
    prepare_executable_graph,
    validate_graph,
)

CARD = "T7 Say Hi"
TWO_ROOTS = "T7 Two Roots"
EMPTY = "T7 Nothing Inside"
LOOP = "T7 Feeds Itself"
OUTER = "T7 Card In A Card"


# -- Graphs -----------------------------------------------------------------


def _node(node_id: str, node_type: str, params: dict | None = None, **data) -> dict:
    return {
        "id": node_id,
        "type": node_type,
        "position": {"x": 0, "y": 0},
        "data": {"params": dict(params or {}), **data},
    }


def _card(node_id: str, preset_name: str) -> dict:
    """A preset card the way the palette drops one."""
    return _node(node_id, f"preset:{preset_name}", internalParams={})


def _trigger(source: str, target: str) -> dict:
    """Start's trigger the way the canvas saves it."""
    return {
        "id": f"t-{source}-{target}",
        "source": source,
        "target": target,
        "sourceHandle": "trigger",
        "targetHandle": "__trigger",
        "type": "trigger",
    }


def _wire(source: str, source_port: str, target: str, target_port: str) -> dict:
    return {
        "id": f"{source}.{source_port}-{target}.{target_port}",
        "source": source,
        "target": target,
        "sourceHandle": source_port,
        "targetHandle": target_port,
        "type": "data",
    }


def _preset(name: str, nodes: list[dict], edges: list[dict],
            outputs: tuple[tuple[str, str, str], ...] = ()) -> dict:
    """``outputs`` are ``(exposed name, internal node, internal port)``."""
    return {
        "preset_name": name,
        "category": "Custom",
        "description": "",
        "tags": [],
        "nodes": nodes,
        "edges": [
            {"source": s, "sourceHandle": sp, "target": t, "targetHandle": tp}
            for s, sp, t, tp in edges
        ],
        "exposed_inputs": [],
        "exposed_outputs": [
            {"name": exposed, "internal_node": inner, "internal_port": port,
             "data_type": "ANY", "description": ""}
            for exposed, inner, port in outputs
        ],
        "exposed_params": [],
    }


def _say_hi() -> dict:
    """A TextInput saying "hi" into a Print, whose value is the card's ``out``."""
    return _preset(
        CARD,
        nodes=[{"id": "a", "type": "TextInput", "params": {"value": "hi"}},
               {"id": "b", "type": "Print", "params": {"label": "card"}}],
        edges=[("a", "text", "b", "value")],
        outputs=(("out", "b", "value"),),
    )


def _two_roots() -> dict:
    """Two TextInputs nothing inside feeds, one of them shown by a Print."""
    return _preset(
        TWO_ROOTS,
        nodes=[{"id": "a", "type": "TextInput", "params": {"value": "one"}},
               {"id": "b", "type": "TextInput", "params": {"value": "two"}},
               {"id": "p", "type": "Print", "params": {}}],
        edges=[("a", "text", "p", "value")],
    )


def _nothing_inside() -> dict:
    return _preset(EMPTY, nodes=[], edges=[])


def _feeds_itself() -> dict:
    """Two Prints, each feeding the other: no node inside is a root."""
    return _preset(
        LOOP,
        nodes=[{"id": "p", "type": "Print", "params": {}},
               {"id": "q", "type": "Print", "params": {}}],
        edges=[("p", "value", "q", "value"), ("q", "value", "p", "value")],
    )


def _card_in_a_card() -> dict:
    """A card of :func:`_say_hi` and a Print it feeds."""
    return _preset(
        OUTER,
        nodes=[{"id": "ic", "type": f"preset:{CARD}", "params": {}},
               {"id": "tail", "type": "Print", "params": {"label": "outer"}}],
        edges=[("ic", "out", "tail", "value")],
    )


def _graph(nodes: list[dict], edges: list[dict], presets: list[dict],
           subgraphs: list[dict] | None = None) -> dict:
    return {"name": "t7-graph", "nodes": nodes, "edges": edges,
            "presets": presets, "subgraphs": subgraphs or []}


def _block(block_id: str, nodes: list[dict], edges: list[dict]) -> dict:
    """A block with no boundary ports and no trigger targets: a trigger into
    an instance starts the block's inner roots."""
    return {
        "id": block_id,
        "name": block_id,
        "description": "",
        "nodes": nodes,
        "edges": edges,
        "interface": {"inputs": [], "outputs": [], "triggerTargets": []},
    }


# -- The three surfaces -----------------------------------------------------


def _validate(graph: dict) -> list[str]:
    return validate_graph(
        graph["nodes"], graph["edges"],
        preset_fallback=build_preset_fallback(graph["presets"]),
        subgraphs=graph["subgraphs"],
    )


def _prepare(graph: dict):
    return prepare_executable_graph(
        graph["nodes"], graph["edges"],
        preset_fallback=build_preset_fallback(graph["presets"]),
        subgraphs=graph["subgraphs"],
    )


async def _run(graph: dict) -> dict:
    return await execute_graph(
        graph["nodes"], graph["edges"],
        context=ExecutionContext(device="cpu", weights_persistent=False,
                                 graph_id="t7-engine"),
        preset_fallback=build_preset_fallback(graph["presets"]),
        subgraphs=graph["subgraphs"],
    )


async def _refused_everywhere(client, graph: dict, sentence: str, *,
                              whole: bool = True) -> None:
    """Validation lists the line once; a run and an export refuse with it alone.

    ``whole=False`` takes *sentence* as the start of the line, and the rest
    is read from validation's own line.
    """
    errors = _validate(graph)
    found = [str(error) for error in errors
             if (error == sentence if whole else error.startswith(sentence))]
    assert len(found) == 1, errors
    line = found[0]
    validated = await client.post("/api/graph/validate", json=graph)
    assert validated.status_code == 200, validated.text
    assert line in validated.json()["errors"], validated.json()

    with pytest.raises(GraphValidationError) as refused:
        _prepare(graph)
    assert str(refused.value) == line

    exported = await client.post("/api/graph/export", json=graph)
    assert exported.status_code == 400, exported.text
    assert exported.json()["detail"] == line


# -- A trigger into a card starts it -----------------------------------------


def test_a_trigger_into_a_card_becomes_one_trigger_per_inner_root():
    nodes = [_node("start", "Start"), _card("c", TWO_ROOTS)]
    edges = [_trigger("start", "c")]

    _nodes, expanded, _mapping = expand_presets(
        nodes, edges, preset_fallback=build_preset_fallback([_two_roots()]))

    triggers = [edge for edge in expanded if edge.get("type") == "trigger"]
    assert [(edge["id"], edge["source"], edge["target"]) for edge in triggers] == [
        ("t-start-c#0", "start", "c__a"),
        ("t-start-c#1", "start", "c__b"),
    ]
    # The handle rides along, as a block's fan-out keeps it.
    assert {edge["targetHandle"] for edge in triggers} == {"__trigger"}
    assert not [edge for edge in expanded if "c" in (edge["source"], edge["target"])]


async def test_start_into_a_card_alone_runs_the_card():
    graph = _graph([_node("start", "Start"), _card("c", CARD)],
                   [_trigger("start", "c")], [_say_hi()])

    assert _validate(graph) == []
    executable, _edges, _mapping = _prepare(graph)
    assert sorted(node["id"] for node in executable) == ["c__a", "c__b", "start"]
    results = await _run(graph)
    assert results["c__b"]["value"] == "hi"


async def test_beside_another_trigger_the_card_still_runs():
    graph = _graph(
        [_node("start", "Start"), _card("c", CARD),
         _node("x", "TextInput", {"value": "q"})],
        [_trigger("start", "c"), _trigger("start", "x")],
        [_say_hi()],
    )

    executable, _edges, _mapping = _prepare(graph)
    assert sorted(node["id"] for node in executable) == [
        "c__a", "c__b", "start", "x"]
    results = await _run(graph)
    assert results["c__b"]["value"] == "hi"
    assert results["x"]["text"] == "q"


async def test_a_card_feeding_a_graph_output_is_wired_and_answers():
    graph = _graph(
        [_node("start", "Start"), _card("c", CARD),
         _node("o", "GraphOutput", {"name": "said", "description": ""})],
        [_trigger("start", "c"), _wire("c", "out", "o", "value")],
        [_say_hi()],
    )
    contract = derive_contract(graph["nodes"])
    wiring = check_wiring(graph["nodes"], graph["edges"], contract)
    assert (wiring.untriggered, wiring.unreachable) == ([], [])

    collected, missing = collect_outputs(contract, await _run(graph))

    assert missing == []
    assert collected == {"said": "hi"}


async def test_a_card_holding_a_card_starts_the_innermost_roots():
    """The fanned trigger names the inner card; the next expansion pass fans
    it out again."""
    graph = _graph([_node("start", "Start"), _card("o", OUTER)],
                   [_trigger("start", "o")], [_say_hi(), _card_in_a_card()])

    assert _validate(graph) == []
    executable, edges, _mapping = _prepare(graph)
    assert sorted(node["id"] for node in executable) == [
        "o__ic__a", "o__ic__b", "o__tail", "start"]
    assert [edge["target"] for edge in edges if edge.get("type") == "trigger"] == [
        "o__ic__a"]
    results = await _run(graph)
    assert results["o__tail"]["value"] == "hi"


async def test_an_empty_card_nothing_triggers_is_left_alone():
    """Only a TRIGGERED card needs a node to start."""
    graph = _graph(
        [_node("start", "Start"), _node("x", "TextInput", {"value": "q"}),
         _card("e", EMPTY)],
        [_trigger("start", "x")],
        [_nothing_inside()],
    )

    assert _validate(graph) == []
    executable, _edges, _mapping = _prepare(graph)
    assert sorted(node["id"] for node in executable) == ["start", "x"]


# -- A triggered card with nothing to start -----------------------------------


@pytest.mark.parametrize(
    ("build", "name", "reason"),
    [
        (_nothing_inside, EMPTY, "it has no nodes"),
        (_feeds_itself, LOOP, "every node inside it is fed by another node inside it"),
    ],
    ids=["no nodes", "every node fed"],
)
async def test_a_triggered_card_with_no_node_to_start_is_refused(
    test_client, build, name, reason,
):
    graph = _graph([_node("start", "Start"), _card("c", name)],
                   [_trigger("start", "c")], [build()])

    await _refused_everywhere(
        test_client, graph,
        f"Node c is triggered, but preset '{name}' has no node to start: {reason}",
    )


async def test_validation_reads_a_card_s_edges_before_bypass_as_a_run_does(test_client):
    """A run expands presets before it resolves bypass, so a muted Start's
    trigger still reaches the card there. Validation checks cards at the same
    point; after bypass the trigger is gone and the graph looked clean."""
    graph = _graph(
        [_node("start", "Start", bypassed=True), _card("c", EMPTY),
         _node("go", "Start"), _node("x", "TextInput", {"value": "q"})],
        [_trigger("start", "c"), _trigger("go", "x")],
        [_nothing_inside()],
    )

    await _refused_everywhere(
        test_client, graph,
        f"Node c is triggered, but preset '{EMPTY}' has no node to start: "
        "it has no nodes",
    )


# -- An edge on a port the card does not expose -------------------------------


def _into_an_unexposed_port() -> dict:
    """Two wires onto the one port: one fault, one line."""
    return _graph(
        [_node("start", "Start"), _node("x", "TextInput", {"value": "q"}),
         _node("y", "TextInput", {"value": "r"}), _card("c", CARD)],
        [_trigger("start", "x"), _trigger("start", "y"),
         _wire("x", "text", "c", "nope"), _wire("y", "text", "c", "nope")],
        [_say_hi()],
    )


def _out_of_an_unexposed_port() -> dict:
    return _graph(
        [_node("start", "Start"), _card("c", CARD), _node("p", "Print")],
        [_trigger("start", "c"), _wire("c", "nope", "p", "value")],
        [_say_hi()],
    )


def _trigger_out_of_a_card() -> dict:
    """A card has no trigger output, so a trigger leaving it is an edge on a
    port it does not expose -- the rule for a block."""
    return _graph(
        [_node("start", "Start"), _card("c", CARD),
         _node("x", "TextInput", {"value": "q"})],
        [_trigger("start", "c"), _trigger("c", "x")],
        [_say_hi()],
    )


def _into_an_unexposed_port_in_a_block() -> dict:
    block = _block(
        "blk",
        [_node("x", "TextInput", {"value": "q"}), _card("c", CARD)],
        [_wire("x", "text", "c", "nope")],
    )
    return _graph([_node("start", "Start"), _node("b", "subgraph:blk")],
                  [_trigger("start", "b")], [_say_hi()], [block])


def _out_of_an_unexposed_port_in_a_block() -> dict:
    block = _block(
        "blk",
        [_card("c", CARD), _node("p", "Print")],
        [_wire("c", "nope", "p", "value")],
    )
    return _graph([_node("start", "Start"), _node("b", "subgraph:blk")],
                  [_trigger("start", "b")], [_say_hi()], [block])


@pytest.mark.parametrize(
    ("build", "sentence"),
    [
        (_into_an_unexposed_port,
         f"Edge targets input port 'nope' which preset '{CARD}' does not "
         "expose (node c)"),
        (_out_of_an_unexposed_port,
         f"Edge sources output port 'nope' which preset '{CARD}' does not "
         "expose (node c)"),
        (_trigger_out_of_a_card,
         f"Edge sources output port 'trigger' which preset '{CARD}' does not "
         "expose (node c)"),
        (_into_an_unexposed_port_in_a_block,
         f"Edge targets input port 'nope' which preset '{CARD}' does not "
         "expose (node b/c)"),
        (_out_of_an_unexposed_port_in_a_block,
         f"Edge sources output port 'nope' which preset '{CARD}' does not "
         "expose (node b/c)"),
    ],
    ids=["into", "out of", "trigger out of", "into, in a block",
         "out of, in a block"],
)
async def test_an_edge_on_a_port_the_card_does_not_expose_is_refused(
    test_client, build, sentence,
):
    await _refused_everywhere(test_client, build(), sentence)


# -- A trigger edge whose node is not in the graph ----------------------------
#
# Only the start of these lines is pinned: the missing node they name. The
# advice after " -- " may change.

FROM_GONE = "A trigger edge comes from node 'gone', which is not in the graph -- "
TO_NOWHERE = "A trigger edge goes to node 'nowhere', which is not in the graph -- "


async def test_a_trigger_from_a_node_the_graph_does_not_have_is_refused(test_client):
    """A hand-edited file: the node a trigger came from is gone, its edge is
    not. The edge used to make its target an entry point anyway."""
    graph = _graph(
        [_node("x", "TextInput", {"value": "q"})],
        [{**_trigger("gone", "x"), "targetHandle": ""}],
        [],
    )

    await _refused_everywhere(test_client, graph, FROM_GONE, whole=False)


async def test_a_trigger_from_a_missing_node_into_a_card_is_one_line(test_client):
    """The run fans the edge out to the card's roots; the line names the
    missing node, not an edge id, so it reads the same before and after."""
    graph = _graph([_card("c", TWO_ROOTS)], [_trigger("gone", "c")],
                   [_two_roots()])

    await _refused_everywhere(test_client, graph, FROM_GONE, whole=False)


def _trigger_to_a_missing_node() -> dict:
    return _graph(
        [_node("start", "Start"), _node("x", "TextInput", {"value": "q"})],
        [_trigger("start", "x"), _trigger("start", "nowhere")],
        [],
    )


def test_validation_lists_a_trigger_to_a_node_the_graph_does_not_have():
    errors = _validate(_trigger_to_a_missing_node())
    assert len(errors) == 1 and errors[0].startswith(TO_NOWHERE), errors


async def test_a_run_and_an_export_refuse_a_trigger_to_a_missing_node(test_client):
    """The run prunes an edge whose target does not run before its closing
    validation, so it checks trigger edges on the whole graph first."""
    await _refused_everywhere(
        test_client, _trigger_to_a_missing_node(), TO_NOWHERE, whole=False)
