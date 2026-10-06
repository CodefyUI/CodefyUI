"""A trigger wired into an empty block is refused by name (#604).

A trigger into a block instance fans out to the block's trigger targets or,
when it names none, to its inner roots. A block with no nodes has neither, so
the fan-out produced nothing and the trigger disappeared: beside another
trigger the block silently did not run, and alone the run said "Graph has no
entry points" although Start was wired straight into it. Validation called
both graphs clean. Preset cards got this refusal in #561; blocks did not.

A trigger into a block with no nodes is now refused, naming the block, in the
same words by validation, a run and an export. Validation keeps the code of
the refusal, so the editor can say it in the user's language and jump to the
block. An empty block nothing triggers is still left alone.

A note inside a block is not a node (#624): a block holding only notes is an
empty block, and a note beside real nodes no longer fails validation.

Graphs and blocks are synthetic and written by the tests.
"""

from __future__ import annotations

import pytest

from app.core.execution_context import ExecutionContext
from app.core.graph_engine import (
    GraphValidationError,
    execute_graph,
    prepare_executable_graph,
    validate_graph,
)
from app.core.validation_issues import issue_payload
from app.schemas.models import SubgraphDefinition

EMPTY = "Nothing Inside"


# -- Graphs -----------------------------------------------------------------


def _node(node_id: str, node_type: str, params: dict | None = None) -> dict:
    return {
        "id": node_id,
        "type": node_type,
        "position": {"x": 0, "y": 0},
        "data": {"params": dict(params or {})},
    }


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


def _block(block_id: str, nodes: list[dict], edges: list[dict], *,
           name: str | None = None, trigger_targets: tuple[str, ...] = ()) -> dict:
    """A block with no boundary ports. ``name`` defaults to the id."""
    return {
        "id": block_id,
        "name": block_id if name is None else name,
        "description": "",
        "nodes": nodes,
        "edges": edges,
        "interface": {"inputs": [], "outputs": [],
                      "triggerTargets": list(trigger_targets)},
    }


def _empty(name: str = EMPTY) -> dict:
    return _block("empty", [], [], name=name)


def _graph(nodes: list[dict], edges: list[dict], subgraphs: list[dict]) -> dict:
    return {"name": "b13-graph", "nodes": nodes, "edges": edges,
            "presets": [], "subgraphs": subgraphs}


def _start_into_the_empty_block(*, beside: bool, name: str = EMPTY) -> dict:
    """Start -> an instance ``e`` of :func:`_empty` and, ``beside``, also
    Start -> TextInput -> Print."""
    nodes = [_node("start", "Start"), _node("e", "subgraph:empty")]
    edges = [_trigger("start", "e")]
    if beside:
        nodes += [_node("x", "TextInput", {"value": "q"}), _node("p", "Print")]
        edges += [_trigger("start", "x"), _wire("x", "text", "p", "value")]
    return _graph(nodes, edges, [_empty(name)])


def _sentence(node_id: str, name: str = EMPTY) -> str:
    return (
        f"Node {node_id} is triggered, but subgraph '{name}' has no node to "
        "start: it has no nodes"
    )


# -- The three surfaces -----------------------------------------------------


def _validate(graph: dict) -> list[str]:
    return validate_graph(graph["nodes"], graph["edges"],
                          subgraphs=graph["subgraphs"])


def _prepare(graph: dict):
    return prepare_executable_graph(graph["nodes"], graph["edges"],
                                    subgraphs=graph["subgraphs"])


async def _run(graph: dict) -> dict:
    return await execute_graph(
        graph["nodes"], graph["edges"],
        context=ExecutionContext(device="cpu", weights_persistent=False,
                                 graph_id="b13-engine"),
        subgraphs=graph["subgraphs"],
    )


# -- A trigger into an empty block --------------------------------------------


@pytest.mark.parametrize("beside", [False, True],
                         ids=["alone", "beside another trigger"])
def test_a_run_refuses_a_trigger_into_an_empty_block_by_name(beside):
    """Alone it used to say "Graph has no entry points"; beside another
    trigger the run went ahead without the block."""
    with pytest.raises(GraphValidationError) as refused:
        _prepare(_start_into_the_empty_block(beside=beside))

    assert str(refused.value) == _sentence("e")
    issue = refused.value.args[0]
    assert getattr(issue, "code", None) == "subgraph_triggered_empty"
    assert (issue.node_id, issue.params) == ("e", {"subgraph": EMPTY})


@pytest.mark.parametrize("beside", [False, True],
                         ids=["alone", "beside another trigger"])
def test_validation_reports_it_once_with_its_code(beside):
    """Expansion raises the refusal; validation reports what it raised, code
    and all, and nothing else about the graph."""
    errors = _validate(_start_into_the_empty_block(beside=beside))

    assert errors == [_sentence("e")]
    assert issue_payload(errors) == [{
        "message": _sentence("e"),
        "code": "subgraph_triggered_empty",
        "node_id": "e",
        "params": {"subgraph": EMPTY},
    }]


async def test_validate_and_export_routes_refuse_it_in_the_same_words(test_client):
    graph = _start_into_the_empty_block(beside=True)

    validated = await test_client.post("/api/graph/validate", json=graph)
    assert validated.status_code == 200, validated.text
    body = validated.json()
    assert body["valid"] is False
    assert body["errors"] == [_sentence("e")]
    assert [(issue["code"], issue["node_id"], issue["params"])
            for issue in body["issues"]] == [
        ("subgraph_triggered_empty", "e", {"subgraph": EMPTY})]

    exported = await test_client.post("/api/graph/export", json=graph)
    assert exported.status_code == 400, exported.text
    assert exported.json()["detail"] == _sentence("e")


def test_a_block_with_no_name_is_named_by_its_id():
    graph = _start_into_the_empty_block(beside=False, name="")

    assert _validate(graph) == [_sentence("e", name="empty")]


@pytest.mark.parametrize("declared", [True, False],
                         ids=["trigger target", "inner root"])
def test_a_nested_empty_block_is_named_by_its_flattened_id(declared):
    """The outer block's trigger reaches the inner instance, either as a
    declared trigger target or as an inner root; the next expansion pass
    finds it empty. The editor maps ``b/inner`` to the outer card."""
    outer = _block("outer", [_node("inner", "subgraph:empty")], [],
                   trigger_targets=("inner",) if declared else ())
    graph = _graph([_node("start", "Start"), _node("b", "subgraph:outer")],
                   [_trigger("start", "b")], [outer, _empty()])

    errors = _validate(graph)

    assert errors == [_sentence("b/inner")]
    assert (errors[0].code, errors[0].node_id) == (
        "subgraph_triggered_empty", "b/inner")
    with pytest.raises(GraphValidationError) as refused:
        _prepare(graph)
    assert str(refused.value) == _sentence("b/inner")


# -- What is left alone --------------------------------------------------------


async def test_an_empty_block_nothing_triggers_is_left_alone():
    """Only a TRIGGERED block needs a node to start."""
    graph = _graph(
        [_node("start", "Start"), _node("x", "TextInput", {"value": "q"}),
         _node("p", "Print"), _node("e", "subgraph:empty")],
        [_trigger("start", "x"), _wire("x", "text", "p", "value")],
        [_empty()],
    )

    assert _validate(graph) == []
    executable, _edges, _mapping = _prepare(graph)
    assert sorted(node["id"] for node in executable) == ["p", "start", "x"]
    results = await _run(graph)
    assert results["p"]["value"] == "q"


def test_a_triggered_block_whose_nodes_all_feed_each_other_is_named_by_its_loop():
    """Not this refusal: such a block has a loop inside it, and the line
    naming the loop is the one that says what to fix."""
    loop = _block(
        "loop",
        [_node("p", "Print"), _node("q", "Print")],
        [_wire("p", "value", "q", "value"), _wire("q", "value", "p", "value")],
    )
    graph = _graph([_node("start", "Start"), _node("l", "subgraph:loop")],
                   [_trigger("start", "l")], [loop])

    errors = _validate(graph)

    codes = [getattr(error, "code", None) for error in errors]
    assert "subgraph_triggered_empty" not in codes
    cycle = errors[codes.index("cycle")]
    assert set(cycle.params["path"]) == {"l/p", "l/q"}


# -- Notes inside a block (#624) ------------------------------------------------
#
# The canvas never writes a note into a block; Import JSON, a hand-edited or
# agent-written file, or a plugin can.

NOTES = "Only Notes"


def _note(note_id: str) -> dict:
    """A canvas note the way the canvas saves one: no ``params`` at all."""
    return {"id": note_id, "type": "note", "position": {"x": 0, "y": 0},
            "data": {"text": "a note"}}


def _notes_only() -> dict:
    return _block("notes", [_note("n1")], [], name=NOTES)


def _with_a_note(trigger_targets: tuple[str, ...] = (), *,
                 note_edge: bool = False) -> dict:
    """Start -> an instance ``e`` of a block holding a note beside
    TextInput -> Print."""
    inner_edges = [_wire("x", "text", "p", "value")]
    if note_edge:
        # The canvas draws no handle on a note; a file can still wire one.
        inner_edges.append(_wire("n1", "text", "x", "value"))
    block = _block(
        "withnote",
        [_note("n1"), _node("x", "TextInput", {"value": "q"}), _node("p", "Print")],
        inner_edges,
        trigger_targets=trigger_targets,
    )
    return _graph([_node("start", "Start"), _node("e", "subgraph:withnote")],
                  [_trigger("start", "e")], [block])


@pytest.mark.parametrize("beside", [False, True],
                         ids=["alone", "beside another trigger"])
def test_a_triggered_block_holding_only_notes_is_refused_as_empty(beside):
    """Validation used to say "Unknown node type: note (node e/n1)", and a
    run alone "Graph has no entry points"."""
    nodes = [_node("start", "Start"), _node("e", "subgraph:notes")]
    edges = [_trigger("start", "e")]
    if beside:
        nodes += [_node("x", "TextInput", {"value": "q"}), _node("p", "Print")]
        edges += [_trigger("start", "x"), _wire("x", "text", "p", "value")]
    graph = _graph(nodes, edges, [_notes_only()])

    errors = _validate(graph)

    assert errors == [_sentence("e", NOTES)]
    assert issue_payload(errors) == [{
        "message": _sentence("e", NOTES),
        "code": "subgraph_triggered_empty",
        "node_id": "e",
        "params": {"subgraph": NOTES},
    }]
    with pytest.raises(GraphValidationError) as refused:
        _prepare(graph)
    assert str(refused.value) == _sentence("e", NOTES)
    assert getattr(refused.value.args[0], "code", None) == "subgraph_triggered_empty"


@pytest.mark.parametrize(
    ("trigger_targets", "note_edge"),
    [((), False), (("x",), False), (("x", "n1"), False), ((), True)],
    ids=["inner roots", "trigger target", "the note a trigger target too",
         "the note wired to a node"],
)
async def test_a_note_beside_nodes_in_a_block_is_left_out(trigger_targets, note_edge):
    """Validation refused the whole graph with "Unknown node type: note", so
    the editor would not run it, and the prepared graph kept the note as a
    node to run."""
    graph = _with_a_note(trigger_targets, note_edge=note_edge)

    assert _validate(graph) == []
    executable, edges, _mapping = _prepare(graph)
    assert sorted(node["id"] for node in executable) == ["e/p", "e/x", "start"]
    assert all("e/n1" not in (edge["source"], edge["target"]) for edge in edges)
    results = await _run(graph)
    assert results["e/p"]["value"] == "q"


async def test_trigger_targets_that_name_only_notes_start_every_inner_root():
    """Notes are ignored, so naming only notes names no target, and a block
    that names none starts each of its inner roots. It is also what the
    definition becomes once the canvas rewrites it: ``definitionFromCanvas``
    drops the note, and its id from the trigger targets."""
    block = _block(
        "tworoots",
        [_note("n1"),
         _node("x", "TextInput", {"value": "q"}), _node("p", "Print"),
         _node("y", "TextInput", {"value": "r"}), _node("s", "Print")],
        [_wire("x", "text", "p", "value"), _wire("y", "text", "s", "value")],
        trigger_targets=("n1",),
    )
    graph = _graph([_node("start", "Start"), _node("e", "subgraph:tworoots")],
                   [_trigger("start", "e")], [block])

    assert _validate(graph) == []
    _executable, edges, _mapping = _prepare(graph)
    assert sorted(
        edge["target"] for edge in edges if edge.get("type") == "trigger"
    ) == ["e/x", "e/y"]
    results = await _run(graph)
    assert (results["e/p"]["value"], results["e/s"]["value"]) == ("q", "r")


def test_a_block_holding_only_notes_that_nothing_triggers_is_left_alone():
    graph = _graph(
        [_node("start", "Start"), _node("x", "TextInput", {"value": "q"}),
         _node("p", "Print"), _node("e", "subgraph:notes")],
        [_trigger("start", "x"), _wire("x", "text", "p", "value")],
        [_notes_only()],
    )

    assert _validate(graph) == []
    executable, _edges, _mapping = _prepare(graph)
    assert sorted(node["id"] for node in executable) == ["p", "start", "x"]


def test_the_definition_every_instance_shares_keeps_its_note():
    """A definition passed in as a model is the one object every instance of
    the block reads, so the notes are left out of a copy of it."""
    graph = _with_a_note()
    definition = SubgraphDefinition(**graph["subgraphs"][0])
    graph["subgraphs"] = [definition]
    graph["nodes"].append(_node("e2", "subgraph:withnote"))
    graph["edges"].append(_trigger("start", "e2"))

    executable, _edges, _mapping = _prepare(graph)

    assert sorted(node["id"] for node in executable) == [
        "e/p", "e/x", "e2/p", "e2/x", "start"]
    assert [node.id for node in definition.nodes] == ["n1", "x", "p"]
