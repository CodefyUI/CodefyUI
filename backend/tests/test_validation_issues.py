"""Validation findings say what they are about, not only what is wrong.

``validate_graph`` returned plain English sentences that named a node by its
raw id -- or, for an edge, by its node TYPE alone, so with two Linear nodes
nobody could tell which one was meant -- and the editor showed them as they
came. Each finding the canvas can produce is now a :class:`ValidationIssue`:
the same ``str`` as before, so every caller, run envelope and test that reads
the text keeps working, which also carries a ``code``, the ``node_id`` it is
about and the ``params`` its sentence was built from. ``POST
/api/graph/validate`` sends them as ``issues`` beside ``errors``, so the
editor can say each one in the user's language, name the node by its title
and jump to it.

Graphs and presets are synthetic and written by the tests.
"""

from __future__ import annotations

import copy
import json
import pickle

import pytest

from app.core.graph_engine import build_preset_fallback, validate_graph
from app.core.node_base import DataType
from app.core.validation_issues import (
    ValidationIssue,
    issue_payload,
    validation_issue,
)

# -- Graphs -----------------------------------------------------------------


def _node(node_id: str, node_type: str, **params) -> dict:
    return {
        "id": node_id,
        "type": node_type,
        "position": {"x": 0, "y": 0},
        "data": {"params": params},
    }


def _trigger(source: str, target: str) -> dict:
    """A trigger edge the way the canvas saves one."""
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


def _bypassed(node: dict) -> dict:
    node["data"]["bypassed"] = True
    return node


def _preset(name: str, nodes: list[dict], edges=(), outputs=()) -> dict:
    """``edges`` are ``(source, port, target, port)``; ``outputs`` are
    ``(exposed name, internal node, internal port)``."""
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


SAY_HI = "T11 Say Hi"


def _say_hi() -> dict:
    """A TextInput into a Print, whose value is the card's ``out``."""
    return _preset(
        SAY_HI,
        nodes=[{"id": "a", "type": "TextInput", "params": {"value": "hi"}},
               {"id": "b", "type": "Print", "params": {}}],
        edges=[("a", "text", "b", "value")],
        outputs=(("out", "b", "value"),),
    )


def _card(node_id: str, preset_name: str) -> dict:
    return {
        "id": node_id,
        "type": f"preset:{preset_name}",
        "position": {"x": 0, "y": 0},
        "data": {"params": {}, "internalParams": {}},
    }


def _issues(errors: list[str], code: str) -> list[ValidationIssue]:
    """The findings carrying *code*. A plain ``str`` has no code at all."""
    return [error for error in errors if getattr(error, "code", None) == code]


def _one(errors: list[str], code: str) -> ValidationIssue:
    found = _issues(errors, code)
    assert len(found) == 1, (code, [(e, getattr(e, "code", None)) for e in errors])
    return found[0]


# -- The str subclass -------------------------------------------------------


def test_an_issue_is_the_message_and_carries_what_it_is_about():
    issue = validation_issue(
        "missing_input", "Missing x on node n1", node_id="n1", port="x")

    assert isinstance(issue, str)
    assert issue == "Missing x on node n1"
    assert issue in ["Missing x on node n1"]
    assert (issue.code, issue.node_id, issue.params) == (
        "missing_input", "n1", {"port": "x"})


def test_params_may_use_any_name_even_code_or_message():
    """``code`` and ``message`` are positional-only, so a param of the same
    name cannot collide with them."""
    issue = validation_issue("c", "text", node_id=None, code="x", message="y")

    assert issue.code == "c"
    assert issue.params == {"code": "x", "message": "y"}


def test_an_issue_survives_copy_and_pickle():
    """A ``str`` subclass is rebuilt by calling the class with the text alone,
    so its own fields must come back from its state, not its constructor."""
    issue = validation_issue("cycle", "Graph contains a cycle: a -> a",
                             node_id="a", path=["a", "a"])

    for clone in (copy.deepcopy(issue), pickle.loads(pickle.dumps(issue))):
        assert clone == issue
        assert (clone.code, clone.node_id, clone.params) == (
            "cycle", "a", {"path": ["a", "a"]})


def test_the_payload_has_one_entry_per_line_and_none_for_a_plain_string():
    issue = validation_issue("unknown_node_type", "Unknown node type: X (node n)",
                             node_id="n", type="X")

    payload = issue_payload([issue, "Duplicate node id: the node 'd' appears more than once"])

    assert payload == [
        {"message": "Unknown node type: X (node n)", "code": "unknown_node_type",
         "node_id": "n", "params": {"type": "X"}},
        {"message": "Duplicate node id: the node 'd' appears more than once",
         "code": None, "node_id": None, "params": {}},
    ]
    # The message is a plain str, so nothing downstream sees the subclass.
    assert type(payload[0]["message"]) is str


# -- Every node-level code, and the old text byte for byte --------------------


def test_missing_input_names_the_node_and_the_port():
    errors = validate_graph(
        [_node("start", "Start"), _node("lin", "Linear")],
        [_trigger("start", "lin")],
    )

    issue = _one(errors, "missing_input")
    assert issue == (
        "Missing required input 'tensor' on node lin (Linear) -- connect an "
        "output to this port"
    )
    assert issue.node_id == "lin"
    assert issue.params == {"port": "tensor", "type": "Linear"}


def test_missing_input_caused_by_a_bypass_names_the_bypassed_node():
    errors = validate_graph(
        [_node("start", "Start"), _bypassed(_node("drop", "Dropout", p=0.5)),
         _node("out", "Print", label="tail")],
        [_trigger("start", "drop"), _wire("drop", "tensor", "out", "value")],
    )

    issue = _one(errors, "missing_input")
    assert issue == (
        "Missing required input 'value' on node out (Print) (input dropped "
        "because 'drop' is bypassed)"
    )
    assert issue.node_id == "out"
    assert issue.params == {"port": "value", "type": "Print", "cause": "drop"}


@pytest.mark.parametrize(
    ("value", "code", "params", "message"),
    [
        (-0.5, "param_below_min", {"param": "p", "value": -0.5, "min": 0.0},
         "Parameter 'p' on node d (Dropout): value -0.5 is below minimum 0.0"),
        (1.5, "param_above_max", {"param": "p", "value": 1.5, "max": 1.0},
         "Parameter 'p' on node d (Dropout): value 1.5 is above maximum 1.0"),
        ("abc", "param_not_number", {"param": "p", "value": "abc"},
         "Parameter 'p' on node d (Dropout): value 'abc' is not a number, so "
         "it cannot be checked against its allowed range"),
        (None, "param_not_number", {"param": "p", "value": None},
         "Parameter 'p' on node d (Dropout): value None is not a number, so "
         "it cannot be checked against its allowed range"),
    ],
    ids=["below min", "above max", "a string", "a cleared box"],
)
def test_a_param_out_of_range_names_the_param_and_the_values(value, code, params, message):
    errors = validate_graph(
        [_node("start", "Start"), _node("src", "_TestSource"),
         _node("d", "Dropout", p=value)],
        [_trigger("start", "src"), _wire("src", "value", "d", "tensor")],
    )

    issue = _one(errors, code)
    assert issue == message
    assert issue.node_id == "d"
    assert issue.params == params


def test_a_misspelt_option_names_the_param_and_the_allowed_values():
    # #698: a SELECT value outside its options passed validation.
    errors = validate_graph(
        [_node("start", "Start"),
         _node("save", "ModelSaver", save_mode="statedict", format="pytorch",
               path="x.pt"),
         _node("opt", "Optimizer", type="Adamm")],
        [_trigger("start", "save"), _trigger("start", "opt")],
    )

    found = {issue.node_id: issue for issue in _issues(errors, "param_not_an_option")}
    assert set(found) == {"save", "opt"}
    assert found["save"] == (
        "Parameter 'save_mode' on node save (ModelSaver): value 'statedict' "
        "is not one of its options (state_dict, full_model)"
    )
    assert found["save"].params == {
        "param": "save_mode", "value": "statedict",
        "options": "state_dict, full_model",
    }
    assert found["opt"].params["param"] == "type"
    assert found["opt"].params["value"] == "Adamm"


def test_every_listed_option_passes():
    from app.core.node_registry import registry

    options = next(p.options for p in registry.get("Optimizer").define_params()
                   if p.name == "type")
    for option in options:
        errors = validate_graph(
            [_node("start", "Start"), _node("opt", "Optimizer", type=option)],
            [_trigger("start", "opt")],
        )
        assert _issues(errors, "param_not_an_option") == [], option


def test_an_unknown_node_type_names_the_node_and_the_type():
    errors = validate_graph(
        [_node("start", "Start"), _node("ghost", "NoSuchNode")],
        [_trigger("start", "ghost")],
    )

    issue = _one(errors, "unknown_node_type")
    assert issue == "Unknown node type: NoSuchNode (node ghost)"
    assert issue.node_id == "ghost"
    assert issue.params == {"type": "NoSuchNode"}


def test_an_unknown_preset_names_the_node_and_the_preset():
    errors = validate_graph(
        [_node("start", "Start"), _card("c", "T11 Nowhere")],
        [_trigger("start", "c")],
    )

    issue = _one(errors, "unknown_preset")
    assert issue == "Unknown preset: T11 Nowhere (node c)"
    assert issue.node_id == "c"
    assert issue.params == {"preset": "T11 Nowhere"}


@pytest.mark.parametrize(
    ("unknown", "code"),
    [(_node("ghost", "NoSuchNode"), "unknown_node_type"),
     (_card("ghost", "T11 Nowhere"), "unknown_preset")],
    ids=["unknown node type", "unknown preset"],
)
def test_an_edge_on_a_node_of_unknown_type_adds_no_line_of_its_own(unknown, code):
    """The node's own line names it, with a code the editor translates. Each
    edge on it also said "Unknown node type: X or Y", with no code, and the
    editor showed that as one more toast, in English."""
    errors = validate_graph(
        [_node("start", "Start"), _node("t", "TextInput", value="q"), unknown,
         _node("p", "Print")],
        [_trigger("start", "t"), _wire("t", "text", "ghost", "value"),
         _wire("ghost", "value", "p", "value")],
    )

    assert [getattr(error, "code", None) for error in errors] == [code]


# -- Edge-level codes: the node, out of two of the same type ------------------


def _two_conv_layers(*edges: dict) -> list[str]:
    return validate_graph(
        [_node("start", "Start"), _node("src", "_TestSource"),
         _node("loss", "Loss"), _node("c1", "Conv2d"), _node("c2", "Conv2d"),
         _node("p", "Print")],
        [_trigger("start", "src"), _trigger("start", "loss"),
         _wire("src", "value", "c1", "tensor"), *edges],
    )


def test_an_invalid_output_port_names_the_source_out_of_two_of_its_type():
    errors = _two_conv_layers(_wire("src", "value", "c2", "tensor"),
                              _wire("c2", "nope", "p", "value"))

    issue = _one(errors, "invalid_output_port")
    assert issue == "Invalid output port 'nope' on Conv2d"
    assert issue.node_id == "c2"
    assert issue.params == {"port": "nope", "type": "Conv2d", "target": "p"}


def test_an_invalid_input_port_names_the_target_out_of_two_of_its_type():
    errors = _two_conv_layers(_wire("src", "value", "c2", "tensor"),
                              _wire("src", "value", "c2", "nope"))

    issue = _one(errors, "invalid_input_port")
    assert issue == "Invalid input port 'nope' on Conv2d"
    assert issue.node_id == "c2"
    assert issue.params == {"port": "nope", "type": "Conv2d", "source": "src"}


def test_a_type_mismatch_names_both_ends_out_of_two_of_their_type():
    errors = _two_conv_layers(_wire("loss", "loss_fn", "c2", "tensor"))

    issue = _one(errors, "type_mismatch")
    # Built the way the engine builds it: how an f-string renders a str-mixin
    # enum changed in Python 3.11 ("LOSS_FN" before, "DataType.LOSS_FN"
    # since), and CI runs both. The params carry the bare value either way.
    assert issue == (
        f"Type mismatch: Loss.loss_fn ({DataType.LOSS_FN}) -> "
        f"Conv2d.tensor ({DataType.TENSOR})"
    )
    assert issue.node_id == "c2"
    assert issue.params == {
        "source": "loss", "source_port": "loss_fn", "source_type": "LOSS_FN",
        "port": "tensor", "target_type": "TENSOR",
    }


# -- Graph-level codes ---------------------------------------------------------


def test_no_entry_points_is_about_no_node():
    errors = validate_graph([_node("src", "_TestSource")], [])

    issue = _one(errors, "no_entry_points")
    assert issue == (
        "Graph has no entry points. Add a Start node and connect it to the "
        "node you want to start execution from."
    )
    assert issue.node_id is None
    assert issue.params == {}


def test_a_cycle_carries_its_path():
    errors = validate_graph(
        [_node("start", "Start"), _node("p1", "Print"), _node("p2", "Print")],
        [_trigger("start", "p1"), _wire("p1", "value", "p2", "value"),
         _wire("p2", "value", "p1", "value")],
    )

    issue = _one(errors, "cycle")
    path = issue.params["path"]
    assert issue == f"Graph contains a cycle: {' -> '.join(path)}"
    assert path[0] == path[-1] and set(path) == {"p1", "p2"}
    assert issue.node_id == path[0]


# -- Preset cards and trigger edges (#561) ------------------------------------


def _with_presets(nodes: list[dict], edges: list[dict], *presets: dict) -> list[str]:
    return validate_graph(nodes, edges,
                          preset_fallback=build_preset_fallback(list(presets)))


def test_an_edge_into_a_port_the_card_does_not_expose():
    errors = _with_presets(
        [_node("start", "Start"), _node("x", "TextInput", value="q"),
         _card("c", SAY_HI)],
        [_trigger("start", "x"), _wire("x", "text", "c", "nope")],
        _say_hi(),
    )

    issue = _one(errors, "preset_input_not_exposed")
    assert issue == (
        f"Edge targets input port 'nope' which preset '{SAY_HI}' does not "
        "expose (node c)"
    )
    assert issue.node_id == "c"
    assert issue.params == {"port": "nope", "preset": SAY_HI}


def test_an_edge_out_of_a_port_the_card_does_not_expose():
    errors = _with_presets(
        [_node("start", "Start"), _card("c", SAY_HI), _node("p", "Print")],
        [_trigger("start", "c"), _wire("c", "nope", "p", "value")],
        _say_hi(),
    )

    issue = _one(errors, "preset_output_not_exposed")
    assert issue == (
        f"Edge sources output port 'nope' which preset '{SAY_HI}' does not "
        "expose (node c)"
    )
    assert issue.node_id == "c"
    assert issue.params == {"port": "nope", "preset": SAY_HI}


@pytest.mark.parametrize(
    ("preset", "code", "reason"),
    [
        (_preset("T11 Empty", nodes=[]), "preset_triggered_empty",
         "it has no nodes"),
        (_preset("T11 Loop",
                 nodes=[{"id": "p", "type": "Print", "params": {}},
                        {"id": "q", "type": "Print", "params": {}}],
                 edges=[("p", "value", "q", "value"), ("q", "value", "p", "value")]),
         "preset_triggered_all_fed",
         "every node inside it is fed by another node inside it"),
    ],
    ids=["no nodes", "every node fed"],
)
def test_a_triggered_card_with_no_node_to_start(preset, code, reason):
    name = preset["preset_name"]
    errors = _with_presets(
        [_node("start", "Start"), _card("c", name)],
        [_trigger("start", "c")],
        preset,
    )

    issue = _one(errors, code)
    assert issue == (
        f"Node c is triggered, but preset '{name}' has no node to start: {reason}")
    assert issue.node_id == "c"
    assert issue.params == {"preset": name}


def test_a_triggered_block_with_no_nodes():
    """Block expansion refuses it, and validation reports the refusal it
    caught as raised, so the code survives (#604)."""
    errors = validate_graph(
        [_node("start", "Start"), _node("b", "subgraph:empty")],
        [_trigger("start", "b")],
        subgraphs=[{"id": "empty", "name": "T11 Empty Block",
                    "nodes": [], "edges": []}],
    )

    issue = _one(errors, "subgraph_triggered_empty")
    assert issue == (
        "Node b is triggered, but subgraph 'T11 Empty Block' has no node to "
        "start: it has no nodes")
    assert issue.node_id == "b"
    assert issue.params == {"subgraph": "T11 Empty Block"}


def test_a_trigger_from_a_missing_node_is_about_the_node_it_reaches():
    """The edge is never drawn: React Flow draws no edge whose node is gone,
    so the canvas cannot delete it. The sentence says where it can go."""
    errors = validate_graph(
        [_node("x", "TextInput", value="q")],
        [_trigger("gone", "x")],
    )

    issue = _one(errors, "trigger_source_missing")
    assert issue == (
        "A trigger edge comes from node 'gone', which is not in the graph -- "
        "remove the edge from the graph file"
    )
    assert issue.node_id == "x"
    assert issue.params == {"source": "gone"}


def test_a_trigger_to_a_missing_node_is_about_the_node_it_leaves():
    errors = validate_graph(
        [_node("start", "Start"), _node("x", "TextInput", value="q")],
        [_trigger("start", "x"), _trigger("start", "nowhere")],
    )

    issue = _one(errors, "trigger_target_missing")
    assert issue == (
        "A trigger edge goes to node 'nowhere', which is not in the graph -- "
        "remove the edge from the graph file"
    )
    assert issue.node_id == "start"
    assert issue.params == {"target": "nowhere"}


# -- Callers that read the list as text ------------------------------------------


def test_the_list_is_still_a_list_of_strings_that_json_can_write():
    errors = _two_conv_layers(_wire("loss", "loss_fn", "c2", "tensor"),
                              _wire("c2", "nope", "p", "value"))

    assert errors and all(isinstance(error, str) for error in errors)
    assert json.loads(json.dumps(errors)) == [str(error) for error in errors]
    assert "; ".join(errors) == "; ".join(str(error) for error in errors)


# -- The validate route ------------------------------------------------------------


async def test_the_route_sends_issues_aligned_with_errors(test_client):
    graph = {
        "name": "t11",
        "nodes": [
            _node("start", "Start"), _node("lin", "Linear"),
            _node("dup", "_TestSource"), _node("dup", "_TestSource"),
            _node("d", "Dropout", p=1.5),
        ],
        "edges": [_trigger("start", "lin"), _trigger("start", "d")],
    }

    response = await test_client.post("/api/graph/validate", json=graph)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["valid"] is False
    issues = body["issues"]
    assert [issue["message"] for issue in issues] == body["errors"]
    by_message = {issue["message"]: issue for issue in issues}
    # A line from a check that has no code yet still arrives, code-less.
    duplicate = by_message["Duplicate node id: the node 'dup' appears more than once"]
    assert duplicate == {
        "message": "Duplicate node id: the node 'dup' appears more than once",
        "code": None, "node_id": None, "params": {},
    }
    missing = next(issue for issue in issues if issue["code"] == "missing_input"
                   and issue["node_id"] == "lin")
    assert missing["params"] == {"port": "tensor", "type": "Linear"}
    above = next(issue for issue in issues if issue["code"] == "param_above_max")
    assert (above["node_id"], above["params"]) == (
        "d", {"param": "p", "value": 1.5, "max": 1.0})


async def test_a_valid_graph_has_no_issues(test_client, sample_graph):
    response = await test_client.post("/api/graph/validate", json=sample_graph)

    assert response.status_code == 200, response.text
    assert response.json() == {"valid": True, "errors": [], "issues": []}


async def test_a_value_json_cannot_write_does_not_break_the_route(test_client):
    """Python's JSON reader takes ``Infinity``, so a request can carry a value
    the response must not try to write back as is."""
    graph = {
        "name": "t11",
        "nodes": [_node("start", "Start"), _node("src", "_TestSource"),
                  _node("d", "Dropout", p=float("inf"))],
        "edges": [_trigger("start", "src"), _wire("src", "value", "d", "tensor")],
    }

    # Written by hand: httpx's own encoder refuses ``Infinity``.
    response = await test_client.post(
        "/api/graph/validate", content=json.dumps(graph),
        headers={"Content-Type": "application/json"})

    assert response.status_code == 200, response.text
    above = next(issue for issue in response.json()["issues"]
                 if issue["code"] == "param_above_max")
    assert above["params"] == {"param": "p", "value": None, "max": 1.0}
