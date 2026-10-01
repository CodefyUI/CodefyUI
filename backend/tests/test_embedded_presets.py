"""ID6: a graph carrying its own preset definition resolves even when the
server's preset registry does not know the preset (portability)."""

import pytest

from app.core.codegen import generate_python
from app.core.graph_engine import (
    GraphValidationError,
    build_preset_fallback,
    expand_presets,
    prepare_executable_graph,
    validate_graph,
)
from app.core.preset_registry import preset_registry
from app.schemas.models import PresetDefinition


def _preset_dict(name="EmbeddedPr", *, inner_type="Print", label="x"):
    # An internal Print node exposed as one input; a name the registry lacks.
    return {
        "preset_name": name,
        "category": "Custom",
        "description": "",
        "tags": [],
        "nodes": [{"id": "inner", "type": inner_type, "params": {"label": label}}],
        "edges": [],
        "exposed_inputs": [
            {"name": "in", "internal_node": "inner", "internal_port": "value",
             "data_type": "ANY", "description": ""},
        ],
        "exposed_outputs": [],
        "exposed_params": [],
    }


def test_registry_lacks_embedded_preset():
    assert preset_registry.get("EmbeddedPr") is None  # sanity: truly absent


def test_build_preset_fallback_maps_names():
    fb = build_preset_fallback([_preset_dict()])
    assert "EmbeddedPr" in fb
    assert fb["EmbeddedPr"].preset_name == "EmbeddedPr"
    # Tolerates junk without raising.
    assert build_preset_fallback([{"bogus": 1}, None]) == {}


def test_validate_unknown_without_fallback():
    nodes = [{"id": "p", "type": "preset:EmbeddedPr", "data": {}}]
    errors = validate_graph(nodes, [])
    assert any("Unknown preset: EmbeddedPr" in e for e in errors)


def test_validate_ok_with_fallback():
    nodes = [{"id": "p", "type": "preset:EmbeddedPr", "data": {}}]
    fb = build_preset_fallback([_preset_dict()])
    errors = validate_graph(nodes, [], preset_fallback=fb)
    assert not any("Unknown preset" in e for e in errors)


def test_expand_uses_fallback():
    nodes = [{"id": "p", "type": "preset:EmbeddedPr",
              "position": {"x": 0, "y": 0}, "data": {}}]
    fb = build_preset_fallback([_preset_dict()])
    expanded, _edges, mapping = expand_presets(nodes, [], preset_fallback=fb)
    assert any(n["id"] == "p__inner" and n["type"] == "Print" for n in expanded)
    assert mapping["p__inner"] == "p"


def test_embedded_definition_wins_same_name_collision(monkeypatch):
    name = "PortableCollision"
    installed = build_preset_fallback([
        _preset_dict(name, inner_type="TextInput", label="installed"),
    ])[name]
    portable = build_preset_fallback([
        _preset_dict(name, inner_type="Print", label="portable"),
    ])[name]
    monkeypatch.setitem(preset_registry._presets, name, installed)

    nodes = [{
        "id": "p",
        "type": f"preset:{name}",
        "position": {"x": 0, "y": 0},
        "data": {},
    }]
    fallback = {name: portable}

    errors = validate_graph(nodes, [], preset_fallback=fallback)
    assert not any("Unknown preset" in error for error in errors)
    expanded, _edges, _mapping = expand_presets(
        nodes, [], preset_fallback=fallback,
    )
    assert expanded[0]["type"] == "Print"
    assert expanded[0]["data"]["params"]["label"] == "portable"


def test_losing_installed_definition_cannot_reject_portable_winner(monkeypatch):
    name = "PortableCollisionWithBrokenInstalled"
    installed = build_preset_fallback([
        _preset_dict(name, inner_type="subgraph:installed-only"),
    ])[name]
    portable = build_preset_fallback([_preset_dict(name)])[name]
    monkeypatch.setitem(preset_registry._presets, name, installed)
    nodes = [{"id": "p", "type": f"preset:{name}", "data": {}}]

    errors = validate_graph(nodes, [], preset_fallback={name: portable})

    assert not any("contains subgraph instance" in error for error in errors)


# -- a graph-owned definition that does not parse (#541) ---------------------
#
# The graph owns the name, so an installed preset of the same name must not
# stand in for it: that ran a definition the graph never meant, and scrubbed
# its keys by the wrong one. Referenced, it is refused by name. Unreferenced,
# it is ignored, as before.

UNREADABLE = "UnreadablePr"


def _unreadable(name=UNREADABLE):
    """``_preset_dict`` with no top-level ``description``: the model refuses it."""
    body = _preset_dict(name)
    del body["description"]
    return body


def _start_into(node_id, node_type):
    """Start, triggering one node."""
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0}, "data": {}},
        {"id": node_id, "type": node_type, "position": {"x": 1, "y": 0}, "data": {}},
    ]
    edges = [{"id": "t", "source": "start", "target": node_id,
              "sourceHandle": "trigger", "targetHandle": "__trigger",
              "type": "trigger"}]
    return nodes, edges


@pytest.fixture
def _installed_same_name(monkeypatch):
    """An installed preset of the same name, there to be wrongly used."""
    monkeypatch.setitem(preset_registry._presets, UNREADABLE, PresetDefinition(
        **_preset_dict(UNREADABLE, inner_type="TextInput", label="installed")))


def _names_it(message, node_id):
    return (f"Preset '{UNREADABLE}'" in message
            and f"node {node_id}" in message
            and "description: Field required" in message)


def test_an_unreadable_entry_is_kept_by_name_with_its_first_error():
    from app.core.graph_engine import MalformedPreset

    entry = build_preset_fallback([_unreadable()])[UNREADABLE]
    assert isinstance(entry, MalformedPreset)
    assert entry.name == UNREADABLE
    assert entry.error == "description: Field required"


def test_a_parsable_definition_wins_over_an_unreadable_one_in_either_order():
    good, bad = _preset_dict(UNREADABLE), _unreadable()
    for presets in ([good, bad], [bad, good]):
        assert isinstance(build_preset_fallback(presets)[UNREADABLE],
                          PresetDefinition)


def test_validate_run_and_export_refuse_a_referenced_unreadable_definition(
        _installed_same_name):
    nodes, edges = _start_into("p", f"preset:{UNREADABLE}")
    presets = [_unreadable()]
    fallback = build_preset_fallback(presets)

    errors = validate_graph(nodes, edges, preset_fallback=fallback)
    assert any(_names_it(error, "p") for error in errors), errors
    assert not any("Unknown" in error for error in errors), errors
    # Run: `execute_graph` starts with this preflight.
    with pytest.raises(GraphValidationError) as run:
        prepare_executable_graph(nodes, edges, preset_fallback=fallback)
    assert _names_it(str(run.value), "p"), run.value
    with pytest.raises(GraphValidationError) as export:
        generate_python(nodes, edges, presets=presets)
    assert _names_it(str(export.value), "p"), export.value


def test_a_definition_that_nests_an_unreadable_one_is_refused_too(
        _installed_same_name):
    outer = _preset_dict("NestsUnreadable", inner_type=f"preset:{UNREADABLE}")
    fallback = build_preset_fallback([outer, _unreadable()])
    nodes, edges = _start_into("p", "preset:NestsUnreadable")

    errors = validate_graph(nodes, edges, preset_fallback=fallback)
    assert any(_names_it(error, "p") for error in errors), errors
    with pytest.raises(GraphValidationError) as run:
        prepare_executable_graph(nodes, edges, preset_fallback=fallback)
    assert f"Preset '{UNREADABLE}'" in str(run.value), run.value


def test_a_card_in_a_block_that_uses_an_unreadable_definition_is_refused(
        _installed_same_name):
    nodes, edges = _start_into("blk", "subgraph:d")
    subgraphs = [{
        "id": "d", "name": "d", "edges": [],
        "nodes": [{"id": "p", "type": f"preset:{UNREADABLE}",
                   "position": {"x": 0, "y": 0}, "data": {}}],
        "interface": {"inputs": [], "outputs": [], "triggerTargets": []},
    }]
    fallback = build_preset_fallback([_unreadable()])

    errors = validate_graph(nodes, edges, preset_fallback=fallback,
                            subgraphs=subgraphs)
    assert any(_names_it(error, "blk/p") for error in errors), errors
    with pytest.raises(GraphValidationError) as run:
        prepare_executable_graph(nodes, edges, preset_fallback=fallback,
                                 subgraphs=subgraphs)
    assert _names_it(str(run.value), "blk/p"), run.value


def test_the_unreadable_lines_on_their_own_are_validation_s_own(
        _installed_same_name):
    """What the publish route refuses with before its secret gate: exactly
    the unreadable-preset lines ``validate_graph`` gives for the same graph,
    at the top level, inside a block and through a nesting preset."""
    from app.core.graph_engine import unreadable_preset_errors

    nodes, edges = _start_into("p", f"preset:{UNREADABLE}")
    nodes += [
        {"id": "n", "type": "preset:NestsUnreadable",
         "position": {"x": 2, "y": 0}, "data": {}},
        {"id": "r", "type": "preset:EmbeddedPr",
         "position": {"x": 3, "y": 0}, "data": {}},
        {"id": "blk", "type": "subgraph:d",
         "position": {"x": 4, "y": 0}, "data": {}},
    ]
    subgraphs = [{
        "id": "d", "name": "d", "edges": [],
        "nodes": [{"id": "q", "type": f"preset:{UNREADABLE}",
                   "position": {"x": 0, "y": 0}, "data": {}}],
        "interface": {"inputs": [], "outputs": [], "triggerTargets": []},
    }]
    fallback = build_preset_fallback([
        _unreadable(),
        _preset_dict("NestsUnreadable", inner_type=f"preset:{UNREADABLE}"),
        _preset_dict(),
    ])

    lines = unreadable_preset_errors(
        nodes, edges, preset_fallback=fallback, subgraphs=subgraphs)
    validated = validate_graph(
        nodes, edges, preset_fallback=fallback, subgraphs=subgraphs)

    assert sorted(lines) == sorted(
        line for line in validated if "could not be read" in line)
    assert sorted(lines) == sorted(
        fallback[UNREADABLE].refusal(node_id) for node_id in ("p", "n", "blk/q"))


def test_an_unreadable_definition_no_node_uses_is_ignored(_installed_same_name):
    """A stray entry never breaks validation, a run or an export."""
    nodes, edges = _start_into("a", "TensorCreate")
    nodes[1]["data"] = {"params": {"shape": "2,2", "fill": "full", "value": 2.0}}
    fallback = build_preset_fallback([_unreadable()])

    assert validate_graph(nodes, edges, preset_fallback=fallback) == []
    prepared, _edges, _mapping = prepare_executable_graph(
        nodes, edges, preset_fallback=fallback)
    assert [node["id"] for node in prepared] == ["start", "a"]
    assert "def " in generate_python(nodes, edges, presets=[_unreadable()])
