"""ID6: a graph carrying its own preset definition resolves even when the
server's preset registry does not know the preset (portability)."""

from app.core.graph_engine import (
    build_preset_fallback,
    expand_presets,
    validate_graph,
)
from app.core.preset_registry import preset_registry


def _preset_dict(name="EmbeddedPr"):
    # An internal Print node exposed as one input; a name the registry lacks.
    return {
        "preset_name": name,
        "category": "Custom",
        "description": "",
        "tags": [],
        "nodes": [{"id": "inner", "type": "Print", "params": {"label": "x"}}],
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
