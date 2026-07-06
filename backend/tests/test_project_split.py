"""Logic/layout split + merge (spec 6.1-6.4): geometry lives only in the
layout file, params only in the logic file, and the layout is a full snapshot."""

import json

from app.core.project import (
    FORMAT_VERSION,
    merge_graph,
    split_graph,
    write_graph_pair,
)


def _payload():
    return {
        "name": "demo",
        "description": "d",
        "nodes": [
            {"id": "a", "type": "Dataset", "position": {"x": 10, "y": 20},
             "data": {"params": {"name": "MNIST"}}},
            {"id": "n1", "type": "note", "position": {"x": 5, "y": 6},
             "data": {"noteKind": "text", "noteContent": "hi",
                      "noteColor": "#333", "boundToNodeId": "a",
                      "boundOffset": {"x": 1, "y": 2}, "noteWidth": 200,
                      "noteHeight": 80}},
        ],
        "edges": [{"id": "e", "source": "a", "target": "a"}],
        "presets": [],
        "segmentGroups": [{"id": "s", "headNodeId": "a", "tailNodeId": "a"}],
    }


def test_split_extracts_geometry_only():
    logic, layout = split_graph(_payload())
    assert logic["format_version"] == FORMAT_VERSION
    # Logic nodes carry NO position and NO note geometry.
    a = next(n for n in logic["nodes"] if n["id"] == "a")
    assert "position" not in a
    assert a["data"]["params"]["name"] == "MNIST"
    note = next(n for n in logic["nodes"] if n["id"] == "n1")
    assert "position" not in note
    for k in ("boundToNodeId", "boundOffset", "noteWidth", "noteHeight"):
        assert k not in note["data"]
    assert note["data"]["noteContent"] == "hi"
    # Layout carries positions (as ints) + note geometry + segmentGroups.
    assert layout["positions"]["a"] == {"x": 10, "y": 20}
    assert layout["notes"]["n1"]["boundToNodeId"] == "a"
    assert layout["notes"]["n1"]["noteWidth"] == 80 or layout["notes"]["n1"]["noteWidth"] == 200
    assert layout["segmentGroups"][0]["id"] == "s"


def test_merge_round_trips():
    logic, layout = split_graph(_payload())
    merged, missing = merge_graph(logic, layout)
    assert missing is False
    a = next(n for n in merged["nodes"] if n["id"] == "a")
    assert a["position"] == {"x": 10, "y": 20}
    note = next(n for n in merged["nodes"] if n["id"] == "n1")
    assert note["data"]["boundToNodeId"] == "a"
    assert merged["segmentGroups"][0]["id"] == "s"
    assert merged["layout_missing"] is False


def test_merge_missing_layout_flags_and_omits_position():
    logic, _ = split_graph(_payload())
    merged, missing = merge_graph(logic, None)
    assert missing is True
    assert merged["layout_missing"] is True
    for n in merged["nodes"]:
        assert "position" not in n  # frontend will auto-layout


def test_positions_pinned_to_int():
    payload = _payload()
    payload["nodes"][0]["position"] = {"x": 10.7, "y": 20.2}
    _logic, layout = split_graph(payload)
    assert layout["positions"]["a"] == {"x": 11, "y": 20}


def test_write_pair_is_full_snapshot(tmp_path):
    logic_path = tmp_path / "graphs" / "demo.graph.json"
    layout_path = tmp_path / "layout" / "demo.layout.json"
    write_graph_pair(logic_path, layout_path, _payload())
    assert json.loads(logic_path.read_text())["name"] == "demo"
    assert set(json.loads(layout_path.read_text())["positions"]) == {"a", "n1"}
    # Save again with the note deleted -> its layout entry is gone (no orphan).
    p2 = _payload()
    p2["nodes"] = [p2["nodes"][0]]
    write_graph_pair(logic_path, layout_path, p2)
    assert set(json.loads(layout_path.read_text())["positions"]) == {"a"}


def test_write_pair_removes_legacy(tmp_path):
    legacy = tmp_path / "graphs" / "demo.json"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("{}")
    logic_path = tmp_path / "graphs" / "demo.graph.json"
    layout_path = tmp_path / "layout" / "demo.layout.json"
    write_graph_pair(logic_path, layout_path, _payload(), legacy_path=legacy)
    assert logic_path.exists()
    assert not legacy.exists()  # upgraded to the pair
