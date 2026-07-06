"""Server-side logic/layout split for project directories (spec 6.1-6.4).

In project mode a saved graph is stored as a PAIR:
  graphs/<name>.graph.json   logic  {format_version, name, description,
                                      nodes[], edges[], presets[]}
  layout/<name>.layout.json  layout {format_version, positions{}, notes{},
                                      segmentGroups[]}

Non-project mode never calls this module (byte-for-byte single-file legacy).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1

# Note-node data keys that are canvas GEOMETRY (spec 6.2): extracted into the
# layout file so a re-bind / resize never dirties the reviewable logic file.
_NOTE_LAYOUT_KEYS = ("boundToNodeId", "boundOffset", "noteWidth", "noteHeight")


def _int_xy(pos: dict) -> dict:
    """Pin x/y to JSON integers so the round-trip is idempotent (spec 6.2)."""
    return {"x": int(round(pos.get("x", 0))), "y": int(round(pos.get("y", 0)))}


def split_graph(payload: dict) -> tuple[dict, dict]:
    """Split a merged graph dict (GraphData.model_dump()) into (logic, layout).

    The layout file is a COMPLETE SNAPSHOT of the current node ids (never a
    patch) so deleted-node orphans cannot accrete (spec 6.2).
    """
    logic_nodes: list[dict] = []
    positions: dict[str, dict] = {}
    notes: dict[str, dict] = {}

    for node in payload.get("nodes", []):
        nid = node.get("id")
        pos = node.get("position")
        if isinstance(pos, dict) and "x" in pos and "y" in pos:
            positions[nid] = _int_xy(pos)
        data = node.get("data")
        data = dict(data) if isinstance(data, dict) else {}
        if node.get("type") == "note":
            note_layout: dict[str, Any] = {}
            for k in _NOTE_LAYOUT_KEYS:
                if k in data:
                    note_layout[k] = data.pop(k)
            if note_layout:
                notes[nid] = note_layout
        logic_nodes.append({"id": nid, "type": node.get("type"), "data": data})

    logic = {
        "format_version": FORMAT_VERSION,
        "name": payload.get("name", "Untitled"),
        "description": payload.get("description", ""),
        "nodes": logic_nodes,
        "edges": payload.get("edges", []),
        "presets": payload.get("presets", []),
    }
    layout = {
        "format_version": FORMAT_VERSION,
        "positions": positions,
        "notes": notes,
        "segmentGroups": payload.get("segmentGroups", []),
    }
    return logic, layout


def merge_graph(logic: dict, layout: dict | None) -> tuple[dict, bool]:
    """Merge a logic + layout pair into the single-file shape the editor loads.

    Returns (merged, layout_missing). layout_missing is True when the layout
    file is absent OR any node ends up without a position; those nodes get NO
    position key so the frontend auto-layouts (spec 6.3). A logic file that
    still carries an embedded position (legacy / transitional `.graph.json`)
    is tolerated. Unknown ids in the layout file are ignored.
    """
    has_layout = isinstance(layout, dict)
    positions = layout.get("positions", {}) if has_layout else {}
    notes = layout.get("notes", {}) if has_layout else {}
    any_missing = not has_layout

    merged_nodes: list[dict] = []
    for node in logic.get("nodes", []):
        nid = node.get("id")
        data = node.get("data")
        data = dict(data) if isinstance(data, dict) else {}
        out: dict[str, Any] = {"id": nid, "type": node.get("type"), "data": data}
        pos = positions.get(nid)
        if pos is None and isinstance(node.get("position"), dict):
            pos = node["position"]
        if isinstance(pos, dict) and "x" in pos and "y" in pos:
            out["position"] = {"x": pos["x"], "y": pos["y"]}
        else:
            any_missing = True
        if node.get("type") == "note":
            note_layout = notes.get(nid, {})
            if isinstance(note_layout, dict):
                for k, v in note_layout.items():
                    data.setdefault(k, v)
        merged_nodes.append(out)

    merged = {
        "format_version": logic.get("format_version", FORMAT_VERSION),
        "name": logic.get("name", "Untitled"),
        "description": logic.get("description", ""),
        "nodes": merged_nodes,
        "edges": logic.get("edges", []),
        "presets": logic.get("presets", []),
        "segmentGroups": layout.get("segmentGroups", []) if has_layout else [],
        "layout_missing": any_missing,
    }
    return merged, any_missing


def _atomic_write(path: Path, text: str) -> None:
    """Write text via a temp file + os.replace (atomic per file, spec 13)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_graph_pair(
    logic_path: Path,
    layout_path: Path,
    payload: dict,
    legacy_path: Path | None = None,
) -> None:
    """Write the split pair atomically. LAYOUT first, LOGIC last (spec 13: the
    logic file is precious; a crash leaves a stale layout that self-heals via
    merge tolerance + auto-layout). A legacy single-file `<name>.json` is
    removed AFTER the logic write (upgrade-on-save, spec 6.4 / ID2).
    """
    logic, layout = split_graph(payload)
    _atomic_write(layout_path, json.dumps(layout, indent=2) + "\n")
    _atomic_write(logic_path, json.dumps(logic, indent=2) + "\n")
    if (
        legacy_path is not None
        and legacy_path != logic_path
        and legacy_path.exists()
    ):
        legacy_path.unlink()
