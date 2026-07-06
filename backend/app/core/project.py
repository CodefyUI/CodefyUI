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
import subprocess
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # 3.10 backport — same API.

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


def git_provenance(project_dir: Path) -> tuple[str | None, bool | None]:
    """(full_commit_sha, dirty) for *project_dir*, or (None, None) when it is
    not a git repo / git is unavailable (this also covers the common
    fresh-scaffold state: `cdui project init` runs `git init` with NO initial
    commit, leaving an unborn HEAD that `rev-parse HEAD` reports as a
    non-zero exit). dirty = any uncommitted change, including untracked files
    (`git status --porcelain`).

    Both git invocations are decoded as utf-8 (errors replaced rather than
    raised) instead of the platform-default text encoding: on this project's
    actual Windows deployments (Traditional Chinese locale, cp950) the
    default encoding cannot decode arbitrary UTF-8 bytes, which would
    otherwise crash on a non-ASCII path in `git status` output. The status
    call is also individually guarded so a timeout/spawn failure there
    degrades to dirty=None instead of losing an already-resolved commit.
    """
    try:
        head = subprocess.run(
            ["git", "-C", str(project_dir), "rev-parse", "HEAD"],
            capture_output=True, encoding="utf-8", errors="replace", timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None, None
    if head.returncode != 0:
        return None, None
    commit = head.stdout.strip() or None
    if commit is None:
        return None, None
    try:
        status = subprocess.run(
            ["git", "-C", str(project_dir), "status", "--porcelain"],
            capture_output=True, encoding="utf-8", errors="replace", timeout=5)
    except (OSError, subprocess.SubprocessError):
        return commit, None
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return commit, dirty


def check_stale_pins_from_manifest(manifest: dict, lockfile: dict) -> list[str]:
    """Plugin ids that are pinned in the manifest but missing or sha-mismatched
    in the installed lockfile (spec 7.4)."""
    pins = manifest.get("plugins", {}) or {}
    installed = lockfile.get("plugins", {})
    stale: list[str] = []
    for pid, pin in pins.items():
        if not isinstance(pin, dict):
            continue
        entry = installed.get(pid)
        if entry is None:
            stale.append(pid)
        elif pin.get("sha") and entry.get("sha") != pin["sha"]:
            stale.append(pid)
    return stale


def check_stale_pins(project_dir: Path, lockfile: dict) -> list[str]:
    """Read the manifest from disk and return stale pin ids (empty on a missing
    or unparseable manifest)."""
    manifest_path = project_dir / "codefyui.project.toml"
    try:
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return []
    return check_stale_pins_from_manifest(manifest, lockfile)
