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
from typing import Any, NamedTuple

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


# ── Canonical-vs-legacy graph file rule (spec ID2/ID7) ─────────────────────
#
# In project mode a graph named <base> may exist on disk as the canonical
# `<base>.graph.json` or the legacy single-file `<base>.json`. The rule,
# defined ONCE here and consumed by the server routes, the `cdui project`
# CLI (validate/adopt), and the /list endpoint (issue #85):
#   - canonical when present;
#   - legacy accepted when only it exists (upgrade-on-save keeps it working);
#   - BOTH present is an error naming both files -- never silently pick one;
#   - `*.layout.json` is layout, never a graph.


class GraphAmbiguityError(Exception):
    """Both `<base>.graph.json` and legacy `<base>.json` exist for the same
    graph — never silently pick one (spec ID2/ID7)."""

    def __init__(self, name: str, canonical: Path, legacy: Path) -> None:
        self.name = name
        self.canonical = canonical
        self.legacy = legacy
        super().__init__(
            f"Graph '{name}' is ambiguous: both {canonical.name} and "
            f"{legacy.name} exist in {canonical.parent}. Remove one "
            "(the legacy single-file form upgrades to the pair on save)."
        )


def resolve_graph_file(
    graphs_dir: Path, base: str, display_name: str | None = None
) -> Path:
    """THE canonical-vs-legacy read rule for one graph *base* (filesystem)
    name: canonical `<base>.graph.json` when present; legacy `<base>.json`
    when only it exists; raises GraphAmbiguityError when BOTH exist; when
    NEITHER exists returns the canonical (non-existent) path so callers'
    not-found handling still fires.

    *display_name* only decorates the error message (e.g. the raw, pre-
    sanitize graph name a route received); it defaults to *base*.
    """
    canonical = graphs_dir / f"{base}.graph.json"
    legacy = graphs_dir / f"{base}.json"
    if canonical.exists() and legacy.exists():
        raise GraphAmbiguityError(display_name or base, canonical, legacy)
    if legacy.exists():
        return legacy
    return canonical


def collect_graph_files(graphs_dir: Path) -> list[tuple[str, Path]]:
    """Directory-wide form of the same rule: sorted `(base, file)` pairs for
    every graph in *graphs_dir* (non-recursive; empty when it is not a
    directory). `*.layout.json` files are skipped. Each base is resolved via
    resolve_graph_file, so the first base present in both forms raises
    GraphAmbiguityError before anything is returned.
    """
    if not graphs_dir.is_dir():
        return []
    bases: set[str] = set()
    for f in graphs_dir.glob("*.json"):
        if f.name.endswith(".layout.json"):
            continue
        if f.name.endswith(".graph.json"):
            bases.add(f.name[: -len(".graph.json")])
        else:
            bases.add(f.stem)
    return [(base, resolve_graph_file(graphs_dir, base))
            for base in sorted(bases)]


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


# ── Stale-pin rule (spec 7.4) ──────────────────────────────────────────────
#
# One classification of the manifest's `[plugins]` pins against the installed
# lockfile, consumed by BOTH the server startup warning (main.py) and
# `cdui project validate [--strict]` (scripts/project.py) -- issue #85.


class PinIssue(NamedTuple):
    """One problematic `[plugins]` pin.

    kind is one of:
      "missing"       pinned id absent from the installed lockfile
      "sha_mismatch"  pin carries a sha and the installed sha differs
      "malformed"     pin value is not a table -- unenforceable, so every
                      surface warns and SKIPS it (never stale, never a
                      --strict failure)
    """

    plugin_id: str
    kind: str
    pinned_sha: str | None = None
    installed_sha: str | None = None


def check_pin_issues(manifest: dict, lockfile: dict) -> list[PinIssue]:
    """THE stale-pin rule: classify every `[plugins]` pin in *manifest*
    against the installed *lockfile* (spec 7.4). A well-formed pin without a
    "sha" key only checks installed-ness. Manifest order is preserved."""
    pins = manifest.get("plugins", {}) or {}
    installed = lockfile.get("plugins", {})
    issues: list[PinIssue] = []
    for pid, pin in pins.items():
        if not isinstance(pin, dict):
            issues.append(PinIssue(pid, "malformed"))
            continue
        entry = installed.get(pid)
        pinned_sha = pin.get("sha")
        if entry is None:
            issues.append(PinIssue(pid, "missing", pinned_sha, None))
        elif pinned_sha and entry.get("sha") != pinned_sha:
            issues.append(
                PinIssue(pid, "sha_mismatch", pinned_sha, entry.get("sha")))
    return issues


def check_stale_pins_from_manifest(manifest: dict, lockfile: dict) -> list[str]:
    """Plugin ids that are pinned in the manifest but missing or sha-mismatched
    in the installed lockfile (spec 7.4). Malformed (non-table) pins are NOT
    stale -- they are reported separately by check_pin_issues and skipped."""
    return [i.plugin_id for i in check_pin_issues(manifest, lockfile)
            if i.kind != "malformed"]


def read_project_manifest(project_dir: Path) -> dict:
    """Parse `codefyui.project.toml`; {} on a missing or unparseable manifest
    (a diagnostics-path reader must never crash startup)."""
    manifest_path = project_dir / "codefyui.project.toml"
    try:
        return tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def check_stale_pins(project_dir: Path, lockfile: dict) -> list[str]:
    """Read the manifest from disk and return stale pin ids (empty on a missing
    or unparseable manifest)."""
    return check_stale_pins_from_manifest(
        read_project_manifest(project_dir), lockfile)
