"""Server-side logic/layout split for project directories (spec 6.1-6.4).

In project mode a saved graph is stored as a PAIR:
  graphs/<name>.graph.json   logic  {format_version, name, description,
                                      nodes[], edges[], presets[], subgraphs[],
                                      settings{} (only when a device or a
                                      seed is set)}
  layout/<name>.layout.json  layout {format_version, positions{}, notes{},
                                      segmentGroups[], subgraphPositions{}}

Non-project mode never uses the SPLIT (byte-for-byte single-file legacy),
but it does share one thing from this module: ``_atomic_write``. Saving and
renaming a single-file graph go through the same temp-file + ``os.replace``
mechanism, because atomicity is not a project-mode feature and because a
bare write leaves a stale directory entry on a case-insensitive filesystem
(see ``routes_graph.save_graph``). The writer stays here, where spec 13's
reasoning for it lives, rather than being copied to a second home.
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

#: The ``.gitignore`` a project directory starts with.
#:
#: Here rather than in ``scripts/project.py`` because there are now two ways
#: a project becomes a repository -- ``cdui project init`` at the command
#: line, and the Source Control tab's "Initialize Repository" button -- and
#: the file they write has to be the same file. The first line is the one
#: that matters most: ``.env`` holds the user's API keys, and the tab
#: refuses to serve it at any ref precisely because it must never be
#: committed in the first place.
PROJECT_GITIGNORE = """.env
*.pt
*.pth
*.safetensors
*.onnx
*.ckpt
*.pkl
__pycache__/
*.db
.DS_Store
# interrupted atomic writes
*.tmp-*
"""

#: The ``.gitattributes`` a project directory starts with.
#:
#: ``*.json text eol=lf`` is the load-bearing line: a project's graph and
#: layout files ARE its source, they are written by the server with ``\n``
#: endings, and a Windows checkout with ``core.autocrlf=true`` would
#: otherwise hand them back with ``\r\n`` -- so every save would rewrite
#: every line of a file nobody edited, and a review of a one-node change
#: would show the whole graph. ``linguist-generated`` keeps the layout half
#: of the pair collapsed in a GitHub diff, where it is noise.
PROJECT_GITATTRIBUTES = """*.json text eol=lf
layout/*.layout.json linguist-generated=true
"""

# Note-node data keys that are canvas GEOMETRY (spec 6.2): extracted into the
# layout file so a re-bind / resize never dirties the reviewable logic file.
_NOTE_LAYOUT_KEYS = ("boundToNodeId", "boundOffset", "noteWidth", "noteHeight")


#: The project manifest, the file that makes a directory a project.
PROJECT_MANIFEST = "codefyui.project.toml"

# The files `cdui project init` and the server's own project creation write
# (phase 4A moved them here from scripts/project.py, so both share them).
_PROJECT_TOML = """[project]
name = "{name}"
format_version = 1
requires_codefyui = ">=1.4"   # advisory metadata; NOT enforced

[plugins]
# written by `cdui project freeze`; installed by `cdui project restore`
# some-pack = {{ url = "https://github.com/owner/repo", ref = "v1.2", sha = "<40-hex>" }}

[publish]                      # optional single default target
# graph = "my-graph"
# slug = "my-service"
# record_io = true
"""

_ENV_EXAMPLE = """# CodefyUI project secrets (runtime only). Copy to .env and fill in.
# .env is gitignored; this .env.example is committed as the required-keys template.
#
# Loaded at server start with os.environ.setdefault semantics (an already-set
# environment variable wins). Values are execution-time only and are never
# written to a saved graph. CODEFYUI_* SERVER CONFIG keys placed here are
# IGNORED (server settings materialize before .env loads); pass `cdui start`
# flags or the process environment for those.
#
# LLM keys, read at node execute time:
# OPENAI_API_KEY=
# CODEFYUI_OPENAI_API_KEY=
# ANTHROPIC_API_KEY=
# CODEFYUI_ANTHROPIC_API_KEY=
"""

_README = """# {name}

A CodefyUI **project directory**: a self-contained git repo where the graph
files ARE the storage. Open it with:

    cdui start --project .

## Layout

    codefyui.project.toml   project manifest (name, plugin pins, default publish target)
    graphs/                 <name>.graph.json  -- logic (nodes/edges/params/presets)
    layout/                 <name>.layout.json -- node positions (reviewable, generated)
    assets/images/          IMAGES_DIR
    assets/models/          MODELS_DIR (weights land here; gitignored)
    assets/data/            CSVs / datasets / FileReader inputs
    assets/output/          ImageWriter output (created on demand)
    .env.example            committed template of required secret keys
    .env                    your secrets (gitignored, never committed)

## Quickstart

    cdui start --project .          # edit graphs in the browser; Save writes the pair
    cdui project validate .         # CI gate: every graph must pass the publish pre-flight
    cdui project publish . --graph my-graph --slug my-service --create   # first publish + provenance

A slug declared in `codefyui.project.toml` `[publish]` creates its app on
first `cdui project publish .` automatically; an explicitly passed `--slug`
needs `--create` the first time (so a typo cannot mint a second app).

## Version control

Set a git identity if you have not:

    git config user.name  "You"
    git config user.email "you@example.com"

Then:

    git add -A && git commit -m "Initial project"

Commit `.env.example`, never `.env`. Commit a small **fetch script** that
downloads large data on demand -- do NOT commit datasets or weights (they are
gitignored). Model checkpoints written to `assets/models/` and images written
to `assets/output/` are local artifacts, not source.

`cdui project freeze` rewrites `codefyui.project.toml` in place: keys you add
are preserved, but comments are not, and `[plugins]` is fully regenerated.

## Continuous integration

Run `cdui project restore .` THEN `cdui project validate .` (restore installs
the plugin pins the graphs need; validate needs the full backend env incl.
torch -- cache the venv in CI). Validate `.` -- do NOT feed a raw `*.json` glob
to the validator, or it would pick up `layout/*.layout.json` files too. A
canvas-only graph (no GraphOutput, e.g. a training graph) fails the publish
pre-flight; give it a real output or validate only your publish targets with
`cdui project validate . --graph <name>`.
"""


#: Directories a new project starts with; empty ones get a ``.gitkeep``.
PROJECT_DIRS: tuple[str, ...] = ("graphs", "layout", "assets/images", "assets/models", "assets/data")


def _write_if_absent(path: Path, content: str) -> bool:
    if path.exists():
        return False
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    return True


def scaffold_project(target: Path, name: str | None = None) -> list[str]:
    """Write a project's directories and starter files into *target*.

    Never overwrites a file that exists. Returns the files it wrote,
    relative to *target*. The caller decides whether *target* may be
    non-empty, and whether to make it a repository (:func:`init_project_repo`).
    """
    target.mkdir(parents=True, exist_ok=True)
    for rel in PROJECT_DIRS:
        (target / rel).mkdir(parents=True, exist_ok=True)
        # Keep otherwise-empty dirs trackable by git.
        if not any((target / rel).iterdir()):
            (target / rel / ".gitkeep").write_text("", encoding="utf-8")
    name = name or target.name
    files = {
        PROJECT_MANIFEST: _PROJECT_TOML.format(name=name),
        # Shared with the Source Control tab's own "Initialize Repository".
        ".gitignore": PROJECT_GITIGNORE,
        ".gitattributes": PROJECT_GITATTRIBUTES,
        ".env.example": _ENV_EXAMPLE,
        "README.md": _README.format(name=name),
    }
    return [rel for rel, text in files.items() if _write_if_absent(target / rel, text)]


def init_project_repo(target: Path) -> bool:
    """``git init`` *target* (no commit). False when git is not installed."""
    import shutil

    if not shutil.which("git"):
        return False
    if not (target / ".git").exists():
        subprocess.run(["git", "init"], cwd=target, capture_output=True)
    return True


class AdoptResult(NamedTuple):
    adopted: list[str]
    skipped: list[tuple[str, str]]


def adopt_graphs(src: Path, target: Path, *,
                 keep_existing: bool = False) -> AdoptResult:
    """Copy every graph in *src* into *target*'s ``graphs/`` and ``layout/``.

    The source files are only read. Raises :class:`GraphAmbiguityError`
    when *src* holds both ``x.json`` and ``x.graph.json``; an unreadable
    file is skipped and reported. *keep_existing* skips (and reports) a
    graph the project already has under the same name instead of
    replacing it -- the editor's import into an existing project.
    """
    graphs_dir = target / "graphs"
    layout_dir = target / "layout"
    # Shared canonical-vs-legacy rule: skips `*.layout.json`, and a source
    # holding BOTH `x.json` and `x.graph.json` aborts naming both files
    # instead of silently letting one overwrite the other (issue #85).
    pairs = collect_graph_files(src)
    adopted: list[str] = []
    skipped: list[tuple[str, str]] = []
    for base, path in pairs:
        if keep_existing and ((graphs_dir / f"{base}.graph.json").exists()
                              or (graphs_dir / f"{base}.json").exists()):
            skipped.append((path.name, "the project already has a graph "
                                       "with this name"))
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            skipped.append((path.name, str(exc)))
            continue
        write_graph_pair(graphs_dir / f"{base}.graph.json",
                         layout_dir / f"{base}.layout.json", payload)
        adopted.append(base)
    if adopted:
        # scaffold_project leaves `.gitkeep` placeholders; once real
        # graph/layout files have landed, drop the stray ones.
        for directory in (graphs_dir, layout_dir):
            (directory / ".gitkeep").unlink(missing_ok=True)
    return AdoptResult(adopted, skipped)


def _int_xy(pos: dict) -> dict:
    """Pin x/y to JSON integers so the round-trip is idempotent (spec 6.2)."""
    return {"x": int(round(pos.get("x", 0))), "y": int(round(pos.get("y", 0)))}


def _assigned_settings(raw: Any) -> dict:
    """The keys of a graph's ``settings`` block that are set.

    ``device`` counts when non-empty (an empty string is "no assignment", as
    in ``GraphSettings``); ``seed`` counts when it is not None, because 0 is
    a seed. A block that is not a dict, which only a hand-edited file holds,
    has none.
    """
    if not isinstance(raw, dict):
        return {}
    assigned: dict[str, Any] = {}
    if raw.get("device"):
        assigned["device"] = raw["device"]
    if raw.get("seed") is not None:
        assigned["seed"] = raw["seed"]
    return assigned


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

    # Subgraph definitions are LOGIC -- they are the block's wiring, and a
    # reviewer needs to see an edit to them. The positions of the nodes INSIDE
    # a definition are canvas geometry like any other, so they follow the same
    # rule the top level follows: dragging inside a sub-canvas must dirty only
    # the layout file (core#137, spec 6.2's rule applied one level down).
    logic_subgraphs: list[dict] = []
    subgraph_positions: dict[str, dict] = {}
    for definition in payload.get("subgraphs", []):
        if not isinstance(definition, dict):
            continue
        sid = definition.get("id")
        inner_positions: dict[str, dict] = {}
        inner_nodes: list[dict] = []
        for node in definition.get("nodes", []):
            if not isinstance(node, dict):
                continue
            nid = node.get("id")
            pos = node.get("position")
            if isinstance(pos, dict) and "x" in pos and "y" in pos:
                inner_positions[nid] = _int_xy(pos)
            inner_nodes.append({
                "id": nid,
                "type": node.get("type"),
                "data": node.get("data") if isinstance(node.get("data"), dict) else {},
            })
        stripped = dict(definition)
        stripped["nodes"] = inner_nodes
        logic_subgraphs.append(stripped)
        if sid is not None and inner_positions:
            subgraph_positions[sid] = inner_positions

    logic = {
        "format_version": FORMAT_VERSION,
        "name": payload.get("name", "Untitled"),
        "description": payload.get("description", ""),
        "nodes": logic_nodes,
        "edges": payload.get("edges", []),
        "presets": payload.get("presets", []),
        "subgraphs": logic_subgraphs,
    }
    # The graph's device and seed belong to the logic file (git-tracked).
    # Each is written only when set, so a graph with neither keeps its logic
    # file's shape.
    logic_settings = _assigned_settings(payload.get("settings"))
    if logic_settings:
        logic["settings"] = logic_settings
    layout = {
        "format_version": FORMAT_VERSION,
        "positions": positions,
        "notes": notes,
        "segmentGroups": payload.get("segmentGroups", []),
        "subgraphPositions": subgraph_positions,
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
            # Geometry-only absence intentionally does NOT set any_missing:
            # a note without a notes{} entry just falls back to default
            # size/binding. Only a missing *position* (above) flags
            # layout_missing and triggers the frontend auto-layout path.
            note_layout = notes.get(nid, {})
            if isinstance(note_layout, dict):
                for k, v in note_layout.items():
                    data.setdefault(k, v)
        merged_nodes.append(out)

    # Put sub-canvas positions back on the definitions' nodes. A missing entry
    # deliberately does NOT flag layout_missing, because flagging here would
    # send the whole TOP-LEVEL graph through auto-layout just because one block
    # had never been opened.
    #
    # That is only safe because the sub-canvas lays itself out: `enterSubgraph`
    # (frontend/src/store/tabStore.ts) checks the definition's raw nodes for a
    # usable position and runs the same auto-layout pass over the block's
    # contents when any is missing. Before core#137's review that claim was
    # made here but not implemented anywhere, and `resolveSerializedNodes`
    # defaults a missing position to the origin -- so an externally authored or
    # layout-less project opened every block as one pile of nodes at (0, 0).
    # If that check is ever removed from `enterSubgraph`, this decision has to
    # be revisited with it.
    subgraph_positions = (
        layout.get("subgraphPositions", {}) if has_layout else {}
    )
    merged_subgraphs: list[dict] = []
    for definition in logic.get("subgraphs", []):
        if not isinstance(definition, dict):
            continue
        inner_positions = subgraph_positions.get(definition.get("id"), {})
        inner_positions = inner_positions if isinstance(inner_positions, dict) else {}
        inner_nodes: list[dict] = []
        for node in definition.get("nodes", []):
            if not isinstance(node, dict):
                continue
            out_node = dict(node)
            pos = inner_positions.get(node.get("id"))
            if pos is None and isinstance(node.get("position"), dict):
                pos = node["position"]
            if isinstance(pos, dict) and "x" in pos and "y" in pos:
                out_node["position"] = {"x": pos["x"], "y": pos["y"]}
            inner_nodes.append(out_node)
        merged_definition = dict(definition)
        merged_definition["nodes"] = inner_nodes
        merged_subgraphs.append(merged_definition)

    merged = {
        "format_version": logic.get("format_version", FORMAT_VERSION),
        "name": logic.get("name", "Untitled"),
        "description": logic.get("description", ""),
        "nodes": merged_nodes,
        "edges": logic.get("edges", []),
        "presets": logic.get("presets", []),
        "subgraphs": merged_subgraphs,
        "segmentGroups": layout.get("segmentGroups", []) if has_layout else [],
        "layout_missing": any_missing,
    }
    merged_settings = _assigned_settings(logic.get("settings"))
    if merged_settings:
        merged["settings"] = merged_settings
    return merged, any_missing


def _atomic_write(path: Path, text: str) -> None:
    """Write text via a temp file + os.replace (atomic per file, spec 13).

    The temp file is unlinked on ANY failure (write or replace) so an
    interrupted save cannot strand a ``*.tmp-*`` orphan next to the pair
    (issue #88; the scaffold .gitignore additionally hides one in-repo).
    On success os.replace has already consumed the temp file, so the
    unlink is a no-op. That unlink is BEST EFFORT and never raises -- see
    the comment on it for what it would otherwise hide.

    os.replace also REPLACES the directory entry rather than reusing one,
    so the file is left spelled exactly as *path* spells it. That is the
    second reason to call this: a bare ``write_text`` to "my_graph.json"
    keeps an existing "My_Graph.json" entry on NTFS/APFS.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        # The cleanup must never become the error the caller sees. ``unlink``
        # can fail on its own account -- WinError 123 on an over-long path is
        # the one reproduced here -- and an exception raised from ``finally``
        # REPLACES the one propagating out of the ``try``. A disk-full or
        # permission failure on the write above would then surface as a
        # complaint about a temp file the caller never named, with the real
        # cause gone. Swallowing this strands at most one ``*.tmp-*`` orphan,
        # which is the lesser of the two and which the scaffold .gitignore
        # already hides.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


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
