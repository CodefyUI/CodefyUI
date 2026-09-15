import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ..config import settings
from ..core.graph_engine import GraphValidationError, build_preset_fallback, validate_graph
from ..core.project import (
    GraphAmbiguityError,
    collect_graph_files,
    resolve_graph_file,
)
from ..core.secret_params import (
    scrub_graph_secrets,
    scrub_preset_definition_secrets,
    scrub_subgraph_definition_secrets,
)
from ..schemas import (
    GraphData,
    GraphExportRequest,
    GraphValidationResponse,
)

router = APIRouter(prefix="/api/graph", tags=["graph"])
logger = logging.getLogger(__name__)


def _sanitize_name(name: str) -> str:
    """Filesystem-safe graph name: every char outside [alnum, '-', '_'] -> '_'."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _project_mode() -> bool:
    return settings.PROJECT_DIR is not None


def _reserved_graph_name(name: str) -> bool:
    """True when the (pre-sanitize) name would collide with the split
    suffixes.

    Checked against the RAW name, not `_sanitize_name(name)`: sanitization
    maps every '.' to '_', so a sanitized name can never contain a literal
    '.' and this check would be unreachable dead code if run post-sanitize.
    """
    return name.endswith(".graph") or name.endswith(".layout")


def _graph_logic_path(name: str) -> Path:
    """Canonical write target for a graph's LOGIC file.

    Non-project: `<GRAPHS_DIR>/<name>.json` (byte-for-byte legacy).
    Project:     `<GRAPHS_DIR>/<name>.graph.json`.
    """
    safe = _sanitize_name(name)
    if not _project_mode():
        return settings.GRAPHS_DIR / f"{safe}.json"
    return settings.GRAPHS_DIR / f"{safe}.graph.json"


def _graph_layout_path(name: str) -> "Path | None":
    """Project-mode LAYOUT file path, else None (non-project has no layout)."""
    if not _project_mode():
        return None
    return settings.LAYOUT_DIR / f"{_sanitize_name(name)}.layout.json"


def _graph_path(name: str) -> Path:
    """Resolve a graph name to the on-disk file to READ.

    Non-project: `<GRAPHS_DIR>/<name>.json` (existence not checked — callers
    guard with `.exists()`; identical to legacy behavior).
    Project: the shared canonical-vs-legacy rule
    (app.core.project.resolve_graph_file) — canonical `<name>.graph.json`
    when present, legacy `<name>.json` accepted, GraphAmbiguityError when
    BOTH exist, and the canonical (non-existent) path when NEITHER does so
    callers' 404 path still fires.
    """
    safe = _sanitize_name(name)
    if not _project_mode():
        return settings.GRAPHS_DIR / f"{safe}.json"
    return resolve_graph_file(settings.GRAPHS_DIR, safe, display_name=name)


@router.post("/validate", response_model=GraphValidationResponse)
async def validate(graph: GraphData):
    nodes = [n.model_dump() for n in graph.nodes]
    edges = [e.model_dump() for e in graph.edges]
    errors = validate_graph(
        nodes, edges,
        preset_fallback=build_preset_fallback([p.model_dump() for p in graph.presets]),
        subgraphs=[s.model_dump() for s in graph.subgraphs],
    )
    return GraphValidationResponse(valid=len(errors) == 0, errors=errors)


@router.post("/save")
async def save_graph(graph: GraphData):
    if _project_mode() and _reserved_graph_name(graph.name):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Graph name '{graph.name}' is reserved: names ending in "
                "'.graph' or '.layout' collide with the project file split."
            ),
        )
    settings.GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    payload = graph.model_dump()
    # ``settings`` is written only when a device is assigned, so a graph
    # file with no assignment stays byte-identical to what it was.
    if not (payload.get("settings") or {}).get("device"):
        payload.pop("settings", None)
    # Defense-in-depth: even if a client bypasses the editor (which already
    # blanks SECRET params before sending), never write a secret to disk.
    #
    # All THREE places a graph carries nodes, or the promise above is only
    # true for whichever the client happened to put the key in: top-level
    # nodes, portable preset definitions, and subgraph definitions (core#137,
    # which is what made `/save` start receiving definitions at all).
    portable_presets = payload.get("presets", [])
    scrub_preset_definition_secrets(portable_presets)
    preset_fallback = build_preset_fallback(portable_presets)
    scrub_graph_secrets(
        payload.get("nodes", []),
        preset_fallback=preset_fallback,
    )
    scrub_subgraph_definition_secrets(
        payload.get("subgraphs", []),
        preset_fallback=preset_fallback,
    )
    if _project_mode():
        from ..core.project import write_graph_pair
        logic_path = _graph_logic_path(graph.name)
        layout_path = _graph_layout_path(graph.name)
        legacy = settings.GRAPHS_DIR / f"{_sanitize_name(graph.name)}.json"
        write_graph_pair(logic_path, layout_path, payload, legacy_path=legacy)
        return {"message": "Graph saved", "path": str(logic_path)}
    path = _graph_path(graph.name)
    path.write_text(json.dumps(payload, indent=2))
    return {"message": "Graph saved", "path": str(path)}


@router.get("/load/{name}")
async def load_graph(name: str):
    try:
        path = _graph_path(name)
    except GraphAmbiguityError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Graph '{name}' not found")
    data = json.loads(path.read_text())
    # Legacy single-file `<name>.json` (project or non-project) loads verbatim
    # with embedded positions (spec 6.4); only the canonical pair is merged.
    if _project_mode() and path.name.endswith(".graph.json"):
        from ..core.project import merge_graph
        layout_path = _graph_layout_path(name)
        layout = None
        if layout_path is not None and layout_path.exists():
            try:
                layout = json.loads(layout_path.read_text())
            except (ValueError, OSError):
                layout = None
        data, _missing = merge_graph(data, layout)
    _warn_if_newer_format(name, data)
    return data


def _warn_if_newer_format(name: str, data: dict) -> None:
    from ..core.project import FORMAT_VERSION
    fmt = data.get("format_version", 1)
    if isinstance(fmt, int) and fmt > FORMAT_VERSION:
        # Read policy: warn, never block (ID8). The editor opens it read-only.
        logger.warning(
            "Graph '%s' has format_version %d newer than this build (%d) -- "
            "opening read-only", name, fmt, FORMAT_VERSION)


@router.get("/list")
async def list_graphs():
    settings.GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    graphs = []
    if not _project_mode():
        for f in settings.GRAPHS_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                graphs.append({
                    "name": data.get("name", f.stem),
                    "file": f.stem,
                    # Raw st_mtime (float epoch seconds), not an int: the
                    # Graphs panel's default order is most-recent-first, and
                    # truncating would shuffle two graphs saved within the
                    # same second into an arbitrary order.
                    "modified": f.stat().st_mtime,
                })
            except Exception:
                continue
        return graphs

    # Project mode: the shared canonical-vs-legacy collection rule
    # (app.core.project.collect_graph_files) — bases never leak the double
    # suffix (spec ID7) and both-forms-present fails loudly naming both.
    try:
        pairs = collect_graph_files(settings.GRAPHS_DIR)
    except GraphAmbiguityError as e:
        raise HTTPException(status_code=409, detail=str(e))
    for base, f in pairs:
        try:
            data = json.loads(f.read_text())
            graphs.append({
                "name": data.get("name", base),
                "file": base,
                # The LOGIC file's mtime -- which is what collect_graph_files
                # resolves to. The layout half is rewritten by every drag, so
                # sorting on it would rank "somebody nudged a node" above
                # "somebody rewired the graph".
                "modified": f.stat().st_mtime,
            })
        except Exception:
            continue
    return graphs


class RenameGraphRequest(BaseModel):
    """``POST /rename``'s body: ``{"from": ..., "to": ...}``.

    ``from`` is a Python keyword, so the field is ``from_name`` carrying the
    wire name as an alias; ``populate_by_name`` keeps the Python spelling
    usable as well, so an in-process caller is not forced to build a dict just
    to say ``from``. ``extra="forbid"`` for the reason routes_packs gives: a
    closed key set should be a property of the schema rather than of every
    handler remembering to check.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_name: str = Field(alias="from", min_length=1)
    to: str = Field(min_length=1)


@router.post("/rename")
async def rename_graph(req: RenameGraphRequest):
    """Rename a saved graph: the file, its layout half, and the ``name`` the
    file calls itself by (which is what the graphs list shows).

    Both names go through ``_sanitize_name`` via the path helpers, so a
    traversal in either direction lands harmlessly inside GRAPHS_DIR as
    underscores rather than escaping it.
    """
    # Same refusal /save gives, and for the same reason: in project mode a
    # graph named `x.graph` would write `x.graph.graph.json` and read back as
    # a graph called `x` with a stray suffix.
    if _project_mode() and _reserved_graph_name(req.to):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Graph name '{req.to}' is reserved: names ending in "
                "'.graph' or '.layout' collide with the project file split."
            ),
        )
    try:
        src = _graph_path(req.from_name)
        # Resolved rather than just probing the write target: this is what
        # catches renaming a canonical pair onto a base that exists only in
        # the legacy form (and the reverse), which would leave both forms of
        # one base on disk -- the state GraphAmbiguityError exists to forbid.
        dst_existing = _graph_path(req.to)
    except GraphAmbiguityError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if not src.exists():
        raise HTTPException(
            status_code=404, detail=f"Graph '{req.from_name}' not found")
    if dst_existing.exists() and dst_existing != src:
        raise HTTPException(
            status_code=409, detail=f"Graph '{req.to}' already exists")
    # The layout half gets the same refusal, checked here so that -- like
    # every check above it -- nothing has moved yet when it fires. A layout
    # can outlive its graph (a `git checkout` of `graphs/` alone, a manual
    # delete, a partial revert), and `Path.replace` overwrites the
    # destination silently on POSIX and on Windows alike, so without this the
    # orphan would be destroyed by a rename the logic half had waved through.
    layout_src = _graph_layout_path(req.from_name)
    layout_dst = _graph_layout_path(req.to)
    if (
        layout_dst is not None
        and layout_dst != layout_src
        and layout_dst.exists()
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                f"A layout file for '{req.to}' already exists without its "
                "graph; remove or restore it before renaming onto that name."
            ),
        )

    raw = src.read_text()
    try:
        data = json.loads(raw)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        # /list silently skips a file it cannot parse, so the panel never
        # offers to rename one. Getting here means a hand-edited file: say so
        # rather than answering a 500 from json.loads.
        raise HTTPException(
            status_code=400,
            detail=f"Graph '{req.from_name}' is not a readable graph file",
        )
    data["name"] = req.to

    safe_to = _sanitize_name(req.to)
    # Rename into the form the file is ALREADY in: a legacy single-file graph
    # stays legacy and upgrades to the pair on its next /save (spec 6.4/ID2).
    # Splitting it here instead would mean re-deriving a layout from embedded
    # positions, which is /save's job and needs the merged shape.
    keeps_pair = _project_mode() and src.name.endswith(".graph.json")
    dst = settings.GRAPHS_DIR / (
        f"{safe_to}.graph.json" if keeps_pair else f"{safe_to}.json")

    # project.py's own writer rather than a second tmp+os.replace here: spec
    # 13's atomicity should stay one mechanism, and a truncated logic file is
    # a lost graph.
    from ..core.project import _atomic_write

    # Move first, rewrite second. The other order leaves a file whose `name`
    # disagrees with its filename when the move then fails; this one leaves a
    # graph that is whole and loadable under its new name, only still titled
    # the old one. The trailing newline is preserved exactly as found so a
    # rename does not surface in a project's git diff as a last-line change.
    if dst != src:
        src.replace(dst)
    _atomic_write(
        dst, json.dumps(data, indent=2) + ("\n" if raw.endswith("\n") else ""))

    # The layout file is keyed by base name, so leaving it behind would orphan
    # it under a name no graph answers to any more. Moved for a legacy source
    # too: if one exists beside a legacy file it is still that graph's. The
    # destination was proved free above, so this move can only create a file.
    if (
        layout_src is not None
        and layout_dst is not None
        and layout_dst != layout_src
        and layout_src.exists()
    ):
        layout_dst.parent.mkdir(parents=True, exist_ok=True)
        layout_src.replace(layout_dst)

    # `file` is what the caller addresses the graph by from here on -- a
    # client holding the old base (the editor's bound graph, say) has no other
    # way to learn the sanitized new one.
    return {"message": "Graph renamed", "name": req.to, "file": safe_to}


# Declared after the literal routes above so a future POST/GET at `/{name}`
# cannot swallow `/rename` or `/list`.
@router.delete("/{name}")
async def delete_graph(name: str):
    """Delete a saved graph, both halves of it.

    404 rather than a quiet success when nothing was there: the panel showed
    the user a row, and "another tab deleted it first" is something they
    should see.
    """
    try:
        path = _graph_path(name)
    except GraphAmbiguityError as e:
        # The same refusal /load gives. Never guess which of the two files the
        # user meant, least of all when the answer decides which one is gone.
        raise HTTPException(status_code=409, detail=str(e))
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Graph '{name}' not found")
    removed = [path]
    path.unlink()
    if _project_mode():
        # `path` IS the legacy single `<name>.json` whenever that is the form
        # on disk (resolve_graph_file returns it, and the both-forms case
        # raised above), so the legacy file needs no unlink of its own. The
        # layout half does: it is not a graph file, so it never takes part in
        # that resolution at all.
        layout_path = _graph_layout_path(name)
        if layout_path is not None and layout_path.exists():
            layout_path.unlink()
            removed.append(layout_path)
    return {"message": "Graph deleted", "removed": [str(p) for p in removed]}


@router.post("/export")
async def export_graph(graph: GraphExportRequest):
    """Export a graph as a single-file, headless CodefyUI Python runner."""
    from ..core.codegen import generate_python
    from ..core.graph_engine import prepare_executable_graph

    # Notes are persisted layout annotations, not executable nodes.  The
    # canvas execution path filters them too; accept saved/imported graph JSON
    # directly instead of making API callers duplicate that UI-only cleanup.
    note_ids = {node.id for node in graph.nodes if node.type == "note"}
    nodes = [n.model_dump() for n in graph.nodes if n.id not in note_ids]
    edges = [
        e.model_dump()
        for e in graph.edges
        if e.source not in note_ids and e.target not in note_ids
    ]

    # M4: parity with /save — never embed a SECRET param value. Portable
    # definitions carry defaults separately, while preset instances carry
    # per-inner-node overrides in internalParams, so scrub both representations.
    presets = [p.model_dump() for p in graph.presets]
    scrub_preset_definition_secrets(presets)
    preset_fallback = build_preset_fallback(presets)
    scrub_graph_secrets(nodes, preset_fallback=preset_fallback)
    # Subgraph internals are ordinary nodes with ordinary params, so they get
    # the same secret scrub the top level gets -- an exported file must never
    # carry a key just because the node holding it sits inside a block. Same
    # `preset_fallback` as the top-level call: a preset node inside a block
    # keeps its secret slots in `internalParams`, and identifying those slots
    # needs the definition, which for a graph-embedded preset exists only in
    # this request's own `presets[]`.
    subgraphs = [s.model_dump() for s in graph.subgraphs]
    scrub_subgraph_definition_secrets(
        subgraphs, preset_fallback=preset_fallback
    )

    try:
        prepare_executable_graph(
            nodes,
            edges,
            preset_fallback=preset_fallback,
            subgraphs=subgraphs,
        )
    except GraphValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        script = generate_python(
            nodes,
            edges,
            name=graph.name,
            presets=presets,
            subgraphs=subgraphs,
            # core#136: the canvas seed travels with the export, so an
            # exported augmenting graph reproduces the crops the canvas
            # produced instead of drawing fresh entropy every invocation.
            seed=graph.seed,
            deterministic=graph.deterministic,
            # The graph's own device is the exported ``--device`` default.
            device=graph.settings.device if graph.settings else None,
        )
        # A successful response must never download syntactically broken
        # Python, even if a future template edit regresses quoting/bracketing.
        compile(script, "<CodefyUI Python export>", "exec")
    except GraphValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

    return {"script": script}
