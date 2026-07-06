import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..core.graph_engine import GraphValidationError, build_preset_fallback, validate_graph
from ..core.node_registry import registry
from ..core.secret_params import scrub_graph_secrets
from ..schemas import GraphData, GraphValidationResponse

router = APIRouter(prefix="/api/graph", tags=["graph"])
logger = logging.getLogger(__name__)


def _sanitize_name(name: str) -> str:
    """Filesystem-safe graph name: every char outside [alnum, '-', '_'] -> '_'."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _project_mode() -> bool:
    return settings.PROJECT_DIR is not None


class GraphAmbiguityError(Exception):
    """Both `<name>.graph.json` and legacy `<name>.json` exist in project
    mode — never silently pick one (spec ID2/ID7)."""

    def __init__(self, name: str, canonical: "Path", legacy: "Path") -> None:
        self.name = name
        self.canonical = canonical
        self.legacy = legacy
        super().__init__(
            f"Graph '{name}' is ambiguous: both {canonical.name} and "
            f"{legacy.name} exist in {canonical.parent}. Remove one "
            "(the legacy single-file form upgrades to the pair on save)."
        )


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
    Project: canonical `<name>.graph.json` when present, else legacy
    `<name>.json`; raises GraphAmbiguityError when BOTH exist; when NEITHER
    exists returns the canonical (non-existent) path so callers' 404 path
    still fires.
    """
    safe = _sanitize_name(name)
    if not _project_mode():
        return settings.GRAPHS_DIR / f"{safe}.json"
    canonical = settings.GRAPHS_DIR / f"{safe}.graph.json"
    legacy = settings.GRAPHS_DIR / f"{safe}.json"
    if canonical.exists() and legacy.exists():
        raise GraphAmbiguityError(name, canonical, legacy)
    if legacy.exists() and not canonical.exists():
        return legacy
    return canonical


@router.post("/validate", response_model=GraphValidationResponse)
async def validate(graph: GraphData):
    nodes = [n.model_dump() for n in graph.nodes]
    edges = [e.model_dump() for e in graph.edges]
    errors = validate_graph(
        nodes, edges,
        preset_fallback=build_preset_fallback([p.model_dump() for p in graph.presets]),
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
    # Defense-in-depth: even if a client bypasses the editor (which already
    # blanks SECRET params before sending), never write a secret to disk.
    scrub_graph_secrets(payload.get("nodes", []))
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
                graphs.append({"name": data.get("name", f.stem), "file": f.stem})
            except Exception:
                continue
        return graphs

    # Project mode: canonical `<name>.graph.json` + legacy `<name>.json`.
    # Path("x.graph.json").stem is "x.graph" — strip the whole ".graph.json"
    # so the file stem never leaks the double suffix (spec ID7).
    files: dict[str, Path] = {}
    for f in settings.GRAPHS_DIR.glob("*.graph.json"):
        files.setdefault(f.name[: -len(".graph.json")], f)
    for f in settings.GRAPHS_DIR.glob("*.json"):
        if f.name.endswith(".graph.json"):
            continue  # already counted as canonical
        base = f.stem
        if base in files:
            # Both forms present -> fail loudly naming both (no silent pick).
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Graph '{base}' is ambiguous: both {files[base].name} "
                    f"and {f.name} exist. Remove one."
                ),
            )
        files[base] = f
    for base, f in sorted(files.items()):
        try:
            data = json.loads(f.read_text())
            graphs.append({"name": data.get("name", base), "file": base})
        except Exception:
            continue
    return graphs


@router.post("/export")
async def export_graph(graph: GraphData):
    """Export graph as a standalone Python script."""
    from ..core.codegen import generate_python
    from ..core.graph_engine import expand_presets, topological_sort

    nodes = [n.model_dump() for n in graph.nodes]
    edges = [e.model_dump() for e in graph.edges]

    # M4: parity with /save — never echo a SECRET param value into exported
    # source. codegen emits raw params in a comment for node types without a
    # template (e.g. LLMChat), so scrub before validation/expansion/codegen.
    # Scrubbing pre-expansion also blanks any secret embedded in a preset
    # node's internalParams before expand_presets injects it downstream.
    scrub_graph_secrets(nodes)

    # ID6: the graph's own presets[] resolve even when the server's preset
    # registry doesn't know them (portability).
    preset_fallback = build_preset_fallback([p.model_dump() for p in graph.presets])

    # Validate the user-authored graph (presets are validated against the
    # preset registry rather than expanded here).
    errors = validate_graph(nodes, edges, preset_fallback=preset_fallback)
    if errors:
        raise HTTPException(status_code=400, detail=errors)

    # Expand preset:* nodes into their real sub-graphs so codegen sees the
    # actual training nodes (Dataset, DataLoader, Optimizer, Loss, etc.).
    try:
        for _ in range(10):  # support nested presets, same depth cap as execution
            if not any(n.get("type", "").startswith("preset:") for n in nodes):
                break
            nodes, edges, _ = expand_presets(nodes, edges, preset_fallback=preset_fallback)
    except GraphValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        order = topological_sort(nodes, edges)
        script = generate_python(nodes, edges, order, name=graph.name)
    except GraphValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

    return {"script": script}
