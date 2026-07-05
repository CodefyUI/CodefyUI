import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..core.graph_engine import GraphValidationError, validate_graph
from ..core.node_registry import registry
from ..core.secret_params import scrub_graph_secrets
from ..schemas import GraphData, GraphValidationResponse

router = APIRouter(prefix="/api/graph", tags=["graph"])


def _sanitize_name(name: str) -> str:
    """Filesystem-safe graph name: every char outside [alnum, '-', '_'] -> '_'."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _graph_path(name: str) -> Path:
    """Resolve a graph name to its on-disk JSON path under GRAPHS_DIR."""
    return settings.GRAPHS_DIR / f"{_sanitize_name(name)}.json"


@router.post("/validate", response_model=GraphValidationResponse)
async def validate(graph: GraphData):
    nodes = [n.model_dump() for n in graph.nodes]
    edges = [e.model_dump() for e in graph.edges]
    errors = validate_graph(nodes, edges)
    return GraphValidationResponse(valid=len(errors) == 0, errors=errors)


@router.post("/save")
async def save_graph(graph: GraphData):
    settings.GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    path = _graph_path(graph.name)
    payload = graph.model_dump()
    # Defense-in-depth: even if a client bypasses the editor (which already
    # blanks SECRET params before sending), never write a secret to disk.
    scrub_graph_secrets(payload.get("nodes", []))
    path.write_text(json.dumps(payload, indent=2))
    return {"message": "Graph saved", "path": str(path)}


@router.get("/load/{name}")
async def load_graph(name: str):
    path = _graph_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Graph '{name}' not found")
    data = json.loads(path.read_text())
    return data


@router.get("/list")
async def list_graphs():
    settings.GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    graphs = []
    for f in settings.GRAPHS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            graphs.append({"name": data.get("name", f.stem), "file": f.stem})
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

    # Validate the user-authored graph (presets are validated against the
    # preset registry rather than expanded here).
    errors = validate_graph(nodes, edges)
    if errors:
        raise HTTPException(status_code=400, detail=errors)

    # Expand preset:* nodes into their real sub-graphs so codegen sees the
    # actual training nodes (Dataset, DataLoader, Optimizer, Loss, etc.).
    try:
        for _ in range(10):  # support nested presets, same depth cap as execution
            if not any(n.get("type", "").startswith("preset:") for n in nodes):
                break
            nodes, edges, _ = expand_presets(nodes, edges)
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
