import json

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..core.node_base import ParamType
from ..core.node_registry import registry as node_registry
from ..core.preset_registry import preset_registry
from ..core.secret_params import scrub_graph_secrets
from ..schemas import CreatePresetRequest, PresetDefinition

router = APIRouter(prefix="/api/presets", tags=["presets"])


@router.get("", response_model=list[PresetDefinition])
async def list_presets():
    return preset_registry.all()


@router.get("/{name}", response_model=PresetDefinition)
async def get_preset(name: str):
    preset = preset_registry.get(name)
    if not preset:
        raise HTTPException(status_code=404, detail=f"Preset '{name}' not found")
    return preset


@router.post("/create", response_model=PresetDefinition)
async def create_preset(request: CreatePresetRequest):
    """Export a graph as a reusable subgraph/preset.

    Auto-detects exposed ports (unconnected ports) and exposed params.
    """
    if not request.nodes:
        raise HTTPException(status_code=400, detail="Graph must have at least one node")

    if preset_registry.get(request.name):
        raise HTTPException(status_code=409, detail=f"Preset '{request.name}' already exists")

    # I3: never persist a SECRET param value into a preset definition file.
    # Blank secrets in the incoming graph (both data.params and any
    # preset-embedded data.internalParams) before any of it is copied into
    # the stored preset. C1 (below) additionally keeps SECRET params out of
    # the exposed-params schema; this guards the raw VALUES.
    scrub_graph_secrets(request.nodes)

    # Create short ID mapping for cleaner JSON
    id_map: dict[str, str] = {}
    for i, node in enumerate(request.nodes):
        old_id = node.get("id", f"node_{i}")
        id_map[old_id] = f"node_{i}"

    # Transform nodes
    internal_nodes = []
    for node in request.nodes:
        old_id = node.get("id", "")
        node_type: str = node.get("type", "")
        params = node.get("data", {}).get("params", {})
        internal_nodes.append({
            "id": id_map.get(old_id, old_id),
            "type": node_type,
            "params": params,
        })

    # Transform edges
    internal_edges = []
    for edge in request.edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        internal_edges.append({
            "source": id_map.get(src, src),
            "target": id_map.get(tgt, tgt),
            "sourceHandle": edge.get("sourceHandle", ""),
            "targetHandle": edge.get("targetHandle", ""),
        })

    # Auto-detect exposed ports (unconnected ports)
    connected_inputs: set[tuple[str, str]] = set()
    connected_outputs: set[tuple[str, str]] = set()
    for edge in internal_edges:
        connected_outputs.add((edge["source"], edge["sourceHandle"]))
        connected_inputs.add((edge["target"], edge["targetHandle"]))

    exposed_inputs = []
    exposed_outputs = []
    exposed_params = []

    for node in internal_nodes:
        node_cls = node_registry.get(node["type"])
        if not node_cls:
            continue

        # Unconnected input ports → exposed inputs
        for port in node_cls.define_inputs():
            if (node["id"], port.name) not in connected_inputs:
                # Build unique name: use node_id prefix if multiple nodes expose same port name
                exposed_inputs.append({
                    "name": f"{node['id']}_{port.name}",
                    "internal_node": node["id"],
                    "internal_port": port.name,
                    "data_type": port.data_type.value,
                    "description": f"{node['type']}: {port.description}",
                })

        # Unconnected output ports → exposed outputs
        for port in node_cls.define_outputs():
            if (node["id"], port.name) not in connected_outputs:
                exposed_outputs.append({
                    "name": f"{node['id']}_{port.name}",
                    "internal_node": node["id"],
                    "internal_port": port.name,
                    "data_type": port.data_type.value,
                    "description": f"{node['type']}: {port.description}",
                })

        # All params → exposed params (grouped by node type)
        for param in node_cls.define_params():
            # C1: never expose a SECRET param (API key) as a preset param. A
            # masked field in the preset config modal would let a user type a
            # key that round-trips into the preset node's internalParams and
            # leaks to disk. The inner node keeps its env-var fallback at
            # runtime, so the preset still works with env keys.
            if param.param_type == ParamType.SECRET:
                continue
            exposed_params.append({
                "internal_node": node["id"],
                "param_name": param.name,
                "display_name": f"{node['type']} - {param.name}",
                "group": node["type"],
            })

    if not exposed_inputs and not exposed_outputs:
        raise HTTPException(
            status_code=400,
            detail="Subgraph has no unconnected ports — it needs at least one exposed input or output",
        )

    # Build preset data
    preset_data = {
        "preset_name": request.name,
        "category": request.category,
        "description": request.description,
        "tags": request.tags,
        "nodes": internal_nodes,
        "edges": internal_edges,
        "exposed_inputs": exposed_inputs,
        "exposed_outputs": exposed_outputs,
        "exposed_params": exposed_params,
    }

    # Save to file
    filename = request.name.lower().replace(" ", "_").replace("/", "_") + ".json"
    filepath = settings.PRESETS_DIR / filename
    settings.PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(preset_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # Reload presets
    preset_registry.clear()
    preset_registry.discover(settings.PRESETS_DIR, node_registry)

    preset = preset_registry.get(request.name)
    if not preset:
        raise HTTPException(status_code=500, detail="Failed to load created preset")
    return preset
