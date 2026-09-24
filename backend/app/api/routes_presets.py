import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..core.data_paths import UnstorableName, check_file_name, resolve_under
from ..core.node_base import ParamType
from ..core.graph_engine import subgraph_id_of
from ..core.node_registry import registry as node_registry
from ..core.preset_registry import preset_registry
from ..core.secret_params import scrub_graph_secrets
from ..schemas import CreatePresetRequest, PresetDefinition

router = APIRouter(prefix="/api/presets", tags=["presets"])


def _coded(status_code: int, code: str, **fields: Any) -> HTTPException:
    """A refusal the panel can act on: ``detail`` is a dict with a ``code``.

    The same shape ``routes_plugins`` answers in, and for the same reason:
    a client that has to tell "pick another name" from "that name is a
    path" cannot do it by matching on a sentence that will be translated.
    The fields beside the code (``character``, ``reserved``, ``filename``)
    are what the message the user reads gets built from. Deliberately no
    ``message``: prose about a coded refusal belongs to whoever is talking
    to the user, in their language.

    Only the #476 name refusals answer this way. The refusals that were
    already here (no nodes, a subgraph instance, no unconnected ports, a
    duplicate name) keep their prose ``detail``, because the editor renders
    those strings today and rewriting them is not this fix.
    """
    return HTTPException(status_code=status_code,
                         detail={"code": code, **fields})


def _preset_filename(name: str) -> str:
    """The file a preset called *name* is stored in, or a coded 400 (#476).

    Half of the answer to "the name became a path". This half runs BEFORE
    anything is joined, because by the time a string is a ``Path`` the
    damage is already expressed in it: ``PRESETS_DIR / "C:\\\\evil"`` is not
    a suspicious path under ``PRESETS_DIR``, it is ``C:\\evil`` -- the join
    threw the left side away. :func:`_resolved_under` is the other half.

    Which names are refused is :func:`app.core.data_paths.check_file_name`,
    the rule the upload routes share since #520; this route answers its
    ``code`` and ``fields``, which the toolbar turns into a sentence in the
    user's language. The characters are checked in the name as TYPED, so a
    refusal points at one the user can see; the length and device names in
    the file actually written, ``.json`` included, since that is what has to
    fit the file system (``name_too_long`` answers ``limit`` for that file
    name).

    What survives, and what does not
    --------------------------------
    The lowercasing and the space-to-underscore are the behaviour this
    endpoint always had, kept exactly: preset titles here are routinely
    Traditional Chinese with spaces, and a guard that renamed or refused
    those would cost more than the hole it closed.

    The ``/``-to-underscore is NOT kept. It looked like sanitising and was
    really a silent rename: ``Vision/Classifier`` and ``Vision Classifier``
    both became ``vision_classifier.json``, so whichever was saved second
    took the first one's file with no warning. A name this endpoint cannot
    store under is now something the user is told about.
    """
    filename = name.lower().replace(" ", "_") + ".json"
    try:
        check_file_name(name, stored_as=filename)
    except UnstorableName as refusal:
        raise _coded(400, refusal.code, **refusal.fields) from None
    return filename


def _resolved_under(directory: Path, filename: str) -> Path:
    """*filename* resolved as a DIRECT CHILD of *directory*, or a coded 400.

    The other half of #476, and the half that holds if the first one is
    ever loosened: whatever the rules above let through, the path that gets
    written is re-derived here and confirmed to be inside the directory it
    was supposed to be inside. Nothing should reach this and be refused --
    that is what makes it worth keeping, not what makes it redundant.

    ``direct_child=True``: the containment check AND a refusal of a nested
    path, which a preset file is never allowed to be (the registry globs one
    flat directory, so a preset written into a subdirectory would vanish
    from the panel).

    The rule is :func:`app.core.data_paths.resolve_under` (#483).
    """
    target = resolve_under(directory, filename, direct_child=True)
    if target is None:
        raise _coded(400, "name_escapes_presets_dir")
    return target


def _preset_path(name: str) -> Path:
    """Where a preset called *name* is written. Checked twice; see #476."""
    return _resolved_under(settings.PRESETS_DIR, _preset_filename(name))


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

    # #476: the name becomes a filename here, before it becomes anything
    # else. Early on purpose -- everything below this line is work done on
    # behalf of a name that may turn out to be unstorable, and a refusal
    # that arrives after the graph has been walked is a refusal that had to
    # walk the graph first.
    filepath = _preset_path(request.name)

    if preset_registry.get(request.name):
        raise HTTPException(status_code=409, detail=f"Preset '{request.name}' already exists")

    # A name the registry does not know can still land on a file that is
    # already taken, because the registry's key is the name as TYPED and
    # the filename is lowercased: with "LLM Preset" saved, "llm preset"
    # passed the check above and then wrote straight over its file, taking
    # its place in the registry on the rediscovery below. The file is the
    # authority on what is already there.
    if filepath.exists():
        raise _coded(409, "preset_file_exists", filename=filepath.name)

    # core#137: a preset is portable and a subgraph id is local to one graph,
    # so a `subgraph:<id>` node baked into a preset is a reference that can
    # never resolve anywhere else -- and does not resolve here either, since
    # expansion runs subgraphs before presets and never revisits. The editor
    # refuses this client-side; this is the same rule at the route, for a
    # hand-rolled request. `graph_engine.preset_subgraph_errors` catches the
    # ones that arrive inside a graph file's own `presets[]`.
    instance_ids = [
        node.get("id", "")
        for node in request.nodes
        if subgraph_id_of(node.get("type", "")) is not None
    ]
    if instance_ids:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Preset '{request.name}' cannot contain subgraph "
                f"instance(s): {', '.join(sorted(instance_ids))}. A subgraph "
                "definition is local to one graph, so it cannot travel with "
                "a preset -- expand the block first."
            ),
        )

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

        # Unconnected input ports → exposed inputs.
        #
        # #196: the DYNAMIC form, because the static one answers for the
        # DEFAULT params. A ComposeTransform(steps=5) reports two ports
        # through `define_inputs()` however many it actually has, so
        # step_3..step_5 never reached `exposed_inputs` and the edges into
        # them had nowhere to reattach when the preset was dropped back onto
        # a canvas. Same for PythonScript, whose `input_ports` /
        # `output_ports` params drive both directions.
        for port in node_cls.define_inputs_dynamic(node["params"]):
            if (node["id"], port.name) not in connected_inputs:
                # Build unique name: use node_id prefix if multiple nodes expose same port name
                exposed_inputs.append({
                    "name": f"{node['id']}_{port.name}",
                    "internal_node": node["id"],
                    "internal_port": port.name,
                    "data_type": port.data_type.value,
                    "description": f"{node['type']}: {port.description}",
                })

        # Unconnected output ports → exposed outputs (see above).
        for port in node_cls.define_outputs_dynamic(node["params"]):
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

    # Save to file (`filepath` was derived and checked at the top, #476)
    settings.PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(preset_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # Reload presets
    preset_registry.clear()
    preset_registry.discover(settings.PRESETS_DIR, node_registry)

    preset = preset_registry.get(request.name)
    if not preset:
        raise HTTPException(status_code=500, detail="Failed to load created preset")
    return preset
