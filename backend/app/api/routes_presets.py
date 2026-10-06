import heapq
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..core.data_paths import UnstorableName, check_file_name, resolve_under
from ..core.node_base import ParamType
from ..core.graph_engine import (
    GraphValidationError,
    build_preset_fallback,
    container_bypass_errors,
    expand_presets,
    is_note_node,
    preset_subgraph_errors,
    resolve_bypass,
    subgraph_id_of,
)
from ..core.node_registry import registry as node_registry
from ..core.preset_registry import preset_registry
from ..core.secret_params import scrub_graph_secrets
from ..schemas import CreatePresetRequest, PresetDefinition

logger = logging.getLogger(__name__)

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

    The name refusals (#476) answer this way, and since #618 so do the
    refusals the editor puts into words itself: ``preset_exists`` (a name
    the registry already holds), ``preset_empty``, ``preset_no_ports``,
    ``preset_card_unknown`` and ``preset_node_unknown``. Two kinds keep a
    prose ``detail``: a subgraph instance, which the editor refuses before
    it asks for a name, and the engine's own sentence about a card or a
    muted node -- the one a Run of the same canvas shows.
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
    """Where a preset called *name* is written. Checked twice; see #476.

    ``USER_PRESETS_DIR``, never the built-in ``PRESETS_DIR`` (#600).
    """
    return _resolved_under(settings.USER_PRESETS_DIR, _preset_filename(name))


def _preset_contents(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The part of a canvas that goes into a preset (#600).

    Export takes the whole canvas, and a canvas needs a Start to be run and
    checked, so Start is left out rather than refused, with every trigger
    wire. Nothing is lost: Start does no work, and a placed card starts its
    inner roots on its own trigger (#561). Kept, the trigger wire was stored
    without its type and expansion brought it back as a DATA edge into a port
    called ``__trigger``, so a graph using the card validated clean and then
    failed to run. A note is left out too: it is not a node, and the registry
    refuses to load a preset that holds one.

    A wire counts as a trigger by its ``type``, by its ``__trigger`` end (a
    hand-rolled request may leave the type out) or by an end on a node left
    out. Block instances are refused below instead, because leaving one out
    WOULD lose part of what the preset computes. Preset cards and muted
    nodes stay in here; :func:`_flattened` turns them into what a run
    executes once the name has been checked (#618).
    """
    def left_out(node: dict[str, Any]) -> bool:
        return node.get("type") == "Start" or is_note_node(node)

    gone = {node.get("id") for node in nodes if left_out(node)}
    kept_edges = [
        edge for edge in edges
        if edge.get("type") != "trigger"
        and edge.get("targetHandle") != "__trigger"
        and edge.get("source") not in gone
        and edge.get("target") not in gone
    ]
    return [node for node in nodes if not left_out(node)], kept_edges


#: How many times nested preset cards are expanded, the budget
#: :func:`~app.core.graph_engine.prepare_executable_graph` gives a run.
_PRESET_DEPTH = 10


def _flattened(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    presets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The canvas the way a run of it executes it (#618).

    A preset card becomes its own nodes, with the settings made on the card,
    and a muted node is left out, with what fed it handed to the nodes after
    it. A stored preset can hold neither: the registry resolves inner types
    through the node registry alone, so a ``preset:<name>`` inner node made
    the file unloadable, and an inner node has no field for a mute, so the
    muted node ran. Both are done by the engine's own ``expand_presets`` and
    ``resolve_bypass``, in the order and to the depth
    :func:`~app.core.graph_engine.prepare_executable_graph` uses, so the
    preset holds what a run of the canvas executes and needs no other preset
    to stay installed. *presets* are the graph's own definitions, which win
    over an installed preset of the same name, as they do in a run. The
    nodes come back in canvas order, a card's nodes where the card stood.

    Refused with a code: a card whose preset neither the graph nor this
    server has (before anything is expanded), and a node type this server
    does not have (once everything is). Every other refusal is the engine's
    sentence, the one a Run of the canvas shows: a muted card, a definition
    that cannot be read or names a block, an edge a card cannot carry,
    nesting past the budget, a muted node with nothing to forward.
    """
    muted_containers = container_bypass_errors(nodes)
    if muted_containers:
        raise HTTPException(status_code=400, detail="; ".join(muted_containers))

    fallback = build_preset_fallback(presets)
    for node in nodes:
        node_type = str(node.get("type", ""))
        if not node_type.startswith("preset:"):
            continue
        name = node_type[len("preset:"):]
        if name not in fallback and preset_registry.get(name) is None:
            raise _coded(400, "preset_card_unknown", preset=name)

    def holds_a_card(graph_nodes: list[dict[str, Any]]) -> bool:
        return any(
            str(node.get("type", "")).startswith("preset:")
            for node in graph_nodes
        )

    for _ in range(_PRESET_DEPTH):
        if not holds_a_card(nodes):
            break
        # On every pass, not only the first: a definition nested in another
        # can name a block as well, and a block left in the stored preset
        # makes the file unloadable.
        refusals = preset_subgraph_errors(nodes, fallback)
        if refusals:
            raise HTTPException(status_code=400, detail="; ".join(refusals))
        try:
            nodes, edges, _ = expand_presets(
                nodes, edges, preset_fallback=fallback)
        except GraphValidationError as exc:
            # ``str``: the message may be a ``ValidationIssue`` (#561).
            raise HTTPException(status_code=400, detail=str(exc)) from None
    if holds_a_card(nodes):
        raise HTTPException(
            status_code=400,
            detail=f"Preset nesting exceeds the maximum depth of {_PRESET_DEPTH}",
        )

    bypass = resolve_bypass(nodes, edges)
    if bypass.errors:
        raise HTTPException(status_code=400, detail="; ".join(bypass.errors))

    # A node type this server does not have -- a plugin's node whose pack is
    # missing or disabled, which a workspace import opens as a placeholder --
    # cannot be stored: the registry refuses to load a file that names one
    # (``bypass`` leaves such a node in place, muted or not). Refused here,
    # by type, rather than written and then refused with a 500.
    for node in bypass.nodes:
        node_type = str(node.get("type") or "")
        if node_registry.get(node_type) is None:
            raise _coded(400, "preset_node_unknown", type=node_type)
    return bypass.nodes, bypass.edges


def _in_flow_order(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """*nodes* in the order data flows through them (#618).

    The stored node order is the order the preset's dialog lists the nodes
    in, and expansion opens a card up where it stood on the canvas, so a
    card placed before the node feeding it was listed ahead of it. Each node
    now comes after every node that feeds it and otherwise keeps its place,
    so an independent chain stays together. The nodes of a cycle, which has
    no such order (a run refuses it), go last, in canvas order.

    For the stored nodes and params only: the exposed ports keep canvas
    order, because Map runs a preset through its first exposed input and
    output (``map_node.py``).
    """
    place: dict[Any, int] = {}
    for index, node in enumerate(nodes):
        place.setdefault(node.get("id"), index)
    feeds: dict[int, list[int]] = defaultdict(list)
    waiting = [0] * len(nodes)
    for edge in edges:
        source, target = place.get(edge.get("source")), place.get(edge.get("target"))
        if source is None or target is None or source == target:
            continue
        feeds[source].append(target)
        waiting[target] += 1

    # The earliest node on the canvas among those whose feeders are all
    # placed goes next.
    ready = [index for index, count in enumerate(waiting) if count == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        index = heapq.heappop(ready)
        order.append(index)
        for fed in feeds[index]:
            waiting[fed] -= 1
            if waiting[fed] == 0:
                heapq.heappush(ready, fed)
    placed = set(order)
    order += [index for index in range(len(nodes)) if index not in placed]
    return [nodes[index] for index in order]


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

    Auto-detects exposed ports (unconnected ports) and exposed params. Start
    nodes, trigger wires and notes are left out (:func:`_preset_contents`).
    A preset card is copied in as its own nodes and a muted node is left
    out, both with the engine's own helpers (:func:`_flattened`).
    """
    nodes, edges = _preset_contents(request.nodes, request.edges)
    if not nodes:
        raise _coded(400, "preset_empty")

    # #476: the name becomes a filename here, before it becomes anything
    # else. Early on purpose -- everything below this line is work done on
    # behalf of a name that may turn out to be unstorable, and a refusal
    # that arrives after the graph has been walked is a refusal that had to
    # walk the graph first.
    filepath = _preset_path(request.name)

    if preset_registry.get(request.name):
        raise _coded(409, "preset_exists", name=request.name)

    # A name the registry does not know can still land on a file that is
    # already taken, because the registry's key is the name as TYPED and
    # the filename is lowercased: with "LLM Preset" saved, "llm preset"
    # passed the check above and then wrote straight over its file, taking
    # its place in the registry. The file is the authority on what is
    # already there, in the built-in dir as well (#600), which also holds
    # the presets older versions exported: nothing there is overwritten,
    # but the new preset would be a second one whose name differs in case.
    if filepath.exists() or (settings.PRESETS_DIR / filepath.name).exists():
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
        for node in nodes
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

    # #618: a card becomes its own nodes and a muted node goes, as in a run.
    nodes, edges = _flattened(nodes, edges, request.presets)
    if not nodes:
        # Every node that was left was muted.
        raise _coded(400, "preset_empty")

    # I3: never persist a SECRET param value into a preset definition file.
    # Blank secrets in the graph before any of it is copied into the stored
    # preset. After `_flattened` on purpose: the params of a card's nodes,
    # with the settings made on the card, only become node params there.
    # C1 (below) additionally keeps SECRET params out of the exposed-params
    # schema; this guards the raw VALUES.
    scrub_graph_secrets(nodes)

    # #618: the nodes are stored, and their params listed, in the order data
    # flows; the exposed ports keep canvas order (see `_in_flow_order`).
    flow = _in_flow_order(nodes, edges)

    # Create short ID mapping for cleaner JSON, numbered in flow order
    id_map: dict[str, str] = {}
    for i, node in enumerate(flow):
        old_id = node.get("id", f"node_{i}")
        id_map[old_id] = f"node_{i}"

    # Transform nodes. Each entry is also kept by its canvas node, so the
    # ports below can be walked in canvas order.
    internal_nodes = []
    internal_of: dict[int, dict[str, Any]] = {}
    for node in flow:
        old_id = node.get("id", "")
        node_type: str = node.get("type", "")
        params = node.get("data", {}).get("params", {})
        entry = {
            "id": id_map.get(old_id, old_id),
            "type": node_type,
            "params": params,
        }
        internal_nodes.append(entry)
        internal_of[id(node)] = entry

    # Transform edges
    internal_edges = []
    for edge in edges:
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

    # #618: a type that occurs more than once is numbered in flow order, as
    # the built-in presets number theirs ("Activation 1"). Two "Activation -
    # function" fields under one heading left the preset's dialog unable to
    # say which was which.
    repeated = {
        node_type
        for node_type, count in Counter(n["type"] for n in internal_nodes).items()
        if count > 1
    }
    seen: Counter[str] = Counter()
    label_of: dict[int, str] = {}
    for node in internal_nodes:
        seen[node["type"]] += 1
        label_of[id(node)] = (
            f"{node['type']} {seen[node['type']]}"
            if node["type"] in repeated else node["type"]
        )

    # Ports in canvas order: Map runs a preset through its first exposed
    # input and output, so which port comes first must not depend on the
    # order the nodes are stored in.
    for node in (internal_of[id(canvas_node)] for canvas_node in nodes):
        node_cls = node_registry.get(node["type"])
        if not node_cls:
            continue
        label = label_of[id(node)]

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
                    "description": f"{label}: {port.description}",
                })

        # Unconnected output ports → exposed outputs (see above).
        for port in node_cls.define_outputs_dynamic(node["params"]):
            if (node["id"], port.name) not in connected_outputs:
                exposed_outputs.append({
                    "name": f"{node['id']}_{port.name}",
                    "internal_node": node["id"],
                    "internal_port": port.name,
                    "data_type": port.data_type.value,
                    "description": f"{label}: {port.description}",
                })

    # All params → exposed params, in flow order, grouped by node as
    # labelled above.
    for node in internal_nodes:
        node_cls = node_registry.get(node["type"])
        if not node_cls:
            continue
        label = label_of[id(node)]
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
                "display_name": f"{label} - {param.name}",
                "group": label,
            })

    if not exposed_inputs and not exposed_outputs:
        raise _coded(400, "preset_no_ports")

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
    settings.USER_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(preset_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # Load the new file alone and add it to what the registry holds (#600).
    # A clear here dropped every preset a plugin pack supplied until the next
    # reload, and rereading the whole dir gave a name that a pack won at
    # startup back to the user's file of that name.
    try:
        return preset_registry.load_file(filepath, node_registry)
    except Exception as exc:
        # Left on disk, the file would refuse every later export under this
        # name with 409 instead of giving this answer again.
        logger.warning("Failed to load %s: %s", filepath.name, exc)
        filepath.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500, detail="Failed to load created preset",
        ) from None
