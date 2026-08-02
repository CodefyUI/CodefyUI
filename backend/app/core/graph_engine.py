from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import logging
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import settings
from .backward_pass import (
    attach_retain_grad,
    capture_grads,
    grad_health,
    run_backward,
    select_backward_target,
    zero_module_grads,
)
from .execution_context import (
    INTERRUPTED_KEY,
    ArtifactSignal,
    CancellationError,
    DroppedSignal,
    EventOutbox,
    ExecutionContext,
    MetricSignal,
    ProgressSignal,
)
from .node_base import BaseNode
from .node_registry import registry
from .step_trace import Step
from .type_system import is_compatible

logger = logging.getLogger(__name__)


class GraphValidationError(Exception):
    pass


def build_preset_fallback(presets: Any) -> dict:
    """Map preset_name -> PresetDefinition for graph-embedded presets (ID6).

    Accepts PresetDefinition objects or plain dicts (json.loads output);
    malformed entries are skipped so a stray preset never breaks a run.
    """
    from ..schemas.models import PresetDefinition

    out: dict[str, Any] = {}
    for p in presets or []:
        try:
            model = p if isinstance(p, PresetDefinition) else PresetDefinition(**p)
        except Exception:
            continue
        out[model.preset_name] = model
    return out


def expand_presets(
    nodes: list[dict],
    edges: list[dict],
    preset_fallback: dict | None = None,
) -> tuple[list[dict], list[dict], dict[str, str]]:
    """Expand preset nodes into their sub-graph of real nodes.

    Returns (expanded_nodes, expanded_edges, internal_to_preset_map).
    internal_to_preset_map maps internal node IDs to the preset node ID they came from.

    ``preset_fallback`` (ID6) is consulted when the server's preset
    registry does not know the preset name -- lets a graph carrying its
    own ``presets[]`` expand on a machine whose registry lacks it.
    """
    from .preset_registry import preset_registry

    expanded_nodes: list[dict] = []
    expanded_edges: list[dict] = list(edges)
    internal_to_preset: dict[str, str] = {}

    for node in nodes:
        node_type: str = node.get("type", "")
        if not node_type.startswith("preset:"):
            expanded_nodes.append(node)
            continue

        preset_name = node_type[len("preset:"):]
        preset = preset_registry.get(preset_name) or (preset_fallback or {}).get(preset_name)
        if not preset:
            raise GraphValidationError(f"Unknown preset: {preset_name}")

        preset_node_id = node["id"]
        internal_params = node.get("data", {}).get("internalParams", {})

        # Build a map of exposed port name -> internal node:port
        input_map: dict[str, tuple[str, str]] = {}
        for ep in preset.exposed_inputs:
            full_id = f"{preset_node_id}__{ep.internal_node}"
            input_map[ep.name] = (full_id, ep.internal_port)

        output_map: dict[str, tuple[str, str]] = {}
        for ep in preset.exposed_outputs:
            full_id = f"{preset_node_id}__{ep.internal_node}"
            output_map[ep.name] = (full_id, ep.internal_port)

        # Add internal nodes with unique IDs
        for internal_node in preset.nodes:
            full_id = f"{preset_node_id}__{internal_node.id}"
            # Merge default params with user overrides
            params = dict(internal_node.params)
            if internal_node.id in internal_params:
                params.update(internal_params[internal_node.id])
            expanded_nodes.append({
                "id": full_id,
                "type": internal_node.type,
                "position": node.get("position", {"x": 0, "y": 0}),
                "data": {"params": params},
            })
            internal_to_preset[full_id] = preset_node_id

        # Add internal edges with remapped IDs
        for internal_edge in preset.edges:
            expanded_edges.append({
                "source": f"{preset_node_id}__{internal_edge.source}",
                "target": f"{preset_node_id}__{internal_edge.target}",
                "sourceHandle": internal_edge.sourceHandle,
                "targetHandle": internal_edge.targetHandle,
            })

        # Remap external edges connected to this preset node
        new_edges = []
        for edge in expanded_edges:
            new_edge = dict(edge)
            # Remap edges where this preset is the target
            if edge.get("target") == preset_node_id:
                target_handle = edge.get("targetHandle", "")
                if target_handle in input_map:
                    internal_id, internal_port = input_map[target_handle]
                    new_edge["target"] = internal_id
                    new_edge["targetHandle"] = internal_port
            # Remap edges where this preset is the source
            if edge.get("source") == preset_node_id:
                source_handle = edge.get("sourceHandle", "")
                if source_handle in output_map:
                    internal_id, internal_port = output_map[source_handle]
                    new_edge["source"] = internal_id
                    new_edge["sourceHandle"] = internal_port
            new_edges.append(new_edge)
        expanded_edges = new_edges

    return expanded_nodes, expanded_edges, internal_to_preset


# ── Bypass / mute (core#128) ─────────────────────────────────────────────
#
# A node the user has bypassed on the canvas carries ``data.bypassed = True``.
# It is not executed; instead every one of its output ports forwards the value
# that arrived on the first type-compatible input port, so downstream nodes see
# the graph as if the bypassed node were not there. This mirrors ComfyUI's
# Ctrl+B, and — like ComfyUI — the match is made on the node's own DECLARED
# port types, positionally, first match wins.
#
# Resolution happens once, structurally: `resolve_bypass` removes the node and
# rewires its outgoing edges to whatever fed the matched input. Everything
# downstream of that (reachability, validation, topological order, the Python
# exporter) therefore needs no bypass awareness at all — it simply never sees
# the node.

BYPASS_KEY = "bypassed"


def _is_bypassed(node: dict) -> bool:
    data = node.get("data")
    return bool(data.get(BYPASS_KEY)) if isinstance(data, dict) else False


def _type_name(data_type: Any) -> str:
    """Readable port type for an error message, enum or bare string."""
    return str(getattr(data_type, "value", data_type))


@dataclass(frozen=True)
class BypassLink:
    """One resolved pass-through: ``node_id.output`` came from ``source``.

    ``source`` is the nearest NON-bypassed producer, so a chain of bypassed
    nodes collapses to the value's real origin. Collected for the Python
    exporter, which comments each bypass with the assignment it stands in for.
    """

    node_id: str
    node_type: str
    output: str
    input: str
    source: str
    source_handle: str


@dataclass
class BypassResolution:
    """Result of :func:`resolve_bypass` — a graph with no bypassed nodes left.

    ``nodes``/``edges`` are the SAME objects that went in when the graph has
    no bypassed node, so the overwhelmingly common case costs one scan and no
    allocation.
    """

    nodes: list[dict]
    edges: list[dict]
    errors: list[str] = field(default_factory=list)
    links: list[BypassLink] = field(default_factory=list)
    #: ``(target_id, target_handle) -> bypassed node id`` for every edge the
    #: resolution DROPPED because the bypass had nothing to forward. Lets the
    #: resulting "Missing required input" name the mute that caused it, rather
    #: than pointing at a port the user never touched.
    dropped: dict[tuple[str, str], str] = field(default_factory=dict)


def resolve_bypass(nodes: list[dict], edges: list[dict]) -> BypassResolution:
    """Remove bypassed nodes, forwarding each output from a matching input.

    Port matching rule (ComfyUI's, spelled out): for output port ``o``, the
    forwarded input is the FIRST declared input ``i`` with
    ``is_compatible(i.data_type, o.data_type)``.

    ``is_compatible`` is deliberately WIDER than the issue's literal "same
    DataType": it is the very predicate the edge validator uses, so an input
    the bypass forwards can only produce an edge validation would already have
    accepted (ANY absorbs everything; IMAGE flows into TENSOR). Requiring
    strict equality would refuse pass-throughs the graph could legally have
    been wired with in the first place. Then:

    * matched input is wired  -> every edge leaving ``o`` is re-pointed at
      whatever fed ``i`` (recursively, so chains of bypassed nodes collapse);
    * matched input is empty  -> the edges leaving ``o`` are dropped, and the
      downstream node reports the missing input like any unconnected port;
    * no compatible input     -> an error naming the incompatibility, because
      silently dropping the edge would hide a wiring mistake.

    Incoming TRIGGER edges are re-pointed at the first non-bypassed node the
    bypassed one feeds, so bypassing an entry point does not silently leave
    the graph with none.

    Never raises. Callers decide what an error means: ``validate_graph``
    reports it, ``prepare_executable_graph`` refuses to run.
    """
    bypassed = {n["id"]: n for n in nodes if _is_bypassed(n)}
    if not bypassed:
        return BypassResolution(nodes, edges)

    errors: list[str] = []
    links: list[BypassLink] = []

    # Imported lazily: api_contract imports find_entry_points from THIS module,
    # so a top-level import would close a cycle. One source of truth beats
    # re-spelling the two type names here.
    from .api_contract import GRAPH_INPUT_TYPE, GRAPH_OUTPUT_TYPE

    # Three kinds of bypassed node are left in place rather than resolved:
    #
    #  - a preset instance, whose ports come from the preset definition and
    #    whose body is expanded elsewhere. Refused loudly (the canvas does not
    #    offer bypass on presets; a hand-edited file might).
    #  - a GraphInput / GraphOutput, which declares the graph's I/O CONTRACT.
    #    `derive_contract` and `check_wiring` scan the raw graph, so silently
    #    deleting one here would leave a published app advertising an output
    #    the run cannot produce. GraphOutput is the sharp case: it declares no
    #    output ports at all, so nothing would ever call `_forward` on it and
    #    no pass-through error would fire. Refused for the same reason presets
    #    are — bypass has no meaning for a node that IS the signature.
    #  - an unregistered node type, whose ports cannot be read at all. Left
    #    alone so the caller reports "Unknown node type", which is the real
    #    problem, instead of a confusing pass-through complaint.
    contract_types = (GRAPH_INPUT_TYPE, GRAPH_OUTPUT_TYPE)
    active: dict[str, dict] = {}
    for node_id, node in bypassed.items():
        node_type = str(node.get("type", ""))
        if node_type.startswith("preset:"):
            errors.append(f"Bypass is not supported on preset node {node_id}")
        elif node_type in contract_types:
            errors.append(
                f"Bypass is not supported on {node_type} node {node_id}: it "
                "declares the graph's I/O contract"
            )
        elif registry.get(node_type) is not None:
            active[node_id] = node
        # else: unregistered type — left in place on purpose (see above).
    if not active:
        return BypassResolution(nodes, edges, errors)

    # Last edge into a handle wins, matching how the engine builds a node's
    # inputs dict (later writes to `inputs[tgt_handle]` overwrite earlier ones).
    incoming: dict[tuple[str, str], tuple[str, str]] = {}
    data_out: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge.get("type", "data") != "data":
            continue
        incoming[(edge["target"], edge.get("targetHandle", ""))] = (
            edge["source"],
            edge.get("sourceHandle", ""),
        )
        data_out[edge["source"]].append(edge["target"])

    ports: dict[str, tuple[list, list]] = {}

    def _ports(node_id: str) -> tuple[list, list]:
        if node_id not in ports:
            node = active[node_id]
            node_cls = registry.get(node["type"])
            params = node.get("data", {}).get("params", {})
            ports[node_id] = (
                node_cls.define_inputs_dynamic(params),
                node_cls.define_outputs_dynamic(params),
            )
        return ports[node_id]

    resolved: dict[tuple[str, str], tuple[str, str] | None] = {}

    def _forward(node_id: str, out_port: str) -> tuple[str, str] | None:
        """The (source, handle) a bypassed output forwards, or None.

        The memo is seeded with ``None`` BEFORE recursing, which doubles as
        the cycle guard: bypassed nodes wired in a ring re-enter, short-circuit
        to ``None``, and the recursion terminates. Note that validate_graph
        will NOT go on to report that cycle — the nodes forming it no longer
        exist by then. Every edge leaving the ring is simply dropped, so what
        the user sees is a missing input on whatever consumed it, attributed
        to the mute via ``dropped`` below. Seeding also keeps the
        no-compatible-input error to one line per port.
        """
        key = (node_id, out_port)
        if key in resolved:
            return resolved[key]
        resolved[key] = None  # provisional

        inputs, outputs = _ports(node_id)
        output = next((p for p in outputs if p.name == out_port), None)
        if output is None:
            # An edge naming a source handle this node does not declare. Edge
            # validation will never see it either (the node and the edge are
            # both about to go), so it lands as a missing input downstream.
            return None

        match = next(
            (p for p in inputs if is_compatible(p.data_type, output.data_type)),
            None,
        )
        if match is None:
            declared = ", ".join(
                f"{p.name} ({_type_name(p.data_type)})" for p in inputs
            )
            errors.append(
                f"Bypassed node {node_id} ({active[node_id].get('type', '')}): "
                f"output '{out_port}' ({_type_name(output.data_type)}) has no "
                f"type-compatible input to forward "
                f"(inputs: {declared or 'none'})"
            )
            return None

        upstream = incoming.get((node_id, match.name))
        if upstream is None:
            return None  # nothing wired in; downstream input stays unconnected

        source, handle = upstream
        if source in active:
            upstream = _forward(source, handle)
            if upstream is None:
                return None
            source, handle = upstream

        resolved[key] = (source, handle)
        links.append(
            BypassLink(
                node_id=node_id,
                node_type=str(active[node_id].get("type", "")),
                output=out_port,
                input=match.name,
                source=source,
                source_handle=handle,
            )
        )
        return resolved[key]

    def _trigger_targets(node_id: str) -> list[str]:
        """First non-bypassed nodes downstream, in breadth-first edge order."""
        out: list[str] = []
        seen = {node_id}
        queue = deque([node_id])
        while queue:
            current = queue.popleft()
            for nxt in data_out.get(current, []):
                if nxt in seen:
                    continue
                seen.add(nxt)
                if nxt in active:
                    queue.append(nxt)
                else:
                    out.append(nxt)
        return out

    new_edges: list[dict] = []
    dropped: dict[tuple[str, str], str] = {}
    for edge in edges:
        edge_type = edge.get("type", "data")
        source, target = edge["source"], edge["target"]

        if target in active:
            # Consumed by the pass-through map — except a trigger, which is a
            # marker rather than a value and moves on to what the node fed.
            if edge_type == "trigger":
                for index, downstream in enumerate(_trigger_targets(target)):
                    moved = dict(edge)
                    moved["target"] = downstream
                    if "id" in moved:
                        moved["id"] = f"{moved['id']}:bypass:{index}"
                    new_edges.append(moved)
            continue

        if source in active:
            if edge_type == "trigger":
                continue  # a bypassed node emits nothing, triggers included
            forwarded = _forward(source, edge.get("sourceHandle", ""))
            if forwarded is None:
                # Nothing to carry across. Remember who caused it so the
                # missing input this creates can name the mute.
                dropped[(target, edge.get("targetHandle", ""))] = source
                continue
            rewired = dict(edge)
            rewired["source"], rewired["sourceHandle"] = forwarded
            new_edges.append(rewired)
            continue

        new_edges.append(edge)

    new_nodes = [n for n in nodes if n["id"] not in active]
    return BypassResolution(new_nodes, new_edges, errors, links, dropped)


def validate_graph(
    nodes: list[dict],
    edges: list[dict],
    preset_fallback: dict | None = None,
) -> list[str]:
    """Validate a graph definition. Returns list of errors (empty = valid).

    ``preset_fallback`` (ID6) lets a graph-embedded preset (one the
    server's registry does not know) validate as present -- see
    ``build_preset_fallback``.

    Bypassed nodes (core#128) are resolved away first, so every check below
    runs against the graph that would actually execute: a node whose only
    upstream is bypassed is checked against what the bypass forwards, not
    against the node the user muted.
    """
    resolution = resolve_bypass(nodes, edges)
    errors: list[str] = list(resolution.errors)
    nodes, edges = resolution.nodes, resolution.edges
    node_map = {n["id"]: n for n in nodes}

    # --- Node-level validation (standalone, before edge checks) ---
    from .preset_registry import preset_registry

    # 1. Node type existence check
    valid_node_ids: set[str] = set()
    preset_node_ids: set[str] = set()
    for node in nodes:
        node_type: str = node.get("type", "")
        # Preset nodes are expanded at execution time; validate they exist in preset registry
        if node_type.startswith("preset:"):
            preset_name = node_type[len("preset:"):]
            if not (preset_registry.get(preset_name) or (preset_fallback or {}).get(preset_name)):
                errors.append(f"Unknown preset: {preset_name} (node {node['id']})")
            else:
                preset_node_ids.add(node["id"])
                valid_node_ids.add(node["id"])
            continue
        node_cls = registry.get(node_type)
        if node_cls is None:
            errors.append(f"Unknown node type: {node_type} (node {node['id']})")
        else:
            valid_node_ids.add(node["id"])

    # 2. Required input connection check (skip preset nodes — they define ports dynamically)
    connected_inputs = {
        (edge["target"], edge.get("targetHandle", ""))
        for edge in edges
    }
    for node in nodes:
        if node["id"] not in valid_node_ids or node["id"] in preset_node_ids:
            continue
        node_cls = registry.get(node["type"])
        node_params = node.get("data", {}).get("params", {}) if isinstance(node.get("data"), dict) else {}
        for inp in node_cls.define_inputs_dynamic(node_params):
            if not inp.optional and (node["id"], inp.name) not in connected_inputs:
                # A port left unconnected BY A BYPASS reads as the user's own
                # wiring mistake unless the message says otherwise (core#128).
                cause = resolution.dropped.get((node["id"], inp.name))
                errors.append(
                    f"Missing required input '{inp.name}' on node {node['id']} ({node['type']})"
                    + (
                        f" (input dropped because '{cause}' is bypassed)"
                        if cause
                        else ""
                    )
                )

    # 3. Parameter range validation (skip preset nodes)
    for node in nodes:
        if node["id"] not in valid_node_ids or node["id"] in preset_node_ids:
            continue
        node_cls = registry.get(node["type"])
        param_values = node.get("data", {}).get("params", {})
        for param_def in node_cls.define_params():
            if param_def.name not in param_values:
                continue
            value = param_values[param_def.name]
            if param_def.min_value is not None and value < param_def.min_value:
                errors.append(
                    f"Parameter '{param_def.name}' on node {node['id']} ({node['type']}): "
                    f"value {value} is below minimum {param_def.min_value}"
                )
            if param_def.max_value is not None and value > param_def.max_value:
                errors.append(
                    f"Parameter '{param_def.name}' on node {node['id']} ({node['type']}): "
                    f"value {value} is above maximum {param_def.max_value}"
                )

    # --- Edge-level validation ---

    for edge in edges:
        # Trigger edges are control-flow markers, not data connections.
        if edge.get("type", "data") == "trigger":
            continue

        src = node_map.get(edge["source"])
        tgt = node_map.get(edge["target"])
        if not src or not tgt:
            errors.append(f"Edge references missing node: {edge}")
            continue

        # Skip edge validation when either end is a preset node (ports are dynamic)
        if src["id"] in preset_node_ids or tgt["id"] in preset_node_ids:
            continue

        src_cls = registry.get(src["type"])
        tgt_cls = registry.get(tgt["type"])
        if not src_cls or not tgt_cls:
            errors.append(f"Unknown node type: {src['type']} or {tgt['type']}")
            continue

        src_port = edge.get("sourceHandle", "")
        tgt_port = edge.get("targetHandle", "")
        # define_{outputs,inputs}_dynamic let param-driven nodes (SplitNode's
        # `chunks`, PythonScriptNode's `input_ports`/`output_ports`) expose
        # their live port set. Nodes that don't override fall back to the
        # static definitions via BaseNode's defaults.
        src_params = src.get("data", {}).get("params", {}) if isinstance(src.get("data"), dict) else {}
        tgt_params = tgt.get("data", {}).get("params", {}) if isinstance(tgt.get("data"), dict) else {}
        src_outputs = {p.name: p for p in src_cls.define_outputs_dynamic(src_params)}
        tgt_inputs = {p.name: p for p in tgt_cls.define_inputs_dynamic(tgt_params)}

        if src_port not in src_outputs:
            errors.append(f"Invalid output port '{src_port}' on {src['type']}")
            continue
        if tgt_port not in tgt_inputs:
            errors.append(f"Invalid input port '{tgt_port}' on {tgt['type']}")
            continue

        if not is_compatible(src_outputs[src_port].data_type, tgt_inputs[tgt_port].data_type):
            errors.append(
                f"Type mismatch: {src['type']}.{src_port} ({src_outputs[src_port].data_type}) "
                f"-> {tgt['type']}.{tgt_port} ({tgt_inputs[tgt_port].data_type})"
            )

    # NEW: Entry-point rules
    entry_ids = find_entry_points(nodes, edges)
    if not entry_ids:
        errors.append(
            "Graph has no entry points. Add a Start node and connect "
            "it to the node you want to start execution from."
        )
        # Still run remaining checks so user sees all problems at once
        executable_node_ids = {n["id"] for n in nodes}
    else:
        executable_node_ids = reachable_from_entry_points(entry_ids, edges)

    # MODIFIED: Run cycle detection on the EXECUTABLE subgraph only.
    # Drafts (nodes outside executable_node_ids) are skipped.
    executable_nodes = [n for n in nodes if n["id"] in executable_node_ids]
    executable_edges = [
        e for e in edges
        if e["source"] in executable_node_ids
        and e["target"] in executable_node_ids
        and e.get("type", "data") == "data"
    ]
    if _has_cycle(executable_nodes, executable_edges):
        errors.append("Graph contains a cycle")

    return errors


def _has_cycle(nodes: list[dict], edges: list[dict]) -> bool:
    in_degree: dict[str, int] = {n["id"]: 0 for n in nodes}
    adj: dict[str, list[str]] = defaultdict(list)

    for edge in edges:
        if edge.get("type", "data") == "trigger":
            continue  # markers, not dependencies
        if edge["source"] not in in_degree or edge["target"] not in in_degree:
            # Edge validation already reports missing endpoints. Keep cycle
            # detection total so a malformed edge produces a clean 4xx error
            # instead of a secondary KeyError.
            continue
        adj[edge["source"]].append(edge["target"])
        in_degree[edge["target"]] += 1

    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return visited != len(nodes)


def find_entry_points(
    nodes: list[dict],
    edges: list[dict],
) -> list[str]:
    """Return ids of nodes that are entry points.

    A node is an entry point if it has at least one incoming trigger edge
    (i.e. it is connected from a Start node). Start nodes themselves are
    NOT entry points — they are markers that designate entry points via
    their trigger edges.

    The order of returned ids matches the order in `nodes` for determinism.
    """
    nodes_with_trigger_in: set[str] = {
        e["target"]
        for e in edges
        if e.get("type", "data") == "trigger"
    }
    return [n["id"] for n in nodes if n["id"] in nodes_with_trigger_in]


def reachable_from_entry_points(
    entry_ids: list[str],
    edges: list[dict],
) -> set[str]:
    """BFS forward from entry_ids through DATA edges only.

    Trigger edges are markers, not data dependencies, and are not
    traversed. The seed entry_ids themselves are always included in the
    result, regardless of edge types.
    """
    reachable: set[str] = set(entry_ids)
    frontier: list[str] = list(entry_ids)
    # Build adjacency list of data edges only.
    adj: dict[str, list[str]] = {}
    for e in edges:
        if e.get("type", "data") == "data":
            adj.setdefault(e["source"], []).append(e["target"])
    while frontier:
        node = frontier.pop()
        for next_node in adj.get(node, []):
            if next_node not in reachable:
                reachable.add(next_node)
                frontier.append(next_node)
    return reachable


def topological_sort(nodes: list[dict], edges: list[dict]) -> list[str]:
    """Kahn's algorithm. Returns ordered node IDs.

    Trigger edges (type="trigger") are excluded from in-degree calculation
    because they are execution markers, not data dependencies.
    """
    in_degree: dict[str, int] = {n["id"]: 0 for n in nodes}
    adj: dict[str, list[str]] = defaultdict(list)

    for edge in edges:
        if edge.get("type", "data") == "trigger":
            continue  # markers, not dependencies
        adj[edge["source"]].append(edge["target"])
        if edge["target"] in in_degree:
            in_degree[edge["target"]] += 1

    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    order: list[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(nodes):
        raise GraphValidationError("Graph contains a cycle")

    return order


def topological_levels(nodes: list[dict], edges: list[dict]) -> list[list[str]]:
    """Kahn's algorithm returning nodes grouped by DAG level for parallel execution.

    Trigger edges (type="trigger") are excluded from in-degree calculation
    because they are execution markers, not data dependencies. A node that
    only receives a trigger edge is still considered a root (level 0).
    """
    in_degree: dict[str, int] = {n["id"]: 0 for n in nodes}
    adj: dict[str, list[str]] = defaultdict(list)

    for edge in edges:
        if edge.get("type", "data") == "trigger":
            continue  # markers, not dependencies
        adj[edge["source"]].append(edge["target"])
        if edge["target"] in in_degree:
            in_degree[edge["target"]] += 1

    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    levels: list[list[str]] = []

    while queue:
        level = list(queue)
        levels.append(level)
        next_queue: deque[str] = deque()
        for node in level:
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    next_queue.append(neighbor)
        queue = next_queue

    total = sum(len(lv) for lv in levels)
    if total != len(nodes):
        raise GraphValidationError("Graph contains a cycle")

    return levels


def invoke_node(
    instance: BaseNode,
    inputs: dict[str, Any],
    params: dict[str, Any],
    *,
    progress_callback: Callable[[dict], None] | None = None,
    context: "ExecutionContext | None" = None,
) -> dict[str, Any]:
    """Call ``instance.execute`` with exactly the keywords it declares.

    Node authors opt in to ``progress_callback`` / ``context`` by naming them
    in their ``execute`` signature; undeclared ones are dropped here. Both the
    graph engine and exported Python runners route every node call through
    this helper, so invocation semantics can never drift between the canvas
    and an exported script.
    """
    sig = inspect.signature(instance.execute)
    call_kwargs: dict[str, Any] = {}
    if "progress_callback" in sig.parameters:
        call_kwargs["progress_callback"] = progress_callback
    if "context" in sig.parameters:
        call_kwargs["context"] = context
    return instance.execute(inputs, params, **call_kwargs)


def prepare_executable_graph(
    nodes: list[dict],
    edges: list[dict],
    *,
    preset_fallback: dict | None = None,
) -> tuple[list[dict], list[dict], dict[str, str]]:
    """Expand presets, resolve bypass, prune drafts, and validate.

    This is the structural preflight used immediately before execution. It is
    also safe for callers such as Python export that need the exact same preset
    grouping and draft-pruning semantics without actually running any nodes.
    """

    # Presets expand into a sub-graph whose ports come from the preset
    # definition rather than a node class, so the pass-through rule has nothing
    # to match on. Refuse before expansion, where the preset node still exists
    # to be named.
    bypassed_presets = [
        node["id"]
        for node in nodes
        if _is_bypassed(node) and str(node.get("type", "")).startswith("preset:")
    ]
    if bypassed_presets:
        raise GraphValidationError(
            "Bypass is not supported on preset node(s): "
            + ", ".join(sorted(bypassed_presets))
        )

    internal_to_preset: dict[str, str] = {}
    expanded_nodes, expanded_edges = nodes, edges
    for _ in range(10):
        if not any(
            node.get("type", "").startswith("preset:")
            for node in expanded_nodes
        ):
            break
        expanded_nodes, expanded_edges, mapping = expand_presets(
            expanded_nodes,
            expanded_edges,
            preset_fallback=preset_fallback,
        )
        internal_to_preset.update(mapping)

    if any(
        node.get("type", "").startswith("preset:")
        for node in expanded_nodes
    ):
        raise GraphValidationError("Preset nesting exceeds the maximum depth of 10")

    # Bypass BEFORE reachability: a bypassed node is not part of the graph, so
    # what is reachable, what the topological order is, and what the exporter
    # emits are all decided on the graph the user actually asked to run.
    bypass = resolve_bypass(expanded_nodes, expanded_edges)
    if bypass.errors:
        raise GraphValidationError("; ".join(bypass.errors))
    expanded_nodes, expanded_edges = bypass.nodes, bypass.edges

    entry_ids = find_entry_points(expanded_nodes, expanded_edges)
    if not entry_ids:
        raise GraphValidationError("Graph has no entry points")

    executable_ids = reachable_from_entry_points(entry_ids, expanded_edges)

    # Preserve Start markers whose trigger targets are executable so the
    # validation pass sees the same entry points.
    for edge in expanded_edges:
        if (
            edge.get("type", "data") == "trigger"
            and edge["target"] in executable_ids
        ):
            executable_ids.add(edge["source"])

    # A preset is one logical unit. If any internal node is reachable, retain
    # all sibling roots (for example Dataset and Loss in a training preset).
    presets_to_include = {
        preset_id
        for internal_id, preset_id in internal_to_preset.items()
        if internal_id in executable_ids
    }
    executable_ids.update(
        internal_id
        for internal_id, preset_id in internal_to_preset.items()
        if preset_id in presets_to_include
    )

    executable_nodes = [
        node for node in expanded_nodes if node["id"] in executable_ids
    ]
    executable_edges = [
        edge
        for edge in expanded_edges
        if edge["source"] in executable_ids and edge["target"] in executable_ids
    ]

    errors = validate_graph(
        executable_nodes,
        executable_edges,
        preset_fallback=preset_fallback,
    )
    if errors:
        raise GraphValidationError("; ".join(errors))

    return executable_nodes, executable_edges, internal_to_preset


async def execute_graph(
    nodes: list[dict],
    edges: list[dict],
    on_progress: Callable[[str, str, dict[str, Any] | None], Any] | None = None,
    context: "ExecutionContext | None" = None,
    error_mode: str = "fail_fast",
    max_retries: int = 0,
    cache: "ExecutionCache | None" = None,
    changed_nodes: list[str] | None = None,
    run_id: str | None = None,
    output_store: "RunOutputStore | None" = None,
    record_outputs: bool = False,
    preset_fallback: dict | None = None,
    on_signal: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Execute the graph with parallel levels, cancellation, error recovery, and caching.

    Args:
        nodes: Graph node definitions.
        edges: Graph edge definitions.
        on_progress: Callback(node_id, status, data).
        context: ExecutionContext for cancellation support.
        error_mode: 'fail_fast', 'continue', or 'retry'.
        max_retries: Number of retries when error_mode is 'retry'.
        cache: Optional ExecutionCache for skipping unchanged nodes.
        changed_nodes: Optional list of node IDs that changed — force re-execute these (bypass cache).
        run_id: Run identifier used as the key for ``output_store``. Required
            when ``record_outputs`` is True.
        output_store: Optional per-run in-memory store. When ``record_outputs``
            is True, each node's full output is written under ``run_id``.
        record_outputs: When True, capture every node's output into
            ``output_store`` for later retrieval via the REST endpoint.
        preset_fallback: Graph-embedded preset definitions (ID6), consulted
            when the server's preset registry lacks a referenced preset.
        on_signal: Callback for everything a node reports that is NOT a node
            status — ``MetricSignal`` (``context.log_metric``),
            ``ArtifactSignal`` (``context.log_artifact``) and
            ``DroppedSignal`` (the outbox shed load). Called on the loop, one
            at a time, in the order the nodes produced them. Omitting it
            simply discards those signals; only ``RunService`` has somewhere
            durable to put them.

    Progress and metric delivery (#122)
    -----------------------------------
    Nodes report from an executor thread, and until #122 the bridge did
    ``run_coroutine_threadsafe(...).result(timeout=10)`` -- the training
    thread blocked until the loop had persisted the event. Now the bridge
    APPENDS to ``context.outbox`` (bounded, drop-oldest, lock-free from the
    node's point of view) and a single pump task drains it on the loop, so a
    per-batch producer never pays for the consumer.

    One pump, not one per node, because ordering is the whole contract of an
    event log: a single consumer delivers signals in the order they were
    produced, across every node running in parallel. The other half of that
    ordering guarantee is the inline drain right after a node's future
    resolves -- without it a node's queued progress would land AFTER its own
    ``completed``.
    """
    expanded_nodes, expanded_edges, internal_to_preset = prepare_executable_graph(
        nodes,
        edges,
        preset_fallback=preset_fallback,
    )

    levels = topological_levels(expanded_nodes, expanded_edges)
    node_map = {n["id"]: n for n in expanded_nodes}

    # Build edge lookup: target_id -> list of (source_id, source_handle, target_handle)
    incoming: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for edge in expanded_edges:
        incoming[edge["target"]].append(
            (edge["source"], edge.get("sourceHandle", ""), edge.get("targetHandle", ""))
        )

    outputs: dict[str, dict[str, Any]] = {}
    node_errors: dict[str, str] = {}  # node_id -> error message
    node_cache_keys: dict[str, str] = {}  # node_id -> cache key
    force_rerun: set[str] = set(changed_nodes) if changed_nodes else set()

    # When verbose step-trace or backward gradient capture is on, the cache
    # is incompatible (verbose adds __steps__; backward needs grad-tracked
    # tensors). Force re-run everything so the run actually produces the
    # requested data instead of returning a cached stale result.
    if context is not None and (context.verbose or context.backward_mode):
        for n in expanded_nodes:
            force_rerun.add(n["id"])

    # Preset aggregation: emit "running" once at start, "completed" only when all internal nodes finish.
    # preset_total[preset_id] = number of internal nodes belonging to that preset
    # preset_done[preset_id] = number of internal nodes that have completed/cached/skipped
    # preset_started[preset_id] = True once we've emitted "running" for the preset
    preset_total: dict[str, int] = defaultdict(int)
    for _internal_id, _preset_id in internal_to_preset.items():
        preset_total[_preset_id] += 1
    preset_done: dict[str, int] = defaultdict(int)
    preset_started: set[str] = set()

    async def _emit_preset_aware(
        node_id: str,
        status: str,
        data: dict[str, Any] | None,
    ) -> None:
        """Emit status to on_progress, aggregating internal preset nodes.

        Internal preset nodes (in internal_to_preset) roll up into a single preset status:
        - First running/cached → emit preset 'running'
        - Every completed/cached/skipped increments done count; emit 'completed' only on last
        - 'error' emits immediately (preset failed)
        - 'interrupted' likewise: a preset whose training node stopped early
          did not complete, so it must never roll up to 'completed'
        - 'progress' passes through as-is with the preset ID (so live charts still work)
        Non-preset nodes pass through unchanged.
        """
        if on_progress is None:
            return
        preset_id = internal_to_preset.get(node_id)
        if preset_id is None:
            # Regular node — pass through
            await _maybe_await(on_progress(node_id, status, data))
            return

        # Internal preset node — aggregate
        if status == "progress":
            # Progress events (e.g. training epochs) should be visible live
            await _maybe_await(on_progress(preset_id, "progress", data))
            return

        if status in ("error", "interrupted"):
            # Any internal failure or early stop settles the whole preset
            await _maybe_await(on_progress(preset_id, status, data))
            return

        if status == "running":
            if preset_id not in preset_started:
                preset_started.add(preset_id)
                await _maybe_await(on_progress(preset_id, "running", None))
            return

        if status in ("completed", "cached", "skipped"):
            preset_done[preset_id] += 1
            # Make sure "running" was emitted at least once
            if preset_id not in preset_started:
                preset_started.add(preset_id)
                await _maybe_await(on_progress(preset_id, "running", None))
            if preset_done[preset_id] >= preset_total[preset_id]:
                await _maybe_await(on_progress(preset_id, "completed", None))

    max_workers = context.max_workers if context else 4
    semaphore = asyncio.Semaphore(max_workers)

    # ── worker-thread → loop delivery ────────────────────────────────────
    #
    # A context-less run (the device smoke script, a bare execute_graph in a
    # test) still gets an outbox so the progress bridge has one code path;
    # nodes just have no way to reach it, since log_metric hangs off the
    # context.
    outbox = context.outbox if context is not None else EventOutbox()
    outbox.bind(asyncio.get_running_loop())
    if context is not None:
        # A fact about THIS run, not a request: whether anything durable is
        # listening. Nodes read it through ``can_record_artifacts`` before
        # creating a file they would otherwise orphan.
        context.signals_recorded = on_signal is not None
    deliver_lock = asyncio.Lock()

    def _signal_node_id(node_id: str | None) -> str | None:
        """Report a preset's id, never its internals — as progress does."""
        if node_id is None:
            return None
        return internal_to_preset.get(node_id, node_id)

    async def _deliver() -> None:
        """Dispatch one drain's worth of signals. Serialised, hence ordered.

        The lock is what makes "one consumer" true even though both the pump
        task and each finishing node call this: they take turns, and a
        caller that finds the queue empty returns immediately.
        """
        async with deliver_lock:
            signals, dropped = outbox.drain()
            if dropped:
                logger.warning(
                    "run %s: dropped %d progress/metric signal(s); the "
                    "producer outran the consumer", run_id or "-", dropped,
                )
                if on_signal is not None:
                    await _maybe_await(on_signal(DroppedSignal(dropped)))
            for signal in signals:
                if isinstance(signal, ProgressSignal):
                    await _emit_preset_aware(
                        signal.node_id, "progress", signal.payload)
                elif on_signal is None:
                    continue
                elif isinstance(signal, MetricSignal):
                    await _maybe_await(on_signal(MetricSignal(
                        name=signal.name, value=signal.value,
                        step=signal.step,
                        node_id=_signal_node_id(signal.node_id))))
                elif isinstance(signal, ArtifactSignal):
                    await _maybe_await(on_signal(ArtifactSignal(
                        kind=signal.kind, path=signal.path, meta=signal.meta,
                        node_id=_signal_node_id(signal.node_id))))
                else:  # pragma: no cover - a node pack's own signal type
                    await _maybe_await(on_signal(signal))

    async def _pump() -> None:
        """Deliver signals while the loop is idle. Lives as long as the run."""
        while True:
            await outbox.wait()
            try:
                await _deliver()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - observability must not kill a run
                logger.warning("signal delivery failed", exc_info=True)

    async def _execute_single_node(node_id: str) -> None:
        """Execute one node with cancellation, caching, and error recovery."""
        if context and context.cancelled:
            raise CancellationError()

        node_def = node_map[node_id]
        node_type = node_def["type"]
        params = node_def.get("data", {}).get("params", {})

        node_cls = registry.get(node_type)
        if not node_cls:
            raise GraphValidationError(f"Unknown node type: {node_type}")

        # Gather inputs from upstream edges
        inputs: dict[str, Any] = {}
        has_failed_input = False
        for src_id, src_handle, tgt_handle in incoming.get(node_id, []):
            if src_id in node_errors:
                has_failed_input = True
                break
            if src_id in outputs and src_handle in outputs[src_id]:
                inputs[tgt_handle] = outputs[src_id][src_handle]

        # Skip if upstream failed (in continue/retry mode)
        if has_failed_input:
            node_errors[node_id] = "skipped: upstream node failed"
            await _emit_preset_aware(node_id, "skipped", None)
            return

        # Check cache (skip for force-rerun nodes from partial re-execution).
        # Stateful nodes opt out via cacheable=False because their internal
        # weights drift across runs.
        #
        # Important: if ANY upstream is non-cacheable, *this* node also cannot
        # be safely cached for this run. The cache key only encodes upstream
        # *cache keys*, not their actual output tensors — so a non-cacheable
        # upstream that produces different shapes between runs would still
        # generate the same downstream cache key, returning a stale tensor.
        # Propagate the non-cacheability instead of silently dropping the
        # upstream from the key.
        node_cacheable = getattr(node_cls, "cacheable", True)
        upstream_all_cached = all(
            src_id in node_cache_keys
            for src_id, _, _ in incoming.get(node_id, [])
        )
        if cache is not None and node_cacheable and upstream_all_cached:
            upstream_keys = [
                node_cache_keys[src_id]
                for src_id, _, _ in incoming.get(node_id, [])
            ]
            cache_key = cache.compute_key(
                node_type, params, upstream_keys,
                device=context.device if context is not None else "cpu",
            )
            node_cache_keys[node_id] = cache_key
            if node_id not in force_rerun:
                cached = cache.get(cache_key)
                if cached is not None:
                    outputs[node_id] = cached
                    # Even on cache hit, capture outputs so the Teaching
                    # Inspector can fetch them. Without this the first run
                    # with Rec OFF primes the cache and a subsequent run
                    # with Rec ON finds nothing to fetch.
                    if record_outputs and output_store is not None and run_id:
                        for port, value in cached.items():
                            if port.startswith("__"):
                                continue
                            await output_store.put(run_id, node_id, port, value)
                    await _emit_preset_aware(node_id, "cached", cached)
                    return

        await _emit_preset_aware(node_id, "running", None)

        attempts = max_retries + 1 if error_mode == "retry" else 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            if context and context.cancelled:
                raise CancellationError()
            try:
                async with semaphore:
                    instance = node_cls()
                    loop = asyncio.get_event_loop()

                    # Thread-safe progress bridge: worker thread → outbox →
                    # the loop-side pump. NON-BLOCKING since #122; a node
                    # reporting every batch must not be paced by sqlite.
                    def _progress_bridge(data: dict) -> None:
                        if on_progress:
                            outbox.put(ProgressSignal(node_id=node_id,
                                                      payload=data))

                    # Tell stateful nodes which node-id they belong to.
                    if context is not None:
                        context.current_node_id = node_id

                    fn = functools.partial(
                        invoke_node,
                        instance,
                        inputs,
                        params,
                        progress_callback=_progress_bridge,
                        context=context,
                    )
                    result = await loop.run_in_executor(None, fn)
                    # Everything this node queued goes out BEFORE its
                    # terminal status, which the queue alone does not
                    # guarantee: the pump might not have been scheduled yet.
                    #
                    # Guarded like the pre-#122 bridge's swallowed
                    # ``future.result(timeout=10)``: a failure to REPORT on a
                    # node that has already produced its result must not
                    # bubble into the retry path below and train the whole
                    # thing over again.
                    try:
                        await _deliver()
                    except asyncio.CancelledError:
                        raise
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            "signal delivery failed after node %s", node_id,
                            exc_info=True)
                outputs[node_id] = result
                # A3: capture floating tensors for the upcoming backward pass.
                if context is not None and getattr(context, "backward_mode", False):
                    attach_retain_grad(result, context.grad_targets, node_id, "")
                    # The empty initial port name yields keys like
                    # ``(node_id, "tensor")`` for ``result["tensor"]`` —
                    # see attach_retain_grad's recursion through dict keys.
                if cache is not None and node_cacheable and node_id in node_cache_keys:
                    cache.put(node_cache_keys[node_id], result)
                if record_outputs and output_store is not None and run_id:
                    for port, value in result.items():
                        if port.startswith("__"):
                            continue
                        await output_store.put(run_id, node_id, port, value)
                    # Expand __steps__ from instrumented nodes into individual
                    # entries so the Teaching Inspector can fetch them via the
                    # standard /api/execution/outputs/{run_id}/{node_id}/{port}
                    # endpoint, with metadata accessible via __steps_index.
                    raw_steps = result.get("__steps__")
                    if raw_steps:
                        for i, step in enumerate(raw_steps):
                            if not isinstance(step, Step):
                                continue
                            for tname, tensor in step.tensors.items():
                                await output_store.put(
                                    run_id, node_id, f"__step__{i}__{tname}", tensor
                                )
                            await output_store.put(
                                run_id, node_id, f"__step__{i}__meta",
                                {
                                    "name": step.name,
                                    "description": step.description,
                                    "scalars": step.scalars,
                                    "tensor_keys": list(step.tensors.keys()),
                                },
                            )
                # A node that stopped on ``context.should_stop()`` returns
                # partial outputs and says so; it did not complete.
                terminal = ("interrupted"
                            if isinstance(result, dict)
                            and result.get(INTERRUPTED_KEY) else "completed")
                await _emit_preset_aware(node_id, terminal, result)
                return
            except Exception as e:
                last_error = e
                if attempt < attempts - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))  # backoff

        # All attempts failed
        assert last_error is not None
        error_detail: dict[str, str] = {"error": str(last_error)}
        if settings.DEBUG:
            error_detail["traceback"] = traceback.format_exc()
        if error_mode == "fail_fast":
            await _emit_preset_aware(node_id, "error", error_detail)
            raise last_error
        else:
            # continue or retry-exhausted
            node_errors[node_id] = str(last_error)
            await _emit_preset_aware(node_id, "error", error_detail)

    pump_task = asyncio.create_task(_pump(), name=f"outbox-pump:{run_id or '-'}")
    try:
        # Before the forward pass: zero any accumulated gradients on persisted
        # modules so backward_mode doesn't keep summing across runs.
        if context is not None and getattr(context, "backward_mode", False):
            zero_module_grads(context.node_state_store, context.graph_id)

        # Execute level by level
        for level in levels:
            if context and context.cancelled:
                raise CancellationError()

            if len(level) == 1:
                await _execute_single_node(level[0])
            else:
                # Run independent nodes in this level concurrently
                tasks = [asyncio.create_task(_execute_single_node(nid)) for nid in level]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, CancellationError):
                        raise result
                    if isinstance(result, Exception):
                        if error_mode == "fail_fast":
                            # Cancel remaining tasks
                            for t in tasks:
                                t.cancel()
                            raise result

        # A stop that arrived during the LAST level has no next iteration to
        # be caught by (#122). Without this, a graph whose final node is the
        # training loop would return normally and the run would be filed
        # ``succeeded`` — after the user pressed Stop and the node returned
        # partial results saying so.
        if context and context.cancelled:
            raise CancellationError()

        # A3: post-forward backward pass + gradient capture.
        if (
            context is not None
            and getattr(context, "backward_mode", False)
            and run_id
            and output_store is not None
        ):
            target = select_backward_target(
                expanded_nodes,
                outputs,
                auto_backward=getattr(context, "auto_backward", False),
            )
            if target is not None:
                loss, _label = target
                try:
                    run_backward(loss)
                    await capture_grads(
                        context.grad_targets,
                        context.node_state_store,
                        context.graph_id,
                        run_id,
                        output_store,
                    )
                except Exception as exc:  # backward errors shouldn't kill the run
                    logger.warning("backward pass failed: %s", exc)

        return outputs
    finally:
        # Unbind first (sync, cannot fail), then flush what is still queued
        # BEFORE stopping the pump: a run that unwound on cancellation has
        # the interrupt checkpoint's artifact signal sitting in the outbox,
        # and losing it would mean a checkpoint on disk that no run row
        # knows about.
        #
        # This flush is the second half of ArtifactSignal's tail-safety
        # obligation (see its docstring). Drop-oldest means the newest
        # survive, so a signal whose loss has consequences outside
        # observability must be queued LAST by its producer and drained
        # here; nothing may be enqueued after this point.
        outbox.unbind()
        try:
            await _deliver()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a lost event must not mask the outcome
            logger.warning("final signal flush failed", exc_info=True)
        finally:
            pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump_task


async def _maybe_await(val: Any) -> Any:
    if asyncio.iscoroutine(val):
        return await val
    return val


# Annotation-only imports: these two would close an import cycle at module
# level. ``execution_context`` no longer needs to be one of them — #122 needs
# ``EventOutbox`` and the signal types at RUNTIME here, and that module
# imports nothing from this one.
if False:  # TYPE_CHECKING
    from .cache import ExecutionCache
    from .run_output_store import RunOutputStore
