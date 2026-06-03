from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import traceback
from collections import defaultdict, deque
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
from .node_base import BaseNode
from .node_registry import registry
from .step_trace import Step
from .type_system import is_compatible

logger = logging.getLogger(__name__)


class GraphValidationError(Exception):
    pass


def expand_presets(
    nodes: list[dict],
    edges: list[dict],
) -> tuple[list[dict], list[dict], dict[str, str]]:
    """Expand preset nodes into their sub-graph of real nodes.

    Returns (expanded_nodes, expanded_edges, internal_to_preset_map).
    internal_to_preset_map maps internal node IDs to the preset node ID they came from.
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
        preset = preset_registry.get(preset_name)
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


def validate_graph(nodes: list[dict], edges: list[dict]) -> list[str]:
    """Validate a graph definition. Returns list of errors (empty = valid)."""
    errors: list[str] = []
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
            if not preset_registry.get(preset_name):
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
        for inp in node_cls.define_inputs():
            if not inp.optional and (node["id"], inp.name) not in connected_inputs:
                errors.append(
                    f"Missing required input '{inp.name}' on node {node['id']} ({node['type']})"
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
        # define_outputs_dynamic lets param-driven nodes (e.g. SplitNode whose
        # `chunks` param decides how many chunk_i ports exist) expose their
        # live port set. Nodes that don't override fall back to the static
        # define_outputs() via BaseNode's default.
        src_params = src.get("data", {}).get("params", {}) if isinstance(src.get("data"), dict) else {}
        src_outputs = {p.name: p for p in src_cls.define_outputs_dynamic(src_params)}
        tgt_inputs = {p.name: p for p in tgt_cls.define_inputs()}

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
        adj[edge["source"]].append(edge["target"])
        if edge["target"] in in_degree:
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
    """
    from .execution_context import CancellationError

    # Expand preset nodes iteratively (handles nested presets)
    internal_to_preset: dict[str, str] = {}
    expanded_nodes, expanded_edges = nodes, edges
    for _ in range(10):  # max nesting depth
        has_preset = any(n.get("type", "").startswith("preset:") for n in expanded_nodes)
        if not has_preset:
            break
        expanded_nodes, expanded_edges, mapping = expand_presets(expanded_nodes, expanded_edges)
        internal_to_preset.update(mapping)

    # Filter to the executable subgraph: the nodes reachable from any entry
    # point via data edges (plus the entry points themselves). Draft
    # components (graph fragments with no entry point) are silently skipped.
    entry_ids = find_entry_points(expanded_nodes, expanded_edges)
    if not entry_ids:
        # validate_graph would catch this, but defend in depth.
        raise GraphValidationError("Graph has no entry points")

    executable_ids = reachable_from_entry_points(entry_ids, expanded_edges)

    # Include Start nodes whose trigger targets are executable, so that
    # trigger edges are preserved for validate_graph's entry-point detection.
    for e in expanded_edges:
        if e.get("type", "data") == "trigger" and e["target"] in executable_ids:
            executable_ids.add(e["source"])

    # Every execution path must originate from a Start node: a node is only
    # executable if it (or an upstream node along its data path) carries an
    # incoming trigger edge. Producers with no trigger — even ones that feed a
    # required input of an otherwise-reachable node — are NOT auto-included;
    # the user must wire a Start node into that producer to run it. This keeps
    # the entry-point contract simple: if it's not reachable from a Start, it
    # doesn't run (validate_graph will then flag any missing required input).

    # If any internal node of a preset is reachable, include ALL sibling
    # nodes from that preset.  A preset is a logical unit — its internal
    # root nodes (e.g. Dataset, Loss) have no incoming external edges and
    # would otherwise be pruned, breaking the internal wiring.
    presets_to_include: set[str] = set()
    for internal_id, preset_id in internal_to_preset.items():
        if internal_id in executable_ids:
            presets_to_include.add(preset_id)
    for internal_id, preset_id in internal_to_preset.items():
        if preset_id in presets_to_include:
            executable_ids.add(internal_id)

    expanded_nodes = [n for n in expanded_nodes if n["id"] in executable_ids]
    expanded_edges = [
        e
        for e in expanded_edges
        if e["source"] in executable_ids and e["target"] in executable_ids
    ]

    errors = validate_graph(expanded_nodes, expanded_edges)
    if errors:
        raise GraphValidationError("; ".join(errors))

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

        if status == "error":
            # Any internal failure fails the whole preset immediately
            await _maybe_await(on_progress(preset_id, "error", data))
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
            cache_key = cache.compute_key(node_type, params, upstream_keys)
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

                    # Thread-safe progress bridge: sync thread → async on_progress
                    def _progress_bridge(data: dict) -> None:
                        if on_progress:
                            future = asyncio.run_coroutine_threadsafe(
                                _emit_preset_aware(node_id, "progress", data),
                                loop,
                            )
                            try:
                                future.result(timeout=10)
                            except Exception:
                                pass

                    # Tell stateful nodes which node-id they belong to.
                    if context is not None:
                        context.current_node_id = node_id

                    # Build the kwargs dict matching the node's execute() signature.
                    sig = inspect.signature(instance.execute)
                    call_kwargs: dict[str, Any] = {}
                    if 'progress_callback' in sig.parameters:
                        call_kwargs['progress_callback'] = _progress_bridge
                    if 'context' in sig.parameters:
                        call_kwargs['context'] = context
                    fn = functools.partial(instance.execute, inputs, params, **call_kwargs)
                    result = await loop.run_in_executor(None, fn)
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
                await _emit_preset_aware(node_id, "completed", result)
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


async def _maybe_await(val: Any) -> Any:
    if asyncio.iscoroutine(val):
        return await val
    return val


# Avoid circular import at module level — these are imported lazily inside execute_graph
if False:  # TYPE_CHECKING
    from .cache import ExecutionCache
    from .execution_context import ExecutionContext
    from .run_output_store import RunOutputStore
