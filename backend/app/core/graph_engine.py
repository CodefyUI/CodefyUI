from __future__ import annotations

import asyncio
import contextlib
import copy
import functools
import inspect
import logging
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..config import settings
from .backward_pass import (
    attach_retain_grad,
    capture_grads,
    run_backward,
    select_backward_target,
    zero_module_grads,
)
from .error_handling import (
    NodeOOMError,
    current_cuda_device,
    is_out_of_memory,
    memory_digest,
    release_cached_memory,
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
from .seeding import deterministic_scope, seed_rngs
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


# ── Subgraphs (core#137) ─────────────────────────────────────────────────
#
# A subgraph is a reusable block of graph that lives in the graph file's own
# ``subgraphs`` list. The canvas shows one ``subgraph:<id>`` INSTANCE node per
# use; every instance of the same id shares one definition, which is the whole
# reuse win over a preset that was flattened once at drop time.
#
# Nothing below this point in the engine knows subgraphs exist. Expansion runs
# as a pre-pass in ``prepare_executable_graph`` and hands back the same
# ``internal node -> container node`` map preset expansion produces, so
# reachability retention, status roll-up and the exporter's grouping treat a
# subgraph instance exactly as they already treat a preset node -- with one
# difference nesting forces: that map is a CHAIN, not a lookup table, so
# anything asking "which box does this node live in?" has to walk it. See
# :func:`outermost_container`.

SUBGRAPH_TYPE_PREFIX = "subgraph:"

#: Flattened inner ids are ``<instance id>/<inner id>``. The separator is not
#: a legal character in a Python identifier, and every place that turns a node
#: id into one already sanitizes (``codegen._slug``, ``checkpoints._safe_part``),
#: so a flattened id never reaches generated code or a file name as-is.
#:
#: What it does NOT give is an injective flattening. Instance ``x`` holding an
#: inner node called ``a/b``, and instance ``x/a`` holding an inner node called
#: ``b``, both produce ``x/a/b``. Editor-made ids are UUIDs and cannot collide
#: this way, but Import JSON, plugin-created nodes and hand-edited files are
#: unconstrained -- so :func:`expand_subgraphs` refuses a duplicate expanded id
#: by name instead of letting one of the two nodes silently disappear.
SUBGRAPH_SEPARATOR = "/"

#: Same budget preset nesting gets. A definition that (transitively) contains
#: itself is rejected BEFORE this by :func:`check_subgraph_recursion`, with a
#: message naming the loop; the budget is the backstop for pathological depth.
MAX_SUBGRAPH_DEPTH = 10


def subgraph_id_of(node_type: Any) -> str | None:
    """``"subgraph:blk"`` -> ``"blk"``; anything else -> ``None``."""
    text = str(node_type or "")
    if text.startswith(SUBGRAPH_TYPE_PREFIX):
        return text[len(SUBGRAPH_TYPE_PREFIX):]
    return None


@dataclass(frozen=True)
class MalformedSubgraph:
    """A ``subgraphs`` entry that does not parse, remembered by id.

    Dropping the entry is still right for a definition nobody instantiates --
    that is what keeps stale data in a graph file from breaking a run. But
    dropping it WITHOUT A TRACE made the next message a lie: an instance that
    referenced it was told ``Unknown subgraph: d``, sending the user to look
    for a definition sitting right there in the file. Keeping the id and the
    reason lets :func:`expand_subgraphs` say what actually happened.

    Deliberately NOT a :class:`SubgraphDefinition` and deliberately without a
    ``nodes`` attribute: anything that walks the index has to notice it rather
    than quietly expand an empty block.
    """

    id: str
    #: Whitespace-collapsed and length-capped, because it is quoted into an
    #: error line that :func:`prepare_executable_graph` joins with "; " before
    #: raising. (:func:`validate_graph` returns its errors as an unjoined
    #: ``list[str]``; the join is on the raise path, not the report path.)
    error: str


def build_subgraph_index(subgraphs: Any) -> dict:
    """Map subgraph id -> definition for a graph's own ``subgraphs``.

    Mirrors :func:`build_preset_fallback`: accepts model objects or plain
    dicts (``json.loads`` output). An entry that does not parse is NOT
    expandable, but it is not forgotten either -- it is recorded as a
    :class:`MalformedSubgraph` so that an unreferenced bad definition still
    cannot break a run, while a REFERENCED one is reported as broken instead
    of missing. A definition that is genuinely absent is caught later, by
    name, in :func:`expand_subgraphs`.
    """
    from ..schemas.models import SubgraphDefinition

    out: dict[str, Any] = {}
    for raw in subgraphs or []:
        try:
            model = (
                raw
                if isinstance(raw, SubgraphDefinition)
                else SubgraphDefinition(**raw)
            )
        except Exception as exc:
            sid = (
                raw.get("id")
                if isinstance(raw, dict)
                else getattr(raw, "id", None)
            )
            # An entry with no usable id cannot be referenced by name either,
            # so there is nothing a later message could say about it.
            # A parsable definition wins over a broken one with the same id,
            # whichever order the two arrive in.
            if sid and not isinstance(out.get(str(sid)), SubgraphDefinition):
                reason = " ".join(str(exc).split())
                if len(reason) > 300:
                    reason = reason[:300] + " ..."
                out[str(sid)] = MalformedSubgraph(str(sid), reason)
            continue
        if model.id:
            out[model.id] = model
    return out


def duplicate_node_id_errors(nodes: list[dict]) -> list[str]:
    """One line per id the graph carries more than once, in first-seen order.

    Wording matches :func:`expand_subgraphs`' same-origin refusal exactly,
    because it IS the same fault: nothing about a boundary is involved, and a
    message mentioning expansion would send the user to the wrong place.

    Lives outside expansion because expansion is not always reached --
    :func:`expand_subgraphs_deep` returns early for a graph with no instances
    -- and "the graph is only checked for duplicate ids when it happens to
    contain a block" is not a rule anyone could have predicted.
    """
    seen: set[str] = set()
    reported: set[str] = set()
    errors: list[str] = []
    for node in nodes:
        node_id = node.get("id", "")
        if node_id in seen and node_id not in reported:
            reported.add(node_id)
            errors.append(
                f"Duplicate node id: the node '{node_id}' appears more "
                "than once"
            )
        seen.add(node_id)
    return errors


def preset_subgraph_errors(
    nodes: list[dict],
    preset_fallback: dict | None = None,
) -> list[str]:
    """One line per preset node whose definition names a subgraph instance.

    A preset is PORTABLE -- it outlives the graph it was made from and can be
    dropped into any other -- while a subgraph id is graph-LOCAL, resolved
    only against the file's own ``subgraphs`` list. A preset carrying a
    ``subgraph:<id>`` node is therefore a reference that is meaningless
    everywhere except in the one graph it came from, and even there it does
    not work: expansion runs subgraphs BEFORE presets and never revisits, so
    the instance the preset unpacks reaches the executor unexpanded. Before
    this check, validate answered ``{"valid": true}`` and the run died with
    ``Unknown subgraph: leaf`` for a definition sitting right there in the
    file -- both of the disagreements this feature set out to remove, at
    once.

    Refused rather than supported: making the expansion loop alternate would
    "work" in the source graph and then break the moment the preset was used
    anywhere else, which is a worse failure than this one. ``/api/presets/
    create`` refuses to build such a preset in the first place; this catches
    the ones that arrive in a file's own ``presets[]``.

    Reported (not raised) so it can join validate's other findings, and
    computed BEFORE expansion so the message can name the preset node the
    user can actually see.
    """
    from .preset_registry import preset_registry

    errors: list[str] = []
    for node in nodes:
        node_type = str(node.get("type", ""))
        if not node_type.startswith("preset:"):
            continue
        preset_name = node_type[len("preset:"):]
        candidates = [
            candidate
            for candidate in (
                preset_registry.get(preset_name),
                (preset_fallback or {}).get(preset_name),
            )
            if candidate is not None
        ]
        offenders: set[str] = set()
        for preset in candidates:
            internal_nodes = (
                preset.get("nodes", [])
                if isinstance(preset, dict)
                else preset.nodes
            )
            for internal in internal_nodes:
                internal_type = (
                    internal.get("type", "")
                    if isinstance(internal, dict)
                    else internal.type
                )
                if subgraph_id_of(internal_type) is not None:
                    internal_id = (
                        internal.get("id", "")
                        if isinstance(internal, dict)
                        else internal.id
                    )
                    offenders.add(str(internal_id))
        if offenders:
            errors.append(
                f"Preset '{preset_name}' contains subgraph instance(s) "
                f"{', '.join(sorted(offenders))} (node {node.get('id', '')}). "
                "A preset is portable and a subgraph id is local to one "
                "graph, so the reference cannot travel with it -- expand the "
                "block before creating the preset."
            )
    return errors


def check_subgraph_recursion(
    index: dict,
    roots: Iterable[str] | None = None,
) -> None:
    """Raise if any definition (transitively) instantiates itself.

    A recursive definition would expand forever. Exhausting
    :data:`MAX_SUBGRAPH_DEPTH` would also stop it, but with a message that
    names nothing; this one names the loop, which is the only way a user can
    find the edit that caused it.

    ``roots`` scopes the scan to the definitions a particular graph actually
    instantiates. Without it EVERY entry in the index is scanned, and one
    stale self-referential definition -- data the run never touches -- refuses
    the whole run. Omitting it keeps that whole-index behaviour, which is all
    a caller holding an index but no graph can ask for.
    """
    state: dict[str, int] = {}  # 0 = on stack, 1 = cleared
    path: list[str] = []

    def visit(sid: str) -> None:
        if state.get(sid) == 1:
            return
        if state.get(sid) == 0:
            loop = path[path.index(sid):] + [sid]
            raise GraphValidationError(
                "Subgraph recursion: "
                + " -> ".join(loop)
                + " (a subgraph cannot contain itself)"
            )
        state[sid] = 0
        path.append(sid)
        # A :class:`MalformedSubgraph` has no ``nodes`` to walk. It cannot
        # close a loop, and an instance that references it is refused by name
        # in :func:`expand_subgraphs`, so there is nothing to say here.
        for node in getattr(index[sid], "nodes", None) or ():
            child = subgraph_id_of(node.type)
            if child is not None and child in index:
                visit(child)
        path.pop()
        state[sid] = 1

    scan = list(index) if roots is None else list(dict.fromkeys(roots))
    for sid in scan:
        # A root naming a definition the index does not have is not this
        # function's problem -- ``expand_subgraphs`` reports it as unknown.
        if sid in index:
            visit(sid)


def _inner_roots(definition: Any) -> list[str]:
    """Inner nodes nothing else inside the definition feeds.

    Used only as the fallback for an instance whose interface declares no
    ``triggerTargets`` -- see :func:`expand_subgraphs`.
    """
    fed: set[str] = {
        edge.target
        for edge in definition.edges
        if edge.target
    }
    return [node.id for node in definition.nodes if node.id not in fed]


def expand_subgraphs(
    nodes: list[dict],
    edges: list[dict],
    index: dict,
) -> tuple[list[dict], list[dict], dict[str, str]]:
    """Inline one level of ``subgraph:<id>`` instance nodes.

    Returns ``(nodes, edges, internal_to_instance)`` -- the same triple shape
    :func:`expand_presets` returns, so callers can merge the two maps.

    Boundary rules
    --------------
    * A DATA edge into the instance is retargeted at the inner node/port the
      interface's matching input names; a data edge out of it is re-sourced
      from the matching output. An edge naming a port the interface does not
      declare is an error, not a silently dropped edge -- dropping it is how
      a preset run turns into a confusing "missing required input" further
      downstream.
    * A TRIGGER edge into the instance is REPLICATED to every inner node in
      ``interface.triggerTargets``. Collapse records exactly the inner nodes
      that had a trigger edge before it ran, so the expanded graph has the
      same entry points -- which is what makes collapse/expand produce the
      same run. When the interface declares none (a hand-written definition,
      or an instance the user wired Start into after collapsing), the trigger
      fans out to the inner ROOTS instead, which is the only reading of
      "start this block" that reaches every one of its sources.
    * Params ride along verbatim from the definition. v1 has no per-instance
      overrides, so two instances of one subgraph are byte-identical blocks.

    Every id the flattened graph ends up with is claimed exactly once. Two
    ids that flatten to the same string (see :data:`SUBGRAPH_SEPARATOR`) are
    refused here, naming both contributors, because the alternative is that
    one of the two nodes disappears and the run dies much later in
    ``topological_levels`` -- reporting a cycle that does not exist, purely
    because its node count no longer matches the graph's.
    """
    expanded_nodes: list[dict] = []
    expanded_edges: list[dict] = list(edges)
    internal_to_instance: dict[str, str] = {}
    #: expanded id -> human description of what produced it.
    origin_of: dict[str, str] = {}

    def claim(node_id: str, origin: str) -> None:
        previous = origin_of.get(node_id)
        if previous == origin:
            # Not a flattening collision at all -- the graph itself carries the
            # id twice. Saying "after subgraph expansion" would send the user
            # looking at a boundary that has nothing to do with it.
            raise GraphValidationError(
                f"Duplicate node id: {origin} appears more than once"
            )
        if previous is not None:
            raise GraphValidationError(
                f"Duplicate node id after subgraph expansion: '{node_id}' is "
                f"produced by both {previous} and {origin}. Ids are flattened "
                f"to '<instance>{SUBGRAPH_SEPARATOR}<inner node>', so rename "
                f"one of them so the two cannot meet."
            )
        origin_of[node_id] = origin

    for node in nodes:
        sid = subgraph_id_of(node.get("type", ""))
        if sid is None:
            # ``.get`` because a node dict with no id is a broken graph, not a
            # crash: a KeyError here would turn a reportable fault into a 500.
            outer_id = node.get("id", "")
            claim(outer_id, f"the node '{outer_id}'")
            expanded_nodes.append(node)
            continue

        instance_id = node["id"]
        # Claimed like any other id even though the instance node itself does
        # NOT survive expansion. Skipping it let two instances share one id
        # whenever their definitions had disjoint inner ids: nothing collided,
        # the run exited clean, and the second block's boundary edges -- which
        # address the id, not the node -- were consumed by the first, so it
        # contributed nothing and nothing said so. The docstring's "every id
        # the flattened graph ends up with is claimed exactly once" was true
        # only because the instance id was consumed rather than kept.
        claim(instance_id, f"the node '{instance_id}'")
        if not sid:
            # ``subgraph:`` with nothing after it. ``subgraph_id_of`` answers
            # "" -- a real answer, not None -- so the node IS an instance, of
            # nothing. Named on its own terms: "Unknown subgraph:  (node x)",
            # with a blank where the id should be, is not actionable.
            raise GraphValidationError(
                f"Node {instance_id} has type '{SUBGRAPH_TYPE_PREFIX}' but "
                "names no subgraph"
            )
        definition = index.get(sid)
        if isinstance(definition, MalformedSubgraph):
            raise GraphValidationError(
                f"Subgraph '{sid}' is in this graph but could not be read, so "
                f"node {instance_id} cannot be expanded: {definition.error}"
            )
        if definition is None:
            raise GraphValidationError(
                f"Unknown subgraph: {sid} (node {instance_id})"
            )

        prefix = f"{instance_id}{SUBGRAPH_SEPARATOR}"
        inner_ids = {inner.id for inner in definition.nodes}

        input_map: dict[str, tuple[str, str]] = {}
        for port in definition.interface.inputs:
            input_map[port.port] = (f"{prefix}{port.innerNode}", port.innerPort)
        output_map: dict[str, tuple[str, str]] = {}
        for port in definition.interface.outputs:
            output_map[port.port] = (f"{prefix}{port.innerNode}", port.innerPort)

        trigger_targets = [
            target for target in definition.interface.triggerTargets
        ]
        unknown_targets = [t for t in trigger_targets if t not in inner_ids]
        if unknown_targets:
            raise GraphValidationError(
                f"Subgraph '{sid}' declares trigger target(s) that are not in "
                f"it: {', '.join(sorted(unknown_targets))} (node {instance_id})"
            )
        if not trigger_targets:
            trigger_targets = _inner_roots(definition)

        for inner in definition.nodes:
            claim(
                f"{prefix}{inner.id}",
                f"the node '{inner.id}' inside instance '{instance_id}'",
            )
            expanded_nodes.append({
                "id": f"{prefix}{inner.id}",
                "type": inner.type,
                # Inner positions are the definition's own layout. Nothing in
                # execution reads them; keeping them truthful means an
                # expanded graph dumped for debugging still looks like the
                # block the user drew.
                "position": dict(inner.position),
                "data": copy.deepcopy(inner.data),
            })
            internal_to_instance[f"{prefix}{inner.id}"] = instance_id

        for inner_edge in definition.edges:
            expanded_edges.append({
                "id": f"{prefix}{inner_edge.id}",
                "source": f"{prefix}{inner_edge.source}",
                "target": f"{prefix}{inner_edge.target}",
                "sourceHandle": inner_edge.sourceHandle,
                "targetHandle": inner_edge.targetHandle,
                "type": inner_edge.type,
            })

        new_edges: list[dict] = []
        for edge in expanded_edges:
            is_trigger = edge.get("type", "data") == "trigger"
            touches_target = edge.get("target") == instance_id
            touches_source = edge.get("source") == instance_id
            if not touches_target and not touches_source:
                new_edges.append(edge)
                continue

            if is_trigger and touches_target:
                for position, inner_id in enumerate(trigger_targets):
                    fanned = dict(edge)
                    fanned["id"] = f"{edge.get('id', 'trigger')}#{position}"
                    fanned["target"] = f"{prefix}{inner_id}"
                    new_edges.append(fanned)
                continue

            new_edge = dict(edge)
            if touches_target:
                handle = edge.get("targetHandle", "")
                if handle not in input_map:
                    raise GraphValidationError(
                        f"Edge targets input port '{handle}' which subgraph "
                        f"'{sid}' does not expose (node {instance_id})"
                    )
                new_edge["target"], new_edge["targetHandle"] = input_map[handle]
            if touches_source:
                handle = edge.get("sourceHandle", "")
                if handle not in output_map:
                    raise GraphValidationError(
                        f"Edge sources output port '{handle}' which subgraph "
                        f"'{sid}' does not expose (node {instance_id})"
                    )
                new_edge["source"], new_edge["sourceHandle"] = output_map[handle]
            new_edges.append(new_edge)
        expanded_edges = new_edges

    return expanded_nodes, expanded_edges, internal_to_instance


def expand_subgraphs_deep(
    nodes: list[dict],
    edges: list[dict],
    index: dict,
) -> tuple[list[dict], list[dict], dict[str, str]]:
    """Expand nested subgraph instances until none are left.

    Recursion is refused up front by name; the depth budget is only reached
    by a definition chain genuinely more than :data:`MAX_SUBGRAPH_DEPTH` deep.

    Every "is there anything to expand?" test below asks ``is not None``, not
    truthiness. ``subgraph_id_of("subgraph:")`` answers ``""`` -- an instance
    of nothing -- and a truthiness test skips it here while
    :func:`validate_graph` still treats it as a valid opaque container, so the
    graph validates clean and dies at run time.
    """
    # Before the early return below, so a duplicate id is refused for every
    # graph rather than only for one that happens to contain a block.
    duplicates = duplicate_node_id_errors(nodes)
    if duplicates:
        raise GraphValidationError("; ".join(duplicates))
    instance_ids = [
        sid
        for sid in (subgraph_id_of(n.get("type", "")) for n in nodes)
        if sid is not None
    ]
    if not instance_ids:
        # Nothing to EXPAND, so nothing further to check: a graph carrying
        # definitions it no longer instantiates still runs.
        return nodes, edges, {}
    # Scoped to what this graph actually uses. Scanning the whole index would
    # let one stale self-recursive definition -- unreachable from every
    # instance here -- refuse a run that never touches it.
    check_subgraph_recursion(index, roots=instance_ids)
    internal_to_instance: dict[str, str] = {}
    for _ in range(MAX_SUBGRAPH_DEPTH):
        if not any(
            subgraph_id_of(n.get("type", "")) is not None for n in nodes
        ):
            return nodes, edges, internal_to_instance
        nodes, edges, mapping = expand_subgraphs(nodes, edges, index)
        internal_to_instance.update(mapping)
    if any(subgraph_id_of(n.get("type", "")) is not None for n in nodes):
        raise GraphValidationError(
            f"Subgraph nesting exceeds the maximum depth of {MAX_SUBGRAPH_DEPTH}"
        )
    return nodes, edges, internal_to_instance


def outermost_container(node_id: str, mapping: dict[str, str]) -> str | None:
    """Resolve ``internal node -> container`` transitively. ``None`` if free.

    The map :func:`prepare_executable_graph` returns is a CHAIN as soon as
    subgraphs nest: an instance ``blk`` holding an instance ``nest`` holding
    node ``mul`` produces ``{'blk/nest': 'blk', 'blk/nest/mul': 'blk/nest'}``.
    A single ``mapping.get(node_id)`` therefore answers ``blk/nest`` -- an id
    that existed only between two expansion passes, that no client can
    resolve to anything on the canvas -- while ``blk``, the box the user is
    actually watching, is never named at all.

    Walks the map rather than splitting on :data:`SUBGRAPH_SEPARATOR`, so an
    instance whose OWN id contains a separator resolves to itself instead of
    to whatever precedes its first slash.

    A malformed map that points at itself, or in a ring, stops at the first
    id seen twice. Nothing here is worth hanging a run for.
    """
    container = mapping.get(node_id)
    if container is None:
        return None
    seen = {node_id}
    while container not in seen:
        seen.add(container)
        parent = mapping.get(container)
        if parent is None:
            break
        container = parent
    return container


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


def container_bypass_errors(
    nodes: list[dict],
    *,
    include_presets: bool = True,
) -> list[str]:
    """Refuse ``bypassed`` on nodes whose ports come from a definition.

    A preset or a subgraph instance expands into a sub-graph whose ports come
    from a definition rather than a node class, so the pass-through rule above
    (first type-compatible input wins) has nothing to match on.

    Every caller runs this BEFORE expansion, because expansion is what removes
    the node that has to be named: once an instance is inlined there is no
    ``bypassed`` flag left for :func:`resolve_bypass` to object to. That is how
    ``POST /api/graph/validate`` came to green-light a graph that
    :func:`prepare_executable_graph` then refused.

    ``include_presets=False`` is for :func:`validate_graph`, which does NOT
    expand preset nodes -- they survive into :func:`resolve_bypass`, which
    names them itself, and saying it twice helps nobody.
    """
    errors: list[str] = []
    if include_presets:
        bypassed_presets = sorted(
            node["id"]
            for node in nodes
            if _is_bypassed(node)
            and str(node.get("type", "")).startswith("preset:")
        )
        if bypassed_presets:
            errors.append(
                "Bypass is not supported on preset node(s): "
                + ", ".join(bypassed_presets)
            )
    bypassed_subgraphs = sorted(
        node["id"]
        for node in nodes
        if _is_bypassed(node)
        and subgraph_id_of(node.get("type", "")) is not None
    )
    if bypassed_subgraphs:
        errors.append(
            "Bypass is not supported on subgraph instance(s): "
            + ", ".join(bypassed_subgraphs)
        )
    return errors


def validate_graph(
    nodes: list[dict],
    edges: list[dict],
    preset_fallback: dict | None = None,
    subgraphs: Any = None,
) -> list[str]:
    """Validate a graph definition. Returns list of errors (empty = valid).

    ``preset_fallback`` (ID6) lets a graph-embedded preset (one the
    server's registry does not know) validate as present -- see
    ``build_preset_fallback``.

    ``subgraphs`` (core#137) is the graph's own definition list. When it is
    given, subgraph instances are INLINED before anything else is checked, so
    validation sees the graph that would actually run: port types across the
    boundary, required inputs inside the definition, and -- the reason this
    matters most -- a cycle that only exists once the boundary is crossed.
    Treating an instance as opaque the way a preset node is treated would let
    a definition-internal loop reach the executor, where it surfaces as a
    stall rather than an error naming the nodes involved.

    Bypassed nodes (core#128) are resolved away first, so every check below
    runs against the graph that would actually execute: a node whose only
    upstream is bypassed is checked against what the bypass forwards, not
    against the node the user muted.
    """
    errors: list[str] = []
    # Before expansion, because expansion is what removes the instance node
    # AND its ``bypassed`` flag -- see :func:`container_bypass_errors`. Preset
    # nodes are excluded: they are still here when `resolve_bypass` runs
    # below, and it names them itself.
    errors.extend(container_bypass_errors(nodes, include_presets=False))
    # Checked here, not only inside expansion: expansion is skipped entirely
    # for a graph with no instances, and validate must not answer "clean" for
    # a graph the run refuses.
    duplicate_errors = duplicate_node_id_errors(nodes)
    errors.extend(duplicate_errors)
    # A preset cannot carry a graph-local subgraph reference. Reported before
    # expansion, while the preset node is still standing to be named.
    errors.extend(preset_subgraph_errors(nodes, preset_fallback))
    if any(subgraph_id_of(n.get("type", "")) is not None for n in nodes):
        try:
            nodes, edges, _ = expand_subgraphs_deep(
                nodes, edges, build_subgraph_index(subgraphs)
            )
        except GraphValidationError as exc:
            # Report and keep going against the UNEXPANDED graph: the file's
            # standing rule is to show every problem at once, and an instance
            # whose definition is missing still has checkable neighbours.
            #
            # Skipped when the sentence is already in `errors`: expansion
            # re-raises the duplicate-id refusal, and one fault has to produce
            # one line.
            #
            # Two comparisons, not one. Expansion raises the duplicate lines
            # JOINED with "; " -- one string for the whole set -- while this
            # function reports them one per entry. With a single duplicate the
            # join equals that one entry and `not in errors` is enough; with
            # two or more it equals nothing in the list and the concatenation
            # is appended as a spurious extra line. Comparing against the join
            # of the list this function actually produced covers both, and
            # only that: a duplicate set expansion finds but this function did
            # not (ids that come into existence during expansion) still gets
            # reported.
            message = str(exc)
            if message not in errors and message != "; ".join(duplicate_errors):
                errors.append(message)

    resolution = resolve_bypass(nodes, edges)
    errors.extend(resolution.errors)
    nodes, edges = resolution.nodes, resolution.edges
    node_map = {n["id"]: n for n in nodes}

    # --- Node-level validation (standalone, before edge checks) ---
    from .preset_registry import preset_registry

    # 1. Node type existence check
    valid_node_ids: set[str] = set()
    # Nodes whose ports are NOT declared by a node class: preset nodes, and
    # subgraph instances that expansion refused. Port-level checks skip them.
    opaque_node_ids: set[str] = set()
    for node in nodes:
        node_type: str = node.get("type", "")
        # A subgraph instance only survives to here when expansion refused it
        # (unknown id, recursion, bad boundary port). That refusal is already
        # in `errors` by name; treat the instance as opaque so the user does
        # not also get "Unknown node type: subgraph:<id>" for the same fault.
        if subgraph_id_of(node_type) is not None:
            opaque_node_ids.add(node["id"])
            valid_node_ids.add(node["id"])
            continue
        # Preset nodes are expanded at execution time; validate they exist in preset registry
        if node_type.startswith("preset:"):
            preset_name = node_type[len("preset:"):]
            if not (preset_registry.get(preset_name) or (preset_fallback or {}).get(preset_name)):
                errors.append(f"Unknown preset: {preset_name} (node {node['id']})")
            else:
                opaque_node_ids.add(node["id"])
                valid_node_ids.add(node["id"])
            continue
        node_cls = registry.get(node_type)
        if node_cls is None:
            errors.append(f"Unknown node type: {node_type} (node {node['id']})")
        else:
            valid_node_ids.add(node["id"])

    # 2. Required input connection check (skip opaque nodes — dynamic ports)
    connected_inputs = {
        (edge["target"], edge.get("targetHandle", ""))
        for edge in edges
    }
    for node in nodes:
        if node["id"] not in valid_node_ids or node["id"] in opaque_node_ids:
            continue
        node_cls = registry.get(node["type"])
        node_params = node.get("data", {}).get("params", {}) if isinstance(node.get("data"), dict) else {}
        for inp in node_cls.define_inputs_dynamic(node_params):
            if not inp.optional and (node["id"], inp.name) not in connected_inputs:
                # A port left unconnected BY A BYPASS reads as the user's own
                # wiring mistake unless the message says otherwise (core#128).
                #
                # The plain case (no bypass cause) gets a short "what to do"
                # too (core#201 part 3): with the sibling-root/data-ancestor
                # rescue in prepare_executable_graph, this message can no
                # longer mean "the edge exists but its source was pruned for
                # lack of a trigger" -- that shape is retained and runs
                # before validation ever sees it. What is left, here or in
                # the editor's own live check, is always a port with no
                # edge at all, so "connect an output" is the complete fix,
                # not a guess among several.
                cause = resolution.dropped.get((node["id"], inp.name))
                errors.append(
                    f"Missing required input '{inp.name}' on node {node['id']} ({node['type']})"
                    + (
                        f" (input dropped because '{cause}' is bypassed)"
                        if cause
                        else " -- connect an output to this port"
                    )
                )

    # 3. Parameter range validation (skip opaque nodes)
    for node in nodes:
        if node["id"] not in valid_node_ids or node["id"] in opaque_node_ids:
            continue
        node_cls = registry.get(node["type"])
        param_values = node.get("data", {}).get("params", {})
        for param_def in node_cls.define_params():
            if param_def.name not in param_values:
                continue
            value = param_values[param_def.name]
            # Both comparisons under one guard, because a bound is only
            # meaningful against a value that can be ORDERED against it and
            # the check used to assume that silently (#193). ``"abc" < 1``
            # and ``None < 1`` both raise TypeError, and the raise escaped
            # ``validate_graph`` entirely: the run route is written to turn
            # this function's return value into a 409 ``invalid_graph``
            # envelope, so a value it could not compare arrived at the
            # client as a 500 with no idea which parameter caused it.
            #
            # Not hypothetical in either direction. A hand-authored
            # graph.json or a direct API POST can put a string on an INT
            # param; and the canvas's own numeric field yields NaN for a
            # CLEARED input, which ``JSON.stringify`` writes as ``null`` --
            # so a saved graph with an emptied box was a 500 waiting to be
            # re-run. A non-comparable value is a validation error, and is
            # now reported as one, naming the parameter like its
            # neighbours.
            #
            # ``except TypeError`` rather than an isinstance check on
            # (int, float): the predicate that matters is "does it order
            # against the bound", which numpy scalars and a plugin's own
            # numeric type satisfy without being either.
            try:
                below = (param_def.min_value is not None
                         and value < param_def.min_value)
                above = (param_def.max_value is not None
                         and value > param_def.max_value)
            except TypeError:
                errors.append(
                    f"Parameter '{param_def.name}' on node {node['id']} ({node['type']}): "
                    f"value {value!r} is not a number, so it cannot be "
                    f"checked against its allowed range"
                )
                continue
            if below:
                errors.append(
                    f"Parameter '{param_def.name}' on node {node['id']} ({node['type']}): "
                    f"value {value} is below minimum {param_def.min_value}"
                )
            if above:
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

        # Skip edge validation when either end is opaque (ports are dynamic)
        if src["id"] in opaque_node_ids or tgt["id"] in opaque_node_ids:
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
    cycle = find_cycle(executable_nodes, executable_edges)
    if cycle is not None:
        errors.append(describe_cycle(cycle))

    return errors


def find_cycle(nodes: list[dict], edges: list[dict]) -> list[str] | None:
    """Return one cycle as a closed path of node ids, or ``None``.

    ``["a", "b", "a"]`` reads left to right as the edges that close the loop.
    Trigger edges are markers, not dependencies, so they are excluded exactly
    as they are from :func:`topological_sort`; an edge whose endpoint is not
    in *nodes* is skipped, because edge validation already reports it and a
    KeyError here would replace a clean 4xx with a 500.
    """
    known = {n["id"] for n in nodes}
    adj: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge.get("type", "data") == "trigger":
            continue
        if edge["source"] not in known or edge["target"] not in known:
            continue
        adj[edge["source"]].append(edge["target"])

    WHITE, GREY, BLACK = 0, 1, 2
    color: dict[str, int] = {nid: WHITE for nid in known}
    path: list[str] = []

    # Iterative DFS: a deeply chained graph must not blow the Python stack on
    # what is only a validation pass.
    for root in (n["id"] for n in nodes):
        if color[root] != WHITE:
            continue
        stack: list[tuple[str, int]] = [(root, 0)]
        color[root] = GREY
        path.append(root)
        while stack:
            node, index = stack[-1]
            neighbors = adj[node]
            if index >= len(neighbors):
                stack.pop()
                color[node] = BLACK
                path.pop()
                continue
            stack[-1] = (node, index + 1)
            nxt = neighbors[index]
            if color[nxt] == GREY:
                return path[path.index(nxt):] + [nxt]
            if color[nxt] == WHITE:
                color[nxt] = GREY
                path.append(nxt)
                stack.append((nxt, 0))
    return None


def describe_cycle(cycle: list[str]) -> str:
    """One error line for a cycle, naming both sides of any boundary crossed.

    Flattened subgraph ids read ``<instance>/<inner>``, so the path itself
    already names the instance node AND the node inside it. The trailing
    clause states which instances are involved, because the loop the user has
    to break may be entirely inside a definition -- invisible on the canvas,
    where the instance is one opaque box.

    EVERY enclosing instance is named, not just the outermost one: an id like
    ``blk/nest/p`` sits inside ``blk`` (which the canvas shows) and inside
    ``blk/nest`` (which only the definition of ``blk`` shows), and the user
    has to open both to reach the edge that closes the loop. Order is
    outermost-first so the clause reads as a path inward.

    This is the one place that reads structure out of the id STRING rather
    than out of the container map, because a cycle is reported from a bare
    list of ids. It is sound for expanded ids -- :func:`expand_subgraphs`
    refuses a flattening that is ambiguous -- but a hand-written top-level id
    containing a slash will still be described as if it were namespaced.
    """
    rendered = " -> ".join(cycle)
    instances: list[str] = []
    for node_id in cycle:
        parts = node_id.split(SUBGRAPH_SEPARATOR)
        for depth in range(1, len(parts)):
            instance = SUBGRAPH_SEPARATOR.join(parts[:depth])
            # A leading separator yields an empty prefix, which names nothing.
            if instance and instance not in instances:
                instances.append(instance)
    if not instances:
        return f"Graph contains a cycle: {rendered}"
    return (
        f"Graph contains a cycle: {rendered} (crosses subgraph instance(s): "
        + ", ".join(instances)
        + ")"
    )


def _has_cycle(nodes: list[dict], edges: list[dict]) -> bool:
    return find_cycle(nodes, edges) is not None


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
    """Call ``instance.execute`` with exactly the keywords it declares, on
    inputs already aligned to the device this node runs on.

    Node authors opt in to ``progress_callback`` / ``context`` by naming them
    in their ``execute`` signature; undeclared ones are dropped here. Both the
    graph engine and exported Python runners route every node call through
    this helper, so invocation semantics can never drift between the canvas
    and an exported script.

    **Device alignment.** ``ExecutionContext.device`` describes one device for
    the whole run, and ``StatefulModuleMixin`` already puts every layer node's
    module there. The other half of that sentence -- getting the *tensors*
    there -- used to be each node's own job, and the record shows what that
    costs: a graph died with "Expected all tensors to be on the same device"
    until #347 taught ``PolicyRollout`` to bridge, and the identical failure
    came back through ``Conv2d`` from a source node that had never learned the
    same lesson. Fourteen nodes create tensors from nothing; four of them
    remembered.

    Aligning here makes it structural instead of remembered. Every node gets
    its inputs on its own device, including third-party plugin nodes and a
    teacher's custom node, neither of which any amount of reviewing the
    builtin set can reach. A tensor already in the right place is returned
    unchanged by ``.to()``, so the aligned case -- which is all of CPU-only
    running, and all of a correctly-behaved graph -- costs nothing.

    Only tensors move; see :func:`align_tensors` for why a module handed
    across a wire deliberately does not. A node whose work is genuinely
    host-side -- one that hands its input straight to numpy, sklearn or
    matplotlib -- says ``align_inputs = False`` and is left alone; see
    :attr:`BaseNode.align_inputs`.
    """
    if (
        context is not None
        and getattr(context, "device", None)
        and inputs
        and getattr(instance, "align_inputs", True)
    ):
        from .device_utils import align_tensors, node_target_device

        inputs = align_tensors(
            inputs, node_target_device(params, context, instance))

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
    subgraphs: Any = None,
) -> tuple[list[dict], list[dict], dict[str, str]]:
    """Expand subgraphs and presets, resolve bypass, prune drafts, validate.

    This is the structural preflight used immediately before execution. It is
    also safe for callers such as Python export that need the exact same preset
    grouping and draft-pruning semantics without actually running any nodes.

    The returned map is ``internal node id -> IMMEDIATELY ENCLOSING container
    node id`` and covers both preset internals and subgraph internals.
    Downstream readers -- reachability retention below, the status roll-up in
    :func:`execute_graph`, the exporter's grouping -- all read that one map,
    but none of them can read it with a single lookup: nesting makes it a
    CHAIN (``blk/nest/mul -> blk/nest -> blk``) whose intermediate links name
    ids that no longer exist in the returned node list. Every reader walks it
    with :func:`outermost_container` -- the retention below, the roll-up, and
    the exporter.
    """

    # Refused before expansion, while the container node still exists to be
    # named -- see :func:`container_bypass_errors`.
    container_bypass = container_bypass_errors(nodes)
    if container_bypass:
        raise GraphValidationError("; ".join(container_bypass))

    # A preset that names a subgraph is refused here, before either expansion
    # runs. The order below is one-way -- subgraphs, then presets, never back
    # -- so a `subgraph:` node arriving out of a preset would reach the
    # executor unexpanded. `validate_graph` reports the same thing, from the
    # same function, so the two surfaces cannot drift.
    preset_subgraphs = preset_subgraph_errors(nodes, preset_fallback)
    if preset_subgraphs:
        raise GraphValidationError("; ".join(preset_subgraphs))

    internal_to_preset: dict[str, str] = {}
    # Subgraphs first: a definition may contain preset nodes, so expanding it
    # feeds the preset loop below. The reverse is REFUSED rather than
    # supported (see `preset_subgraph_errors` just above) -- a preset's
    # internals can name a graph-local subgraph id, because
    # `build_preset_fallback` reads the graph's own client-supplied
    # `presets[]` and `/api/presets/create` used to copy node types verbatim.
    nodes, edges, subgraph_map = expand_subgraphs_deep(
        nodes, edges, build_subgraph_index(subgraphs)
    )
    internal_to_preset.update(subgraph_map)

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

    # `outermost_of` depends only on `internal_to_preset`, which the loop
    # below never modifies -- computed once rather than on every iteration.
    #
    # Grouped by the OUTERMOST container, not the immediately enclosing
    # one. Under nesting the only reachable node may be `blk/nest/mul`,
    # whose immediate container is `blk/nest`; grouping there retains
    # `nest`'s own contents and prunes the roots sitting directly inside
    # `blk`, so how deeply a user happened to nest a block would decide
    # whether its roots survive.
    outermost_of = {
        internal_id: outermost_container(internal_id, internal_to_preset)
        for internal_id in internal_to_preset
    }

    # Two rescue passes, iterated to a fixpoint (#201). Forward reachability
    # only ever walks from a node already known executable to what it
    # FEEDS -- a root with no INCOMING edge of its own (Dataset, Loss, the
    # head of a transform chain) is never the source side of that walk, so
    # it gets pruned even though something reachable has a port -- required
    # OR optional, this does not check which -- wired to its output. Both
    # passes below retain such a root once something it feeds survives;
    # iterating them together (rather than picking a fixed order) handles a
    # root that is itself inside a preset, or a preset sibling that itself
    # depends on an outside root, without having to prove one pass always
    # finishes what the other would still need to do.
    while True:
        before = len(executable_ids)

        # A container is one logical unit. If any internal node is
        # reachable, retain all sibling roots (for example Dataset and Loss
        # in a training preset) -- they have no incoming edge, so
        # reachability alone would prune them out of a block the user did
        # ask to run.
        containers_to_include = {
            container
            for internal_id, container in outermost_of.items()
            if internal_id in executable_ids
        }
        executable_ids.update(
            internal_id
            for internal_id, container in outermost_of.items()
            if container in containers_to_include
        )

        # A training graph built from plain nodes -- no preset, no
        # subgraph -- has the identical shape: `Dataset`, `Loss`, and the
        # head of each transform chain have no incoming edge at all and are
        # not part of any container, so the rescue above never sees them.
        # Retain every node that feeds a DATA edge into something already
        # executable -- the same "one logical unit" reasoning as the
        # container rescue above, applied to an ordinary wired connection
        # instead of a preset boundary. A root pulled in this way runs
        # unconditionally, with no trigger of its own, exactly like a
        # rescued preset sibling already does today (topological
        # scheduling only orders by DATA edges; trigger edges are markers,
        # not a precondition to run -- see topological_sort/_levels).
        for edge in expanded_edges:
            if (
                edge.get("type", "data") == "data"
                and edge["target"] in executable_ids
            ):
                executable_ids.add(edge["source"])

        if len(executable_ids) == before:
            break

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


#: How one internal node's terminal status contributes to the status its
#: CONTAINER (preset box or subgraph instance) reports (#260). The highest
#: rank any internal reached wins, so the box answers the only question the
#: user is actually asking of it -- "did this thing do anything?" -- and the
#: three answers are distinguishable instead of all being "completed":
#:
#:  - ``completed`` (2) beats everything: one node that really ran means the
#:    box really ran. A MIXED box (some cached, some executed) is genuinely
#:    a box that did work, so it must not be demoted to ``cached``.
#:  - ``skipped`` (1) beats ``cached``: ``skipped`` is emitted in exactly one
#:    place (``_execute_single_node``, "skipped: upstream node failed") and it
#:    means a node was NOT ATTEMPTED because something upstream blew up. That
#:    is strictly worse news than a cache hit, and reporting the box as
#:    ``cached`` would claim a complete, reusable result when a piece of it is
#:    missing. Note that a failure INSIDE the box already settles it as
#:    ``error`` below and stops the done-count from ever filling, so the case
#:    that reaches here is the one where the failing node was OUTSIDE: the
#:    whole box was passed over, and ``skipped`` says so.
#:  - ``cached`` (0) is what is left: every single internal was a cache hit,
#:    so nothing executed. This is the case #260 is about.
_CONTAINER_STATUS_RANK: dict[str, int] = {
    "cached": 0,
    "skipped": 1,
    "completed": 2,
}
#: Rank -> the status the container emits. Inverse of the map above.
_CONTAINER_RANK_STATUS: dict[int, str] = {
    rank: status for status, rank in _CONTAINER_STATUS_RANK.items()
}


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
    subgraphs: Any = None,
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
        subgraphs: Graph-local subgraph definitions (core#137). Every
            ``subgraph:<id>`` instance node is inlined before anything runs,
            and its internals roll up to the instance in every status event.
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
        subgraphs=subgraphs,
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

    # Container aggregation: emit "running" once at start, and one terminal
    # status only when all internal nodes finish -- which of
    # completed/skipped/cached depends on what those internals actually did
    # (see _CONTAINER_STATUS_RANK).
    #
    # `container_of` is resolved ONCE, transitively, and only over ids that are
    # actually in the executable node list. Both halves matter:
    #  - transitively, because `internal_to_preset` is a chain under nesting and
    #    a single lookup answers `blk/nest` -- an id the canvas never had, so
    #    the box the user is watching would sit at `idle` for the whole run
    #    while events go to something unresolvable;
    #  - over executable nodes only, because those intermediate ids ARE keys in
    #    the map. Counting them would inflate `container_total` past the number
    #    of nodes that can ever report, and the `>=` gate below would never
    #    fire -- no "completed" for the container, ever.
    container_of: dict[str, str] = {}
    for _node in expanded_nodes:
        _container = outermost_container(_node["id"], internal_to_preset)
        if _container is not None:
            container_of[_node["id"]] = _container
    # container_total[cid] = internal nodes belonging to that container
    # container_done[cid] = internal nodes that have completed/cached/skipped
    # container_started[cid] = True once "running" has been emitted for it
    # container_outcome[cid] = the strongest terminal status seen so far,
    #   as a rank (see _CONTAINER_STATUS_RANK) -- this is what the container
    #   itself reports once every internal node has finished (#260).
    container_total: dict[str, int] = defaultdict(int)
    for _container in container_of.values():
        container_total[_container] += 1
    container_done: dict[str, int] = defaultdict(int)
    container_started: set[str] = set()
    container_outcome: dict[str, int] = {}

    async def _emit_preset_aware(
        node_id: str,
        status: str,
        data: dict[str, Any] | None,
    ) -> None:
        """Emit status to on_progress, aggregating nodes inside a container.

        A container is a preset node or a subgraph instance -- whatever box
        the canvas draws in place of the nodes that expansion produced. Its
        internals (in ``container_of``) roll up into one status:
        - First running/cached -> emit container 'running'
        - Every completed/cached/skipped increments done count; on the last
          one the container emits the WINNING status by
          ``_CONTAINER_STATUS_RANK`` -- 'completed' if anything actually
          ran, 'skipped' if nothing ran and something was passed over,
          'cached' when every internal was a cache hit (#260). Before that
          the box said 'completed' unconditionally, so a preset that did
          nothing at all was indistinguishable from one that trained
        - 'error' emits immediately (the container failed)
        - 'interrupted' likewise: a container whose training node stopped
          early did not complete, so it must never roll up to 'completed'
        - 'progress' passes through as-is with the container ID (so live
          charts still work)
        Nodes that are inside no container pass through unchanged.
        """
        if on_progress is None:
            return
        container_id = container_of.get(node_id)
        if container_id is None:
            # Regular node — pass through
            await _maybe_await(on_progress(node_id, status, data))
            return

        # Node inside a container — aggregate
        if status == "progress":
            # Progress events (e.g. training epochs) should be visible live
            await _maybe_await(on_progress(container_id, "progress", data))
            return

        if status in ("error", "interrupted"):
            # Any internal failure or early stop settles the whole container
            await _maybe_await(on_progress(container_id, status, data))
            return

        if status == "running":
            if container_id not in container_started:
                container_started.add(container_id)
                await _maybe_await(on_progress(container_id, "running", None))
            return

        if status in _CONTAINER_STATUS_RANK:
            container_done[container_id] += 1
            # Remember the strongest thing that happened inside this box, so
            # the terminal status below can tell "it trained" from "every
            # node was a cache hit" (#260).
            rank = _CONTAINER_STATUS_RANK[status]
            if rank > container_outcome.get(container_id, -1):
                container_outcome[container_id] = rank
            # Make sure "running" was emitted at least once. Kept even for a
            # fully-cached box: whether the box will settle 'cached' is not
            # knowable until its LAST internal reports, so the alternative is
            # to withhold every status until then and leave the node at
            # 'idle' while its internals work.
            if container_id not in container_started:
                container_started.add(container_id)
                await _maybe_await(on_progress(container_id, "running", None))
            if container_done[container_id] >= container_total[container_id]:
                terminal = _CONTAINER_RANK_STATUS[container_outcome[container_id]]
                await _maybe_await(on_progress(container_id, terminal, None))

    # A SEEDED run is a SERIAL run (#134). Per-node seeding below sets the
    # PROCESS-GLOBAL RNGs, and two nodes running concurrently would reseed
    # each other mid-execute -- the same seed would then give a different
    # answer on every scheduling, which is the precise opposite of what a
    # seed is for. Reproducibility beats the throughput of a graph whose
    # independent branches could have overlapped; an unseeded run is
    # unaffected and keeps its parallelism.
    max_workers = context.max_workers if context else 4
    if context is not None and context.seed is not None:
        max_workers = 1
    semaphore = asyncio.Semaphore(max_workers)

    # Deterministic kernels are a process-wide torch setting, not a per-call
    # argument, so they are applied once for the whole run and — since
    # #188's review — TAKEN BACK when it ends. The scope is entered
    # unconditionally, not only when determinism was requested: a
    # ``TrainingLoop`` whose own ``deterministic`` param is on would
    # otherwise latch the setting for every later run in the server, so
    # merely opening someone else's saved graph would change how the next
    # one computes.
    wants_determinism = context is not None and context.deterministic

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
        """Report the container's id, never its internals — as progress does.

        Same transitive resolution as `_emit_preset_aware`: a nested node must
        surface as the box on the canvas, not as the intermediate id that only
        existed between two expansion passes.
        """
        if node_id is None:
            return None
        return container_of.get(node_id, node_id)

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

    def _handle_oom(exc: Exception, node_id: str, node_type: str) -> Exception:
        """Recover from an accelerator OOM and rewrite it as a NodeOOMError.

        Order matters. The allocator digest is read FIRST, so it describes
        the failure rather than the cleanup that follows it. Then the two
        things actually worth doing: drop whatever this node had cached
        (those tensors are on the device that just ran out, and they are by
        construction the shape of thing that failed to allocate), and hand
        the caching allocator's free blocks back.

        Neither recovers the run -- it is over -- and that is the point.
        They recover the PROCESS, so the next run the user starts after
        halving their batch size finds the card in the state they expect
        rather than inheriting a fragmented reserve. What cannot be freed
        here is whatever the traceback still holds a reference to; that
        goes when the exception does.
        """
        device = context.device if context is not None else ""
        if not device:
            # A bare ``execute_graph`` (an exported script, a test) has no
            # context and therefore no run device. Naming the current CUDA
            # device keeps the digest and the cleanup pointed at the same
            # place, instead of one of them silently going quiet.
            device = current_cuda_device()
        digest = memory_digest(device)
        dropped = 0
        if cache is not None:
            dropped = cache.clear_node(node_id)
        release_cached_memory(device)
        logger.warning(
            "Node %s (%s) ran out of memory on %s; dropped %d cached "
            "entr%s for it and released the allocator's free blocks.%s",
            node_id, node_type, device or "the current device", dropped,
            "y" if dropped == 1 else "ies",
            f" {digest}" if digest else "",
        )
        return NodeOOMError.from_exception(
            exc, node_id=node_id, node_type=node_type, device=device,
            digest=digest,
        )

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
        # Nodes opt out via cacheable=False when their recorded outputs stop
        # describing what they did -- drifting weights, a live model or
        # optimizer handle something downstream mutates, or a side effect
        # no output carries. See ``BaseNode.cacheable`` for the four cases.
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
            # #144/#145: a node whose output depends on external state a
            # path param only NAMES (not contains) folds a cheap descriptor
            # of that state in here, so a change on disk changes the key.
            # None for every node that does not override the hook.
            fingerprint = node_cls.cache_fingerprint(params)
            cache_key = cache.compute_key(
                node_type, params, upstream_keys,
                device=context.device if context is not None else "cpu",
                fingerprint=fingerprint,
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
        #: Captured INSIDE the handler. ``traceback.format_exc()`` read
        #: after the ``except`` block has exited returns the string
        #: "NoneType: None", because CPython restores the exception state
        #: on the way out -- so the DEBUG traceback below used to be that
        #: placeholder for every failure, including the out-of-memory one
        #: this release goes out of its way to make legible.
        last_traceback: str | None = None

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
                    #
                    # On a PER-NODE VIEW of the context, not the shared one
                    # (#253). ``current_node_id`` used to be assigned here
                    # and read much later, inside the worker thread, with an
                    # ``await`` in between -- so on an unseeded run, where
                    # ``max_workers`` is 4, the sibling that acquired the
                    # semaphore next overwrote the field before this node's
                    # thread ever read it, and `StatefulModuleMixin` filed
                    # the node's weights under a DIFFERENT node's id. It
                    # never showed up because a seeded run is a serial run
                    # (see ``max_workers`` above) and the stateful nodes that
                    # existed were all mid-chain, one per level.
                    #
                    # ``copy.copy`` and not ``replace``/``deepcopy``: every
                    # collaborator on the context -- the stop event, the
                    # outbox, ``node_state_store``, ``grad_targets`` -- must
                    # stay the SAME object, so cancellation, logging and
                    # gradient capture are unaffected. Only the scalar field
                    # differs per node.
                    node_context = context
                    if context is not None:
                        node_context = copy.copy(context)
                        node_context.current_node_id = node_id

                    # Per-node seeding (#134). Every node starts from a seed
                    # derived from (run seed, node id), so its weight init,
                    # dropout mask and shuffle order depend on the run seed
                    # and its own identity -- never on what the rest of the
                    # graph drew first. Re-applied on every RETRY attempt
                    # too, so a retried node re-runs from the same state
                    # rather than continuing the failed attempt's stream.
                    if context is not None and context.seed is not None:
                        seed_rngs(context.derive_seed(node_id))

                    fn = functools.partial(
                        invoke_node,
                        instance,
                        inputs,
                        params,
                        progress_callback=_progress_bridge,
                        context=node_context,
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
                    # ``node_id`` is carried so an out-of-memory failure can
                    # drop exactly this node's cached results (#135) rather
                    # than the whole cache or nothing at all.
                    cache.put(node_cache_keys[node_id], result, node_id)
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
                last_traceback = traceback.format_exc()
                if is_out_of_memory(e):
                    last_error = _handle_oom(e, node_id, node_type)
                    # Never retried. The same allocation on the same device
                    # gets the same answer, and the only thing a retry adds
                    # is another few seconds before the user sees the
                    # message telling them what to change. Auto-shrinking
                    # the batch instead was considered and rejected: it
                    # would change the numbers a run produces without
                    # saying so, so the same graph would mean two different
                    # things depending on how much VRAM happened to be free.
                    break
                last_error = e
                if attempt < attempts - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))  # backoff

        # All attempts failed
        assert last_error is not None
        # `str(exc)` never contains the exception's class name -- str(KeyError('tensor'))
        # is "'tensor'". The UI's beginner-friendly error mapping used to scan for a
        # "KeyError:" prefix that this payload could not produce, so every rule in it
        # was unreachable. Send the type as its own field instead of asking the client
        # to recover it from the message shape.
        error_detail: dict[str, str] = {
            "error": str(last_error),
            "error_type": type(last_error).__name__,
        }
        if settings.DEBUG and last_traceback:
            error_detail["traceback"] = last_traceback
        if error_mode == "fail_fast":
            await _emit_preset_aware(node_id, "error", error_detail)
            raise last_error
        else:
            # continue or retry-exhausted
            node_errors[node_id] = str(last_error)
            await _emit_preset_aware(node_id, "error", error_detail)

    # Entered by hand rather than with a ``with`` block so the whole body
    # below keeps its indentation; the paired exit lives in the ``finally``
    # that already owns this function's teardown.
    #
    # Hand-entering means the pairing is not the language's job any more,
    # so every statement between the two has to be looked at (#190). The
    # cost of getting it wrong went UP when the scope became refcounted:
    # a skipped exit used to cost one run its restore, whereas now the
    # depth does not return to 0, so ``use_deterministic_algorithms`` stays
    # at the failed run's setting and every LATER scope's baseline capture
    # is suppressed as well (see ``seeding.deterministic_scope``).
    #
    # How far that actually reaches, measured rather than assumed, because
    # #190 overstates it: ``deterministic_scope`` is a generator-based
    # context manager, so once the traceback holding this frame is released
    # CPython finalises the suspended generator and runs the very
    # ``finally`` that was skipped. The window is therefore however long
    # something holds the traceback -- a stored exception, an unretrieved
    # task -- not the life of the process. Still worth closing: it is a
    # process-global setting whose correctness would otherwise rest on an
    # interpreter detail nothing here states, and which no implementation
    # without prompt refcounting provides.
    determinism = deterministic_scope(wants_determinism)
    determinism.__enter__()

    # Gap 1: outside the try below, and ``create_task`` raises RuntimeError
    # on a loop that is closing. ``BaseException`` because a leaked depth
    # outlives whatever is being torn down.
    try:
        pump_task = asyncio.create_task(_pump(),
                                        name=f"outbox-pump:{run_id or '-'}")
    except BaseException:
        determinism.__exit__(None, None, None)
        raise

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
            # Gap 2: ``await pump_task`` re-raises anything ``_pump`` died
            # of that is not CancelledError, and the exit used to be its
            # PEER rather than its finally — so the comment below ("nothing
            # above can skip it") was true of the outer chain and false of
            # the three lines it sat under. ``_pump`` catches Exception
            # around ``_deliver`` but not around its own ``outbox.wait()``,
            # and never catches BaseException at all, so the path is narrow
            # rather than closed.
            try:
                pump_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pump_task
            finally:
                # Hand determinism back to whatever the process had before
                # this run. Last, and now genuinely unskippable.
                determinism.__exit__(None, None, None)


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
