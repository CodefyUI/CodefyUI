"""SECRET-typed parameter handling for saved graphs.

A node may declare a param as ``ParamType.SECRET`` (e.g. LLMChat's
``openai_api_key``). Such values live only in canvas / runtime state; they
must never be written to a saved graph file, a published snapshot, or an
exported JSON. This module is the single source of truth for:

- which params of a given node type are secret (registry lookup),
- scrubbing secret values out of a graph's nodes before persistence
  (defense-in-depth for the save endpoint), and
- detecting a leftover non-empty secret in an already-on-disk graph
  (publish pre-flight for hand-edited files).

A graph carries nodes in three places, and every function here that walks
one has to walk all three or the guard is only as strong as the shallowest
walk: top-level ``nodes``, portable ``presets[].nodes``, and -- since
core#137 -- ``subgraphs[].nodes``. A node inside a collapsed block is an
ordinary node with ordinary params; nothing about sitting in a block makes
its API key less of a secret.

Every function operates on the serialized node shape ``{"id", "type",
"data": {"params": {...}}}`` used by GraphData / the saved JSON, so the
save path (pydantic dump -> dict) and the publish path (json.loads dict)
share one implementation. A node type the registry does not know (a plugin
or custom node that is not loaded, a preset nobody has) carries no known
secret params, and the save, export and publish walks leave it untouched.

The run path is stricter (#537). The server cannot name the secret params
of a type it does not know, and a run's row is written before the engine
refuses the type, so the two submit lanes call ``split_graph_secrets`` with
``unknown_types_as_secret=True``: every value of such a node is kept out of
the row the way a SECRET value is. The startup sweep of finished rows does
not ask for that, and putting values back never consults the registry at
all -- ``split_graph_secrets`` and ``restore_graph_secrets`` say why.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Iterator, Mapping

from .node_base import ParamType
from .node_registry import registry

#: One extracted secret's address inside a graph -- see
#: :func:`iter_secret_slots`. Positional rather than id-based on purpose: a
#: hand-edited graph may carry duplicate node ids, and two slots sharing a
#: vault key would restore each other's value.
SecretAddress = tuple[Any, ...]
#: ``{address: original value}``. Held in memory only; never serialized.
SecretVault = dict[SecretAddress, Any]


def secret_param_names(node_type: str) -> set[str]:
    """Names of the SECRET-typed params for ``node_type``.

    Empty set when the type is unknown to the registry (notes, presets,
    unloaded plugin nodes) or declares no secret params.
    """
    if not node_type:
        return set()
    node_cls = registry.get(node_type)
    if node_cls is None:
        return set()
    return {
        p.name
        for p in node_cls.define_params()
        if p.param_type == ParamType.SECRET
    }


def _type_of(node: Mapping[str, Any]) -> str:
    """A node's type as a string, whatever the file actually contains.

    ``node.get("type", "")`` is NOT enough: a key present with a null value
    returns ``None``, and ``_preset_secret_param_map`` then calls
    ``.startswith`` on it and raises ``AttributeError`` (core#200 item 6). The
    walkers here are the only readers of a graph that never passed through
    pydantic -- the publish gate reads a file straight off disk, while
    ``/save`` and ``/export`` are protected by ``NodeData.type: str`` -- so
    this is where the coercion belongs.
    """
    return str(node.get("type") or "")


def _params_of(node: dict[str, Any]) -> dict[str, Any] | None:
    data = node.get("data")
    if not isinstance(data, dict):
        return None
    params = data.get("params")
    return params if isinstance(params, dict) else None


def _internal_params_of(node: dict[str, Any]) -> dict[str, Any] | None:
    """The ``data.internalParams`` map of a preset node, keyed by internal
    node id -> {param name: value}. None when absent or malformed."""
    data = node.get("data")
    if not isinstance(data, dict):
        return None
    internal = data.get("internalParams")
    return internal if isinstance(internal, dict) else None


def _preset_definitions(
    node_type: str,
    preset_fallback: Mapping[str, Any] | None,
) -> list[Any]:
    """The definitions a ``preset:<name>`` type resolves to, installed first.

    The preset registry's and the graph's own portable one, whichever exist.
    Empty when the type is not a preset or neither knows the name -- the
    case the engine reports as ``Unknown preset``.
    """
    if not node_type.startswith("preset:"):
        return []
    # Lazy import: preset_registry pulls in schemas/node_registry; importing
    # it at module load would risk a cycle (graph_engine uses the same
    # lazy-import pattern for exactly this reason).
    from .preset_registry import preset_registry

    preset_name = node_type[len("preset:"):]
    registered = preset_registry.get(preset_name)
    fallback = (preset_fallback or {}).get(preset_name)
    return [p for p in (registered, fallback) if p is not None]


def _preset_secret_param_map(
    node_type: str,
    preset_fallback: Mapping[str, Any] | None = None,
) -> dict[str, set[str]]:
    """For a ``preset:<name>`` node type, map each internal node id to the
    set of its SECRET-typed param names.

    A preset node embeds per-inner-node overrides in ``data.internalParams``;
    those inner nodes are real registry types (e.g. an inner ``LLMChat``),
    so a hand-edited graph could bake an API key into
    ``internalParams["<inner_id>"]["openai_api_key"]`` — invisible to the
    plain ``data.params`` scrub. We resolve the preset through the preset
    registry (to learn each inner node's id + type) and the node registry
    (to learn which of its params are secret). Empty when the type is not a
    preset, the preset is unknown, or no inner node declares a secret.
    """
    candidates = _preset_definitions(node_type, preset_fallback)
    if not candidates:
        return {}
    result: dict[str, set[str]] = {}
    # For execution, an installed preset intentionally wins over a portable
    # same-name fallback. For scrubbing, take the union: the downloaded graph
    # may later run on a machine where only the embedded definition exists.
    for preset in candidates:
        internal_nodes = (
            preset.get("nodes", []) if isinstance(preset, dict) else preset.nodes
        )
        for internal in internal_nodes:
            internal_type = (
                internal.get("type", "")
                if isinstance(internal, dict)
                else internal.type
            )
            internal_id = (
                internal.get("id", "")
                if isinstance(internal, dict)
                else internal.id
            )
            names = secret_param_names(internal_type)
            if names:
                result.setdefault(internal_id, set()).update(names)
    return result


def _is_nonempty_secret(value: Any) -> bool:
    """A secret value counts as "present" for lint/scrub purposes when it is
    anything other than the empty string or ``None`` (the scrubbed form)."""
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    # A non-string truthy value (should not happen for a text key, but a
    # hand-edited file could contain one) is still a leaked secret.
    return bool(value)


def scrub_graph_secrets(
    nodes: Iterable[dict[str, Any]],
    *,
    preset_fallback: Mapping[str, Any] | None = None,
) -> int:
    """Blank every SECRET-typed param value in ``nodes`` (in place).

    Returns the number of values changed. Only params that are both
    declared secret AND currently non-empty are rewritten to ``""`` — a
    node with no secret params, or an unknown type, is untouched.
    """
    changed = 0
    for node in nodes:
        node_type = _type_of(node)
        # Regular node: blank its own declared SECRET params.
        names = secret_param_names(node_type)
        if names:
            params = _params_of(node)
            if params is not None:
                for name in names:
                    if name in params and _is_nonempty_secret(params[name]):
                        params[name] = ""
                        changed += 1
        # Preset node: blank SECRET params embedded per inner node in
        # data.internalParams (secret_param_names is empty for a preset:*
        # type, so the block above no-ops for it — no double counting).
        preset_secrets = _preset_secret_param_map(node_type, preset_fallback)
        if preset_secrets:
            internal_params = _internal_params_of(node)
            if internal_params is not None:
                for internal_id, secret_names in preset_secrets.items():
                    inner = internal_params.get(internal_id)
                    if not isinstance(inner, dict):
                        continue
                    for name in secret_names:
                        if name in inner and _is_nonempty_secret(inner[name]):
                            inner[name] = ""
                            changed += 1
    return changed


def scrub_preset_definition_secrets(
    presets: Iterable[dict[str, Any]],
) -> int:
    """Blank SECRET defaults stored in portable preset definitions.

    ``scrub_graph_secrets`` handles instantiated graph nodes and a preset
    node's ``internalParams`` overrides. A portable graph can additionally
    carry defaults in ``presets[].nodes[].params``; those definitions are
    serialized separately and therefore need this matching pass.
    """

    changed = 0
    for preset in presets:
        internal_nodes = preset.get("nodes")
        if not isinstance(internal_nodes, list):
            continue
        for internal in internal_nodes:
            if not isinstance(internal, dict):
                continue
            params = internal.get("params")
            if not isinstance(params, dict):
                continue
            for name in secret_param_names(str(internal.get("type", ""))):
                if name in params and _is_nonempty_secret(params[name]):
                    params[name] = ""
                    changed += 1
    return changed


def scrub_subgraph_definition_secrets(
    subgraphs: Iterable[Any],
    *,
    preset_fallback: Mapping[str, Any] | None = None,
) -> int:
    """Blank every SECRET-typed value inside subgraph definitions (in place).

    A definition's ``nodes`` are ORDINARY serialized nodes -- same shape,
    same params, same ``internalParams`` for a preset among them -- so this
    is just :func:`scrub_graph_secrets` applied per definition. It exists as
    one named function because the save route and the export route both need
    it: two hand-written loops is how one of them ended up without the
    ``preset_fallback`` the other passed.

    Nesting needs no recursion: ``subgraphs`` is a FLAT list and a block
    inside a block is a ``subgraph:<id>`` reference into it, so every node at
    every depth is reached by iterating the list once.

    Returns the number of values changed.
    """
    changed = 0
    for definition in subgraphs or []:
        if not isinstance(definition, dict):
            continue
        inner_nodes = definition.get("nodes")
        if not isinstance(inner_nodes, list):
            continue
        changed += scrub_graph_secrets(
            [n for n in inner_nodes if isinstance(n, dict)],
            preset_fallback=preset_fallback,
        )
    return changed


def _is_unknown_type(
    node_type: str,
    preset_fallback: Mapping[str, Any] | None,
) -> bool:
    """Would the engine refuse ``node_type`` as a type it does not know?

    The rules of the type check in ``graph_engine.validate_graph``, so the
    run path withholds what the run itself could not place and nothing
    else. A note is an annotation the engine drops, not a type. A
    ``subgraph:<id>`` instance is a reference; the nodes of its definition
    are walked in their own right. A ``preset:<name>`` is known when the
    preset registry or the graph's own ``presets[]`` has it. Any other type
    is known when ``registry.get`` resolves it, a bare plugin name included.
    An empty or missing type is unknown.
    """
    # Lazy import: keeping graph_engine out of this module's import graph.
    from .graph_engine import NOTE_NODE_TYPE, subgraph_id_of

    if node_type == NOTE_NODE_TYPE or subgraph_id_of(node_type) is not None:
        return False
    if node_type.startswith("preset:"):
        return not _preset_definitions(node_type, preset_fallback)
    return registry.get(node_type) is None


def _unplaced_inner_ids(
    node_type: str,
    internal_params: Mapping[Any, Any],
    preset_fallback: Mapping[str, Any] | None,
) -> list[Any]:
    """The ``internalParams`` entries of a KNOWN preset the server cannot place.

    An entry is placed when its id names an inner node and every type that
    id has is known, across the installed and the portable definition
    together -- the union ``_preset_secret_param_map`` takes. A placed entry
    keeps the SECRET rule. An unplaced one is withheld whole: nothing says
    which of its values is secret. Empty for a type that is not a known
    preset.
    """
    definitions = _preset_definitions(node_type, preset_fallback)
    if not definitions:
        return []
    inner_types: dict[Any, set[str]] = {}
    for preset in definitions:
        internal_nodes = (
            preset.get("nodes", []) if isinstance(preset, dict) else preset.nodes
        )
        for internal in internal_nodes:
            internal_type = (
                internal.get("type", "")
                if isinstance(internal, dict)
                else internal.type
            )
            internal_id = (
                internal.get("id", "")
                if isinstance(internal, dict)
                else internal.id
            )
            inner_types.setdefault(internal_id, set()).add(
                str(internal_type or ""))
    return [
        internal_id for internal_id in internal_params
        if internal_id not in inner_types
        or any(_is_unknown_type(t, preset_fallback)
               for t in inner_types[internal_id])
    ]


def _iter_param_slots(
    params: Any, prefix: tuple[Any, ...],
) -> Iterator[tuple[SecretAddress, dict[str, Any], str]]:
    """Every key of ONE params dict, as slots addressed ``(*prefix, key)``.

    Unfiltered: this is how a value is found when the registry cannot say
    whether it is secret, or must not be asked.
    """
    if isinstance(params, dict):
        for name in params:
            yield (*prefix, name), params, name


def _iter_node_value_slots(
    node: dict[str, Any], prefix: tuple[Any, ...],
) -> Iterator[tuple[SecretAddress, dict[str, Any], str]]:
    """EVERY param slot of one serialized graph node, whatever its type.

    ``data.params`` and each entry of ``data.internalParams``, at the
    addresses :func:`_iter_node_secret_slots` gives the same slots.
    """
    yield from _iter_param_slots(_params_of(node), (*prefix, "params"))
    internal_params = _internal_params_of(node)
    if internal_params is not None:
        for internal_id, inner in internal_params.items():
            yield from _iter_param_slots(
                inner, (*prefix, "internalParams", internal_id))


def _iter_node_secret_slots(
    node: dict[str, Any],
    prefix: tuple[Any, ...],
    preset_fallback: Mapping[str, Any] | None,
    unknown_types_as_secret: bool,
) -> Iterator[tuple[SecretAddress, dict[str, Any], str]]:
    """Yield every SECRET-typed slot of ONE serialized node.

    A "slot" is ``(address, container, key)`` where ``container[key]`` is the
    value — a writable handle, so a caller can blank or replace the value
    where it sits.

    Only keys that are PRESENT are yielded. That is what makes the two
    directions symmetric: extraction blanks a slot to ``""`` rather than
    deleting it, so the same slot is still there to be found on the way back.

    With ``unknown_types_as_secret``, every slot of a node of unknown type
    counts as secret, and so does every slot of a known preset's
    ``internalParams`` entry the server cannot place.
    """
    node_type = _type_of(node)
    if unknown_types_as_secret and _is_unknown_type(node_type, preset_fallback):
        yield from _iter_node_value_slots(node, prefix)
        return
    names = secret_param_names(node_type)
    if names:
        params = _params_of(node)
        if params is not None:
            for name in sorted(names):
                if name in params:
                    yield (*prefix, "params", name), params, name
    internal_params = _internal_params_of(node)
    if internal_params is None:
        return
    unplaced = (
        _unplaced_inner_ids(node_type, internal_params, preset_fallback)
        if unknown_types_as_secret else []
    )
    # Preset instance: secrets can also sit per inner node in internalParams.
    preset_secrets = _preset_secret_param_map(node_type, preset_fallback)
    for internal_id in sorted(preset_secrets):
        if internal_id in unplaced:
            continue  # withheld whole, below
        inner = internal_params.get(internal_id)
        if not isinstance(inner, dict):
            continue
        for name in sorted(preset_secrets[internal_id]):
            if name in inner:
                yield ((*prefix, "internalParams", internal_id, name),
                       inner, name)
    for internal_id in unplaced:
        yield from _iter_param_slots(
            internal_params[internal_id],
            (*prefix, "internalParams", internal_id))


def _iter_definition_secret_slots(
    node: dict[str, Any],
    prefix: tuple[Any, ...],
    preset_fallback: Mapping[str, Any] | None,
    unknown_types_as_secret: bool,
) -> Iterator[tuple[SecretAddress, dict[str, Any], str]]:
    """The same, for one node of a portable preset DEFINITION.

    A flatter shape: ``params`` directly on the node, not under ``data``,
    and no ``internalParams`` -- the engine reads a definition node as
    ``{id, type, params}`` and nothing else.
    """
    params = node.get("params")
    if not isinstance(params, dict):
        return
    node_type = _type_of(node)
    if unknown_types_as_secret and _is_unknown_type(node_type, preset_fallback):
        yield from _iter_param_slots(params, (*prefix, "params"))
        return
    for name in sorted(secret_param_names(node_type)):
        if name in params:
            yield (*prefix, "params", name), params, name


def _iter_graph_nodes(
    graph: Mapping[str, Any],
) -> Iterator[tuple[tuple[Any, ...], dict[str, Any], bool]]:
    """Every node a graph carries, with the positional prefix of its address.

    Yields ``(prefix, node, is_definition)``; ``is_definition`` marks a node
    of a portable preset DEFINITION. The SECRET walk and the restore walk
    both go through here, so the address a value is vaulted under and the
    address it is put back from are built by the same code.
    """
    nodes = graph.get("nodes")
    if isinstance(nodes, list):
        for index, node in enumerate(nodes):
            if isinstance(node, dict):
                yield ("nodes", index), node, False

    # Subgraph definitions hold ORDINARY nodes. Nesting needs no recursion:
    # the list is flat and a block inside a block is a `subgraph:<id>`
    # reference into this same list.
    subgraphs = graph.get("subgraphs")
    if isinstance(subgraphs, list):
        for outer, definition in enumerate(subgraphs):
            if not isinstance(definition, dict):
                continue
            inner_nodes = definition.get("nodes")
            if not isinstance(inner_nodes, list):
                continue
            for index, inner in enumerate(inner_nodes):
                if isinstance(inner, dict):
                    yield ("subgraphs", outer, "nodes", index), inner, False

    # Portable preset DEFINITIONS carry defaults in a flatter shape:
    # `params` directly on the node, not under `data`.
    presets = graph.get("presets")
    if isinstance(presets, list):
        for outer, preset in enumerate(presets):
            if not isinstance(preset, dict):
                continue
            inner_nodes = preset.get("nodes")
            if not isinstance(inner_nodes, list):
                continue
            for index, inner in enumerate(inner_nodes):
                if isinstance(inner, dict):
                    yield ("presets", outer, "nodes", index), inner, True


def iter_secret_slots(
    graph: Mapping[str, Any],
    *,
    preset_fallback: Mapping[str, Any] | None = None,
    unknown_types_as_secret: bool = False,
) -> Iterator[tuple[SecretAddress, dict[str, Any], str]]:
    """Yield every SECRET-typed slot of a WHOLE graph, writable handles and all.

    Where the ``scrub_*`` functions above each take one piece of a graph and
    leave the caller to combine them (four calls, in the right order, with
    the right ``preset_fallback`` — see ``routes_graph``), this takes the
    whole ``{"nodes", "edges", "presets", "subgraphs"}`` envelope and covers
    all of it. The run path needs exactly that: one call that cannot be
    half-applied.

    ``preset_fallback`` defaults to one built from the graph's OWN
    ``presets[]``, so the caller cannot forget it — without it a preset
    node's ``internalParams`` slots are not identifiable as secret at all.

    Addresses are POSITIONAL (``("nodes", 3, "params", "openai_api_key")``)
    rather than id-based. Ids would read better, but a hand-edited graph may
    carry duplicates, and two slots colliding on one vault key would restore
    each other's value. Indices are unique by construction and survive the
    ``json.dumps``/``loads`` round trip the snapshot column makes, which is
    the only round trip an address has to outlive.

    ``unknown_types_as_secret`` adds every slot whose value the server cannot
    classify at all; :func:`split_graph_secrets` says who asks for it and why.
    """
    if not isinstance(graph, Mapping):
        return
    if preset_fallback is None:
        # Lazy import, for the same reason preset_registry is imported lazily
        # above: keeping graph_engine out of this module's import graph.
        from .graph_engine import build_preset_fallback
        raw_presets = graph.get("presets")
        preset_fallback = build_preset_fallback(
            raw_presets if isinstance(raw_presets, list) else [])

    for prefix, node, is_definition in _iter_graph_nodes(graph):
        walk = (_iter_definition_secret_slots if is_definition
                else _iter_node_secret_slots)
        yield from walk(node, prefix, preset_fallback, unknown_types_as_secret)


def _iter_all_param_slots(
    graph: Mapping[str, Any],
) -> Iterator[tuple[SecretAddress, dict[str, Any], str]]:
    """EVERY param slot of a whole graph: the walk restore goes by.

    Everything :func:`iter_secret_slots` yields in either mode is in here,
    at the same address, and none of it is found by asking a registry.
    """
    if not isinstance(graph, Mapping):
        return
    for prefix, node, is_definition in _iter_graph_nodes(graph):
        if is_definition:
            yield from _iter_param_slots(node.get("params"), (*prefix, "params"))
        else:
            yield from _iter_node_value_slots(node, prefix)


def split_graph_secrets(
    graph: Mapping[str, Any],
    *,
    unknown_types_as_secret: bool = False,
) -> tuple[Mapping[str, Any], SecretVault]:
    """Separate a graph from its secrets: ``(scrubbed graph, vault)``.

    The input is NEVER modified — the caller usually still needs the real
    values to execute with. When the graph carries no secret at all the same
    object is returned unchanged (nothing to protect, and a deep copy of a
    large graph on every submit is not free); otherwise the returned graph is
    a deep copy with every secret blanked to ``""``.

    ``unknown_types_as_secret`` is what the two submit lanes pass (#537). It
    counts as secret every present value of a node whose type the server
    does not know (see :func:`_is_unknown_type` for what that means), and
    of a known preset's ``internalParams`` entry that no known inner node
    accounts for. The server cannot name the secret params of a type it
    does not know, and the run row is written before the engine refuses the
    type, so without this a key typed into such a node would be stored as
    sent. Withholding every value costs the run nothing: the queued lane
    gets them back from the vault, and the interactive lane runs the live
    graph.

    The startup sweep of finished rows does NOT pass it. "Unknown" is a fact
    about one boot: a plugin that fails to load once would lose its settings
    from every past run, for good, since the sweep keeps no vault.

    The vault is plain in-memory Python, deliberately: it is the half that
    must NOT be written anywhere. See ``RunService`` for the lifetime
    argument that makes that safe.
    """
    if not any(_is_nonempty_secret(container[key])
               for _address, container, key in iter_secret_slots(
                   graph, unknown_types_as_secret=unknown_types_as_secret)):
        return graph, {}
    scrubbed = copy.deepcopy(dict(graph))
    vault: SecretVault = {}
    for address, container, key in iter_secret_slots(
            scrubbed, unknown_types_as_secret=unknown_types_as_secret):
        value = container[key]
        if _is_nonempty_secret(value):
            vault[address] = value
            container[key] = ""
    return scrubbed, vault


def restore_graph_secrets(
    graph: Mapping[str, Any], vault: SecretVault | None,
) -> int:
    """Put a vault's values back into ``graph`` (in place). Returns how many.

    The inverse of :func:`split_graph_secrets`. Its addresses come from the
    same traversal, so an address is found where it was taken from, but the
    walk is NOT the filtered one that chose what to take: it visits every
    param slot and puts back whatever the vault holds for it, without asking
    the registry anything. The vault already records what was taken. A
    queued run can wait for hours, and a type that gets registered in that
    time (a custom node re-enabled) would make a filtered walk see only its
    SECRET slots: the key would come back, every other value of that node
    would stay ``""``, and the run would execute with them, silently.

    A missing address is silently skipped rather than raised on, and that is
    the safe direction: the slot keeps the ``""`` it was scrubbed to, and the
    node fails with whatever "no API key" error it already raises. The
    opposite failure — a graph that quietly runs with a key it should not
    have, or a crash that strands a run — is worse.
    """
    if not vault:
        return 0
    restored = 0
    for address, container, key in _iter_all_param_slots(graph):
        if address in vault:
            container[key] = vault[address]
            restored += 1
    return restored


def find_secret_violations(
    nodes: Iterable[dict[str, Any]],
    *,
    subgraphs: Iterable[Any] | None = None,
    presets: Iterable[Any] | None = None,
    preset_fallback: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Report every non-empty SECRET-typed param still present in ``nodes``.

    Each entry is ``{"node_id": <id>, "param": <param name>}``. Used by the
    publish pre-flight to reject a hand-edited graph file that a client
    dropped into the graphs dir with a secret baked in.

    ``subgraphs`` (core#137) is the graph's own definition list. A node
    inside a collapsed block is an ORDINARY node with ordinary params, so it
    can hold a key exactly the way a top-level node can -- and a walk that
    stops at the top level lets such a file publish cleanly into an
    immutable, API-exposed snapshot. Inner violations are addressed
    ``<definition id>/<inner id>``, mirroring the ``<container>/<inner>``
    shape expansion flattens ids to, so the message names a slot the user can
    navigate to. Definitions carry no nesting of their own -- a block inside
    a block is a ``subgraph:<id>`` REFERENCE into this same flat list -- so
    walking every entry covers arbitrary depth without recursing.

    ``presets`` is the graph's own portable preset DEFINITIONS (core#200 item
    5). This is the third and last place a file can carry a node, and it was
    the one this walk missed: ``scrub_preset_definition_secrets`` exists to
    clean it on the save path and ``iter_secret_slots`` covers it on the run
    path, so a key baked into a portable preset's own defaults published
    cleanly while the identical key one level up was refused. A definition
    node's params sit directly on the node (``presets[].nodes[].params``),
    not under ``data`` -- a flatter shape than everywhere else. Violations are
    addressed ``preset:<name>/<inner id>``, which is the type string an
    instance of that preset actually wears, so the report names something the
    reader can look up rather than an index into a list.

    ``preset_fallback`` resolves presets defined only in the graph's own
    ``presets[]``; without it a preset node's ``internalParams`` slots cannot
    be identified as secret at all.
    """
    violations: list[dict[str, str]] = []

    def scan(node: dict[str, Any], node_id: str) -> None:
        node_type = _type_of(node)
        # Regular node: report its own declared SECRET params.
        names = secret_param_names(node_type)
        if names:
            params = _params_of(node)
            if params is not None:
                for name in sorted(names):
                    if _is_nonempty_secret(params.get(name)):
                        violations.append({"node_id": node_id, "param": name})
        # Preset node: report secrets baked into data.internalParams. The
        # param is reported as ``<inner_id>.<param>`` so the message names the
        # exact inner slot the client must clear.
        preset_secrets = _preset_secret_param_map(node_type, preset_fallback)
        if preset_secrets:
            internal_params = _internal_params_of(node) or {}
            for internal_id in sorted(preset_secrets):
                inner = internal_params.get(internal_id)
                if not isinstance(inner, dict):
                    continue
                for name in sorted(preset_secrets[internal_id]):
                    if _is_nonempty_secret(inner.get(name)):
                        violations.append({
                            "node_id": node_id,
                            "param": f"{internal_id}.{name}",
                        })

    for node in nodes:
        scan(node, str(node.get("id", "")))

    # Lazy import for the same reason preset_registry is imported lazily
    # above: keeping graph_engine out of this module's import graph.
    from .graph_engine import SUBGRAPH_SEPARATOR

    for definition in subgraphs or []:
        if not isinstance(definition, dict):
            continue
        definition_id = str(definition.get("id", ""))
        inner_nodes = definition.get("nodes")
        if not isinstance(inner_nodes, list):
            continue
        for inner in inner_nodes:
            if not isinstance(inner, dict):
                continue
            inner_id = str(inner.get("id", ""))
            scan(inner, f"{definition_id}{SUBGRAPH_SEPARATOR}{inner_id}")

    # Portable preset definitions. Not `scan`: a definition node keeps its
    # params directly on the node rather than under `data`, and it can never
    # itself be a `preset:` instance, so the internalParams half does not
    # apply. Nesting needs no recursion here either -- a preset whose body
    # uses another preset carries a `preset:<name>` REFERENCE, and that inner
    # definition is its own entry in this same list.
    for preset in presets or []:
        if not isinstance(preset, dict):
            continue
        preset_name = str(preset.get("preset_name") or preset.get("name") or "")
        inner_nodes = preset.get("nodes")
        if not isinstance(inner_nodes, list):
            continue
        for inner in inner_nodes:
            if not isinstance(inner, dict):
                continue
            params = inner.get("params")
            if not isinstance(params, dict):
                continue
            inner_id = str(inner.get("id", ""))
            address = f"preset:{preset_name}{SUBGRAPH_SEPARATOR}{inner_id}"
            for name in sorted(secret_param_names(_type_of(inner))):
                if _is_nonempty_secret(params.get(name)):
                    violations.append({"node_id": address, "param": name})

    return violations
