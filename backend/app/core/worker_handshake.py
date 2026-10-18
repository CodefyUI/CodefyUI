"""What a worker's ``hello`` must satisfy, and what it tells us (#781).

``api/ws_worker`` owns the socket; the decisions are here, pure and
testable without one:

- :func:`raw_refusal`: the version rules read off the raw JSON before the
  frame is validated, so an agent whose ``hello`` this build cannot parse
  (another release changed its shape) still hears ``version mismatch:
  worker 3.1.0, control plane 3.1.2`` rather than a bare protocol error.
- :func:`refusal`: why a first frame cannot register a worker at all. Not
  a ``hello``; another :data:`~worker_protocol.PROTOCOL_VERSION`; another
  CodefyUI **version** (both sides must run the same release, #764, so the
  reason names both); a malformed or reserved name.
- :func:`identity_refusal`: why this credential may not register under the
  name the ``hello`` uses (#780).
- :func:`one_shot_refusal`: why a one-shot agent (#790) may not take a
  name another agent process holds; :func:`same_agent` decides who that
  is, after a restart of this server too (#845).
- :func:`check_catalog`: how the worker's node catalog compares with this
  server's (``node_registry.node_catalog``). Equal hashes: ``match``.
  Different: ``mismatch``, with the node types this server has and the
  worker lacks (``missing``) and the other way round (``extra``); the
  worker is then ``degraded``. A worker that reported no catalog (its
  facts helper failed) is ``unknown``: not degraded, and a run naming it
  is not checked, but ``target: any`` skips it, as it skips a worker that
  LACKS types (``missing`` non-empty). A worker with every type this
  server has and more (``extra`` only) stays eligible for ``any``; each
  run is still checked against it before ``assign``.
- :func:`verified_fingerprint`: the worker's environment document and the
  digest THIS side computes from it.
- :func:`catalog_refusal`: before a run is given to a mismatched worker
  (at submit for a named target, and again just before ``assign``), the
  node types the graph uses (:func:`graph_node_types`: after preset and
  subgraph expansion, plus everything an installed preset a ``Map`` node
  runs contains) must all be ones the worker has.

None of this imports torch or the engine at module level: the agent
imports ``worker_protocol``, never this module, but the control plane
loads it on every connection.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable

from . import worker_protocol as wp

logger = logging.getLogger(__name__)

CATALOG_MATCH = "match"
CATALOG_MISMATCH = "mismatch"
CATALOG_UNKNOWN = "unknown"

#: Node types named in a refusal before the rest are counted.
MAX_TYPES_NAMED = 10


# ── refusals ──────────────────────────────────────────────────────────────


def version_mismatch(worker_version: str, control_version: str) -> str:
    return (f"version mismatch: worker {worker_version}, control plane "
            f"{control_version}; install the same CodefyUI version on both")


def raw_refusal(raw: str | bytes, *,
                control_version: str) -> tuple[str | None, str] | None:
    """``(name, reason)`` when the raw first frame is a ``hello`` of another
    protocol or CodefyUI version, else None. Read before validation, so a
    schema this build cannot parse still gets the version pair. *name* is
    the frame's name when it is a string, else None."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("type") != wp.HELLO:
        return None
    name = data.get("name") if isinstance(data.get("name"), str) else None
    version = data.get("version")
    if isinstance(version, str) and version != control_version:
        return name, version_mismatch(version, control_version)
    protocol = data.get("protocol_version")
    if isinstance(protocol, int) and protocol != wp.PROTOCOL_VERSION:
        return name, protocol_mismatch(protocol)
    return None


def protocol_mismatch(protocol: int) -> str:
    return (f"protocol version {protocol} is not this server's "
            f"{wp.PROTOCOL_VERSION}; update the agent")


def refusal(frame: Any, *, control_version: str) -> str | None:
    """Why a first frame cannot register a worker, or None."""
    if not isinstance(frame, wp.Hello):
        return f"expected hello, got {frame.type}"
    if frame.protocol_version != wp.PROTOCOL_VERSION:
        return protocol_mismatch(frame.protocol_version)
    if frame.version != control_version:
        return version_mismatch(frame.version, control_version)
    if wp.WORKER_NAME_PATTERN.fullmatch(frame.name) is None:
        return (f"worker name {frame.name!r} must be lowercase letters, "
                "digits and '-', at most 63 characters")
    if frame.name in wp.RESERVED_WORKER_NAMES:
        return f"worker name {frame.name!r} is reserved"
    if frame.instance is not None and not frame.one_shot:
        # Several agents on one credential are for batch jobs that each
        # take one run (#845); regular agents keep one name per credential.
        return ("--instance is for one-shot agents (--one-shot) only; a "
                "regular agent connects under its own name")
    return None


def instance_id_hash(instance_id: str | None) -> str | None:
    """How an agent process's ``hello.instance_id`` is recorded (#845):
    its sha256 hex, so the stored form cannot be replayed as the id."""
    if not instance_id:
        return None
    return hashlib.sha256(instance_id.encode("utf-8")).hexdigest()


def identity_refusal(hello: wp.Hello, *, credential: str,
                     enrolled_as: str | None, registry: Any) -> str | None:
    """Why a connection that authenticated with *credential* (naming
    *enrolled_as*, or None) may not register as ``hello.name``, or None.

    A worker token names its worker, so the ``hello`` must use that name.
    And a connection may replace a live one under the same name only if it
    authenticated the same way: the session token cannot take over an
    enrolled worker's name, nor a worker token a live same-host agent's.

    ``hello.name`` may already be an instance's worker id (#845,
    ``<name>.<instance>``, as ``ws_worker`` rewrites it): the credential
    must then name the part before the separator, and the session token
    may not open instances under an enrolled worker's name.
    """
    from .workers import CREDENTIAL_SESSION

    name, instance = wp.instance_parts(hello.name)
    if enrolled_as is not None and name != enrolled_as:
        return (f"this worker token belongs to worker {enrolled_as!r}, not "
                f"{name!r}; start the agent with --name {enrolled_as}, "
                f"or enroll {name!r} with a new token")
    if instance is not None and credential == CREDENTIAL_SESSION:
        base = registry.get(name)
        if base is not None and base.enrolled:
            return (f"worker {name!r} is enrolled; its instances connect "
                    "with its worker token, not the session token")
    # The placeholder of runs adopted after a restart counts (#845): its
    # one-shot run belongs to the agent process that took it.
    existing = getattr(registry, "holder", registry.get)(hello.name)
    if existing is None:
        return None
    if credential == CREDENTIAL_SESSION and existing.enrolled:
        return (f"worker {hello.name!r} is enrolled; it connects with its "
                "worker token, not the session token. Pick another --name")
    if existing.online and existing.credential not in (None, credential):
        return (f"a worker named {hello.name!r} is connected with another "
                "kind of credential; stop it first")
    return one_shot_refusal(hello, existing)


def one_shot_refusal(hello: wp.Hello, existing: Any) -> str | None:
    """Why *hello* may not take over *existing*'s name because one of the
    two is a one-shot agent (#790), or None.

    A regular worker's reconnect replaces the old connection, and its runs
    move over. Between two DIFFERENT agent processes (``instance_id``) that
    would hand one agent's run to another that does not have it, which then
    ends ``interrupted``: two batch jobs started with the same credential.
    So while the name is connected, or still has runs in flight, only the
    same agent (:func:`same_agent`) may take it back.
    """
    if not (hello.one_shot or getattr(existing, "one_shot", False)):
        return None
    if not (existing.online or existing.running):
        return None
    if same_agent(hello, existing):
        return None
    state = "connected" if existing.online else "its run is in flight"
    return (f"worker {hello.name!r} is busy: another agent holds the name "
            f"({state}). One one-shot agent per worker name at a time; "
            "start concurrent jobs on one credential with --instance, each "
            "with an instance id of its own")


def same_agent(hello: wp.Hello, existing: Any) -> bool:
    """Whether *hello* comes from the agent process that holds *existing*
    (#790): the same ``instance_id``. After a restart of this server the
    holder of an adopted one-shot run knows only the id's hash, recorded
    when the run started (#845, ``Worker.instance_hash``); a known worker
    loaded from the database, or a placeholder, has no ``instance_id``.
    Every agent on one credential holds the same worker token, so the
    token alone cannot tell them apart."""
    if not hello.instance_id:
        return False
    current = getattr(existing, "instance_id", None)
    if current is not None:
        return hello.instance_id == current
    recorded = getattr(existing, "instance_hash", None)
    return recorded is not None and hmac.compare_digest(
        instance_id_hash(hello.instance_id) or "", recorded)


# ── the node catalog ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class CatalogCheck:
    """A worker's node catalog against this server's."""

    status: str = CATALOG_UNKNOWN
    #: The worker's ``node_catalog_hash``.
    hash: str | None = None
    #: Types this server has and the worker does not, sorted.
    missing: tuple[str, ...] = ()
    #: Types the worker has and this server does not, sorted.
    extra: tuple[str, ...] = ()

    @property
    def degraded(self) -> bool:
        return self.status == CATALOG_MISMATCH

    @property
    def lacks_types(self) -> bool:
        """It lacks node types this server has: ``any`` skips it."""
        return self.degraded and bool(self.missing)

    def describe(self) -> dict[str, Any]:
        """The ``node_catalog`` entry of ``GET /api/workers``."""
        return {"status": self.status, "hash": self.hash,
                "missing": list(self.missing), "extra": list(self.extra)}


def check_catalog(control: Any, worker_hash: str | None,
                  worker_types: Iterable[str]) -> CatalogCheck:
    """Compare a worker's catalog with *control* (a ``NodeCatalog``)."""
    if not worker_hash:
        return CatalogCheck()
    if worker_hash == control.hash:
        return CatalogCheck(CATALOG_MATCH, worker_hash)
    types = set(worker_types)
    ours = set(control.node_types)
    return CatalogCheck(CATALOG_MISMATCH, worker_hash,
                        missing=tuple(sorted(ours - types)),
                        extra=tuple(sorted(types - ours)))


def _named(types: list[str]) -> str:
    shown = ", ".join(types[:MAX_TYPES_NAMED])
    more = len(types) - MAX_TYPES_NAMED
    return shown + (f" and {more} more" if more > 0 else "")


#: The node that runs an installed preset by name, and its param naming it.
MAP_NODE_TYPE = "Map"
MAP_PRESET_PARAM = "subgraph"


def graph_node_types(graph: dict[str, Any]) -> set[str]:
    """The node types a run of *graph* executes. Blocking: call it in a
    thread from async code.

    After preset and subgraph expansion, bypass and pruning, exactly as
    the engine prepares it (``prepare_executable_graph``), plus the types
    inside every preset a ``Map`` node runs by name: Map looks its
    ``subgraph`` up in the installed presets (``preset_registry``) when it
    runs, so those types execute too (:func:`preset_node_types`). A graph
    that does not prepare here (it uses a type this server does not know,
    or is malformed) is read conservatively instead: every type named at
    the top level, inside its subgraph definitions and inside its embedded
    preset definitions, containers themselves left out.
    """
    types: set[str] = set()
    for node in _prepared_nodes(graph):
        node_type = node.get("type")
        if not isinstance(node_type, str) or not node_type \
                or node_type.startswith("subgraph:"):
            continue
        if node_type.startswith("preset:"):
            types |= preset_node_types(node_type[len("preset:"):], graph)
            continue
        types.add(node_type)
        if node_type == MAP_NODE_TYPE:
            types |= preset_node_types(_map_preset(node), graph)
    return types


def _iter_nodes(graph: dict[str, Any]):
    """Every node dict of *graph*: top level, subgraph and embedded preset
    definitions."""
    yield from (n for n in graph.get("nodes") or [] if isinstance(n, dict))
    for container in (graph.get("subgraphs") or []) + \
            (graph.get("presets") or []):
        if isinstance(container, dict):
            yield from (n for n in container.get("nodes") or []
                        if isinstance(n, dict))


def _map_preset(node: dict[str, Any]) -> str | None:
    """The preset a Map node runs: ``data.params`` on a graph node,
    ``params`` on a preset's internal node."""
    data = node.get("data")
    params = data.get("params") if isinstance(data, dict) \
        else node.get("params")
    value = params.get(MAP_PRESET_PARAM) if isinstance(params, dict) else None
    return value if isinstance(value, str) and value else None


def preset_node_types(name: str | None, graph: dict[str, Any] | None = None,
                      _seen: set[str] | None = None) -> set[str]:
    """The node types an installed preset *name* runs, recursively: its
    nodes, the presets its ``preset:`` nodes name and the presets its own
    ``Map`` nodes run. Resolved as the engine resolves them: the graph's own
    ``presets[]`` entry of that name first, else the installed one. An
    unknown name contributes nothing (the run fails on it anyway)."""
    from .preset_registry import preset_registry

    seen = set() if _seen is None else _seen
    if not name or name in seen:
        return set()
    seen.add(name)
    definition: Any = None
    for entry in (graph or {}).get("presets") or []:
        if isinstance(entry, dict) and entry.get("preset_name") == name:
            definition = entry
            break
    if definition is None:
        installed = preset_registry.get(name)
        definition = None if installed is None else installed.model_dump()
    if not isinstance(definition, dict):
        return set()
    types: set[str] = set()
    for node in definition.get("nodes") or []:
        node_type = node.get("type") if isinstance(node, dict) else None
        if not isinstance(node_type, str) or not node_type:
            continue
        if node_type.startswith("preset:"):
            types |= preset_node_types(node_type[len("preset:"):], graph,
                                       seen)
            continue
        types.add(node_type)
        if node_type == MAP_NODE_TYPE:
            types |= preset_node_types(_map_preset(node), graph, seen)
    return types


def _prepared_nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    """The nodes the engine would execute, or, for a graph that does not
    prepare here, every node it names anywhere (:func:`_iter_nodes`)."""
    from .graph_engine import (
        GraphValidationError,
        build_preset_fallback,
        prepare_executable_graph,
    )

    try:
        nodes, _edges, _containers = prepare_executable_graph(
            list(graph.get("nodes") or []), list(graph.get("edges") or []),
            preset_fallback=build_preset_fallback(graph.get("presets") or []),
            subgraphs=graph.get("subgraphs") or [])
        return [n for n in nodes if isinstance(n, dict)]
    except (GraphValidationError, KeyError, TypeError, ValueError,
            AttributeError):
        return list(_iter_nodes(graph))


def missing_node_types(graph_types: Iterable[str],
                       worker_types: Iterable[str]) -> list[str]:
    """The types of *graph_types* (:func:`graph_node_types`) that
    *worker_types* lacks, sorted.

    A bare type (``Edu-KNN``, from before plugin namespacing) counts as
    present when the worker has it under any plugin prefix, the way
    ``NodeRegistry.get`` resolves it.
    """
    have = set(worker_types)
    suffixes = {t.partition(":")[2] for t in have if ":" in t}
    return sorted(t for t in set(graph_types)
                  if t not in have and not (":" not in t and t in suffixes))


def catalog_refusal(worker: Any, graph_types: Iterable[str]) -> str | None:
    """Why a graph using *graph_types* may not run on *worker*, or None.

    Only a worker whose catalog differs is checked: one whose catalog
    matches has every type this server has, and one that reported none is
    not checked (see the module docstring).
    """
    check: CatalogCheck = getattr(worker, "catalog", None) or CatalogCheck()
    if not check.degraded:
        return None
    missing = missing_node_types(graph_types, worker.node_types)
    if not missing:
        return None
    return (f"worker {worker.name!r} does not have the node type(s) this "
            f"graph uses: {_named(missing)}. Its node catalog differs from "
            "this server's; install the same plugins and packs on it, or "
            "run on another target")


# ── the fingerprint ───────────────────────────────────────────────────────


def verified_fingerprint(
    report: wp.FingerprintReport | None, *, worker: str = "?",
) -> tuple[dict[str, Any] | None, str | None]:
    """``(document, digest)`` from a ``hello``'s fingerprint, or Nones.

    The digest is recomputed here from the document's canonical bytes
    (``env_fingerprint.fingerprint_digest``), so the one recorded is the
    one the stored document hashes to. A worker that reported another
    digest is logged; its document still stands.
    """
    if report is None:
        return None, None
    from .env_fingerprint import fingerprint_digest

    try:
        digest = fingerprint_digest(report.document)
    except (TypeError, ValueError) as exc:
        logger.warning("worker %s sent a fingerprint that cannot be "
                       "canonicalised (%s); none recorded", worker, exc)
        return None, None
    if digest != report.digest:
        logger.warning("worker %s reported fingerprint digest %s, but its "
                       "document hashes to %s; recording %s", worker,
                       report.digest, digest, digest)
    return report.document, digest
