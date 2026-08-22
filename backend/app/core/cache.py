"""Node-level execution cache with LRU eviction.

Bounded by TWO limits since core#135, and both are needed. The entry count
stops an enormous number of tiny results from growing the dict without
limit; the byte budget stops a handful of enormous ones from filling a
graphics card. 256 cached node outputs is either 40 MB of scalars or 200 GB
of feature maps, and only one of those numbers can be read off a count.

See ``core.memory_budget`` for what "bytes" means here (storage-level,
recursive through containers, measured once at insertion).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .memory_budget import format_bytes, value_bytes

logger = logging.getLogger(__name__)

#: Every live cache, so ``/api/health`` can report what they hold. Each
#: WebSocket connection owns one, and they come and go with their sockets --
#: hence weak references, which keep this registry from being the reason a
#: closed socket's cache is never collected.
_LIVE: "weakref.WeakSet[ExecutionCache]" = weakref.WeakSet()
_LIVE_LOCK = threading.Lock()


@dataclass
class _Entry:
    """One cached node result, and what it cost to keep."""

    outputs: dict[str, Any]
    nbytes: int
    node_id: str


class ExecutionCache:
    """Cache node outputs keyed by a hash of (type, params, upstream keys).

    Evicts least-recently-used entries when either ``max_entries`` or
    ``max_bytes`` would be exceeded.
    """

    def __init__(
        self,
        max_entries: int = 256,
        *,
        max_bytes: int | None = None,
    ) -> None:
        self._store: OrderedDict[str, _Entry] = OrderedDict()
        self._max_entries = max_entries
        # Read from settings at CONSTRUCTION rather than defaulted in the
        # signature: a default argument is evaluated at import time, which
        # would freeze the value before a test (or an env var read during
        # startup) could change it.
        if max_bytes is None:
            from ..config import settings

            max_bytes = int(settings.EXECUTION_CACHE_MAX_MB) * 1024 * 1024
        self._max_bytes = max(0, int(max_bytes))
        self._total_bytes = 0
        #: node id -> the keys it owns, so an OOM can drop exactly the
        #: entries belonging to the node that could not allocate.
        self._by_node: dict[str, set[str]] = {}
        with _LIVE_LOCK:
            _LIVE.add(self)

    @staticmethod
    def compute_key(
        node_type: str,
        params: dict[str, Any],
        upstream_keys: list[str],
        device: str = "cpu",
        fingerprint: Any = None,
    ) -> str:
        """Compute a deterministic SHA-256 cache key.

        ``device`` is part of the key: a device-aware source node (e.g.
        TensorCreate) produces tensors on the run's global device, so a CPU
        result must not be served for a later MPS run. It is the RESOLVED
        device, so ``cuda:0`` and ``cuda:1`` are different keys -- which is
        the point on a multi-GPU box, where serving one card's tensors to a
        run on the other is a device-mismatch crash rather than a slow path.

        ``upstream_keys`` is one entry PER INCOMING EDGE, and each entry must
        identify the edge's ports as well as its source -- build them with
        :meth:`upstream_ref`. The bare source keys are not enough (#360):
        they say which upstream NODES feed this one, and ``sorted()`` below
        then reduces even that to an unordered bag, so two different wirings
        hash the same. ``Split.chunk_0`` and ``Split.chunk_1`` both reduce to
        the one ``split`` key, and ``MatMul(a=A, b=B)`` and
        ``MatMul(a=B, b=A)`` both reduce to ``{A, B}`` -- in each case the
        second node gets served the first one's outputs on a re-run. The
        ``sorted()`` is still right in INTENT (edge iteration order must not
        change the key) as long as what it sorts keeps the port pair.

        ``fingerprint`` is ``node_cls.cache_fingerprint(params)`` (#144,
        #145): ``None`` for the overwhelming majority of nodes, whose
        params already describe their output completely. A node that reads
        external state a path param only NAMES (a file's bytes, a
        directory's contents) returns a cheap descriptor of that state here
        so a change on disk changes the key even though the path string did
        not. Two calls that both omit it (or both pass ``None``) still
        agree, which is the only compatibility guarantee that matters here:
        ``ExecutionCache`` is in-memory and per-connection (see the module
        docstring), never persisted, so the hash VALUE shifting once this
        field joined the payload -- for every node, not just the ones that
        use it -- has no stale-cache-on-disk to migrate.
        """
        payload = json.dumps(
            {
                "type": node_type,
                "params": params,
                "upstream": sorted(upstream_keys),
                "device": device,
                "fingerprint": fingerprint,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def upstream_ref(
        target_handle: str, source_handle: str, source_key: str
    ) -> str:
        """One ``upstream_keys`` entry: an edge, not just its source node.

        JSON rather than a delimiter-joined string so no handle name can
        forge a boundary -- ``json.dumps`` escapes whatever is in a port
        name, where a ``"a<-b"`` separator would let a handle literally
        called ``a<-b`` collide with a different pair. The list order
        ``(target, source, key)`` puts the receiving port first so that
        ``compute_key``'s ``sorted()`` groups a node's edges by the input
        they land on, which is the reading order a human debugging a key
        wants.
        """
        return json.dumps(
            [target_handle, source_handle, source_key], separators=(",", ":")
        )

    def get(self, key: str) -> dict[str, Any] | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        self._store.move_to_end(key)
        return entry.outputs

    def put(self, key: str, outputs: dict[str, Any], node_id: str = "") -> None:
        """Store *outputs* under *key*, evicting LRU entries to stay in budget.

        A single result LARGER than the whole budget is not stored at all --
        caching it would evict everything else and then still not fit under
        the limit, which is the worst of both. It is a cache miss next time,
        which is the correct outcome for a value the budget cannot hold.
        Note that any PREVIOUS value under the same key goes with it: the
        key encodes the whole computation, so an entry being replaced is
        one the node has just recomputed, and keeping the old copy would
        make the budget's refusal invisible.
        """
        self._drop(key)
        nbytes = value_bytes(outputs)
        if self._max_bytes and nbytes > self._max_bytes:
            logger.info(
                "not caching %s: one result is %s, over the whole %s cache "
                "budget", node_id or key[:12], format_bytes(nbytes),
                format_bytes(self._max_bytes),
            )
            return

        self._store[key] = _Entry(outputs=outputs, nbytes=nbytes,
                                  node_id=node_id)
        self._total_bytes += nbytes
        if node_id:
            self._by_node.setdefault(node_id, set()).add(key)

        while len(self._store) > self._max_entries:
            self._evict_oldest()
        while self._max_bytes and self._total_bytes > self._max_bytes:
            if not self._evict_oldest():
                break

    def clear_node(self, node_id: str) -> int:
        """Drop every entry produced by one node. Returns how many went.

        Called on an out-of-memory failure: whatever that node cached on an
        earlier run is, by construction, the shape of thing that just failed
        to allocate, and it is sitting on the device that ran out.
        """
        keys = list(self._by_node.get(node_id, ()))
        for key in keys:
            self._drop(key)
        return len(keys)

    def clear(self) -> None:
        self._store.clear()
        self._by_node.clear()
        self._total_bytes = 0

    def stats(self) -> dict[str, int]:
        """What this cache holds, for ``/api/health``."""
        return {
            "entries": len(self._store),
            "max_entries": self._max_entries,
            "bytes": self._total_bytes,
            "max_bytes": self._max_bytes,
        }

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def __len__(self) -> int:
        return len(self._store)

    # ── internals ────────────────────────────────────────────────────────

    def _drop(self, key: str) -> bool:
        entry = self._store.pop(key, None)
        if entry is None:
            return False
        self._total_bytes -= entry.nbytes
        owned = self._by_node.get(entry.node_id)
        if owned is not None:
            owned.discard(key)
            if not owned:
                self._by_node.pop(entry.node_id, None)
        return True

    def _evict_oldest(self) -> bool:
        try:
            key = next(iter(self._store))
        except StopIteration:
            return False
        return self._drop(key)


def execution_cache_stats() -> dict[str, int]:
    """Totals across every live ``ExecutionCache`` in this process.

    There is one per WebSocket connection rather than one per server, so
    "how much is the execution cache holding" is a sum over sockets. The
    instance count is reported alongside because a total of 4 GB means
    something different across one connection than across twelve.
    """
    from ..config import settings

    with _LIVE_LOCK:
        caches = list(_LIVE)
    # The configured budget, not ``caches[0].max_bytes``: with no editor
    # connected there is no instance to read it off, and reporting 0 there
    # would say "no budget" rather than "nothing is using it".
    return {
        "instances": len(caches),
        "entries": sum(len(c) for c in caches),
        "bytes": sum(c.total_bytes for c in caches),
        "max_bytes_each": int(settings.EXECUTION_CACHE_MAX_MB) * 1024 * 1024,
    }
