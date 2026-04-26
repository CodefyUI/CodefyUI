"""In-memory store for persistent ``nn.Module`` instances per (graph, node).

Every layer-style node opting in via :class:`StatefulModuleMixin` looks up its
underlying ``nn.Module`` here on each execute. The first lookup builds the
module; subsequent lookups (with matching structural params) reuse the same
instance so its weights persist across multiple Run clicks. This is what makes
"Conv2d that students can actually train" possible.

Key composition: ``(graph_id, node_id, structure_hash)``.

  * ``graph_id``: stable per-tab UUID assigned by the frontend (``tabStore``).
    Falls back to the empty string for legacy clients.
  * ``node_id``: tab-unique node identifier from React Flow.
  * ``structure_hash``: SHA-256 hash of the structural params declared by the
    node (e.g. for Conv2d: in_channels, out_channels, kernel_size, stride,
    padding). When the user changes a structural param the hash changes,
    we evict any sibling key for the same (graph_id, node_id), and build a
    fresh module — old weights are no longer dimensionally compatible.

LRU eviction caps total stored modules at ``max_modules`` to bound memory.

Thread safety: graph_engine runs ``execute()`` on a thread-pool executor
(``loop.run_in_executor``). All mutating accessors are guarded by a
``threading.Lock``. A per-key lock is also exposed so callers can serialise
forward passes through the same module instance — autograd is not generally
re-entrant.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Iterator

if TYPE_CHECKING:
    import torch


class NodeStateStore:
    def __init__(self, max_modules: int = 200) -> None:
        self._max = max_modules
        self._store: dict[tuple[str, str, str], "torch.nn.Module"] = {}
        self._lru: deque[tuple[str, str, str]] = deque()
        self._lock = threading.Lock()
        self._key_locks: dict[tuple[str, str, str], threading.Lock] = {}

    # ── core lookup ─────────────────────────────────────────────────

    def get_or_create(
        self,
        graph_id: str,
        node_id: str,
        structure_hash: str,
        builder: Callable[[], "torch.nn.Module"],
    ) -> "torch.nn.Module":
        """Return the persisted module or build a new one.

        On structural change for the same (graph_id, node_id), all sibling
        entries for that node are evicted before the new module is created.
        Weights from a different shape would crash the forward pass anyway,
        and keeping them around would just waste memory.
        """
        key = (graph_id, node_id, structure_hash)
        with self._lock:
            existing = self._store.get(key)
            if existing is not None:
                # Touch LRU.
                try:
                    self._lru.remove(key)
                except ValueError:
                    pass
                self._lru.append(key)
                return existing

            # Structure changed for this node — drop stale siblings.
            stale = [
                k for k in self._store
                if k[0] == graph_id and k[1] == node_id
            ]
            for s in stale:
                self._store.pop(s, None)
                try:
                    self._lru.remove(s)
                except ValueError:
                    pass
                self._key_locks.pop(s, None)

        # Builder runs OUTSIDE the lock — module construction can be slow
        # and may itself acquire locks (e.g. CUDA initialisation).
        module = builder()

        with self._lock:
            self._store[key] = module
            self._lru.append(key)
            while len(self._lru) > self._max:
                evict = self._lru.popleft()
                self._store.pop(evict, None)
                self._key_locks.pop(evict, None)
        return module

    def per_key_lock(self, graph_id: str, node_id: str, structure_hash: str) -> threading.Lock:
        """Return a lock that callers can use to serialise forward passes
        through the same persisted module instance."""
        key = (graph_id, node_id, structure_hash)
        with self._lock:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock

    # ── reset / introspection ───────────────────────────────────────

    def reset_node(self, graph_id: str, node_id: str) -> int:
        """Drop all persisted modules for one node. Returns count evicted."""
        with self._lock:
            stale = [
                k for k in self._store
                if k[0] == graph_id and k[1] == node_id
            ]
            for s in stale:
                self._store.pop(s, None)
                try:
                    self._lru.remove(s)
                except ValueError:
                    pass
                self._key_locks.pop(s, None)
            return len(stale)

    def reset_graph(self, graph_id: str) -> int:
        """Drop all persisted modules for an entire graph. Returns count evicted."""
        with self._lock:
            stale = [k for k in self._store if k[0] == graph_id]
            for s in stale:
                self._store.pop(s, None)
                try:
                    self._lru.remove(s)
                except ValueError:
                    pass
                self._key_locks.pop(s, None)
            return len(stale)

    def reset_all(self) -> int:
        with self._lock:
            n = len(self._store)
            self._store.clear()
            self._lru.clear()
            self._key_locks.clear()
            return n

    def iter_for_graph(
        self, graph_id: str,
    ) -> Iterator[tuple[tuple[str, str, str], "torch.nn.Module"]]:
        """Iterate (key, module) pairs belonging to a graph. Snapshot — safe to mutate
        the store after this returns."""
        with self._lock:
            items = [
                (k, m) for k, m in self._store.items() if k[0] == graph_id
            ]
        return iter(items)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def keys_for_node(self, graph_id: str, node_id: str) -> list[tuple[str, str, str]]:
        with self._lock:
            return [
                k for k in self._store
                if k[0] == graph_id and k[1] == node_id
            ]
