"""In-memory LRU store for captured per-run node outputs.

Scope:
  Outputs live only in server memory; they are lost on restart. Indexed
  by (run_id, node_id, port). When the number of tracked runs exceeds
  ``max_runs`` -- or the bytes they hold exceed ``max_bytes`` (core#135) --
  the oldest run is evicted.

  Eviction is by RUN, not by port, under both limits. Half a run's captures
  is worse than none of them: the Teaching Inspector would show a node's
  input and not its output, and nothing in the UI can distinguish "this port
  was never captured" from "this port was captured and then quietly thrown
  away". A whole run disappearing is at least a fact a user can act on.

Thread safety:
  All mutating operations take an ``asyncio.Lock`` so concurrent WebSocket
  runs don't corrupt the internal dict.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any

from .memory_budget import format_bytes, value_bytes

logger = logging.getLogger(__name__)


class RunOutputStore:
    def __init__(self, max_runs: int = 20, *, max_bytes: int | None = None) -> None:
        self._max_runs = max_runs
        if max_bytes is None:
            from ..config import settings

            max_bytes = int(settings.RUN_OUTPUT_STORE_MAX_MB) * 1024 * 1024
        self._max_bytes = max(0, int(max_bytes))
        # Each slot holds ``(value, write_serial)``. See ``get_with_version``
        # for why the serial exists; ``get`` unwraps it, so every caller that
        # only wants the value is unaffected.
        self._store: dict[str, dict[str, dict[str, tuple[Any, int]]]] = {}
        self._order: deque[str] = deque()
        #: Bytes per run, so eviction and ``/api/health`` read one number
        #: instead of re-walking every captured tensor.
        self._run_bytes: dict[str, int] = {}
        #: Bytes per (run, node, port), so OVERWRITING a port can give the
        #: previous value's bytes back without walking it a second time --
        #: a node inside a loop rewrites one port on every iteration.
        self._slot_bytes: dict[tuple[str, str, str], int] = {}
        self._total_bytes = 0
        self._writes = 0
        self._lock = asyncio.Lock()

    async def put(self, run_id: str, node_id: str, port: str, value: Any) -> None:
        # Measured OUTSIDE the lock: walking a captured value is pure
        # reading, and doing it under the lock would serialise every
        # concurrent run's captures behind the largest tensor in flight.
        nbytes = value_bytes(value)
        async with self._lock:
            if run_id not in self._store:
                self._store[run_id] = {}
                self._run_bytes[run_id] = 0
                self._order.append(run_id)
            nodes = self._store[run_id]
            if node_id not in nodes:
                nodes[node_id] = {}
            slot = (run_id, node_id, port)
            previous = self._slot_bytes.pop(slot, 0)
            self._run_bytes[run_id] += nbytes - previous
            self._total_bytes += nbytes - previous
            self._slot_bytes[slot] = nbytes
            self._writes += 1
            nodes[node_id][port] = (value, self._writes)
            self._evict_locked(keep=run_id)

    def _evict_locked(self, *, keep: str | None = None) -> None:
        """Drop oldest runs until both limits hold. Caller holds the lock.

        *keep* is the run just written to, and it is skipped over rather
        than evicted. Evicting the run currently executing to make room for
        the run currently executing is a loop that ends with an empty store
        and the same overrun; a run whose own captures exceed the whole
        budget simply exceeds it, visibly, in ``/api/health``.

        Note ``keep`` is not necessarily the newest run: a long training run
        can outlive several short ones started after it. So the scan walks
        from the oldest and skips it, rather than stopping at it.
        """
        while len(self._order) > self._max_runs:
            self._drop(self._order[0])
        if not self._max_bytes:
            return
        index = 0
        while self._total_bytes > self._max_bytes and index < len(self._order):
            oldest = self._order[index]
            if oldest == keep:
                index += 1
                continue
            self._drop(oldest)
            logger.info(
                "run output store over budget (%s of %s); dropped the "
                "captures of run %s",
                format_bytes(self._total_bytes), format_bytes(self._max_bytes),
                oldest,
            )

    def _drop(self, run_id: str) -> None:
        """Forget one run and everything it captured. Caller holds the lock."""
        nodes = self._store.pop(run_id, {})
        for node_id, ports in nodes.items():
            for port in ports:
                self._slot_bytes.pop((run_id, node_id, port), None)
        self._total_bytes -= self._run_bytes.pop(run_id, 0)
        try:
            self._order.remove(run_id)
        except ValueError:
            pass

    async def get(self, run_id: str, node_id: str, port: str) -> Any | None:
        async with self._lock:
            slot = self._store.get(run_id, {}).get(node_id, {}).get(port)
            return None if slot is None else slot[0]

    async def get_with_version(
        self, run_id: str, node_id: str, port: str
    ) -> tuple[Any, int] | None:
        """The value plus the write serial that produced it, or None.

        The serial is what makes it safe to memoise anything DERIVED from a
        captured value (see ``port_stats``). Object identity is not: a node
        inside a loop overwrites one port several times in a run, and torch's
        caching allocator hands the freed block straight to the replacement —
        so the new tensor can carry the old one's ``data_ptr``, ``shape`` and
        ``dtype`` and be indistinguishable from it. The serial only ever goes
        up, so a cache keyed on it cannot answer for a value that is gone.
        """
        async with self._lock:
            return self._store.get(run_id, {}).get(node_id, {}).get(port)

    async def has_run(self, run_id: str) -> bool:
        async with self._lock:
            return run_id in self._store

    async def delete_run(self, run_id: str) -> bool:
        async with self._lock:
            if run_id not in self._store:
                return False
            self._drop(run_id)
            return True

    async def list_ports(self, run_id: str) -> list[tuple[str, str]] | None:
        """Return (node_id, port) pairs for a run, or None if run not found."""
        async with self._lock:
            run = self._store.get(run_id)
            if run is None:
                return None
            return [
                (node_id, port)
                for node_id, ports in run.items()
                for port in ports.keys()
            ]

    async def list_runs(self) -> list[str]:
        """Return run_ids in LRU order (oldest first)."""
        async with self._lock:
            return list(self._order)

    async def stats(self) -> dict[str, int]:
        """What this store holds, for ``/api/health``."""
        async with self._lock:
            return {
                "runs": len(self._order),
                "max_runs": self._max_runs,
                "bytes": self._total_bytes,
                "max_bytes": self._max_bytes,
            }

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
            self._order.clear()
            self._run_bytes.clear()
            self._slot_bytes.clear()
            self._total_bytes = 0
