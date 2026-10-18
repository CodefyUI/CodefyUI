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
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable

from .memory_budget import format_bytes, value_bytes

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptureAlias:
    """What a recorded run keeps under a port no node of its own produced.

    A preset card, a block instance and a bypassed node never run --
    expansion and bypass resolution replace them -- but the canvas draws
    their ports, and every reader of ``/api/execution/outputs`` asks for
    those (#553). This names the port whose capture holds the value, and the
    route follows it. The value is not stored a second time: the store
    measures every slot against its byte budget, so a copy would count a
    tensor twice and evict other runs early. An alias itself measures 0
    bytes.

    It always names a port that RAN, never another alias, so one step
    reaches the value, and it is written only after that port's value. The
    two writes take the store's lock one at a time, so a run dropped between
    them -- evicted, or deleted mid-run -- comes back holding the alias
    alone, and reading it answers 404. That is the truth: the value is gone.

    Defined here rather than in ``graph_engine`` (which still names it) so
    that the readers of a worker's recorded outputs (#789) need not import
    the engine.
    """

    node_id: str
    port: str


#: Runs the store retains unless told otherwise: the server's store
#: (``main.py``) and a worker's recorded outputs (``agent.outputs``, #789).
MAX_RUNS = 20
#: Worker runs evicted from retention that the store remembers as such
#: (#789), newest kept: a read of one answers as for any run not held,
#: instead of finding it again through its run record.
EXPIRED_REMEMBERED = 1000


class RunOutputStore:
    def __init__(self, max_runs: int = MAX_RUNS, *,
                 max_bytes: int | None = None) -> None:
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
        #: Worker runs whose captures the WORKER holds (#789): run id ->
        #: worker name. Each is in ``_order`` like a run held here, holding
        #: no bytes here, so this store's run-count retention applies to
        #: them by run id; evicting one calls :attr:`on_remote_drop`.
        self._remote: dict[str, str] = {}
        #: ``(run_id, worker)`` when a worker run is evicted or deleted
        #: here: its worker is to be told to drop the captures.
        #: Synchronous; it schedules the message. None: nobody to tell.
        self.on_remote_drop: Callable[[str, str], None] | None = None
        #: Worker runs evicted by retention or deleted, oldest first, at
        #: most :data:`EXPIRED_REMEMBERED` (#789): never retained again.
        self._expired: OrderedDict[str, None] = OrderedDict()
        #: Called with the run id whenever a worker run stops being
        #: retained, for any reason (#789): its cached stats go with it.
        self.on_remote_forget: Callable[[str], None] | None = None
        self._lock = asyncio.Lock()

    @property
    def max_runs(self) -> int:
        """The run-count bound, read-only.

        A sweep refuses ``record_outputs`` when it would compile more
        variants than this (#140, spec 8.3), read from the LIVE store rather
        than from a literal 20 -- so raising the bound raises that limit with
        it. Already surfaced in ``stats()``; this is the same number without
        the async hop.
        """
        return self._max_runs

    async def put(self, run_id: str, node_id: str, port: str, value: Any) -> None:
        # Measured OUTSIDE the lock: walking a captured value is pure
        # reading, and doing it under the lock would serialise every
        # concurrent run's captures behind the largest tensor in flight.
        nbytes = value_bytes(value)
        async with self._lock:
            if run_id not in self._store:
                if self._remote.pop(run_id, None) is not None:
                    self._order.remove(run_id)
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
            self._drop(self._order[0], tell=True, evicted=True)
        if not self._max_bytes:
            return
        index = 0
        while self._total_bytes > self._max_bytes and index < len(self._order):
            oldest = self._order[index]
            if oldest == keep or oldest in self._remote:
                # A worker's run holds no bytes here: dropping it frees none.
                index += 1
                continue
            # Read BEFORE the drop, so the message reports the overrun it
            # is reacting to rather than the total left after the fix.
            over = self._total_bytes
            self._drop(oldest)
            logger.info(
                "run output store over budget (%s of %s); dropped the "
                "captures of run %s, leaving %s",
                format_bytes(over), format_bytes(self._max_bytes), oldest,
                format_bytes(self._total_bytes),
            )

    def _drop(self, run_id: str, *, tell: bool = False,
              evicted: bool = False) -> None:
        """Forget one run and everything it captured. Caller holds the lock.

        For a worker's run (#789): *tell* has its worker told
        (:attr:`on_remote_drop`); *evicted* (a retention limit dropped it)
        remembers it as expired (:meth:`was_evicted`).
        """
        worker = self._remote.pop(run_id, None)
        if worker is not None and evicted:
            self._expire(run_id)
        if worker is not None and self.on_remote_forget is not None:
            try:
                self.on_remote_forget(run_id)
            except Exception:  # noqa: BLE001 -- a cache, not a duty
                logger.warning("could not forget the cached stats of run %s",
                               run_id, exc_info=True)
        if worker is not None and tell and self.on_remote_drop is not None:
            try:
                self.on_remote_drop(run_id, worker)
            except Exception:  # noqa: BLE001 -- eviction must not fail a put
                logger.warning("could not tell worker %s to drop the "
                               "outputs of run %s", worker, run_id,
                               exc_info=True)
        nodes = self._store.pop(run_id, {})
        for node_id, ports in nodes.items():
            for port in ports:
                self._slot_bytes.pop((run_id, node_id, port), None)
        self._total_bytes -= self._run_bytes.pop(run_id, 0)
        try:
            self._order.remove(run_id)
        except ValueError:
            pass

    def _expire(self, run_id: str) -> None:
        """Remember *run_id* as a worker run never to retain again.
        Caller holds the lock."""
        self._expired[run_id] = None
        self._expired.move_to_end(run_id)
        while len(self._expired) > EXPIRED_REMEMBERED:
            self._expired.popitem(last=False)

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
        """Forget *run_id*: its captures here, or, for a worker's run
        (#789), its retention, telling its worker to drop them. False when
        neither is held."""
        async with self._lock:
            if run_id not in self._store and run_id not in self._remote:
                return False
            self._drop(run_id, tell=True)
            return True

    # -- worker runs (#789) -------------------------------------------------

    async def add_remote(self, run_id: str, worker: str) -> None:
        """Retain *run_id*, whose captures *worker* holds, by run id.

        Newest in the retention order, as a run that starts writing here
        would be, evicting older runs past ``max_runs`` (and telling their
        workers) as :meth:`put` does. A run held here, and one evicted or
        deleted before (:meth:`was_evicted`), are left as they are: taking
        one back would evict another run for captures that are gone.
        """
        async with self._lock:
            if run_id in self._store or run_id in self._expired:
                return
            if run_id in self._remote:
                self._order.remove(run_id)
            self._remote[run_id] = worker
            self._order.append(run_id)
            self._evict_locked(keep=run_id)

    async def remote_holder(self, run_id: str) -> str | None:
        """The worker holding *run_id*'s captures, or None (#789)."""
        async with self._lock:
            return self._remote.get(run_id)

    async def was_evicted(self, run_id: str) -> bool:
        """*run_id* is a worker's run this store evicted or deleted
        (#789): its reads answer as for a run not held."""
        async with self._lock:
            return run_id in self._expired

    async def forget_remote(self, run_id: str, *,
                            expire: bool = False) -> bool:
        """Stop retaining a worker's run without telling the worker (its
        captures are deleted there, or gone). *expire* also remembers it
        as never to retain again, retained now or not. False when it was
        not retained."""
        async with self._lock:
            if expire and run_id not in self._store:
                self._expire(run_id)
            if run_id not in self._remote:
                return False
            self._drop(run_id)
            return True

    async def snapshot(self, run_id: str) -> list[tuple[str, str, Any, int, int]]:
        """``(node_id, port, value, write serial, bytes)`` for every slot of
        *run_id*, in :meth:`list_ports` order; ``[]`` for a run not held.

        What a worker's run subprocess writes to disk when the run ends
        (#789, ``agent.outputs``), so that its agent serves it afterwards.
        """
        async with self._lock:
            run = self._store.get(run_id) or {}
            return [
                (node_id, port, slot[0], slot[1],
                 self._slot_bytes.get((run_id, node_id, port), 0))
                for node_id, ports in run.items()
                for port, slot in ports.items()
            ]

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
            self._remote.clear()
            self._expired.clear()
            self._order.clear()
            self._run_bytes.clear()
            self._slot_bytes.clear()
            self._total_bytes = 0
