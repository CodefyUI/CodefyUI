"""Placement: which device slots a waiting run may take (#768).

``RunService._pump`` owns the queue (one FIFO per queue key, submit order)
and asks a :class:`Scheduler` one question about the run at the front: can
it start now, and on which slots? The answer is a :class:`Placement` or
``None`` ("not yet"; the run stays at the front of its line).

A slot belongs to a device ON A WORKER, so it is keyed by
``(worker_id, device)``: ``("local", "cuda:0")`` is this process's first
card. A run may hold several slots at once (a device set, #774), which is
why a placement carries a tuple of them and why releasing a run gives back
every one.

:class:`LocalScheduler` reproduces the 3.0 rule exactly: a run on queue key
``K`` takes the single slot ``("local", K)`` while fewer than
``limits.for_key(K)`` runs hold it. Worker-aware placement is #772/#773.

This module imports nothing from ``run_service``: the scheduler sees a
waiting run through :class:`QueuedEntry` and the caps through
:class:`SlotLimits`, both structural, so ``run_service`` can depend on it
without a cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .run_store import WORKER_LOCAL

#: ``(worker_id, device)``, e.g. ``("local", "cuda:0")``.
SlotKey = tuple[str, str]

#: The worker id of this process. The same string ``exec_runs.worker_id``
#: stores for a local run (``run_store.WORKER_LOCAL``), aliased rather than
#: spelled again so the two cannot drift.
LOCAL_WORKER_ID = WORKER_LOCAL


@dataclass(frozen=True)
class Placement:
    """Where a run starts: one worker, the devices it gets, the slots it holds.

    ``slots`` is what admission accounting takes on start and gives back on
    release; ``devices`` is what the run executes on. They coincide for a
    local single-device run and are kept apart so a later placement can
    hold slots it does not name as an execution device (or the reverse)
    without changing this type.
    """

    worker_id: str
    devices: tuple[str, ...]
    slots: tuple[SlotKey, ...]


class QueuedEntry(Protocol):
    """What a scheduler may read off a waiting run (``run_service._QueuedRun``)."""

    run_id: str
    queue_key: str
    options: dict[str, Any]


class SlotLimits(Protocol):
    """The per-key concurrency caps (``run_service.QueueLimits``)."""

    def for_key(self, queue_key: str) -> int: ...


class Scheduler(Protocol):
    """Decides whether the run at the front of its queue may start now.

    ``taken`` maps each slot to how many runs currently hold it; a slot
    nobody holds may be absent. Must be synchronous and must not mutate
    ``taken``: ``_pump`` calls it between taking the entry's place in line
    and starting it, with no ``await`` in between, and does the accounting
    itself once the start succeeds.
    """

    def place(self, entry: QueuedEntry,
              taken: Mapping[SlotKey, int]) -> Placement | None: ...


class LocalScheduler:
    """Everything runs in this process; one slot per run, capped per device."""

    def __init__(self, limits: SlotLimits) -> None:
        self._limits = limits

    def place(self, entry: QueuedEntry,
              taken: Mapping[SlotKey, int]) -> Placement | None:
        queue_key = entry.queue_key
        slot: SlotKey = (LOCAL_WORKER_ID, queue_key)
        if taken.get(slot, 0) >= self._limits.for_key(queue_key):
            return None
        return Placement(LOCAL_WORKER_ID, (queue_key,), (slot,))
