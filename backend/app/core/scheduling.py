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
``limits.for_key(K)`` runs hold it. :class:`WorkerScheduler` (#772) adds
the connected worker agents; per-device slots for local device sets are #773.

This module imports nothing from ``run_service``: the scheduler sees a
waiting run through :class:`QueuedEntry` and the caps through
:class:`SlotLimits`, both structural, so ``run_service`` can depend on it
without a cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Protocol

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

    A placement must name at least one slot; ``_pump`` treats an empty
    ``slots`` (and an exception) as a refusal, since an uncounted start
    would let the whole FIFO drain past its cap.
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


#: ``target.worker`` values with a meaning of their own (#769).
TARGET_LOCAL = LOCAL_WORKER_ID
TARGET_ANY = "any"


def entry_target(options: Mapping[str, Any]) -> tuple[str, list[str] | None]:
    """``(worker, devices)`` of a run's normalised options; ``local`` without
    a target."""
    target = options.get("target") if isinstance(options, Mapping) else None
    if not isinstance(target, Mapping):
        return TARGET_LOCAL, None
    return (target.get("worker") or TARGET_LOCAL), target.get("devices")


class WorkerScheduler:
    """Placement across this process and the connected workers (#772).

    - no target, or ``target.worker == "local"``: :class:`LocalScheduler`,
      unchanged (the same FIFO key, the same single slot);
    - a named worker: that worker, if it is connected and not draining,
      on the run's device set, or its requested device, or (a bare kind
      such as ``cuda``, or ``auto``) the first device of that kind it
      offers with a free slot;
    - ``any``: the first connected, non-draining worker that can take it
      the same way, in name order, and this process last (on the device
      *resolve_local* gives the request, under the local cap).

    Every worker slot is ``(worker_id, device)`` capped by
    ``limits.for_key(device)``; a device set takes one slot per device and
    starts only when all of them are free. A local run keeps the one slot
    it has always taken.
    """

    def __init__(self, limits: SlotLimits, registry: Any,
                 resolve_local: Callable[[str], str]) -> None:
        self._limits = limits
        self._registry = registry
        self._resolve_local = resolve_local
        self._local = LocalScheduler(limits)

    def place(self, entry: QueuedEntry,
              taken: Mapping[SlotKey, int]) -> Placement | None:
        worker_id, devices = entry_target(entry.options)
        if worker_id == TARGET_LOCAL:
            return self._local.place(entry, taken)
        requested = str(entry.options.get("device") or "cpu")
        if worker_id != TARGET_ANY:
            worker = self._registry.get(worker_id)
            if worker is None or not worker.accepts_runs:
                return None
            return self._place_on(worker, requested, devices, taken)
        for worker in self._registry.workers():
            if worker.accepts_runs:
                placement = self._place_on(worker, requested, devices, taken)
                if placement is not None:
                    return placement
        local = self._resolve_local(requested)
        slot: SlotKey = (LOCAL_WORKER_ID, local)
        if taken.get(slot, 0) >= self._limits.for_key(local):
            return None
        return Placement(LOCAL_WORKER_ID, (local,), (slot,))

    def _free(self, slot: SlotKey, taken: Mapping[SlotKey, int]) -> bool:
        return taken.get(slot, 0) < self._limits.for_key(slot[1])

    def _place_on(self, worker: Any, requested: str,
                  devices: list[str] | None,
                  taken: Mapping[SlotKey, int]) -> Placement | None:
        if devices:
            if not all(worker.offers(d) for d in devices):
                return None
            slots = tuple((worker.id, d) for d in devices)
            if all(self._free(slot, taken) for slot in slots):
                return Placement(worker.id, tuple(devices), slots)
            return None
        for device in candidate_devices(worker.device_values(), requested):
            slot: SlotKey = (worker.id, device)
            if self._free(slot, taken):
                return Placement(worker.id, (device,), (slot,))
        return None


def candidate_devices(offered: list[str], requested: str) -> list[str]:
    """The devices of *offered* that can serve *requested*, best first.

    A concrete device (``cuda:1``, ``cpu``) is itself when offered; a bare
    kind is every offered device of that kind; ``auto`` is CUDA, then MPS,
    then the CPU, like ``resolve_device``.
    """
    requested = requested.strip().lower()
    if requested == "auto":
        order = ("cuda", "mps", "cpu")
        return sorted((d for d in offered if d.split(":")[0] in order),
                      key=lambda d: order.index(d.split(":")[0]))
    if requested == "mps":
        requested = "mps:0"
    if ":" in requested or requested == "cpu":
        return [requested] if requested in offered else []
    return [d for d in offered if d.split(":")[0] == requested]
