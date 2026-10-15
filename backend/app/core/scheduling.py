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
the connected worker agents.

Admission (#773) is count-based, per slot: ``limits.for_slot(slot)`` runs
may hold a slot at once (1 per accelerator, 2 for ``cpu`` by default,
``CODEFYUI_RUN_QUEUE_MAX_CONCURRENT`` overrides per slot). A run whose
placement names several slots (a device set, here or on a worker) starts
only when EVERY one has room, takes them all in one step and gives them
all back in one step; there is no partial hold, so two set runs cannot
deadlock each other. There is no reservation either: while a set run waits
for its second card, a single-device run on that card (in another FIFO)
may start first. Priorities and preemption are out of scope.

Same-host workers (#773): an agent that dialled in from loopback runs on
this machine, so its ``cuda:0`` is this process's ``cuda:0``. Its slots are
ALIASED onto this process's (:func:`slot_for`): ``("local", "cuda:0")``
for both, one budget under the local cap, so an in-process run and a
worker run cannot both take one physical card under ``gpu=1``. The device
names are matched as spelled; an agent that renumbers its cards (say,
started under a different ``CUDA_VISIBLE_DEVICES``) is outside what this
can see.

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
from .worker_protocol import TARGET_ANY, TARGET_LOCAL, canonical_device

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


def on_a_worker(placement: "Placement | None") -> bool:
    """Whether *placement* puts a run on a worker agent rather than here."""
    return placement is not None and placement.worker_id != LOCAL_WORKER_ID


class QueuedEntry(Protocol):
    """What a scheduler may read off a waiting run (``run_service._QueuedRun``)."""

    run_id: str
    queue_key: str
    options: dict[str, Any]


class SlotLimits(Protocol):
    """The per-slot concurrency caps (``run_service.QueueLimits``)."""

    def for_slot(self, slot: SlotKey) -> int: ...


def slot_for(worker: Any, device: str) -> SlotKey:
    """The slot a run on *device* of *worker* takes.

    ``(worker.id, device)``, except for a worker on this machine
    (``worker.same_host``, recorded at the handshake from a loopback peer
    address): its cards are this process's cards, so its slot is
    ``("local", device)`` and shares this process's budget and cap.
    """
    if getattr(worker, "same_host", False):
        return (LOCAL_WORKER_ID, device)
    return (worker.id, device)


def local_set_slots(queue_key: str,
                    devices: list[str] | None) -> tuple[SlotKey, ...]:
    """This process's slots for a run on *queue_key*, with its device set.

    One slot without a set (the 3.0 rule). With one (#774), a slot on every
    device of the set, the run's own queue key first: the in-process run
    executes on the first device, and holding the rest keeps another run
    off the cards its nodes may address.
    """
    slots: list[SlotKey] = [(LOCAL_WORKER_ID, queue_key)]
    for device in devices or ():
        try:
            device = canonical_device(device)
        except ValueError:
            continue
        if (LOCAL_WORKER_ID, device) not in slots:
            slots.append((LOCAL_WORKER_ID, device))
    return tuple(slots)


def _all_free(limits: SlotLimits, slots: tuple[SlotKey, ...],
              taken: Mapping[SlotKey, int]) -> bool:
    return all(taken.get(slot, 0) < limits.for_slot(slot) for slot in slots)


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
    """Everything runs in this process; one slot per device, capped per slot.

    A run without a device set takes the one slot ``("local", queue_key)``,
    the 3.0 rule unchanged. A ``target.worker: local`` run with a device set
    takes a slot on every device of it, all or nothing
    (:func:`local_set_slots`); it still executes on its queue key.
    """

    def __init__(self, limits: SlotLimits) -> None:
        self._limits = limits

    def place(self, entry: QueuedEntry,
              taken: Mapping[SlotKey, int]) -> Placement | None:
        queue_key = entry.queue_key
        _, devices = entry_target(entry.options)
        slots = local_set_slots(queue_key, devices)
        if not _all_free(self._limits, slots, taken):
            return None
        return Placement(LOCAL_WORKER_ID, (queue_key,), slots)




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
      the same way, in name order, and this process last, judged the same
      way against *local_devices* (this process's devices, canonical).
      With a device set, this process qualifies only when it has every
      device of the set; it then executes on the set's first device, as a
      local run with a set does.

    Every worker slot is ``(worker_id, device)`` capped by
    ``limits.for_slot(slot)`` (a same-host worker's is this process's slot,
    see :func:`slot_for`); a device set takes one slot per device, here or
    on a worker, and starts only when all of them are free.

    :meth:`explain` says why a run that :meth:`place` refused cannot start
    for a reason other than a busy slot (the Runs list shows it).
    """

    def __init__(self, limits: SlotLimits, registry: Any,
                 local_devices: Callable[[], list[str]]) -> None:
        self._limits = limits
        self._registry = registry
        self._local_devices = local_devices
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
        return self._place_here(requested, devices, taken)

    def explain(self, entry: QueuedEntry) -> str | None:
        """Why *entry* is waiting, when it is not just a busy slot."""
        worker_id, devices = entry_target(entry.options)
        if worker_id == TARGET_LOCAL:
            return None
        requested = str(entry.options.get("device") or "cpu")
        if worker_id != TARGET_ANY:
            worker = self._registry.get(worker_id)
            if worker is None or not worker.online:
                return f"waiting for worker {worker_id} to reconnect"
            if worker.draining:
                return f"worker {worker_id} is draining"
            return None
        if not self.can_serve_any(requested, devices):
            wanted = ", ".join(devices) if devices else requested
            return f"waiting for a worker that offers {wanted}"
        return None

    def can_serve_any(self, requested: str,
                      devices: list[str] | None) -> bool:
        """Whether this process or a connected worker could ever run it."""
        if _serves(self._local_devices(), requested, devices):
            return True
        return any(w.online and _serves(w.device_values(), requested, devices)
                   for w in self._registry.workers())

    def _place_here(self, requested: str, devices: list[str] | None,
                    taken: Mapping[SlotKey, int]) -> Placement | None:
        offered = self._local_devices()
        if devices:
            if not _serves(offered, requested, devices):
                return None
            first = canonical_device(devices[0])
            slots = local_set_slots(first, devices)
            if _all_free(self._limits, slots, taken):
                return Placement(LOCAL_WORKER_ID, (first,), slots)
            return None
        for device in candidate_devices(offered, requested):
            slot: SlotKey = (LOCAL_WORKER_ID, device)
            if _all_free(self._limits, (slot,), taken):
                return Placement(LOCAL_WORKER_ID, (device,), (slot,))
        return None

    def _place_on(self, worker: Any, requested: str,
                  devices: list[str] | None,
                  taken: Mapping[SlotKey, int]) -> Placement | None:
        if devices:
            if not all(worker.offers(d) for d in devices):
                return None
            slots = tuple(slot_for(worker, d) for d in devices)
            if _all_free(self._limits, slots, taken):
                return Placement(worker.id, tuple(devices), slots)
            return None
        for device in candidate_devices(worker.device_values(), requested):
            slot = slot_for(worker, device)
            if _all_free(self._limits, (slot,), taken):
                return Placement(worker.id, (device,), (slot,))
        return None


def _serves(offered: list[str], requested: str,
            devices: list[str] | None) -> bool:
    """Whether a host offering *offered* can run the request at all."""
    if devices:
        return all(canonical_device(d) in offered for d in devices)
    return bool(candidate_devices(offered, requested))


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
    try:
        requested = canonical_device(requested)
    except ValueError:
        return []
    if ":" in requested or requested == "cpu":
        return [requested] if requested in offered else []
    return [d for d in offered if d.split(":")[0] == requested]
