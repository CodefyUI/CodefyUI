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

Admission (#773) is count-based, per slot: while fewer than
``limits.for_slot(slot)`` runs hold a slot, another may take it (1 per
accelerator and 2 for ``cpu`` by default; ``CODEFYUI_RUN_QUEUE_MAX_CONCURRENT``
overrides one slot). :class:`LocalScheduler` reproduces the 3.0 rule for a
run without a device set: queue key ``K`` takes the single slot
``("local", K)``. :class:`WorkerScheduler` (#772) adds the connected
worker agents.

Device sets: a run whose placement names several slots (``target.devices``,
here or on a worker) starts only when EVERY one has room, takes them all
in one step and gives them all back in one step; it never holds part of a
set. While it waits at the front of its FIFO, ``RunService`` RESERVES the
slots it needs (:meth:`WorkerScheduler.reservation`): no other run may
start on a reserved card until the set run starts, runs already on those
cards finish normally, and so the set run starts once they have. The
price is idle capacity: a reserved card that is free stays idle until the
set's other cards free too, at most as long as the longest run already
holding one of them. Reservations are first come, first served (an
earlier reservation also blocks a later set run), so two set runs cannot
reserve each other into a deadlock. An ``any`` run with a set reserves
nothing: which worker's cards it needs is not known until it is placed.
Nor does it ever run in this process (:class:`WorkerScheduler`).

Same-host workers (#773): a worker whose ``hello.machine_id`` equals this
process's runs on this machine. Each card it shares with this process
(``Worker.shared``: ``cpu``, ``mps:0``, and every CUDA card whose UUID
matches one here, whatever its index on either side) is ALIASED onto this
process's slot (:func:`slot_for`), one budget under the local cap, so an
in-process run and a worker run cannot both take one physical card under
``gpu=1``. A card with no UUID or no match keeps a slot of its own.

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

    ``(worker.id, device)``, except for a card a same-host worker shares
    with this process (``worker.shared``, see
    ``WorkerRegistry._shared_devices``): that card's slot is this
    process's, ``("local", <its device here>)``, with this process's
    budget and cap.
    """
    shared = getattr(worker, "shared", None) or {}
    local = shared.get(device)
    if local is not None:
        return (LOCAL_WORKER_ID, local)
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


def _free(limits: SlotLimits, slot: SlotKey,
          taken: Mapping[SlotKey, int]) -> bool:
    return taken.get(slot, 0) < limits.for_slot(slot)


def _all_free(limits: SlotLimits, slots: tuple[SlotKey, ...],
              taken: Mapping[SlotKey, int]) -> bool:
    return all(_free(limits, slot, taken) for slot in slots)


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

    A scheduler may also offer ``reservation(entry)``: the slots a refused
    front-of-line run needs, which ``_pump`` then keeps every other run
    off until it starts (see the module docstring). Without it nothing is
    reserved.
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

    def reservation(self, entry: QueuedEntry) -> tuple[SlotKey, ...] | None:
        """The slots to reserve while *entry* waits: its set's, or None."""
        _, devices = entry_target(entry.options)
        slots = local_set_slots(entry.queue_key, devices)
        return slots if len(slots) > 1 else None




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
      With a device set, this process never qualifies: a worker's run
      subprocess sees the set renumbered from ``cuda:0`` through
      ``CUDA_VISIBLE_DEVICES``, so a node pinned to ``cuda:1`` means the
      set's second card; in this process it would mean physical card 1.
      An ``any`` set run therefore waits for a worker that offers the set.

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

    def reservation(self, entry: QueuedEntry) -> tuple[SlotKey, ...] | None:
        """The slots to reserve while *entry* waits at the front, or None.

        Only a device-set run on this process or on a named worker that
        accepts runs and offers the whole set: those are the runs refused
        for lack of slots alone, and the slots they need are known. A
        single-device run needs no reservation (its FIFO is its line), an
        ``any`` set run's slots are not known until it is placed, and a run
        waiting for a worker to come back or undrain must not hold cards
        idle for it.
        """
        worker_id, devices = entry_target(entry.options)
        if worker_id == TARGET_LOCAL:
            return self._local.reservation(entry)
        if worker_id == TARGET_ANY or not devices or len(devices) < 2:
            return None
        worker = self._registry.get(worker_id)
        if worker is None or not worker.accepts_runs \
                or not all(worker.offers(d) for d in devices):
            return None
        return tuple(slot_for(worker, d) for d in devices)

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
        if not self.can_serve_any(requested, devices, connected_only=True):
            wanted = ", ".join(devices) if devices else requested
            return f"waiting for a worker that offers {wanted}"
        return None

    def can_serve_any(self, requested: str, devices: list[str] | None,
                      *, connected_only: bool = False) -> bool:
        """Whether this process or a worker could ever run it.

        This process counts only for a run without a device set (see the
        class docstring: a set is renumbered on a worker, not here). By
        default every worker this server has seen counts, connected or
        not (#777): a run submitted while the only worker offering its
        device is away waits for it to reconnect rather than being refused.
        *connected_only* asks about the workers connected now, which is
        what a waiting run's reason describes.
        """
        if not devices and _serves(self._local_devices(), requested, None):
            return True
        return any((w.online or not connected_only)
                   and _serves(w.device_values(), requested, devices)
                   for w in self._registry.workers())

    def _place_here(self, requested: str, devices: list[str] | None,
                    taken: Mapping[SlotKey, int]) -> Placement | None:
        if devices:  # renumbered only on a worker; see the class docstring
            return None
        offered = self._local_devices()
        for device in candidate_devices(offered, requested):
            slot: SlotKey = (LOCAL_WORKER_ID, device)
            if _free(self._limits, slot, taken):
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
            if _free(self._limits, slot, taken):
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
