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
A CUDA set a worker would renumber never runs in this process
(:func:`renumbered_on_a_worker`, :class:`WorkerScheduler`).

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

from .mounts import holds_mount
from .transfer import refusal as transfer_refusal
from .transfer import transferable
from .run_store import WORKER_LOCAL
from .worker_protocol import (
    CAPABILITY_TRANSFER,
    TARGET_ANY,
    TARGET_LOCAL,
    canonical_device,
)

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
    # An entry may also carry ``needs`` (``run_needs.needs``, #782) for a
    # run that targets a worker; read through :func:`entry_mount`.


def entry_mount(entry: Any) -> str | None:
    """The mount label *entry*'s run needs (``run_needs``), or None (#782)."""
    needs = getattr(entry, "needs", None)
    mount = needs.get("mount") if isinstance(needs, Mapping) else None
    return mount if isinstance(mount, str) and mount else None


def entry_transfers(entry: Any) -> bool:
    """Whether *entry*'s run may use the ``transfer`` backend (#785):
    ``needs["transfer"]``, decided at submit by ``core.transfer``."""
    return transferable(getattr(entry, "needs", None))


def speaks_transfer(worker: Any) -> bool:
    """Whether *worker*'s agent listed the ``transfer`` capability in its
    ``hello`` (#781's rule: an additive feature is used only when both ends
    name it)."""
    return CAPABILITY_TRANSFER in (getattr(worker, "capabilities", None) or ())


def reaches(worker: Any, entry: Any) -> bool:
    """Whether *worker* can serve *entry*'s files: the run needs none, the
    worker holds the mount (#782), or the run can use the transfer backend
    and the worker speaks it (#785)."""
    mount = entry_mount(entry)
    return (mount is None or holds_mount(worker, mount)
            or (entry_transfers(entry) and speaks_transfer(worker)))


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




def renumbered_on_a_worker(devices: list[str] | None) -> bool:
    """Whether a worker would renumber *devices* for the run.

    A worker's run subprocess sees a CUDA set through
    ``CUDA_VISIBLE_DEVICES``, numbered from ``cuda:0`` in the set's order;
    this process does not renumber. The two agree only for no CUDA set or
    the one card ``cuda:0``: a set of several cards, or one other card
    (``["cuda:1"]``, which a worker calls ``cuda:0``), would have a node
    pinned to ``cuda:N`` address different cards here and there. ``cpu``
    and ``mps`` sets are not renumbered.
    """
    cuda: list[str] = []
    for device in devices or ():
        try:
            device = canonical_device(device)
        except ValueError:
            continue
        if device.split(":")[0] == "cuda":
            cuda.append(device)
    return len(cuda) > 1 or (len(cuda) == 1 and cuda[0] != "cuda:0")


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
      A device set a worker would renumber (:func:`renumbered_on_a_worker`:
      several CUDA cards, or one other than ``cuda:0``) never runs here: a
      worker's run subprocess sees it numbered from ``cuda:0`` through
      ``CUDA_VISIBLE_DEVICES``, so a node pinned to ``cuda:1`` means the
      set's second card there and physical card 1 here. Such a run waits
      for a worker that offers the set. Any other set qualifies here when
      this process has every device of it, and executes on its first
      device, as a local run with a set does.

    Every worker slot is ``(worker_id, device)`` capped by
    ``limits.for_slot(slot)`` (a same-host worker's is this process's slot,
    see :func:`slot_for`); a device set takes one slot per device, here or
    on a worker, and starts only when all of them are free.

    Storage (#782): a run whose ``needs`` name a mount (``run_needs``) goes
    to a worker that holds it (``mounts.holds_mount``: it declared the
    label, or it is a same-host worker and the label is this server's),
    or, when the run can use the ``transfer`` backend (#785,
    ``needs["transfer"]``), to one that does not: the files travel over
    HTTP instead. The mount is preferred: ``any`` tries the workers that
    hold it first, then the others through the transfer backend, then this
    process (which always holds its own). A run the transfer backend cannot
    serve goes only where the mount is (:func:`reaches`).

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
        mount = entry_mount(entry)
        if worker_id != TARGET_ANY:
            worker = self._registry.get(worker_id)
            if worker is None or not worker.accepts_runs:
                return None
            if not reaches(worker, entry):
                return None
            return self._place_on(worker, requested, devices, taken)
        workers = [w for w in self._registry.workers() if w.accepts_runs
                   # A worker that lacks node types this server has, or
                   # reported no catalog, takes only runs that name it (#781).
                   and not _skipped_by_any(w)]
        # The mount first (#785): a worker that holds it, then one that
        # gets the files through the transfer backend.
        ordered = [w for w in workers
                   if mount is None or holds_mount(w, mount)]
        if mount is not None and entry_transfers(entry):
            ordered += [w for w in workers
                        if not holds_mount(w, mount) and speaks_transfer(w)]
        for worker in ordered:
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
        if not reaches(worker, entry):
            return None
        return tuple(slot_for(worker, d) for d in devices)

    def explain(self, entry: QueuedEntry) -> str | None:
        """Why *entry* is waiting, when it is not just a busy slot."""
        worker_id, devices = entry_target(entry.options)
        if worker_id == TARGET_LOCAL:
            return None
        requested = str(entry.options.get("device") or "cpu")
        mount = entry_mount(entry)
        if worker_id != TARGET_ANY:
            worker = self._registry.get(worker_id)
            if worker is None or not worker.online:
                return f"waiting for worker {worker_id} to reconnect"
            if worker.draining:
                return f"worker {worker_id} is draining"
            if not reaches(worker, entry):
                return no_mount(worker_id, mount or "",
                                transfer_refusal(getattr(entry, "needs",
                                                         None)))
            return None
        wanted = ", ".join(devices) if devices else requested
        if not self.can_serve_any(requested, devices, connected_only=True,
                                  mount=mount,
                                  transfer=entry_transfers(entry)):
            if mount is not None and not entry_transfers(entry):
                return (f"waiting for a worker that offers {wanted} and "
                        f"has the `{mount}` mount")
            return f"waiting for a worker that offers {wanted}"
        if not self.can_serve_any(requested, devices, connected_only=True,
                                  mount=mount, usable_only=True,
                                  transfer=entry_transfers(entry)):
            # The workers the first check counted and this one did not:
            # the same predicate as ``can_serve_any`` (``reaches`` is its
            # mount/transfer rule), so the list is never empty.
            skipped = sorted(
                w.name for w in self._registry.workers()
                if w.online and _skipped_by_any(w) and reaches(w, entry)
                and _serves(w.device_values(), requested, devices))
            return (f"waiting for a worker that offers {wanted}: "
                    f"{', '.join(skipped)} "
                    f"{'is' if len(skipped) == 1 else 'are'} skipped by "
                    "`any` (node types missing, or not reported)")
        return None

    def can_serve_any(self, requested: str, devices: list[str] | None,
                      *, connected_only: bool = False,
                      mount: str | None = None,
                      usable_only: bool = False,
                      transfer: bool = False) -> bool:
        """Whether this process or a worker could ever run it.

        This process does not count for a device set a worker would
        renumber (see the class docstring). By default every worker this server has seen counts, connected or
        not (#777): a run submitted while the only worker offering its
        device is away waits for it to reconnect rather than being refused.
        *connected_only* asks about the workers connected now, which is
        what a waiting run's reason describes. With *mount* (#782), a
        worker counts only when it holds that mount, or, with *transfer*
        (#785), when it speaks the transfer backend; this process always
        does. *usable_only* leaves out the workers ``any`` skips (#781:
        one that lacks node types, or reported no catalog).
        """
        if not renumbered_on_a_worker(devices) \
                and _serves(self._local_devices(), requested, devices):
            return True
        return any((w.online or not connected_only)
                   and (mount is None or holds_mount(w, mount)
                        or (transfer and speaks_transfer(w)))
                   and not (usable_only and _skipped_by_any(w))
                   and _serves(w.device_values(), requested, devices)
                   for w in self._registry.workers())

    def _place_here(self, requested: str, devices: list[str] | None,
                    taken: Mapping[SlotKey, int]) -> Placement | None:
        if renumbered_on_a_worker(devices):  # see the class docstring
            return None
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


def no_mount(worker_id: str, mount: str, why_not_transfer: str | None = None
             ) -> str:
    """Why a run that needs *mount* cannot go to *worker_id* (#782): the
    submit refusal and the waiting reason, one sentence. *why_not_transfer*
    is why the transfer backend (#785) cannot serve it instead."""
    message = (f"worker {worker_id} has no `{mount}` mount; add "
               f"`--mount {mount}=...` to its agent")
    if why_not_transfer:
        message += (f", or make the run fit the transfer backend: "
                    f"{why_not_transfer}")
    return message


def _skipped_by_any(worker: Any) -> bool:
    return bool(getattr(worker, "skipped_by_any", False))


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
