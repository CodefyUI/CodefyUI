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

One device, several slots (#773, #842): every slot stays its offerer's
own ``(worker, device)``, and the slots that are one physical device
are grouped (``WorkerRegistry.card_slots``): a CUDA card by the UUID its
offerers report, whatever number each gives it, and a same-host worker's
``cpu`` and ``mps:0`` with this process's (a worker whose
``hello.machine_id`` equals this process's runs on this machine). Two
agents on one GPU server, or a same-host agent and this process, then
admit as one card (:class:`CardBudget`): the runs on every slot of it
count against each of them, under the smallest cap among the offerers
that are online (or hold a run there), so an in-process run and worker
runs never share one card past ``gpu=1``. A card's memory ledger is read
across its slots, and it admits by memory only while every online worker
offering it has it ``shared`` (the stricter policy wins). A CUDA card
without a UUID keeps a slot of its own; a device no other slot names
admits exactly as before.

Memory admission (#788): a worker's card whose policy is ``shared``
(``WorkerRegistry.device_policy``) is admitted by memory instead of by
count: a run with a budget (``entry.memory_budget``) takes it while the
budget fits the card's headroom in the
:class:`~.memory_admission.MemoryLedger` and no run without a budget holds
it; a run without a budget never takes it. A run without a budget on its
slot keeps the count rule, so it waits for the budgeted runs to finish
(and they for it). Only a worker that speaks ``memory_budget`` gets such
runs: its agent enforces the budget. See ``core.memory_admission``.

This module imports nothing from ``run_service``: the scheduler sees a
waiting run through :class:`QueuedEntry` and the caps through
:class:`SlotLimits`, both structural, so ``run_service`` can depend on it
without a cycle.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from . import memory_admission as mem
from .coded_text import freeze, thaw
from .mounts import holds_mount
from .transfer import refusal as transfer_refusal
from .transfer import transferable
from .run_store import WORKER_LOCAL
from .worker_protocol import (
    CAPABILITY_MEMORY_BUDGET,
    CAPABILITY_S3,
    CAPABILITY_TRANSFER,
    POOL_ONE_SHOT,
    TARGET_ANY,
    TARGET_LOCAL,
    canonical_device,
)

logger = logging.getLogger(__name__)

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
    #: Memory admission (#788): the run's budget in bytes on each of
    #: ``budget_devices`` (the devices of ``devices`` it was admitted to by
    #: memory, as the worker numbers them), whose slots are
    #: ``budget_slots``. None and empty: admitted by count alone.
    memory_budget: int | None = None
    budget_devices: tuple[str, ...] = ()
    budget_slots: tuple[SlotKey, ...] = ()


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


#: A slot count that means "reserved": ``RunService._admission_view`` fills
#: the slots of an earlier device-set run's reservation with it, so every
#: later run sees them full (#773). Memory admission tells it apart from a
#: run holding the card (#788).
RESERVED_SLOT = 1 << 30


def entry_budget(entry: Any) -> int | None:
    """The memory budget in bytes *entry*'s run was given at submit (#788),
    or None."""
    budget = getattr(entry, "memory_budget", None)
    nbytes = getattr(budget, "bytes", budget)
    return nbytes if isinstance(nbytes, int) and nbytes > 0 else None


def target_wait(options: Mapping[str, Any], *, default: bool) -> bool:
    """``target.wait`` (#788), or *default* when not given."""
    target = options.get("target") if isinstance(options, Mapping) else None
    wait = target.get("wait") if isinstance(target, Mapping) else None
    return default if wait is None else bool(wait)


def entry_mount(entry: Any) -> str | None:
    """The mount label *entry*'s run needs (``run_needs``), or None (#782)."""
    needs = getattr(entry, "needs", None)
    mount = needs.get("mount") if isinstance(needs, Mapping) else None
    return mount if isinstance(mount, str) and mount else None


def entry_transfers(entry: Any) -> bool:
    """Whether *entry*'s run may use the ``transfer`` backend (#785):
    ``needs["transfer"]``, decided at submit by ``core.transfer``."""
    return transferable(getattr(entry, "needs", None))


def no_mount_backend(worker: Any) -> str | None:
    """The storage backend a run that needs the mount would use on
    *worker*, which does not hold it: ``s3`` (#791) when this server has it
    on and the agent listed it, else ``transfer`` (#785) when the agent
    listed that (#781's rule: an additive feature is used only when both
    ends name it) and it is on here, else None."""
    from ..config import settings
    from . import s3

    capabilities = getattr(worker, "capabilities", None) or ()
    if CAPABILITY_S3 in capabilities and s3.enabled():
        return CAPABILITY_S3
    if CAPABILITY_TRANSFER in capabilities and settings.TRANSFER_ENABLED:
        return CAPABILITY_TRANSFER
    return None


def ships_files(worker: Any) -> bool:
    """Whether *worker* can take a run without the mount: its agent speaks
    a backend that ships the files (:func:`no_mount_backend`)."""
    return no_mount_backend(worker) is not None


def reaches(worker: Any, entry: Any) -> bool:
    """Whether *worker* can serve *entry*'s files: the run needs none, the
    worker holds the mount (#782), or the run can use the transfer backend
    and the worker speaks it (#785)."""
    mount = entry_mount(entry)
    return (mount is None or holds_mount(worker, mount)
            or (entry_transfers(entry) and ships_files(worker)))


class SlotLimits(Protocol):
    """The per-slot concurrency caps (``run_service.QueueLimits``)."""

    def for_slot(self, slot: SlotKey) -> int: ...


def slot_for(worker: Any, device: str) -> SlotKey:
    """The slot a run on *device* of *worker* takes: ``(worker.id,
    device)``. A device several slots name (a same-host worker's card that
    is this process's, one card two workers offer) is grouped by
    :class:`CardBudget`, never aliased (#842)."""
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


class SlotBudget:
    """Count admission per slot, the 3.0 rule: a slot admits while fewer
    than its cap hold it. :class:`CardBudget` counts per physical card."""

    def __init__(self, limits: SlotLimits) -> None:
        self.limits = limits

    def held(self, slot: SlotKey, taken: Mapping[SlotKey, int]) -> int:
        """How many runs hold *slot* (a reserved slot reads
        :data:`RESERVED_SLOT` or more)."""
        return taken.get(slot, 0)

    def limit(self, slot: SlotKey, taken: Mapping[SlotKey, int]
              ) -> tuple[int, SlotKey]:
        """``(cap, the slot whose cap it is)`` of *slot*."""
        return self.limits.for_slot(slot), slot

    def free(self, slot: SlotKey, taken: Mapping[SlotKey, int]) -> bool:
        return self.held(slot, taken) < self.limit(slot, taken)[0]

    def all_free(self, slots: tuple[SlotKey, ...],
                 taken: Mapping[SlotKey, int]) -> bool:
        return all(self.free(slot, taken) for slot in slots)


class CardBudget(SlotBudget):
    """Count admission per physical device (#773, #842).

    A slot's device is every slot ``registry.card_slots`` groups with it.
    What it holds is the runs on all of them; its cap is the smallest cap
    among the slot asked about and the others whose offerer is online
    (``registry.slot_online``) or holds a run there, so an offerer that
    went away does not keep the card's cap down. A same-host worker's slot
    without an override of its own takes this server's cap for the device
    (:meth:`slot_cap`), so ``cuda:0=2`` raises a card a same-host worker
    shares, as before #842. A registry that does not group leaves every
    slot its own; one whose grouping FAILS makes the slot read busy
    (logged once), never free.
    """

    def __init__(self, limits: SlotLimits, registry: Any) -> None:
        super().__init__(limits)
        self._registry = registry
        #: The last grouping failure logged, so a broken index is logged
        #: once per distinct error rather than on every placement.
        self._failure: str | None = None

    def _group(self, slot: SlotKey) -> tuple[SlotKey, ...] | None:
        """Every slot on *slot*'s device, or None when grouping failed."""
        cards = getattr(self._registry, "card_slots", None)
        if cards is None:
            return (slot,)
        try:
            group = tuple(cards(slot)) or (slot,)
        except Exception as exc:  # noqa: BLE001 -- fail safe, see held()
            failure = f"{type(exc).__name__}: {exc}"
            if failure != self._failure:
                self._failure = failure
                logger.exception("could not group slot %s/%s with the other "
                                 "offerers of its device; it reads busy "
                                 "until the grouping works", *slot)
            return None
        self._failure = None
        return group

    def card(self, slot: SlotKey) -> tuple[SlotKey, ...]:
        """Every slot on *slot*'s device, *slot* included (just *slot*
        when the grouping fails: for descriptions, never for admission)."""
        return self._group(slot) or (slot,)

    def online(self, slot: SlotKey) -> bool:
        """Whether *slot*'s offerer can take a run now
        (``registry.slot_online``; every slot without a registry)."""
        online = getattr(self._registry, "slot_online", None)
        return True if online is None else bool(online(slot))

    def held(self, slot: SlotKey, taken: Mapping[SlotKey, int]) -> int:
        group = self._group(slot)
        if group is None:
            return RESERVED_SLOT  # unknown: busy, never free
        return sum(taken.get(s, 0) for s in group)

    def slot_cap(self, slot: SlotKey) -> int:
        """*slot*'s own cap: ``limits.for_slot``, except that a same-host
        worker's device with no ``<worker>/<device>`` override takes this
        server's cap for the same device (it IS that device)."""
        override_for = getattr(self.limits, "override_for", None)
        if slot[0] != LOCAL_WORKER_ID and override_for is not None \
                and override_for(slot) is None:
            get = getattr(self._registry, "get", None)
            worker = None if get is None else get(slot[0])
            local = (getattr(worker, "shared", None) or {}).get(slot[1])
            if local is not None:
                return self.limits.for_slot((LOCAL_WORKER_ID, local))
        return self.limits.for_slot(slot)

    def limit(self, slot: SlotKey, taken: Mapping[SlotKey, int]
              ) -> tuple[int, SlotKey]:
        counted = [s for s in self.card(slot) if s == slot
                   or taken.get(s, 0) > 0 or self.online(s)]
        # The smallest cap; on a tie, the slot asked about.
        by = min(counted, key=lambda s: (self.slot_cap(s), s != slot))
        return self.slot_cap(by), by


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

    def __init__(self, limits: "SlotLimits | SlotBudget") -> None:
        self._budget = (limits if isinstance(limits, SlotBudget)
                        else SlotBudget(limits))

    def place(self, entry: QueuedEntry,
              taken: Mapping[SlotKey, int]) -> Placement | None:
        queue_key = entry.queue_key
        _, devices = entry_target(entry.options)
        slots = local_set_slots(queue_key, devices)
        if not self._budget.all_free(slots, taken):
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


def entry_pool(options: Mapping[str, Any]) -> str | None:
    """``target.pool`` of a run's normalised options (#790), or None."""
    target = options.get("target") if isinstance(options, Mapping) else None
    pool = target.get("pool") if isinstance(target, Mapping) else None
    return pool if isinstance(pool, str) and pool else None


def _one_shot(worker: Any) -> bool:
    return bool(getattr(worker, "one_shot", False))


def _takes(worker: Any, entry: Any) -> bool:
    """The one-shot rules (#790): a regular worker takes any run; a
    one-shot one only its first, and only its ``--run-id`` if it named one."""
    takes = getattr(worker, "one_shot_takes", None)
    return True if takes is None else bool(takes(entry.run_id))


def entry_target(options: Mapping[str, Any]) -> tuple[str, list[str] | None]:
    """``(worker, devices)`` of a run's normalised options; ``local`` without
    a target."""
    target = options.get("target") if isinstance(options, Mapping) else None
    if not isinstance(target, Mapping):
        return TARGET_LOCAL, None
    return (target.get("worker") or TARGET_LOCAL), target.get("devices")


#: Codes of :class:`QueueReason` (#841). The Runs list words each one in
#: the viewer's language from its ``params``; an unknown code falls back to
#: the English ``text``.
REASON_WORKER_RECONNECT = "worker_reconnect"
REASON_WORKER_DRAINING = "worker_draining"
REASON_WORKER_NO_MOUNT = "worker_no_mount"
REASON_ANY_NO_WORKER = "any_no_worker"
REASON_ANY_NO_WORKER_MOUNT = "any_no_worker_mount"
REASON_ANY_UNAVAILABLE = "any_unavailable"
#: Wave D (#788, #790, #794).
REASON_ONE_SHOT_START = "one_shot_start"
REASON_ONE_SHOT_TAKEN = "one_shot_taken"
REASON_ONE_SHOT_RUN_ID = "one_shot_run_id"
REASON_ONE_SHOT_POOL = "one_shot_pool"
REASON_MEMORY_NO_BUDGET = "memory_no_budget"
REASON_MEMORY_WAIT = "memory_wait"
#: Only the raise of an accepted estimate (#794); beside another reason
#: the raise rides in that reason's ``params.raised``.
REASON_BUDGET_RAISED = "budget_raised"


@dataclass(frozen=True, eq=False)
class QueueReason:
    """Why a waiting run cannot start, beyond a busy slot (#772, #841).

    Immutable: *params* is frozen (``coded_text.freeze``) on construction,
    and ``eq=False`` keeps it hashable by identity; :meth:`params_dict`
    is the plain copy the API sends.
    """

    code: str
    params: Mapping[str, Any]
    #: The English sentence: ``queue_reason`` in the API and the CLI.
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", freeze(self.params))

    def params_dict(self) -> dict[str, Any]:
        return thaw(self.params)


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
    ``limits.for_slot(slot)``, counted per physical device
    (:class:`CardBudget`); a device set takes one slot per device, here or
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

    One-shot workers (#790, ``Worker.one_shot``): ``any`` never places a
    run on one, nor counts one in :meth:`can_serve_any`. A run naming one,
    or ``target.pool == "one-shot"`` (any one-shot worker, never this
    process), goes there only while it has not taken a run, and, for an
    agent started with ``--run-id``, only when it is that run
    (``Worker.one_shot_takes``).

    :meth:`explain` says why a run that :meth:`place` refused cannot start
    for a reason other than a busy slot (the Runs list shows it).
    """

    def __init__(self, limits: SlotLimits, registry: Any,
                 local_devices: Callable[[], list[str]],
                 ledger: "mem.MemoryLedger | None" = None) -> None:
        self._registry = registry
        # Counts and caps per physical device (#773, #842).
        self._budget = CardBudget(limits, registry)
        self._local_devices = local_devices
        self._local = LocalScheduler(self._budget)
        #: The budgets promised per slot (#788); ``RunService`` reserves,
        #: starts and releases them, this class reads them.
        self.ledger = mem.MemoryLedger() if ledger is None else ledger

    def card_slots(self, slot: SlotKey) -> tuple[SlotKey, ...]:
        """Every slot on *slot*'s physical device (#842)."""
        return self._budget.card(slot)

    def card_limit(self, slot: SlotKey, taken: Mapping[SlotKey, int]
                   ) -> tuple[int, SlotKey]:
        """``(cap, the slot whose cap it is)`` of *slot*'s device (#842)."""
        return self._budget.limit(slot, taken)

    def card_held(self, slot: SlotKey, taken: Mapping[SlotKey, int]) -> int:
        """How many runs hold *slot*'s device, on any of its slots."""
        return self._budget.held(slot, taken)

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
            if not reaches(worker, entry) or not _takes(worker, entry):
                return None
            return self._place_on(worker, requested, devices, taken,
                                  entry_budget(entry))
        budget = entry_budget(entry)
        if entry_pool(entry.options) == POOL_ONE_SHOT:
            # The one-shot pool (#790): a one-shot worker that has not
            # taken its run, never this process, never a regular worker;
            # like ``any``, not one that lacks node types this server has
            # (or reported none, #781): the run names no worker either.
            for worker in self._registry.workers():
                if _one_shot(worker) and worker.accepts_runs \
                        and not _skipped_by_any(worker) \
                        and _takes(worker, entry) and reaches(worker, entry):
                    placement = self._place_on(worker, requested, devices,
                                               taken, budget)
                    if placement is not None:
                        return placement
            return None
        workers = [w for w in self._registry.workers() if w.accepts_runs
                   # A worker that lacks node types this server has, or
                   # reported no catalog, takes only runs that name it (#781).
                   and not _skipped_by_any(w)
                   # A one-shot worker takes only runs that name it or its
                   # pool (#790).
                   and not _one_shot(w)]
        # The mount first (#785): a worker that holds it, then one that
        # gets the files through the transfer backend.
        ordered = [w for w in workers
                   if mount is None or holds_mount(w, mount)]
        if mount is not None and entry_transfers(entry):
            ordered += [w for w in workers
                        if not holds_mount(w, mount) and ships_files(w)]
        for worker in ordered:
            placement = self._place_on(worker, requested, devices, taken,
                                       budget)
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
                or not all(worker.offers(d) for d in devices) \
                or not _takes(worker, entry):
            return None
        if not reaches(worker, entry):
            return None
        return tuple(slot_for(worker, d) for d in devices)

    def explain(self, entry: QueuedEntry,
                taken: Mapping[SlotKey, int] | None = None) -> str | None:
        """Why *entry* is waiting, when it is not just a busy slot: the
        English sentence of :meth:`explain_reason`."""
        reason = self.explain_reason(entry, taken)
        return None if reason is None else reason.text

    def explain_reason(self, entry: QueuedEntry,
                       taken: Mapping[SlotKey, int] | None = None
                       ) -> QueueReason | None:
        """Why *entry* is waiting, as a code, its parameters and the
        English sentence (#841): the Runs list words the code in the
        viewer's language, the API and CLI keep the sentence. *taken* is
        the slot counts it is judged against (memory, #788)."""
        taken = taken or {}
        worker_id, devices = entry_target(entry.options)
        if worker_id == TARGET_LOCAL:
            return None
        requested = str(entry.options.get("device") or "cpu")
        mount = entry_mount(entry)
        if worker_id != TARGET_ANY:
            worker = self._registry.get(worker_id)
            if worker is not None and _one_shot(worker) \
                    and not _takes(worker, entry):
                return one_shot_busy(worker)
            if worker is None or not worker.online:
                if worker is not None and _one_shot(worker):
                    return QueueReason(
                        REASON_ONE_SHOT_START, {"worker": worker_id},
                        f"waiting for one-shot worker {worker_id} to "
                        "start (cdui agent start --one-shot)")
                return QueueReason(
                    REASON_WORKER_RECONNECT, {"worker": worker_id},
                    f"waiting for worker {worker_id} to reconnect")
            if worker.draining:
                return QueueReason(
                    REASON_WORKER_DRAINING, {"worker": worker_id},
                    f"worker {worker_id} is draining")
            if not reaches(worker, entry):
                why_not = transfer_refusal(getattr(entry, "needs", None))
                return QueueReason(
                    REASON_WORKER_NO_MOUNT,
                    {"worker": worker_id, "mount": mount or "",
                     "transfer": why_not or None},
                    no_mount(worker_id, mount or "", why_not))
            return self._memory_wait(entry.options, entry_budget(entry),
                                     taken)
        wanted = ", ".join(devices) if devices else requested
        if entry_pool(entry.options) == POOL_ONE_SHOT:
            return QueueReason(
                REASON_ONE_SHOT_POOL, {"devices": wanted},
                f"waiting for a one-shot worker that offers {wanted} "
                "(cdui agent start --one-shot)")
        if not self.can_serve_any(requested, devices, connected_only=True,
                                  mount=mount,
                                  transfer=entry_transfers(entry)):
            if mount is not None and not entry_transfers(entry):
                return QueueReason(
                    REASON_ANY_NO_WORKER_MOUNT,
                    {"devices": wanted, "mount": mount},
                    f"waiting for a worker that offers {wanted} and "
                    f"has the `{mount}` mount")
            return QueueReason(
                REASON_ANY_NO_WORKER, {"devices": wanted},
                f"waiting for a worker that offers {wanted}")
        if not self.can_serve_any(requested, devices, connected_only=True,
                                  mount=mount, usable_only=True,
                                  accepting_only=True,
                                  transfer=entry_transfers(entry)):
            # The workers the first check counted and this one did not:
            # the same predicate as ``can_serve_any`` (``reaches`` is its
            # mount/transfer rule), so the two lists are never both empty.
            # Draining (#841): an `any` run behind a draining worker said
            # nothing at all, while a run naming it said "is draining".
            able = [w for w in self._registry.workers()
                    if w.online and reaches(w, entry)
                    and _serves(w.device_values(), requested, devices)]
            draining = sorted(w.name for w in able
                              if _draining(w) and not _skipped_by_any(w))
            skipped = sorted(w.name for w in able if _skipped_by_any(w))
            parts = []
            if draining:
                parts.append(f"{', '.join(draining)} "
                             f"{'is' if len(draining) == 1 else 'are'} "
                             "draining")
            if skipped:
                parts.append(f"{', '.join(skipped)} "
                             f"{'is' if len(skipped) == 1 else 'are'} "
                             "skipped by `any` (node types missing, or not "
                             "reported)")
            return QueueReason(
                REASON_ANY_UNAVAILABLE,
                {"devices": wanted, "draining": draining, "skipped": skipped},
                f"waiting for a worker that offers {wanted}: "
                + "; ".join(parts))
        return None

    def can_serve_any(self, requested: str, devices: list[str] | None,
                      *, connected_only: bool = False,
                      mount: str | None = None,
                      usable_only: bool = False,
                      accepting_only: bool = False,
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
        one that lacks node types, or reported no catalog);
        *accepting_only* the draining ones (#841), which take no new run.
        """
        if not renumbered_on_a_worker(devices) \
                and _serves(self._local_devices(), requested, devices):
            return True
        return any((w.online or not connected_only)
                   # ``any`` never places a run on a one-shot worker (#790).
                   and not _one_shot(w)
                   and (mount is None or holds_mount(w, mount)
                        or (transfer and ships_files(w)))
                   and not (usable_only and _skipped_by_any(w))
                   and not (accepting_only and _draining(w))
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
            if self._budget.all_free(slots, taken):
                return Placement(LOCAL_WORKER_ID, (first,), slots)
            return None
        for device in candidate_devices(offered, requested):
            slot: SlotKey = (LOCAL_WORKER_ID, device)
            if self._budget.free(slot, taken):
                return Placement(LOCAL_WORKER_ID, (device,), (slot,))
        return None

    def _place_on(self, worker: Any, requested: str,
                  devices: list[str] | None,
                  taken: Mapping[SlotKey, int],
                  budget: int | None = None) -> Placement | None:
        if devices:
            if not all(worker.offers(d) for d in devices):
                return None
            chosen = tuple(canonical_device(d) for d in devices)
            if not all(self._admits(worker, d, taken, budget)
                       for d in chosen):
                return None
            placed = tuple(devices)
        else:
            chosen = next(((d,) for d in candidate_devices(
                worker.device_values(), requested)
                if self._admits(worker, d, taken, budget)), ())
            if not chosen:
                return None
            placed = chosen
        slots = tuple(slot_for(worker, d) for d in chosen)
        by_memory = tuple(d for d in chosen if self.shared(worker, d))
        if not by_memory:
            return Placement(worker.id, placed, slots)
        return Placement(worker.id, placed, slots, memory_budget=budget,
                         budget_devices=by_memory,
                         budget_slots=tuple(slot_for(worker, d)
                                            for d in by_memory))

    # ── memory admission (#788) ───────────────────────────────────────────

    def shared(self, worker: Any, device: str) -> bool:
        """Whether *worker*'s *device* admits by memory: it does
        (:meth:`_by_memory`), and so does every other online worker that
        offers the same physical card (#842): the stricter policy wins, so
        a card one offerer keeps ``exclusive`` admits by count. This
        process's runs on the card are holders without a budget, as
        before."""
        if not self._by_memory(worker, device):
            return False
        try:
            device = canonical_device(device)
        except ValueError:  # pragma: no cover - _by_memory said it parses
            return False
        for peer_id, peer_device in self._budget.card((worker.id, device)):
            if peer_id in (LOCAL_WORKER_ID, worker.id):
                continue
            if not self._budget.online((peer_id, peer_device)):
                continue
            peer = self._registry.get(peer_id)
            if peer is not None and not self._by_memory(peer, peer_device):
                return False
        return True

    def _by_memory(self, worker: Any, device: str) -> bool:
        """Whether *worker*'s *device* is set to admit by memory: its
        policy is ``shared``, it is a CUDA or MPS device, and the agent
        enforces budgets (the ``memory_budget`` capability)."""
        policy = getattr(self._registry, "device_policy", None)
        if policy is None:
            return False
        try:
            device = canonical_device(device)
        except ValueError:
            return False
        return (device.split(":")[0] in mem.SHAREABLE_KINDS
                and CAPABILITY_MEMORY_BUDGET
                in (getattr(worker, "capabilities", None) or ())
                and policy(worker.id, device) == mem.POLICY_SHARED)

    def room(self, worker: Any, device: str
             ) -> tuple[int | None, int | None, "mem.SlotAccount"]:
        """``(total, headroom, account)`` of *worker*'s *device*: its total
        bytes as last reported, what a new footprint may take
        (``memory_admission.headroom``), and what the ledger holds on its
        slot."""
        report = _report(worker, device)
        total = None if report is None else report.total_bytes
        free = None if report is None else report.free_bytes
        # One account per physical card (#842).
        account = self.ledger.account(*self._budget.card(
            slot_for(worker, device)))
        return total, mem.headroom(total, free, account), account

    def admission(self, worker: Any, device: str) -> dict[str, Any]:
        """A shared card's ledger as ``GET /api/workers`` shows it."""
        report = _report(worker, device)
        total, room, account = self.room(worker, device)
        return {
            "by_memory": self.shared(worker, device),
            "total_bytes": total,
            "free_bytes": None if report is None else report.free_bytes,
            "reserved_bytes": account.reserved,
            "running_bytes": account.running,
            "headroom_bytes": room,
            "context_bytes": mem.context_bytes(device),
            "runs": list(account.runs),
        }

    def card_state(self, worker: Any, device: str,
                   taken: Mapping[SlotKey, int], budget: int
                   ) -> tuple[bool, bool, str | None]:
        """``(fits_ever, fits_now, why_not)`` of a budgeted run on a shared
        card: the one judgement placement, the submit refusal and the
        waiting reason share. *why_not* names the real blocker: no total,
        a total too small, a holder without a budget, or too little room."""
        need = mem.footprint(budget, device)
        total, room, account = self.room(worker, device)
        if total is None:
            return False, False, mem.no_total_reported(device, worker.id)
        if total < need:
            return False, False, mem.total_too_small(device, worker.id, total)
        slot = slot_for(worker, device)
        held = self._budget.held(slot, taken)
        if held >= RESERVED_SLOT:
            return True, False, mem.reserved_for_a_set(device, worker.id)
        # A holder without a budget (a same-host card's in-process run, a
        # run from before the card became shared, a reserved set) has the
        # card to itself until it ends.
        if held > self.ledger.holders(*self._budget.card(slot)):
            return True, False, mem.held_without_budget(device, worker.id)
        if room is None or room < need:
            return True, False, mem.does_not_fit(device, worker.id, room,
                                                 account.runs)
        return True, True, None

    def _admits(self, worker: Any, device: str,
                taken: Mapping[SlotKey, int], budget: int | None) -> bool:
        if not self.shared(worker, device):
            return self._budget.free(slot_for(worker, device), taken)
        return budget is not None \
            and self.card_state(worker, device, taken, budget)[1]

    def places(self, options: Mapping[str, Any]
               ) -> tuple[list[list[tuple[Any, str]]], bool]:
        """``(places, here)`` for a run with *options*: every place on a
        worker it may go (a device set is one place per worker, all of its
        devices; otherwise one card each), and whether this process may take
        it (``any``, by count). The one enumeration ``wants_budget``,
        ``memory_refusal`` and the waiting reason use."""
        worker_id, devices = entry_target(options)
        if worker_id == TARGET_LOCAL:
            return [], True
        requested = str(options.get("device") or "cpu")
        here = False
        if worker_id != TARGET_ANY:
            worker = self._registry.get(worker_id)
            workers = [] if worker is None else [worker]
        else:
            # As ``place`` picks them: the one-shot pool (#790) is its
            # one-shot workers only, never this process; ``any`` is every
            # other worker, then this process.
            pool = entry_pool(options) == POOL_ONE_SHOT
            here = (not pool and not renumbered_on_a_worker(devices)
                    and _serves(self._local_devices(), requested, devices))
            workers = [w for w in self._registry.workers()
                       if not _skipped_by_any(w)
                       and _one_shot(w) == pool
                       and _serves(w.device_values(), requested, devices)]
        places: list[list[tuple[Any, str]]] = []
        for worker in workers:
            if devices:
                cards = [canonical_device(d) for d in devices]
            else:
                cards = candidate_devices(worker.device_values(), requested)
            cards = [d for d in cards if worker.offers(d)]
            if devices:
                places.append([(worker, d) for d in cards])
            else:
                places.extend([(worker, d)] for d in cards)
        return places, here

    def wants_budget(self, options: Mapping[str, Any]) -> bool:
        """Whether a run with *options* could be placed on a shared card,
        so a budget matters."""
        places, _ = self.places(options)
        return any(self.shared(w, d) for place in places for w, d in place)

    def memory_refusal(self, options: Mapping[str, Any],
                       budget: int | None, taken: Mapping[SlotKey, int],
                       *, wait: bool) -> str | None:
        """Why a run with *options* and *budget* is refused at submit
        (#788 spec 4), or None.

        Only shared cards are judged: a run that may also go to a card
        admitted by count (this process, for ``any``) is never refused
        here, it queues as before. Refused: no budget, for a run that can
        only go to shared cards; a footprint larger than every candidate's
        total (it would wait forever); and, unless *wait*, one no candidate
        fits now. A worker that is away is judged on its last report for
        the total only: whether it fits NOW is not known, so the run waits.
        """
        places, here = self.places(options)
        if here or not places or any(
                not any(self.shared(w, d) for w, d in place)
                for place in places):
            return None  # nowhere to judge, or somewhere admits by count
        if budget is None:
            worker, device = places[0][0]
            return mem.no_budget_message(
                f"{device} on {worker.id}" if len(places) == 1
                else "every device this run may use")
        reasons: list[str] = []
        fits_ever = fits_now = False
        kind = places[0][0][1]
        for place in places:
            place_ever = place_now = True
            for worker, device in place:
                if not self.shared(worker, device):
                    continue
                ever, now, why = self.card_state(worker, device, taken,
                                                 budget)
                if not getattr(worker, "online", True) and ever:
                    now, why = True, None  # stale figures: wait for it
                place_ever = place_ever and ever
                place_now = place_now and now
                if why is not None:
                    reasons.append(why)
            fits_ever = fits_ever or place_ever
            fits_now = fits_now or place_now
        if not fits_ever or (not fits_now and not wait):
            return mem.needs_message(budget, reasons, kind)
        return None

    def _memory_wait(self, options: Mapping[str, Any],
                     budget: int | None,
                     taken: Mapping[SlotKey, int]) -> QueueReason | None:
        """Why a run for a named worker waits for memory, or None (it waits
        for its turn, or for a card admitted by count)."""
        places, _ = self.places(options)
        if not places or any(not any(self.shared(w, d) for w, d in place)
                             for place in places):
            return None
        if budget is None:
            worker, device = places[0][0]
            return QueueReason(
                REASON_MEMORY_NO_BUDGET,
                {"device": device, "worker": worker.id},
                mem.no_budget_message(f"{device} on {worker.id}",
                                      waiting=True))
        reasons = []
        for place in places:
            for worker, device in place:
                if self.shared(worker, device):
                    _, now, why = self.card_state(worker, device, taken,
                                                  budget)
                    if not now and why is not None:
                        reasons.append(why)
        if not reasons:
            return None
        device = places[0][0][1]
        context = mem.context_bytes(device)
        return QueueReason(
            REASON_MEMORY_WAIT,
            {"need": mem.format_gib(budget),
             "context": mem.format_gib(context) if context else None,
             "reasons": [mem.coded_payload(why) for why in reasons]},
            "waiting for memory: " + mem.needs_message(budget, reasons,
                                                       device))


def one_shot_busy(worker: Any) -> QueueReason:
    """Why a run naming one-shot *worker* waits although it is there (#790)."""
    if worker.one_shot_claim is not None:
        return QueueReason(
            REASON_ONE_SHOT_TAKEN,
            {"worker": worker.name, "run": worker.one_shot_claim},
            f"one-shot worker {worker.name} already took run "
            f"{worker.one_shot_claim}; start another one-shot agent")
    return QueueReason(
        REASON_ONE_SHOT_RUN_ID,
        {"worker": worker.name, "run": worker.one_shot_run_id},
        f"one-shot worker {worker.name} takes only run "
        f"{worker.one_shot_run_id} (--run-id)")


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


def _report(worker: Any, device: str) -> Any:
    """*worker*'s last ``DeviceReport`` for *device*, or None."""
    for report in getattr(worker, "devices", None) or ():
        try:
            if canonical_device(report.value) == device:
                return report
        except ValueError:
            continue
    return None


def _draining(worker: Any) -> bool:
    return bool(getattr(worker, "draining", False))


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
