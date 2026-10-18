"""Memory-based admission for shared GPUs (#788).

A worker's card admits runs one of two ways, its **policy**, set per
``(worker, device)`` in the Workers panel and stored in
``worker_device_policies`` (migration 015):

- ``exclusive`` (the default, and what every card had before #788): the
  per-slot count limit (``QueueLimits``), one run per card;
- ``shared``: runs declare a memory **budget** and share the card while the
  budgets fit. A run's budget is enforced in its own subprocess on the
  worker (``torch.cuda.set_per_process_memory_fraction``, ``agent.child``),
  so a run that outgrows it runs out of memory alone and its neighbours are
  untouched.

The ledger
----------
:class:`MemoryLedger` is the control plane's account of what it promised on
each slot (``(worker_id, device)``, ``scheduling.slot_for``; the slots of
one physical card, a same-host worker's and this process's or two
workers', are read as one account, #842): the budgets of runs
assigned but not yet started (**reserved**) and of runs whose subprocess
reported the cap in place (**running**). The card's **total** and **free**
bytes come from the worker's heartbeat (NVML, device-wide: other users'
jobs on a shared machine are counted, which is what a ledger needs).

A run on a CUDA card also holds its CUDA context (cuBLAS, kernels), which
the allocator cap does not cover: :func:`footprint` is its budget plus
``CODEFYUI_EXECUTION_CUDA_CONTEXT_MIB`` (768 MiB; ~650 MiB measured), and
the ledger holds footprints. A run fits on a card when its footprint is at
most the card's headroom::

    headroom = min(free - reserved, total - reserved - running)

The first term is the spec's (``free - reserved``): memory nobody holds now,
minus what was promised to runs that have not started allocating. The
second keeps the promises themselves within the card: a run that started a
moment ago holds almost nothing yet, so ``free`` alone would admit a third
6 GiB run next to two on a 16 GiB card and let all three grow into each
other. Without a ``free`` figure the second term alone decides; without a
``total`` the card cannot be shared (it is refused, saying so).

Budgets
-------
``target.memory_gib`` declares one (GiB per device of the run). A run that
declares none gets one from a :class:`BudgetSource` (the default,
``memory_estimate.EstimateBudgetSource``: the last peak of a succeeded run
of the same graph, by its structural hash (#794), times
:data:`HISTORY_MARGIN`, :func:`history_suggestion`; a CUDA peak only: MPS
reports what the driver holds at the end, which is not a peak). A static estimate from the graph's parameter counts is
never a default: the dialog offers it, and a run carries it only as a
declared ``target.memory_gib`` with ``target.memory_source: "estimate"``.
A run with no budget cannot go to a shared card.

``free`` is read every heartbeat (10 s) for the whole card; a run's own
growth after it started is not visible in it, which the ``total`` term
covers for our runs but not for other users' jobs.

Nothing here runs unless a card is ``shared``: with every card exclusive
(no row in the table), admission is the count limit, unchanged.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol

logger = logging.getLogger(__name__)

POLICY_EXCLUSIVE = "exclusive"
POLICY_SHARED = "shared"
POLICIES = (POLICY_EXCLUSIVE, POLICY_SHARED)

#: The device kinds a policy may be set for: the CPU is never "shared" in
#: this sense (its runs are count-limited, and no allocator cap exists).
SHAREABLE_KINDS = ("cuda", "mps")

GIB = 1024 ** 3
#: The largest ``target.memory_gib`` accepted: no card is bigger, and a
#: typo of a few zeros should read as a refusal, not a run that waits.
MAX_MEMORY_GIB = 4096.0
#: The history default (#788 spec 3): the last peak times this.
HISTORY_MARGIN = 1.2

#: ``Budget.source`` values.
SOURCE_DECLARED = "declared"
SOURCE_HISTORY = "history"
#: A static estimate from the graph's parameter counts (#794) the user
#: accepted (``target.memory_source``), never a measured peak.
SOURCE_ESTIMATE = "estimate"

SlotKey = tuple[str, str]


def gib_to_bytes(gib: float) -> int:
    return int(math.ceil(gib * GIB))


def context_bytes(device: str) -> int:
    """What a run's CUDA context takes on *device* beyond its allocator cap
    (``CODEFYUI_EXECUTION_CUDA_CONTEXT_MIB``); 0 for anything but CUDA."""
    if not str(device).startswith("cuda"):
        return 0
    from ..config import settings

    return max(0, int(settings.EXECUTION_CUDA_CONTEXT_MIB)) * 1024 ** 2


def footprint(budget: int, device: str) -> int:
    """What a run with *budget* holds on *device*: the budget plus its
    CUDA context. What the ledger reserves and admission compares."""
    return int(budget) + context_bytes(device)


def format_gib(nbytes: int | float | None) -> str:
    """``6.3 GiB``; one decimal, at least ``0.1`` for a positive figure."""
    if nbytes is None:
        return "? GiB"
    value = nbytes / GIB
    if 0 < value < 0.1:
        value = 0.1
    return f"{value:.1f} GiB"


def short_run(run_id: str) -> str:
    """How a refusal names a run: its first 8 characters."""
    return run_id[:8]


@dataclass(frozen=True)
class Budget:
    """A run's memory budget per device, and where it came from."""

    bytes: int
    source: str = SOURCE_DECLARED
    #: For a ``history`` budget: the run whose peak it was taken from.
    from_run: str | None = None
    #: For a ``history`` budget that replaced a smaller accepted estimate
    #: while the run waited (#794): that estimate, in bytes.
    raised_from: int | None = None

    def describe(self) -> dict[str, Any]:
        entry: dict[str, Any] = {"bytes": self.bytes, "source": self.source}
        if self.from_run:
            entry["from_run"] = self.from_run
        if self.raised_from:
            entry["raised_from"] = self.raised_from
        return entry


@dataclass
class _Held:
    slots: tuple[SlotKey, ...]
    bytes: int
    started: bool = False


@dataclass
class SlotAccount:
    """What the ledger holds on one slot."""

    reserved: int = 0
    running: int = 0
    runs: list[str] = field(default_factory=list)


class MemoryLedger:
    """Budgets promised per slot: reserved (assigned, not started) and
    running. Event-loop only; every method is synchronous."""

    def __init__(self) -> None:
        self._runs: dict[str, _Held] = {}

    def reserve(self, run_id: str, slots: Iterable[SlotKey],
                nbytes: int) -> None:
        """*run_id* was placed with *nbytes* on each of *slots*."""
        self._runs[run_id] = _Held(tuple(slots), int(nbytes))

    def started(self, run_id: str) -> bool:
        """*run_id*'s cap is in place: its budget moves from reserved to
        running. False when the ledger does not hold it."""
        held = self._runs.get(run_id)
        if held is None:
            return False
        held.started = True
        return True

    def release(self, run_id: str) -> None:
        """Forget *run_id*'s budget. Idempotent."""
        self._runs.pop(run_id, None)

    def holds(self, run_id: str) -> bool:
        return run_id in self._runs

    def budget_of(self, run_id: str) -> int | None:
        held = self._runs.get(run_id)
        return None if held is None else held.bytes

    def account(self, slot: SlotKey, *more: SlotKey) -> SlotAccount:
        """What the ledger holds on *slot*, or on any of *slot* and *more*
        (the slots of one physical card, #842): each run counted once."""
        wanted = {slot, *more}
        account = SlotAccount()
        for run_id, held in self._runs.items():
            if wanted.isdisjoint(held.slots):
                continue
            if held.started:
                account.running += held.bytes
            else:
                account.reserved += held.bytes
            account.runs.append(run_id)
        return account

    def holders(self, slot: SlotKey, *more: SlotKey) -> int:
        """How many budgeted runs hold *slot* (or any of *more*, #842)."""
        wanted = {slot, *more}
        return sum(1 for held in self._runs.values()
                   if not wanted.isdisjoint(held.slots))


def headroom(total: int | None, free: int | None,
             account: SlotAccount) -> int | None:
    """The bytes a new budget may take on a card (see the module docstring),
    or None when the card reports no total."""
    if total is None:
        return None
    room = total - account.reserved - account.running
    if free is not None:
        room = min(room, free - account.reserved)
    return max(0, room)


# ── the default budget source: history ────────────────────────────────────


def run_shape(options: Mapping[str, Any] | None) -> str:
    """The device kind and the number of devices a run with *options* asks
    for (``cuda x2``): ``target.devices`` when given, else ``device``. Part
    of the history key (:func:`graph_hash`): a per-card peak of a run on two
    cards is not a budget for the same graph on one."""
    options = options if isinstance(options, Mapping) else {}
    target = options.get("target")
    devices = target.get("devices") if isinstance(target, Mapping) else None
    if isinstance(devices, list) and devices:
        first, count = str(devices[0]), len(devices)
    else:
        first, count = str(options.get("device") or "cpu"), 1
    return f"{first.split(':')[0].strip().lower() or 'cpu'} x{count}"


def graph_hash(graph: Mapping[str, Any],
               options: Mapping[str, Any] | None = None) -> str:
    """A digest of what a graph computes, and on what, for "the same run"
    (#788): the key a run's peak is stored under and its default budget is
    looked up by. Since #794 the structural hash of the executable graph
    (``memory_estimate.structural_hash``: node types and the params that
    shape it, the wiring, no ids, positions, labels or seeds) and the run's
    :func:`run_shape` (device kind and count; without *options*, one CPU).
    """
    from .memory_estimate import structural_hash

    return structural_hash(graph, options)


def peak_bytes(peak_memory: Any) -> int | None:
    """The largest CUDA figure of an ``execution_facts.peak_memory`` (#770):
    a card's ``reserved_bytes`` (what the allocator held, which is what the
    cap limits), else its ``allocated_bytes``. None when there is none.

    MPS's ``held_bytes`` is NOT used: it is what the driver held at the END
    of the run, not a peak, so a budget derived from it could sit below what
    the run needs and make it run out of memory under its own cap."""
    if not isinstance(peak_memory, Mapping):
        return None
    best: int | None = None
    for device, entry in peak_memory.items():
        if not isinstance(entry, Mapping) or not str(device).startswith("cuda"):
            continue
        for key in ("reserved_bytes", "allocated_bytes"):
            value = entry.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) \
                    and value > 0:
                best = max(best or 0, int(value))
                break
    return best


@dataclass(frozen=True)
class Suggestion:
    """A default budget for a run that declared none."""

    budget: Budget
    #: The peak the budget was derived from, in bytes.
    peak_bytes: int | None = None


class BudgetSource(Protocol):
    """Where a run's default ``memory_gib`` comes from (#788; #794 keys it
    by the graph's structure). ``suggest`` gets the graph (secrets
    scrubbed), the run's normalised options (its target and device) and,
    when the caller has it, the graph's history key *digest*
    (:func:`graph_hash` of the graph before unknown node types' params were
    scrubbed), and returns None when it has nothing to say."""

    async def suggest(self, graph: Mapping[str, Any],
                      options: Mapping[str, Any], *,
                      digest: str | None = None) -> Suggestion | None: ...


def history_suggestion(run_id: str | None,
                       peak_memory: Any) -> Suggestion | None:
    """The default budget a measured run gives its graph (#788): its CUDA
    peak (:func:`peak_bytes`) times :data:`HISTORY_MARGIN`, from *run_id*;
    None for a run with no CUDA peak. The one place the rule lives: the
    budget source's suggestion, a waiting estimate's raise and a submitted
    estimate's raise (#794) all go through it."""
    peak = peak_bytes(peak_memory)
    if peak is None:
        return None
    return Suggestion(Budget(int(math.ceil(peak * HISTORY_MARGIN)),
                             SOURCE_HISTORY, from_run=run_id),
                      peak_bytes=peak)


# ── messages ──────────────────────────────────────────────────────────────


def held_without_budget(device: str, worker: str) -> str:
    """Why a shared card cannot take a budgeted run although memory is
    free: a run without a budget holds it (#788 review 3)."""
    return (f"{device} on {worker} is held by a run without a memory budget "
            "(a run on this server, or one started before the card was "
            "shared) until it ends")


def no_budget_message(where: str, *, waiting: bool = False) -> str:
    """No budget for a run that can only go to shared cards. *waiting*: the
    run is already queued, so it says how to get it going."""
    if waiting:
        return (f"{where} is shared by memory and this run has no memory "
                "budget: it takes one once a run of the same graph on the "
                "same devices reports its peak, or cancel it and queue it "
                "again with target.memory_gib")
    return (f"{where} is shared by memory and this run has no memory budget: "
            "declare target.memory_gib (no earlier run of this graph on these "
            "devices reported its peak memory)")


def reserved_for_a_set(device: str, worker: str) -> str:
    """Why a shared card cannot take a run although memory is free: an
    earlier run waiting for its whole device set holds it (#788)."""
    return (f"{device} on {worker} is reserved for an earlier run that waits "
            "for its whole device set")


def does_not_fit(device: str, worker: str, room: int | None,
                 runs: list[str]) -> str:
    """``cuda:1 on gpu-box-1 has 6.3 GiB free (runs a1b2c3d4, ...)``."""
    if room is None:
        return (f"{device} on {worker} reports no memory total, so it cannot "
                "admit by memory")
    text = f"{device} on {worker} has {format_gib(room)} free"
    if runs:
        text += f" (runs {', '.join(short_run(r) for r in runs)})"
    return text


def needs_message(budget: int, reasons: list[str], device: str = "") -> str:
    """``needs 6.0 GiB (+0.8 GiB CUDA context), <reasons>``."""
    context = context_bytes(device)
    need = format_gib(budget)
    if context:
        need += f" (+{format_gib(context)} CUDA context)"
    return f"needs {need}, " + "; ".join(reasons)
