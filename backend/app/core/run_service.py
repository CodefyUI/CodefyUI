"""Server-owned graph runs — the Run Service (#120).

A run used to live inside the WebSocket handler: ``ws_execution.py`` created
the ``asyncio.Task``, and it cancelled that task on ``WebSocketDisconnect``.
Closing the browser tab killed a training job. Submitting a second graph on
the same socket killed the first. Both are properties of *where the task was
owned*, not of the engine — ``graph_engine.execute_graph`` is already a
perfectly reusable coroutine.

This module owns the task instead. The service is a plain object (no
FastAPI, no HTTP — the ``db.py``/``api_contract.py`` precedent), constructed
once in the lifespan and reachable at ``app.state.run_service``. Its
responsibilities, and nothing else:

- **submit** — persist the run (``RunStore``), then start it. Between those
  two steps is the seam #123's FIFO queue slots into; in THIS issue every
  submitted run starts immediately.
- **observe** — every engine callback is appended to ``exec_run_events`` and
  fanned out to live subscribers; scalar progress values are batched into
  ``exec_run_metrics``.
- **cancel** — cooperative, via the ``ExecutionContext`` flag (never
  ``Task.cancel``; see ``cancel``).
- **survive** — startup recovery retires rows a dead process left
  ``running``; shutdown drains in-flight tasks BEFORE the database closes.

Layering
--------
``core`` never imports ``api``… with one deliberate, function-local
exception: the #117 output-entry builders currently live in
``api/ws_execution.py``. They are imported lazily in ``_progress_bridge``
rather than duplicated, because a second copy of that logic is exactly the
drift #117 removed. #121 rewrites the WS handler on top of this service and
should move those two helpers into ``core`` at that point.

Ids
---
ONE id per run: ``RunStore``'s ``uuid4().hex``. It is the ``exec_runs.id``,
the ``ExecutionContext.execution_id``, and the ``run_id`` handed to
``execute_graph`` (so ``RunOutputStore`` entries key off the same string).
``ExecutionContext``'s own ``str(uuid4())`` default is never used here —
two id shapes for one run is how outputs stop being findable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, NamedTuple

from ..config import settings
from .db import utc_now_iso
from .device_utils import resolve_device
from .execution_context import CancellationError, ExecutionContext
from .graph_engine import GraphValidationError, build_preset_fallback, execute_graph
from .run_output_store import RunOutputStore
from .run_store import (
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    EventRecord,
    MetricPoint,
    RunStore,
    json_safe,
)

logger = logging.getLogger(__name__)


# ── event vocabulary ──────────────────────────────────────────────────────
#
# Deliberately the WebSocket message types, verbatim. The durable event log
# exists to be replayed onto that wire (#121: attach to a running run and
# catch up), so a 1:1 mapping means the bridge is
# ``{"type": event.type, **(event.payload or {})}`` and there is no second
# vocabulary to keep in sync. The ``type`` key itself is NOT stored in the
# payload — the column already carries it.
EVENT_RUN_STARTED = "execution_start"
EVENT_NODE_STATUS = "node_status"
EVENT_RUN_COMPLETED = "execution_complete"
EVENT_RUN_FAILED = "execution_error"
EVENT_RUN_STOPPED = "execution_stopped"

#: ``execution_stopped`` payload discriminator. The WS protocol has one
#: "stopped" frame; the run ROW distinguishes cancelled from interrupted,
#: and this key carries the same fact into the replay log.
STOP_REASON_CANCELLED = "cancelled"
STOP_REASON_INTERRUPTED = "interrupted"

# ── submit options ────────────────────────────────────────────────────────

#: The complete option vocabulary. Unknown keys are REJECTED rather than
#: ignored: ``{"devcie": "cuda"}`` silently running on CPU for forty minutes
#: is the failure mode this closes. Extending the set is a one-line additive
#: change — #121 (canvas flags: verbose/backward/graph_id) and #123 (queue
#: priority) will each do exactly that, deliberately.
OPTION_KEYS = frozenset({"device", "seed", "record_outputs", "lane"})

#: Default lane. Named for #123's queue; no queue exists yet, so today it is
#: a label carried through to ``exec_runs.options`` unchanged.
DEFAULT_LANE = "queued"
MAX_LANE_LENGTH = 64
MAX_NAME_LENGTH = 64
MAX_SEED = 2 ** 32 - 1

#: The device vocabulary ``resolve_device`` actually understands. Validated
#: as a VALUE, not just a key: strict option keys exist to stop a typo from
#: silently running forty minutes on the wrong device, and ``{"device":
#: "cudda"}`` defeats that entirely if only the key is checked — it sails
#: through, hits ``resolve_device``'s unknown-value branch, and degrades to
#: CPU with nothing but a log line. ``auto`` is accepted because the canvas
#: device selector emits it; note that ``resolve_device`` currently maps it
#: to CPU (pre-existing behaviour, deliberately not changed here).
DEVICE_PATTERN = re.compile(r"^(cpu|auto|cuda(:\d+)?|mps(:\d+)?)$")

#: Progress-payload keys that describe the LOOP, not a measurement. Turning
#: ``epoch`` into a series named "epoch" whose value equals its own step is
#: noise; ``total_epochs`` is a constant repeated once per point.
NON_METRIC_KEYS = frozenset({"event", "step", "epoch", "total_epochs",
                             "start_epoch"})

#: Flush thresholds for the metric batcher (see ``_flush_metrics``).
METRIC_FLUSH_MAX_POINTS = 64
METRIC_FLUSH_INTERVAL_S = 0.5

#: Default per-event payload ceiling; overridden from
#: ``settings.RUN_EVENT_PAYLOAD_CAP_BYTES`` in the lifespan. See
#: ``cap_event_payload`` for what "over cap" does.
EVENT_PAYLOAD_CAP_BYTES = 128 * 1024

#: How long ``shutdown`` lets a run stop cooperatively before hard-cancelling.
DEFAULT_SHUTDOWN_GRACE_S = 5.0

#: Per-subscriber queue depth. Overflow drops (see ``RunSubscription``).
DEFAULT_SUBSCRIBER_QUEUE_SIZE = 256


class RunSubmitError(ValueError):
    """Malformed submit envelope or options. REST maps this to 400."""


class RunServiceUnavailable(RuntimeError):
    """The service cannot take work right now. REST maps this to 503.

    Distinct from a bare ``RuntimeError`` on purpose: "the server is
    shutting down, retry" and "something inside broke" are different answers
    and must not share a status code.
    """


class SubmitResult(NamedTuple):
    """What ``submit`` hands back: the id, and where the run actually is.

    ``status`` is ``running`` today. It is not hardcoded at the call site
    because #123's queue will legitimately return ``queued`` from the same
    call, and a client that reads this field keeps working.
    """

    run_id: str
    status: str


class CancelOutcome(NamedTuple):
    """Result of a cancel request. ``cancelled`` is False for a no-op."""

    run_id: str
    status: str
    cancelled: bool


def normalize_options(raw: Any) -> dict[str, Any]:
    """Validate the submit options and fill in defaults.

    Strict on purpose (see ``OPTION_KEYS``). Returns a NEW dict with every
    key present, so ``exec_runs.options`` always records the full effective
    configuration rather than "whatever the client happened to send" — a
    stored run has to be readable years later without knowing which defaults
    were in force at submit time.

    ``device`` is normalised (trimmed, lowercased) but NOT availability-
    checked here: that resolution happens at start time via
    ``resolve_device`` and lands in ``exec_runs.queue_key``, so the row keeps
    both what was asked for and what was used.
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise RunSubmitError(
            f"options must be an object, got {type(raw).__name__}")
    unknown = sorted(set(raw) - OPTION_KEYS)
    if unknown:
        raise RunSubmitError(
            f"unknown option(s) {unknown}; supported: {sorted(OPTION_KEYS)}")

    device = raw.get("device", "cpu")
    if not isinstance(device, str) or not device.strip():
        raise RunSubmitError("device must be a non-empty string")
    device = device.strip().lower()
    if DEVICE_PATTERN.match(device) is None:
        raise RunSubmitError(
            f"unknown device {device!r}; expected cpu, auto, cuda, cuda:N, "
            "mps or mps:N")

    seed = raw.get("seed")
    if seed is not None:
        # bool is an int subclass; True is not a seed.
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise RunSubmitError("seed must be an integer or null")
        if not 0 <= seed <= MAX_SEED:
            raise RunSubmitError(f"seed must be between 0 and {MAX_SEED}")

    record_outputs = raw.get("record_outputs", False)
    if not isinstance(record_outputs, bool):
        raise RunSubmitError("record_outputs must be true or false")

    lane = raw.get("lane", DEFAULT_LANE)
    if not isinstance(lane, str) or not lane.strip():
        raise RunSubmitError("lane must be a non-empty string")
    if len(lane) > MAX_LANE_LENGTH:
        raise RunSubmitError(f"lane must be at most {MAX_LANE_LENGTH} chars")

    return {
        "device": device,
        "seed": seed,
        "record_outputs": record_outputs,
        "lane": lane.strip(),
    }


def normalize_name(raw: Any) -> str | None:
    """Validate the optional run label. Bounded like ``lane``.

    A run name is display text on a list row, not a document: an unbounded
    one would be stored on every row and re-sent on every poll of the Runs
    panel. Blank (or whitespace-only) means unnamed, stored as SQL NULL
    rather than an empty string nobody can tell apart from "not set".
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise RunSubmitError(f"name must be a string, got {type(raw).__name__}")
    name = raw.strip()
    if not name:
        return None
    if len(name) > MAX_NAME_LENGTH:
        raise RunSubmitError(f"name must be at most {MAX_NAME_LENGTH} chars")
    return name


def normalize_graph(raw: Any) -> dict[str, Any]:
    """Validate the submit ENVELOPE only — never the graph's semantics.

    Structural validation (unknown node types, missing entry points, cycles)
    stays where it already lives: ``prepare_executable_graph``, inside the
    run. Duplicating it here would mean two validators drifting apart, and
    would make ``POST /api/runs`` reject things the engine accepts. A graph
    that cannot execute becomes a ``failed`` run with the engine's own error
    on the row, which is a far more useful artifact than a 400.
    """
    if not isinstance(raw, dict):
        raise RunSubmitError(
            "graph must be an object with 'nodes' and 'edges'")
    nodes = raw.get("nodes")
    # Absent means "none of these"; anything else is checked, never coerced.
    # ``raw.get(k) or []`` would quietly accept a malformed ``"edges": {}``
    # as an empty edge list because an empty dict is falsy.
    edges = [] if raw.get("edges") is None else raw["edges"]
    presets = [] if raw.get("presets") is None else raw["presets"]
    if not isinstance(nodes, list):
        raise RunSubmitError("graph.nodes must be a list")
    if not nodes:
        raise RunSubmitError("graph.nodes is empty")
    if not isinstance(edges, list):
        raise RunSubmitError("graph.edges must be a list")
    if not isinstance(presets, list):
        raise RunSubmitError("graph.presets must be a list")
    return {"nodes": nodes, "edges": edges, "presets": presets}


def json_size(value: Any) -> int:
    """Bytes this value would occupy as a stored JSON column.

    Matches ``run_store._dumps``' encoding (compact separators, the default
    ``ensure_ascii=True``), so the number is the real storage cost and not
    an estimate. ``ensure_ascii`` also means every character encodes to one
    byte, which is why ``len`` of the string is the byte count with no
    encode step. ``default=str`` keeps MEASURING from ever raising on
    something exotic — a sizer that throws would fail the run it was only
    trying to bound.
    """
    try:
        return len(json.dumps(value, separators=(",", ":"), default=str))
    except (TypeError, ValueError, RecursionError):  # pragma: no cover
        return 0


def _elide_entry(entry: Any, budget_bytes: int) -> Any:
    """Replace one over-budget output entry with a placeholder.

    ``output_kind`` and ``port`` survive so a UI can render "image, elided"
    in the right slot rather than a hole. The payload key (``image``,
    ``tensor_summary``, …) is DROPPED rather than stuffed with a marker
    object: a consumer that reads ``entry["text"]`` expecting a string must
    not silently receive a dict instead. ``"elided" in entry`` is the check.
    """
    if not isinstance(entry, dict):
        return entry
    size = json_size(entry)
    if size <= budget_bytes:
        return entry
    elided: dict[str, Any] = {"output_kind": entry.get("output_kind"),
                              "elided": True, "bytes": size,
                              "cap_bytes": budget_bytes}
    if "port" in entry:
        elided["port"] = entry["port"]
    return elided


def cap_event_payload(payload: Any, *, cap_bytes: int) -> Any:
    """Bound one event payload before it becomes a durable row.

    Retention counts RUNS, not bytes. Nothing else stops a graph whose nodes
    emit base64 PNGs (#117 image entries are the full image) from writing
    hundreds of megabytes of ``exec_run_events`` for a single run — with
    ``record_outputs`` off, because this path is the live status stream, not
    the output capture. Same for a training payload whose per-batch array
    grows every epoch.

    Three steps, cheapest first:

    1. Measure once. Under cap — the overwhelmingly common case, an ordinary
       ``node_status`` is a couple of KB — and the payload is returned
       untouched, having cost exactly one ``json.dumps``.
    2. Over cap, elide the output entries that are pulling more than their
       equal share of the budget. Equal share, rather than the whole cap,
       so a payload made of several medium entries degrades entry by entry
       instead of collapsing wholesale.
    3. Still over — nothing left to trim, e.g. a single enormous ``error``
       string — and the payload collapses to a marker that keeps the
       identifying keys. Losing the body beats refusing the event.

    ``cap_bytes <= 0`` disables the whole thing.
    """
    if cap_bytes <= 0 or payload is None:
        return payload
    size = json_size(payload)
    if size <= cap_bytes:
        return payload

    entries = payload.get("outputs") if isinstance(payload, dict) else None
    if isinstance(entries, list) and entries:
        budget = max(1, cap_bytes // len(entries))
        payload = {**payload,
                   "outputs": [_elide_entry(e, budget) for e in entries]}
        size = json_size(payload)
        if size <= cap_bytes:
            return payload

    marker: dict[str, Any] = {"elided": True, "bytes": size,
                              "cap_bytes": cap_bytes}
    if isinstance(payload, dict):
        for key in ("node_id", "status", "run_id", "reason"):
            if key in payload:
                marker[key] = payload[key]
    return marker


def scalar_metrics(
    payload: Any, *, node_id: str | None, step: int,
) -> list[MetricPoint]:
    """Extract chartable scalars from one progress payload.

    Every finite top-level number that is not a loop descriptor becomes a
    point of a series named after its key. Booleans are excluded (a flag is
    not a measurement), as are strings, lists and nested objects — a
    ``losses`` array is per-batch detail that belongs in the event payload,
    not as one row per element in the metrics table.
    """
    if not isinstance(payload, dict):
        return []
    points: list[MetricPoint] = []
    for key, value in payload.items():
        if key in NON_METRIC_KEYS or isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue
        points.append(MetricPoint(name=key, value=float(value), step=step,
                                  node_id=node_id))
    return points


def metric_step(payload: Any, *, fallback: int) -> int:
    """The step a progress payload belongs to.

    ``step`` wins over ``epoch``; both beat the caller's running counter, so
    a producer that reports nothing still gets a monotonic x-axis instead of
    every point piling up on step 0.
    """
    if isinstance(payload, dict):
        for key in ("step", "epoch"):
            value = payload.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
    return fallback


def apply_seed(seed: int | None) -> None:
    """Seed the RNGs a run might touch. Best effort, PROCESS-GLOBAL.

    Deliberately honest about its limits: ``torch.manual_seed`` and friends
    set global state, so two concurrent runs with different seeds do NOT get
    independent streams — the second submit reseeds the first run's
    generator mid-flight. Reproducibility here means "run this one alone and
    get the same numbers", which is what a stored ``seed`` is worth today.
    Per-run generator isolation is a node-level change, not a service one.
    """
    if seed is None:
        return
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy always present in practice
        logger.debug("numpy seeding skipped", exc_info=True)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch always present in practice
        logger.debug("torch seeding skipped", exc_info=True)


# ── in-process fan-out ────────────────────────────────────────────────────


class _Broadcast:
    """Edge-triggered wake-up for long pollers. No busy waiting anywhere.

    ``asyncio.Event`` is level-triggered and single-shot; a shared one would
    have to be cleared, and whoever clears it races everyone who has not
    woken yet. Instead each ``notify`` SETS the current event and installs a
    fresh one for the next generation. A waiter captures the event object
    BEFORE it reads the store, so a notification that lands during that read
    still wakes it — the classic lost-wakeup window is closed by ordering,
    not by a timeout.

    Not an ``asyncio.Condition``: a Condition makes every waiter contend for
    one lock just to be told to go read sqlite, and its ``wait()`` reacquires
    that lock before returning, serialising the wake-ups.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def waiter(self) -> asyncio.Event:
        """The generation to wait on. Capture it before checking the store."""
        return self._event

    def notify(self) -> None:
        event, self._event = self._event, asyncio.Event()
        event.set()


# eq=False: a subscription is a live handle, not a value. Identity equality
# (and therefore identity hashing) is what lets it live in the per-run
# subscriber set; the dataclass default would generate __eq__ and set
# __hash__ to None, making it unhashable.
@dataclass(eq=False)
class RunSubscription:
    """A live feed of one run's events, for in-process push consumers (#121).

    Drop-tolerant BY DESIGN. The queue is bounded and a full queue drops the
    new event rather than applying backpressure — a stalled consumer must
    never be able to slow down or deadlock the run that is feeding it. The
    durable log is the source of truth: a consumer that sees ``take_dropped()
    > 0`` re-reads the tail from ``RunStore.get_events`` starting at its last
    cursor and loses nothing. ``None`` on the queue is the end-of-run
    sentinel; ``closed`` says the same thing for a consumer that never drains.
    """

    run_id: str
    queue: "asyncio.Queue[EventRecord | None]"
    closed: bool = False
    _dropped: int = 0

    def offer(self, event: "EventRecord | None") -> None:
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self._dropped += 1

    def take_dropped(self) -> int:
        """How many events were dropped since the last call; resets to 0."""
        dropped, self._dropped = self._dropped, 0
        return dropped


@dataclass(eq=False)
class _ActiveRun:
    """Everything the service holds for one in-flight run."""

    run_id: str
    context: ExecutionContext
    signal: _Broadcast
    record_outputs: bool = False
    task: "asyncio.Task[None] | None" = None
    subscribers: set[RunSubscription] = field(default_factory=set)
    #: "cancelled" (a user asked) or "interrupted" (the server is going
    #: away). Set BEFORE the context flag so the unwinding task can tell the
    #: two apart — they are different facts about why a run stopped.
    stop_reason: str | None = None
    #: True once the OUTCOME is decided — set on entry to ``_finalize``,
    #: before any of its awaits. A cancel arriving after this cannot change
    #: anything, so it must not claim it did.
    terminating: bool = False
    #: True once the terminal event is durable. Long pollers stop waiting on
    #: this, not on registry removal, which happens a few statements later.
    #: Strictly after ``terminating``: between the two the closing event is
    #: still being written, and a poller should keep waiting for it.
    finished: bool = False
    pending_metrics: list[MetricPoint] = field(default_factory=list)
    last_metric_step: int = 0
    last_flush_monotonic: float = field(default_factory=time.monotonic)

    def close(self) -> None:
        """End every subscription and wake every poller. Sync — never fails."""
        self.finished = True
        for subscription in list(self.subscribers):
            subscription.offer(None)
            subscription.closed = True
        self.subscribers.clear()
        self.signal.notify()


def _output_entry_builders() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """The #117 output-entry helpers, imported lazily (see module docstring)."""
    from ..api.ws_execution import build_node_output_entries, declared_image_ports

    return build_node_output_entries, declared_image_ports


# ── the service ───────────────────────────────────────────────────────────


class RunService:
    """Owns every server-side run: registry, tasks, events, lifecycle.

    One instance per process, held on ``app.state.run_service``. Everything
    it touches concurrently lives on the event loop (plain dict/set mutation
    with no ``await`` in between), so it needs no lock of its own; the only
    shared resource with real contention is the database, and ``Database.run``
    already serialises that.
    """

    def __init__(
        self,
        store: RunStore,
        *,
        output_store: RunOutputStore | None = None,
        retention_keep_last: int | None = None,
        event_payload_cap_bytes: int = EVENT_PAYLOAD_CAP_BYTES,
        shutdown_grace_s: float = DEFAULT_SHUTDOWN_GRACE_S,
        subscriber_queue_size: int = DEFAULT_SUBSCRIBER_QUEUE_SIZE,
        metric_flush_max_points: int = METRIC_FLUSH_MAX_POINTS,
        metric_flush_interval_s: float = METRIC_FLUSH_INTERVAL_S,
    ) -> None:
        self.store = store
        self._output_store = output_store
        self._retention_keep_last = retention_keep_last
        self._event_payload_cap_bytes = event_payload_cap_bytes
        self._shutdown_grace_s = shutdown_grace_s
        self._subscriber_queue_size = subscriber_queue_size
        self._metric_flush_max_points = metric_flush_max_points
        self._metric_flush_interval_s = metric_flush_interval_s
        self._runs: dict[str, _ActiveRun] = {}
        self._shutting_down = False

    # ── introspection ─────────────────────────────────────────────────────

    def active_run_ids(self) -> list[str]:
        """Runs THIS process is driving, submit order."""
        return list(self._runs)

    def is_active(self, run_id: str) -> bool:
        return run_id in self._runs

    def execution_id(self, run_id: str) -> str | None:
        """The ``ExecutionContext.execution_id`` of a live run (== its id)."""
        active = self._runs.get(run_id)
        return None if active is None else active.context.execution_id

    # ── submit ────────────────────────────────────────────────────────────

    async def submit(
        self,
        graph: Any,
        *,
        options: Any = None,
        name: str | None = None,
    ) -> SubmitResult:
        """Persist a run, then start it. Raises ``RunSubmitError`` on input.

        The two halves are separate on purpose. ``create_run`` makes the run
        exist and queryable before anything can execute, so a crash between
        the two leaves a row that startup recovery retires honestly instead
        of losing the submission entirely.

        **The #123 seam is between them.** A queue implementation stops
        calling ``_start`` here and calls it from its scheduler; the only
        extra work is re-reading the graph with
        ``store.get_graph_snapshot(run_id)``, since a deferred start cannot
        rely on the submitting request still being in memory. Nothing else in
        this module assumes immediacy.
        """
        if self._shutting_down:
            raise RunServiceUnavailable("run service is shutting down")
        normalized_graph = normalize_graph(graph)
        normalized_options = normalize_options(options)
        normalized_name = normalize_name(name)

        record = await self.store.create_run(
            graph_snapshot=normalized_graph,
            options=normalized_options,
            name=normalized_name,
            status=STATUS_QUEUED,
        )
        # ── #123 inserts scheduling HERE ─────────────────────────────────
        return self._start(record.id, normalized_graph, normalized_options)

    def _start(
        self, run_id: str, graph: dict[str, Any], options: dict[str, Any],
    ) -> SubmitResult:
        """Register the run and hand it to its own task. SYNCHRONOUS.

        Not a coroutine, and that is the load-bearing property: there is no
        ``await`` anywhere between putting the run in the registry and
        creating the task that owns it, so no cancellation of the SUBMITTING
        coroutine can land in the middle and strand a registry entry that
        nothing will ever finish. Everything that needs the database
        (``mark_running``, the start event) happens inside the task, where a
        cancellation is the RUN's cancellation and unwinds through the normal
        terminal path.

        The reported status is therefore a promise the task fulfils on its
        first step; a client that reads the row in that microsecond sees
        ``queued``, which is simply true.

        The shutdown re-check is not redundant with ``submit``'s. That one
        happens before ``create_run`` awaits (provenance capture can shell
        out to git), and ``shutdown`` snapshots the registry at any await —
        so a submit parked in that window would resume here and register a
        task nobody drains, with ``main.py`` closing the database underneath
        it. Refusing here leaves the durable ``queued`` row for startup
        recovery, exactly like a cancelled submit.
        """
        if self._shutting_down:
            raise RunServiceUnavailable(
                "run service began shutting down while the run was being "
                "persisted; it stays queued and will be retired at startup")
        context = ExecutionContext(
            execution_id=run_id,
            device=resolve_device(options.get("device")),
            max_workers=settings.MAX_PARALLEL_NODES,
            # A server-owned run is an isolated, reproducible unit: its
            # stored graph_snapshot + options must fully describe what ran.
            # Inheriting live nn.Module weights from a previous run would
            # make that description a lie, and two concurrent runs of the
            # same graph would train one shared module. The canvas's
            # keep-training workflow keeps working over the WS path; #121
            # re-introduces it here as an explicit option.
            weights_persistent=False,
            node_state_store=None,
        )
        active = _ActiveRun(
            run_id=run_id, context=context, signal=_Broadcast(),
            record_outputs=bool(options.get("record_outputs")),
        )
        self._runs[run_id] = active
        active.task = asyncio.create_task(
            self._drive(active, graph, options), name=f"run:{run_id}")
        return SubmitResult(run_id=run_id, status=STATUS_RUNNING)

    # ── the run task ──────────────────────────────────────────────────────

    async def _drive(
        self, active: _ActiveRun, graph: dict[str, Any],
        options: dict[str, Any],
    ) -> None:
        """The server-owned task. Nothing outside this module awaits it.

        A hard ``Task.cancel`` (shutdown's last resort) is BaseException, so
        it skips every ``except`` below and with them ``_finalize`` — the
        row stays ``running`` for the next boot's recovery to retire, which
        is deliberate: a forcibly-killed run must not write a tidy terminal
        status it cannot vouch for. The ``finally`` still runs (that is what
        closes subscriptions and deregisters the run); only the terminal
        bookkeeping is skipped. Everything else — including a broken
        database — is caught, because an unretrieved task exception is a
        silent death nobody would ever see.
        """
        run_id = active.run_id
        try:
            if not await self._begin(active):
                return
            status, error = await self._execute(active, graph, options)
            await self._finalize(active, status, error)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("run %s: unhandled error in the run task", run_id)
        finally:
            active.close()
            self._runs.pop(run_id, None)

    async def _begin(self, active: _ActiveRun) -> bool:
        """Claim the row and announce the run. False means "do not run".

        The only way this fails is the row not existing any more — deleted
        out from under a submission — in which case there is nothing left to
        write events or a status to.
        """
        started = await self.store.mark_running(
            active.run_id, queue_key=active.context.device)
        if not started:
            logger.warning("run %s vanished before it could start",
                           active.run_id)
            return False
        await self._emit(active.run_id, EVENT_RUN_STARTED,
                         {"run_id": active.run_id})
        return True

    async def _execute(
        self, active: _ActiveRun, graph: dict[str, Any],
        options: dict[str, Any],
    ) -> tuple[str, str | None]:
        """Run the graph and classify the outcome. Never raises but cancel."""
        try:
            apply_seed(options.get("seed"))
            await execute_graph(
                graph["nodes"],
                graph["edges"],
                on_progress=self._progress_bridge(active, graph["nodes"]),
                context=active.context,
                # No ExecutionCache: nothing is re-executed within a run, and
                # sharing one ACROSS runs would serve a later run stale
                # tensors under a matching key.
                cache=None,
                run_id=active.run_id,
                output_store=self._output_store,
                record_outputs=active.record_outputs,
                preset_fallback=build_preset_fallback(graph["presets"]),
            )
            return STATUS_SUCCEEDED, None
        except CancellationError:
            # Why it stopped is a different fact from that it stopped: a user
            # pressing Stop is `cancelled`, the server going away is
            # `interrupted`. The engine raises the same exception for both.
            return ((STATUS_INTERRUPTED
                     if active.stop_reason == STOP_REASON_INTERRUPTED
                     else STATUS_CANCELLED), None)
        except GraphValidationError as exc:
            return STATUS_FAILED, str(exc)
        except Exception as exc:
            logger.warning("run %s failed: %s", active.run_id, exc,
                           exc_info=True)
            return STATUS_FAILED, str(exc)

    async def _finalize(
        self, active: _ActiveRun, status: str, error: str | None,
    ) -> None:
        """Terminal bookkeeping, in the one order that stays consistent.

        Metrics before the terminal event (so a reader that stops at the
        terminal event has the full series), the terminal event before the
        row (so an event-log follower never sees a finished row with no
        closing frame), and ``finished`` set before the row write so a long
        poller woken by the terminal event does not go back to sleep.

        ``terminating`` is set FIRST, before any await: from here the
        outcome is decided, and a cancel that arrives during the metric
        flush must not report that it cancelled a run about to file
        ``succeeded``.
        """
        active.terminating = True
        await self._flush_metrics(active, force=True)
        if status == STATUS_SUCCEEDED:
            await self._emit(active.run_id, EVENT_RUN_COMPLETED, None)
        elif status == STATUS_FAILED:
            await self._emit(active.run_id, EVENT_RUN_FAILED,
                             {"error": error or "run failed"})
        else:
            reason = (STOP_REASON_INTERRUPTED if status == STATUS_INTERRUPTED
                      else STOP_REASON_CANCELLED)
            await self._emit(active.run_id, EVENT_RUN_STOPPED,
                             {"reason": reason})
        active.finished = True

        # mark_finished guards on the ACTIVE statuses and reports whether it
        # landed, which is how a cancel racing a completion is resolved
        # (#119): the loser simply did not write, instead of a succeeded run
        # being filed forever as cancelled by a late writer.
        landed = await self.store.mark_finished(active.run_id, status,
                                                error=error)
        if not landed:
            logger.info(
                "run %s was already terminal when %s was reported",
                active.run_id, status)
        await self.prune_retention()

    def _progress_bridge(
        self, active: _ActiveRun, nodes: list[dict[str, Any]],
    ) -> Callable[[str, str, dict[str, Any] | None], Any]:
        """Build the engine's ``on_progress`` callback for one run.

        The message body is byte-for-byte what ``ws_execution`` puts on the
        wire minus its ``type`` key — see the event-vocabulary block above.
        """
        build_entries, declared_image_ports = _output_entry_builders()
        image_ports = declared_image_ports(nodes)

        async def on_progress(
            node_id: str, status: str, result: dict[str, Any] | None,
        ) -> None:
            message: dict[str, Any] = {"node_id": node_id, "status": status}
            if result and status == "error":
                message["error"] = result.get("error", "")
            entries = build_entries(status, result, image_ports.get(node_id, ()))
            if entries:
                message["outputs"] = entries
            await self._emit(active.run_id, EVENT_NODE_STATUS, message)
            if status == "progress":
                self._collect_metrics(active, node_id, result)
                await self._flush_metrics(active)

        return on_progress

    # ── events ────────────────────────────────────────────────────────────

    async def _emit(
        self, run_id: str, event_type: str, payload: Any,
    ) -> EventRecord | None:
        """Append one event durably, then hand it to live subscribers.

        Persist-then-fan-out, never the reverse: the store is the source of
        truth and a subscriber must not learn of an event that failed to
        land. A failed append is logged and swallowed — the event log is
        observability, and losing a progress row must not fail the run that
        was only trying to describe itself. ``CancelledError`` is re-raised
        untouched (it is the run stopping, not a logging problem); note that
        an append cancelled AFTER its COMMIT leaves a durable event whose
        cursor is lost here, which is safe precisely because every consumer
        re-reads the tail from the store rather than trusting this return.

        Subscribers receive EXACTLY the object that was stored — sanitised
        (``json_safe``: no NaN/Infinity, which are CPython tokens that
        ``JSON.parse`` rejects, and which would otherwise reach #121's
        WebSocket) and capped (``cap_event_payload``). The tempting
        alternative — store the capped copy, push the full one, since
        subscribers are in-process — is rejected because it would break the
        drop-tolerance contract: a subscriber is told that when it lags it
        may re-read the tail from the store "and lose nothing". If the live
        copy carried more than the stored one, a lagging consumer would
        silently get a DIFFERENT payload, so a UI would show an image live
        and a placeholder after any hiccup. One payload, one truth; #121 can
        revisit the split deliberately if it ever needs the full frame.
        """
        stamp = utc_now_iso()
        stored = cap_event_payload(json_safe(payload),
                                   cap_bytes=self._event_payload_cap_bytes)
        try:
            cursor = await self.store.append_event(run_id, event_type, stored,
                                                   ts=stamp)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("run %s: could not append %s event", run_id,
                           event_type, exc_info=True)
            self._wake(run_id)
            return None
        record = EventRecord(run_id=run_id, cursor=cursor, type=event_type,
                             payload=stored, ts=stamp)
        self._fan_out(run_id, record)
        return record

    def _fan_out(self, run_id: str, record: EventRecord) -> None:
        active = self._runs.get(run_id)
        if active is None:
            return
        for subscription in list(active.subscribers):
            subscription.offer(record)
        active.signal.notify()

    def _wake(self, run_id: str) -> None:
        active = self._runs.get(run_id)
        if active is not None:
            active.signal.notify()

    @asynccontextmanager
    async def subscribe(
        self, run_id: str, *, maxsize: int | None = None,
    ) -> AsyncIterator[RunSubscription]:
        """Live event feed for an in-process consumer (#121's WS bridge).

        A run that is already finished yields an immediately-closed
        subscription: there is nothing left to push, and the consumer should
        replay from ``RunStore.get_events``. Always used as a context manager
        so a dropped consumer cannot leak a queue onto a long-lived run.

        *maxsize* overrides the queue depth; ``0`` means unbounded, as
        ``asyncio.Queue`` defines it. (``maxsize or default`` would have
        quietly turned that request into the bounded default.)
        """
        depth = self._subscriber_queue_size if maxsize is None else maxsize
        subscription = RunSubscription(run_id=run_id,
                                       queue=asyncio.Queue(depth))
        active = self._runs.get(run_id)
        if active is None or active.finished:
            subscription.closed = True
            yield subscription
            return
        active.subscribers.add(subscription)
        try:
            yield subscription
        finally:
            active.subscribers.discard(subscription)

    async def wait_for_events(
        self,
        run_id: str,
        *,
        after_cursor: int = 0,
        limit: int = 500,
        wait: float = 0.0,
    ) -> list[EventRecord]:
        """Events after *after_cursor*, optionally long-polling for the next.

        The wake-up is edge-triggered (``_Broadcast``) — there is no sleep
        loop and no polling interval anywhere in this function. Order is
        load-bearing: the generation is captured BEFORE the store read, so an
        event that lands during the read still satisfies the wait instead of
        being missed until the next one.

        Returns as soon as anything is available, and immediately (possibly
        empty) once the run is finished or unknown — a long poll on a dead
        run must never hang for its full timeout.
        """
        deadline = time.monotonic() + max(wait, 0.0)
        while True:
            active = self._runs.get(run_id)
            waiter = None if active is None or active.finished \
                else active.signal.waiter()
            events = await self.store.get_events(
                run_id, after_cursor=after_cursor, limit=limit)
            if events or waiter is None:
                return events
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return []
            try:
                await asyncio.wait_for(waiter.wait(), remaining)
            except asyncio.TimeoutError:
                return []

    # ── metrics ───────────────────────────────────────────────────────────

    def _collect_metrics(
        self, active: _ActiveRun, node_id: str, payload: Any,
    ) -> None:
        step = metric_step(payload, fallback=active.last_metric_step + 1)
        active.last_metric_step = step
        active.pending_metrics.extend(
            scalar_metrics(payload, node_id=node_id, step=step))

    async def _flush_metrics(
        self, active: _ActiveRun, *, force: bool = False,
    ) -> int:
        """Write buffered points in ONE transaction (``log_metrics``).

        Never one row per point: a training loop emits several series per
        epoch and per-point inserts would mean a transaction (and an fsync)
        each. The buffer drains when it is big enough OR when it has been
        sitting for ``metric_flush_interval_s`` — the second rule is what
        keeps a live chart current on a slow producer, checked on the emit
        path so no timer task is needed. ``force`` drains it at terminal.
        """
        if not active.pending_metrics:
            return 0
        now = time.monotonic()
        if (not force
                and len(active.pending_metrics) < self._metric_flush_max_points
                and now - active.last_flush_monotonic
                < self._metric_flush_interval_s):
            return 0
        batch, active.pending_metrics = active.pending_metrics, []
        active.last_flush_monotonic = now
        try:
            return await self.store.log_metrics(active.run_id, batch)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("run %s: metric flush failed (%d point(s) lost)",
                           active.run_id, len(batch), exc_info=True)
            return 0

    # ── cancel ────────────────────────────────────────────────────────────

    async def cancel(self, run_id: str) -> CancelOutcome | None:
        """Ask a run to stop. ``None`` when the run does not exist.

        COOPERATIVE, never ``Task.cancel``: the ``ExecutionContext`` flag is
        set and the engine unwinds at its next checkpoint (between nodes and
        between levels). A node already executing in a worker thread runs to
        completion first — a training node with a 90-second epoch stops after
        that epoch, not mid-backward — because there is no safe way to
        interrupt arbitrary third-party node code, and killing the task
        outright is what leaves half-written rows and wedged CUDA state.
        Cancellation is therefore ACKNOWLEDGED here and OBSERVED on the row.

        A run that never started (a ``queued`` row — #123's lane, or an
        orphan) is retired directly through the guarded ``mark_finished``, so
        a cancel racing a start cannot rewrite a run that is already going.

        The ``terminating`` check is what keeps the answer truthful once a
        run's outcome is decided but it is still in the registry — from the
        first line of ``_finalize`` (before the metric flush and the
        terminal event, both of which await) until the task deregisters
        itself. A registry hit alone would report "cancelled: true" for a
        run that is on its way to filing ``succeeded``.
        """
        active = self._runs.get(run_id)
        if active is not None and not active.terminating:
            if active.stop_reason is None:
                active.stop_reason = STOP_REASON_CANCELLED
            active.context.cancel()
            return CancelOutcome(run_id=run_id, status=STATUS_RUNNING,
                                 cancelled=True)

        record = await self.store.get_run(run_id)
        if record is None:
            return None
        if record.status == STATUS_QUEUED:
            landed = await self.store.mark_finished(
                run_id, STATUS_CANCELLED, expected=(STATUS_QUEUED,))
            if landed:
                return CancelOutcome(run_id=run_id, status=STATUS_CANCELLED,
                                     cancelled=True)
            record = await self.store.get_run(run_id) or record
        return CancelOutcome(run_id=run_id, status=record.status,
                             cancelled=False)

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def recover_interrupted(self) -> int:
        """Retire rows a dead process left active. STARTUP ONLY.

        Nothing resumes a ``running`` row after a restart, so leaving one
        claiming to be alive is both a lie to the user and an unbounded
        metrics leak: retention never deletes an active run, so an
        abandoned one would keep its events and metrics forever. This is why
        it must run BEFORE ``prune_retention``.

        Guarded rather than merely documented — calling it while this process
        has live runs would mark those very runs interrupted underneath
        themselves.
        """
        if self._runs:
            raise RuntimeError(
                "recover_interrupted is a startup call; "
                f"{len(self._runs)} active run(s) are in flight")
        count = await self.store.interrupt_active_runs()
        if count:
            logger.warning(
                "marked %d run(s) interrupted (they did not survive a "
                "restart)", count)
        return count

    async def prune_retention(self) -> int:
        """Apply the keep-last-N retention policy. ``None`` disables it."""
        keep_last = self._retention_keep_last
        if keep_last is None or keep_last < 0:
            return 0
        deleted = await self.store.prune(keep_last=keep_last)
        if deleted:
            logger.info(
                "run retention: removed %d finished run(s), keeping the "
                "newest %d", deleted, keep_last)
        return deleted

    async def shutdown(self) -> None:
        """Stop every run and DRAIN its task. Must complete before db.close().

        The #119 carry-forward this closes: ``Database.close`` races any
        merely-slow ``db.run`` that is still in flight, and the run tasks are
        the only thing in the process that issues database work nobody is
        awaiting. Every one of them is awaited to completion here, so by the
        time this returns there is no worker thread left holding the
        connection.

        Two phases, in this order:

        1. Cooperative. The stop reason is ``interrupted`` (the server is
           going away, the user did not ask), so the tasks write an honest
           terminal row on their way out.
        2. Hard, after ``shutdown_grace_s``. ``Task.cancel`` on whatever is
           still stuck in a long node; those rows stay ``running`` and the
           next boot's ``recover_interrupted`` retires them. The cancelled
           tasks are still awaited — cancelling is not draining.
        """
        self._shutting_down = True
        active = list(self._runs.values())
        for entry in active:
            # First reason wins: a run the user already cancelled is filed as
            # cancelled even if the server goes down before it unwinds. Their
            # intent is the more specific fact, and it came first.
            if entry.stop_reason is None:
                entry.stop_reason = STOP_REASON_INTERRUPTED
            entry.context.cancel()
        tasks = [entry.task for entry in active if entry.task is not None]
        if not tasks:
            return
        logger.info("shutdown: draining %d in-flight run(s)", len(tasks))
        _done, pending = await asyncio.wait(tasks,
                                            timeout=self._shutdown_grace_s)
        if pending:
            logger.warning(
                "shutdown: %d run(s) did not stop within %.1fs; cancelling "
                "(they will be marked interrupted on the next start)",
                len(pending), self._shutdown_grace_s)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
