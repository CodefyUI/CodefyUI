"""Where a run's graph executes: the Executor seam (#766).

``RunService`` decides WHEN a run executes (the queue, admission, the
lanes) and owns everything that outlives it (the row, the event log, the
metric batcher). This module is the part in between: given one run that is
ready to go, execute its graph and report how it ended.

Three pieces:

- :class:`RunSinks`: the loop-side persistence a run reports into. Built by
  ``RunService`` per run; every callable lands in the service (``_emit``, the
  metric batcher, the artifact row). An executor never touches the store.
- :class:`ExecutionRequest`: everything one execution needs, gathered by
  ``RunService._execute``.
- :class:`Executor`: ``run(request) -> ExecutionOutcome``. Today there is one,
  :class:`InProcessExecutor`, which is ``RunService._execute``'s body as it was
  before #766, unchanged: it calls ``execute_graph`` in this process and turns
  its callbacks into payloads with ``core.run_events`` and into rows with the
  sinks.

The split exists so a run can execute somewhere else (a worker subprocess,
#771/#772) and feed the same payload builders and the same sinks, without
the service learning where the graph ran.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol

from .device_utils import split_device
from .execution_context import (
    ArtifactSignal,
    CancellationError,
    DroppedSignal,
    ExecutionContext,
    MetricSignal,
    WarningSignal,
)
from .graph_engine import GraphValidationError, build_preset_fallback, execute_graph
from .output_entries import declared_media_ports
from .run_events import (
    DEFAULT_ERROR_MODE,
    EVENT_INNER_NODE_STATUS,
    EVENT_NODE_STATUS,
    EVENT_WARNING,
    MediaPorts,
    dropped_signals_payload,
    inner_node_status_payload,
    metric_point,
    metric_points_from_progress,
    node_status_payload,
)
from .run_output_store import RunOutputStore
from .run_store import (
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_SUCCEEDED,
    MetricPoint,
)
from .seeding import seed_rngs

if TYPE_CHECKING:  # pragma: no cover - typing only; run_service imports us
    from .run_service import InteractiveSession
    from .scheduling import Placement
    from .workers import WorkerRegistry

logger = logging.getLogger(__name__)


@dataclass
class RunSinks:
    """Where one run's reports land. Loop-side, owned by ``RunService``.

    Every callable is a coroutine function and is awaited on the event loop,
    in the order the engine delivered the callback it came from.

    - ``emit(event_type, payload)``: append one event and fan it out
      (``RunService._emit``; capped and sanitised there).
    - ``record_metrics(points)``: buffer points and let the batcher decide
      whether to flush. Called with an EMPTY list too, on every progress
      frame: the batcher's time rule is checked on that path.
    - ``record_artifact(signal)``: store the artifact row, then announce it.
    - ``warn(signal)``: a ``WarningSignal`` a node sent, as a ``run_warning``.
    """

    emit: Callable[[str, dict[str, Any] | None], Awaitable[Any]]
    record_metrics: Callable[[list[MetricPoint]], Awaitable[Any]]
    record_artifact: Callable[[ArtifactSignal], Awaitable[Any]]
    warn: Callable[[WarningSignal], Awaitable[Any]]
    #: Merge a patch into the run's ``execution_facts`` (#772: what a worker
    #: reports about where the run executes). None: nothing records them.
    record_execution_facts: (
        Callable[[dict[str, Any]], Awaitable[Any]] | None) = None


@dataclass
class ExecutionRequest:
    """One run, ready to execute. Built by ``RunService._execute``.

    ``explicit_metric_nodes`` is the run's own set (``_ActiveRun``'s), shared
    rather than copied: a node that logs an explicit metric is added to it
    here, and from then on its progress payloads are not mined (#122).
    """

    run_id: str
    graph: dict[str, Any]
    options: dict[str, Any]
    context: ExecutionContext
    session: "InteractiveSession | None"
    output_store: RunOutputStore | None
    record_outputs: bool
    sinks: RunSinks
    explicit_metric_nodes: set[str]
    #: Where the scheduler put the run (#768); ``WorkerExecutor`` reads its
    #: worker and devices. None on the interactive lane.
    placement: "Placement | None" = None


@dataclass(frozen=True)
class ExecutionOutcome:
    """How an execution ended: a terminal run status and, when failed, why.

    ``status`` is ``succeeded``, ``failed`` or ``cancelled``. ``cancelled``
    means "the engine stopped cooperatively"; whether that was a user or the
    server going away is the service's to say (``_stopped_status``).
    """

    status: str
    error: str | None = None
    #: The run's device memory (#770), merged into the row's
    #: ``execution_facts.peak_memory`` by ``RunService._finalize``: a map of
    #: ``"cuda:N"`` to its peaks, or ``"mps"`` to what the driver held at
    #: the end; None when no card holds memory (a plain CPU run), without
    #: torch or a CUDA context, or when the allocator could not be read. See :meth:`PeakMemoryScope.read`.
    peak_memory: dict[str, Any] | None = None


class Executor(Protocol):
    """Executes one run's graph. Never raises but a hard ``Task.cancel``."""

    async def run(self, request: ExecutionRequest) -> ExecutionOutcome: ...


class InProcessExecutor:
    """Execute the graph in this process, on this event loop's threads.

    ``RunService._execute``'s body before #766, moved verbatim: the bridges
    now build their payloads with ``core.run_events`` and persist through
    ``request.sinks``, and the outcome comes back as an
    :class:`ExecutionOutcome` instead of a tuple.
    """

    async def run(self, request: ExecutionRequest) -> ExecutionOutcome:
        """Run the graph, classify the outcome and attach its peak memory.

        Never raises but cancel. The peak is read whatever the outcome: a
        run that failed with an out-of-memory error is the one whose number
        matters most. ``exact`` is False here: in-process runs share one
        CUDA context (see :class:`PeakMemoryScope`).
        """
        scope = PeakMemoryScope(request.context.device).open()
        try:
            outcome = await self._execute(request)
            return replace(outcome, peak_memory=scope.read(exact=False))
        finally:
            scope.close()

    async def _execute(self, request: ExecutionRequest) -> ExecutionOutcome:
        """Run the graph and classify the outcome. Never raises but cancel."""
        graph = request.graph
        options = request.options
        session = request.session
        media_ports = declared_media_ports(graph["nodes"])
        on_progress = _progress_bridge(request, media_ports)
        try:
            # The RUN-level baseline (``seeding.seed_rngs``): anything that
            # draws outside a node's ``execute`` (preset expansion, a
            # validator) is covered by it. What makes a run reproducible is
            # the per-node derivation in ``graph_engine`` (#134); this call
            # is process-global, as it always was.
            seed_rngs(options.get("seed"))
            # Determinism is NOT applied here any more: ``execute_graph``
            # owns it through ``deterministic_scope``, which also takes it
            # back when the run ends. Setting it here would latch the
            # process outside that scope — the one-way latch the #188
            # review found.
            await execute_graph(
                graph["nodes"],
                graph["edges"],
                on_progress=on_progress,
                on_inner_status=_inner_status_bridge(
                    request, on_progress, media_ports),
                # The nodes inside blocks and preset cards declare media too:
                # read once, from the node list the run executes (#559).
                on_expanded=lambda expanded: media_ports.update(
                    declared_media_ports(expanded)),
                on_signal=_signal_bridge(request),
                context=request.context,
                error_mode=options.get("error_mode", DEFAULT_ERROR_MODE),
                max_retries=int(options.get("max_retries", 0)),
                # No ExecutionCache by default: nothing is re-executed
                # within a run, and sharing one ACROSS server-owned runs
                # would serve a later run stale tensors under a matching
                # key. The canvas lends its own — one per socket, cleared by
                # the same reconnect that always cleared it — because
                # incremental re-execution IS the editing loop.
                cache=None if session is None else session.cache,
                changed_nodes=(None if session is None
                               else list(session.changed_nodes) or None),
                run_id=request.run_id,
                output_store=request.output_store,
                record_outputs=request.record_outputs,
                preset_fallback=build_preset_fallback(graph["presets"]),
                subgraphs=graph["subgraphs"],
            )
            return ExecutionOutcome(STATUS_SUCCEEDED, None)
        except CancellationError:
            # Why it stopped is a different fact from that it stopped: a user
            # pressing Stop is `cancelled`, the server going away is
            # `interrupted`. The engine raises the same exception for both,
            # so this reports `cancelled` and the service tells them apart.
            return ExecutionOutcome(STATUS_CANCELLED, None)
        except GraphValidationError as exc:
            return ExecutionOutcome(STATUS_FAILED, str(exc))
        except Exception as exc:
            logger.warning("run %s failed: %s", request.run_id, exc,
                           exc_info=True)
            return ExecutionOutcome(STATUS_FAILED, str(exc))


# ── peak memory (#770) ────────────────────────────────────────────────────
#
# torch is read off ``sys.modules`` rather than imported: this module never
# imported it, and a run that never loaded torch has allocated no device
# memory, so "not loaded" is the honest null rather than a reason to pay
# for the import. Nothing here initialises CUDA either: every CUDA call is
# behind ``torch.cuda.is_initialized()``.
#
# CUDA's peak counters are per PROCESS and per card, so in-process runs
# share them. ``_ACTIVE_SCOPES`` is the set of in-process runs currently
# measuring, by the card they run on. It is only touched on the event loop,
# between awaits, so it needs no lock. A run resets a card's counters only
# when no other in-process run is on that card; otherwise it leaves them
# alone (resetting would wipe the other run's peak) and records the card as
# ``shared``.


class PeakMemoryScope:
    """One run's peak-memory measurement. ``open`` -> ``read`` -> ``close``.

    Opened before the graph executes, read after it (whatever the outcome),
    closed in a ``finally`` so a hard cancel still unregisters it.
    """

    def __init__(self, device: str | None) -> None:
        self.kind, index = split_device(device or "")
        self.card: int | None = None
        #: Another in-process run shared this run's card at some point.
        self.shared = False
        #: Counters were all zero when this run started (no CUDA context).
        self.fresh = False
        #: Cards whose counters this run reset at its start.
        self.reset_cards: set[int] = set()
        self._index = index

    def open(self) -> "PeakMemoryScope":
        """Register on the run's card and reset what is safe to reset."""
        if self.kind == "mps":
            return self
        torch = sys.modules.get("torch")
        initialised = _cuda_initialised(torch)
        others = [scope for scope in _ACTIVE_SCOPES if scope is not self]
        # A CPU run has no card of its own (``card`` stays None) but is
        # measured the same way: a node can pin a CUDA card.
        if self.kind == "cuda":
            # A bare ``cuda`` is whichever card torch points at. Asking
            # would initialise CUDA; without a context nothing has moved it
            # off 0.
            self.card = self._index
            if self.card is None:
                self.card = (int(torch.cuda.current_device()) if initialised
                             else 0)
            for scope in others:
                if scope.card == self.card:
                    scope.shared = True
                    self.shared = True
        _ACTIVE_SCOPES.append(self)
        if not initialised:
            self.fresh = torch is not None
            return self
        if self.shared:
            return self
        busy = {scope.card for scope in others}
        try:
            for card in range(int(torch.cuda.device_count())):
                if card not in busy:
                    torch.cuda.reset_peak_memory_stats(card)
                    self.reset_cards.add(card)
        except Exception:  # noqa: BLE001 - observability never fails a run
            logger.debug("could not reset the CUDA peak memory statistics",
                         exc_info=True)
        return self

    def close(self) -> None:
        """Unregister. Idempotent."""
        if self in _ACTIVE_SCOPES:
            _ACTIVE_SCOPES.remove(self)

    def read(self, *, exact: bool) -> dict[str, Any] | None:
        """The run's device memory, as ``execution_facts.peak_memory``.

        - CUDA: one ``"cuda:N"`` entry per card this process holds memory
          on (a nonzero allocation peak, or ``memory_reserved > 0`` on a
          card whose counters were not reset), the run's own card
          first, so a node pinned to another card is counted too. Each is
          ``{"allocated_bytes", "reserved_bytes", "exact"}`` from
          ``max_memory_allocated`` / ``max_memory_reserved``, plus
          ``"shared": true`` when the number is not this run's alone:
          another in-process run used the card while this one ran, or its
          counters could not be reset at the start (they then still hold
          an earlier peak).
        - MPS: ``{"mps": {"held_bytes", "exact"}}``,
          ``torch.mps.driver_allocated_memory()`` at the END of the run:
          everything the Metal driver holds for this process, the
          allocator's cache included. MPS has no peak counter, so this is
          not a peak and not a bound on one.
        - A CPU run is read like a CUDA one, with no card of its own: a
          node that pinned a CUDA card puts that card in the map.
        - torch never loaded, CUDA never initialised, no card holding
          memory (a plain CPU run), or a read failing: None.

        *exact* is the caller's to say: in-process runs share one CUDA
        context and pass False; a run alone in its own process (the worker
        subprocess, #771) can pass True. A ``shared`` card is never exact.
        """
        torch = sys.modules.get("torch")
        if torch is None:
            return None
        try:
            if self.kind == "mps":
                backend = getattr(torch.backends, "mps", None)
                if backend is None or not backend.is_available():
                    return None
                return {"mps": {
                    "held_bytes": int(torch.mps.driver_allocated_memory()),
                    "exact": False,
                }}
            if not _cuda_initialised(torch):
                return None
            count = int(torch.cuda.device_count())
            order = sorted(range(count), key=lambda card: card != self.card)
            busy = {scope.card for scope in _ACTIVE_SCOPES
                    if scope is not self}
            entries: dict[str, Any] = {}
            for card in order:
                allocated = int(torch.cuda.max_memory_allocated(card))
                # On a card reset at the start, reserved memory with no
                # allocation since is cache an EARLIER run left behind (the
                # caching allocator keeps it), not this run's use.
                if allocated <= 0 and (
                        card in self.reset_cards
                        or torch.cuda.memory_reserved(card) <= 0):
                    continue
                shared = (
                    (self.shared and card == self.card)
                    or card in busy
                    or (not self.fresh and card not in self.reset_cards)
                )
                entry: dict[str, Any] = {
                    "allocated_bytes": allocated,
                    "reserved_bytes": int(
                        torch.cuda.max_memory_reserved(card)),
                    "exact": bool(exact) and not shared,
                }
                if shared:
                    entry["shared"] = True
                entries[f"cuda:{card}"] = entry
            return entries or None
        except Exception:  # noqa: BLE001 - observability never fails a run
            logger.debug("could not read the %s peak memory", self.kind,
                         exc_info=True)
            return None


#: In-process runs currently measuring (see the block comment above).
_ACTIVE_SCOPES: list[PeakMemoryScope] = []


def _cuda_initialised(torch: Any) -> bool:
    """True when *torch* has a live CUDA context. Never initialises one."""
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_initialized())
    except Exception:  # noqa: BLE001 - no driver, a broken build
        return False


# ── engine bridges ────────────────────────────────────────────────────────
#
# Each one is the engine callback for one run: a ``run_events`` builder for
# the payload, then a sink for the persistence. Moved out of ``RunService``
# by #766, which is where their history lives.


def _progress_bridge(
    request: ExecutionRequest, media_ports: MediaPorts,
) -> Callable[..., Any]:
    """Build the engine's ``on_progress`` callback for one run.

    One ``node_status`` event per frame (see
    :func:`run_events.node_status_payload` for its shape). A ``progress``
    frame is also mined for metrics (#122) unless its node logs its own;
    the step counter is this run's, starting at 0.
    """
    sinks = request.sinks
    last_metric_step = 0

    async def on_progress(
        node_id: str, status: str, result: dict[str, Any] | None,
        inner_node_id: str | None = None,
    ) -> None:
        nonlocal last_metric_step
        await sinks.emit(EVENT_NODE_STATUS, node_status_payload(
            node_id, status, result, media_ports,
            inner_node_id=inner_node_id))
        if status == "progress":
            points, last_metric_step = metric_points_from_progress(
                node_id, result,
                explicit_metric_nodes=request.explicit_metric_nodes,
                last_step=last_metric_step)
            await sinks.record_metrics(points)

    return on_progress


def _inner_status_bridge(
    request: ExecutionRequest,
    on_progress: Callable[..., Any],
    media_ports: MediaPorts,
) -> Callable[..., Any]:
    """Build the engine's ``on_inner_status`` callback for one run (#559).

    A status becomes one ``inner_node_status`` event (see
    :func:`run_events.inner_node_status_payload`). A progress frame goes to
    ``on_progress`` as the card's, naming the inner node, so its metrics are
    filed under the card as before.
    """
    sinks = request.sinks

    async def on_inner_status(
        node_id: str, container_id: str, status: str,
        result: dict[str, Any] | None,
    ) -> None:
        if status == "progress":
            await on_progress(container_id, status, result,
                              inner_node_id=node_id)
            return
        await sinks.emit(EVENT_INNER_NODE_STATUS, inner_node_status_payload(
            node_id, container_id, status, result, media_ports))

    return on_inner_status


def _signal_bridge(request: ExecutionRequest) -> Callable[[Any], Any]:
    """Build the engine's ``on_signal`` callback for one run (#122).

    Everything a node reports that is not a node status arrives here,
    already serialised onto the loop by the engine's single pump — so
    this can await the database without a lock and without reordering
    anything.

    Nothing in here is allowed to raise. A metric, an artifact row and a
    dropped-signal notice are all observability; a run must not fail
    because the thing describing it did.
    """
    sinks = request.sinks

    async def on_signal(signal: Any) -> None:
        try:
            if isinstance(signal, MetricSignal):
                # Claim the node BEFORE recording: from this point its
                # progress payloads are not mined (#122).
                request.explicit_metric_nodes.add(signal.node_id or "")
                await sinks.record_metrics([metric_point(signal)])
            elif isinstance(signal, ArtifactSignal):
                await sinks.record_artifact(signal)
            elif isinstance(signal, DroppedSignal):
                await sinks.emit(EVENT_WARNING,
                                 dropped_signals_payload(signal))
            elif isinstance(signal, WarningSignal):
                await sinks.warn(signal)
            else:  # pragma: no cover - a node pack's own signal type
                logger.debug("run %s: ignoring unknown signal %r",
                             request.run_id, type(signal).__name__)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("run %s: could not record a %s", request.run_id,
                           type(signal).__name__, exc_info=True)

    return on_signal


# ── WorkerExecutor (#772) ─────────────────────────────────────────────────

#: How often the cancel watcher looks at the run's stop flag. The flag is a
#: ``threading.Event`` with no loop hook; a fifth of a second is invisible
#: next to a run subprocess's cancel grace.
CANCEL_POLL_S = 0.2

#: ``run_warning`` kind when a run ends because its worker went away.
WARNING_WORKER_DISCONNECTED = "worker_disconnected"


class WorkerExecutor:
    """Execute the graph on a worker agent (#771) over ``/ws/worker``.

    The run's placement names the worker and its devices. ``run`` sends
    ``assign`` (the graph arrives with its secret params restored; the
    worker has no vault), then turns the worker's frames for the run into
    the same sink calls ``InProcessExecutor`` makes:

    - ``run_started`` -> its ``execution_facts`` (where the run executes,
      #774) merged into the row's;
    - ``event`` -> ``sinks.emit`` (``RunService._emit`` caps and sanitises
      it again, which changes nothing: the run subprocess applied the same
      policy), then ``sinks.record_metrics([])``, the batcher's clock check
      the in-process bridge makes on every progress frame;
    - ``metrics`` -> ``sinks.record_metrics``; ``artifact`` ->
      ``sinks.record_artifact``; ``warning`` -> ``sinks.warn``;
    - ``run_finished`` -> the outcome. ``succeeded`` / ``failed`` /
      ``cancelled`` map one to one (a crash's output tail is appended to its
      error). The agent's ``interrupted`` (the subprocess ended without
      reporting, after a cancel) is a stop when THIS server asked for one,
      so ``cancelled`` and the service's ``_stopped_status`` decides; with
      no stop asked it is ``interrupted``, with the agent's error.

    A stop is watched for (``context.should_stop()``) and forwarded as a
    ``cancel``. A worker that goes away ends the run after a ``run_warning``
    saying so: ``interrupted``, or ``cancelled`` when a stop had been asked
    (a user's Stop stays the user's).

    Every send looks the worker up in the registry AT SEND TIME: a
    reconnect replaces the ``Worker`` object, and a frame sent through the
    old one would go to a closed socket.
    """

    def __init__(self, registry: "WorkerRegistry") -> None:
        self._registry = registry

    async def _send(self, worker_id: str, frame: Any) -> bool:
        """Send *frame* to *worker_id*'s current connection. False if none."""
        worker = self._registry.get(worker_id)
        if worker is None or not worker.online:
            return False
        await worker.send(frame)
        return True

    async def run(self, request: ExecutionRequest) -> ExecutionOutcome:
        from . import worker_protocol as wp
        from .workers import WORKER_GONE

        placement = request.placement
        if placement is None:
            return ExecutionOutcome(STATUS_FAILED,
                                    "a worker run needs a placement")
        worker_id = placement.worker_id
        worker = self._registry.get(worker_id)
        if worker is None or not worker.online:
            detail = f"worker {worker_id} is not connected"
            await _worker_gone_warning(request, detail)
            return self._gone(request, detail)
        devices = list(placement.devices)
        run = self._registry.open_run(worker, request.run_id)
        watcher: asyncio.Task | None = None
        try:
            await worker.send(wp.Assign(
                run_id=request.run_id, graph=request.graph,
                options=request.options, device=devices[0],
                devices=devices))
            watcher = asyncio.create_task(
                self._forward_cancel(request, worker_id),
                name=f"cancel-watch:{request.run_id}")
            while True:
                item = await run.inbox.get()
                if isinstance(item, tuple) and item and item[0] is WORKER_GONE:
                    detail = f"worker {worker_id} went away: {item[1]}"
                    await _worker_gone_warning(request, detail)
                    return self._gone(request, detail)
                outcome = await self._deliver(request, item, devices)
                if outcome is not None:
                    return outcome
        except asyncio.CancelledError:
            # A hard Task.cancel (shutdown's last resort): tell the worker,
            # best effort, so the subprocess does not outlive its row.
            try:
                await self._send(worker_id, wp.Cancel(
                    run_id=request.run_id, grace_s=0))
            except Exception:  # noqa: BLE001
                logger.warning("run %s: could not tell worker %s to stop",
                               request.run_id, worker_id, exc_info=True)
            raise
        except Exception as exc:  # noqa: BLE001 -- e.g. the socket closed under send
            logger.warning("run %s: worker %s failed: %s", request.run_id,
                           worker_id, exc, exc_info=True)
            detail = f"could not reach worker {worker_id}: {exc}"
            await _worker_gone_warning(request, detail)
            return self._gone(request, detail)
        finally:
            if watcher is not None:
                watcher.cancel()
            self._registry.close_run(worker_id, request.run_id)

    @staticmethod
    def _gone(request: ExecutionRequest, detail: str) -> ExecutionOutcome:
        """The outcome of a run whose worker is gone: a stop if one was
        asked (``_stopped_status`` keeps a user's Stop ``cancelled``),
        else ``interrupted``."""
        if request.context.should_stop():
            return ExecutionOutcome(STATUS_CANCELLED, None)
        return ExecutionOutcome(STATUS_INTERRUPTED, detail)

    async def _forward_cancel(self, request: ExecutionRequest,
                              worker_id: str) -> None:
        """Watcher task: forward the run's stop as ``cancel``. Never raises."""
        from . import worker_protocol as wp

        try:
            while not request.context.should_stop():
                await asyncio.sleep(CANCEL_POLL_S)
            if not await self._send(worker_id,
                                    wp.Cancel(run_id=request.run_id)):
                logger.info("run %s: stop requested, but worker %s is not "
                            "connected", request.run_id, worker_id)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- the run ends via WORKER_GONE
            logger.warning("run %s: could not forward the stop to worker %s",
                           request.run_id, worker_id, exc_info=True)

    async def _deliver(self, request: ExecutionRequest, frame: Any,
                       devices: list[str]) -> ExecutionOutcome | None:
        """Hand one frame to the sinks; the outcome on ``run_finished``."""
        from . import worker_protocol as wp

        sinks = request.sinks
        if isinstance(frame, wp.RunStarted):
            if frame.execution_facts and sinks.record_execution_facts:
                try:
                    await sinks.record_execution_facts(frame.execution_facts)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 -- observability
                    logger.warning("run %s: could not record the worker's "
                                   "execution facts", request.run_id,
                                   exc_info=True)
        elif isinstance(frame, wp.Event):
            await sinks.emit(frame.event_type, frame.payload)
            await sinks.record_metrics([])
        elif isinstance(frame, wp.Metrics):
            await sinks.record_metrics([
                MetricPoint(name=p.name, value=p.value, step=p.step,
                            node_id=p.node_id) for p in frame.points])
        elif isinstance(frame, wp.Artifact):
            await sinks.record_artifact(ArtifactSignal(
                kind=frame.kind, path=frame.path, meta=frame.meta,
                node_id=frame.node_id))
        elif isinstance(frame, wp.RunWarning):
            await sinks.warn(WarningSignal(kind=frame.kind,
                                           detail=frame.detail,
                                           node_id=frame.node_id))
        elif isinstance(frame, wp.RunFinished):
            return self._outcome(frame, devices,
                                 stop_asked=request.context.should_stop())
        return None

    @staticmethod
    def _outcome(frame: Any, devices: list[str], *,
                 stop_asked: bool) -> ExecutionOutcome:
        peak = _physical_peak(frame.peak_memory, devices)
        if frame.status == STATUS_SUCCEEDED:
            return ExecutionOutcome(STATUS_SUCCEEDED, None, peak_memory=peak)
        if frame.status == STATUS_CANCELLED or (
                frame.status == STATUS_INTERRUPTED and stop_asked):
            return ExecutionOutcome(STATUS_CANCELLED, None, peak_memory=peak)
        error = frame.error or "the run failed on its worker"
        if frame.stderr_tail:
            error = f"{error}\n--- output tail ---\n{frame.stderr_tail}"
        if frame.status == STATUS_INTERRUPTED:
            return ExecutionOutcome(STATUS_INTERRUPTED, error,
                                    peak_memory=peak)
        return ExecutionOutcome(STATUS_FAILED, error, peak_memory=peak)


async def _worker_gone_warning(request: ExecutionRequest, detail: str) -> None:
    try:
        await request.sinks.emit(EVENT_WARNING, {
            "kind": WARNING_WORKER_DISCONNECTED, "detail": detail,
            "node_id": None})
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 -- observability
        logger.warning("run %s: could not record %s", request.run_id, detail)


def _physical_peak(peak: Any, devices: list[str]) -> Any:
    """A worker's peak-memory map with its card keys renamed back to this
    worker's numbering: the run subprocess saw only its set, so its
    ``cuda:0`` is the set's first CUDA card."""
    if not isinstance(peak, dict):
        return None
    cuda = [d for d in devices if d.startswith("cuda:")]
    renamed: dict[str, Any] = {}
    for key, value in peak.items():
        if isinstance(key, str) and key.startswith("cuda:"):
            try:
                key = cuda[int(key.split(":", 1)[1])]
            except (ValueError, IndexError):
                pass
        renamed[key] = value
    return renamed
