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
- :class:`Executor`: ``run(request) -> ExecutionOutcome``. Two exist.
  :class:`InProcessExecutor`, the default, is ``RunService._execute``'s body
  as it was before #766, unchanged: it calls ``execute_graph`` in this
  process and turns its callbacks into payloads with ``core.run_events`` and
  into rows with the sinks. :class:`WorkerExecutor` (#772) hands a run
  placed on a worker agent to that worker and feeds its frames into the same
  sinks.

The split exists so a run can execute somewhere else (a worker subprocess,
#771/#772) and feed the same payload builders and the same sinks, without
the service learning where the graph ran.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from collections.abc import Mapping
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


@dataclass(frozen=True)
class MountedArtifactSignal(ArtifactSignal):
    """An ``ArtifactSignal`` from a worker whose ``path`` is relative to the
    mount ``mount_label`` names (#782). ``RunService`` stores it as a
    ``mount`` row. A subclass, so ``ArtifactSignal`` itself (part of the
    runtime exported scripts import) stays as it is."""

    mount_label: str | None = None


@dataclass(frozen=True)
class TransferredArtifactSignal(ArtifactSignal):
    """An ``ArtifactSignal`` for a file a worker uploaded to this server
    (#785): ``path`` is the key the upload answered with
    (``core.transfer.key_location``). ``RunService`` stores a ``transfer``
    row. A subclass, like :class:`MountedArtifactSignal`. ``backend`` is
    ``s3`` (#791) for a file a worker uploaded to the run's bucket instead:
    ``path`` is then its object key, ``bucket`` the bucket and ``prefix``
    the run's key prefix it was assigned (both recorded with the row, so a
    later settings change does not lose the object)."""

    backend: str = "transfer"
    bucket: str | None = None
    prefix: str | None = None


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
    #: ``record_worker_connection(False)`` when the run's worker dropped
    #: and the run waits for it, ``(True)`` when it is back (#783): the
    #: row's ``disconnected`` <-> ``running``. None: nothing records it.
    record_worker_connection: (
        Callable[[bool], Awaitable[Any]] | None) = None
    #: ``record_worker_seq(seq)`` before a worker run's frames up to *seq*
    #: are acknowledged (#793): make what they carried durable (the metric
    #: buffer) and persist *seq*, so a restarted server resumes the stream
    #: after it. None: nothing is persisted.
    record_worker_seq: Callable[[int], Awaitable[Any]] | None = None
    #: Store one node's provenance record (#344, ``core.node_provenance``)
    #: for a run with the ``provenance`` option. None: nothing records it,
    #: and the executor collects none.
    record_provenance: (
        Callable[[dict[str, Any]], Awaitable[Any]] | None) = None
    #: ``record_environment(document, numerics)``: the environment the
    #: run's worker subprocess reported (#344): its fingerprint document
    #: (or None) and its torch numerics switches. None: nothing records it.
    record_environment: (
        Callable[[dict[str, Any] | None, dict[str, Any] | None],
                 Awaitable[Any]] | None) = None


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
    #: The node types the run executes, computed once at submit for a run
    #: that targets a worker (#781); None when not computed.
    node_types: frozenset[str] | None = None
    #: ``run_needs.needs`` of a worker run (#782): which storage backend
    #: its assignment uses (#785). None for a run here.
    needs: dict[str, Any] | None = None
    #: A worker run adopted after a restart of this server (#793): the
    #: last frame its previous process recorded. ``WorkerExecutor`` sends
    #: no ``assign`` and waits for the worker to report the run
    #: (``WorkerRegistry.adopt``). None: a run that starts now.
    resume_seq: int | None = None
    #: An adopted ``s3`` run's ``execution_facts.worker_s3`` (bucket,
    #: prefix, input keys, #791): its worker is issued new credentials.
    resume_s3: dict[str, Any] | None = None
    #: Where the run's memory budget came from (#794): ``declared``,
    #: ``history`` or ``estimate`` (``memory_admission.SOURCE_*``), for the
    #: note an out-of-memory failure under that budget gets. None: no budget.
    memory_budget_source: str | None = None


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
    #: ``{code, params}`` of *error* when it is a coded message (#841: a
    #: worker run's "worker X went away: ..."), stored as the row's
    #: ``execution_facts.error_reason`` so the Runs panel can word it.
    error_reason: dict[str, Any] | None = None


class Executor(Protocol):
    """Executes one run's graph. Never raises but a hard ``Task.cancel``."""

    async def run(self, request: ExecutionRequest) -> ExecutionOutcome: ...


class InProcessExecutor:
    """Execute the graph in this process, on this event loop's threads.

    ``RunService._execute``'s body before #766, moved verbatim: the bridges
    now build their payloads with ``core.run_events`` and persist through
    ``request.sinks``, and the outcome comes back as an
    :class:`ExecutionOutcome` instead of a tuple.

    *exclusive_process* says this process executes this run and nothing
    else (the worker child, ``agent.child``, #771). Its peak-memory figures
    are then exact: the counters are reset unconditionally and no card is
    ever ``shared``. The server's executor keeps the default, False.
    """

    def __init__(self, *, exclusive_process: bool = False,
                 worker_run: bool = False,
                 run_cards: Mapping[str, str] | None = None) -> None:
        self.exclusive_process = exclusive_process
        #: This executes a run a worker was assigned (``agent.child``),
        #: for the ``device_fallback`` report (#839): the run's own device
        #: is not judged there, and the sentence names the worker.
        self.worker_run = worker_run
        #: In a worker run, each in-run CUDA name and the worker's card it
        #: is (``{"cuda:0": "cuda:1"}``; empty when it sees no card). None
        #: when the numbering is not the run's own (the server, or a worker
        #: run that kept the agent's visibility).
        self.run_cards = None if run_cards is None else dict(run_cards)

    async def run(self, request: ExecutionRequest) -> ExecutionOutcome:
        """Run the graph, classify the outcome and attach its peak memory.

        Never raises but cancel. The peak is read whatever the outcome: a
        run that failed with an out-of-memory error is the one whose number
        matters most. ``exact`` is False unless the process is exclusive:
        runs in the server process share one CUDA context (see
        :class:`PeakMemoryScope`).
        """
        scope = PeakMemoryScope(request.context.device,
                                exclusive=self.exclusive_process).open()
        try:
            outcome = await self._execute(request)
            return replace(outcome, peak_memory=scope.read(
                exact=self.exclusive_process))
        finally:
            scope.close()

    async def _warn_device_fallbacks(self, request: ExecutionRequest,
                                     expanded: list[dict[str, Any]]) -> None:
        """Send a ``run_warning`` (``device_fallback``) for every device the
        run asked for and will not get (#839).

        Called from ``execute_graph``'s ``on_expanded``, with the node list
        it already expanded, before any node runs: no second expansion, and
        a graph that does not prepare (or a bypassed node) warns about
        nothing. The fallback itself is unchanged (``resolve_device``); a
        run that meets none emits nothing. In a worker's run subprocess the
        run's own device is not judged (its cards are renumbered there and
        submit checked the set), and a CUDA node is described in the run's
        own card names (``run_cards``). Never raises: a failure here is
        logged and the run goes on.
        """
        from .device_fallback import WARNING_DEVICE_FALLBACK, device_fallbacks

        try:
            found = device_fallbacks(
                expanded, request.options.get("device"),
                request.context.device,
                where="this worker" if self.worker_run else "this server",
                check_run=not self.worker_run,
                run_cards=self.run_cards)
            for entry in found:
                # Emitted directly, like a worker's coded warnings, so the
                # payload carries ``code`` and ``params`` beside ``detail``.
                await request.sinks.emit(EVENT_WARNING, {
                    "kind": WARNING_DEVICE_FALLBACK,
                    "detail": entry["detail"], "node_id": entry["node_id"],
                    "code": entry["code"], "params": entry["params"]})
        except Exception:  # noqa: BLE001 -- a report, never the run's failure
            logger.warning("run %s: could not report device fallbacks",
                           request.run_id, exc_info=True)

    async def _execute(self, request: ExecutionRequest) -> ExecutionOutcome:
        """Run the graph and classify the outcome. Never raises but cancel."""
        graph = request.graph
        options = request.options
        session = request.session
        media_ports = declared_media_ports(graph["nodes"])
        on_progress = _progress_bridge(request, media_ports)
        on_inner_status = _inner_status_bridge(request, on_progress,
                                               media_ports)
        recorder = _provenance_recorder(request)

        async def on_expanded(expanded: list[dict[str, Any]]) -> None:
            # The nodes inside blocks and preset cards declare media too:
            # read once, from the node list the run executes (#559).
            media_ports.update(declared_media_ports(expanded))
            if recorder is not None:
                recorder.add(expanded)
            await self._warn_device_fallbacks(request, expanded)

        if recorder is not None:
            on_progress, on_inner_status = _observed(
                recorder, on_progress, on_inner_status)

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
                on_inner_status=on_inner_status,
                on_expanded=on_expanded,
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


def _provenance_recorder(request: ExecutionRequest) -> Any:
    """A ``ProvenanceRecorder`` for a run with the ``provenance`` option
    and a sink to report into (#344), else None."""
    from .node_provenance import MODES, ProvenanceRecorder

    mode = request.options.get("provenance")
    sink = request.sinks.record_provenance
    if mode not in MODES or sink is None:
        return None
    return ProvenanceRecorder(request.graph.get("nodes") or [], mode, sink)


def _observed(recorder: Any, on_progress: Callable[..., Any],
              on_inner_status: Callable[..., Any],
              ) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """The two status bridges, each followed by *recorder*: a node's
    status event goes out first, then its provenance is described."""

    async def progress(node_id: str, status: str,
                       result: dict[str, Any] | None,
                       inner_node_id: str | None = None) -> None:
        await on_progress(node_id, status, result,
                          inner_node_id=inner_node_id)
        await recorder.observe(node_id, status, result)

    async def inner(node_id: str, container_id: str, status: str,
                    result: dict[str, Any] | None) -> None:
        await on_inner_status(node_id, container_id, status, result)
        await recorder.observe(node_id, status, result)

    return progress, inner


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

    def __init__(self, device: str | None, *, exclusive: bool = False) -> None:
        self.kind, index = split_device(device or "")
        #: The process runs this run and nothing else (a worker child,
        #: #771): reset every card unconditionally, never shared.
        self.exclusive = exclusive
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
        if self.exclusive:
            # Nothing else runs in this process, so nothing to share with
            # or to protect: no registration, every card reset.
            self.fresh = torch is not None and not initialised
            if initialised:
                self._reset_cards(torch, busy=set())
            return self
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
        self._reset_cards(torch, busy={scope.card for scope in others})
        return self

    def _reset_cards(self, torch: Any, *, busy: set[int | None]) -> None:
        """Reset the peak counters of every card not in *busy*."""
        try:
            for card in range(int(torch.cuda.device_count())):
                if card not in busy:
                    torch.cuda.reset_peak_memory_stats(card)
                    self.reset_cards.add(card)
        except Exception:  # noqa: BLE001 - observability never fails a run
            logger.debug("could not reset the CUDA peak memory statistics",
                         exc_info=True)

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

        *exact* is the caller's to say: runs in the server process share
        one CUDA context and pass False; a run alone in its own process
        (the worker child, #771, opened with ``exclusive=True``) passes
        True, and an exclusive scope never marks a card ``shared``. A
        ``shared`` card is never exact. The MPS figure stays
        ``exact: false`` either way: it includes the driver's cache.
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
                shared = not self.exclusive and (
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

#: ``run_warning`` kinds (#772, #783): the run's worker went away (the run
#: waits for it, or ends), or came back.
WARNING_WORKER_DISCONNECTED = "worker_disconnected"
WARNING_WORKER_RECONNECTED = "worker_reconnected"

#: ``execution_facts`` key of an ``s3`` worker run's bucket, prefix and
#: input keys (#793): a restarted server re-issues its credentials.
WORKER_S3_KEY = "worker_s3"

#: A worker run's frames are acknowledged (``ack``, #783) once this many
#: are recorded, or :data:`ACK_INTERVAL_S` after the first unacknowledged
#: one, whichever comes first; and always on ``run_finished``.
ACK_EVERY = 64
ACK_INTERVAL_S = 1.0

#: An ``s3`` run's temporary credentials are renewed once this share of
#: their remaining lifetime has passed (#791), never sooner than
#: :data:`REFRESH_MIN_S`; a failed renewal is retried every
#: :data:`REFRESH_RETRY_S` until it succeeds.
REFRESH_AT = 2 / 3
REFRESH_MIN_S = 5.0
REFRESH_RETRY_S = 60.0


@dataclass
class _S3Run:
    """An ``s3`` run's storage on this side (#791): the bucket and prefix
    its rows record, and the latest credentials sent (re-sent after a
    reconnect)."""

    bucket: str
    prefix: str
    access: Any
    renewed: bool = False


def _with_budget_note(outcome: ExecutionOutcome, placement: Any,
                      source: str | None = None) -> ExecutionOutcome:
    """A failed run that ran out of memory under a budget (#788) says so:
    the cap is the run's own, and raising it is the fix the generic advice
    does not mention. The note follows where the budget came from
    (*source*, #794): a ``history`` budget was given by default from an
    earlier run's peak, an ``estimate`` one was a static estimate the user
    accepted; either way a larger ``target.memory_gib`` is the fix."""
    budget = getattr(placement, "memory_budget", None)
    error = outcome.error
    if outcome.status != STATUS_FAILED or not budget or not error \
            or "out of memory" not in error.lower():
        return outcome
    from .memory_admission import format_gib

    devices = ", ".join(getattr(placement, "budget_devices", ()) or ())
    if source == "history":
        return replace(outcome, error=(
            f"{error}\n\nThis run was capped to the memory budget it was "
            f"given by default, {format_gib(budget)} on {devices} (its "
            "graph's last measured peak plus 20%): declare a larger "
            "target.memory_gib, or make the run need less."))
    if source == "estimate":
        return replace(outcome, error=(
            f"{error}\n\nThis run was capped to {format_gib(budget)} on "
            f"{devices}, a static estimate of its memory (no earlier run of "
            "its graph had reported a peak): the estimate was too low; "
            "declare a larger target.memory_gib, or make the run need "
            "less."))
    return replace(outcome, error=(
        f"{error}\n\nThis run was capped to its memory budget of "
        f"{format_gib(budget)} on {devices} (target.memory_gib): raise the "
        "budget, or make the run need less."))


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
    ``cancel`` (again after a reconnect, in case the first one was lost).

    Frames are acknowledged (``ack``, cumulative) every :data:`ACK_EVERY`
    recorded frames or :data:`ACK_INTERVAL_S` after the first one not yet
    acknowledged (one ``call_later`` timer puts a marker in the inbox), and
    on ``run_finished`` (#783). A worker whose connection drops does not end the run: the
    registry puts ``WORKER_DISCONNECTED`` in the inbox, the run gets a
    ``worker_disconnected`` warning and its row goes ``disconnected``, and
    it keeps its slots. ``WORKER_RECONNECTED`` (the worker's ``run_alive``
    listed it) brings it back to ``running`` with a ``worker_reconnected``
    warning, and the agent's replay continues the stream (the registry
    drops what was already delivered). ``WORKER_GONE`` (no reconnect within
    the grace, or none allowed) ends the run after a ``run_warning``:
    ``interrupted``, or ``cancelled`` when a stop had been asked (a user's
    Stop stays the user's).

    A run adopted after a restart of this server (``request.resume_seq``,
    #793) is not assigned again: the registry already holds it, detached
    (``WorkerRegistry.adopt``). The run gets a ``worker_disconnected``
    warning saying the server restarted, its row reads ``disconnected``,
    and from there it is a run whose worker dropped: the worker's
    reconnect brings it back, the replay continues after ``resume_seq``,
    or the grace ends it. Before each ``ack`` the frames it covers are
    made durable (``sinks.record_worker_seq``), which is what lets a
    restarted server resume the stream after the persisted seq.

    Before ``assign``, a degraded worker (its node catalog differs from
    this server's, #781) must have every node type the graph uses, else
    the run fails with ``worker_handshake.catalog_refusal``'s message and
    nothing is sent.

    Every send looks the worker up in the registry AT SEND TIME: a
    reconnect replaces the ``Worker`` object, and a frame sent through the
    old one would go to a closed socket.
    """

    def __init__(self, registry: "WorkerRegistry",
                 content_index: Any = None,
                 on_started: Callable[[str], Any] | None = None) -> None:
        self._registry = registry
        #: Called with the run id on its ``run_started`` (#788): a run
        #: admitted by memory has its cap in place, so the ledger moves its
        #: budget from reserved to running.
        self._on_started = on_started
        #: ``core.transfer.ContentIndex``: the inputs' hashes for a run
        #: that uses the transfer backend (#785). None: no such run is
        #: assigned (it fails with a message instead).
        self._content_index = content_index
        #: The ``s3`` runs in flight (#791), by run id.
        self._s3_runs: dict[str, _S3Run] = {}

    async def _send(self, worker_id: str, frame: Any) -> bool:
        """Send *frame* to *worker_id*'s current connection. False if none."""
        worker = self._registry.get(worker_id)
        if worker is None or not worker.online:
            return False
        await worker.send(frame)
        return True

    async def run(self, request: ExecutionRequest) -> ExecutionOutcome:
        from . import worker_protocol as wp
        from .workers import (
            not_connected_message,
            s3_transfer_failed_message,
            storage_fallback_message,
            transfer_failed_message,
        )

        placement = request.placement
        if placement is None:
            return ExecutionOutcome(STATUS_FAILED,
                                    "a worker run needs a placement")
        worker_id = placement.worker_id
        if request.resume_seq is not None:
            return await self._resume(request, worker_id)
        worker = self._registry.get(worker_id)
        if worker is None or not worker.online:
            message = not_connected_message(worker_id)
            await _worker_gone_warning(request, message)
            return self._gone(request, message)
        # The submit-time check again, against the worker as it is now: it
        # may have reconnected with fewer node types since (#781).
        refusal = await self._catalog_refusal(request, worker)
        if refusal is not None:
            return ExecutionOutcome(STATUS_FAILED, refusal)
        if placement.memory_budget and wp.CAPABILITY_MEMORY_BUDGET not in (
                getattr(worker, "capabilities", None) or ()):
            # Admitted to a shared card by its budget, but the worker that is
            # connected now (an older agent that reconnected) cannot cap it:
            # never run it uncapped beside the card's other runs (#788).
            return ExecutionOutcome(STATUS_FAILED, (
                f"worker {worker_id} no longer enforces memory budgets (its "
                f"agent reconnected without the memory_budget capability), so "
                f"this run, admitted to a shared card by its budget, was not "
                f"started; update the agent, or set the card to Exclusive, "
                f"and queue the run again"))
        if request.record_outputs and wp.CAPABILITY_RECORD_OUTPUTS not in (
                getattr(worker, "capabilities", None) or ()):
            # Only an agent that keeps the captures can be asked for them
            # (#789); the submit-time check could not see this one's.
            return ExecutionOutcome(STATUS_FAILED, (
                f"worker {worker_id} cannot record outputs: its agent does "
                f"not support it; update the agent, or queue the run without "
                f"Record outputs or on this server"))
        if request.record_outputs and getattr(worker, "one_shot", False):
            from .run_service import ONE_SHOT_REFUSES_RECORD_OUTPUTS

            return ExecutionOutcome(STATUS_FAILED,
                                    ONE_SHOT_REFUSES_RECORD_OUTPUTS)
        devices = list(placement.devices)
        storage = assignment_storage(worker)
        backend = storage_backend(worker, request.needs)
        if backend is not None:
            try:
                if backend == "s3":
                    storage = await s3_storage(
                        storage.mount, request.run_id, request.needs,
                        self._content_index)
                else:
                    storage = await transfer_storage(
                        storage.mount, request.needs, self._content_index)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- TransferRefused, I/O
                message = (s3_transfer_failed_message(worker_id, exc)
                           if backend == "s3"
                           else transfer_failed_message(worker_id, exc))
                return ExecutionOutcome(STATUS_FAILED, message.text,
                                        error_reason=_reason_of(message))
            if backend == "transfer" and _s3_skipped(worker):
                from . import s3
                await _warning(request, s3.WARNING_STORAGE_FALLBACK,
                               storage_fallback_message(worker_id))
        input_keys: list[str] = []
        if storage.s3 is not None:
            self._s3_runs[request.run_id] = _S3Run(
                bucket=storage.s3.bucket, prefix=storage.s3.prefix,
                access=storage.s3)
            input_keys = [i.key for i in storage.inputs or [] if i.key]
            # What a restarted server needs to issue the run's worker new
            # credentials after adopting it (#793): before the assign, so a
            # crash right after it still finds them.
            await _record_facts(request, {WORKER_S3_KEY: {
                "bucket": storage.s3.bucket, "prefix": storage.s3.prefix,
                "input_keys": input_keys}})
        run = self._registry.open_run(worker, request.run_id)
        assign = wp.Assign(
            run_id=request.run_id, graph=request.graph,
            options=request.options, device=devices[0],
            devices=devices, storage=storage,
            memory_budgets=({device: placement.memory_budget
                             for device in placement.budget_devices}
                            if placement.memory_budget else {}))
        return await self._serve(request, worker_id, run, devices,
                                 assign=(worker, assign),
                                 s3_input_keys=input_keys)

    async def _resume(self, request: ExecutionRequest,
                      worker_id: str) -> ExecutionOutcome:
        """A run adopted after a restart of this server (#793).

        Its recorded outputs, if any, are still recording on its worker:
        retained here again from now on (#789), once its worker still has
        it.
        """
        from .workers import adopted_message, lost_after_restart_message

        run = self._registry.claim_adopted(request.run_id)
        if run is None:
            message = lost_after_restart_message(worker_id)
            await _worker_gone_warning(request, message)
            return self._gone(request, message)
        await _retain_outputs(request, worker_id)
        grace = getattr(self._registry, "resume_grace_s", 0)
        await _warning(request, WARNING_WORKER_DISCONNECTED,
                       adopted_message(worker_id, grace))
        await _record_connection(request, False)
        s3_reissue = None
        resume_s3 = request.resume_s3
        if isinstance(resume_s3, Mapping) and resume_s3.get("bucket"):
            s3_reissue = [k for k in resume_s3.get("input_keys") or []
                          if isinstance(k, str)]
            self._s3_runs[request.run_id] = _S3Run(
                bucket=str(resume_s3["bucket"]),
                prefix=str(resume_s3.get("prefix") or ""), access=None)
        return await self._serve(request, worker_id, run,
                                 list(request.placement.devices),
                                 resume_seq=request.resume_seq or 0,
                                 s3_reissue=s3_reissue)

    async def _serve(self, request: ExecutionRequest, worker_id: str,
                     run: Any, devices: list[str], *, assign: Any = None,
                     resume_seq: int = 0,
                     s3_input_keys: list[str] | None = None,
                     s3_reissue: list[str] | None = None) -> ExecutionOutcome:
        """Send *assign* (``(worker, frame)``; none for an adopted run),
        then turn the run's inbox into sink calls until it ends."""
        from . import worker_protocol as wp
        from .workers import (
            WORKER_DISCONNECTED,
            WORKER_GONE,
            WORKER_RECONNECTED,
            gone_message,
            unreachable_message,
        )

        watcher: asyncio.Task | None = None
        refresher: asyncio.Task | None = None
        acks: _Acks | None = None
        finished = False
        try:
            if assign is not None:
                target, frame = assign
                await target.send(frame)
                # The agent has it: a one-shot worker's claim stays (#790).
                assigned = getattr(self._registry, "one_shot_assigned", None)
                if assigned is not None:
                    assigned(worker_id, request.run_id)
            watcher = asyncio.create_task(
                self._forward_cancel(request, worker_id),
                name=f"cancel-watch:{request.run_id}")
            state = self._s3_runs.get(request.run_id)
            if s3_reissue is not None and state is not None:
                # An adopted s3 run (#793): its credentials were issued by
                # the previous process; fresh ones go out once the worker
                # is back, then are renewed as for any run.
                refresher = asyncio.create_task(
                    self._reissue_credentials(request, worker_id,
                                              s3_reissue),
                    name=f"s3-credentials:{request.run_id}")
            elif state is not None and state.access is not None \
                    and state.access.expires_at is not None:
                refresher = asyncio.create_task(
                    self._refresh_credentials(request, worker_id,
                                              s3_input_keys or []),
                    name=f"s3-credentials:{request.run_id}")
            acks = _Acks(self, request, worker_id, run.inbox,
                         recorded=resume_seq)
            while True:
                try:
                    item = run.inbox.get_nowait()
                except asyncio.QueueEmpty:
                    item = await run.inbox.get()
                if item is _ACK_DUE:
                    await acks.flush()
                    continue
                if isinstance(item, tuple) and item and item[0] is WORKER_GONE:
                    message = gone_message(worker_id, item[1])
                    await _worker_gone_warning(request, message)
                    return self._gone(request, message)
                if isinstance(item, tuple) and item \
                        and item[0] is WORKER_DISCONNECTED:
                    await self._disconnected(request, worker_id, item[1])
                    continue
                if isinstance(item, tuple) and item \
                        and item[0] is WORKER_RECONNECTED:
                    await self._reconnected(request, worker_id, acks)
                    continue
                outcome = await self._deliver(request, item, devices)
                acks.delivered(item.seq)
                if outcome is not None:
                    finished = True
                    await acks.flush()
                    return _with_budget_note(
                        outcome, request.placement,
                        request.memory_budget_source)
                if acks.recorded - acks.acked >= ACK_EVERY:
                    await acks.flush()
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
            message = unreachable_message(worker_id, exc)
            await _worker_gone_warning(request, message)
            return self._gone(request, message)
        finally:
            if watcher is not None:
                watcher.cancel()
            if refresher is not None:
                refresher.cancel()
            self._s3_runs.pop(request.run_id, None)
            if acks is not None:
                acks.cancel_timer()
            self._registry.close_run(worker_id, request.run_id,
                                     finished=finished)

    async def _disconnected(self, request: ExecutionRequest, worker_id: str,
                            reason: Any) -> None:
        """The worker's connection dropped; the run waits for it (#783)."""
        from .workers import disconnected_message

        grace = getattr(self._registry, "resume_grace_s", 0)
        await _warning(request, WARNING_WORKER_DISCONNECTED,
                       disconnected_message(worker_id, reason, grace))
        await _record_connection(request, False)

    async def _reconnected(self, request: ExecutionRequest, worker_id: str,
                           acks: "_Acks") -> None:
        """The worker is back and still has the run (#783)."""
        from . import worker_protocol as wp
        from .workers import reconnected_message

        await _warning(request, WARNING_WORKER_RECONNECTED,
                       reconnected_message(worker_id))
        await _record_connection(request, True)
        # Its replay starts after the last ack it got; tell it what this
        # side already has, and repeat a stop it may never have received.
        await acks.resend()
        state = self._s3_runs.get(request.run_id)
        if state is not None and state.renewed and state.access is not None:
            # New credentials may have gone to the dropped connection (#791).
            await self._send_quietly(worker_id, wp.Credentials(
                run_id=request.run_id, s3=state.access))
        if request.context.should_stop():
            await self._send(worker_id, wp.Cancel(run_id=request.run_id))

    @staticmethod
    async def _catalog_refusal(request: ExecutionRequest,
                               worker: Any) -> str | None:
        """Why *worker* cannot run *request*'s node types (#781), or None.

        The types were computed at submit (``request.node_types``); a run
        without them (none was computed) has them computed here, off the
        loop. A failure to compute is logged and refuses nothing: the
        worker reports an unknown type itself.
        """
        from .worker_handshake import catalog_refusal, graph_node_types

        if not getattr(getattr(worker, "catalog", None), "degraded", False):
            return None
        try:
            types = request.node_types
            if types is None:
                types = frozenset(await asyncio.to_thread(
                    graph_node_types, request.graph))
            return catalog_refusal(worker, types)
        except Exception:  # noqa: BLE001
            logger.warning("run %s: could not check its node types against "
                           "worker %s", request.run_id, worker.name,
                           exc_info=True)
            return None

    @staticmethod
    def _gone(request: ExecutionRequest, message: Any) -> ExecutionOutcome:
        """The outcome of a run whose worker is gone: a stop if one was
        asked (``_stopped_status`` keeps a user's Stop ``cancelled``),
        else ``interrupted`` with *message* (a ``workers.WorkerReason``)."""
        if request.context.should_stop():
            return ExecutionOutcome(STATUS_CANCELLED, None)
        return ExecutionOutcome(STATUS_INTERRUPTED, str(message),
                                error_reason=_reason_of(message))

    async def _forward_cancel(self, request: ExecutionRequest,
                              worker_id: str) -> None:
        """Watcher task: forward the run's stop as ``cancel``. Never raises.

        A worker that is away gets the cancel when it is back (#783). One
        that is removed meanwhile does not keep this waiting: removal
        (``WorkerRegistry.forget``) ends the run, which cancels this task.
        """
        from . import worker_protocol as wp

        try:
            while not request.context.should_stop():
                await asyncio.sleep(CANCEL_POLL_S)
            told = False
            while not await self._send(worker_id,
                                       wp.Cancel(run_id=request.run_id)):
                if not told:
                    told = True
                    logger.info("run %s: stop requested, but worker %s is "
                                "not connected; waiting for it",
                                request.run_id, worker_id)
                await asyncio.sleep(CANCEL_POLL_S)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- the run ends via WORKER_GONE
            logger.warning("run %s: could not forward the stop to worker %s",
                           request.run_id, worker_id, exc_info=True)

    async def _send_quietly(self, worker_id: str, frame: Any) -> bool:
        """:meth:`_send`, a failed send (the socket closed after the online
        check) being False rather than an exception."""
        try:
            return await self._send(worker_id, frame)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- a reconnect re-sends
            logger.debug("worker %s: %s not sent", worker_id,
                         getattr(frame, "type", frame), exc_info=True)
            return False

    async def _renewed_access(self, request: ExecutionRequest,
                              input_keys: list[str]) -> Any:
        """New credentials for the run, asked for every
        :data:`REFRESH_RETRY_S` until they come (#791)."""
        from . import s3

        while True:
            try:
                return await asyncio.to_thread(s3.issue_access,
                                               request.run_id, input_keys)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- STS refused, network
                logger.warning("run %s: could not renew its S3 credentials "
                               "(%s); trying again in %.0f s",
                               request.run_id, exc, REFRESH_RETRY_S)
                await asyncio.sleep(REFRESH_RETRY_S)

    async def _reissue_credentials(self, request: ExecutionRequest,
                                   worker_id: str,
                                   input_keys: list[str]) -> None:
        """Watcher task of an adopted ``s3`` run (#793): the credentials
        the worker holds came from the previous process and may expire
        before the run ends. New ones are issued now and sent once the
        worker is back (a reconnect re-sends them too), then renewed as
        :meth:`_refresh_credentials` does. Never raises but a cancel."""
        from . import worker_protocol as wp

        state = self._s3_runs[request.run_id]
        fresh = await self._renewed_access(request, input_keys)
        state.access, state.renewed = fresh, True
        while not await self._send_quietly(worker_id, wp.Credentials(
                run_id=request.run_id, s3=fresh)):
            await asyncio.sleep(CANCEL_POLL_S)
        if fresh.expires_at is not None:
            await self._refresh_credentials(request, worker_id, input_keys)

    async def _refresh_credentials(self, request: ExecutionRequest,
                                   worker_id: str,
                                   input_keys: list[str]) -> None:
        """Watcher task of an ``s3`` run with temporary credentials (#791):
        new ones are asked for once :data:`REFRESH_AT` of the current ones'
        remaining lifetime has passed (a refusal retried every
        :data:`REFRESH_RETRY_S`) and go out as ``credentials``, for as long
        as the run lasts. A send that fails (the worker is away, or its
        socket closed under the send) is tried again until it goes; a
        reconnect re-sends the latest too (:meth:`_reconnected`). Never
        raises but a cancel."""
        from . import worker_protocol as wp

        state = self._s3_runs[request.run_id]
        while state.access.expires_at is not None:
            left = state.access.expires_at - time.time()
            await asyncio.sleep(max(REFRESH_MIN_S, left * REFRESH_AT))
            fresh = await self._renewed_access(request, input_keys)
            state.access, state.renewed = fresh, True
            while not await self._send_quietly(worker_id, wp.Credentials(
                    run_id=request.run_id, s3=fresh)):
                await asyncio.sleep(CANCEL_POLL_S)

    async def _deliver(self, request: ExecutionRequest, frame: Any,
                       devices: list[str]) -> ExecutionOutcome | None:
        """Hand one frame to the sinks; the outcome on ``run_finished``."""
        from . import worker_protocol as wp

        sinks = request.sinks
        if isinstance(frame, wp.RunStarted):
            if self._on_started is not None:
                self._on_started(request.run_id)
            if request.placement is not None:
                # Started there: only now does a recorded run hold outputs
                # on its worker to retain (#789).
                await _retain_outputs(request, request.placement.worker_id)
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
            if frame.backend in ("transfer", "s3"):
                state = self._s3_runs.get(request.run_id) \
                    if frame.backend == "s3" else None
                signal: ArtifactSignal = TransferredArtifactSignal(
                    kind=frame.kind, path=frame.path, meta=frame.meta,
                    node_id=frame.node_id, backend=frame.backend,
                    bucket=state.bucket if state else None,
                    prefix=state.prefix if state else None)
            elif frame.mount_label:
                signal = MountedArtifactSignal(
                    kind=frame.kind, path=frame.path, meta=frame.meta,
                    node_id=frame.node_id, mount_label=frame.mount_label)
            else:
                signal = ArtifactSignal(
                    kind=frame.kind, path=frame.path, meta=frame.meta,
                    node_id=frame.node_id)
            await sinks.record_artifact(signal)
        elif isinstance(frame, wp.RunWarning):
            await sinks.warn(WarningSignal(kind=frame.kind,
                                           detail=frame.detail,
                                           node_id=frame.node_id))
        elif isinstance(frame, wp.RunFacts):
            await self._run_facts(request, frame)
        elif isinstance(frame, wp.RunFinished):
            return self._outcome(frame, devices,
                                 stop_asked=request.context.should_stop())
        return None

    @staticmethod
    async def _run_facts(request: ExecutionRequest, frame: Any) -> None:
        """A ``run_facts`` frame (#344): the subprocess's environment to
        ``record_environment``, a provenance record to
        ``record_provenance``. Observability: a failure is logged."""
        from .worker_handshake import verified_fingerprint

        sinks = request.sinks
        try:
            if (frame.fingerprint is not None or frame.numerics is not None) \
                    and sinks.record_environment is not None:
                document, _ = verified_fingerprint(
                    frame.fingerprint, worker=request.placement.worker_id
                    if request.placement is not None else "?")
                await sinks.record_environment(document, frame.numerics)
            if frame.provenance is not None \
                    and sinks.record_provenance is not None:
                await sinks.record_provenance(frame.provenance)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- observability
            logger.warning("run %s: could not record what its worker "
                           "reported about it", request.run_id,
                           exc_info=True)

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


def assignment_storage(worker: Any) -> Any:
    """``assign.storage`` for *worker* (#782).

    Always this server's mount label. For a worker this server lends its
    data directories to (``mounts.lends_mount``: same host, no directory of
    its own for the label) also those directories, as configured here. A
    same-host worker's files are on this machine, so it reports them with
    their absolute path (``relative_artifacts`` off): they become ``local``
    rows, never ``mount`` rows.
    """
    from . import worker_protocol as wp
    from .mounts import control_plane_label, lends_mount, lent_dirs

    label = control_plane_label()
    return wp.Storage(
        mount=label,
        dirs=lent_dirs() if lends_mount(worker, label) else None,
        relative_artifacts=not getattr(worker, "same_host", False))


def storage_backend(worker: Any, needs: Any) -> str | None:
    """The backend that ships a run's files to *worker* (#785, #791):
    ``s3`` or ``transfer`` when the run needs the mount, *worker* does not
    hold it, the run fits (``needs["transfer"]``) and the worker speaks one
    (``scheduling.no_mount_backend``, S3 first). The scheduler placed it
    there for that reason (``scheduling.reaches``). None: the run is
    assigned the mount as before."""
    from .mounts import holds_mount
    from .scheduling import no_mount_backend
    from .transfer import transferable

    mount = needs.get("mount") if isinstance(needs, Mapping) else None
    if not mount or holds_mount(worker, mount) or not transferable(needs):
        return None
    return no_mount_backend(worker)


def _s3_skipped(worker: Any) -> bool:
    """S3 is on here, but *worker*'s agent did not list it (#791)."""
    from . import s3
    from .worker_protocol import CAPABILITY_S3

    return s3.enabled() and CAPABILITY_S3 not in (
        getattr(worker, "capabilities", None) or ())


async def s3_storage(label: str, run_id: str, needs: Mapping[str, Any],
                     content_index: Any) -> Any:
    """``assign.storage`` for an ``s3`` run (#791): the inputs named by
    content hash (as for ``transfer``) and by the object each was put in
    (``s3.ensure_inputs``, which uploads one the bucket lacks), and the
    bucket with the credentials the worker uses (``s3.issue_access``)."""
    from . import s3
    from . import worker_protocol as wp
    from .transfer import TransferRefused

    inputs = list(needs.get("inputs") or [])
    if inputs and content_index is None:
        raise TransferRefused("this server has no content index")
    manifest = await content_index.manifest(inputs) if inputs else []
    entries = await asyncio.to_thread(s3.ensure_inputs, manifest)
    access = await asyncio.to_thread(
        s3.issue_access, run_id, [e["key"] for e in entries])
    return wp.Storage(
        backend="s3", mount=label, relative_artifacts=False,
        inputs=[wp.TransferInput(**entry) for entry in entries], s3=access)


async def transfer_storage(label: str, needs: Mapping[str, Any],
                           content_index: Any) -> Any:
    """``assign.storage`` for a transfer run (#785): the inputs named by
    content hash (``ContentIndex.manifest``, which refuses a file gone or
    a total over the cap) and the largest upload this server takes."""
    from . import worker_protocol as wp
    from ..config import settings
    from .transfer import TransferRefused

    inputs = list(needs.get("inputs") or [])
    if inputs and content_index is None:
        raise TransferRefused("this server has no content index")
    manifest = await content_index.manifest(inputs) if inputs else []
    return wp.Storage(
        backend="transfer", mount=label, relative_artifacts=False,
        inputs=[wp.TransferInput(**entry) for entry in manifest],
        max_upload_bytes=int(settings.MAX_UPLOAD_SIZE))


async def _retain_outputs(request: ExecutionRequest, worker_id: str) -> None:
    """A recorded worker run's captures are on *worker_id* (#789): the
    output store retains the run by id, so the outputs routes ask that
    worker and the store's retention evicts the run as its own. Called
    once the run has started there (``run_started``, or an adopted run
    its worker still has), never for one that failed before."""
    if request.record_outputs and request.output_store is not None:
        await request.output_store.add_remote(request.run_id, worker_id)


def _reason_of(message: Any) -> dict[str, Any] | None:
    """``{code, params}`` of a coded message, or None for plain text
    (``workers.reason_dict``)."""
    from .workers import reason_dict

    return reason_dict(message)


async def _worker_gone_warning(request: ExecutionRequest,
                               message: Any) -> None:
    await _warning(request, WARNING_WORKER_DISCONNECTED, message)


async def _warning(request: ExecutionRequest, kind: str,
                   message: Any) -> None:
    """A ``run_warning`` about the run's worker. *message* is a
    ``workers.WorkerReason``: ``detail`` is its English text, and ``code``
    and ``params`` let the Runs panel word it (#841). Only worker runs
    send these, so a run with no target never carries the two keys."""
    payload: dict[str, Any] = {"kind": kind, "detail": str(message),
                               "node_id": None}
    reason = _reason_of(message)
    if reason is not None:
        payload.update(reason)
    try:
        await request.sinks.emit(EVENT_WARNING, payload)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 -- observability
        logger.warning("run %s: could not record %s", request.run_id,
                       message)


async def _record_facts(request: ExecutionRequest,
                        patch: dict[str, Any]) -> None:
    record = request.sinks.record_execution_facts
    if record is None:
        return
    try:
        await record(patch)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 -- the run goes on; adoption degrades
        logger.warning("run %s: could not record %s", request.run_id,
                       ", ".join(patch), exc_info=True)


async def _record_connection(request: ExecutionRequest,
                             connected: bool) -> None:
    record = request.sinks.record_worker_connection
    if record is None:
        return
    try:
        await record(connected)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 -- observability
        logger.warning("run %s: could not record its worker connection",
                       request.run_id, exc_info=True)


#: Put in a run's inbox by the ack timer: acknowledge what is recorded.
_ACK_DUE = object()


class _Acks:
    """When to acknowledge a worker run's frames, and doing it (#783).

    Before an ack goes out, ``sinks.record_worker_seq`` makes the frames it
    covers durable and persists the seq (#793): the agent forgets what is
    acknowledged, so a restarted server must have all of it, and knows
    where to resume (*recorded*, for an adopted run). No ack goes out when
    that fails. A frame recorded after the last persisted seq is delivered
    again after a restart.
    """

    def __init__(self, executor: WorkerExecutor, request: ExecutionRequest,
                 worker_id: str, inbox: asyncio.Queue, *,
                 recorded: int = 0) -> None:
        self._executor = executor
        self._request = request
        self._worker_id = worker_id
        self._inbox = inbox
        #: Highest seq handed to the sinks, and highest acknowledged.
        self.recorded = recorded
        self.acked = recorded
        self._timer: asyncio.TimerHandle | None = None

    def delivered(self, seq: int) -> None:
        self.recorded = max(self.recorded, seq)
        if self._timer is None and self.recorded > self.acked:
            self._timer = asyncio.get_running_loop().call_later(
                ACK_INTERVAL_S, self._inbox.put_nowait, _ACK_DUE)

    def cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    async def flush(self) -> None:
        """Acknowledge everything recorded."""
        self.cancel_timer()
        if self.recorded <= self.acked:
            return
        seq = self.recorded
        if not await self._persist(seq):
            # Not acknowledged: the agent keeps the frames, so a restart
            # that resumes after the stored seq still finds them. The next
            # frame (or run_finished) tries again.
            return
        self.acked = seq
        await self._send(seq)

    async def _persist(self, seq: int) -> bool:
        """``sinks.record_worker_seq(seq)``. False when it failed (logged):
        the frames up to *seq* must then not be acknowledged."""
        record = self._request.sinks.record_worker_seq
        if record is None:
            return True
        try:
            await record(seq)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- retried at the next ack
            logger.warning("run %s: could not persist seq %d; not "
                           "acknowledging it yet", self._request.run_id, seq,
                           exc_info=True)
            return False
        return True

    async def resend(self) -> None:
        """Repeat the last ack (its first may not have reached the agent)."""
        if self.recorded > self.acked:
            await self.flush()
        elif self.acked:
            await self._send(self.acked)

    async def _send(self, seq: int) -> None:
        from . import worker_protocol as wp

        try:
            await self._executor._send(self._worker_id, wp.Ack(
                run_id=self._request.run_id, seq=seq))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- a lost ack means a replay
            logger.debug("run %s: ack %d not sent", self._request.run_id,
                         seq, exc_info=True)


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
