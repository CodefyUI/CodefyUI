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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol

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
    STATUS_SUCCEEDED,
    MetricPoint,
)
from .seeding import seed_rngs

if TYPE_CHECKING:  # pragma: no cover - typing only; run_service imports us
    from .run_service import InteractiveSession

#: The run service's logger, by name. These lines were ``run_service``'s
#: before #766 moved them here, and a log filter keyed on that name must keep
#: matching them.
logger = logging.getLogger("app.core.run_service")

#: ``graph_engine.execute_graph``'s default error policy. Mirrors
#: ``run_service.DEFAULT_ERROR_MODE``, which ``normalize_options`` validates
#: against; not imported from there because ``run_service`` imports this
#: module.
DEFAULT_ERROR_MODE = "fail_fast"


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


@dataclass(frozen=True)
class ExecutionOutcome:
    """How an execution ended: a terminal run status and, when failed, why.

    ``status`` is ``succeeded``, ``failed`` or ``cancelled``. ``cancelled``
    means "the engine stopped cooperatively"; whether that was a user or the
    server going away is the service's to say (``_stopped_status``).
    """

    status: str
    error: str | None = None


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
        """Run the graph and classify the outcome. Never raises but cancel."""
        graph = request.graph
        options = request.options
        session = request.session
        media_ports = declared_media_ports(graph["nodes"])
        on_progress = _progress_bridge(request, media_ports)
        try:
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
