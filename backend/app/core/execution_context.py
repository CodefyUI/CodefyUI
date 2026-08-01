"""Execution context for tracking and cancelling graph runs.

Three things live here, and they are all about the boundary between the
event loop and the worker threads nodes execute in:

* :class:`ExecutionContext` -- the per-run bag of flags every node sees.
* :func:`ExecutionContext.should_stop` -- the cooperative stop flag, readable
  from an executor thread with no loop interaction at all (#122).
* :class:`EventOutbox` -- a bounded, drop-oldest, thread-safe hand-off so a
  node can report progress, metrics and artifacts from its worker thread
  WITHOUT ever blocking on the loop (#122).

Why the outbox exists
---------------------
Before #122 the only way out of a node was ``graph_engine``'s progress
bridge, which did ``asyncio.run_coroutine_threadsafe(...).result(timeout=10)``
-- it parked the training thread until the event loop had persisted the
event, and swallowed whatever went wrong. On the per-EPOCH cadence that was
merely wasteful; at the per-BATCH cadence this issue introduces it would be
a training loop whose speed is set by sqlite. The outbox inverts it: the
worker appends to a deque under a plain ``threading.Lock`` and pokes the
loop with ``call_soon_threadsafe``; a loop-side pump drains and dispatches.

The queue is BOUNDED and drops the OLDEST item when full, for the same
reason ``RunSubscription`` does: a stalled consumer must never be able to
slow down, deadlock or memory-starve the run feeding it. Dropped items are
counted and reported once per drain, so the loss is visible rather than
silent.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from .node_state_store import NodeStateStore

logger = logging.getLogger(__name__)


# ── signals ───────────────────────────────────────────────────────────────
#
# What a worker thread can hand to the loop. Frozen value objects, never
# live handles: an item may sit in the queue across an arbitrary delay, so
# it has to carry everything the loop-side consumer needs.


@dataclass(frozen=True)
class ProgressSignal:
    """One ``progress`` node_status frame, queued instead of blocking.

    ``node_id`` is stamped by ``graph_engine``'s per-node bridge, which is
    the only producer -- nodes keep calling ``progress_callback`` and never
    see this type.
    """

    node_id: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class MetricSignal:
    """One point of a named series, from :meth:`ExecutionContext.log_metric`."""

    name: str
    value: float
    step: int
    node_id: str | None = None


@dataclass(frozen=True)
class ArtifactSignal:
    """A file the run produced, from :meth:`ExecutionContext.log_artifact`.

    Registering an artifact is a database write, which a worker thread
    cannot do; the node writes the FILE synchronously and hands the ROW over
    here.
    """

    kind: str
    path: str
    meta: dict[str, Any] | None = None
    node_id: str | None = None


@dataclass(frozen=True)
class DroppedSignal:
    """*count* items the outbox discarded to stay bounded.

    Synthesized by the loop-side drain, never queued (queueing a
    drop-notice on a full queue would drop something else to say so).
    """

    count: int


#: Result key a long-running node sets to say "I stopped early because
#: ``should_stop()`` flipped, and these outputs are partial". ``graph_engine``
#: turns it into an ``interrupted`` node status instead of ``completed``.
#: Dunder-prefixed like ``__steps__`` / ``__log__``, so it is filtered out of
#: recorded outputs and port summaries for free. The value is a dict of
#: whatever the node knows about where it stopped (epoch, batch, checkpoint
#: path); only its truthiness is load-bearing.
INTERRUPTED_KEY = "__interrupted__"


#: Outbox depth. Sized for the burst a per-batch producer makes while the
#: loop is busy persisting an event -- hundreds of items -- not for holding
#: a whole run's metrics, which belong in ``exec_run_metrics``.
DEFAULT_OUTBOX_CAPACITY = 1024


class EventOutbox:
    """Bounded thread-to-loop hand-off. ``put`` NEVER blocks and never raises.

    Producer side (any thread): :meth:`put`. Consumer side (the loop that
    called :meth:`bind`): ``await`` :meth:`wait`, then :meth:`drain`.

    The wake-up has no lost-notification window because clearing the waker
    and taking the items happen under the SAME lock as appending: a ``put``
    either lands before the drain takes the items, or after the drain
    released the lock -- in which case it re-arms the waker. The only
    possible anomaly is a spurious wake with nothing to deliver, which costs
    one empty drain.
    """

    __slots__ = ("_items", "_capacity", "_lock", "_dropped", "_loop", "_waker")

    def __init__(self, capacity: int = DEFAULT_OUTBOX_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._items: deque[Any] = deque()
        self._capacity = capacity
        self._lock = threading.Lock()
        self._dropped = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._waker: asyncio.Event | None = None

    @property
    def capacity(self) -> int:
        return self._capacity

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """Attach the loop that will drain. Call from that loop's thread.

        Re-arms the waker when items are already buffered, so anything a
        node queued before the run started (or between two runs sharing a
        context) is delivered rather than stranded until the next ``put``.
        """
        with self._lock:
            self._loop = loop
            self._waker = asyncio.Event()
            if self._items or self._dropped:
                self._waker.set()

    def unbind(self) -> None:
        """Detach the loop. Buffered items survive for a final drain."""
        with self._lock:
            self._loop = None
            self._waker = None

    def put(self, signal: Any) -> None:
        """Queue one signal from ANY thread. Drops the oldest when full.

        Deliberately total: a metric API that can raise is a metric API that
        eventually kills a six-hour training run.
        """
        with self._lock:
            if len(self._items) >= self._capacity:
                self._items.popleft()
                self._dropped += 1
            self._items.append(signal)
            loop, waker = self._loop, self._waker
            if loop is not None and waker is not None:
                try:
                    loop.call_soon_threadsafe(waker.set)
                except RuntimeError:
                    # The loop closed under us. The items stay buffered; a
                    # final drain still finds them.
                    pass

    def drain(self) -> tuple[list[Any], int]:
        """Take everything queued plus the drop count since the last drain."""
        with self._lock:
            if self._waker is not None:
                self._waker.clear()
            items = list(self._items)
            self._items.clear()
            dropped, self._dropped = self._dropped, 0
        return items, dropped

    async def wait(self) -> None:
        """Block the caller until something is queued.

        An unbound outbox has nothing to wake it, so this yields on a short
        sleep rather than parking forever -- a pump started before
        :meth:`bind` must not spin, and must not hang either.
        """
        waker = self._waker
        if waker is None:
            await asyncio.sleep(0.05)
            return
        await waker.wait()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


@dataclass
class ExecutionContext:
    """Shared context for a single graph execution run.

    Carries cancellation, parallelism settings, plus per-run feature flags
    (verbose step trace, weight persistence, backward gradient capture)
    that nodes consult to opt into educational behaviour.
    """

    execution_id: str = field(default_factory=lambda: str(uuid4()))
    max_workers: int = 4
    #: A ``threading`` Event, not an ``asyncio`` one: it is SET on the loop
    #: (a Stop click, a shutdown) and READ inside a training loop running in
    #: an executor thread. ``asyncio.Event.set`` is not thread-safe and its
    #: waiters are useless here -- nothing awaits cancellation, everything
    #: polls it.
    _stop_event: threading.Event = field(default_factory=threading.Event)

    #: Worker-thread -> loop hand-off for progress, metrics and artifacts.
    #: Shared by every node of the run; ``graph_engine`` binds and drains it.
    outbox: EventOutbox = field(default_factory=EventOutbox)

    # Global compute device ("cpu" / "cuda" / "mps") for the whole run. Already
    # resolved (availability-checked) at the execution entry point. Tensor-
    # source nodes create on it and StatefulModuleMixin moves layer modules to
    # it, so the entire graph — not just the device-aware sink nodes — runs on
    # the chosen accelerator.
    device: str = "cpu"

    # A1: verbose step-trace mode. Instrumented nodes emit __steps__ when True.
    verbose: bool = False

    # A2: per-node weight persistence.
    graph_id: str = ""
    weights_persistent: bool = True
    node_state_store: "NodeStateStore | None" = None

    # A3: backward-pass gradient capture.
    backward_mode: bool = False
    auto_backward: bool = False

    # Mutated per-node by graph_engine before each execute() call so that
    # StatefulModuleMixin.get_or_build_module knows which node it is in.
    current_node_id: str = ""

    # Populated during a run when backward_mode is True. Maps
    # (node_id, port) -> tensor reference whose .grad we want to capture
    # after the backward pass. graph_engine writes here; capture_grads reads.
    grad_targets: dict[tuple[str, str], Any] = field(default_factory=dict)

    def cancel(self) -> None:
        """Signal cancellation. Safe from any thread."""
        self._stop_event.set()

    @property
    def cancelled(self) -> bool:
        return self._stop_event.is_set()

    def should_stop(self) -> bool:
        """True once someone asked this run to stop. Call it every batch.

        The whole point is that it is cheap and thread-safe: a plain
        ``threading.Event`` read, no loop, no lock contention with the
        engine, nothing to await. A long-running node calls it inside its
        innermost loop and returns partial results when it flips; see
        ``TrainingLoopNode`` for the interrupt-checkpoint pattern that goes
        with it.

        Identical to the ``cancelled`` property; both exist because
        ``cancelled`` reads as engine bookkeeping while ``should_stop()`` is
        the verb a node author is looking for.
        """
        return self._stop_event.is_set()

    # ── loop-side reporting from a worker thread ─────────────────────────

    def log_metric(
        self,
        name: str,
        value: Any,
        step: int,
        node_id: str | None = None,
    ) -> None:
        """Record one point of a named series. Thread-safe and NON-BLOCKING.

        The point is queued for the run's loop-side consumer, which writes
        it to ``exec_run_metrics`` through the BATCHED
        ``RunStore.log_metrics`` (never a transaction per point) and emits a
        ``metric`` event. When the queue is full the OLDEST point is dropped
        and the loss is reported once per drain -- a training loop is never
        slowed down or failed by its own instrumentation.

        *step* is the series' x-axis: the absolute epoch for a per-epoch
        series, the global batch index for a per-batch one. *node_id*
        defaults to the node currently executing. Under parallel execution
        that field is shared, so a node running concurrently with others
        should pass its own id explicitly if the label matters.

        A value that is not a finite number is still recorded (as NULL) by
        the store, so a chart breaks exactly where the training diverged;
        a value that is not a number at ALL is dropped with a log line
        rather than raised.
        """
        try:
            numeric = float(value)
            index = int(step)
        except (TypeError, ValueError):
            logger.debug(
                "log_metric(%r) ignored: value=%r step=%r is not numeric",
                name, value, step,
            )
            return
        self.outbox.put(MetricSignal(
            name=str(name), value=numeric, step=index,
            node_id=node_id or self.current_node_id or None,
        ))

    def log_artifact(
        self,
        kind: str,
        path: str,
        meta: dict[str, Any] | None = None,
        node_id: str | None = None,
    ) -> None:
        """Register a file this run produced. Thread-safe and NON-BLOCKING.

        The node writes the file itself (synchronously, in its own thread);
        this only hands the ROW to the loop, which inserts it into
        ``exec_run_artifacts``. *kind* is an open vocabulary --
        ``checkpoint``, ``export``, ``image`` are the ones core uses.
        """
        self.outbox.put(ArtifactSignal(
            kind=str(kind), path=str(path), meta=meta,
            node_id=node_id or self.current_node_id or None,
        ))


class CancellationError(Exception):
    """Raised when a graph execution is cancelled."""
