"""``context.log_metric`` / ``log_artifact`` and the outbox behind them (#122).

Three properties, in the order they matter:

1. **Thread safety.** The producer is a node running in an executor thread;
   the consumer is the event loop. A metric logged from a worker thread has
   to arrive, exactly once, in order, without the worker ever waiting for
   the loop.
2. **Bounded, drop-oldest.** The queue must shed load rather than grow
   without limit or apply backpressure to a training loop. When it does, the
   loss is reported instead of being silent.
3. **Never raises.** A metric API that can throw is a metric API that
   eventually kills a six-hour run, so bad input is dropped with a log line.

The end-to-end half (points reaching ``exec_run_metrics``, ``metric``
events, artifacts) lives in ``test_run_service.py``; this file is about the
mechanism.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from app.core.execution_context import (
    DEFAULT_OUTBOX_CAPACITY,
    ArtifactSignal,
    EventOutbox,
    ExecutionContext,
    MetricSignal,
)
from app.core.graph_engine import execute_graph
from app.core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from app.core.node_registry import registry


# ── test nodes ────────────────────────────────────────────────────────────


class _MetricLoggerNode(BaseNode):
    """Logs ``count`` points from its executor thread, like a training loop."""

    NODE_NAME = "_MetricLogger"
    CATEGORY = "Test"
    DESCRIPTION = "Logs metrics from the worker thread"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="count", param_type=ParamType.INT, default=5),
            ParamDefinition(name="artifact", param_type=ParamType.BOOL,
                            default=False),
        ]

    def execute(self, inputs, params, progress_callback=None, *, context=None):
        assert context is not None
        thread = threading.current_thread()
        for step in range(1, int(params.get("count", 5)) + 1):
            context.log_metric("train_loss", 1.0 / step, step)
        if params.get("artifact"):
            context.log_artifact("checkpoint", "/tmp/never-written.pt",
                                 {"reason": "test"})
        return {"value": thread.name}


_TEST_NODES = {"_MetricLogger": _MetricLoggerNode}


@pytest.fixture(autouse=True)
def _register_test_nodes():
    registry._nodes.update(_TEST_NODES)
    yield
    for name in _TEST_NODES:
        registry._nodes.pop(name, None)


def _graph(**params):
    return (
        [
            {"id": "start", "type": "Start", "data": {"params": {}}},
            {"id": "logger", "type": "_MetricLogger", "data": {"params": params}},
        ],
        [{"id": "et", "source": "start", "target": "logger",
          "sourceHandle": "trigger", "type": "trigger"}],
    )


# ── the outbox itself ─────────────────────────────────────────────────────


def test_put_and_drain_round_trip():
    outbox = EventOutbox(capacity=8)
    outbox.put(MetricSignal("loss", 1.0, 1))
    outbox.put(MetricSignal("loss", 0.5, 2))

    items, dropped = outbox.drain()

    assert [s.value for s in items] == [1.0, 0.5]
    assert dropped == 0
    assert outbox.drain() == ([], 0), "a drained outbox is empty"


def test_drop_oldest_keeps_the_newest_and_counts_the_loss():
    """Over capacity, the OLDEST goes — a chart's tail matters more."""
    outbox = EventOutbox(capacity=3)
    for step in range(1, 7):
        outbox.put(MetricSignal("loss", float(step), step))

    items, dropped = outbox.drain()

    assert [s.step for s in items] == [4, 5, 6]
    assert dropped == 3
    # The count resets, so a consumer is told about each loss exactly once.
    assert outbox.drain() == ([], 0)


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        EventOutbox(capacity=0)


def test_default_capacity_is_bounded():
    assert 1 <= DEFAULT_OUTBOX_CAPACITY < 100_000
    assert EventOutbox().capacity == DEFAULT_OUTBOX_CAPACITY


def test_put_never_blocks_or_raises_without_a_loop():
    """An unbound outbox (a direct unit-test call) still accepts points."""
    context = ExecutionContext()
    for step in range(1, 4):
        context.log_metric("loss", step, step)
    items, dropped = context.outbox.drain()
    assert [s.name for s in items] == ["loss"] * 3
    assert dropped == 0


@pytest.mark.asyncio
async def test_put_from_another_thread_wakes_the_loop():
    """The wake-up is the whole point: no polling interval anywhere."""
    outbox = EventOutbox(capacity=4)
    outbox.bind(asyncio.get_running_loop())

    waiter = asyncio.ensure_future(outbox.wait())
    await asyncio.sleep(0)          # let the waiter park
    assert not waiter.done()

    await asyncio.to_thread(outbox.put, MetricSignal("loss", 1.0, 1))
    await asyncio.wait_for(waiter, timeout=2.0)

    items, _dropped = outbox.drain()
    assert [s.name for s in items] == ["loss"]


@pytest.mark.asyncio
async def test_a_redundant_wake_up_is_skipped_without_losing_anything():
    """A put onto an already-armed outbox must not schedule a second poke.

    ``call_soon_threadsafe`` is real loop work — a lock plus a self-pipe
    write — and a per-batch producer would pay for it on every batch while
    the consumer is busy. Skipping it is only safe because the pending wake
    drains EVERYTHING queued since, which is the second half of this test.
    """
    outbox = EventOutbox(capacity=16)
    loop = asyncio.get_running_loop()
    outbox.bind(loop)

    pokes = 0
    real = loop.call_soon_threadsafe

    def counting(callback, *args, **kwargs):
        nonlocal pokes
        pokes += 1
        return real(callback, *args, **kwargs)

    loop.call_soon_threadsafe = counting     # type: ignore[method-assign]
    try:
        outbox.put(MetricSignal("loss", 1.0, 1))
        assert pokes == 1, "the first put has to arm the waker"
        await asyncio.sleep(0)               # let the scheduled set() run
        for step in range(2, 6):
            outbox.put(MetricSignal("loss", 1.0, step))
        assert pokes == 1, "the waker was already set; those were redundant"

        items, dropped = outbox.drain()
        assert [s.step for s in items] == [1, 2, 3, 4, 5]
        assert dropped == 0

        # The drain cleared the waker, so the next put re-arms it.
        outbox.put(MetricSignal("loss", 1.0, 6))
        assert pokes == 2
    finally:
        del loop.call_soon_threadsafe        # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_bind_rearms_for_items_queued_before_the_run():
    """Nothing queued before ``bind`` may be stranded until the next put."""
    outbox = EventOutbox(capacity=4)
    outbox.put(MetricSignal("loss", 1.0, 1))
    outbox.bind(asyncio.get_running_loop())

    await asyncio.wait_for(outbox.wait(), timeout=2.0)
    assert len(outbox.drain()[0]) == 1


# ── ExecutionContext.log_metric ───────────────────────────────────────────


def test_log_metric_defaults_the_node_id_to_the_running_node():
    context = ExecutionContext()
    context.current_node_id = "train-1"
    context.log_metric("train_loss", 0.25, 3)
    context.log_metric("train_loss", 0.25, 3, node_id="explicit")

    items, _ = context.outbox.drain()
    assert [s.node_id for s in items] == ["train-1", "explicit"]


def test_log_metric_coerces_and_drops_non_numbers():
    """Bad input is a dropped point and a log line, never an exception."""
    context = ExecutionContext()
    context.log_metric("loss", "0.5", "2")           # coercible strings
    context.log_metric("loss", object(), 1)          # not a number
    context.log_metric("loss", 1.0, "not-a-step")    # not a step

    items, _ = context.outbox.drain()
    assert [(s.value, s.step) for s in items] == [(0.5, 2)]


def test_log_metric_keeps_nan_for_the_store_to_null_out():
    """A diverged loss is a real observation; the store decides how to file it."""
    import math

    context = ExecutionContext()
    context.log_metric("loss", float("nan"), 1)
    items, _ = context.outbox.drain()
    assert len(items) == 1 and math.isnan(items[0].value)


def test_log_artifact_queues_a_row():
    context = ExecutionContext()
    context.current_node_id = "train-1"
    context.log_artifact("checkpoint", "c.pt", {"epoch": 2})

    items, _ = context.outbox.drain()
    assert items == [ArtifactSignal(kind="checkpoint", path="c.pt",
                                    meta={"epoch": 2}, node_id="train-1")]


# ── thread safety, end to end through the engine ──────────────────────────


@pytest.mark.asyncio
async def test_metrics_logged_from_an_executor_thread_reach_the_loop():
    """The acceptance shape: a node logs from its thread, the loop receives.

    Also pins that it really is another thread — a bug that moved node
    execution onto the loop would make this test pass for the wrong reason.
    """
    signals: list[object] = []
    seen_loop_thread = threading.current_thread().name
    context = ExecutionContext()
    nodes, edges = _graph(count=4, artifact=True)

    assert context.can_record_artifacts() is False, "nothing has run yet"
    outputs = await execute_graph(nodes, edges, context=context,
                                  on_signal=signals.append)

    assert outputs["logger"]["value"] != seen_loop_thread, (
        "the node did not run in a worker thread; this test proves nothing"
    )
    assert context.can_record_artifacts() is True, (
        "the engine must stamp the capability from its on_signal"
    )
    metrics = [s for s in signals if isinstance(s, MetricSignal)]
    assert [s.step for s in metrics] == [1, 2, 3, 4], "order must be preserved"
    assert all(s.node_id == "logger" for s in metrics)
    artifacts = [s for s in signals if isinstance(s, ArtifactSignal)]
    assert [(a.kind, a.path) for a in artifacts] == [
        ("checkpoint", "/tmp/never-written.pt")]


@pytest.mark.asyncio
async def test_logging_never_blocks_the_worker_thread():
    """A stalled consumer must not slow the producer down.

    ``on_signal`` sleeps 50 ms per signal, so a BLOCKING hand-off would make
    100 points cost 5 s inside the node. Measured in the node itself, which
    is the only place the claim means anything.
    """
    async def slow_consumer(_signal):
        await asyncio.sleep(0.05)

    elapsed: dict[str, float] = {}

    class _TimedLogger(_MetricLoggerNode):
        NODE_NAME = "_TimedLogger"

        def execute(self, inputs, params, progress_callback=None, *, context=None):
            started = time.monotonic()
            for step in range(1, 101):
                context.log_metric("loss", 1.0 / step, step)
            elapsed["seconds"] = time.monotonic() - started
            return {"value": "done"}

    registry._nodes["_TimedLogger"] = _TimedLogger
    try:
        nodes, edges = _graph()
        nodes[1]["type"] = "_TimedLogger"
        await execute_graph(nodes, edges, context=ExecutionContext(),
                            on_signal=slow_consumer)
    finally:
        registry._nodes.pop("_TimedLogger", None)

    assert elapsed["seconds"] < 1.0, (
        f"100 log_metric calls took {elapsed['seconds']:.2f}s — the hand-off "
        "is blocking on the consumer"
    )


@pytest.mark.asyncio
async def test_overflow_reports_a_dropped_count_and_keeps_the_run_healthy():
    """Drop-oldest under a tiny queue, surfaced as a DroppedSignal.

    The overflow is FORCED, not raced for (#171). Every drain in the engine
    goes through one delivery lock, so a consumer that parks inside its first
    dispatch holds that lock and nothing can empty the queue while the node
    floods it. The previous version logged 40 points and trusted the pump to
    fall behind; on an idle CI runner it kept up and dropped nothing, which
    is a flaky test rather than a passing contract.
    """
    from app.core.execution_context import DroppedSignal

    capacity, total = 4, 40
    loop = asyncio.get_running_loop()
    consumer_parked = threading.Event()   # loop → node: no drain can happen
    burst_queued = asyncio.Event()        # node → loop: the flood is queued
    observed: dict[str, bool] = {}

    class _FloodingLogger(_MetricLoggerNode):
        """Waits until the consumer is provably stuck, THEN overruns it."""

        NODE_NAME = "_FloodingLogger"

        def execute(self, inputs, params, progress_callback=None, *, context=None):
            # One point to wake the pump, which parks in the consumer below.
            context.log_metric("train_loss", 1.0, 1)
            observed["parked"] = consumer_parked.wait(timeout=5.0)
            for step in range(2, int(params.get("count", 5)) + 1):
                context.log_metric("train_loss", 1.0 / step, step)
            loop.call_soon_threadsafe(burst_queued.set)
            return {"value": "flooded"}

    signals: list[object] = []

    async def gated_consumer(signal):
        signals.append(signal)
        if consumer_parked.is_set():
            return
        consumer_parked.set()
        # Still inside the engine's delivery lock, so every later drain waits
        # here — the node's remaining points have nowhere to go but out.
        await asyncio.wait_for(burst_queued.wait(), timeout=5.0)

    context = ExecutionContext(outbox=EventOutbox(capacity=capacity))
    nodes, edges = _graph(count=total)
    nodes[1]["type"] = "_FloodingLogger"
    registry._nodes["_FloodingLogger"] = _FloodingLogger
    try:
        outputs = await execute_graph(nodes, edges, context=context,
                                      on_signal=gated_consumer)
    finally:
        registry._nodes.pop("_FloodingLogger", None)

    assert observed.get("parked"), "the consumer never parked; nothing was held"
    assert outputs["logger"]["value"], "the run completed regardless"

    metrics = [s for s in signals if isinstance(s, MetricSignal)]
    # Point 1 was delivered on its own; of the other 39 only the last
    # `capacity` can survive, so the shed count is exact rather than "> 0".
    expected_dropped = total - 1 - capacity
    assert [s.count for s in signals if isinstance(s, DroppedSignal)] == [
        expected_dropped
    ], "one stalled drain, one drop-count covering everything it missed"
    assert len(metrics) + expected_dropped == total, (
        "every point is either delivered or accounted for as dropped"
    )
    # The tail survives: what a chart needs most is where the run got to.
    assert [s.step for s in metrics] == [
        1, *range(total - capacity + 1, total + 1)]


@pytest.mark.asyncio
async def test_signals_are_discarded_without_a_consumer():
    """No ``on_signal`` (the REST contract runner) is a no-op, not a crash.

    And the run says so: ``can_record_artifacts`` is how a node about to
    write a FILE alongside its artifact row finds out that the row has
    nowhere to go — see ``loop_control.save_interrupt_checkpoint``.
    """
    context = ExecutionContext()
    nodes, edges = _graph(count=3, artifact=True)
    outputs = await execute_graph(nodes, edges, context=context)
    assert outputs["logger"]["value"]
    assert context.can_record_artifacts() is False
    assert context.outbox.drain() == ([], 0), "the engine drained on the way out"
