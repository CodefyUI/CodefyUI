"""``WorkerExecutor``: a worker's frames into a run's sinks (#772)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.core import worker_protocol as wp
from app.core.execution_context import ExecutionContext
from app.core.executors import ExecutionRequest, RunSinks, WorkerExecutor
from app.core.scheduling import Placement
from app.core.workers import WorkerRegistry


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def sinks(self) -> RunSinks:
        async def emit(event_type, payload):
            self.calls.append(("emit", event_type, payload))

        async def record_metrics(points):
            self.calls.append(("metrics", list(points)))

        async def record_artifact(signal):
            self.calls.append(("artifact", signal))

        async def warn(signal):
            self.calls.append(("warn", signal))

        async def facts(patch):
            self.calls.append(("facts", patch))

        return RunSinks(emit=emit, record_metrics=record_metrics,
                        record_artifact=record_artifact, warn=warn,
                        record_execution_facts=facts)


class _Agent:
    """A registered worker whose ``send`` is captured."""

    def __init__(self, registry: WorkerRegistry, *, fail: bool = False) -> None:
        self.sent: list[Any] = []
        self.fail = fail
        self.got = asyncio.Event()

        async def send(frame: Any) -> None:
            if self.fail:
                raise ConnectionError("socket closed")
            self.sent.append(frame)
            self.got.set()

        async def close() -> None:
            return None

        self.worker, _ = registry.connect(
            wp.Hello(name="box", version="t", devices=[
                wp.DeviceReport(value="cuda:2", kind="cuda", index=2),
                wp.DeviceReport(value="cuda:5", kind="cuda", index=5)]),
            send=send, close=close)

    async def wait_for(self, kind: type) -> Any:
        while True:
            for frame in self.sent:
                if isinstance(frame, kind):
                    return frame
            self.got.clear()
            await asyncio.wait_for(self.got.wait(), 5)


@pytest.fixture
async def registry():
    registry = WorkerRegistry()
    yield registry
    await registry.shutdown()


def _request(recorder: _Recorder, *, worker: str = "box") -> ExecutionRequest:
    context = ExecutionContext(execution_id="r1", device="cuda:2")
    return ExecutionRequest(
        run_id="r1", graph={"nodes": []}, options={"seed": 1},
        context=context, session=None, output_store=None,
        record_outputs=False, sinks=recorder.sinks(),
        explicit_metric_nodes=set(),
        placement=Placement(worker, ("cuda:2", "cuda:5"),
                            ((worker, "cuda:2"), (worker, "cuda:5"))))


async def _run(registry, recorder, frames, **kw):
    agent = _Agent(registry, **kw)
    task = asyncio.create_task(WorkerExecutor(registry).run(
        _request(recorder)))
    assign = await agent.wait_for(wp.Assign)
    for frame in frames:
        registry.route(agent.worker, frame)
    outcome = await asyncio.wait_for(task, 5)
    return agent, assign, outcome


async def test_frames_become_sink_calls_in_order(registry):
    recorder = _Recorder()
    frames = [
        wp.RunStarted(run_id="r1", seq=1, execution_facts={
            "visible_devices": "2,5"}),
        wp.Event(run_id="r1", seq=2, event_type="node_status",
                 payload={"node_id": "a", "status": "running"}),
        wp.Metrics(run_id="r1", seq=3, points=[
            wp.MetricPointReport(name="loss", value=1.0, step=2,
                                 node_id="a")]),
        wp.Artifact(run_id="r1", seq=4, kind="checkpoint", path="/c.pt",
                    meta={"epoch": 1}, node_id="a"),
        wp.RunWarning(run_id="r1", seq=5, kind="k", detail="d"),
        wp.RunFinished(run_id="r1", seq=6, status="succeeded",
                       peak_memory={"cuda:0": {"peak_bytes": 1},
                                    "cuda:1": {"peak_bytes": 2}}),
    ]
    agent, assign, outcome = await _run(registry, recorder, frames)
    assert (assign.device, assign.devices, assign.options) == (
        "cuda:2", ["cuda:2", "cuda:5"], {"seed": 1})
    kinds = [call[0] for call in recorder.calls]
    assert kinds == ["facts", "emit", "metrics", "metrics", "artifact",
                     "warn"]
    assert recorder.calls[0] == ("facts", {"visible_devices": "2,5"})
    assert recorder.calls[3][1][0].name == "loss"
    assert recorder.calls[4][1].path == "/c.pt"
    assert outcome.status == "succeeded"
    # The subprocess's cuda:0/1 are this worker's cuda:2/5.
    assert outcome.peak_memory == {"cuda:2": {"peak_bytes": 1},
                                   "cuda:5": {"peak_bytes": 2}}
    assert agent.worker.running == {}


@pytest.mark.parametrize("finished, status, error", [
    (dict(status="cancelled"), "cancelled", None),
    # No stop was asked here: the agent's own ``interrupted`` stands.
    (dict(status="interrupted", error="exited while stopping"),
     "interrupted", "exited while stopping"),
    (dict(status="failed", error="boom"), "failed", "boom"),
    (dict(status="failed", error="run subprocess exited with code 3",
          stderr_tail="Traceback"), "failed",
     "run subprocess exited with code 3\n--- output tail ---\nTraceback"),
])
async def test_run_finished_maps_to_an_outcome(registry, finished, status,
                                               error):
    _, _, outcome = await _run(registry, _Recorder(), [
        wp.RunFinished(run_id="r1", seq=1, **finished)])
    assert (outcome.status, outcome.error) == (status, error)


async def test_a_worker_that_goes_away_interrupts_with_a_warning(registry):
    recorder = _Recorder()
    agent = _Agent(registry)
    task = asyncio.create_task(WorkerExecutor(registry).run(
        _request(recorder)))
    await agent.wait_for(wp.Assign)
    registry.disconnect(agent.worker, "test")
    outcome = await asyncio.wait_for(task, 5)
    assert outcome.status == "interrupted"
    assert recorder.calls[-1][1] == "run_warning"
    assert recorder.calls[-1][2]["kind"] == "worker_disconnected"


async def test_after_a_stop_the_agents_interrupted_is_a_stop(registry):
    recorder = _Recorder()
    agent = _Agent(registry)
    request = _request(recorder)
    task = asyncio.create_task(WorkerExecutor(registry).run(request))
    await agent.wait_for(wp.Assign)
    request.context.cancel()
    await agent.wait_for(wp.Cancel)
    registry.route(agent.worker, wp.RunFinished(
        run_id="r1", seq=1, status="interrupted", error="terminated"))
    assert (await asyncio.wait_for(task, 5)).status == "cancelled"


async def test_a_worker_gone_after_a_stop_is_still_a_stop(registry):
    recorder = _Recorder()
    agent = _Agent(registry)
    request = _request(recorder)
    task = asyncio.create_task(WorkerExecutor(registry).run(request))
    await agent.wait_for(wp.Assign)
    request.context.cancel()
    registry.disconnect(agent.worker, "test")
    assert (await asyncio.wait_for(task, 5)).status == "cancelled"


async def test_cancel_reaches_the_replacing_connection(registry):
    """The stop goes to the worker's CURRENT socket, not the assign's."""
    recorder = _Recorder()
    old = _Agent(registry)
    request = _request(recorder)
    task = asyncio.create_task(WorkerExecutor(registry).run(request))
    await old.wait_for(wp.Assign)
    new = _Agent(registry)  # same name: replaces, inherits the run
    request.context.cancel()
    cancel = await new.wait_for(wp.Cancel)
    assert cancel.run_id == "r1"
    assert not any(isinstance(f, wp.Cancel) for f in old.sent)
    registry.route(new.worker, wp.RunFinished(run_id="r1", seq=1,
                                              status="cancelled"))
    assert (await asyncio.wait_for(task, 5)).status == "cancelled"


async def test_an_unreachable_or_unknown_worker_interrupts(registry):
    recorder = _Recorder()
    _Agent(registry, fail=True)
    outcome = await WorkerExecutor(registry).run(_request(recorder))
    assert outcome.status == "interrupted"
    recorder = _Recorder()
    outcome = await WorkerExecutor(registry).run(
        _request(recorder, worker="nobody"))
    assert outcome.status == "interrupted"
    assert "not connected" in recorder.calls[0][2]["detail"]


async def test_a_stop_is_forwarded_as_cancel(registry):
    recorder = _Recorder()
    agent = _Agent(registry)
    request = _request(recorder)
    task = asyncio.create_task(WorkerExecutor(registry).run(request))
    await agent.wait_for(wp.Assign)
    request.context.cancel()
    cancel = await agent.wait_for(wp.Cancel)
    assert cancel.run_id == "r1" and cancel.grace_s is None
    registry.route(agent.worker, wp.RunFinished(
        run_id="r1", seq=1, status="cancelled"))
    assert (await asyncio.wait_for(task, 5)).status == "cancelled"


async def test_duplicate_and_stray_frames_are_dropped(registry):
    agent = _Agent(registry)
    run = registry.open_run(agent.worker, "r1")
    registry.route(agent.worker, wp.RunStarted(run_id="r1", seq=2))
    registry.route(agent.worker, wp.RunStarted(run_id="r1", seq=2))
    registry.route(agent.worker, wp.RunStarted(run_id="r1", seq=1))
    registry.route(agent.worker, wp.RunStarted(run_id="other", seq=1))
    assert run.inbox.qsize() == 1
