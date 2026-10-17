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

        async def connection(connected):
            self.calls.append(("connection", connected))

        return RunSinks(emit=emit, record_metrics=record_metrics,
                        record_artifact=record_artifact, warn=warn,
                        record_execution_facts=facts,
                        record_worker_connection=connection)


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


@pytest.fixture
async def no_grace():
    """A registry whose disconnected workers' runs end at once (#783's
    ``CODEFYUI_WORKER_RESUME_GRACE_S=0``, the behaviour before it)."""
    registry = WorkerRegistry(resume_grace_s=0)
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
    # Acknowledged once recorded (#783).
    acks = [f for f in agent.sent if isinstance(f, wp.Ack)]
    assert [(a.run_id, a.seq) for a in acks] == [("r1", 6)]
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


async def test_a_worker_that_goes_away_interrupts_with_a_warning(no_grace):
    registry = no_grace
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


async def test_a_worker_gone_after_a_stop_is_still_a_stop(no_grace):
    registry = no_grace
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


# ── disconnected, resume, late results (#783) ────────────────────────────


async def test_a_disconnect_waits_and_a_reconnect_resumes(registry):
    recorder = _Recorder()
    agent = _Agent(registry)
    request = _request(recorder)
    task = asyncio.create_task(WorkerExecutor(registry).run(request))
    await agent.wait_for(wp.Assign)
    registry.route(agent.worker, wp.RunStarted(run_id="r1", seq=1))
    registry.disconnect(agent.worker, "the connection closed")
    await _until(lambda: ("connection", False) in recorder.calls)
    assert not task.done()
    warning = [c for c in recorder.calls if c[0] == "emit"][-1]
    assert warning[2]["kind"] == "worker_disconnected"
    assert "waits up to 300 s" in warning[2]["detail"]
    state = registry.run_state("box", "r1")
    assert state.detached_wall is not None

    back = _Agent(registry)  # same name, after the drop
    registry.run_alive(back.worker, wp.RunAlive(runs=[
        wp.RunAliveEntry(run_id="r1", last_seq=3)]))
    await _until(lambda: ("connection", True) in recorder.calls)
    assert registry.run_state("box", "r1").detached_wall is None
    # The replay overlaps what arrived before the drop: seq 1 is dropped.
    for frame in (wp.RunStarted(run_id="r1", seq=1),
                  wp.Event(run_id="r1", seq=2, event_type="node_status",
                           payload={"node_id": "a", "status": "running"}),
                  wp.RunFinished(run_id="r1", seq=3, status="succeeded")):
        registry.route(back.worker, frame)
    outcome = await asyncio.wait_for(task, 5)
    assert outcome.status == "succeeded"
    kinds = [c[2]["kind"] for c in recorder.calls
             if c[0] == "emit" and c[1] == "run_warning"]
    assert kinds == ["worker_disconnected", "worker_reconnected"]
    node_events = [c for c in recorder.calls
                   if c[0] == "emit" and c[1] == "node_status"]
    assert len(node_events) == 1
    assert [f.seq for f in back.sent if isinstance(f, wp.Ack)][-1] == 3


@pytest.mark.parametrize("dropped_first", [True, False])
async def test_a_worker_stopped_on_purpose_ends_its_runs_at_once(
        registry, dropped_first):
    """#784: an SSH worker's Disable / Remove / shutdown. Its runs do not
    wait out the resume grace: nothing is coming back for them."""
    recorder = _Recorder()
    agent = _Agent(registry)
    task = asyncio.create_task(WorkerExecutor(registry).run(
        _request(recorder)))
    await agent.wait_for(wp.Assign)
    registry.route(agent.worker, wp.RunStarted(run_id="r1", seq=1))
    if dropped_first:
        # The tunnel went first: the run is waiting for the worker.
        registry.disconnect(agent.worker, "the connection closed")
        await _until(lambda: ("connection", False) in recorder.calls)
        assert not task.done()
    registry.stopped("box", "SSH worker box was stopped")
    # ``DELETE /api/workers/ssh/{id}`` forgets the worker right after the
    # launcher stopped it: the run is not ended a second time.
    assert registry.run_state("box", "r1") is None
    registry.forget("box", "the SSH worker was deleted")
    outcome = await asyncio.wait_for(task, 5)
    assert outcome.status == "interrupted"
    assert "SSH worker box was stopped" in outcome.error
    assert "deleted" not in outcome.error
    registry.stopped("box", "again")  # forgotten: harmless
    registry.stopped("nobody", "unknown")


async def test_past_the_grace_the_run_is_interrupted_and_late_results_noted(
        registry):
    clock = [100.0]
    registry._clock = lambda: clock[0]
    late: list[tuple] = []

    async def on_late(worker, run_id, frame):
        late.append((worker, run_id, frame.status))

    registry.on_late_result = on_late
    recorder = _Recorder()
    agent = _Agent(registry)
    task = asyncio.create_task(WorkerExecutor(registry).run(
        _request(recorder)))
    await agent.wait_for(wp.Assign)
    registry.route(agent.worker, wp.RunStarted(run_id="r1", seq=1))
    registry.disconnect(agent.worker, "test")
    await _until(lambda: ("connection", False) in recorder.calls)
    clock[0] += 299
    registry.check()
    assert not task.done()
    clock[0] += 2
    registry.check()
    outcome = await asyncio.wait_for(task, 5)
    assert outcome.status == "interrupted"
    assert "did not reconnect within 300 s" in outcome.error

    back = _Agent(registry)
    registry.run_alive(back.worker, wp.RunAlive(runs=[
        wp.RunAliveEntry(run_id="r1", last_seq=5)]))
    # Its slots went back: the worker is told to stop it at once.
    cancel = await back.wait_for(wp.Cancel)
    assert (cancel.run_id, cancel.grace_s) == ("r1", 0)
    registry.route(back.worker, wp.RunStarted(run_id="r1", seq=1))  # dup
    registry.route(back.worker, wp.RunFinished(run_id="r1", seq=5,
                                               status="succeeded"))
    await _until(lambda: late)
    assert late == [("box", "r1", "succeeded")]
    await _until(lambda: any(isinstance(f, wp.Ack) for f in back.sent))
    assert [f.seq for f in back.sent if isinstance(f, wp.Ack)] == [5]
    # Reported once: a replayed run_finished is acknowledged, not recorded.
    registry.route(back.worker, wp.RunFinished(run_id="r1", seq=5,
                                               status="succeeded"))
    await asyncio.sleep(0.05)
    assert len(late) == 1


async def test_a_stop_while_disconnected_reaches_the_worker_when_back(
        registry):
    recorder = _Recorder()
    agent = _Agent(registry)
    request = _request(recorder)
    task = asyncio.create_task(WorkerExecutor(registry).run(request))
    await agent.wait_for(wp.Assign)
    registry.disconnect(agent.worker, "test")
    await _until(lambda: ("connection", False) in recorder.calls)
    request.context.cancel()
    await asyncio.sleep(0.5)
    back = _Agent(registry)
    registry.run_alive(back.worker, wp.RunAlive(runs=[
        wp.RunAliveEntry(run_id="r1", last_seq=0)]))
    cancel = await back.wait_for(wp.Cancel)
    assert cancel.run_id == "r1"
    registry.route(back.worker, wp.RunFinished(run_id="r1", seq=1,
                                               status="cancelled"))
    assert (await asyncio.wait_for(task, 5)).status == "cancelled"


async def test_removing_a_disconnected_worker_ends_its_runs(registry):
    """Remove while the run waits for a reconnect: it ends at once, and the
    registry has nothing left to expire (the run used to hang forever)."""
    recorder = _Recorder()
    agent = _Agent(registry)
    task = asyncio.create_task(WorkerExecutor(registry).run(
        _request(recorder)))
    await agent.wait_for(wp.Assign)
    registry.disconnect(agent.worker, "network dropped")
    await _until(lambda: ("connection", False) in recorder.calls)
    registry.forget("box", "the worker was removed")
    outcome = await asyncio.wait_for(task, 5)
    assert outcome.status == "interrupted"
    assert "the worker was removed" in outcome.error
    assert registry.get("box") is None
    assert agent.worker.running == {}


async def test_removing_an_online_worker_ends_its_runs(registry):
    recorder = _Recorder()
    agent = _Agent(registry)
    task = asyncio.create_task(WorkerExecutor(registry).run(
        _request(recorder)))
    await agent.wait_for(wp.Assign)
    registry.forget("box")
    outcome = await asyncio.wait_for(task, 5)
    assert outcome.status == "interrupted"


async def test_a_stop_waiting_for_a_removed_worker_ends_the_run(registry):
    """A Stop waiting for a disconnected worker does not retry forever once
    the worker is removed: the run ends, as the user's stop."""
    recorder = _Recorder()
    agent = _Agent(registry)
    request = _request(recorder)
    task = asyncio.create_task(WorkerExecutor(registry).run(request))
    await agent.wait_for(wp.Assign)
    registry.disconnect(agent.worker, "network dropped")
    await _until(lambda: ("connection", False) in recorder.calls)
    request.context.cancel()
    await asyncio.sleep(0.3)  # the watcher is retrying the cancel
    assert not task.done()
    registry.forget("box")
    assert (await asyncio.wait_for(task, 5)).status == "cancelled"


async def test_acks_come_every_n_frames_and_after_a_quiet_second(
        registry, monkeypatch):
    from app.core import executors

    monkeypatch.setattr(executors, "ACK_EVERY", 3)
    monkeypatch.setattr(executors, "ACK_INTERVAL_S", 0.2)
    recorder = _Recorder()
    agent = _Agent(registry)
    task = asyncio.create_task(WorkerExecutor(registry).run(
        _request(recorder)))
    await agent.wait_for(wp.Assign)
    for seq in range(1, 5):
        registry.route(agent.worker, wp.Event(
            run_id="r1", seq=seq, event_type="node_status",
            payload={"node_id": "a", "status": "progress"}))
    await _until(lambda: [f.seq for f in agent.sent
                          if isinstance(f, wp.Ack)] == [3, 4], timeout=3)
    registry.route(agent.worker, wp.RunFinished(run_id="r1", seq=5,
                                                status="succeeded"))
    assert (await asyncio.wait_for(task, 5)).status == "succeeded"
    assert [f.seq for f in agent.sent if isinstance(f, wp.Ack)] == [3, 4, 5]


async def test_orphan_frames_are_acknowledged(registry):
    agent = _Agent(registry)
    registry.route(agent.worker, wp.Event(
        run_id="ghost", seq=64, event_type="node_status", payload={}))
    registry.route(agent.worker, wp.RunFinished(run_id="ghost", seq=65,
                                                status="failed"))
    await _until(lambda: len([f for f in agent.sent
                              if isinstance(f, wp.Ack)]) == 2)


async def _until(predicate: Any, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not met")
        await asyncio.sleep(0.01)


async def test_a_heartbeat_naming_an_untracked_run_stops_it(registry):
    """Architecture review of #772: a subprocess nothing here tracks would
    keep its card while the scheduler hands it to another run."""
    agent = _Agent(registry)
    registry.open_run(agent.worker, "tracked")
    registry.heartbeat(agent.worker, wp.Heartbeat(running=["tracked",
                                                           "stray"]))
    cancel = await agent.wait_for(wp.Cancel)
    assert (cancel.run_id, cancel.grace_s) == ("stray", 0)
    await asyncio.sleep(0.05)
    assert [f.run_id for f in agent.sent if isinstance(f, wp.Cancel)] == [
        "stray"]
    registry.close_run("box", "tracked")


async def test_a_gap_in_seq_is_logged_and_the_frame_kept(registry, caplog):
    agent = _Agent(registry)
    run = registry.open_run(agent.worker, "r1")
    registry.route(agent.worker, wp.RunStarted(run_id="r1", seq=1))
    with caplog.at_level("WARNING", logger="app.core.workers"):
        registry.route(agent.worker, wp.RunStarted(run_id="r1", seq=4))
    assert "frames 2-3 never arrived" in caplog.text
    assert run.inbox.qsize() == 2 and run.last_seq == 4


async def test_a_long_replay_pauses_the_socket_until_the_inbox_drains(
        registry, monkeypatch):
    from app.core import workers

    monkeypatch.setattr(workers, "INBOX_HIGH_WATER", 10)
    monkeypatch.setattr(workers, "INBOX_DRAIN_POLL_S", 0.01)
    agent = _Agent(registry)
    run = registry.open_run(agent.worker, "r1")
    frame = None
    for seq in range(1, 9):
        frame = wp.RunStarted(run_id="r1", seq=seq)
        registry.route(agent.worker, frame)
    await asyncio.wait_for(registry.backpressure(agent.worker, frame), 1)
    for seq in range(9, 13):
        frame = wp.RunStarted(run_id="r1", seq=seq)
        registry.route(agent.worker, frame)
    waiter = asyncio.create_task(registry.backpressure(agent.worker, frame))
    await asyncio.sleep(0.05)
    assert not waiter.done()  # 12 queued: past the mark, reading pauses
    while run.inbox.qsize() > 5:
        run.inbox.get_nowait()
    await asyncio.wait_for(waiter, 1)
    registry.close_run("box", "r1")


async def test_a_run_that_just_finished_here_is_not_a_stray(registry):
    """Review of #783: a heartbeat may still list a run whose subprocess
    is exiting after its run_finished; that is no reason to cancel it."""
    clock = [500.0]
    registry._clock = lambda: clock[0]
    recorder = _Recorder()
    agent = _Agent(registry)
    task = asyncio.create_task(WorkerExecutor(registry).run(
        _request(recorder)))
    await agent.wait_for(wp.Assign)
    registry.route(agent.worker, wp.RunFinished(run_id="r1", seq=1,
                                                status="succeeded"))
    assert (await asyncio.wait_for(task, 5)).status == "succeeded"
    registry.heartbeat(agent.worker, wp.Heartbeat(running=["r1"]))
    await asyncio.sleep(0.05)
    assert not any(isinstance(f, wp.Cancel) for f in agent.sent)
    clock[0] += 61  # long past its exit: a stray after all
    registry.heartbeat(agent.worker, wp.Heartbeat(running=["r1"]))
    cancel = await agent.wait_for(wp.Cancel)
    assert (cancel.run_id, cancel.grace_s) == ("r1", 0)


async def test_any_frame_keeps_a_worker_alive(registry):
    """Review of #783: a long replay must not read as missed heartbeats."""
    clock = [100.0]
    registry._clock = lambda: clock[0]
    agent = _Agent(registry)
    clock[0] += 25
    registry.touch(agent.worker)  # a replayed frame, say
    clock[0] += 25
    registry.check()
    assert agent.worker.status == "degraded"  # late, not gone
    registry.touch(agent.worker)
    assert agent.worker.status == "online"
    clock[0] += 31
    registry.check()
    assert agent.worker.status == "disconnected"
# ── storage (#782) ───────────────────────────────────────────────────────


async def test_assign_names_the_mount_and_a_mounted_artifact_keeps_it(
        registry, monkeypatch):
    from app.config import settings
    from app.core.executors import MountedArtifactSignal

    monkeypatch.setattr(settings, "MOUNT_LABEL", "data")
    recorder = _Recorder()
    frames = [
        wp.Artifact(run_id="r1", seq=1, kind="checkpoint",
                    path="models/periodic/a.pt", mount_label="data"),
        wp.RunFinished(run_id="r1", seq=2, status="succeeded"),
    ]
    _, assign, _ = await _run(registry, recorder, frames)
    # Another machine: the label only, never this server's path.
    assert assign.storage == wp.Storage(mount="data")
    signal = recorder.calls[0][1]
    assert isinstance(signal, MountedArtifactSignal)
    assert (signal.path, signal.mount_label) == ("models/periodic/a.pt",
                                                 "data")


async def test_a_same_host_worker_without_a_mount_borrows_this_servers_path(
        registry, monkeypatch, tmp_path):
    from app.config import settings
    from app.core.executors import assignment_storage

    monkeypatch.setattr(settings, "MOUNT_LABEL", "data")
    # Configured directories that do not share one parent: what is lent is
    # what is configured, not a layout derived from MODELS_DIR.parent.
    monkeypatch.setattr(settings, "DATA_FILES_DIR", tmp_path / "uploads")
    monkeypatch.setattr(settings, "IMAGES_DIR", tmp_path / "pics")
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "m" / "models")
    monkeypatch.setattr(settings, "MEDIA_DIR", tmp_path / "media")
    agent = _Agent(registry)
    agent.worker.same_host = True
    resolved = tmp_path.resolve()
    assert assignment_storage(agent.worker) == wp.Storage(
        mount="data", relative_artifacts=False, dirs={
            "CODEFYUI_DATA_FILES_DIR": str(resolved / "uploads"),
            "CODEFYUI_IMAGES_DIR": str(resolved / "pics"),
            "CODEFYUI_MODELS_DIR": str(resolved / "m" / "models"),
            "CODEFYUI_MEDIA_DIR": str(resolved / "media")})
    # Its own directory for the label wins: nothing is lent, and its files
    # are still on this machine (absolute paths).
    agent.worker.mounts = {"data": "/mnt/own"}
    assert assignment_storage(agent.worker) == wp.Storage(
        mount="data", relative_artifacts=False)
