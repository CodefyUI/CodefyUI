"""Worker messages carry a code and parameters beside their text (#841).

A worker run's ``run_warning`` and its error read "worker X went away:
<reason>" in English, which the API and the CLI keep. The Runs panel words
them in the viewer's language from ``code`` / ``params`` (the warning's
payload, the row's ``execution_facts.error_reason``) instead of matching
the English. These tests pin every reason a worker's runs lose it for, and
where the codes land.

Fixtures and helpers are local, following the house rule that test modules
stay independent.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.core import worker_protocol as wp
from app.core.db import Database
from app.core.execution_context import ExecutionContext
from app.core.executors import ExecutionRequest, RunSinks, WorkerExecutor
from app.core.run_store import RunProvenance, RunStore
from app.core.scheduling import Placement
from app.core.workers import (
    REKEYED,
    WorkerReason,
    WorkerRegistry,
    gone_message,
    no_reconnect,
    reason_dict,
    removed,
    ssh_stopped,
)


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def sinks(self) -> RunSinks:
        async def emit(event_type, payload):
            self.calls.append(("emit", event_type, payload))

        async def nothing(*_args):
            return None

        async def connection(connected):
            self.calls.append(("connection", connected))

        return RunSinks(emit=emit, record_metrics=nothing,
                        record_artifact=nothing, warn=nothing,
                        record_execution_facts=nothing,
                        record_worker_connection=connection)

    def warnings(self) -> list[dict[str, Any]]:
        return [c[2] for c in self.calls
                if c[0] == "emit" and c[1] == "run_warning"]


class _Agent:
    def __init__(self, registry: WorkerRegistry, *, fail: bool = False) -> None:
        self.sent: list[Any] = []
        self.got = asyncio.Event()

        async def send(frame: Any) -> None:
            if fail:
                raise ConnectionError("socket closed")
            self.sent.append(frame)
            self.got.set()

        async def close() -> None:
            return None

        self.worker, _ = registry.connect(
            wp.Hello(name="box", version="t", devices=[
                wp.DeviceReport(value="cuda:0", kind="cuda", index=0)]),
            send=send, close=close)

    async def assigned(self) -> None:
        while not any(isinstance(f, wp.Assign) for f in self.sent):
            self.got.clear()
            await asyncio.wait_for(self.got.wait(), 5)


@pytest.fixture
async def registry():
    registry = WorkerRegistry()
    yield registry
    await registry.shutdown()


def _request(recorder: _Recorder) -> ExecutionRequest:
    return ExecutionRequest(
        run_id="r1", graph={"nodes": []}, options={"seed": 1},
        context=ExecutionContext(execution_id="r1", device="cuda:0"),
        session=None, output_store=None, record_outputs=False,
        sinks=recorder.sinks(), explicit_metric_nodes=set(),
        placement=Placement("box", ("cuda:0",), (("box", "cuda:0"),)))


async def _until(predicate) -> None:
    for _ in range(500):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never held")


async def _started(registry: WorkerRegistry, recorder: _Recorder):
    agent = _Agent(registry)
    task = asyncio.create_task(WorkerExecutor(registry).run(
        _request(recorder)))
    await agent.assigned()
    registry.route(agent.worker, wp.RunStarted(run_id="r1", seq=1))
    return agent, task


def _gone(reason: dict[str, Any]) -> dict[str, Any]:
    return {"code": "worker_gone",
            "params": {"worker": "box", "reason": reason}}


# ── the reasons ───────────────────────────────────────────────────────────


def test_every_reason_keeps_its_english_and_names_its_code():
    assert (removed().code, removed().text, removed().params) == (
        "removed", "the worker was removed", {"revoked": False})
    assert removed(revoked=True).text == (
        "the worker was removed and its credential revoked")
    assert (removed(ssh=True).code, removed(ssh=True, revoked=True).text) == (
        "ssh_deleted", "the SSH worker was deleted and its credential revoked")
    assert (ssh_stopped("box").code, str(ssh_stopped("box"))) == (
        "ssh_stopped", "SSH worker box was stopped")
    assert REKEYED.code == "rekeyed"
    # A caller that still passes text gets the catch-all code.
    assert WorkerReason.of("the moon is full").payload() == {
        "code": "other", "params": {}, "text": "the moon is full"}
    assert WorkerReason.of(REKEYED) is REKEYED
    assert gone_message("box", "x").text == "worker box went away: x"


def test_wave_d_messages_keep_their_english_and_name_their_codes():
    """The adoption (#793), shutdown and S3 (#791) messages, coded in the
    #841 scheme; their English is what they said before."""
    from app.core.workers import (
        adopted_message,
        lost_after_restart_message,
        no_reconnect_after_restart,
        s3_transfer_failed_message,
        shutdown_not_connected,
        storage_fallback_message,
    )

    cases = [
        (adopted_message("box", 300), "worker_adopted",
         {"worker": "box", "seconds": 300},
         "this server restarted while the run was executing on worker box; "
         "the run keeps running there and waits up to 300 s for the worker "
         "to reconnect"),
        (lost_after_restart_message("box"), "worker_lost_after_restart",
         {"worker": "box"},
         "this server restarted and worker box no longer has the run"),
        (no_reconnect_after_restart(300), "no_reconnect_after_restart",
         {"seconds": 300},
         "it did not reconnect with the run within 300 s after this server "
         "restarted"),
        (shutdown_not_connected("box"), "shutdown_not_connected",
         {"worker": "box"},
         "this server is shutting down and worker box is not connected"),
        (s3_transfer_failed_message("box", "denied"),
         "worker_s3_transfer_failed", {"worker": "box", "error": "denied"},
         "cannot send this run's input files to the S3 bucket: denied"),
        (storage_fallback_message("box"), "worker_storage_fallback",
         {"worker": "box"},
         "worker box cannot use the S3 bucket (its agent has no boto3); "
         "this run's files go through this server instead"),
    ]
    for message, code, params, text in cases:
        assert (message.code, message.reason_dict()["params"], message.text) \
            == (code, params, text)


def test_reasons_are_immutable_and_hashable():
    """#848 review: the module constants are shared by every run, so their
    parameters cannot be changed through one, nested values included, and
    a reason can be a dict key or a set member."""
    from app.core.scheduling import QueueReason

    message = gone_message("box", no_reconnect("box", 300))
    with pytest.raises(TypeError):
        message.params["worker"] = "other"  # type: ignore[index]
    with pytest.raises(TypeError):
        message.params["reason"]["params"]["seconds"] = 1  # type: ignore[index]
    # What leaves it is a plain copy: changing that changes nothing here.
    copy = message.reason_dict()
    copy["params"]["reason"]["params"]["seconds"] = 1
    assert message.reason_dict()["params"]["reason"]["params"] == {
        "worker": "box", "seconds": 300}
    queued = QueueReason("any_unavailable",
                         {"devices": "cuda", "draining": ["a"]}, "text")
    assert queued.params["draining"] == ("a",)
    assert queued.params_dict() == {"devices": "cuda", "draining": ["a"]}
    assert len({message, REKEYED, queued, REKEYED}) == 3
    assert reason_dict("plain text") is None
    assert reason_dict(REKEYED) == {"code": "rekeyed", "params": {}}


# ── where they land ───────────────────────────────────────────────────────


async def test_a_disconnect_and_a_reconnect_warn_with_codes(registry):
    recorder = _Recorder()
    agent, task = await _started(registry, recorder)
    registry.disconnect(agent.worker)
    await _until(lambda: ("connection", False) in recorder.calls)
    warning = recorder.warnings()[-1]
    assert warning["detail"] == (
        "worker box disconnected (the connection closed); the run keeps "
        "running there and waits up to 300 s for it to reconnect")
    assert (warning["code"], warning["params"]) == ("worker_disconnected", {
        "worker": "box", "seconds": 300, "reason": {
            "code": "connection_closed", "params": {},
            "text": "the connection closed"}})

    back = _Agent(registry)
    registry.run_alive(back.worker, wp.RunAlive(runs=[
        wp.RunAliveEntry(run_id="r1", last_seq=1)]))
    await _until(lambda: ("connection", True) in recorder.calls)
    warning = recorder.warnings()[-1]
    assert (warning["code"], warning["params"]) == (
        "worker_reconnected", {"worker": "box"})
    registry.route(back.worker, wp.RunFinished(run_id="r1", seq=2,
                                               status="succeeded"))
    assert (await asyncio.wait_for(task, 5)).error_reason is None


async def test_a_reconnect_without_the_run_ends_it_with_a_code(registry):
    clock = [100.0]
    registry._clock = lambda: clock[0]
    recorder = _Recorder()
    agent, task = await _started(registry, recorder)
    registry.disconnect(agent.worker, "test")
    await _until(lambda: ("connection", False) in recorder.calls)
    back = _Agent(registry)
    clock[0] += back.worker.heartbeat_s + 0.1
    registry.check()
    outcome = await asyncio.wait_for(task, 5)
    assert outcome.error == ("worker box went away: after reconnecting it "
                             "did not report this run as still running")
    assert outcome.error_reason == _gone({
        "code": "alive_missing", "params": {},
        "text": "after reconnecting it did not report this run as still "
                "running"})
    # The warning that closes the run says the same, with the same code.
    assert recorder.warnings()[-1]["code"] == "worker_gone"


async def test_an_agent_that_lost_the_run_says_so_with_a_code(registry):
    recorder = _Recorder()
    agent, task = await _started(registry, recorder)
    registry.disconnect(agent.worker)
    back = _Agent(registry)
    registry.run_alive(back.worker, wp.RunAlive(runs=[]))
    registry.check()
    await _until(lambda: task.done())
    reason = task.result().error_reason["params"]["reason"]
    assert reason["code"] == "agent_lost_run"


async def test_silence_and_an_expired_grace_have_codes(registry):
    clock = [100.0]
    registry._clock = lambda: clock[0]
    recorder = _Recorder()
    agent, task = await _started(registry, recorder)
    clock[0] += agent.worker.heartbeat_s * registry.missed_heartbeats + 1
    registry.check()
    await _until(lambda: ("connection", False) in recorder.calls)
    reason = recorder.warnings()[-1]["params"]["reason"]
    assert reason["code"] == "no_heartbeat"
    assert reason["params"]["seconds"] == round(
        agent.worker.heartbeat_s * registry.missed_heartbeats + 1)
    assert reason["text"] == f"no heartbeat for {reason['params']['seconds']} s"

    clock[0] += registry.resume_grace_s
    registry.check()
    outcome = await asyncio.wait_for(task, 5)
    assert outcome.error_reason == _gone({
        "code": "no_reconnect", "params": {"worker": "box", "seconds": 300},
        "text": "worker box did not reconnect within 300 s"})


@pytest.mark.parametrize("end, code, params", [
    (lambda r: r.forget("box"), "removed", {"revoked": False}),
    (lambda r: r.forget("box", removed(revoked=True, ssh=True)),
     "ssh_deleted", {"revoked": True}),
    (lambda r: r.stopped("box", ssh_stopped("box")), "ssh_stopped",
     {"worker": "box"}),
    (lambda r: r.forget("box", "a plain reason"), "other", {}),
])
async def test_a_removed_or_stopped_worker_ends_its_runs_with_a_code(
        registry, end, code, params):
    recorder = _Recorder()
    _agent, task = await _started(registry, recorder)
    end(registry)
    outcome = await asyncio.wait_for(task, 5)
    assert outcome.status == "interrupted"
    reason = outcome.error_reason["params"]["reason"]
    assert (reason["code"], reason["params"]) == (code, params)
    assert outcome.error == f"worker box went away: {reason['text']}"


async def test_a_rekey_and_a_shutdown_have_codes(registry):
    recorder = _Recorder()
    agent, _task = await _started(registry, recorder)
    registry.kick("box", REKEYED)
    await _until(lambda: ("connection", False) in recorder.calls)
    assert recorder.warnings()[-1]["params"]["reason"]["code"] == "rekeyed"
    _task.cancel()

    other = _Recorder()
    agent, task = await _started(registry, other)
    await registry.shutdown()
    outcome = await asyncio.wait_for(task, 5)
    assert outcome.error_reason["params"]["reason"]["code"] == "shutting_down"


async def test_an_absent_or_unreachable_worker_has_a_code(registry):
    recorder = _Recorder()
    outcome = await WorkerExecutor(registry).run(_request(recorder))
    assert outcome.error == "worker box is not connected"
    assert outcome.error_reason == {"code": "worker_not_connected",
                                    "params": {"worker": "box"}}
    assert recorder.warnings()[-1]["code"] == "worker_not_connected"

    _Agent(registry, fail=True)
    outcome = await WorkerExecutor(registry).run(_request(_Recorder()))
    assert outcome.error == "could not reach worker box: socket closed"
    assert outcome.error_reason == {
        "code": "worker_unreachable",
        "params": {"worker": "box", "error": "socket closed"}}


# ── the row ───────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    try:
        yield RunStore(database)
    finally:
        database.close()


async def test_the_rows_error_keeps_its_code_in_execution_facts(store):
    reason = {"code": "worker_removed", "params": {"worker": "box"}}
    # A queued run has no facts document, and gets none: NULL stays "not
    # recorded" (#848 review). Its code is on its execution_error event.
    queued = await store.create_run(graph_snapshot={"nodes": []}, options={},
                                    provenance=RunProvenance())
    assert await store.mark_finished(queued.id, "failed",
                                     error="worker box was removed",
                                     expected=("queued",),
                                     error_reason=reason)
    row = await store.get_run(queued.id)
    assert (row.error, row.execution_facts) == (
        "worker box was removed", None)

    # A started run's facts keep everything else.
    started = await store.create_run(graph_snapshot={"nodes": []}, options={},
                                     provenance=RunProvenance())
    await store.mark_running(started.id, execution_facts={"device": "cuda:0"})
    assert await store.mark_finished(started.id, "interrupted", error="gone",
                                     error_reason=reason)
    assert (await store.get_run(started.id)).execution_facts == {
        "device": "cuda:0", "error_reason": reason}
    # A write that does not land changes nothing.
    assert not await store.mark_finished(started.id, "failed", error="again",
                                         error_reason={"code": "x"})
    assert (await store.get_run(started.id)).execution_facts[
        "error_reason"] == reason
