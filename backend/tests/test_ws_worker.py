"""``WS /ws/worker`` with a fake agent over ``httpx-ws`` (#772).

The fake agent speaks ``core.worker_protocol`` by hand, so every test can
say exactly what a worker sends and when; the real agent is exercised end
to end in ``test_worker_e2e.py``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from httpx import AsyncClient
from httpx_ws import WebSocketDisconnect, aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport

from app.core import worker_protocol as wp
from app.core.auth import TOKEN_HEADER, TOKEN_QUERY_PARAM, session_token
from app.core.db import Database
from app.core.run_service import RunService
from app.core.run_store import RunStore
from app.core.workers import WorkerRegistry
from app.main import app

BASE = "http://127.0.0.1:8000"
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


def _path(token: str | None = None) -> str:
    return (f"{wp.WORKER_WS_PATH}?{TOKEN_QUERY_PARAM}="
            f"{session_token() if token is None else token}")


def _graph() -> dict[str, Any]:
    return {"nodes": [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "x", "type": "TensorCreate",
         "data": {"params": {"shape": "2", "fill": "ones"}}}],
        "edges": [{"id": "t", "source": "start", "target": "x",
                   "sourceHandle": "trigger", "type": "trigger"}]}


@pytest.fixture
async def env(tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    registry = WorkerRegistry(heartbeat_s=10)
    service = RunService(RunStore(database), workers=registry,
                         shutdown_grace_s=5.0)
    app.state.db = database
    app.state.run_service = service
    app.state.worker_registry = registry
    try:
        yield service, registry
    finally:
        await service.shutdown()
        await registry.shutdown()
        database.close()
        for attribute in ("db", "run_service", "worker_registry"):
            if hasattr(app.state, attribute):
                delattr(app.state, attribute)


def _client(token: bool = True) -> AsyncClient:
    headers = {TOKEN_HEADER: session_token()} if token else {}
    return AsyncClient(transport=ASGIWebSocketTransport(app=app),
                       base_url=BASE, headers=headers)


class FakeAgent:
    def __init__(self, ws: Any) -> None:
        self.ws = ws

    async def send(self, frame: Any) -> None:
        await self.ws.send_text(wp.encode_frame(frame))

    async def recv(self, timeout: float = 10.0) -> Any:
        raw = await asyncio.wait_for(self.ws.receive_text(), timeout)
        return wp.decode_control_frame(raw)

    async def hello(self, name: str = "box", devices: tuple = ("cpu",),
                    **kw: Any) -> Any:
        await self.send(wp.Hello(name=name, version="t", devices=[
            wp.DeviceReport(value=d, kind=d.split(":")[0]) for d in devices],
            **kw))
        return await self.recv()

    async def finish_run(self, assign: wp.Assign, status: str = "succeeded",
                         **kw: Any) -> None:
        run_id = assign.run_id
        await self.send(wp.RunStarted(run_id=run_id, seq=1, pid=1))
        await self.send(wp.Event(run_id=run_id, seq=2, event_type="node_status",
                                 payload={"node_id": "x",
                                          "status": "running"}))
        await self.send(wp.Event(run_id=run_id, seq=3, event_type="node_status",
                                 payload={"node_id": "x",
                                          "status": "completed"}))
        await self.send(wp.RunFinished(run_id=run_id, seq=4, status=status,
                                       **kw))


async def _terminal(service: RunService, run_id: str,
                    timeout: float = 10.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = await service.store.get_run(run_id)
        if record is not None and record.status in TERMINAL \
                and run_id not in service.active_run_ids():
            return record
        await asyncio.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish")


async def _until(predicate: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met")
        await asyncio.sleep(0.02)


# ── handshake ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("path, headers, code", [
    (_path("wrong"), {}, 4401),
    (wp.WORKER_WS_PATH, {}, 4401),
    (_path(), {"origin": "http://127.0.0.1:8000"}, 4003),
    (_path(), {"host": "evil.example"}, 4003),
])
async def test_a_refused_handshake_never_registers(env, path, headers, code):
    _, registry = env
    async with _client(token=False) as client:
        with pytest.raises(WebSocketDisconnect) as refused:
            async with aconnect_ws(path, client, headers=headers):
                pass
    assert refused.value.code == code
    assert registry.workers() == []


@pytest.mark.parametrize("hello, reason", [
    (wp.Hello(name="local", version="t"), "reserved"),
    (wp.Hello(name="any", version="t"), "reserved"),
    (wp.Hello(name="box", version="t", protocol_version=99),
     "protocol version"),
])
async def test_a_bad_hello_is_rejected(env, hello, reason):
    _, registry = env
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        agent = FakeAgent(ws)
        await agent.send(hello)
        reply = await agent.recv()
    assert isinstance(reply, wp.Reject) and reason in reply.reason
    assert registry.workers() == []


async def test_hello_registers_and_heartbeats_update(env):
    service, registry = env
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        agent = FakeAgent(ws)
        welcome = await agent.hello(devices=("cpu", "cuda:0"))
        assert isinstance(welcome, wp.Welcome)
        assert (welcome.worker_id, welcome.heartbeat_s, welcome.grace_s) == (
            "box", 10, wp.DEFAULT_GRACE_S)
        worker = registry.get("box")
        assert worker.online and worker.device_values() == ["cpu", "cuda:0"]
        await agent.send(wp.Heartbeat(devices=[wp.DeviceReport(
            value="cpu", kind="cpu", total_bytes=8, free_bytes=4,
            source="psutil")]))
        await _until(lambda: worker.devices[0].free_bytes == 4)
        listed = (await client.get("/api/workers")).json()["workers"]
        assert listed[0]["name"] == "box"
        assert listed[0]["devices"][0]["memory"]["free_bytes"] == 4
    await _until(lambda: not registry.get("box").online)


# ── runs ─────────────────────────────────────────────────────────────────


async def test_a_run_targeting_a_worker_executes_there(env):
    service, registry = env
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        agent = FakeAgent(ws)
        await agent.hello(devices=("cpu", "cuda:0", "cuda:1"))
        response = await client.post("/api/runs", json={
            "graph": _graph(), "options": {
                "device": "cuda", "seed": 1,
                "target": {"worker": "box"}}})
        assert response.status_code == 200, response.text
        run_id = response.json()["run_id"]
        assign = await agent.recv()
        assert isinstance(assign, wp.Assign)
        assert (assign.run_id, assign.device, assign.devices) == (
            run_id, "cuda:0", ["cuda:0"])
        assert assign.graph["nodes"][1]["type"] == "TensorCreate"
        await agent.send(wp.RunStarted(
            run_id=run_id, seq=1, pid=7,
            execution_facts={"visible_devices": "0",
                             "device_aliases": {"cuda:0": "cuda:0"}}))
        await agent.send(wp.Metrics(run_id=run_id, seq=2, points=[
            wp.MetricPointReport(name="loss", value=0.5, step=1)]))
        await agent.send(wp.RunWarning(run_id=run_id, seq=3, kind="k",
                                       detail="d"))
        await agent.send(wp.RunWarning(run_id=run_id, seq=3, kind="dup",
                                       detail="a duplicate seq is dropped"))
        await agent.send(wp.RunFinished(run_id=run_id, seq=4,
                                        status="succeeded",
                                        peak_memory={"cuda:0": {"peak": 1}}))
        record = await _terminal(service, run_id)
    assert record.status == "succeeded"
    assert record.queue_key == "box/cuda:0"
    assert record.worker_id == "box" and record.devices == ["cuda:0"]
    assert record.execution_facts["visible_devices"] == "0"
    assert record.execution_facts["device_aliases"] == {"cuda:0": "cuda:0"}
    assert record.execution_facts["peak_memory"] == {"cuda:0": {"peak": 1}}
    assert record.fingerprint_digest is None  # not this process's environment
    facts = record.execution_facts
    assert (facts["facts_source"], facts["worker_id"]) == ("worker", "box")
    assert facts["resolved"]["device"] == "cuda:0"
    assert facts["resolved"]["devices"] == ["cuda:0"]
    assert "numerics" not in facts["resolved"]  # this process's switches
    events = await service.store.get_events(run_id)
    warnings = [e.payload["kind"] for e in events if e.type == "run_warning"]
    assert warnings == ["k"]
    metrics = await service.store.get_metrics(run_id)
    assert [(m.name, m.value) for m in metrics] == [("loss", 0.5)]


async def test_events_crash_and_slot_release(env):
    service, registry = env
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        agent = FakeAgent(ws)
        await agent.hello(devices=("cpu",))
        first = (await client.post("/api/runs", json={
            "graph": _graph(), "options": {"target": {"worker": "box"}}})
            ).json()["run_id"]
        second = (await client.post("/api/runs", json={
            "graph": _graph(), "options": {"target": {"worker": "box"}}})
            ).json()["run_id"]
        third = (await client.post("/api/runs", json={
            "graph": _graph(), "options": {"target": {"worker": "box"}}})
            ).json()["run_id"]
        # cpu admits two at a time on a worker too.
        a, b = await agent.recv(), await agent.recv()
        assert {a.run_id, b.run_id} == {first, second}
        assert (await service.store.get_run(third)).status == "queued"
        await agent.finish_run(a)
        await agent.finish_run(b, status="failed",
                               error="run subprocess exited with code 3",
                               exit_code=3, stderr_tail="Traceback: boom")
        c = await agent.recv()
        assert c.run_id == third  # a freed slot served the queue
        await agent.finish_run(c)
        records = [await _terminal(service, r) for r in (a.run_id, b.run_id,
                                                         third)]
    assert [r.status for r in records] == ["succeeded", "failed", "succeeded"]
    assert "exited with code 3" in records[1].error
    assert "Traceback: boom" in records[1].error
    events = await service.store.get_events(a.run_id)
    assert [(e.type, (e.payload or {}).get("status")) for e in events] == [
        ("execution_start", None), ("node_status", "running"),
        ("node_status", "completed"), ("execution_complete", None)]
    assert service._slots == {}


async def test_stop_is_forwarded_and_the_workers_answer_ends_the_run(env):
    service, registry = env
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        agent = FakeAgent(ws)
        await agent.hello()
        run_id = (await client.post("/api/runs", json={
            "graph": _graph(), "options": {"target": {"worker": "box"}}})
            ).json()["run_id"]
        assign = await agent.recv()
        await agent.send(wp.RunStarted(run_id=run_id, seq=1))
        assert (await client.post(f"/api/runs/{run_id}/cancel")
                ).status_code == 200
        cancel = await agent.recv()
        assert isinstance(cancel, wp.Cancel) and cancel.run_id == run_id
        assert cancel.grace_s is None  # the agent's own chain decides
        # The agent had to terminate it: still the stop a user asked for.
        await agent.send(wp.RunFinished(run_id=assign.run_id, seq=2,
                                        status="interrupted"))
        record = await _terminal(service, run_id)
    assert record.status == "cancelled"


async def test_a_worker_that_goes_away_interrupts_its_runs(env):
    service, registry = env
    async with _client() as client:
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello()
            run_id = (await client.post("/api/runs", json={
                "graph": _graph(), "options": {"target": {"worker": "box"}}})
                ).json()["run_id"]
            await agent.recv()
        record = await _terminal(service, run_id)
        assert record.status == "interrupted"
        events = await service.store.get_events(run_id)
        assert any(e.type == "run_warning"
                   and e.payload["kind"] == "worker_disconnected"
                   for e in events)
        assert events[-1].type == "execution_stopped"
        assert service._slots == {}
        # And a new submission to it is refused while it is away.
        refused = await client.post("/api/runs", json={
            "graph": _graph(), "options": {"target": {"worker": "box"}}})
        assert refused.status_code == 400
        assert "not connected" in refused.json()["detail"]


async def test_heartbeat_timeout_disconnects_and_interrupts(env):
    service, registry = env
    clock = [1000.0]
    registry._clock = lambda: clock[0]
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        agent = FakeAgent(ws)
        await agent.hello()
        run_id = (await client.post("/api/runs", json={
            "graph": _graph(), "options": {"target": {"worker": "box"}}})
            ).json()["run_id"]
        await agent.recv()
        clock[0] += 16  # 1.5 heartbeats late
        registry.check()
        assert registry.get("box").status == "degraded"
        clock[0] += 15  # past 3 x 10 s
        registry.check()
        assert registry.get("box").status == "disconnected"
        record = await _terminal(service, run_id)
    assert record.status == "interrupted"


async def test_a_new_connection_replaces_the_old_one_and_keeps_live_runs(env):
    service, registry = env
    async with _client() as client:
        async with aconnect_ws(_path(), client) as old:
            first = FakeAgent(old)
            await first.hello()
            kept = (await client.post("/api/runs", json={
                "graph": _graph(), "options": {"target": {"worker": "box"}}})
                ).json()["run_id"]
            lost = (await client.post("/api/runs", json={
                "graph": _graph(), "options": {"target": {"worker": "box"}}})
                ).json()["run_id"]
            await first.recv()
            await first.recv()
            await first.send(wp.RunStarted(run_id=kept, seq=1))
            async with aconnect_ws(_path(), client) as new:
                second = FakeAgent(new)
                welcome = await second.hello()
                assert isinstance(welcome, wp.Welcome)
                await second.send(wp.RunAlive(runs=[
                    wp.RunAliveEntry(run_id=kept, last_seq=1)]))
                lost_record = await _terminal(service, lost)
                assert lost_record.status == "interrupted"
                await second.send(wp.RunFinished(run_id=kept, seq=2,
                                                 status="succeeded"))
                kept_record = await _terminal(service, kept)
    assert kept_record.status == "succeeded"
    assert len(registry.workers()) == 1


async def test_drain_holds_runs_and_undrain_releases_them(env):
    service, registry = env
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        agent = FakeAgent(ws)
        await agent.hello()
        drained = (await client.post("/api/workers/box/drain")).json()
        assert (drained["draining"], drained["status"]) == (True, "draining")
        run_id = (await client.post("/api/runs", json={
            "graph": _graph(), "options": {"target": {"worker": "box"}}})
            ).json()["run_id"]
        await asyncio.sleep(0.1)
        row = (await client.get(f"/api/runs/{run_id}")).json()
        assert (row["status"], row["queue_reason"]) == (
            "queued", "worker box is draining")
        assert (await client.post("/api/workers/box/undrain")).json()[
            "draining"] is False
        assign = await agent.recv()
        assert assign.run_id == run_id
        await agent.finish_run(assign)
        assert (await _terminal(service, run_id)).status == "succeeded"


async def test_any_prefers_a_worker_and_falls_back_to_this_process(env):
    service, registry = env
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        agent = FakeAgent(ws)
        await agent.hello()
        ids = [(await client.post("/api/runs", json={
            "graph": _graph(), "options": {"target": {"worker": "any"}}})
            ).json()["run_id"] for _ in range(3)]
        on_worker = [await agent.recv(), await agent.recv()]
        assert [a.run_id for a in on_worker] == ids[:2]
        # The third found the worker full and ran here.
        local = await _terminal(service, ids[2], timeout=60)
        assert local.status == "succeeded"
        assert (local.queue_key, local.worker_id) == ("cpu", "local")
        for assign in on_worker:
            await agent.finish_run(assign)
        for run_id in ids[:2]:
            record = await _terminal(service, run_id)
            assert (record.queue_key, record.worker_id) == ("box/cpu", "box")


async def test_a_queued_run_says_which_worker_it_waits_for(env):
    service, registry = env
    async with _client() as client:
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello()
            first = (await client.post("/api/runs", json={
                "graph": _graph(), "options": {"target": {"worker": "box"}}})
                ).json()["run_id"]
            second = (await client.post("/api/runs", json={
                "graph": _graph(), "options": {"target": {"worker": "box"}}})
                ).json()["run_id"]
            third = (await client.post("/api/runs", json={
                "graph": _graph(), "options": {"target": {"worker": "box"}}})
                ).json()["run_id"]
            await agent.recv()
            await agent.recv()
            row = (await client.get(f"/api/runs/{third}")).json()
            assert (row["queue_position"], row["queue_reason"]) == (1, None)
        await _terminal(service, first)
        await _terminal(service, second)
        listed = {r["id"]: r for r in
                  (await client.get("/api/runs")).json()["runs"]}
        assert listed[third]["status"] == "queued"
        assert listed[third]["queue_reason"] == (
            "waiting for worker box to reconnect")
        # It starts once the worker is back.
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello()
            assign = await agent.recv()
            assert assign.run_id == third
            await agent.finish_run(assign)
            assert (await _terminal(service, third)).status == "succeeded"


async def test_a_slow_run_does_not_starve_heartbeats(env):
    """Per-run inboxes never block the socket's read loop (#772 review)."""
    service, registry = env
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        agent = FakeAgent(ws)
        await agent.hello()
        worker = registry.get("box")
        run = registry.open_run(worker, "slowrun")  # nobody consumes it
        for seq in range(1, 2001):
            await agent.send(wp.Event(run_id="slowrun", seq=seq,
                                      event_type="node_status",
                                      payload={"node_id": "x",
                                               "status": "progress"}))
        await agent.send(wp.Heartbeat(devices=[wp.DeviceReport(
            value="cpu", kind="cpu", free_bytes=123, source="psutil")]))
        await _until(lambda: worker.devices[0].free_bytes == 123)
        assert run.inbox.qsize() == 2000
        registry.close_run("box", "slowrun")
