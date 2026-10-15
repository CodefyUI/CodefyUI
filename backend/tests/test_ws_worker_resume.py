"""A worker that drops off the network and comes back (#783).

A fake agent with a real spool (``app.agent.spool.Spool``) talks to the
real ``/ws/worker`` endpoint: it closes its socket mid-run, keeps
producing frames into the spool, reconnects, says ``run_alive`` and
replays what the control plane has not acknowledged. Within the resume
grace the run carries on to a complete, gap-free event log and a
``succeeded`` row; past it the run is ``interrupted`` and the worker's
eventual ``run_finished`` is recorded as a late result in
``execution_facts`` (never as an event after the terminal one). A stop
that waited for the worker cannot reach it before ``welcome``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from httpx_ws import aconnect_ws

from app.agent.spool import Spool
from app.core import worker_protocol as wp

from . import test_ws_worker
from .test_ws_worker import (
    FakeAgent,
    _client,
    _graph,
    _path,
    _terminal,
    _until,
)

#: The database, run service and registry behind the app (a fixture).
env = test_ws_worker.env


class SpoolingAgent:
    """One run's frames, spooled before they are sent, as the agent does."""

    def __init__(self, tmp_path: Any, run_id: str) -> None:
        self.run_id = run_id
        self.spool = Spool(tmp_path / run_id, run_id)
        self.seq = 0
        self.agent: FakeAgent | None = None
        #: Control frames other than ``ack`` read while waiting for acks.
        self.received: list[Any] = []

    def _next(self) -> int:
        self.seq += 1
        return self.seq

    def frame(self, kind: str, **fields: Any) -> Any:
        seq = self._next()
        if kind == "started":
            return wp.RunStarted(run_id=self.run_id, seq=seq, pid=1)
        if kind == "progress":
            return wp.Event(run_id=self.run_id, seq=seq,
                            event_type="node_status",
                            payload={"node_id": "x", "status": "progress",
                                     "step": fields["step"]})
        if kind == "completed":
            return wp.Event(run_id=self.run_id, seq=seq,
                            event_type="node_status",
                            payload={"node_id": "x", "status": "completed"})
        return wp.RunFinished(run_id=self.run_id, seq=seq,
                              status=fields.get("status", "succeeded"))

    async def produce(self, frame: Any) -> None:
        """Spool *frame*, then send it if connected."""
        self.spool.append(frame.seq, wp.encode_frame(frame),
                          final=isinstance(frame, wp.RunFinished))
        if self.agent is not None:
            await self.agent.send(frame)

    def take_acks(self) -> None:
        if self.agent is None:
            return
        for ack in self.agent.acks:
            if ack.run_id == self.run_id:
                self.spool.ack(ack.seq)
        self.agent.acks.clear()

    async def resume(self, agent: FakeAgent) -> None:
        """``run_alive``, then every unacknowledged frame, in order."""
        self.agent = agent
        await agent.send(wp.RunAlive(runs=[wp.RunAliveEntry(
            run_id=self.run_id, last_seq=self.seq)]))
        for _seq, line in self.spool.read_after(self.spool.acked, 10**6):
            await agent.ws.send_text(line)


async def _drain_acks(spooler: SpoolingAgent, until_seq: int) -> None:
    """Read control frames until an ack covering *until_seq* arrived."""
    agent = spooler.agent
    while not any(a.seq >= until_seq for a in agent.acks):
        frame = await agent.recv(acks=True)
        if not isinstance(frame, wp.Ack):
            spooler.received.append(frame)
    spooler.take_acks()


async def _status(client: Any, run_id: str) -> dict[str, Any]:
    return (await client.get(f"/api/runs/{run_id}")).json()


async def test_a_drop_within_the_grace_resumes_to_a_gap_free_log(env,
                                                                  tmp_path):
    service, registry = env
    async with _client() as client:
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello()
            run_id = (await client.post("/api/runs", json={
                "graph": _graph(), "options": {"target": {"worker": "box"}}})
                ).json()["run_id"]
            assert isinstance(await agent.recv(), wp.Assign)
            spooler = SpoolingAgent(tmp_path, run_id)
            spooler.agent = agent
            await spooler.produce(spooler.frame("started"))
            for step in range(3):
                await spooler.produce(spooler.frame("progress", step=step))
            # The control plane acks within a second; part of the stream
            # is then acknowledged and part is not.
            await _drain_acks(spooler, 4)
            await spooler.produce(spooler.frame("progress", step=3))
        # The socket is gone. The run waits; its frames keep coming.
        spooler.agent = None
        await _until(lambda: registry.get("box").status == "disconnected")
        for _ in range(250):
            row = await _status(client, run_id)
            if row["status"] == "disconnected":
                break
            await asyncio.sleep(0.02)
        assert row["status"] == "disconnected"
        assert row["active"] is True
        status = row["worker_status"]
        assert (status["worker_id"], status["state"]) == ("box",
                                                         "disconnected")
        assert status["resume_deadline"] - status["disconnected_at"] == 300
        listed = {r["id"]: r for r in
                  (await client.get("/api/runs")).json()["runs"]}
        assert listed[run_id]["worker_status"]["state"] == "disconnected"
        assert service.slot_usage(("box", "cpu"))["taken"] == 1
        for step in range(4, 7):
            await spooler.produce(spooler.frame("progress", step=step))
        await spooler.produce(spooler.frame("completed"))
        final = spooler.frame("finished")
        await spooler.produce(final)

        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            assert isinstance(await agent.hello(), wp.Welcome)
            await spooler.resume(agent)
            record = await _terminal(service, run_id)
            await _drain_acks(spooler, final.seq)
    assert record.status == "succeeded", record.error
    assert spooler.spool.complete
    events = await service.store.get_events(run_id)
    steps = [e.payload.get("step") for e in events
             if e.type == "node_status" and e.payload["status"] == "progress"]
    assert steps == list(range(7))  # complete, in order, no duplicate
    kinds = [e.payload["kind"] for e in events if e.type == "run_warning"]
    assert kinds == ["worker_disconnected", "worker_reconnected"]
    assert [e.cursor for e in events] == list(range(1, len(events) + 1))
    assert events[-1].type == "execution_complete"
    assert service.worker_status(run_id) is None
    assert service._slots == {}


async def test_past_the_grace_the_run_is_interrupted_and_a_late_result_noted(
        env, tmp_path):
    service, registry = env
    clock = [1000.0]
    registry._clock = lambda: clock[0]
    async with _client() as client:
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello()
            run_id = (await client.post("/api/runs", json={
                "graph": _graph(), "options": {"target": {"worker": "box"}}})
                ).json()["run_id"]
            await agent.recv()
            spooler = SpoolingAgent(tmp_path, run_id)
            spooler.agent = agent
            await spooler.produce(spooler.frame("started"))
        spooler.agent = None
        await _until(lambda: (service.worker_status(run_id) or {}).get(
            "state") == "disconnected")
        clock[0] += 301
        registry.check()
        record = await _terminal(service, run_id)
        assert record.status == "interrupted"
        assert "did not reconnect within 300 s" in record.error
        assert service._slots == {}
        await spooler.produce(spooler.frame("completed"))
        final = spooler.frame("finished")
        await spooler.produce(final)

        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello()
            await spooler.resume(agent)
            await _drain_acks(spooler, final.seq)

            for _ in range(200):
                row = await _status(client, run_id)
                if row["late_result"] is not None:
                    break
                await asyncio.sleep(0.02)
    assert spooler.spool.complete  # acknowledged, so the agent lets go
    # Its slots went back when it was interrupted, so the worker was told
    # to stop it at once (architecture review of #772).
    cancels = [f for f in spooler.received if isinstance(f, wp.Cancel)]
    assert [(c.run_id, c.grace_s) for c in cancels] == [(run_id, 0)]
    record = await service.store.get_run(run_id)
    assert record.status == "interrupted"  # not overwritten
    late = row["late_result"]
    assert (late["status"], late["error"], late["worker_id"]) == (
        "succeeded", None, "box")
    assert late["at"]
    assert record.execution_facts["late_result"] == late
    events = await service.store.get_events(run_id)
    # Nothing after the terminal event: neither the late frames nor a note.
    assert events[-1].type == "execution_stopped"
    assert not any(e.type == "node_status" for e in events)


async def test_a_stop_waiting_for_the_worker_never_precedes_welcome(env):
    """Review of #783: once a reconnecting worker is registered, executors
    send through its socket; ``welcome`` must already be out."""
    service, registry = env
    async with _client() as client:
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello()
            run_id = (await client.post("/api/runs", json={
                "graph": _graph(), "options": {"target": {"worker": "box"}}})
                ).json()["run_id"]
            await agent.recv()
            await agent.send(wp.RunStarted(run_id=run_id, seq=1))
        for _ in range(250):
            if (await _status(client, run_id))["status"] == "disconnected":
                break
            await asyncio.sleep(0.02)
        assert (await client.post(f"/api/runs/{run_id}/cancel")
                ).status_code == 200
        await asyncio.sleep(0.5)  # the stop waits for the worker
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.send(wp.Hello(name="box", version="t", devices=[
                wp.DeviceReport(value="cpu", kind="cpu")]))
            first = await agent.recv(acks=True)
            assert isinstance(first, wp.Welcome)
            await agent.send(wp.RunAlive(runs=[
                wp.RunAliveEntry(run_id=run_id, last_seq=1)]))
            cancel = await agent.recv(acks=False)
            assert isinstance(cancel, wp.Cancel) and cancel.run_id == run_id
            await agent.send(wp.RunFinished(run_id=run_id, seq=2,
                                            status="cancelled"))
            record = await _terminal(service, run_id)
    assert record.status == "cancelled"
