"""Regressions from the review of #793's adoption (PR #843).

Each test pins one fix: a re-attached run loses its "adopted" reason, a
closed run leaves an unlisted placeholder, no ``ack`` goes out unless its
seq was stored, the seq is stored alone and only with a resume grace, an
adopted run starts through the ordinary ``_start`` / ``_drive`` path, and
a graceful shutdown ends an adopted run whose worker is away at once.
Fixtures and helpers come from ``test_worker_adoption``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from httpx_ws import aconnect_ws

from app.core import worker_protocol as wp
from app.core.execution_context import ExecutionContext
from app.core.executors import _Acks
from app.core.run_service import WORKER_RESUME_KEY, _ActiveRun, _Broadcast
from app.core.scheduling import Placement

from .test_worker_adoption import _start_and_crash, db  # noqa: F401
from .test_ws_worker_resume import SpoolingAgent, _drain_acks
from .test_ws_worker import FakeAgent, _client, _graph, _path, _terminal, _until

RESUME = {WORKER_RESUME_KEY: {"last_seq": 0, "slots": [["box", "cpu"]]}}


async def _worker_row(store: Any, run_id: str = "box-run", *,
                      options: dict[str, Any] | None = None,
                      facts: dict[str, Any] | None = None) -> None:
    await store.create_run(run_id=run_id, graph_snapshot=_graph(),
                           options=options or {}, worker_id="box")
    await store.mark_running(run_id, queue_key="box/cpu", worker_id="box",
                             devices=["cpu"],
                             execution_facts=RESUME if facts is None
                             else facts)


async def test_a_reattached_run_dropped_again_gets_the_ordinary_reason(
        db, tmp_path):  # noqa: F811
    clock = [1000.0]
    async with _client() as client:
        _first, run_id, spooler = await _start_and_crash(db, tmp_path,
                                                         client)
        second = db(clock=lambda: clock[0])
        await second.service.recover_interrupted()
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello()
            await spooler.resume(agent)
            await _until(lambda: second.service.worker_status(run_id)[
                "state"] == "connected")
            assert not second.registry.run_state("box", run_id).adopted
        await _until(lambda: second.service.worker_status(run_id)[
            "state"] == "disconnected")
        clock[0] += 301
        second.registry.check()
        record = await _terminal(second.service, run_id)
    assert record.error == ("worker box went away: worker box did not "
                            "reconnect within 300 s")


async def test_closing_a_run_drops_it_from_an_unlisted_placeholder(
        db):  # noqa: F811
    registry = db().registry
    registry.adopt("box", "r1", last_seq=3)
    registry.adopt("box", "r2", last_seq=0)
    registry.close_run("box", "r1")
    assert registry.run_state("box", "r1") is None
    assert registry.claim_adopted("r1") is None
    assert list(registry._adopted["box"].running) == ["r2"]
    registry.close_run("box", "r2")
    assert registry._adopted == {}


class _Sinks:
    def __init__(self, fail: bool) -> None:
        self.fail = fail
        self.persisted: list[int] = []

    async def record_worker_seq(self, seq: int) -> None:
        if self.fail:
            raise OSError("disk full")
        self.persisted.append(seq)


class _Request:
    run_id = "r1"

    def __init__(self, sinks: _Sinks) -> None:
        self.sinks = sinks


class _Executor:
    def __init__(self) -> None:
        self.sent: list[Any] = []

    async def _send(self, _worker_id: str, frame: Any) -> bool:
        self.sent.append(frame)
        return True


@pytest.mark.parametrize("fail", [True, False])
async def test_no_ack_goes_out_when_the_seq_cannot_be_stored(fail):
    sinks, executor = _Sinks(fail), _Executor()
    acks = _Acks(executor, _Request(sinks), "box", asyncio.Queue())
    acks.delivered(5)
    await acks.flush()
    acks.cancel_timer()
    if fail:
        # The agent keeps frames 1-5: the stored seq does not cover them.
        assert (executor.sent, acks.acked) == ([], 0)
    else:
        assert [f.seq for f in executor.sent] == [5]
        assert (sinks.persisted, acks.acked) == ([5], 5)


async def test_the_seq_is_stored_alone_and_only_with_a_grace(
        db):  # noqa: F811
    gen = db()
    store = gen.service.store
    await _worker_row(store, facts={"resolved": {"device": "cpu"}, **RESUME})
    assert await store.set_worker_seq("box-run", 42)
    assert (await store.get_run("box-run")).execution_facts == {
        "resolved": {"device": "cpu"},
        WORKER_RESUME_KEY: {"last_seq": 42, "slots": [["box", "cpu"]]}}
    await _worker_row(store, "old-run", facts={})
    assert not await store.set_worker_seq("old-run", 1)
    assert (await store.get_run("old-run")).execution_facts == {}

    active = _ActiveRun(run_id="box-run", context=ExecutionContext(),
                        signal=_Broadcast(), placement=Placement(
                            "box", ("cpu",), (("box", "cpu"),)))
    assert gen.service._sinks(active).record_worker_seq is not None
    assert db(grace=0.0).service._sinks(active).record_worker_seq is None


async def test_an_adopted_run_starts_through_the_ordinary_path(
        db):  # noqa: F811
    first = db()
    store = first.service.store
    await _worker_row(store, options={"verbose": True, "backward_mode": True,
                                      "seed": 5})
    await first.close()
    second = db()
    await second.service.recover_interrupted()
    active = second.service._runs["box-run"]
    assert (active.context.verbose, active.context.backward_mode,
            active.context.seed, active.resume_seq) == (True, True, 5, 0)
    await asyncio.sleep(0.05)
    # The previous process began it: no second execution_start.
    assert [e.type for e in await store.get_events("box-run")] == [
        "run_warning"]


async def test_a_graceful_shutdown_ends_a_waiting_adopted_run_at_once(
        db):  # noqa: F811
    first = db()
    store = first.service.store
    await _worker_row(store)
    await first.close()
    second = db()
    second.service._shutdown_grace_s = 30.0
    await second.service.recover_interrupted()
    await asyncio.sleep(0.05)
    started = time.monotonic()
    await second.service.shutdown()
    assert time.monotonic() - started < 5.0
    record = await store.get_run("box-run")
    assert record.status == "interrupted"
    events = await store.get_events("box-run")
    assert events[-1].type == "execution_stopped"
    assert any("shutting down" in (e.payload or {}).get("detail", "")
               for e in events if e.type == "run_warning")
    # Terminal now, so the next start has nothing to adopt again.
    third = db()
    assert await third.service.recover_interrupted() == 0
    assert third.service.active_run_ids() == []


async def test_an_artifact_replayed_after_a_restart_is_one_row(
        db, tmp_path):  # noqa: F811
    """Seen on Meow: a checkpoint reported after the last ack was recorded
    twice once the agent replayed it to the restarted server."""
    first = db()
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
            await _drain_acks(spooler, 1)
            await spooler.produce(wp.Artifact(
                run_id=run_id, seq=spooler._next(), kind="checkpoint",
                path="/w/ckpt-e1.pt", meta={"epoch": 1}))
            for _ in range(250):
                if await first.service.store.list_artifacts(run_id):
                    break
                await asyncio.sleep(0.02)
        spooler.agent = None
        await _until(lambda: first.registry.get("box").status
                     == "disconnected")
        await first.crash()
        second = db()
        await second.service.recover_interrupted()
        await spooler.produce(wp.Artifact(
            run_id=run_id, seq=spooler._next(), kind="checkpoint",
            path="/w/ckpt-e2.pt", meta={"epoch": 2}))
        await spooler.produce(spooler.frame("finished"))
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello()
            await spooler.resume(agent)
            record = await _terminal(second.service, run_id)
    assert record.status == "succeeded"
    rows = await second.service.store.list_artifacts(run_id)
    assert sorted(r.meta["epoch"] for r in rows) == [1, 2]
