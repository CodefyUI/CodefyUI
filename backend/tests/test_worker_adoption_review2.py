"""Regressions from the second review of #793's adoption (PR #843).

Auto-resume and an interrupted cancel of a study whose child was adopted,
a Stop that survives a crash, the artifact replay check limited to rows
the previous process wrote, a graceful shutdown that cannot stop a
connected worker run, the seq stored only together with the buffered
metrics, a placeholder's runs found after the name became known, and no
write when nothing moved.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from httpx_ws import aconnect_ws

from app.core import worker_protocol as wp
from app.core.execution_context import ArtifactSignal
from app.core.run_service import WORKER_RESUME_KEY
from app.core.run_store import MetricPoint

from .test_worker_adoption import (  # noqa: F401
    _start_and_crash,
    _study,
    _study_status,
    db,
)
from .test_worker_adoption_review import _worker_row
from .test_ws_worker import FakeAgent, _client, _graph, _path, _terminal, _until
from .test_ws_worker_resume import SpoolingAgent, _drain_acks


async def _study_on_box(db: Any, tmp_path: Any, client: Any,  # noqa: F811
                        **extra: Any) -> tuple[Any, str, Any, Any]:
    """A study whose first child runs on ``box``; the agent then drops."""
    first = db(studies=True)
    async with aconnect_ws(_path(), client) as ws:
        agent = FakeAgent(ws)
        await agent.hello()
        response = await client.post("/api/studies",
                                     json={**_study(), **extra})
        assert response.status_code == 201, response.text
        study_id = response.json()["study_id"]
        assign = await agent.recv()
        assert isinstance(assign, wp.Assign)
        spooler = SpoolingAgent(tmp_path, assign.run_id)
        spooler.agent = agent
        await spooler.produce(spooler.frame("started"))
        await _drain_acks(spooler, 1)
    spooler.agent = None
    await _until(lambda: first.registry.get("box").status == "disconnected")
    return first, study_id, assign, spooler


async def _events(coordinator: Any, study_id: str) -> list[str]:
    return [e.type for e in await coordinator.store.get_events(study_id)]


async def test_an_auto_study_resumes_when_its_adopted_child_expires(
        db, tmp_path):  # noqa: F811
    clock = [1000.0]
    async with _client() as client:
        first, study_id, assign, _sp = await _study_on_box(
            db, tmp_path, client, resume_policy="auto")
        await first.crash()
        second = db(studies=True, clock=lambda: clock[0])
        await second.service.recover_interrupted()
        await second.coordinator.recover()
        assert "study_resumed" not in await _events(second.coordinator,
                                                    study_id)
        clock[0] += 301
        second.registry.check()
        await _terminal(second.service, assign.run_id)
        for _ in range(250):
            if "study_resumed" in await _events(second.coordinator,
                                                study_id):
                break
            await asyncio.sleep(0.02)
    events = await _events(second.coordinator, study_id)
    assert "study_interrupted" in events and "study_resumed" in events
    assert second.coordinator._restarted == set()


async def test_a_cancel_in_progress_is_sent_again_to_an_adopted_child(
        db, tmp_path):  # noqa: F811
    async with _client() as client:
        first, study_id, assign, spooler = await _study_on_box(db, tmp_path,
                                                               client)
        # The cancel was asked, but the child's stop never left (the
        # worker was away) before the server died.
        await first.coordinator.store.transition(
            study_id, to="cancel_requested", expected=("running",),
            event_type="study_cancel_requested")
        await first.crash()
        second = db(studies=True)
        await second.service.recover_interrupted()
        await second.coordinator.recover()
        assert await _study_status(second.coordinator, study_id) \
            == "cancel_requested"
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello()
            await spooler.resume(agent)
            frame = await agent.recv()
            while not isinstance(frame, wp.Cancel):
                frame = await agent.recv()
            assert frame.run_id == assign.run_id
            await spooler.produce(spooler.frame("finished",
                                                status="cancelled"))
            record = await _terminal(second.service, assign.run_id)
        for _ in range(250):
            if await _study_status(second.coordinator, study_id) \
                    == "cancelled":
                break
            await asyncio.sleep(0.02)
    assert record.status == "cancelled"
    assert await _study_status(second.coordinator, study_id) == "cancelled"


async def test_a_stop_asked_before_a_crash_is_asked_again_after_it(
        db, tmp_path):  # noqa: F811
    async def stop(first: Any, run_id: str) -> None:
        outcome = await first.service.cancel(run_id)
        assert outcome.cancelled
        facts = (await first.service.store.get_run(run_id)).execution_facts
        assert facts["stop_requested"] == "cancelled"

    async with _client() as client:
        _first, run_id, spooler = await _start_and_crash(
            db, tmp_path, client, before_crash=stop)
        second = db()
        await second.service.recover_interrupted()
        assert second.service._runs[run_id].context.should_stop()
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello()
            await spooler.resume(agent)
            frame = await agent.recv()
            while not isinstance(frame, wp.Cancel):
                frame = await agent.recv()
            # Even a worker that finished anyway does not file a success
            # over the user's Stop.
            await spooler.produce(spooler.frame("finished",
                                                status="interrupted"))
            record = await _terminal(second.service, run_id)
    assert record.status == "cancelled"


async def test_only_a_replay_of_a_previous_row_is_skipped(db):  # noqa: F811
    first = db()
    store = first.service.store
    await _worker_row(store)
    await store.add_artifact("box-run", "checkpoint", "/w/latest.pt",
                             meta={"epoch": 1}, worker_id="box")
    await first.close()
    second = db()
    await second.service.recover_interrupted()
    active = second.service._runs["box-run"]
    record = second.service._record_artifact
    # The replay of what the previous process wrote: skipped.
    await record(active, ArtifactSignal(kind="checkpoint",
                                        path="/w/latest.pt",
                                        meta={"epoch": 1}))
    # The same file rewritten (other meta), then twice with equal meta
    # after the adoption: both are new writes and recorded.
    await record(active, ArtifactSignal(kind="checkpoint",
                                        path="/w/latest.pt",
                                        meta={"epoch": 2}))
    await record(active, ArtifactSignal(kind="plot", path="/w/loss.png"))
    await record(active, ArtifactSignal(kind="plot", path="/w/loss.png"))
    rows = await store.list_artifacts("box-run")
    assert [(r.kind, r.meta) for r in rows] == [
        ("checkpoint", {"epoch": 1}), ("checkpoint", {"epoch": 2}),
        ("plot", None), ("plot", None)]


async def test_a_worker_run_that_outlives_the_shutdown_grace_is_retired(
        db, tmp_path):  # noqa: F811
    gen = db()
    gen.service._shutdown_grace_s = 0.3
    async with _client() as client:
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello()
            run_id = (await client.post("/api/runs", json={
                "graph": _graph(), "options": {"target": {"worker": "box"}}})
            ).json()["run_id"]
            assert isinstance(await agent.recv(), wp.Assign)
            await agent.send(wp.RunStarted(run_id=run_id, seq=1, pid=1))
            await _until(lambda: gen.service.worker_status(run_id)
                         is not None)
            # The agent ignores the stop: still running after the grace.
            await gen.service.shutdown()
    record = await gen.service.store.get_run(run_id)
    assert record.status == "interrupted"
    assert "shut down before the run on worker box stopped" in record.error
    events = await gen.service.store.get_events(run_id)
    assert events[-1].type == "execution_stopped"
    nxt = db()
    assert await nxt.service.recover_interrupted() == 0
    assert nxt.service.active_run_ids() == []


async def test_the_seq_is_stored_only_with_the_points_and_never_alone(
        db, monkeypatch):  # noqa: F811
    first = db()
    store = first.service.store
    await _worker_row(store)
    await first.close()
    second = db()
    await second.service.recover_interrupted()
    service = second.service
    store = service.store
    active = service._runs["box-run"]
    point = MetricPoint(name="loss", value=1.0, step=1, node_id="x")
    active.pending_metrics.append(point)
    real = store.checkpoint_worker_run

    async def broken(*_args: Any, **_kwargs: Any) -> bool:
        raise OSError("disk full")
    monkeypatch.setattr(store, "checkpoint_worker_run", broken)
    with pytest.raises(OSError):
        await service._record_worker_seq(active, 5)
    assert active.pending_metrics == [point]  # kept for the next try
    assert active.stored_seq == 0
    monkeypatch.setattr(store, "checkpoint_worker_run", real)
    await service._record_worker_seq(active, 5)
    assert active.pending_metrics == [] and active.stored_seq == 5
    facts = (await store.get_run("box-run")).execution_facts
    assert facts[WORKER_RESUME_KEY]["last_seq"] == 5
    assert [m.step for m in await store.get_metrics("box-run")] == [1]

    # Nothing moved: no write at all; no points: no metric event.
    calls: list[Any] = []

    async def counting(*args: Any, **kwargs: Any) -> bool:
        calls.append(args)
        return await real(*args, **kwargs)
    monkeypatch.setattr(store, "checkpoint_worker_run", counting)
    await service._record_worker_seq(active, 5)
    assert calls == []
    before = [e.type for e in await store.get_events("box-run")]
    await service._record_worker_seq(active, 6)
    assert len(calls) == 1
    assert [e.type for e in await store.get_events("box-run")] == before


async def test_a_placeholders_runs_stay_visible_once_the_name_is_known(
        db):  # noqa: F811
    registry = db().registry
    registry.adopt("box", "r1", last_seq=2)

    class Known:
        name = "box"
        platform: dict = {}
        mounts: dict = {}
        devices: list = []
        last_seen_at = None
    registry.load_known([Known()])  # enrolled during the grace
    assert registry.get("box") is not None
    assert registry.run_state("box", "r1").last_seq == 2
    run = registry.run_state("box", "r1")
    assert registry.end_waiting("box", "r1", "this server is shutting down")
    assert run.gone
