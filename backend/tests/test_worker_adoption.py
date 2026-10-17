"""Worker runs adopted after a control-plane restart (#793).

A control plane that dies (``kill -9``, a laptop that loses power) leaves
its worker runs executing: the run subprocesses keep going and the agent
spools their frames. The next process must not retire those rows as
``interrupted`` the way it retires its own runs: it adopts them
(``RunService.recover_interrupted``), they read ``disconnected`` while
the worker is away, and the worker's reconnect, ``run_alive`` and replay
carry them to their real outcome, in the same event log, deduplicated
against the ``seq`` the dead process persisted. A worker that does not
come back within ``CODEFYUI_WORKER_RESUME_GRACE_S`` ends the run
``interrupted``, saying so. Runs that executed in the dead process, and
every ``queued`` row, are retired exactly as before.

The "crash" here hard-cancels the first service's tasks after its agent
dropped, the way a killed process leaves its rows: nothing terminal is
written. A second service and registry then start on the same database.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from httpx_ws import aconnect_ws

from app.core import worker_protocol as wp
from app.core.db import Database
from app.core.run_service import (
    WORKER_RESUME_KEY,
    RunService,
)
from app.core.run_store import RunStore
from app.core.study_coordinator import StudyCoordinator
from app.core.study_store import StudyStore
from app.core.workers import WorkerRegistry
from app.main import app

from .test_ws_worker import FakeAgent, _client, _graph, _path, _terminal, _until
from .test_ws_worker_resume import SpoolingAgent, _drain_acks


class Generation:
    """One control-plane process: a registry and a run service."""

    def __init__(self, database: Database, *, grace: float = 300.0,
                 clock: Any = None, studies: bool = False) -> None:
        kwargs = {} if clock is None else {"clock": clock}
        self.registry = WorkerRegistry(heartbeat_s=10, resume_grace_s=grace,
                                       **kwargs)
        self.service = RunService(RunStore(database), workers=self.registry,
                                  shutdown_grace_s=0.5)
        self.coordinator = (StudyCoordinator(self.service, StudyStore(database))
                            if studies else None)
        app.state.run_service = self.service
        app.state.worker_registry = self.registry
        if self.coordinator is not None:
            app.state.study_coordinator = self.coordinator

    async def crash(self) -> None:
        """Die without writing anything terminal (its agents are gone)."""
        tasks = [active.task for active in self.service._runs.values()
                 if active.task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.coordinator is not None:
            await self.coordinator.shutdown()
        await self.registry.shutdown()

    async def close(self) -> None:
        if self.coordinator is not None:
            await self.coordinator.shutdown()
        await self.service.shutdown()
        await self.registry.shutdown()


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    app.state.db = database
    generations: list[Generation] = []

    def start(**kwargs: Any) -> Generation:
        generation = Generation(database, **kwargs)
        generations.append(generation)
        return generation

    try:
        yield start
    finally:
        for generation in reversed(generations):
            await generation.close()
        database.close()
        for attribute in ("db", "run_service", "worker_registry",
                          "study_coordinator"):
            if hasattr(app.state, attribute):
                delattr(app.state, attribute)


async def _submit(client: Any, worker: str = "box") -> str:
    response = await client.post("/api/runs", json={
        "graph": _graph(), "options": {"target": {"worker": worker}}})
    assert response.status_code in (200, 201, 202), response.text
    return response.json()["run_id"]


async def _row(client: Any, run_id: str) -> dict[str, Any]:
    return (await client.get(f"/api/runs/{run_id}")).json()


async def _start_and_crash(db: Any, tmp_path: Any, client: Any, *,
                           steps: int = 3, unacked: int = 0,
                           before_crash: Any = None,
                           **kwargs: Any) -> tuple[Generation, str,
                                                   SpoolingAgent]:
    """A run on worker ``box`` that recorded *steps* acknowledged progress
    frames (and *unacked* more), then the control plane dies."""
    first = db()
    async with aconnect_ws(_path(), client) as ws:
        agent = FakeAgent(ws)
        await agent.hello()
        run_id = await _submit(client)
        assert isinstance(await agent.recv(), wp.Assign)
        spooler = SpoolingAgent(tmp_path, run_id)
        spooler.agent = agent
        await spooler.produce(spooler.frame("started"))
        for step in range(steps):
            await spooler.produce(spooler.frame("progress", step=step))
        await _drain_acks(spooler, steps + 1)
        for step in range(steps, steps + unacked):
            await spooler.produce(spooler.frame("progress", step=step))
        if unacked:
            async def recorded() -> bool:
                events = await first.service.store.get_events(run_id)
                return sum(e.type == "node_status" for e in events) \
                    == steps + unacked
            for _ in range(250):
                if await recorded():
                    break
                await asyncio.sleep(0.02)
    spooler.agent = None
    await _until(lambda: first.registry.get("box").status == "disconnected")
    if before_crash is not None:
        await before_crash(first, run_id)
    await first.crash()
    record = await first.service.store.get_run(run_id)
    assert record.status in ("running", "disconnected")
    return first, run_id, spooler


def _steps(events: list[Any]) -> list[Any]:
    return [e.payload.get("step") for e in events
            if e.type == "node_status" and e.payload["status"] == "progress"]


async def test_a_restarted_control_plane_adopts_the_run_and_it_succeeds(
        db, tmp_path):
    async with _client() as client:
        first, run_id, spooler = await _start_and_crash(db, tmp_path, client)
        facts = (await first.service.store.get_run(run_id)).execution_facts
        assert facts[WORKER_RESUME_KEY] == {"last_seq": 4,
                                            "slots": [["box", "cpu"]]}
        # Frames keep coming on the worker while no control plane is up.
        for step in range(3, 6):
            await spooler.produce(spooler.frame("progress", step=step))

        second = db()
        assert await second.service.recover_interrupted() == 0
        row = await _row(client, run_id)
        assert row["status"] == "disconnected"
        assert row["active"] is True
        assert row["worker_status"]["state"] == "disconnected"
        assert second.service.running_on(run_id) == "box"
        assert second.service.slot_usage(("box", "cpu"))["taken"] == 1

        await spooler.produce(spooler.frame("completed"))
        final = spooler.frame("finished")
        await spooler.produce(final)
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            assert isinstance(await agent.hello(), wp.Welcome)
            await spooler.resume(agent)
            record = await _terminal(second.service, run_id)
            await _drain_acks(spooler, final.seq)
    assert record.status == "succeeded", record.error
    assert spooler.spool.complete
    events = await second.service.store.get_events(run_id)
    assert _steps(events) == list(range(6))  # one log, no gap, no repeat
    assert [e.cursor for e in events] == list(range(1, len(events) + 1))
    warnings = [e.payload for e in events if e.type == "run_warning"]
    assert [w["kind"] for w in warnings] == [
        "worker_disconnected", "worker_disconnected", "worker_reconnected"]
    assert "this server restarted" in warnings[1]["detail"]
    assert events[-1].type == "execution_complete"
    assert [e.type for e in events].count("execution_start") == 1
    assert second.service._slots == {}
    assert second.service.worker_status(run_id) is None


async def test_frames_recorded_after_the_last_ack_arrive_again_not_lost(
        db, tmp_path):
    """The persisted seq is the last ACKNOWLEDGED one: what the dead process
    recorded after it is replayed, so nothing is missing (at most a second
    of frames is seen twice)."""
    async with _client() as client:
        _first, run_id, spooler = await _start_and_crash(
            db, tmp_path, client, steps=2, unacked=2)
        second = db()
        await second.service.recover_interrupted()
        final = spooler.frame("finished")
        await spooler.produce(final)
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello()
            await spooler.resume(agent)
            record = await _terminal(second.service, run_id)
    assert record.status == "succeeded"
    assert sorted(set(_steps(await second.service.store.get_events(
        run_id)))) == [0, 1, 2, 3]


async def test_no_worker_within_the_grace_ends_the_run_saying_why(
        db, tmp_path):
    clock = [1000.0]
    async with _client() as client:
        _first, run_id, spooler = await _start_and_crash(db, tmp_path,
                                                         client)
        second = db(clock=lambda: clock[0])
        await second.service.recover_interrupted()
        assert second.service.slot_usage(("box", "cpu"))["taken"] == 1
        clock[0] += 299
        second.registry.check()
        await asyncio.sleep(0.05)
        assert second.service.is_active(run_id)
        clock[0] += 2
        second.registry.check()
        record = await _terminal(second.service, run_id)
        assert record.status == "interrupted"
        assert record.error == (
            "worker box went away: it did not reconnect with the run within "
            "300 s after this server restarted")
        assert second.service._slots == {}
        assert second.registry._adopted == {}
        events = await second.service.store.get_events(run_id)
        assert events[-1].type == "execution_stopped"
        assert events[-1].payload == {"reason": "interrupted"}

        # The worker comes back later and reports the end: a late result,
        # never a status change or an event after the terminal one.
        await spooler.produce(spooler.frame("completed"))
        final = spooler.frame("finished")
        await spooler.produce(final)
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello()
            await spooler.resume(agent)

            async def noted() -> bool:
                row = await second.service.store.get_run(run_id)
                return "late_result" in (row.execution_facts or {})
            for _ in range(250):
                if await noted():
                    break
                await asyncio.sleep(0.02)
    row = await second.service.store.get_run(run_id)
    assert row.status == "interrupted"
    assert row.execution_facts["late_result"]["status"] == "succeeded"
    assert (await second.service.store.get_events(run_id))[-1].type \
        == "execution_stopped"


async def test_a_reconnected_agent_without_the_run_interrupts_it(
        db, tmp_path):
    async with _client() as client:
        _first, run_id, _spooler = await _start_and_crash(db, tmp_path,
                                                          client)
        second = db()
        await second.service.recover_interrupted()
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello()
            # A restarted agent: its runs died with it, so it lists none.
            await agent.send(wp.RunAlive(runs=[]))
            record = await _terminal(second.service, run_id)
    assert record.status == "interrupted"
    assert "does not have it" in record.error


async def test_a_user_stop_while_adopted_reaches_the_worker_when_it_is_back(
        db, tmp_path):
    async with _client() as client:
        _first, run_id, spooler = await _start_and_crash(db, tmp_path,
                                                         client)
        second = db()
        await second.service.recover_interrupted()
        outcome = await second.service.cancel(run_id)
        assert outcome.cancelled
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello()
            await spooler.resume(agent)
            frame = await agent.recv()
            while not isinstance(frame, wp.Cancel):
                frame = await agent.recv()
            assert frame.run_id == run_id
            await spooler.produce(spooler.frame("finished",
                                                status="cancelled"))
            record = await _terminal(second.service, run_id)
    assert record.status == "cancelled"


async def test_local_and_queued_rows_are_retired_as_before(db, tmp_path):
    first = db()
    store = first.service.store
    for run_id, worker, status in (("local-run", "local", "running"),
                                   ("queued-run", "box", "queued"),
                                   ("any-run", "any", "queued"),
                                   ("legacy-run", "box", "running"),
                                   ("box-run", "box", "running")):
        await store.create_run(run_id=run_id, graph_snapshot=_graph(),
                               options={}, worker_id=worker)
        if status == "running":
            await store.mark_running(run_id, queue_key=(
                "cpu" if worker == "local" else f"{worker}/cpu"),
                worker_id=worker, devices=None if worker == "local"
                else ["cpu"], execution_facts=(
                    {WORKER_RESUME_KEY: {"last_seq": 0, "slots": [["box", "cpu"]]}} if run_id == "box-run" else {}))
    await first.close()
    second = db()
    assert await second.service.recover_interrupted() == 4
    # ``legacy-run`` recorded no resume point (a 3.1.0 row): its slots
    # cannot be known, so it is retired rather than double-booking a card.
    for run_id in ("local-run", "queued-run", "any-run", "legacy-run"):
        record = await store.get_run(run_id)
        assert record.status == "interrupted", run_id
        assert (await store.get_events(run_id))[-1].type == "execution_stopped"
    adopted = await store.get_run("box-run")
    assert adopted.status == "disconnected"
    assert second.service.active_run_ids() == ["box-run"]
    assert second.service.slot_usage(("box", "cpu"))["taken"] == 1


@pytest.mark.parametrize("grace, registry", [(0.0, True), (300.0, False)])
async def test_without_a_grace_or_registry_every_row_is_retired(
        db, tmp_path, grace, registry):
    database = app.state.db
    store = RunStore(database)
    await store.create_run(run_id="box-run", graph_snapshot=_graph(),
                           options={}, worker_id="box")
    await store.mark_running("box-run", queue_key="box/cpu",
                             worker_id="box", devices=["cpu"],
                             execution_facts={WORKER_RESUME_KEY: {"last_seq": 0, "slots": [["box", "cpu"]]}})
    if registry:
        service = db(grace=grace).service
    else:
        service = RunService(store, shutdown_grace_s=5.0)
    assert await service.recover_interrupted() == 1
    assert (await store.get_run("box-run")).status == "interrupted"
    assert service.active_run_ids() == []


async def test_a_run_whose_log_already_closed_is_not_adopted(db, tmp_path):
    """The previous process died between the terminal event and the row:
    the run finished, so it is retired, without a second closing frame."""
    first = db()
    store = first.service.store
    await store.create_run(run_id="box-run", graph_snapshot=_graph(),
                           options={}, worker_id="box")
    await store.mark_running("box-run", queue_key="box/cpu", worker_id="box",
                             devices=["cpu"],
                             execution_facts={WORKER_RESUME_KEY: {"last_seq": 0, "slots": [["box", "cpu"]]}})
    await store.append_event("box-run", "execution_complete", None)
    await first.close()
    second = db()
    assert await second.service.recover_interrupted() == 1
    assert second.service.active_run_ids() == []
    events = await store.get_events("box-run")
    assert [e.type for e in events] == ["execution_complete"]


async def test_a_row_with_no_device_is_retired(db, tmp_path):
    first = db()
    store = first.service.store
    await store.create_run(run_id="box-run", graph_snapshot=_graph(),
                           options={}, worker_id="box")
    await store.mark_running("box-run", queue_key="cpu", worker_id="box",
                             execution_facts={WORKER_RESUME_KEY: {
                                 "last_seq": "bad", "slots": "bad"}})
    await first.close()
    second = db()
    assert await second.service.recover_interrupted() == 1
    assert second.service.active_run_ids() == []


async def test_a_run_that_cannot_be_adopted_ends_interrupted(db, tmp_path,
                                                             monkeypatch):
    first = db()
    store = first.service.store
    await store.create_run(run_id="box-run", graph_snapshot=_graph(),
                           options={}, worker_id="box")
    await store.mark_running("box-run", queue_key="box/cpu", worker_id="box",
                             devices=["cpu"],
                             execution_facts={WORKER_RESUME_KEY: {"last_seq": 0, "slots": [["box", "cpu"]]}})
    await first.close()
    second = db()

    def broken(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("no")
    monkeypatch.setattr(second.registry, "adopt", broken)
    assert await second.service.recover_interrupted() == 1
    record = await store.get_run("box-run")
    assert record.status == "interrupted"
    assert "could not take the run over" in record.error
    assert (await store.get_events("box-run"))[-1].type == "execution_stopped"
    assert second.service.active_run_ids() == []
    assert second.service._slots == {}


async def test_an_enrolled_worker_holds_its_adopted_run_while_offline(
        db, tmp_path):
    """A worker known from the table (offline at boot) lists the run in the
    Workers panel; removing it ends the run at once."""
    first = db()
    store = first.service.store
    await store.create_run(run_id="box-run", graph_snapshot=_graph(),
                           options={}, worker_id="box")
    await store.mark_running(
        "box-run", queue_key="box/cuda:0", worker_id="box",
        devices=["cuda:0"], execution_facts={WORKER_RESUME_KEY: {
            "last_seq": 7, "slots": [["local", "cuda:0"]]}})
    await first.close()
    second = db()

    class Known:
        name = "box"
        platform: dict = {}
        mounts: dict = {}
        devices: list = []
        last_seen_at = None
    second.registry.load_known([Known()])
    await second.service.recover_interrupted()
    worker = second.registry.get("box")
    assert worker.describe()["running"] == ["box-run"]
    assert worker.running["box-run"].last_seq == 7
    # The slot it recorded (a same-host card's slot is this process's).
    assert second.service.slot_usage(("local", "cuda:0"))["taken"] == 1
    second.registry.forget("box")
    record = await _terminal(second.service, "box-run")
    assert record.status == "interrupted"
    assert "removed" in record.error


async def test_stopping_an_unknown_workers_adopted_run_ends_it(db, tmp_path):
    first = db()
    store = first.service.store
    await store.create_run(run_id="box-run", graph_snapshot=_graph(),
                           options={}, worker_id="box")
    await store.mark_running("box-run", queue_key="box/cpu", worker_id="box",
                             devices=["cpu"],
                             execution_facts={WORKER_RESUME_KEY: {"last_seq": 0, "slots": [["box", "cpu"]]}})
    await first.close()
    second = db()
    await second.service.recover_interrupted()
    assert second.registry.get("box") is None  # not listed
    assert second.service.worker_status("box-run")["state"] == "disconnected"
    second.registry.stopped("box", "the SSH worker was disabled")
    record = await _terminal(second.service, "box-run")
    assert record.status == "interrupted"
    assert "disabled" in record.error


# ── studies ───────────────────────────────────────────────────────────────


def _study() -> dict[str, Any]:
    return {
        "objective": {"node_id": "x", "metric": "val_loss",
                      "direction": "minimize"},
        "base_graph": _graph(),
        "variants": [{"label": "only", "ops": [
            {"node_id": "x", "param": "value", "value": 1.0}]}],
        "repetitions": 2, "seed": 11,
        "options": {"target": {"worker": "box"}},
        "budget": {"max_parallel_runs": 1},
    }


async def _study_status(coordinator: StudyCoordinator, study_id: str) -> str:
    return (await coordinator.store.get_study(study_id)).status


@pytest.mark.parametrize("outcome", ["succeeded", "expired"])
async def test_a_study_with_an_adopted_child_carries_on_with_it(
        db, tmp_path, outcome):
    """The adopted child is an ordinary active child: a read (tick) before
    it ends changes nothing and strands nothing; its success lets the
    study continue to the end, its interruption interrupts the study."""
    clock = [1000.0]
    first = db(studies=True)
    async with _client() as client:
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello()
            response = await client.post("/api/studies", json=_study())
            assert response.status_code == 201, response.text
            study_id = response.json()["study_id"]
            assign = await agent.recv()
            assert isinstance(assign, wp.Assign)
            spooler = SpoolingAgent(tmp_path, assign.run_id)
            spooler.agent = agent
            await spooler.produce(spooler.frame("started"))
            await _drain_acks(spooler, 1)
        spooler.agent = None
        await _until(lambda: first.registry.get("box").status
                     == "disconnected")
        await first.crash()

        second = db(studies=True, clock=lambda: clock[0])
        await second.service.recover_interrupted()
        assert await second.coordinator.recover() == 0
        assert await _study_status(second.coordinator, study_id) == "running"
        # A read reconciles the study first (the review's stuck case): it
        # stays running and schedules nothing past its budget of one.
        response = await client.get(f"/api/studies/{study_id}")
        assert response.status_code == 200
        assert await _study_status(second.coordinator, study_id) == "running"
        assert second.service.active_run_ids() == [assign.run_id]

        if outcome == "succeeded":
            await spooler.produce(spooler.frame("finished"))
            async with aconnect_ws(_path(), client) as ws:
                agent = FakeAgent(ws)
                await agent.hello()
                await spooler.resume(agent)
                await _terminal(second.service, assign.run_id)
                nxt = await agent.recv()
                while not isinstance(nxt, wp.Assign):
                    nxt = await agent.recv()
                assert nxt.options["seed"] == 12  # repetition 2
                await agent.finish_run(nxt)
                await _terminal(second.service, nxt.run_id)
                wanted = "succeeded"
                for _ in range(250):
                    if await _study_status(second.coordinator,
                                           study_id) == wanted:
                        break
                    await asyncio.sleep(0.02)
        else:
            clock[0] += 301
            second.registry.check()
            await _terminal(second.service, assign.run_id)
            wanted = "interrupted"
            for _ in range(250):
                if await _study_status(second.coordinator,
                                       study_id) == wanted:
                    break
                await asyncio.sleep(0.02)
    assert await _study_status(second.coordinator, study_id) == wanted
    attempts = await second.coordinator.store.list_attempts(study_id)
    statuses = sorted(a.status for a in attempts)
    if outcome == "succeeded":
        assert statuses == ["succeeded", "succeeded"]
    else:
        assert statuses == ["interrupted", "pending"]
