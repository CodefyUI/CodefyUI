"""Startup recovery leaves a closing frame in the run's log (#552).

``RunService.recover_interrupted`` retires every ``running`` or ``queued``
row a dead process left behind. It used to do that with a bare UPDATE, so the
run's event log simply stopped where the process died. A canvas tab that
re-attached after the restart got ``attached{status: "interrupted"}`` and
then a replay with no last frame, so nothing on the wire said the run was
over and the tab stayed on Running.

These tests pin the closing event recovery now appends -- the same
``execution_stopped{reason: "interrupted"}`` a graceful shutdown writes --
and that the row, ``GET /api/runs/{id}`` and the Runs panel's event feed
still describe the run the way they did.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from httpx_ws import aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport

from app.config import settings
from app.core.auth import TOKEN_HEADER, TOKEN_QUERY_PARAM, session_token
from app.core.db import Database
from app.core.run_service import (
    EVENT_NODE_STATUS,
    EVENT_RUN_COMPLETED,
    EVENT_RUN_FAILED,
    EVENT_RUN_STARTED,
    EVENT_RUN_STOPPED,
    STOP_REASON_INTERRUPTED,
    RunService,
)
from app.core.run_store import (
    STATUS_INTERRUPTED,
    STATUS_SUCCEEDED,
    RunProvenance,
    RunStore,
)
from app.main import app

_BASE_URL = f"http://127.0.0.1:{settings.PORT}"
_WS_PATH = f"/ws/execution?{TOKEN_QUERY_PARAM}={session_token()}"

#: Every frame here comes straight from the database; this only guards
#: against a hang if a replay never ends.
_RECV_TIMEOUT = 10.0

#: The frames that tell a follower a run is over.
_CLOSING_TYPES = {EVENT_RUN_COMPLETED, EVENT_RUN_FAILED, EVENT_RUN_STOPPED}

_GRAPH = {"nodes": [{"id": "start", "type": "Start", "data": {"params": {}}}],
          "edges": []}


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def store(db):
    return RunStore(db)


@pytest.fixture
async def restarted(db, store):
    """The process that boots over the dead one's database, on ``app.state``.

    Teardown mirrors the lifespan: drain the service, then the database
    fixture closes it.
    """
    service = RunService(store)
    app.state.db = db
    app.state.run_service = service
    try:
        yield service
    finally:
        await service.shutdown()
        for attribute in ("db", "run_service"):
            if hasattr(app.state, attribute):
                delattr(app.state, attribute)


async def _orphan(store: RunStore, *, started: bool = True) -> str:
    """A row a killed process left active, its log cut off mid-run."""
    record = await store.create_run(graph_snapshot=_GRAPH, options={},
                                    provenance=RunProvenance())
    if started:
        await store.mark_running(record.id)
        await store.append_event(record.id, EVENT_RUN_STARTED,
                                 {"run_id": record.id})
        await store.append_event(record.id, EVENT_NODE_STATUS,
                                 {"node_id": "start", "status": "running"})
    return record.id


async def _recv(ws) -> dict[str, Any]:
    return json.loads(await asyncio.wait_for(ws.receive_text(), _RECV_TIMEOUT))


# ── the store ─────────────────────────────────────────────────────────────


async def test_a_retired_run_gets_exactly_one_closing_event(store):
    run_id = await _orphan(store)

    assert await RunService(store).recover_interrupted() == 1

    events = await store.get_events(run_id)
    assert [e.type for e in events] == [
        EVENT_RUN_STARTED, EVENT_NODE_STATUS, EVENT_RUN_STOPPED]
    closing = events[-1]
    assert closing.cursor == 3
    assert closing.payload == {"reason": STOP_REASON_INTERRUPTED}
    row = await store.get_run(run_id)
    assert row is not None and row.status == STATUS_INTERRUPTED
    # One moment, one stamp: the frame says the run ended when the row does.
    assert closing.ts == row.finished_at


async def test_a_run_that_never_left_the_queue_gets_one_too(store):
    """No ``execution_start`` to balance it, exactly like a retired queue.

    A follower needs a frame that says the run ended; inventing a start
    event to pair with it would be the actual lie.
    """
    run_id = await _orphan(store, started=False)

    assert await RunService(store).recover_interrupted() == 1

    events = await store.get_events(run_id)
    assert [(e.cursor, e.type, e.payload) for e in events] == [
        (1, EVENT_RUN_STOPPED, {"reason": STOP_REASON_INTERRUPTED})]


async def test_every_retired_run_gets_its_own(store):
    running = await _orphan(store)
    queued = await _orphan(store, started=False)

    assert await RunService(store).recover_interrupted() == 2

    for run_id in (running, queued):
        events = await store.get_events(run_id)
        assert {e.run_id for e in events} == {run_id}
        assert [e.type for e in events].count(EVENT_RUN_STOPPED) == 1
        assert events[-1].type == EVENT_RUN_STOPPED


async def test_a_second_boot_adds_no_second_closing_event(store):
    run_id = await _orphan(store)
    assert await RunService(store).recover_interrupted() == 1

    # Another restart over the same database.
    assert await RunService(store).recover_interrupted() == 0

    events = await store.get_events(run_id)
    assert sum(e.type in _CLOSING_TYPES for e in events) == 1


async def test_a_log_that_already_closed_is_not_closed_again(store):
    """The process died between its terminal event and its row write.

    ``_finalize`` makes the terminal event durable before ``mark_finished``,
    so a kill in between leaves an active row whose log is already closed.
    Recovery still retires the row; a second closing frame would make a
    replay end twice.
    """
    run_id = await _orphan(store)
    await store.append_event(run_id, EVENT_RUN_COMPLETED, None)

    assert await RunService(store).recover_interrupted() == 1

    events = await store.get_events(run_id)
    assert events[-1].type == EVENT_RUN_COMPLETED
    assert sum(e.type in _CLOSING_TYPES for e in events) == 1
    row = await store.get_run(run_id)
    assert row is not None and row.status == STATUS_INTERRUPTED


async def test_finished_runs_are_left_alone(store):
    run_id = await _orphan(store)
    await store.append_event(run_id, EVENT_RUN_COMPLETED, None)
    await store.mark_finished(run_id, STATUS_SUCCEEDED)
    before = await store.get_events(run_id)

    assert await RunService(store).recover_interrupted() == 0

    assert await store.get_events(run_id) == before
    row = await store.get_run(run_id)
    assert row is not None and row.status == STATUS_SUCCEEDED


# ── the wire ──────────────────────────────────────────────────────────────


async def test_a_reconnecting_tab_is_told_the_run_is_over(restarted, store):
    """The canvas re-attaches from the last cursor it rendered.

    Before #552 the acknowledgement was the last thing it ever received.
    """
    run_id = await _orphan(store)
    assert await restarted.recover_interrupted() == 1

    async with AsyncClient(transport=ASGIWebSocketTransport(app=app),
                           base_url=_BASE_URL) as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            await ws.send_text(json.dumps({"action": "attach",
                                           "run_id": run_id, "cursor": 2}))
            ack = await _recv(ws)
            closing = await _recv(ws)

    assert ack == {"type": "attached", "run_id": run_id, "cursor": 2,
                   "status": STATUS_INTERRUPTED}
    assert closing == {"type": EVENT_RUN_STOPPED,
                       "reason": STOP_REASON_INTERRUPTED,
                       "run_id": run_id, "cursor": 3}


async def test_a_replay_from_the_start_ends_with_it(restarted, store):
    run_id = await _orphan(store)
    assert await restarted.recover_interrupted() == 1

    frames: list[dict[str, Any]] = []
    async with AsyncClient(transport=ASGIWebSocketTransport(app=app),
                           base_url=_BASE_URL) as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            await ws.send_text(json.dumps({"action": "attach",
                                           "run_id": run_id}))
            assert (await _recv(ws))["type"] == "attached"
            while not frames or frames[-1]["type"] not in _CLOSING_TYPES:
                frames.append(await _recv(ws))

    assert [f["type"] for f in frames] == [
        EVENT_RUN_STARTED, EVENT_NODE_STATUS, EVENT_RUN_STOPPED]
    assert [f["cursor"] for f in frames] == [1, 2, 3]


# ── REST: the row and the Runs panel ──────────────────────────────────────


async def test_the_runs_api_reports_the_row_and_its_closing_event(
    restarted, store,
):
    run_id = await _orphan(store)
    assert await restarted.recover_interrupted() == 1

    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url=_BASE_URL,
                           headers={TOKEN_HEADER: session_token()}) as client:
        one = (await client.get(f"/api/runs/{run_id}")).json()
        listed = (await client.get("/api/runs")).json()
        feed = (await client.get(f"/api/runs/{run_id}/events")).json()

    assert one["status"] == STATUS_INTERRUPTED
    assert one["active"] is False
    assert one["finished_at"] is not None
    # Where a follower resumes: just past the closing frame.
    assert one["last_cursor"] == 3
    assert [r["status"] for r in listed["runs"] if r["id"] == run_id] == [
        STATUS_INTERRUPTED]
    assert feed["status"] == STATUS_INTERRUPTED
    assert [e["type"] for e in feed["events"]][-1] == EVENT_RUN_STOPPED
    assert feed["events"][-1]["payload"] == {"reason": STOP_REASON_INTERRUPTED}
    assert feed["cursor"] == 3
