"""One-shot workers, control plane side (#790).

``cdui agent start --one-shot`` says ``hello.one_shot``: ``any`` never
places a run there, a run naming it or ``target.pool: "one-shot"`` does,
and it takes ONE run. A run waiting for one fails after
``CODEFYUI_WORKER_ONE_SHOT_QUEUE_TIMEOUT_S``. The agent side is
``tests/agent/test_one_shot.py``; a real agent, ``test_one_shot_e2e.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from httpx_ws import aconnect_ws

from app.core import worker_protocol as wp
from app.core.run_service import (
    ONE_SHOT_QUEUE_PREFIX,
    RunSubmitError,
    normalize_target,
    one_shot_timed_out,
)
from app.core.scheduling import Placement
from app.core.version import get_version
from app.core.worker_handshake import one_shot_refusal
from app.core.workers import STATUS_OFFLINE, WorkerRegistry

from . import test_ws_worker
from .test_worker_scheduler import _Entry, _scheduler, _target, _worker
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

POOL = {"worker": "any", "pool": "one-shot"}


def _one_shot(name: str, *devices: str, run_id: str | None = None,
              claim: str | None = None, **kw: Any) -> Any:
    worker = _worker(name, *devices, **kw)
    worker.one_shot = True
    worker.one_shot_run_id = run_id
    worker.one_shot_claim = claim
    worker.instance_id = "i-1"
    return worker


def _pool(device: str = "cpu") -> dict[str, Any]:
    return {"device": device, "target": {**POOL, "devices": None}}


# ── target.pool ──────────────────────────────────────────────────────────


def test_a_target_without_a_pool_normalises_as_before():
    assert normalize_target({"worker": "box"}) == {"worker": "box",
                                                   "devices": None}


@pytest.mark.parametrize("raw", [{"pool": "one-shot"},
                                 {"pool": " One-Shot ", "worker": "any"}])
def test_the_one_shot_pool_targets_any(raw):
    assert normalize_target(raw) == {"worker": "any", "devices": None,
                                     "pool": "one-shot"}


@pytest.mark.parametrize("raw, words", [
    ({"pool": "gpu"}, "target.pool must be 'one-shot'"),
    ({"pool": 3}, "target.pool must be 'one-shot'"),
    ({"pool": "one-shot", "worker": "box"}, "target.pool picks the worker"),
    ({"pool": "one-shot", "worker": "local"}, "target.pool picks the worker"),
    ({"poll": "one-shot"}, "unknown target key"),
])
def test_a_bad_pool_is_refused(raw, words):
    with pytest.raises(RunSubmitError, match=words):
        normalize_target(raw)


def test_the_timeout_message_names_the_span_and_the_setting():
    assert "within 24 h" in one_shot_timed_out(86400)
    assert "within 5 min" in one_shot_timed_out(300)
    assert "within 1.5 s" in one_shot_timed_out(1.5)
    assert "CODEFYUI_WORKER_ONE_SHOT_QUEUE_TIMEOUT_S" in one_shot_timed_out(1)


# ── placement ────────────────────────────────────────────────────────────


def test_any_never_places_a_run_on_a_one_shot_worker():
    scheduler = _scheduler(_one_shot("slurm", "cpu", "cuda:0"), here=("cpu",))
    entry = _Entry("r1", "any/cuda", _target("any", "cuda"))
    assert scheduler.place(entry, {}) is None
    # ... and does not count it as able to serve the device either, so a
    # submit is refused rather than left waiting forever.
    assert not scheduler.can_serve_any("cuda", None)
    assert scheduler.place(_Entry("r2", "any/cpu", _target("any")), {}) \
        == Placement("local", ("cpu",), (("local", "cpu"),))


def test_the_pool_places_on_a_one_shot_worker_only():
    regular = _worker("box", "cpu", "cuda:0")
    one = _one_shot("slurm", "cpu", "cuda:0")
    scheduler = _scheduler(regular, one, here=("cpu", "cuda:0"))
    entry = _Entry("r1", f"{ONE_SHOT_QUEUE_PREFIX}/cuda", _pool("cuda"))
    assert scheduler.place(entry, {}) == Placement(
        "slurm", ("cuda:0",), (("slurm", "cuda:0"),))
    # Its card busy: nowhere else, not here and not the regular worker.
    assert scheduler.place(entry, {("slurm", "cuda:0"): 1}) is None
    assert "waiting for a one-shot worker that offers cuda" in \
        scheduler.explain(entry)
    coded = scheduler.explain_reason(entry)
    assert (coded.code, coded.params_dict()) == ("one_shot_pool",
                                                 {"devices": "cuda"})


def test_a_one_shot_worker_takes_one_run_and_only_its_run_id():
    claimed = _one_shot("slurm", "cpu", claim="r0")
    scheduler = _scheduler(claimed)
    entry = _Entry("r1", "slurm/cpu", _target("slurm"))
    assert scheduler.place(entry, {}) is None
    assert scheduler.place(_Entry("r1", "p/cpu", _pool()), {}) is None
    assert scheduler.explain(entry) == (
        "one-shot worker slurm already took run r0; start another one-shot "
        "agent")
    coded = scheduler.explain_reason(entry)
    assert (coded.code, coded.params_dict()) == (
        "one_shot_taken", {"worker": "slurm", "run": "r0"})
    pinned = _one_shot("slurm", "cpu", run_id="r7")
    scheduler = _scheduler(pinned)
    assert scheduler.place(_Entry("r1", "p/cpu", _pool()), {}) is None
    assert scheduler.explain(entry) == (
        "one-shot worker slurm takes only run r7 (--run-id)")
    coded = scheduler.explain_reason(entry)
    assert (coded.code, coded.params_dict()) == (
        "one_shot_run_id", {"worker": "slurm", "run": "r7"})
    assert scheduler.place(_Entry("r7", "p/cpu", _pool()), {}) == Placement(
        "slurm", ("cpu",), (("slurm", "cpu"),))
    # No reservation for a set run it will not take.
    pinned.devices = [wp.DeviceReport(value=d, kind="cuda")
                      for d in ("cuda:0", "cuda:1")]
    sets = _Entry("r1", "slurm/cuda:0", _target("slurm", "cuda:0",
                                                ["cuda:0", "cuda:1"]))
    assert scheduler.reservation(sets) is None


def test_an_away_one_shot_worker_is_waited_for_with_its_own_words():
    away = _one_shot("slurm", "cpu", status="offline")
    entry = _Entry("r1", "slurm/cpu", _target("slurm"))
    assert _scheduler(away).explain(entry) == (
        "waiting for one-shot worker slurm to start (cdui agent start "
        "--one-shot)")
    coded = _scheduler(away).explain_reason(entry)
    assert (coded.code, coded.params_dict()) == ("one_shot_start",
                                                 {"worker": "slurm"})


# ── the registry ─────────────────────────────────────────────────────────


async def _nothing(*_: Any) -> None:
    return None


def _hello(name: str = "slurm", **kw: Any) -> wp.Hello:
    return wp.Hello(name=name, version="t", one_shot=True,
                    devices=[wp.DeviceReport(value="cpu", kind="cpu")], **kw)


async def test_the_registry_claims_carries_and_releases_the_run():
    registry = WorkerRegistry()
    try:
        worker, _ = registry.connect(_hello(instance_id="a"), send=_nothing,
                                     close=_nothing)
        assert worker.one_shot and worker.one_shot_available
        registry.claim_one_shot("slurm", "r1")
        registry.claim_one_shot("slurm", "r2")  # first one wins
        assert worker.one_shot_claim == "r1" and not worker.one_shot_available
        # The same agent reconnecting keeps it; another agent starts fresh.
        again, _ = registry.connect(_hello(instance_id="a"), send=_nothing,
                                    close=_nothing)
        assert again.one_shot_claim == "r1"
        registry.release_one_shot("slurm", "other")
        assert again.one_shot_claim == "r1"
        registry.release_one_shot("slurm", "r1")
        assert again.one_shot_claim is None
        registry.claim_one_shot("slurm", "r3")
        fresh, _ = registry.connect(_hello(instance_id="b"), send=_nothing,
                                    close=_nothing)
        assert fresh.one_shot_claim is None
        assert fresh.describe()["one_shot"] == {"run_id": None, "took": None}
        # A regular worker is never claimed, and says so.
        regular, _ = registry.connect(wp.Hello(name="box", version="t"),
                                      send=_nothing, close=_nothing)
        registry.claim_one_shot("box", "r9")
        assert regular.one_shot_claim is None
        assert regular.describe()["one_shot"] is None
        registry.release_one_shot("nobody", "r9")
        registry.claim_one_shot("nobody", "r9")
    finally:
        await registry.shutdown()


async def test_a_one_shot_worker_that_leaves_with_nothing_running_is_offline():
    registry = WorkerRegistry()
    try:
        worker, _ = registry.connect(_hello(one_shot_run_id="r1"),
                                     send=_nothing, close=_nothing)
        registry.claim_one_shot("slurm", "r1")
        registry.disconnect(worker)
        assert worker.status == STATUS_OFFLINE
        described = worker.describe()
        assert described["status"] == "offline"
        assert described["one_shot"] == {"run_id": "r1", "took": "r1"}
        # Unlike an enrolled worker that never connected, it says when.
        assert described["connected_at"] is not None
        registry.disconnect(worker)  # idempotent
        # With a run in flight it is disconnected, and the run waits.
        busy, _ = registry.connect(_hello(name="other"), send=_nothing,
                                   close=_nothing)
        run = registry.open_run(busy, "r2")
        registry.disconnect(busy)
        assert busy.status == "disconnected" and run.detached_at is not None
    finally:
        await registry.shutdown()


def test_another_agent_may_not_take_a_busy_one_shot_name():
    online = _one_shot("slurm", "cpu")
    assert one_shot_refusal(_hello(instance_id="i-1"), online) is None
    refusal = one_shot_refusal(_hello(instance_id="i-2"), online)
    assert "is busy: another agent holds the name (connected)" in refusal
    assert one_shot_refusal(_hello(), online) is not None  # no instance id
    # A regular agent may not take it either, nor a one-shot agent a
    # connected regular worker's name.
    assert one_shot_refusal(wp.Hello(name="slurm", version="t"),
                            online) is not None
    regular = _worker("slurm", "cpu")
    assert one_shot_refusal(_hello(instance_id="i-1"), regular) is not None
    assert one_shot_refusal(wp.Hello(name="slurm", version="t"),
                            regular) is None
    # Gone, with nothing in flight: a new job may take the name.
    gone = _one_shot("slurm", "cpu", status="offline")
    assert one_shot_refusal(_hello(instance_id="i-2"), gone) is None
    gone.running = {"r1": object()}
    assert "its run is in flight" in one_shot_refusal(
        _hello(instance_id="i-2"), gone)


# ── over /ws/worker ──────────────────────────────────────────────────────


async def _submit(client: Any, target: dict[str, Any],
                  device: str = "cpu") -> str:
    response = await client.post("/api/runs", json={
        "graph": _graph(), "options": {"device": device, "target": target}})
    assert response.status_code == 200, response.text
    return response.json()["run_id"]


async def test_a_pool_run_waits_for_a_one_shot_worker_which_takes_only_it(env):
    service, registry = env
    async with _client() as client:
        first = await _submit(client, POOL)
        second = await _submit(client, {"pool": "one-shot"})
        row = (await client.get(f"/api/runs/{first}")).json()
        assert row["status"] == "queued"
        assert row["queue_reason"] == (
            "waiting for a one-shot worker that offers cpu (cdui agent "
            "start --one-shot)")
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            welcome = await agent.hello(name="slurm", one_shot=True,
                                        instance_id="a")
            assert wp.CAPABILITY_ONE_SHOT in welcome.capabilities
            assign = await agent.recv()
            assert isinstance(assign, wp.Assign) and assign.run_id == first
            listed = (await client.get("/api/workers")).json()["workers"]
            assert listed[0]["one_shot"] == {"run_id": None, "took": first}
            await agent.finish_run(assign)
            record = await _terminal(service, first)
            assert record.status == "succeeded"
            assert record.worker_id == "slurm"
            # The second run is not given to it.
            with pytest.raises(asyncio.TimeoutError):
                await agent.recv(timeout=0.5)
            row = (await client.get(f"/api/runs/{second}")).json()
            assert row["status"] == "queued"
            assert row["queue_key"] == f"{ONE_SHOT_QUEUE_PREFIX}/cpu"
        await _until(lambda: registry.get("slurm").status == "offline")
        listed = (await client.get("/api/workers")).json()["workers"]
        assert (listed[0]["status"], listed[0]["one_shot"]["took"]) == (
            "offline", first)
        # The next job's agent takes the second run.
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello(name="slurm", one_shot=True, instance_id="b")
            assign = await agent.recv()
            assert assign.run_id == second
            await agent.finish_run(assign, status="failed", error="boom")
            assert (await _terminal(service, second)).status == "failed"


async def test_any_runs_stay_here_beside_a_one_shot_worker(env):
    service, registry = env
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        agent = FakeAgent(ws)
        await agent.hello(name="slurm", devices=("cpu", "cuda:1"),
                          one_shot=True, instance_id="a")
        response = await client.post("/api/runs", json={
            "graph": _graph(), "options": {"device": "cuda:1",
                                           "target": {"worker": "any"}}})
        assert response.status_code == 400
        assert "neither this server nor a known worker offers cuda:1" in \
            response.text
        run_id = await _submit(client, {"worker": "any"})
        record = await _terminal(service, run_id)
        assert record.worker_id == "local"
        with pytest.raises(asyncio.TimeoutError):
            await agent.recv(timeout=0.3)


async def test_a_named_one_shot_worker_takes_the_first_run_naming_it(env):
    service, registry = env
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        agent = FakeAgent(ws)
        await agent.hello(name="slurm", one_shot=True, instance_id="a")
        first = await _submit(client, {"worker": "slurm"})
        second = await _submit(client, {"worker": "slurm"})
        assign = await agent.recv()
        assert assign.run_id == first
        row = (await client.get(f"/api/runs/{second}")).json()
        assert row["queue_reason"] == (
            f"one-shot worker slurm already took run {first}; start "
            "another one-shot agent")
        await agent.finish_run(assign)
        await _terminal(service, first)
        assert (await client.post(f"/api/runs/{second}/cancel")).status_code \
            == 200


@pytest.mark.parametrize("setup, words", [
    ("missing", "does not exist on this server"),
    ("other", "does not wait for a one-shot worker"),
    ("finished", "is not waiting to start (it is succeeded)"),
])
async def test_a_run_id_that_does_not_wait_for_the_agent_is_refused(
        env, setup, words):
    service, registry = env
    async with _client() as client:
        run_id = "0" * 32
        if setup == "other":
            async with aconnect_ws(_path(), client) as ws:
                await FakeAgent(ws).hello(name="box")
                run_id = await _submit(client, {"worker": "box"})
                # Not placed: the fake agent never answers, so it waits as
                # a running run on box. A run running elsewhere is refused.
        elif setup == "finished":
            run_id = await _submit(client, {"worker": "local"})
            await _terminal(service, run_id)
        async with aconnect_ws(_path(), client) as ws:
            reply = await FakeAgent(ws).hello(
                name="slurm", one_shot=True, one_shot_run_id=run_id,
                instance_id="a")
        assert isinstance(reply, wp.Reject), reply
        assert words in reply.reason or (
            setup == "other" and "already running" in reply.reason)
        assert registry.rejections()[0]["name"] == "slurm"


async def test_a_run_id_waiting_in_the_pool_is_the_one_it_gets(env):
    service, registry = env
    async with _client() as client:
        decoy = await _submit(client, POOL)
        wanted = await _submit(client, POOL)
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            welcome = await agent.hello(name="slurm", one_shot=True,
                                        one_shot_run_id=wanted,
                                        instance_id="a")
            assert isinstance(welcome, wp.Welcome)
            assign = await agent.recv()
            assert assign.run_id == wanted
            # Reconnecting mid-run (same agent): no second check, no new run.
            assert await service.one_shot_run_refusal(wanted, "slurm") is None
            assert "already running (on slurm)" in \
                await service.one_shot_run_refusal(wanted, "elsewhere")
            await agent.finish_run(assign)
            await _terminal(service, wanted)
        assert (await client.get(f"/api/runs/{decoy}")).json()["status"] \
            == "queued"


async def test_a_second_agent_under_a_busy_one_shot_name_is_refused(env):
    _, registry = env
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        await FakeAgent(ws).hello(name="slurm", one_shot=True,
                                  instance_id="a")
        async with aconnect_ws(_path(), client) as other:
            reply = await FakeAgent(other).hello(name="slurm", one_shot=True,
                                                 instance_id="b")
        assert isinstance(reply, wp.Reject) and "is busy" in reply.reason
        assert registry.get("slurm").online
        assert registry.get("slurm").instance_id == "a"


async def test_a_pool_run_fails_when_no_one_shot_worker_comes(env):
    service, registry = env
    service._one_shot_timeout_s = 0.3
    async with _client() as client:
        run_id = await _submit(client, POOL)
        local = await _submit(client, {"worker": "local"})
        record = await _terminal(service, run_id)
        assert record.status == "failed"
        assert "no one-shot worker took this run within 0.3 s" in record.error
        events = await service.store.get_events(run_id)
        assert events[-1].type == "execution_error"
        # Only runs that wait for a one-shot worker are timed.
        assert (await _terminal(service, local)).status == "succeeded"
        assert service._one_shot_timers == {}


async def test_a_run_that_starts_or_is_cancelled_is_not_timed_out(env):
    service, registry = env
    service._one_shot_timeout_s = 30
    async with _client() as client:
        cancelled = await _submit(client, POOL)
        assert cancelled in service._one_shot_timers
        await client.post(f"/api/runs/{cancelled}/cancel")
        assert cancelled not in service._one_shot_timers
        started = await _submit(client, POOL)
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello(name="slurm", one_shot=True, instance_id="a")
            assign = await agent.recv()
            assert assign.run_id == started
            assert started not in service._one_shot_timers
            await agent.finish_run(assign)
            await _terminal(service, started)
        waiting = await _submit(client, POOL)
        assert waiting in service._one_shot_timers
    await service.shutdown()
    assert service._one_shot_timers == {}


async def test_a_named_offline_worker_is_not_timed_and_none_is_disabled(env):
    service, registry = env
    service._one_shot_timeout_s = 0
    async with _client() as client:
        assert await _submit(client, POOL) not in service._one_shot_timers
        assert service._one_shot_timers == {}
    service._one_shot_timeout_s = 30
    assert not service._waits_for_one_shot({"target": {"worker": "local"}})
    assert not service._waits_for_one_shot({"target": {"worker": "ghost"}})


async def test_a_pool_needs_worker_agents():
    from app.core.run_service import RunService

    class _Store:
        pass

    service = RunService(_Store())  # no registry: no workers at all
    with pytest.raises(RunSubmitError, match="needs worker agents"):
        service._submit_queue_key({"device": "cpu", "target": POOL})


async def test_a_run_refused_before_its_assign_hands_the_claim_back(
        env, monkeypatch):
    from app.core.executors import WorkerExecutor

    async def refuse(request: Any, worker: Any) -> str:
        return "its node types are missing"

    monkeypatch.setattr(WorkerExecutor, "_catalog_refusal",
                        staticmethod(refuse))
    service, registry = env
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        await FakeAgent(ws).hello(name="slurm", one_shot=True,
                                  instance_id="a")
        run_id = await _submit(client, POOL)
        record = await _terminal(service, run_id)
        assert (record.status, record.error) == (
            "failed", "its node types are missing")
        assert registry.get("slurm").one_shot_claim is None


def test_the_hello_carries_the_one_shot_fields():
    hello = wp.decode_agent_frame(wp.encode_frame(wp.Hello(
        name="slurm", version=get_version(), one_shot=True,
        one_shot_run_id="abc", instance_id="ff")))
    assert (hello.one_shot, hello.one_shot_run_id, hello.instance_id) == (
        True, "abc", "ff")
    plain = wp.decode_agent_frame('{"type":"hello","name":"b","version":"v"}')
    assert (plain.one_shot, plain.one_shot_run_id, plain.instance_id) == (
        False, None, None)
    with pytest.raises(wp.ProtocolError):
        wp.decode_agent_frame(wp.encode_frame(wp.Hello(
            name="b", version="v")).replace(
                '"one_shot_run_id":null', '"one_shot_run_id":"../x"'))


# ── review fixes (#835) ──────────────────────────────────────────────────


async def test_a_run_cancelled_before_its_assign_hands_the_claim_back(env):
    """Every path that ends a run before ``assign`` went out frees the
    one-shot worker; here a cancel that lands before the run task starts."""
    service, registry = env
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        agent = FakeAgent(ws)
        await agent.hello(name="slurm", one_shot=True, instance_id="a")
        result = await service.submit(_graph(), options={
            "device": "cpu", "target": POOL})
        assert registry.get("slurm").one_shot_claim == result.run_id
        await service.cancel(result.run_id)  # before its task ever ran
        record = await _terminal(service, result.run_id)
        assert record.status == "cancelled"
        worker = registry.get("slurm")
        assert (worker.one_shot_claim, worker.one_shot_assigned) == (
            None, False)
        # It takes the next run.
        second = await _submit(client, POOL)
        assign = await agent.recv()
        assert assign.run_id == second
        assert registry.get("slurm").one_shot_assigned
        await agent.finish_run(assign)
        await _terminal(service, second)
        # An assigned claim stays after the run ends.
        assert registry.get("slurm").one_shot_claim == second


async def test_a_failed_assign_send_hands_the_claim_back(env):
    service, registry = env
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        await FakeAgent(ws).hello(name="slurm", one_shot=True,
                                  instance_id="a")
        worker = registry.get("slurm")
        real_send = worker.send

        async def broken(frame: Any) -> None:
            if isinstance(frame, wp.Assign):
                raise ConnectionError("the socket went away")
            await real_send(frame)

        worker.send = broken
        run_id = await _submit(client, POOL)
        record = await _terminal(service, run_id)
        assert record.status == "interrupted"
        assert registry.get("slurm").one_shot_claim is None


async def test_a_restarted_server_learns_the_taken_run_from_hello(env):
    """After a control-plane restart the agent's ``one_shot_taken`` is the
    only record that it took a run: no second run goes there, and its
    ``--run-id`` is not checked again."""
    service, registry = env
    async with _client() as client:
        waiting = await _submit(client, POOL)
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            reply = await agent.hello(
                name="slurm", one_shot=True, instance_id="a",
                one_shot_run_id="0" * 32, one_shot_taken="0" * 32)
            assert isinstance(reply, wp.Welcome), reply
            worker = registry.get("slurm")
            assert (worker.one_shot_claim, worker.one_shot_assigned) == (
                "0" * 32, True)
            with pytest.raises(asyncio.TimeoutError):
                await agent.recv(timeout=0.5)
        row = (await client.get(f"/api/runs/{waiting}")).json()
        assert row["status"] == "queued"
    # A regular hello's one_shot_taken means nothing.
    regular, _ = registry.connect(
        wp.Hello(name="box", version="t", one_shot_taken="r1"),
        send=_nothing, close=_nothing)
    assert regular.one_shot_claim is None


def test_the_pool_skips_a_one_shot_worker_any_would_skip():
    """A one-shot worker that lacks node types (or reported none) gets no
    pool run, as ``any`` gives it none; naming it still works."""
    from app.core.worker_handshake import CATALOG_UNKNOWN, CatalogCheck

    lacking = _one_shot("slurm", "cpu")
    lacking.catalog_checked = True
    lacking.catalog = CatalogCheck(status=CATALOG_UNKNOWN)
    entry = _Entry("r1", "pool:one-shot/cpu", _pool())
    assert _scheduler(lacking).place(entry, {}) is None
    assert _scheduler(lacking).place(
        _Entry("r1", "slurm/cpu", _target("slurm")), {}) is not None
    lacking.catalog = CatalogCheck(status="match")
    assert _scheduler(lacking).place(entry, {}) is not None


async def test_two_agents_racing_for_one_name_cannot_both_register(
        env, monkeypatch):
    """The busy-name check is repeated with no await before ``connect``,
    so an agent that passed the first check while another was registering
    is refused instead of replacing it."""
    from app.api import ws_worker

    service, registry = env
    gate = asyncio.Event()
    real = ws_worker._one_shot_run_refusal
    calls = []

    async def held(ws: Any, reg: Any, hello: Any) -> Any:
        calls.append(hello.instance_id)
        if hello.instance_id == "slow":
            await gate.wait()
        return await real(ws, reg, hello)

    monkeypatch.setattr(ws_worker, "_one_shot_run_refusal", held)
    async with _client() as client:
        async with aconnect_ws(_path(), client) as slow_ws:
            slow = FakeAgent(slow_ws)
            await slow.send(wp.Hello(
                name="slurm", version=get_version(), one_shot=True,
                instance_id="slow",
                devices=[wp.DeviceReport(value="cpu", kind="cpu")]))
            await _until(lambda: calls == ["slow"])
            async with aconnect_ws(_path(), client) as fast_ws:
                welcome = await FakeAgent(fast_ws).hello(
                    name="slurm", one_shot=True, instance_id="fast")
                assert isinstance(welcome, wp.Welcome)
                gate.set()
                assert isinstance(await slow.recv(), wp.Welcome)
                refused = await slow.recv()
                assert isinstance(refused, wp.Reject)
                assert "is busy" in refused.reason
                assert registry.get("slurm").instance_id == "fast"
                assert registry.get("slurm").online


async def test_an_expiry_that_cannot_write_is_retried_not_lost(
        env, monkeypatch):
    """A failed expiry puts the run back in its line with a retry timer;
    the retry fails it."""
    from app.core import run_service as rs

    service, registry = env
    service._one_shot_timeout_s = 0.2
    monkeypatch.setattr(rs, "ONE_SHOT_EXPIRY_RETRY_S", 0.2)
    real = service.store.mark_finished
    calls = []

    async def flaky(*args: Any, **kw: Any) -> Any:
        calls.append(args)
        if len(calls) == 1:
            raise OSError("database is locked")
        return await real(*args, **kw)

    monkeypatch.setattr(service.store, "mark_finished", flaky)
    async with _client() as client:
        run_id = await _submit(client, POOL)
        await _until(lambda: len(calls) == 1)
        await _until(lambda: run_id in service._one_shot_timers)
        assert service.queue_snapshot()["pool:one-shot/cpu"] == [run_id]
        record = await _terminal(service, run_id)
    assert record.status == "failed" and "one-shot" in record.error
    assert len(calls) == 2


async def test_shutdown_waits_for_an_expiry_in_flight(env, monkeypatch):
    """An expiry writing when the server stops is cancelled and awaited;
    the run is retired with the rest of the queue."""
    service, registry = env
    service._one_shot_timeout_s = 0.1
    entered = asyncio.Event()
    real = service.store.mark_finished

    async def stuck(run_id: str, status: str, *args: Any, **kw: Any) -> Any:
        if status == "failed":
            entered.set()
            await asyncio.sleep(3600)
        return await real(run_id, status, *args, **kw)

    monkeypatch.setattr(service.store, "mark_finished", stuck)
    async with _client() as client:
        run_id = await _submit(client, POOL)
    await asyncio.wait_for(entered.wait(), 5)
    tasks = list(service._one_shot_expiries)
    assert tasks
    await asyncio.wait_for(service.shutdown(), 10)
    assert all(task.done() for task in tasks)
    record = await service.store.get_run(run_id)
    assert record.status == "interrupted"


async def test_pool_runs_share_one_line_and_run_id_takes_its_run_out_of_it(
        env):
    """One FIFO per pool device; a ``--run-id`` agent takes its run out of
    turn, from the pool and from a line naming the worker."""
    service, registry = env
    async with _client() as client:
        first = await _submit(client, POOL)
        second = await _submit(client, POOL)
        assert service.queue_snapshot()["pool:one-shot/cpu"] == [first,
                                                                 second]
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello(name="slurm", one_shot=True, instance_id="a",
                              one_shot_run_id=second)
            assign = await agent.recv()
            assert assign.run_id == second
            await agent.finish_run(assign)
            await _terminal(service, second)
        named_a = await _submit(client, {"worker": "slurm"})
        named_b = await _submit(client, {"worker": "slurm"})
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello(name="slurm", one_shot=True, instance_id="b",
                              one_shot_run_id=named_b)
            assign = await agent.recv()
            assert assign.run_id == named_b
            await agent.finish_run(assign)
            await _terminal(service, named_b)
        for run_id in (first, named_a):
            assert (await client.get(f"/api/runs/{run_id}")).json()[
                "status"] == "queued"
