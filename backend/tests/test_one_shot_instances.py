"""One-shot follow-ups (#845): the persisted one-shot flag, and several
concurrent instances on one credential.

1. ``workers.one_shot`` (migration 016) records that an enrolled worker's
   last agent was one-shot, so a server that restarted times a run naming
   it (``CODEFYUI_WORKER_ONE_SHOT_QUEUE_TIMEOUT_S``) before that agent is
   back.
2. ``cdui agent start --one-shot --instance ID``: the agent says
   ``hello.instance`` and registers as ``<name>.<ID>``; the credential
   must name ``<name>``, a regular agent may not do it, the instances are
   listed with ``instance_of``, revoking the credential removes them all,
   and no instance can take over another's run (the socket, a restart's
   adoption, the transfer upload).

The agent side is ``tests/agent/test_one_shot_instances.py``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from httpx_ws import aconnect_ws

from app.core import worker_protocol as wp
from app.core.auth import TOKEN_HEADER
from app.core.db import Database
from app.core.run_service import WORKER_RESUME_KEY, RunService
from app.core.run_store import RunStore
from app.core.run_service import QueueLimits, parse_queue_overrides
from app.core.worker_handshake import (
    identity_refusal,
    instance_id_hash,
    one_shot_refusal,
    refusal,
    same_agent,
)
from app.core.worker_store import WorkerStore
from app.core.workers import (
    CREDENTIAL_SESSION,
    CREDENTIAL_WORKER,
    MAX_LEFT_INSTANCES,
    STATUS_DISCONNECTED,
    STATUS_OFFLINE,
    WorkerRegistry,
    policy_owner,
)
from app.main import app

from .test_transfer_backend import _upload, data_root, db, http  # noqa: F401
from .test_worker_adoption import Generation
from .test_ws_worker import (
    FakeAgent,
    _client,
    _graph,
    _path,
    _terminal,
    _until,
    env,  # noqa: F401
)
from .test_ws_worker_resume import SpoolingAgent

POOL = {"worker": "any", "pool": "one-shot"}


async def _nothing(*_: Any) -> None:
    return None


def _hello(name: str = "gpu", **kw: Any) -> wp.Hello:
    kw.setdefault("one_shot", True)
    return wp.Hello(name=name, version="t",
                    devices=[wp.DeviceReport(value="cpu", kind="cpu")], **kw)


async def _submit(client: Any, target: dict[str, Any]) -> str:
    response = await client.post("/api/runs", json={
        "graph": _graph(), "options": {"device": "cpu", "target": target}})
    assert response.status_code == 200, response.text
    return response.json()["run_id"]


# ── the protocol ──────────────────────────────────────────────────────────


def test_an_instance_id_is_the_credentials_name_and_the_instance():
    assert wp.instance_worker_id("gpu", None) == "gpu"
    assert wp.instance_worker_id("gpu", "4821-3") == "gpu.4821-3"
    assert wp.instance_parts("gpu.4821-3") == ("gpu", "4821-3")
    assert wp.instance_parts("gpu") == ("gpu", None)
    # Worker ids: a name, or a name and an instance; nothing else.
    for good in ("gpu", "gpu.3", "a-b.c-d"):
        assert wp.WORKER_ID_PATTERN.fullmatch(good)
    for bad in ("gpu.", ".3", "gpu.3.4", "gpu/3", "GPU.3", "gpu.-3"):
        assert wp.WORKER_ID_PATTERN.fullmatch(bad) is None
    # The separator is outside the name alphabet: no enrolled name is ever
    # an instance's id.
    assert wp.WORKER_NAME_PATTERN.fullmatch("gpu.3") is None
    assert wp.CAPABILITY_INSTANCES in wp.CAPABILITIES


@pytest.mark.parametrize("instance", ["", "A", "-3", "x" * 33, "a.b", "a/b"])
def test_a_malformed_instance_is_not_a_hello(instance):
    with pytest.raises(ValueError):
        _hello(instance=instance)


def test_only_a_one_shot_agent_may_be_an_instance():
    assert refusal(_hello(instance="3"), control_version="t") is None
    assert "--instance is for one-shot agents" in refusal(
        _hello(instance="3", one_shot=False), control_version="t")


async def test_a_credential_opens_instances_under_its_own_name_only():
    registry = WorkerRegistry()
    try:
        hello = _hello("gpu.3", instance="3")
        assert identity_refusal(hello, credential=CREDENTIAL_WORKER,
                                enrolled_as="gpu", registry=registry) is None
        assert "belongs to worker 'gpu', not 'cpu-box'" in identity_refusal(
            _hello("cpu-box.3", instance="3"), credential=CREDENTIAL_WORKER,
            enrolled_as="gpu", registry=registry)
        # The session token may open instances of a name nobody enrolled,
        # never of an enrolled worker's.
        assert identity_refusal(hello, credential=CREDENTIAL_SESSION,
                                enrolled_as=None, registry=registry) is None
        registry.load_known([SimpleNamespace(
            name="gpu", platform={}, mounts={}, devices=[],
            last_seen_at=None)])
        assert "its instances connect with its worker token" in \
            identity_refusal(hello, credential=CREDENTIAL_SESSION,
                             enrolled_as=None, registry=registry)
    finally:
        await registry.shutdown()


def test_the_same_agent_is_its_instance_id_or_after_a_restart_its_hash():
    taken = instance_id_hash("secret-a")
    assert instance_id_hash(None) is None and taken != "secret-a"
    live = SimpleNamespace(instance_id="secret-a", instance_hash=None)
    # A restarted server knows the process only by the hash it recorded.
    known = SimpleNamespace(instance_id=None, instance_hash=taken)
    nothing = SimpleNamespace(instance_id=None, instance_hash=None)
    for holder in (live, known):
        assert same_agent(_hello(instance_id="secret-a"), holder)
        for other in ("secret-b", None):
            assert not same_agent(_hello(instance_id=other), holder)
    assert not same_agent(_hello(instance_id="secret-a"), nothing)


def test_a_one_shot_run_adopted_on_a_known_worker_is_its_agents():
    """Review of #855: a known (enrolled) one-shot worker has no
    ``instance_id`` after a restart; the hash recorded with its run lets
    its own agent take the run back, and nobody else."""
    known = SimpleNamespace(one_shot=True, online=False, running={"r1": 1},
                            instance_id=None,
                            instance_hash=instance_id_hash("secret-a"))
    assert one_shot_refusal(_hello(instance_id="secret-a"), known) is None
    assert "its run is in flight" in one_shot_refusal(
        _hello(instance_id="secret-b"), known)


# ── the registry ──────────────────────────────────────────────────────────


async def test_instances_are_listed_under_their_credential():
    registry = WorkerRegistry()
    try:
        a, _ = registry.connect(_hello("gpu.a", instance="a"),
                                send=_nothing, close=_nothing)
        b, _ = registry.connect(_hello("gpu.b", instance="b"),
                                send=_nothing, close=_nothing)
        regular, _ = registry.connect(wp.Hello(name="box", version="t"),
                                      send=_nothing, close=_nothing)
        assert (a.instance_of, a.instance) == ("gpu", "a")
        assert a.describe()["instance_of"] == "gpu"
        assert a.describe()["instance"] == "a"
        assert regular.describe()["instance_of"] is None
        assert regular.describe()["instance"] is None
        assert {w.id for w in registry.instances_of("gpu")} == {"gpu.a",
                                                                "gpu.b"}
        assert registry.instances_of("box") == []
        # Each is an ordinary one-shot worker with a claim of its own.
        registry.claim_one_shot("gpu.a", "r1")
        assert a.one_shot_claim == "r1" and b.one_shot_available
        assert registry.forget_instances("gpu", "revoked") == ["gpu.a",
                                                               "gpu.b"]
        assert registry.get("gpu.a") is None and registry.get("box")
    finally:
        await registry.shutdown()


async def test_instances_that_dropped_mid_run_are_pruned_once_it_ended():
    """Review of #855: an instance that dropped with its run in flight
    stays ``disconnected`` after the resume grace ended the run; it left
    all the same, and is pruned like one that left after its run."""
    registry = WorkerRegistry(resume_grace_s=60)
    try:
        dropped = []
        for index in range(MAX_LEFT_INSTANCES + 2):
            worker, _ = registry.connect(
                _hello(f"gpu.{index}", instance=str(index)),
                send=_nothing, close=_nothing)
            registry.open_run(worker, f"r{index}")
            registry.disconnect(worker)
            assert worker.status == STATUS_DISCONNECTED and worker.running
            dropped.append(worker)
        registry.connect(_hello("gpu.live", instance="live"),
                         send=_nothing, close=_nothing)
        # Their runs still wait for them: nothing is pruned.
        assert len(registry.instances_of("gpu")) == MAX_LEFT_INSTANCES + 3
        for worker in dropped:
            for run in list(worker.running.values()):
                registry._end(worker, run, "grace over")
        registry.connect(_hello("gpu.next", instance="next"),
                         send=_nothing, close=_nothing)
        ids = [w.id for w in registry.instances_of("gpu")]
        assert len(ids) == MAX_LEFT_INSTANCES + 2
        assert ids[:2] == ["gpu.2", "gpu.3"]  # the two oldest went
        assert ids[-2:] == ["gpu.live", "gpu.next"]
        # The index follows a reconnect (newest last) and a removal.
        registry.connect(_hello("gpu.2", instance="2"), send=_nothing,
                         close=_nothing)
        assert [w.id for w in registry.instances_of("gpu")][-1] == "gpu.2"
        registry.forget("gpu.2")
        assert "gpu.2" not in {w.id for w in registry.instances_of("gpu")}
        registry.forget_instances("gpu", "revoked")
        assert registry.instances_of("gpu") == [] and not registry._instances
    finally:
        await registry.shutdown()


async def test_instances_inherit_the_credentials_drain_and_policies():
    """Review of #855: what is set on the credential applies to its
    instances: draining, device admission policies, queue caps."""
    registry = WorkerRegistry()
    try:
        registry.load_known([SimpleNamespace(
            name="gpu", platform={}, mounts={}, devices=[],
            last_seen_at=None)])
        a, _ = registry.connect(_hello("gpu.a", instance="a"),
                                send=_nothing, close=_nothing)
        registry.drain("gpu")
        assert a.draining and not a.accepts_runs
        # One starting while the credential drains starts drained.
        b, _ = registry.connect(_hello("gpu.b", instance="b"),
                                send=_nothing, close=_nothing)
        assert b.draining
        registry.drain("gpu", False)
        assert not a.draining and not b.draining
        # Draining one instance leaves the others alone.
        registry.drain("gpu.a")
        assert a.draining and not b.draining
        assert registry.get("gpu").draining is False

        assert policy_owner("gpu.a") == "gpu" and policy_owner("gpu") == "gpu"
        registry.set_device_policy("gpu", "cuda:0", "shared")
        assert registry.device_policy("gpu.b", "cuda:0") == "shared"
        assert registry._shares_by_memory(b)
        # Set through an instance, it is the credential's.
        registry.set_device_policy("gpu.a", "cuda:1", "shared")
        assert registry.device_policy("gpu", "cuda:1") == "shared"
        assert registry.device_policy("other", "cuda:0") == "exclusive"
    finally:
        await registry.shutdown()


def test_an_instance_takes_its_credentials_queue_cap():
    overrides = parse_queue_overrides(
        "gpu/cuda:0=3,gpu.b/cuda:0=5,box.x/cpu=4")
    assert ("gpu.b/cuda:0", 5) in overrides  # an instance's id is accepted
    limits = QueueLimits(cpu=2, gpu=1, overrides=overrides)
    assert limits.for_slot(("gpu", "cuda:0")) == 3
    assert limits.for_slot(("gpu.a", "cuda:0")) == 3  # the credential's
    assert limits.for_slot(("gpu.b", "cuda:0")) == 5  # its own wins
    assert limits.for_slot(("gpu.a", "cuda:1")) == 1
    assert limits.for_slot(("box.x", "cpu")) == 4
    assert limits.for_slot(("box", "cpu")) == 2


async def test_only_the_newest_instances_that_left_stay_listed():
    registry = WorkerRegistry()
    try:
        for index in range(MAX_LEFT_INSTANCES + 3):
            worker, _ = registry.connect(
                _hello(f"gpu.{index}", instance=str(index)),
                send=_nothing, close=_nothing)
            worker.connected_at = float(index)
            registry.disconnect(worker)
            assert worker.status == STATUS_OFFLINE
        # One more connects: the oldest that left go, the live one stays.
        live, _ = registry.connect(_hello("gpu.live", instance="live"),
                                   send=_nothing, close=_nothing)
        ids = {w.id for w in registry.instances_of("gpu")}
        assert len(ids) == MAX_LEFT_INSTANCES + 1 and live.id in ids
        assert {"gpu.0", "gpu.1", "gpu.2"}.isdisjoint(ids)
    finally:
        await registry.shutdown()


async def test_a_run_adopted_under_an_instance_waits_for_its_process():
    registry = WorkerRegistry()
    try:
        registry.adopt("gpu.3", "r1", instance_hash="h", one_shot=True)
        placeholder = registry.holder("gpu.3")
        assert placeholder.instance_hash == "h" and placeholder.one_shot
        assert registry.get("gpu.3") is None  # unlisted
        assert registry.holder("gpu.4") is None
        registry.adopt("box", "r2")
        assert registry.holder("box").instance_hash is None
        assert not registry.holder("box").one_shot
        # A regular run records its agent too (uploads), and stays regular.
        registry.adopt("lab", "r3", instance_hash="k")
        assert registry.holder("lab").instance_hash == "k"
        assert not registry.holder("lab").one_shot
        # Revoking the credential ends the adopted run too.
        assert registry.forget_instances("gpu", "revoked") == ["gpu.3"]
        assert registry.holder("gpu.3") is None
        assert registry.run_state("gpu.3", "r1") is None
    finally:
        await registry.shutdown()


# ── over /ws/worker, with an enrolled credential ─────────────────────────


@pytest.fixture
async def enrolled(env):  # noqa: F811 -- the imported fixture
    """``env`` plus a worker store with ``gpu`` enrolled."""
    service, registry = env
    store = WorkerStore(app.state.db)
    app.state.worker_store = store
    created = await store.create_enrollment()
    worker_token, _ = await store.enroll(
        created["token"], name="gpu", platform={}, mounts={}, devices=[])
    registry.load_known(await store.list_workers())
    try:
        yield service, registry, store, worker_token
    finally:
        del app.state.worker_store


def _ws(client: Any, token: str) -> Any:
    """A ``/ws/worker`` dial with the worker *token* (the client's own
    session token header is replaced)."""
    return aconnect_ws(wp.WORKER_WS_PATH, client,
                       headers={TOKEN_HEADER: token})


async def test_one_credential_serves_concurrent_instances_of_a_sweep(
        enrolled):
    service, registry, store, token = enrolled
    async with _client() as client:
        runs = [await _submit(client, POOL) for _ in range(3)]
        async with _ws(client, token) as ws_a, \
                _ws(client, token) as ws_b:
            a, b = FakeAgent(ws_a), FakeAgent(ws_b)
            welcome_a = await a.hello("gpu", one_shot=True, instance="a",
                                      instance_id="secret-a")
            welcome_b = await b.hello("gpu", one_shot=True, instance="b",
                                      instance_id="secret-b")
            assert welcome_a.worker_id == "gpu.a"
            assert welcome_b.worker_id == "gpu.b"
            assign_a, assign_b = await a.recv(), await b.recv()
            # Oldest first, one run each; the third waits for a third job.
            assert {assign_a.run_id, assign_b.run_id} == set(runs[:2])
            listed = (await client.get("/api/workers")).json()["workers"]
            by_id = {w["id"]: w for w in listed}
            assert by_id["gpu.a"]["instance_of"] == "gpu"
            assert by_id["gpu.a"]["instance"] == "a"
            assert by_id["gpu.a"]["enrolled"] is False
            assert by_id["gpu.a"]["credential"] == CREDENTIAL_WORKER
            assert by_id["gpu"]["enrolled"] is True
            assert by_id["gpu"]["instance_of"] is None
            # The credential's row is its own agent's: no instance writes
            # it (review of #855).
            row = (await store.list_workers())[0]
            assert row.one_shot is False and row.devices == []
            assert not registry.get("gpu").one_shot
            await a.finish_run(assign_a)
            await b.finish_run(assign_b)
            for assign in (assign_a, assign_b):
                record = await _terminal(service, assign.run_id)
                assert record.status == "succeeded"
                assert record.worker_id in ("gpu.a", "gpu.b")
        assert (await service.store.get_run(runs[2])).status == "queued"


async def test_a_second_agent_under_a_live_instance_id_is_refused(enrolled):
    _, registry, _, token = enrolled
    async with _client() as client, \
            _ws(client, token) as ws_a, \
            _ws(client, token) as ws_b:
        await FakeAgent(ws_a).hello("gpu", one_shot=True, instance="3",
                                    instance_id="secret-a")
        reject = await FakeAgent(ws_b).hello("gpu", one_shot=True,
                                             instance="3",
                                             instance_id="secret-b")
        assert isinstance(reject, wp.Reject)
        assert "is busy" in reject.reason and "--instance" in reject.reason
        assert registry.get("gpu.3").instance_id == "secret-a"


async def test_a_regular_agent_cannot_use_instances(enrolled):
    _, registry, _, token = enrolled
    async with _client() as client, \
            _ws(client, token) as ws:
        reject = await FakeAgent(ws).hello("gpu", instance="3")
        assert isinstance(reject, wp.Reject)
        assert "--instance is for one-shot agents" in reject.reason
    assert registry.get("gpu.3") is None


async def test_revoking_the_credential_removes_every_instance(enrolled):
    service, registry, store, token = enrolled
    async with _client() as client:
        run_id = await _submit(client, POOL)
        async with _ws(client, token) as ws_a, \
                _ws(client, token) as ws_b:
            a = FakeAgent(ws_a)
            await a.hello("gpu", one_shot=True, instance="a",
                          instance_id="secret-a")
            assert (await a.recv()).run_id == run_id
            await a.send(wp.RunStarted(run_id=run_id, seq=1, pid=1))
            await FakeAgent(ws_b).hello("gpu", one_shot=True, instance="b",
                                        instance_id="secret-b")
            waiting = await _submit(client, {"worker": "gpu.b"})
            await _until(lambda: registry.get("gpu.b").one_shot_claim
                         == waiting)
            removed = await client.delete("/api/workers/gpu")
            assert removed.status_code == 200, removed.text
            assert removed.json()["revoked"] is True
            record = await _terminal(service, run_id)
            assert record.status == "interrupted"
            assert registry.instances_of("gpu") == []
            assert registry.get("gpu") is None
        assert await store.authenticate(token) is None


async def test_a_queued_run_naming_an_instance_fails_on_revocation(enrolled):
    service, registry, _, token = enrolled
    async with _client() as client:
        async with _ws(client, token) as ws:
            await FakeAgent(ws).hello("gpu", one_shot=True, instance="a",
                                      instance_id="secret-a",
                                      one_shot_run_id=None)
            registry.claim_one_shot("gpu.a", "elsewhere")
            queued = await _submit(client, {"worker": "gpu.a"})
            removed = (await client.delete("/api/workers/gpu")).json()
        assert queued in removed["failed_runs"]
        assert (await service.store.get_run(queued)).status == "failed"


async def test_removing_one_instance_keeps_the_credential(enrolled):
    _, registry, store, token = enrolled
    async with _client() as client:
        async with _ws(client, token) as ws:
            await FakeAgent(ws).hello("gpu", one_shot=True, instance="a",
                                      instance_id="secret-a")
            # Second review of #855: a live instance is not removed on its
            # own (its agent would only reconnect, its run would end).
            live = await client.delete("/api/workers/gpu.a")
            assert live.status_code == 409
            assert "remove the credential 'gpu'" in live.text
            assert registry.get("gpu.a").online
        await _until(lambda: registry.get("gpu.a").status == STATUS_OFFLINE)
        removed = await client.delete("/api/workers/gpu.a")
        assert removed.status_code == 200
        assert removed.json()["revoked"] is False
        assert registry.get("gpu.a") is None
        assert await store.authenticate(token) == "gpu"


async def test_revoking_fails_runs_naming_instances_no_longer_listed(
        enrolled):
    """Second review of #855: a run queued for an instance that left the
    list (pruned, or not back since a restart) failed only if the
    registry still indexed it; it stayed queued for ever."""
    service, registry, _, token = enrolled
    async with _client() as client:
        async with _ws(client, token) as ws:
            await FakeAgent(ws).hello("gpu", one_shot=True, instance="b",
                                      instance_id="secret-b")
            registry.claim_one_shot("gpu.b", "elsewhere")
            queued = await _submit(client, {"worker": "gpu.b"})
        await _until(lambda: registry.get("gpu.b").status == STATUS_OFFLINE)
        registry.forget("gpu.b")  # as pruning does
        assert registry.instances_of("gpu") == []
        removed = (await client.delete("/api/workers/gpu")).json()
    assert removed["failed_runs"] == [queued]
    assert (await service.store.get_run(queued)).status == "failed"


async def test_removing_a_session_token_worker_leaves_its_instances(env):  # noqa: F811
    """Second review of #855: a same-host worker on the session token has
    no credential to revoke; its instances did not use one, and stay."""
    service, registry = env
    async with _client() as client:
        async with aconnect_ws(_path(), client) as ws_box, \
                aconnect_ws(_path(), client) as ws_i:
            await FakeAgent(ws_box).hello("box", instance_id="box")
            await FakeAgent(ws_i).hello("box", one_shot=True, instance="1",
                                        instance_id="secret-1")
            registry.claim_one_shot("box.1", "elsewhere")
            queued = await _submit(client, {"worker": "box.1"})
            removed = await client.delete("/api/workers/box")
            assert removed.status_code == 200, removed.text
            assert removed.json()["revoked"] is False
            assert removed.json()["failed_runs"] == []
            assert registry.get("box") is None
            assert registry.get("box.1").online
        assert (await service.store.get_run(queued)).status == "queued"


async def test_record_outputs_is_refused_on_an_instance(enrolled):
    """#789 refuses record_outputs on one-shot workers; an instance
    (#845) is one: refused at submit, named or through the pool, and at
    start (``WorkerExecutor``) on the instance's ``one_shot``."""
    from app.core.executors import ExecutionRequest, WorkerExecutor
    from app.core.run_output_store import RunOutputStore
    from app.core.run_service import ONE_SHOT_REFUSES_RECORD_OUTPUTS
    from app.core.scheduling import Placement

    _, registry, _, token = enrolled
    async with _client() as client, _ws(client, token) as ws:
        await FakeAgent(ws).hello("gpu", one_shot=True, instance="a",
                                  instance_id="secret-a",
                                  capabilities=list(wp.CAPABILITIES))
        for target in ({"worker": "gpu.a"}, POOL):
            response = await client.post("/api/runs", json={
                "graph": _graph(), "options": {
                    "device": "cpu", "target": target,
                    "record_outputs": True}})
            assert response.status_code == 400, response.text
            assert "not available on one-shot workers" in response.text
        executor = WorkerExecutor(registry)
        outcome = await executor.run(ExecutionRequest(
            run_id="r1", graph={"nodes": []}, options={},
            context=SimpleNamespace(should_stop=lambda: False),
            session=None, output_store=RunOutputStore(max_runs=1,
                                                      max_bytes=0),
            record_outputs=True,
            sinks=SimpleNamespace(warn=lambda *_: None,
                                  emit=lambda *_: None),
            explicit_metric_nodes=set(),
            placement=Placement(worker_id="gpu.a", devices=("cpu",),
                                slots=())))
        assert outcome.status == "failed"
        assert outcome.error == ONE_SHOT_REFUSES_RECORD_OUTPUTS


def test_a_run_naming_an_instance_is_timed_listed_or_not():
    """Second review of #855: an instance is known in memory only; a run
    naming one is timed by the one-shot rule even when the registry no
    longer lists it (pruned, or gone with a restart)."""
    service = RunService.__new__(RunService)
    service._workers = WorkerRegistry()
    assert service._waits_for_one_shot({"target": {"worker": "gpu.7"}})
    assert not service._waits_for_one_shot({"target": {"worker": "gpu"}})
    assert not service._waits_for_one_shot({"target": {"worker": "local"}})


# ── 1. the one-shot flag survives a restart (migration 016) ─────────────


async def test_a_run_naming_a_one_shot_worker_is_timed_after_a_restart(
        tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    store = WorkerStore(database)
    created = await store.create_enrollment()
    await store.enroll(created["token"], name="slurm", platform={},
                       mounts={}, devices=[{"value": "cpu", "kind": "cpu"}])
    assert (await store.list_workers())[0].one_shot is False
    await store.touch("slurm", one_shot=True)  # its last hello
    registry = WorkerRegistry()
    service = RunService(RunStore(database), workers=registry,
                         shutdown_grace_s=1.0, one_shot_queue_timeout_s=0.3)
    app.state.db = database
    app.state.run_service = service
    app.state.worker_registry = registry
    try:
        # A new process: the worker is known offline, and known one-shot.
        known = registry.load_known(await store.list_workers())
        assert known[0].one_shot and known[0].status == STATUS_OFFLINE
        assert known[0].describe()["one_shot"] == {"run_id": None,
                                                   "took": None}
        async with _client() as client:
            run_id = await _submit(client, {"worker": "slurm"})
            assert run_id in service._one_shot_timers
            # An instance is known in memory only: after a restart a run
            # naming one is refused at once instead of waiting.
            refused = await client.post("/api/runs", json={
                "graph": _graph(), "options": {
                    "device": "cpu", "target": {"worker": "slurm.9000-3"}}})
            assert refused.status_code == 400
            assert "has not connected" in refused.text
            record = await _terminal(service, run_id)
        assert record.status == "failed"
        assert "no one-shot worker took this run within 0.3 s" in \
            record.error
        # A regular agent's hello clears it again.
        await store.touch("slurm", one_shot=False)
        assert (await store.list_workers())[0].one_shot is False
        await store.touch("slurm")  # no hello: left as it is
        assert (await store.list_workers())[0].one_shot is False
    finally:
        await service.shutdown()
        await registry.shutdown()
        database.close()
        for attribute in ("db", "run_service", "worker_registry"):
            if hasattr(app.state, attribute):
                delattr(app.state, attribute)


async def test_a_regular_workers_hello_records_it_as_not_one_shot(enrolled):
    _, _, store, token = enrolled
    await store.touch("gpu", one_shot=True)
    async with _client() as client, \
            _ws(client, token) as ws:
        assert isinstance(await FakeAgent(ws).hello("gpu"), wp.Welcome)
        await _until(lambda: True)
    for _ in range(100):
        if not (await store.list_workers())[0].one_shot:
            break
        await asyncio.sleep(0.02)
    assert (await store.list_workers())[0].one_shot is False


# ── a restart with an instance's run in flight ───────────────────────────


@pytest.fixture
async def generations(tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    app.state.db = database
    started: list[Generation] = []

    def start(**kwargs: Any) -> Generation:
        generation = Generation(database, **kwargs)
        started.append(generation)
        return generation

    try:
        yield start
    finally:
        for generation in reversed(started):
            await generation.close()
        database.close()
        for attribute in ("db", "run_service", "worker_registry"):
            if hasattr(app.state, attribute):
                delattr(app.state, attribute)


async def test_an_adopted_instance_run_is_taken_back_by_its_process_only(
        generations, tmp_path):
    async with _client() as client:
        first = generations()
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello("box", one_shot=True, instance="7",
                              instance_id="secret-7")
            run_id = await _submit(client, POOL)
            assert isinstance(await agent.recv(), wp.Assign)
            spooler = SpoolingAgent(tmp_path, run_id)
            spooler.agent = agent
            await spooler.produce(spooler.frame("started"))
            await _until(lambda: first.service.running_on(run_id) == "box.7")
        spooler.agent = None
        await _until(lambda: first.registry.get("box.7").status
                     == "disconnected")
        facts = (await first.service.store.get_run(run_id)).execution_facts
        assert facts[WORKER_RESUME_KEY]["instance"] == instance_id_hash(
            "secret-7")
        await first.crash()

        second = generations()
        await second.service.recover_interrupted()
        assert second.service.running_on(run_id) == "box.7"
        # Another instance under the same id (or one without its id) is
        # refused: it would be handed the run and end it.
        for other in ("secret-8", None):
            async with aconnect_ws(_path(), client) as ws:
                reject = await FakeAgent(ws).hello(
                    "box", one_shot=True, instance="7", instance_id=other)
                assert isinstance(reject, wp.Reject)
                assert "its run is in flight" in reject.reason
        assert second.service.running_on(run_id) == "box.7"
        final = spooler.frame("finished")
        await spooler.produce(final)
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            welcome = await agent.hello("box", one_shot=True, instance="7",
                                        instance_id="secret-7",
                                        one_shot_taken=run_id)
            assert isinstance(welcome, wp.Welcome)
            await spooler.resume(agent)
            record = await _terminal(second.service, run_id)
    assert record.status == "succeeded", record.error


# ── the transfer upload ──────────────────────────────────────────────────


async def test_an_instance_cannot_upload_into_the_credentials_own_run(
        http, monkeypatch):  # noqa: F811
    """Second review of #855 (security): every agent of a credential holds
    its token; an upload for the credential's own worker's run is taken
    only from the agent process running it, live or (after a restart)
    by the hash its run recorded."""
    http.running["r1"] = "gpu-1"
    holders = {"gpu-1": SimpleNamespace(instance_id="regular",
                                        instance_hash=None)}
    registry = SimpleNamespace(holder=holders.get, get=holders.get)
    monkeypatch.setattr(app.state, "worker_registry", registry,
                        raising=False)

    def upload(instance: str | None) -> Any:
        headers = {TOKEN_HEADER: "token-gpu-1"}
        if instance is not None:
            headers[wp.INSTANCE_HEADER] = instance
        return http.post("/api/runs/r1/artifacts/upload", headers=headers,
                         data={"kind": "checkpoint", "name": "a.pt"},
                         files={"file": ("a.pt", b"w",
                                         "application/octet-stream")})

    for intruder in ("secret-a", None, ""):  # an instance, or no id at all
        assert (await upload(intruder)).status_code == 409
    assert (await upload("regular")).status_code == 200
    # After a restart, before the agent is back: the recorded hash.
    holders["gpu-1"] = SimpleNamespace(
        instance_id=None, instance_hash=instance_id_hash("regular"))
    assert (await upload("secret-a")).status_code == 409
    assert (await upload("regular")).status_code == 200


async def test_an_instance_uploads_for_its_own_run_only(http, monkeypatch):  # noqa: F811
    http.running["r1"] = "gpu-1.a"
    registry = SimpleNamespace(get=lambda worker_id: {
        "gpu-1.a": SimpleNamespace(instance_id="secret-a")}.get(worker_id))
    monkeypatch.setattr(app.state, "worker_registry", registry,
                        raising=False)

    def upload(token: str, instance: str | None) -> Any:
        headers = {TOKEN_HEADER: token}
        if instance is not None:
            headers[wp.INSTANCE_HEADER] = instance
        return http.post("/api/runs/r1/artifacts/upload", headers=headers,
                         data={"kind": "checkpoint", "name": "a.pt"},
                         files={"file": ("a.pt", b"w",
                                         "application/octet-stream")})

    # The credential's token alone, or with another instance's id: no.
    for instance in (None, "secret-b", ""):
        refused = await upload("token-gpu-1", instance)
        assert refused.status_code == 409, refused.text
    # Another credential with the right id: no.
    assert (await upload("token-gpu-2", "secret-a")).status_code == 409
    accepted = await upload("token-gpu-1", "secret-a")
    assert accepted.status_code == 200, accepted.text
    # The registry no longer knowing the instance: no.
    monkeypatch.setattr(app.state, "worker_registry",
                        SimpleNamespace(get=lambda _id: None), raising=False)
    assert (await upload("token-gpu-1", "secret-a")).status_code == 409
    # A regular worker's upload is unchanged.
    http.running["r1"] = "gpu-1"
    assert (await _upload(http, "token-gpu-1")).status_code == 200


async def test_the_handshake_checks_an_adopted_id_before_registering(
        env, monkeypatch):  # noqa: F811
    """``ws_worker`` looks up who holds an instance under its full id (not
    the credential's name)."""
    _, registry = env
    seen: list[str] = []
    original = registry.holder

    def spy(worker_id: str) -> Any:
        seen.append(worker_id)
        return original(worker_id)

    monkeypatch.setattr(registry, "holder", spy)
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        assert isinstance(await FakeAgent(ws).hello(
            "box", one_shot=True, instance="9", instance_id="i"), wp.Welcome)
    assert seen and set(seen) == {"box.9"}


# ── review of #855: a restart with the credential's own agent ────────────


@pytest.fixture
async def enrolled_generations(generations):
    """``generations`` plus a worker store with ``gpu`` enrolled; each
    generation knows the enrolled workers, as ``main`` loads them."""
    database = app.state.db
    store = WorkerStore(database)
    app.state.worker_store = store
    created = await store.create_enrollment()
    token, _ = await store.enroll(created["token"], name="gpu", platform={},
                                  mounts={}, devices=[])

    async def start() -> Generation:
        generation = generations()
        generation.registry.load_known(await store.list_workers())
        return generation

    try:
        yield start, store, token
    finally:
        del app.state.worker_store


async def test_an_instance_never_costs_the_credentials_own_agent_its_run(
        enrolled_generations, tmp_path):
    """Review of #855 (item 1): an instance's hello wrote the credential's
    row (one-shot, its devices) while the credential's regular agent was
    connected; after a restart that agent was refused as busy and the run
    it was running died with it."""
    start, store, token = enrolled_generations
    async with _client() as client:
        first = await start()
        async with _ws(client, token) as ws:
            regular = FakeAgent(ws)
            assert isinstance(await regular.hello(
                "gpu", devices=("cpu", "cuda:0"), instance_id="regular"),
                wp.Welcome)
            run_id = await _submit(client, {"worker": "gpu"})
            assert isinstance(await regular.recv(), wp.Assign)
            spooler = SpoolingAgent(tmp_path, run_id)
            spooler.agent = regular
            await spooler.produce(spooler.frame("started"))
            await _until(lambda: first.service.running_on(run_id) == "gpu")
            async with _ws(client, token) as ws_i:
                assert isinstance(await FakeAgent(ws_i).hello(
                    "gpu", one_shot=True, instance="7",
                    instance_id="secret-7"), wp.Welcome)
                await _until(lambda: first.registry.get("gpu.7") is not None)
            row = (await store.list_workers())[0]
            assert row.one_shot is False
            assert [d["value"] for d in row.devices] == ["cpu", "cuda:0"]
            assert not first.registry.get("gpu").one_shot
        spooler.agent = None
        await first.crash()

        second = await start()
        assert not second.registry.get("gpu").one_shot
        await second.service.recover_interrupted()
        assert second.service.running_on(run_id) == "gpu"
        await spooler.produce(spooler.frame("finished"))
        async with _ws(client, token) as ws:
            agent = FakeAgent(ws)
            assert isinstance(await agent.hello(
                "gpu", devices=("cpu", "cuda:0"), instance_id="regular"),
                wp.Welcome)
            await spooler.resume(agent)
            record = await _terminal(second.service, run_id)
    assert record.status == "succeeded", record.error


async def test_an_enrolled_one_shot_agent_takes_its_run_back_after_a_restart(
        enrolled_generations, tmp_path):
    """Review of #855 (item 2): a known one-shot worker loaded from its row
    has no ``instance_id``; the hash its run recorded lets its own agent,
    and only that one, take the run back."""
    start, store, token = enrolled_generations
    async with _client() as client:
        first = await start()
        async with _ws(client, token) as ws:
            agent = FakeAgent(ws)
            await agent.hello("gpu", one_shot=True, instance_id="secret-a")
            run_id = await _submit(client, POOL)
            assert isinstance(await agent.recv(), wp.Assign)
            spooler = SpoolingAgent(tmp_path, run_id)
            spooler.agent = agent
            await spooler.produce(spooler.frame("started"))
            await _until(lambda: first.service.running_on(run_id) == "gpu")
        spooler.agent = None
        facts = (await first.service.store.get_run(run_id)).execution_facts
        assert facts[WORKER_RESUME_KEY]["instance"] == instance_id_hash(
            "secret-a")
        await first.crash()

        second = await start()
        known = second.registry.get("gpu")
        assert known.one_shot and known.instance_id is None
        await second.service.recover_interrupted()
        assert known.instance_hash == instance_id_hash("secret-a")
        async with _ws(client, token) as ws:
            reject = await FakeAgent(ws).hello("gpu", one_shot=True,
                                               instance_id="secret-b")
            assert isinstance(reject, wp.Reject)
            assert "its run is in flight" in reject.reason
        await spooler.produce(spooler.frame("finished"))
        async with _ws(client, token) as ws:
            agent = FakeAgent(ws)
            assert isinstance(await agent.hello(
                "gpu", one_shot=True, instance_id="secret-a",
                one_shot_run_id=run_id, one_shot_taken=run_id), wp.Welcome)
            await spooler.resume(agent)
            record = await _terminal(second.service, run_id)
    assert record.status == "succeeded", record.error


async def test_a_policy_set_on_an_instance_is_its_credentials(enrolled):
    """Review of #855 (item 7): an instance's card admits as the
    credential says, so a policy set through it is stored for the
    credential, and every instance (and the next array) uses it."""
    _, registry, store, token = enrolled
    async with _client() as client, _ws(client, token) as ws:
        await FakeAgent(ws).hello("gpu", devices=("cpu", "cuda:0"),
                                  one_shot=True, instance="a",
                                  instance_id="secret-a")
        response = await client.put(
            "/api/workers/gpu.a/devices/cuda:0/policy",
            json={"policy": "shared"})
        assert response.status_code == 200, response.text
        assert response.json()["devices"][1]["policy"] == "shared"
    assert await store.list_device_policies() == [("gpu", "cuda:0",
                                                   "shared")]
    assert registry.device_policy("gpu", "cuda:0") == "shared"
