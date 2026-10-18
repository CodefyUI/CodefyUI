"""``record_outputs`` on worker runs, the control plane's half (#789).

The store's retention of a worker's runs by id, the shared reads
(``core.output_queries``) and their disk form, the reads in flight to a
worker (``core.worker_outputs``), the outputs routes forwarding a worker
run's reads, and where ``WorkerExecutor`` / ``RunService`` take part. The
agent's half is ``tests/agent/test_outputs.py``; the two together, with a
real agent, ``test_worker_record_outputs_e2e.py``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from app.core import output_queries as q
from app.core import worker_protocol as wp
from app.core.port_stats import PortStatsCache, compute_port_stats
from app.core.run_output_store import CaptureAlias, RunOutputStore
from app.core.worker_outputs import (
    RemoteAnswer,
    WorkerOutputs,
    WorkerTimeout,
    WorkerUnreachable,
)
from app.main import app


# ── the store retains a worker's runs by id ──────────────────────────────


async def test_a_worker_run_takes_a_place_in_the_retention_order():
    dropped: list[tuple[str, str]] = []
    store = RunOutputStore(max_runs=2, max_bytes=0)
    store.on_remote_drop = lambda run, worker: dropped.append((run, worker))
    await store.add_remote("w1", "box")
    await store.put("l1", "n", "p", 1)
    assert await store.remote_holder("w1") == "box"
    assert not await store.has_run("w1")  # nothing of it is held here
    await store.put("l2", "n", "p", 2)  # a third run: the oldest goes
    assert dropped == [("w1", "box")]
    assert await store.remote_holder("w1") is None
    assert await store.list_runs() == ["l1", "l2"]
    await store.add_remote("w2", "box")  # evicts l1, a local run: no call
    assert dropped == [("w1", "box")]
    assert await store.list_runs() == ["l2", "w2"]


async def test_adding_a_worker_run_again_makes_it_newest():
    store = RunOutputStore(max_runs=5, max_bytes=0)
    await store.add_remote("w1", "box")
    await store.add_remote("w2", "box")
    await store.add_remote("w1", "box")
    assert await store.list_runs() == ["w2", "w1"]
    await store.put("w1", "n", "p", 1)  # captures here win over the worker
    assert await store.remote_holder("w1") is None
    assert await store.list_runs() == ["w2", "w1"]
    await store.add_remote("w1", "other")  # a run held here is left alone
    assert await store.remote_holder("w1") is None


async def test_the_byte_budget_skips_worker_runs():
    """A worker's run holds no bytes here: evicting it would free none."""
    dropped: list[str] = []
    store = RunOutputStore(max_runs=10, max_bytes=100)
    store.on_remote_drop = lambda run, worker: dropped.append(run)
    await store.add_remote("w1", "box")
    await store.put("l1", "n", "p", torch.zeros(20))   # 80 bytes
    await store.put("l2", "n", "p", torch.zeros(20))   # over: l1 goes
    assert dropped == []
    assert await store.list_runs() == ["w1", "l2"]


async def test_a_failing_drop_callback_does_not_fail_the_put(caplog):
    store = RunOutputStore(max_runs=1, max_bytes=0)

    def boom(run: str, worker: str) -> None:
        raise RuntimeError("socket gone")

    store.on_remote_drop = boom
    await store.add_remote("w1", "box")
    await store.put("l1", "n", "p", 1)
    assert await store.list_runs() == ["l1"]
    assert "could not tell worker box" in caplog.text


async def test_an_evicted_worker_run_is_remembered_as_expired(monkeypatch):
    from app.core import run_output_store as module

    told: list[str] = []
    store = RunOutputStore(max_runs=1, max_bytes=0)
    store.on_remote_drop = lambda run, worker: told.append(run)
    await store.add_remote("w1", "box")
    await store.add_remote("w2", "box")
    assert told == ["w1"] and await store.was_evicted("w1")
    assert not await store.was_evicted("w2")
    # Retained again (a reconnect's ``_retain_outputs``): no longer expired.
    await store.add_remote("w1", "box")
    assert not await store.was_evicted("w1") and await store.was_evicted("w2")
    # Deleted is not evicted, and the memory is bounded.
    await store.delete_run("w1")
    assert not await store.was_evicted("w1")
    monkeypatch.setattr(module, "EXPIRED_REMEMBERED", 1)
    await store.add_remote("w3", "box")
    await store.add_remote("w4", "box")
    assert await store.was_evicted("w3") and not await store.was_evicted("w2")
    await store.clear()
    assert not await store.was_evicted("w3")


async def test_deleting_a_worker_run_here_tells_its_worker():
    told: list[tuple[str, str]] = []
    store = RunOutputStore(max_runs=5, max_bytes=0)
    store.on_remote_drop = lambda run, worker: told.append((run, worker))
    await store.add_remote("w1", "box")
    assert await store.delete_run("w1") is True
    assert told == [("w1", "box")]
    assert await store.list_runs() == [] and not await store.was_evicted("w1")
    assert await store.delete_run("w1") is False


async def test_forget_remote_and_delete_do_not_tell_the_worker():
    told: list[str] = []
    store = RunOutputStore(max_runs=5, max_bytes=0)
    store.on_remote_drop = lambda run, worker: told.append(run)
    await store.add_remote("w1", "box")
    assert await store.forget_remote("w1") is True
    assert await store.forget_remote("w1") is False
    assert told == [] and await store.list_runs() == []
    await store.add_remote("w2", "box")
    await store.clear()
    assert await store.remote_holder("w2") is None


async def test_snapshot_lists_every_slot_in_order():
    store = RunOutputStore(max_runs=5, max_bytes=0)
    await store.put("r", "a", "out", torch.zeros(4))
    await store.put("r", "b", "out", "text")
    await store.put("r", "a", "other", CaptureAlias("b", "out"))
    rows = await store.snapshot("r")
    assert [(n, p) for n, p, *_ in rows] == [("a", "out"), ("a", "other"),
                                              ("b", "out")]
    assert rows[0][4] == 16
    assert rows[0][3] < rows[2][3] or rows[0][3] != rows[2][3]
    assert await store.snapshot("nope") == []


# ── portable values and the disk form ────────────────────────────────────


def test_portable_keeps_data_and_stubs_the_rest():
    big = torch.arange(100.0)
    view = big[:3]
    model = torch.nn.Linear(2, 3)
    value = {"t": view, "n": [1, "a", None, (2.5, True)], "m": model,
             "s": {1, 2}, 3: {"x": torch.ones(2, requires_grad=True)}}
    out = q.portable(value)
    assert out["t"].untyped_storage().nbytes() == 3 * 4  # not the whole 100
    assert out["n"] == [1, "a", None, (2.5, True)]
    assert out[3]["x"].requires_grad is False
    back = q.restore(out)
    assert back["m"] == q.ModuleStub("Linear", 9, 9, repr(model))
    assert back["s"] == q.OpaqueStub("set", repr({1, 2}))
    # a dict with a key that is not a primitive is opaque as a whole
    odd = q.restore(q.portable({(1, 2): 1}))
    assert odd == q.OpaqueStub("dict", repr({(1, 2): 1}))
    # as is a dict that looks like a stub
    assert isinstance(q.restore(q.portable({"__codefyui_stub__": 1})),
                      q.OpaqueStub)


def test_stubs_read_like_the_values_they_stand_for():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    stub = q.restore(q.portable(model))
    assert q.serialize_value(stub, "", 10) == q.serialize_value(model, "", 10)
    assert q.type_label(stub) == q.type_label(model) == "model"
    assert q.stub_stats(stub) == compute_port_stats(model)
    raw = {1, 2}
    opaque = q.restore(q.portable(raw))
    assert q.serialize_value(opaque, "", 10) == q.serialize_value(raw, "", 10)
    assert q.type_label(opaque) == q.type_label(raw) == "set"
    assert q.stub_stats(opaque) == compute_port_stats(raw)
    assert q.stub_stats(3) is None


async def test_a_dump_reads_back_like_the_store(tmp_path):
    store = RunOutputStore(max_runs=5, max_bytes=0)
    await store.put("r", "x", "tensor", torch.arange(12.0).reshape(3, 4))
    await store.put("r", "x", "none", None)
    await store.put("r", "card", "tensor", CaptureAlias("x", "tensor"))
    await store.put("r", "bad", "alias", CaptureAlias("card", "tensor"))
    await store.put("r", "s", "__step__0__meta", {"name": "a"})
    index = await q.dump_run(store, "r", tmp_path / "outputs")
    measured = sum(row[4] for row in await store.snapshot("r"))
    assert index["run_id"] == "r" and index["bytes"] == measured > 0
    disk = q.DiskRunReader(tmp_path / "outputs")
    assert await disk.has_run("r") and not await disk.has_run("other")
    assert await disk.list_ports("r") == await store.list_ports("r")
    assert await disk.list_ports("other") is None
    for op, params in [("list", {}),
                       ("value", {"node_id": "card", "port": "tensor",
                                  "slice": "1"}),
                       ("value", {"node_id": "bad", "port": "alias"}),
                       ("value", {"node_id": "x", "port": "none"}),
                       ("stats", {"node_id": "x", "port": "tensor"}),
                       ("steps", {"node_id": "s"}),
                       ("grads", {"node_id": "x"}),
                       ("value", {"node_id": "x", "port": "gone"})]:
        assert await q.dispatch(disk, op, "r", params, compute=_compute) == \
            await q.dispatch(store, op, "r", params, compute=_compute), op
    assert await disk.get("r", "x", "missing") is None
    assert await disk.get_with_version("other", "x", "tensor") is None


async def test_no_index_holds_nothing(tmp_path):
    assert q.read_index(tmp_path) is None
    (tmp_path / q.INDEX_FILE).write_text("[1, 2]")
    assert q.read_index(tmp_path) is None
    (tmp_path / q.INDEX_FILE).write_text("{not json")
    disk = q.DiskRunReader(tmp_path)
    assert await disk.has_run("r") is False
    assert await q.dispatch(disk, "list", "r", {}, compute=_compute) == (
        404, {"detail": "run 'r' not found"})


async def _compute(value: Any) -> dict[str, Any]:
    return compute_port_stats(value)


async def test_dispatch_answers_every_outcome():
    store = RunOutputStore(max_runs=5, max_bytes=0)
    await store.put("r", "n", "t", torch.ones(3))
    assert (await q.dispatch(store, "nope", "r", {}, compute=_compute))[0] \
        == 400
    status, body = await q.dispatch(store, "value", "r",
                                    {"node_id": "n", "port": "t",
                                     "max_elements": 1}, compute=_compute)
    assert status == 413 and "max 1" in body["detail"]

    async def broken(value: Any) -> dict[str, Any]:
        raise ValueError("boom")

    status, body = await q.dispatch(store, "stats", "r",
                                    {"node_id": "n", "port": "t"},
                                    compute=broken)
    assert status == 500 and "ValueError: boom" in body["detail"]
    assert json.loads(q.encode_body({"v": float("nan")})) == {"v": None}


# ── reads in flight to a worker ──────────────────────────────────────────


@dataclass(eq=False)
class _FakeWorker:
    name: str = "box"
    online: bool = True
    capabilities: tuple[str, ...] = tuple(wp.CAPABILITIES)
    sent: list[Any] = field(default_factory=list)
    fail: bool = False

    async def send(self, frame: Any) -> None:
        if self.fail:
            raise RuntimeError("closed")
        self.sent.append(frame)


class _FakeRegistry:
    def __init__(self, *workers: _FakeWorker) -> None:
        self.workers = {w.name: w for w in workers}

    def get(self, name: str) -> Any:
        return self.workers.get(name)


async def _answer_when_sent(outputs: WorkerOutputs, worker: _FakeWorker,
                            parts: list[str], status: int = 200,
                            final: bool = True) -> None:
    while not worker.sent:
        await asyncio.sleep(0)
    request = worker.sent[-1]
    for number, body in enumerate(parts):
        outputs.deliver(worker.name, wp.OutputAnswer(
            request_id=request.request_id, status=status, part=number,
            parts=len(parts), body=body, final=final))


async def test_a_fetch_reassembles_its_parts():
    worker = _FakeWorker()
    outputs = WorkerOutputs(_FakeRegistry(worker), timeout_s=5)
    helper = asyncio.ensure_future(_answer_when_sent(
        outputs, worker, ['{"a": ', '[1, 2]', '}']))
    answer = await outputs.fetch("box", "r1", "value", {"node_id": "n"})
    await helper
    assert answer == RemoteAnswer(200, {"a": [1, 2]}, True)
    assert worker.sent[0].params == {"node_id": "n"}
    assert outputs.in_flight() == 0


async def test_an_empty_answer_is_a_204():
    worker = _FakeWorker()
    outputs = WorkerOutputs(_FakeRegistry(worker), timeout_s=5)
    helper = asyncio.ensure_future(_answer_when_sent(outputs, worker, [""],
                                                     status=204))
    answer = await outputs.fetch("box", "r1", "value")
    await helper
    assert (answer.status, answer.body) == (204, None)


@pytest.mark.parametrize("worker, needle", [
    (None, "has not connected since this server started"),
    (_FakeWorker(online=False), "is not connected"),
    (_FakeWorker(capabilities=("ack",)), "cannot serve them"),
    (_FakeWorker(fail=True), "could not be sent the request"),
])
async def test_a_worker_that_cannot_be_asked_is_unreachable_at_once(
        worker, needle):
    registry = _FakeRegistry(*([worker] if worker else []))
    outputs = WorkerOutputs(registry, timeout_s=30)
    with pytest.raises(WorkerUnreachable, match=needle) as raised:
        await asyncio.wait_for(outputs.fetch("box", "r1", "list"), 2)
    assert "worker 'box'" in str(raised.value)
    assert outputs.in_flight() == 0


async def test_a_silent_worker_times_out():
    outputs = WorkerOutputs(_FakeRegistry(_FakeWorker()), timeout_s=0.05)
    with pytest.raises(WorkerTimeout, match="did not answer within 0.05 s"):
        await outputs.fetch("box", "r1", "list")
    assert outputs.in_flight() == 0


async def test_a_disconnect_ends_the_reads_in_flight():
    worker = _FakeWorker()
    outputs = WorkerOutputs(_FakeRegistry(worker, _FakeWorker(name="other")),
                            timeout_s=30)
    read = asyncio.ensure_future(outputs.fetch("box", "r1", "list"))
    other = asyncio.ensure_future(outputs.fetch("other", "r2", "list"))
    while outputs.in_flight() < 2:
        await asyncio.sleep(0)
    outputs.worker_gone("box", "the connection closed")
    with pytest.raises(WorkerUnreachable, match="disconnected"):
        await asyncio.wait_for(read, 2)
    assert not other.done()
    other.cancel()


async def test_stray_and_malformed_answers():
    worker = _FakeWorker()
    outputs = WorkerOutputs(_FakeRegistry(worker), timeout_s=5)
    outputs.deliver("box", wp.OutputAnswer(request_id="nobody", status=200))
    read = asyncio.ensure_future(outputs.fetch("box", "r1", "list"))
    while not worker.sent:
        await asyncio.sleep(0)
    rid = worker.sent[0].request_id
    # another worker answering this request is ignored
    outputs.deliver("other", wp.OutputAnswer(request_id=rid, status=200))
    outputs.deliver("box", wp.OutputAnswer(request_id=rid, status=200,
                                           part=0, parts=2, body="{"))
    outputs.deliver("box", wp.OutputAnswer(request_id=rid, status=200,
                                           part=1, parts=3, body="}"))
    with pytest.raises(WorkerUnreachable, match="malformed"):
        await read
    read = asyncio.ensure_future(outputs.fetch("box", "r1", "list"))
    while len(worker.sent) < 2:
        await asyncio.sleep(0)
    outputs.deliver("box", wp.OutputAnswer(
        request_id=worker.sent[1].request_id, status=200, body="{nope"))
    with pytest.raises(WorkerUnreachable, match="not JSON"):
        await read


async def test_a_fetch_says_its_wait_and_cancels_what_it_gives_up():
    worker = _FakeWorker()
    outputs = WorkerOutputs(_FakeRegistry(worker), timeout_s=0.05)
    with pytest.raises(WorkerTimeout):
        await outputs.fetch("box", "r1", "list")
    for _ in range(5):
        await asyncio.sleep(0)
    fetch, cancel = worker.sent
    assert fetch.timeout_s == 0.05
    assert cancel == wp.CancelFetch(request_id=fetch.request_id)
    # The browser going away cancels the read on the worker too.
    worker.sent.clear()
    read = asyncio.ensure_future(outputs.fetch("box", "r1", "list",
                                               timeout_s=30))
    while not worker.sent:
        await asyncio.sleep(0)
    read.cancel()
    with pytest.raises(asyncio.CancelledError):
        await read
    for _ in range(5):
        await asyncio.sleep(0)
    assert isinstance(worker.sent[-1], wp.CancelFetch)
    # An answered read cancels nothing.
    worker.sent.clear()
    helper = asyncio.ensure_future(_answer_when_sent(outputs, worker, ["1"]))
    outputs.timeout_s = 5
    await outputs.fetch("box", "r1", "list")
    await helper
    for _ in range(5):
        await asyncio.sleep(0)
    assert [type(f) for f in worker.sent] == [wp.FetchOutput]


async def test_drops_for_an_absent_worker_wait_for_its_next_connection(
        monkeypatch):
    from app.core import worker_outputs as module

    away = _FakeWorker(name="away", online=False)
    old = _FakeWorker(name="old", capabilities=("ack",), online=False)
    outputs = WorkerOutputs(_FakeRegistry(away, old))
    outputs.drop("r1", "away")
    outputs.drop("r2", "away")
    outputs.drop("r1", "away")  # once is enough
    outputs.drop("r9", "nobody")
    assert outputs.pending_drops("away") == ["r1", "r2"]
    assert outputs.flush_drops("away") == 0  # still not connected
    assert outputs.pending_drops("away") == ["r1", "r2"]
    away.online = True
    assert outputs.flush_drops("away") == 2
    for _ in range(5):
        await asyncio.sleep(0)
    assert [f.run_id for f in away.sent] == ["r1", "r2"]
    assert outputs.pending_drops("away") == []
    assert outputs.flush_drops("away") == 0
    # An agent that cannot serve outputs has nothing to drop.
    outputs.drop("r3", "old")
    old.online = True
    assert outputs.flush_drops("old") == 0 and outputs.pending_drops("old") == []
    # Bounded: the oldest go first.
    monkeypatch.setattr(module, "PENDING_DROPS_MAX", 2)
    away.online = False
    for run_id in ("a", "b", "c"):
        outputs.drop(run_id, "away")
    assert outputs.pending_drops("away") == ["b", "c"]
    assert outputs.pending_drops("nobody") == ["r9"]


async def test_drop_tells_a_connected_worker_only():
    worker = _FakeWorker()
    away = _FakeWorker(name="away", online=False)
    failing = _FakeWorker(name="failing", fail=True)
    outputs = WorkerOutputs(_FakeRegistry(worker, away, failing))
    outputs.drop("r1", "box")
    outputs.drop("r2", "away")
    outputs.drop("r3", "nobody")
    outputs.drop("r4", "failing")
    for _ in range(5):
        await asyncio.sleep(0)
    assert [f.run_id for f in worker.sent] == ["r1"]
    assert isinstance(worker.sent[0], wp.DropOutputs)
    assert away.sent == [] and failing.sent == []


def test_drop_without_a_loop_is_a_no_op():
    outputs = WorkerOutputs(_FakeRegistry(_FakeWorker()))
    outputs.drop("r1", "box")  # no running loop: nothing is sent


async def test_the_registry_ends_reads_when_a_worker_goes():
    from app.core.workers import WorkerRegistry

    registry = WorkerRegistry(output_fetch_timeout_s=30)

    async def send(frame: Any) -> None:
        return None

    async def close() -> None:
        return None

    hello = wp.Hello(name="box", version="t",
                     capabilities=list(wp.CAPABILITIES))
    worker, _ = registry.connect(hello, send=send, close=close)
    read = asyncio.ensure_future(registry.outputs.fetch("box", "r", "list"))
    while registry.outputs.in_flight() < 1:
        await asyncio.sleep(0)
    # A replacing connection: the old socket's reads end.
    registry.connect(hello, send=send, close=close)
    with pytest.raises(WorkerUnreachable, match="replaced"):
        await asyncio.wait_for(read, 2)
    read = asyncio.ensure_future(registry.outputs.fetch("box", "r", "list"))
    while registry.outputs.in_flight() < 1:
        await asyncio.sleep(0)
    registry.disconnect(registry.get("box"))
    with pytest.raises(WorkerUnreachable, match="the connection closed"):
        await asyncio.wait_for(read, 2)
    await registry.shutdown()


# ── the routes, for a worker run ─────────────────────────────────────────


class _FakeOutputs:
    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, str, str, dict[str, Any]]] = []

    async def fetch(self, worker: str, run_id: str, op: str,
                    params: dict[str, Any]) -> RemoteAnswer:
        self.calls.append((worker, run_id, op, params))
        answer = self.answers[op]
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def worker_app(monkeypatch):
    """The app with a fresh store, a fake registry and a run record."""
    store = RunOutputStore(max_runs=5, max_bytes=0)
    outputs = _FakeOutputs({})
    records: dict[str, Any] = {}

    async def get_run(run_id: str) -> Any:
        record = records.get(run_id)
        if isinstance(record, Exception):
            raise record
        return record

    monkeypatch.setattr(app.state, "run_output_store", store, raising=False)
    monkeypatch.setattr(app.state, "port_stats_cache",
                        PortStatsCache(max_bytes=1 << 20), raising=False)
    monkeypatch.setattr(app.state, "worker_registry",
                        SimpleNamespace(outputs=outputs), raising=False)
    monkeypatch.setattr(app.state, "run_service", SimpleNamespace(
        store=SimpleNamespace(get_run=get_run)), raising=False)
    return SimpleNamespace(store=store, outputs=outputs, records=records)


def _record(worker: str = "box", record: bool = True) -> Any:
    return SimpleNamespace(worker_id=worker,
                           options={"record_outputs": record})


async def test_every_read_of_a_worker_run_is_forwarded(test_client,
                                                       worker_app):
    await worker_app.store.add_remote("r1", "box")
    ok = RemoteAnswer(200, {"type": "tensor"}, False)
    worker_app.outputs.answers.update(
        list=RemoteAnswer(200, [], False), value=ok, stats=ok, steps=ok,
        grads=ok)
    base = "/api/execution/outputs/r1"
    reads = [
        (base, ("list", {})),
        (f"{base}/n/out?slice=0&max_elements=7&preview=false",
         ("value", {"node_id": "n", "port": "out", "slice": "0",
                    "max_elements": 7, "preview": False})),
        (f"{base}/value?node_id=b/n&port=out",
         ("value", {"node_id": "b/n", "port": "out", "slice": "",
                    "max_elements": 4096, "preview": False})),
        (f"{base}/n/out/stats", ("stats", {"node_id": "n", "port": "out"})),
        (f"{base}/n/__steps_index?inner=true",
         ("steps", {"node_id": "n", "inner": True})),
        (f"{base}/n/__grad_index", ("grads", {"node_id": "n",
                                              "inner": False})),
    ]
    for url, (op, params) in reads:
        response = await test_client.get(url)
        assert response.status_code == 200, (url, response.text)
        assert worker_app.outputs.calls[-1] == ("box", "r1", op, params)


async def test_worker_answers_become_responses(test_client, worker_app):
    await worker_app.store.add_remote("r1", "box")
    answers = worker_app.outputs.answers
    url = "/api/execution/outputs/r1/n/out"
    answers["value"] = RemoteAnswer(204, None, True)
    assert (await test_client.get(url)).status_code == 204
    answers["value"] = RemoteAnswer(413, {"detail": "too big"}, True)
    response = await test_client.get(url)
    assert (response.status_code, response.json()["detail"]) == (413,
                                                                 "too big")
    answers["value"] = RemoteAnswer(500, ["not a dict"], True)
    response = await test_client.get(url)
    assert response.json()["detail"] == "the worker refused the read"
    answers["value"] = WorkerUnreachable("worker 'box' is not connected")
    response = await test_client.get(url)
    assert response.status_code == 503
    assert response.json()["detail"] == "worker 'box' is not connected"
    answers["value"] = WorkerTimeout("worker 'box' did not answer")
    assert (await test_client.get(url)).status_code == 504


async def test_stats_of_an_ended_worker_run_are_cached(test_client,
                                                       worker_app):
    await worker_app.store.add_remote("r1", "box")
    url = "/api/execution/outputs/r1/n/out/stats"
    worker_app.outputs.answers["stats"] = RemoteAnswer(200, {"k": 1}, False)
    await test_client.get(url)
    await test_client.get(url)
    assert len(worker_app.outputs.calls) == 2  # live: not cached
    worker_app.outputs.answers["stats"] = RemoteAnswer(200, {"k": 2}, True)
    await test_client.get(url)
    response = await test_client.get(url)
    assert response.json() == {"k": 2}
    assert len(worker_app.outputs.calls) == 3


async def test_after_a_restart_the_record_names_the_worker(test_client,
                                                           worker_app):
    worker_app.records["r1"] = _record()
    worker_app.outputs.answers["list"] = RemoteAnswer(404, {
        "detail": "run 'r1' not found"}, True)
    response = await test_client.get("/api/execution/outputs/r1")
    assert response.status_code == 404
    assert await worker_app.store.remote_holder("r1") is None  # not found
    worker_app.outputs.answers["list"] = RemoteAnswer(200, [], True)
    assert (await test_client.get("/api/execution/outputs/r1")).json() == []
    assert await worker_app.store.remote_holder("r1") == "box"


@pytest.mark.parametrize("record", [
    None, _record(worker="local"), _record(worker="any"),
    _record(record=False), RuntimeError("db gone")])
async def test_a_run_no_worker_holds_reads_as_before(test_client,
                                                     worker_app, record):
    worker_app.records["r1"] = record
    response = await test_client.get("/api/execution/outputs/r1")
    assert (response.status_code, response.json()) == (
        404, {"detail": "run 'r1' not found"})
    assert worker_app.outputs.calls == []


async def test_without_a_run_service_or_registry(test_client, worker_app,
                                                 monkeypatch):
    monkeypatch.setattr(app.state, "run_service", None)
    assert (await test_client.get("/api/execution/outputs/r1")
            ).status_code == 404
    await worker_app.store.add_remote("r1", "box")
    monkeypatch.setattr(app.state, "worker_registry", None)
    response = await test_client.get("/api/execution/outputs/r1")
    assert response.status_code == 503
    assert "no worker registry" in response.json()["detail"]


async def test_deleting_a_worker_run_forgets_it_either_way(test_client,
                                                           worker_app):
    await worker_app.store.add_remote("r1", "box")
    worker_app.outputs.answers["delete"] = RemoteAnswer(
        200, {"run_id": "r1", "deleted": True}, True)
    response = await test_client.delete("/api/execution/outputs/r1")
    assert response.json() == {"run_id": "r1", "deleted": True}
    assert await worker_app.store.remote_holder("r1") is None
    await worker_app.store.add_remote("r2", "box")
    worker_app.outputs.answers["delete"] = WorkerUnreachable("gone")
    response = await test_client.delete("/api/execution/outputs/r2")
    assert response.status_code == 503
    assert await worker_app.store.remote_holder("r2") is None


async def test_an_evicted_worker_run_is_not_found_again(test_client,
                                                        worker_app):
    """Evicted while its worker was away, a run's record still names its
    worker: a later read answers as for any run not held, and does not
    retain it again (which would evict another run)."""
    store = worker_app.store
    worker_app.records["w0"] = _record()
    await store.add_remote("w0", "box")
    for number in range(1, 6):
        await store.add_remote(f"w{number}", "box")
    assert await store.was_evicted("w0")
    worker_app.outputs.answers["list"] = RemoteAnswer(200, [], True)
    response = await test_client.get("/api/execution/outputs/w0")
    assert (response.status_code, response.json()) == (
        404, {"detail": "run 'w0' not found"})
    response = await test_client.get("/api/execution/outputs/w0/n/p/stats")
    assert response.status_code == 404
    assert "Record outputs" in response.json()["detail"]
    assert worker_app.outputs.calls == []
    assert await store.list_runs() == [f"w{n}" for n in range(1, 6)]


async def test_a_run_held_here_never_asks_a_worker(test_client, worker_app):
    await worker_app.store.put("r1", "n", "out", 3)
    worker_app.records["r1"] = _record()
    response = await test_client.get("/api/execution/outputs/r1/n/out")
    assert response.json()["value"] == 3
    assert worker_app.outputs.calls == []


# ── WorkerExecutor and RunService take part ──────────────────────────────


async def test_the_executor_retains_a_recorded_run_and_refuses_old_agents():
    from app.core.executors import ExecutionRequest, WorkerExecutor
    from app.core.scheduling import Placement

    store = RunOutputStore(max_runs=5, max_bytes=0)
    old = SimpleNamespace(online=True, capabilities=("ack",),
                          node_types=None)

    class Registry:
        resume_grace_s = 0

        def get(self, name: str) -> Any:
            return old

        def claim_adopted(self, run_id: str) -> Any:
            return None

    executor = WorkerExecutor(Registry())
    executor._catalog_refusal = _no_refusal  # type: ignore[method-assign]

    def request(**extra: Any) -> ExecutionRequest:
        return ExecutionRequest(
            run_id="r1", graph={"nodes": []}, options={},
            context=SimpleNamespace(should_stop=lambda: False),
            session=None, output_store=store, record_outputs=True,
            sinks=SimpleNamespace(warn=_ignore, emit=_ignore),
            explicit_metric_nodes=set(),
            placement=Placement(worker_id="old", devices=("cpu",), slots=()),
            **extra)

    outcome = await executor.run(request())
    assert outcome.status == "failed"
    assert "cannot record outputs" in outcome.error
    assert await store.remote_holder("r1") is None
    # An adopted run (#793) is retained again before it waits.
    outcome = await executor.run(request(resume_seq=3))
    assert await store.remote_holder("r1") == "old"


async def _no_refusal(*_: Any) -> None:
    return None


async def _ignore(*_: Any, **__: Any) -> None:
    return None
