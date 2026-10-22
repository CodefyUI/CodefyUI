"""``WorkerScheduler``: placement across this process and its workers (#772)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from app.core import worker_protocol as wp
from app.core.run_service import (
    QueueLimits,
    RunService,
    RunSubmitError,
    worker_not_connected,
)
from app.core.scheduling import (
    LocalScheduler,
    Placement,
    WorkerScheduler,
    candidate_devices,
)
from app.core.workers import Worker, WorkerRegistry


@dataclass
class _Entry:
    run_id: str
    queue_key: str
    options: dict[str, Any] = field(default_factory=dict)


async def _noop(*_: Any) -> None:
    return None


def _worker(name: str, *devices: str, draining: bool = False,
            status: str = "online") -> Worker:
    return Worker(
        id=name, name=name, version="t", fingerprint=None, platform={},
        mounts={}, devices=[wp.DeviceReport(value=d, kind=d.split(":")[0])
                            for d in devices],
        send=_noop, close=_noop, heartbeat_s=10, status=status,
        draining=draining)


class _Registry:
    def __init__(self, *workers: Worker) -> None:
        self._workers = {w.id: w for w in workers}

    def get(self, worker_id: str) -> Worker | None:
        return self._workers.get(worker_id)

    def workers(self) -> list[Worker]:
        return sorted(self._workers.values(), key=lambda w: w.name)


LIMITS = QueueLimits()


def _scheduler(*workers: Worker, here: tuple = ("cpu",)) -> WorkerScheduler:
    return WorkerScheduler(LIMITS, _Registry(*workers), lambda: list(here))


def _target(worker: str, device: str = "cpu",
            devices: list[str] | None = None) -> dict[str, Any]:
    return {"device": device, "target": {"worker": worker,
                                         "devices": devices}}


def test_without_a_target_it_is_the_local_scheduler():
    entry = _Entry("r1", "cuda:0", {"device": "cuda:0"})
    scheduler = _scheduler(_worker("box", "cpu", "cuda:0"))
    for taken in ({}, {("local", "cuda:0"): 1}):
        assert scheduler.place(entry, taken) == LocalScheduler(LIMITS).place(
            entry, taken)
    local = _Entry("r2", "cpu", {"device": "cpu",
                                 "target": {"worker": "local",
                                            "devices": None}})
    assert scheduler.place(local, {}) == Placement(
        "local", ("cpu",), (("local", "cpu"),))


def test_a_named_worker_gets_its_device_while_the_slot_is_free():
    scheduler = _scheduler(_worker("box", "cpu", "cuda:0", "cuda:1"))
    entry = _Entry("r1", "box/cuda:1", _target("box", "cuda:1"))
    assert scheduler.place(entry, {}) == Placement(
        "box", ("cuda:1",), (("box", "cuda:1"),))
    assert scheduler.place(entry, {("box", "cuda:1"): 1}) is None
    # The same card number on another worker, or here, is another slot.
    assert scheduler.place(entry, {("local", "cuda:1"): 1,
                                   ("other", "cuda:1"): 1}) is not None


def test_a_bare_kind_takes_the_first_free_card_and_auto_prefers_cuda():
    scheduler = _scheduler(_worker("box", "cpu", "cuda:0", "cuda:1"))
    entry = _Entry("r1", "box/cuda", _target("box", "cuda"))
    assert scheduler.place(entry, {("box", "cuda:0"): 1}).devices == (
        "cuda:1",)
    auto = _Entry("r2", "box/auto", _target("box", "auto"))
    assert scheduler.place(auto, {}).devices == ("cuda:0",)
    assert scheduler.place(auto, {("box", "cuda:0"): 1,
                                  ("box", "cuda:1"): 1}).devices == ("cpu",)


def test_a_device_set_holds_every_slot_or_waits():
    scheduler = _scheduler(_worker("box", "cuda:0", "cuda:1"))
    entry = _Entry("r1", "box/cuda:0",
                   _target("box", "cuda:0", ["cuda:0", "cuda:1"]))
    assert scheduler.place(entry, {}) == Placement(
        "box", ("cuda:0", "cuda:1"), (("box", "cuda:0"), ("box", "cuda:1")))
    assert scheduler.place(entry, {("box", "cuda:1"): 1}) is None


@pytest.mark.parametrize("worker", [
    _worker("box", "cpu", draining=True),
    _worker("box", "cpu", status="disconnected"),
])
def test_a_draining_or_gone_worker_takes_nothing(worker):
    scheduler = _scheduler(worker)
    assert scheduler.place(_Entry("r1", "box/cpu", _target("box")), {}) is None
    assert scheduler.place(_Entry("r1", "nobody/cpu", _target("nobody")),
                           {}) is None


def test_any_tries_workers_in_name_order_then_this_process():
    scheduler = _scheduler(_worker("b-box", "cpu"), _worker("a-box", "cpu"),
                           _worker("c-box", "cpu", draining=True))
    entry = _Entry("r1", "any/cpu", _target("any"))
    assert scheduler.place(entry, {}).worker_id == "a-box"
    full = {("a-box", "cpu"): 2, ("b-box", "cpu"): 2}
    assert scheduler.place(entry, full) == Placement(
        "local", ("cpu",), (("local", "cpu"),))
    assert scheduler.place(entry, {**full, ("local", "cpu"): 2}) is None


def test_any_with_a_renumbered_cuda_set_never_runs_here():
    """A CUDA set is renumbered from cuda:0 on a worker
    (CUDA_VISIBLE_DEVICES), not in this process, so node pins would address
    other cards here: such an ``any`` run waits for a worker even when this
    server has every card of the set."""
    entry = _Entry("r1", "any/cuda:1",
                   _target("any", "cuda:1", ["cuda:1", "cuda:0"]))
    for here in (("cpu", "cuda:1"), ("cpu", "cuda:0", "cuda:1")):
        scheduler = _scheduler(here=here)
        assert scheduler.place(entry, {}) is None
        assert scheduler.reservation(entry) is None
        assert not scheduler.can_serve_any("cuda:1", ["cuda:1", "cuda:0"])
        assert scheduler.explain(entry) == (
            "waiting for a worker that offers cuda:1, cuda:0")
    # One card other than cuda:0 is renumbered too (a worker calls it cuda:0).
    one = _Entry("r2", "any/cuda:1", _target("any", "cuda:1", ["cuda:1"]))
    assert _scheduler(here=("cpu", "cuda:1")).place(one, {}) is None
    # Without a set, this process still qualifies.
    assert _scheduler(here=("cpu", "cuda:1")).can_serve_any("cuda:1", None)
    # A worker with the set takes it, in the order the set lists it.
    with_box = _scheduler(_worker("box", "cpu", "cuda:0", "cuda:1"),
                          here=("cpu", "cuda:0", "cuda:1"))
    assert with_box.place(entry, {}) == Placement(
        "box", ("cuda:1", "cuda:0"), (("box", "cuda:1"), ("box", "cuda:0")))


@pytest.mark.parametrize("devices,first", [
    (["cuda:0"], "cuda:0"),
    (["mps:0"], "mps:0"),
    (["cpu"], "cpu"),
])
def test_any_with_a_set_nothing_renumbers_may_run_here(devices, first):
    """``cuda:0`` alone, ``mps`` and ``cpu`` mean the same card here and on
    a worker, so this server still qualifies for them."""
    here = ("cpu", "cuda:0", "mps:0")
    entry = _Entry("r1", f"any/{first}", _target("any", first, devices))
    scheduler = _scheduler(here=here)
    assert scheduler.can_serve_any(first, devices)
    assert scheduler.place(entry, {}) == Placement(
        "local", (first,), (("local", first),))
    assert scheduler.place(entry, {("local", first): 2}) is None


def test_renumbered_on_a_worker():
    from app.core.scheduling import renumbered_on_a_worker

    for devices in (None, [], ["cuda:0"], ["cpu"], ["mps"], ["mps:0"]):
        assert not renumbered_on_a_worker(devices), devices
    for devices in (["cuda:1"], ["cuda:0", "cuda:1"], ["cuda:1", "cuda:0"]):
        assert renumbered_on_a_worker(devices), devices


def test_explain_names_the_worker_a_run_waits_for():
    scheduler = _scheduler(_worker("box", "cpu", draining=True),
                           _worker("gone", "cpu", status="disconnected"))
    assert scheduler.explain(_Entry("r", "box/cpu", _target("box"))) == (
        "worker box is draining")
    for name in ("gone", "nobody"):
        assert scheduler.explain(_Entry("r", f"{name}/cpu", _target(name))) \
            == f"waiting for worker {name} to reconnect"
    assert scheduler.explain(_Entry("r", "cpu", {"device": "cpu"})) is None


def test_candidate_devices():
    offered = ["cpu", "cuda:0", "cuda:1", "mps:0"]
    assert candidate_devices(offered, "cuda:1") == ["cuda:1"]
    assert candidate_devices(offered, "cuda:7") == []
    assert candidate_devices(offered, "cuda") == ["cuda:0", "cuda:1"]
    assert candidate_devices(offered, "mps") == ["mps:0"]
    assert candidate_devices(offered, "CUDA:01") == ["cuda:1"]
    assert candidate_devices(offered, "tpu") == []
    assert candidate_devices(offered, "auto") == ["cuda:0", "cuda:1",
                                                  "mps:0", "cpu"]


# ── submit-time refusals (RunService._submit_queue_key) ──────────────────


def _service(registry: Any = None) -> RunService:
    scheduler = (None if registry is None
                 else WorkerScheduler(LIMITS, registry, lambda: ["cpu"]))
    return RunService(store=None, workers=registry,  # type: ignore[arg-type]
                      scheduler=scheduler)


def _registry_with(*workers: Worker) -> WorkerRegistry:
    registry = WorkerRegistry()
    for worker in workers:
        registry._workers[worker.id] = worker
    return registry


def _options(**target: Any) -> dict[str, Any]:
    from app.core.run_service import normalize_options

    raw = {"target": target}
    if "device" in target:
        raw["device"] = target.pop("device")
    return normalize_options(raw)


def test_submit_keys_and_refusals():
    service = _service(_registry_with(_worker("box", "cpu", "cuda:0")))
    key = service._submit_queue_key
    assert key(_options(worker="box")) == "box/cpu"
    assert key(_options(worker="box", device="cuda")) == "box/cuda"
    assert key(_options(worker="box", devices=["cuda:0"])) == "box/cuda:0"
    assert key(_options(worker="any", device="cuda")) == "any/cuda"
    assert key(_options(worker="any", devices=["cuda:0"])) == "any/cuda:0"
    with pytest.raises(RunSubmitError, match="neither this server"):
        key(_options(worker="any", device="cuda:3"))
    with pytest.raises(RunSubmitError, match="no known worker offers the "
                                             "device set cuda:0, cuda:1"):
        key(_options(worker="any", devices=["cuda:0", "cuda:1"]))
    assert key(_options(worker="local")) == "cpu"
    with pytest.raises(RunSubmitError, match="has not connected"):
        key(_options(worker="nobody"))
    with pytest.raises(RunSubmitError, match="does not offer cuda:1"):
        key(_options(worker="box", devices=["cuda:1"]))
    with pytest.raises(RunSubmitError, match="does not offer mps"):
        key(_options(worker="box", device="mps"))
    # #789: a worker run may record outputs. ``box`` never said hello (no
    # capabilities known), so it is checked when the run starts instead.
    assert key({**_options(worker="box"), "record_outputs": True}) == \
        "box/cpu"
    assert key({**_options(worker="any"), "record_outputs": True}) == \
        "any/cpu"
    with pytest.raises(RunSubmitError, match="interactive"):
        key({**_options(worker="box"), "lane": "interactive"})
    assert worker_not_connected("nobody") in str(
        pytest.raises(RunSubmitError, key, _options(worker="nobody")).value)


def test_submit_refuses_an_any_set_only_this_server_could_take():
    """This server has both cards but never takes an ``any`` run whose CUDA
    set a worker would renumber; ``cuda:0`` alone it still takes."""
    service = RunService(
        store=None, workers=_registry_with(_worker("box", "cpu")),  # type: ignore[arg-type]
        scheduler=WorkerScheduler(LIMITS, _registry_with(_worker("box", "cpu")),
                                  lambda: ["cpu", "cuda:0", "cuda:1"]))
    key = service._submit_queue_key
    with pytest.raises(RunSubmitError, match="never on this server"):
        key(_options(worker="any", devices=["cuda:0", "cuda:1"]))
    with pytest.raises(RunSubmitError, match="never on this server"):
        key(_options(worker="any", devices=["cuda:1"]))
    assert key(_options(worker="any", devices=["cuda:0"])) == "any/cuda:0"
    assert key(_options(worker="any", device="cuda:1")) == "any/cuda:1"


def test_a_known_worker_that_is_away_is_waited_for_not_refused():
    """#777: one uniform rule. A worker this server has seen but that is
    disconnected now queues the run (it waits for the reconnect); only an
    unknown name or a device it never offered is refused."""
    away = _worker("box", "cpu", "cuda:0", status="disconnected")
    service = _service(_registry_with(away))
    key = service._submit_queue_key
    assert key(_options(worker="box", device="cuda")) == "box/cuda"
    assert key(_options(worker="any", device="cuda")) == "any/cuda"
    with pytest.raises(RunSubmitError, match="does not offer cuda:1"):
        key(_options(worker="box", devices=["cuda:1"]))
    with pytest.raises(RunSubmitError, match="neither this server"):
        key(_options(worker="any", device="cuda:1"))
    with pytest.raises(RunSubmitError, match="has not connected"):
        key(_options(worker="other"))
    # The waiting run's reason still describes the workers connected now.
    scheduler = _scheduler(away)
    entry = _Entry("r1", "any/cuda", _target("any", "cuda"))
    assert scheduler.can_serve_any("cuda", None)
    assert not scheduler.can_serve_any("cuda", None, connected_only=True)
    assert scheduler.explain(entry) == "waiting for a worker that offers cuda"
    assert scheduler.explain(_Entry("r2", "box/cuda", _target(
        "box", "cuda"))) == "waiting for worker box to reconnect"


def test_without_a_registry_a_named_worker_is_refused():
    with pytest.raises(RunSubmitError, match="has not connected"):
        _service()._submit_queue_key(_options(worker="box"))


def test_a_heartbeat_that_changes_the_devices_reschedules():
    """A card appearing in a heartbeat is capacity a waiting run may take."""
    worker = _worker("box", "cpu")
    registry = _registry_with(worker)
    calls: list[None] = []
    registry.add_listener(lambda: calls.append(None))
    same = [wp.DeviceReport(value="cpu", kind="cpu")]
    registry.heartbeat(worker, wp.Heartbeat(devices=same))
    assert calls == []
    grown = [*same, wp.DeviceReport(value="cuda:0", kind="cuda", index=0,
                                    uuid="abc", free_bytes=100)]
    registry.heartbeat(worker, wp.Heartbeat(devices=grown))
    assert calls == [None]
    assert worker.device_values() == ["cpu", "cuda:0"]
    # Memory figures move on every heartbeat: stored, but not a change.
    busier = [*same, wp.DeviceReport(value="cuda:0", kind="cuda", index=0,
                                     uuid="abc", free_bytes=10)]
    registry.heartbeat(worker, wp.Heartbeat(devices=busier))
    assert calls == [None]
    assert worker.devices[1].free_bytes == 10
    swapped = [*same, wp.DeviceReport(value="cuda:0", kind="cuda", index=0,
                                      uuid="def", free_bytes=10)]
    registry.heartbeat(worker, wp.Heartbeat(devices=swapped))
    assert calls == [None, None]
# ── the mount constraint (#782) ──────────────────────────────────────────

NEEDS_DATA = {"mount": "data", "files": ["train.csv"]}


@dataclass
class _NeedyEntry(_Entry):
    needs: dict[str, Any] | None = None


def _mounted(worker: Worker, **mounts: str) -> Worker:
    worker.mounts = dict(mounts)
    return worker


@pytest.fixture
def data_label(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "MOUNT_LABEL", "data")


def test_a_run_that_needs_the_mount_goes_only_to_a_worker_holding_it(
        data_label):
    bare = _worker("bare", "cpu")
    nas = _mounted(_worker("nas", "cpu"), data="/mnt/nas")
    scheduler = _scheduler(bare, nas)
    needy = _NeedyEntry("r1", "bare/cpu", _target("bare"), NEEDS_DATA)
    assert scheduler.place(needy, {}) is None
    assert scheduler.reservation(_NeedyEntry(
        "r1", "bare/cpu", _target("bare", "cpu", ["cpu", "cpu"]),
        NEEDS_DATA)) is None
    assert scheduler.explain(needy) == (
        "worker bare has no `data` mount; add `--mount data=...` to its "
        "agent")
    assert scheduler.place(_NeedyEntry("r2", "nas/cpu", _target("nas"),
                                       NEEDS_DATA), {}).worker_id == "nas"
    # A run that needs nothing goes anywhere, as before.
    assert scheduler.place(_NeedyEntry("r3", "bare/cpu", _target("bare"),
                                       {"mount": None, "files": []}),
                           {}).worker_id == "bare"


def test_any_skips_the_workers_without_the_mount_and_falls_back_here(
        data_label):
    scheduler = _scheduler(_worker("a-bare", "cpu"),
                           _mounted(_worker("b-nas", "cpu"), data="/n"))
    entry = _NeedyEntry("r1", "any/cpu", _target("any"), NEEDS_DATA)
    assert scheduler.place(entry, {}).worker_id == "b-nas"
    # This process always holds its own data directory.
    assert scheduler.place(entry, {("b-nas", "cpu"): 2}).worker_id == "local"


def test_a_same_host_worker_holds_the_control_planes_label(data_label):
    near = _worker("near", "cpu")
    near.same_host = True
    scheduler = _scheduler(near)
    entry = _NeedyEntry("r1", "near/cpu", _target("near"), NEEDS_DATA)
    assert scheduler.place(entry, {}).worker_id == "near"
    # Only the control plane's own label: another one it must declare.
    other = _NeedyEntry("r2", "near/cpu", _target("near"),
                        {"mount": "scratch", "files": []})
    assert scheduler.place(other, {}) is None


def test_any_explains_a_wait_for_a_mounted_worker(data_label):
    scheduler = _scheduler(_worker("gpu", "cpu", "cuda:0"))
    entry = _NeedyEntry("r1", "any/cuda", _target("any", "cuda"), NEEDS_DATA)
    assert scheduler.place(entry, {}) is None
    assert scheduler.explain(entry) == (
        "waiting for a worker that offers cuda and has the `data` mount")


def test_submit_refuses_a_named_worker_without_the_mount(data_label):
    bare = _worker("gpu-box-1", "cpu", "cuda:0")
    nas = _mounted(_worker("nas-box", "cpu"), data="/n")
    away = _mounted(_worker("away-box", "cpu", status="disconnected"),
                    data="/n")
    service = _service(_registry_with(bare, nas, away))
    check = service._check_storage
    with pytest.raises(RunSubmitError) as refused:
        check(_options(worker="gpu-box-1"), NEEDS_DATA)
    assert str(refused.value) == (
        "worker gpu-box-1 has no `data` mount; add `--mount data=...` to "
        "its agent")
    # A pure-tensor graph is accepted there.
    check(_options(worker="gpu-box-1"), {"mount": None, "files": []})
    check(_options(worker="nas-box"), NEEDS_DATA)
    # #777's rule still holds: a known worker that is away is waited for.
    check(_options(worker="away-box"), NEEDS_DATA)
    # any: this server holds the mount, so a cpu run is fine...
    check(_options(worker="any"), NEEDS_DATA)
    # ...but only gpu-box-1 offers cuda, and it has no mount.
    with pytest.raises(RunSubmitError, match="`data` mount"):
        check(_options(worker="any", device="cuda"), NEEDS_DATA)


def test_prepare_computes_needs_only_for_a_worker_run(data_label, tmp_path,
                                                      monkeypatch):
    """The 3.0 path never reads the graph for storage; a worker run's
    needs are what submit checks and the queue entry carries."""
    from app.config import settings

    # An upload store of its own: whatever the install's holds is not this
    # test's business.
    monkeypatch.setattr(settings, "DATA_FILES_DIR", tmp_path / "files")
    service = _service(_registry_with(_worker("gpu-box-1", "cpu")))
    graph = {"nodes": [
        {"id": "s", "type": "Start", "data": {"params": {}}},
        {"id": "c", "type": "CSVReader",
         "data": {"params": {"path": "train.csv"}}}],
        "edges": [{"id": "t", "source": "s", "target": "c",
                   "sourceHandle": "trigger", "type": "trigger"}]}
    *_, needs = service._prepare(graph, {"device": "cpu"})
    assert needs is None
    # train.csv is not an upload here, so the transfer backend (#785)
    # cannot ship it either, and the refusal says so.
    with pytest.raises(RunSubmitError, match=(
            "has no `data` mount; add `--mount data=...` to its agent, or "
            "make the run fit the transfer backend: `train.csv` is not an "
            "upload on this server")):
        service._prepare(graph, {"target": {"worker": "gpu-box-1"}})
    *_, needs = service._prepare(graph, {"target": {"worker": "any"}})
    assert needs["mount"] == "data" and needs["files"] == ["train.csv"]
    assert needs["transfer"]["ok"] is False


def test_cheap_refusals_come_before_the_graph_is_read(data_label,
                                                      monkeypatch):
    service = _service(_registry_with(_worker("gpu-box-1", "cpu")))

    def never(_graph):
        raise AssertionError("run_needs must not run for a refused submit")

    monkeypatch.setattr(service, "_needs_of", never)
    graph = {"nodes": [{"id": "s", "type": "Start", "data": {"params": {}}}]}
    with pytest.raises(RunSubmitError, match="has not connected"):
        service._prepare(graph, {"target": {"worker": "nobody"}})
    service._workers.get("gpu-box-1").capabilities = ("ack", "mounts")
    with pytest.raises(RunSubmitError, match="cannot record outputs"):
        service._prepare(graph, {"target": {"worker": "gpu-box-1"},
                                 "record_outputs": True})


def test_record_outputs_needs_the_capability_on_a_named_worker():
    """#789: refused at submit only for an agent that said hello without
    ``record_outputs`` (an older one); allowed with it."""
    from app.core.run_service import worker_cannot_record_outputs

    old = _worker("old", "cpu")
    old.capabilities = ("ack", "mounts", "transfer")
    new = _worker("new", "cpu")
    new.capabilities = tuple(wp.CAPABILITIES)
    service = _service(_registry_with(old, new))
    key = service._submit_queue_key
    with pytest.raises(RunSubmitError) as refused:
        key({**_options(worker="old"), "record_outputs": True})
    assert str(refused.value) == worker_cannot_record_outputs("old")
    assert key({**_options(worker="old")}) == "old/cpu"
    assert key({**_options(worker="new"), "record_outputs": True}) == \
        "new/cpu"


def test_record_outputs_is_refused_for_one_shot_workers():
    """#789 x #790: a one-shot agent exits when its run ends, so what the
    run recorded could never be read: refused at submit, for the pool and
    for a named one-shot worker, with a message saying why."""
    from app.core.run_service import ONE_SHOT_REFUSES_RECORD_OUTPUTS

    once = _worker("once", "cpu")
    once.capabilities = tuple(wp.CAPABILITIES)
    once.one_shot = True
    service = _service(_registry_with(once))
    key = service._submit_queue_key
    for options in ({"target": {"worker": "any", "pool": "one-shot"}, "device": "cpu"},
                    _options(worker="once")):
        with pytest.raises(RunSubmitError) as refused:
            key({**options, "record_outputs": True})
        assert str(refused.value) == ONE_SHOT_REFUSES_RECORD_OUTPUTS
        assert "one-shot" in str(refused.value)
    assert key({"target": {"worker": "any", "pool": "one-shot"}, "device": "cpu"}) == \
        "pool:one-shot/cpu"


def test_a_batch_check_returns_each_childs_needs(data_label):
    """Needs depend on the whole graph, so every child is read once and
    its needs handed back for submit; settings-only checks stay per
    distinct settings."""
    service = _service(_registry_with(_mounted(_worker("nas", "cpu"),
                                               data="/n")))

    def graph(path: str) -> dict:
        nodes = [{"id": "s", "type": "Start", "data": {"params": {}}}]
        edges = []
        if path:
            nodes.append({"id": "c", "type": "CSVReader",
                          "data": {"params": {"path": path}}})
            edges.append({"id": "t", "source": "s", "target": "c",
                          "sourceHandle": "trigger", "type": "trigger"})
        return {"nodes": nodes, "edges": edges}

    result = service.check_placeable_batch(
        [graph("a.csv"), graph(""), graph("b.csv")],
        {"target": {"worker": "nas"}})
    assert [None if r is None else (r["mount"], r["files"])
            for r in result] == [("data", ["a.csv"]), (None, []),
                                 ("data", ["b.csv"])]
    assert "transfer" in result[0] and "transfer" not in result[1]
    assert service.check_placeable_batch([graph("a.csv")], {}) == [None]


async def test_a_heartbeat_updates_the_mounts_and_wakes_the_queue():
    """A share that appears on a worker after it connected (#782) is
    placeable from its next heartbeat: the registry's listeners re-pump."""
    registry = WorkerRegistry()
    try:
        woken: list[int] = []
        registry.add_listener(lambda: woken.append(1))
        worker, _ = registry.connect(
            wp.Hello(name="box", version="t", mounts={}), send=_noop,
            close=_noop)
        woken.clear()
        registry.heartbeat(worker, wp.Heartbeat(mounts={"data": "/n"}))
        assert worker.mounts == {"data": "/n"} and woken == [1]
        registry.heartbeat(worker, wp.Heartbeat())  # None: unchanged
        registry.heartbeat(worker, wp.Heartbeat(mounts={"data": "/n"}))
        assert worker.mounts == {"data": "/n"} and woken == [1]
    finally:
        await registry.shutdown()



# ── node catalogs (#781) ─────────────────────────────────────────────────


def _cataloged(name: str, *devices: str, status: str = "mismatch",
               missing: tuple = ("X",), extra: tuple = (),
               has: tuple = ("Start",)) -> Worker:
    """A connected worker of a registry that compares catalogs."""
    from app.core.worker_handshake import CatalogCheck

    worker = _worker(name, *devices)
    worker.catalog_checked = True
    worker.node_types = tuple(has)
    worker.catalog = CatalogCheck(status, None if status == "unknown"
                                  else "h", missing=missing, extra=extra)
    return worker


def test_any_skips_a_worker_lacking_types_and_says_why_it_waits():
    scheduler = _scheduler(_cataloged("a-box", "cpu", "cuda:0"),
                           _worker("b-box", "cpu"))
    entry = _Entry("r1", "any/cpu", _target("any"))
    assert scheduler.place(entry, {}).worker_id == "b-box"
    # Only that worker offers the card: the run waits, and says so.
    cuda = _Entry("r2", "any/cuda", _target("any", "cuda"))
    assert scheduler.place(cuda, {}) is None
    assert scheduler.can_serve_any("cuda", None, connected_only=True)
    assert not scheduler.can_serve_any("cuda", None, connected_only=True,
                                       usable_only=True)
    assert scheduler.explain(cuda) == (
        "waiting for a worker that offers cuda: a-box is skipped by `any` "
        "(node types missing, or not reported)")


def test_any_skips_a_worker_without_a_catalog_but_not_a_superset_one():
    unknown = _cataloged("a-box", "cpu", status="unknown", missing=())
    superset = _cataloged("b-box", "cpu", missing=(), extra=("mine:Only",))
    entry = _Entry("r1", "any/cpu", _target("any"))
    assert _scheduler(unknown, superset).place(entry, {}).worker_id == "b-box"
    assert unknown.skipped_by_any and not superset.skipped_by_any
    # A registry that compares nothing skips nothing.
    unknown.catalog_checked = False
    assert _scheduler(unknown).place(entry, {}).worker_id == "a-box"


def test_a_named_worker_lacking_types_is_placed():
    """Its submit was checked (and the executor checks again): placement
    itself does not refuse a run that names it."""
    scheduler = _scheduler(_cataloged("box", "cpu"))
    entry = _Entry("r1", "box/cpu", _target("box"))
    assert scheduler.place(entry, {}).worker_id == "box"
    assert scheduler.explain(entry) is None


def test_submit_to_a_degraded_worker_needs_every_node_type_it_uses():
    service = _service(_registry_with(
        _cataloged("box", "cpu", has=("Start", "Print")),
        _cataloged("more", "cpu", missing=(), has=("Start",)),
        _cataloged("blind", "cpu", status="unknown", missing=()),
        _worker("full", "cpu")))
    check = service._check_worker_catalog
    check(_options(worker="box"), frozenset({"Start", "Print"}))
    with pytest.raises(RunSubmitError) as refused:
        check(_options(worker="box"),
              frozenset({"Start", "foundations:Edu-KNN", "Print"}))
    assert str(refused.value).startswith(
        "worker 'box' does not have the node type(s) this graph uses: "
        "foundations:Edu-KNN.")
    # A superset worker is still checked per run.
    with pytest.raises(RunSubmitError, match="Print"):
        check(_options(worker="more"), frozenset({"Start", "Print"}))
    # A matching worker, one without a catalog, any, local: not checked.
    for target in ("full", "blind", "any", "local"):
        check(_options(worker=target), frozenset({"foundations:Edu-KNN"}))


def test_a_batch_is_checked_per_child_not_per_settings():
    """Two children with the same settings but different node types: the
    second is refused for a worker that lacks its type (#781)."""
    def chain(*types: str) -> dict[str, Any]:
        nodes = [{"id": "s", "type": "Start", "data": {"params": {}}}]
        edges = []
        for i, node_type in enumerate(types):
            nodes.append({"id": f"n{i}", "type": node_type,
                          "data": {"params": {}}})
            edges.append({"id": f"e{i}", "source": "s", "target": f"n{i}",
                          "sourceHandle": "trigger", "type": "trigger"})
        return {"nodes": nodes, "edges": edges, "settings": {"seed": 1}}

    service = _service(_registry_with(
        _cataloged("box", "cpu", has=("Start", "Print"))))
    options = {"target": {"worker": "box"}}
    service.check_placeable_batch([chain("Print"), chain("Print")], options)
    with pytest.raises(RunSubmitError, match="foundations:Edu-KNN"):
        service.check_placeable_batch(
            [chain("Print"), chain("foundations:Edu-KNN")], options)


def test_a_busy_transfer_run_has_no_made_up_reason(data_label):
    """A run that reaches a worker by the transfer backend (#785) and only
    waits for a busy slot gets no reason: the usable-worker check counts
    the transfer route too (it used to read "... are skipped by `any`"
    naming nobody)."""
    gpu = _worker("gpu", "cpu", "cuda:0")
    gpu.capabilities = (wp.CAPABILITY_TRANSFER,)
    scheduler = _scheduler(gpu)
    entry = _NeedyEntry("r1", "any/cuda", _target("any", "cuda"), {
        **NEEDS_DATA, "transfer": {"ok": True}})
    assert scheduler.place(entry, {}).worker_id == "gpu"
    assert scheduler.place(entry, {("gpu", "cuda:0"): 1}) is None
    assert scheduler.explain(entry) is None


def test_the_skipped_list_names_only_workers_that_could_take_the_run(
        data_label):
    """A worker ``any`` skips is named only when it would otherwise reach
    the run's files: one without the mount (and no transfer) is not."""
    held = _mounted(_cataloged("a-box", "cpu", "cuda:0"), data="/n")
    bare = _cataloged("c-bare", "cpu", "cuda:0")
    scheduler = _scheduler(held, bare)
    entry = _NeedyEntry("r1", "any/cuda", _target("any", "cuda"), NEEDS_DATA)
    assert scheduler.place(entry, {}) is None
    assert scheduler.explain(entry) == (
        "waiting for a worker that offers cuda: a-box is skipped by `any` "
        "(node types missing, or not reported)")


# ── reason codes (#841) ───────────────────────────────────────────────────


def test_an_any_run_behind_a_draining_worker_says_so():
    """U5 of the 3.1.0 audit: a run naming a draining worker said "worker X
    is draining", an ``any`` run waiting for the same card said nothing."""
    scheduler = _scheduler(_worker("meow", "cpu", "cuda:0", draining=True))
    entry = _Entry("r1", "any/cuda", _target("any", "cuda"))
    assert scheduler.place(entry, {}) is None
    reason = scheduler.explain_reason(entry)
    assert (reason.code, reason.params_dict()) == ("any_unavailable", {
        "devices": "cuda", "draining": ["meow"], "skipped": []})
    assert reason.text == scheduler.explain(entry) == (
        "waiting for a worker that offers cuda: meow is draining")
    # A worker that can take it makes it a plain wait for a slot again.
    other = _scheduler(_worker("meow", "cpu", "cuda:0", draining=True),
                       _worker("zed", "cuda:0"))
    assert other.explain_reason(entry) is None


def test_draining_and_skipped_workers_are_both_named():
    scheduler = _scheduler(_worker("b-box", "cuda:0", draining=True),
                           _worker("c-box", "cuda:0", draining=True),
                           _cataloged("a-box", "cuda:0"))
    reason = scheduler.explain_reason(
        _Entry("r1", "any/cuda", _target("any", "cuda")))
    assert reason.params_dict() == {"devices": "cuda",
                             "draining": ["b-box", "c-box"],
                             "skipped": ["a-box"]}
    assert reason.text == (
        "waiting for a worker that offers cuda: b-box, c-box are draining; "
        "a-box is skipped by `any` (node types missing, or not reported)")


def test_every_reason_carries_a_code_and_params(data_label):
    scheduler = _scheduler(_worker("box", "cpu", draining=True),
                           _mounted(_worker("bare", "cpu")))

    def reason(entry: _Entry) -> tuple[str, dict[str, Any]]:
        found = scheduler.explain_reason(entry)
        return found.code, found.params_dict()

    assert reason(_Entry("r", "box/cpu", _target("box"))) == (
        "worker_draining", {"worker": "box"})
    assert reason(_Entry("r", "gone/cpu", _target("gone"))) == (
        "worker_reconnect", {"worker": "gone"})
    code, params = reason(_NeedyEntry("r", "bare/cpu", _target("bare"),
                                      NEEDS_DATA))
    assert (code, params["worker"], params["mount"]) == (
        "worker_no_mount", "bare", "data")
    assert reason(_Entry("r", "any/cuda", _target("any", "cuda"))) == (
        "any_no_worker", {"devices": "cuda"})
    assert reason(_NeedyEntry("r", "any/cuda", _target("any", "cuda"),
                              NEEDS_DATA)) == (
        "any_no_worker_mount", {"devices": "cuda", "mount": "data"})
    assert scheduler.explain_reason(
        _Entry("r", "cpu", {"device": "cpu"})) is None
