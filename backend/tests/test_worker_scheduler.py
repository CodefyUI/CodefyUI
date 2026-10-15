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


def test_any_with_a_device_set_falls_back_here_only_with_every_device():
    entry = _Entry("r1", "any/cuda:0",
                   _target("any", "cuda:0", ["cuda:0", "cuda:1"]))
    only_one = _scheduler(here=("cpu", "cuda:0"))
    assert only_one.place(entry, {}) is None
    assert only_one.explain(entry) == (
        "waiting for a worker that offers cuda:0, cuda:1")
    both = _scheduler(here=("cpu", "cuda:0", "cuda:1"))
    # Here, it executes on the first card and holds both (#773).
    assert both.place(entry, {}) == Placement(
        "local", ("cuda:0",), (("local", "cuda:0"), ("local", "cuda:1")))
    assert both.place(entry, {("local", "cuda:1"): 1}) is None
    assert both.explain(entry) is None


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
    with pytest.raises(RunSubmitError, match="neither this server"):
        key(_options(worker="any", devices=["cuda:0", "cuda:1"]))
    assert key(_options(worker="local")) == "cpu"
    with pytest.raises(RunSubmitError, match="has not connected"):
        key(_options(worker="nobody"))
    with pytest.raises(RunSubmitError, match="does not offer cuda:1"):
        key(_options(worker="box", devices=["cuda:1"]))
    with pytest.raises(RunSubmitError, match="does not offer mps"):
        key(_options(worker="box", device="mps"))
    with pytest.raises(RunSubmitError, match="record_outputs"):
        key({**_options(worker="box"), "record_outputs": True})
    with pytest.raises(RunSubmitError, match="record_outputs"):
        key({**_options(worker="any"), "record_outputs": True})
    with pytest.raises(RunSubmitError, match="interactive"):
        key({**_options(worker="box"), "lane": "interactive"})
    assert worker_not_connected("nobody") in str(
        pytest.raises(RunSubmitError, key, _options(worker="nobody")).value)


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
