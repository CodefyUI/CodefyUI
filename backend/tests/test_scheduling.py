"""The scheduler seam under ``RunService._pump`` (#768).

``LocalScheduler`` must reproduce the #123 rule exactly (``test_run_queue``
pins that end to end); this module pins the seam itself: the pump defers to
whatever scheduler it is given, slots are keyed by ``(worker_id, device)``,
and a placement holding several slots gives every one of them back.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Mapping
from typing import Any

import pytest

from app.core import run_service as run_service_module
from app.core.db import Database
from app.core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from app.core.node_registry import registry
from app.core.run_service import QueueLimits, RunService
from app.core.run_store import (
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    WORKER_LOCAL,
    RunStore,
)
from app.core.scheduling import (
    LOCAL_WORKER_ID,
    LocalScheduler,
    Placement,
    QueuedEntry,
    SlotKey,
)

# ── a node that holds its run open until the test lets go ─────────────────

_gates: dict[str, threading.Event] = {}
_entered: dict[str, threading.Event] = {}


class _HoldNode(BaseNode):
    NODE_NAME = "_SchedHold"
    CATEGORY = "Test"
    DESCRIPTION = "Blocks until the test releases its label"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [ParamDefinition(name="label", param_type=ParamType.STRING,
                                default="")]

    def execute(self, inputs: dict[str, Any],
                params: dict[str, Any]) -> dict[str, Any]:
        label = str(params.get("label", ""))
        entered = _entered.get(label)
        if entered is not None:
            entered.set()
        gate = _gates.get(label)
        if gate is not None and not gate.wait(timeout=10.0):
            raise TimeoutError(f"hold for {label!r} was never released")
        return {"value": inputs.get("value")}


@pytest.fixture(autouse=True)
def _register_hold_node():
    registry._nodes["_SchedHold"] = _HoldNode
    _gates.clear()
    _entered.clear()
    yield
    for gate in _gates.values():
        gate.set()
    registry._nodes.pop("_SchedHold", None)


@pytest.fixture(autouse=True)
def _fake_devices(monkeypatch):
    """Device strings resolve to themselves (no GPU on CI); see test_run_queue."""
    monkeypatch.setattr(
        run_service_module, "resolve_device",
        lambda requested: (requested or "cpu").strip().lower() or "cpu")
    monkeypatch.setattr(run_service_module, "_current_cuda_index", lambda: 0)


@pytest.fixture
def store(tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    try:
        yield RunStore(database)
    finally:
        database.close()


@pytest.fixture
async def make_service(store):
    built: list[RunService] = []

    def _make(**kwargs: Any) -> RunService:
        service = RunService(store, limits=QueueLimits(cpu=2, gpu=1),
                             **kwargs)
        built.append(service)
        return service

    try:
        yield _make
    finally:
        for service in built:
            await service.shutdown()


def _graph(label: str) -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "start", "type": "Start", "data": {"params": {}}},
            {"id": "src", "type": "_TestSource",
             "data": {"params": {"val": label}}},
            {"id": "hold", "type": "_SchedHold",
             "data": {"params": {"label": label}}},
        ],
        "edges": [
            {"id": "et", "source": "start", "target": "src",
             "sourceHandle": "trigger", "type": "trigger"},
            {"id": "e1", "source": "src", "target": "hold",
             "sourceHandle": "value", "targetHandle": "value"},
        ],
    }


def _hold(label: str) -> threading.Event:
    _entered[label] = threading.Event()
    _gates[label] = threading.Event()
    return _gates[label]


async def _wait_entered(label: str) -> None:
    assert await asyncio.to_thread(_entered[label].wait, 5.0), (
        f"{label!r} never started executing")


async def _submit(service: RunService, label: str, device: str = "cpu"):
    return await service.submit(_graph(label), options={"device": device},
                                name=label)


async def _await_terminal(store: RunStore, run_id: str,
                          timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = await store.get_run(run_id)
        assert record is not None
        if record.finished_at is not None:
            return record
        await asyncio.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s")


# ── LocalScheduler: the #123 rule, keyed by (worker, device) ──────────────


class _Entry:
    def __init__(self, queue_key: str) -> None:
        self.run_id = "r"
        self.queue_key = queue_key
        self.options: dict[str, Any] = {}


def test_the_local_worker_id_is_the_one_the_run_rows_store():
    assert LOCAL_WORKER_ID == WORKER_LOCAL == "local"


def test_local_scheduler_places_one_local_slot_under_the_cap():
    scheduler = LocalScheduler(QueueLimits(cpu=2, gpu=1))
    assert scheduler.place(_Entry("cuda:0"), {}) == Placement(
        "local", ("cuda:0",), (("local", "cuda:0"),))
    assert scheduler.place(_Entry("cpu"), {("local", "cpu"): 1}) == Placement(
        "local", ("cpu",), (("local", "cpu"),))


def test_local_scheduler_refuses_at_the_cap_and_reads_only_its_own_slot():
    scheduler = LocalScheduler(QueueLimits(cpu=2, gpu=1,
                                           overrides=(("cuda:1", 2),)))
    assert scheduler.place(_Entry("cuda:0"), {("local", "cuda:0"): 1}) is None
    assert scheduler.place(_Entry("cpu"), {("local", "cpu"): 2}) is None
    # The override still means the local worker's card.
    assert scheduler.place(_Entry("cuda:1"),
                           {("local", "cuda:1"): 1}) is not None
    # Another worker's card of the same name is a different slot.
    assert scheduler.place(_Entry("cuda:0"),
                           {("w1", "cuda:0"): 5}) is not None


# ── the pump defers to the scheduler ──────────────────────────────────────


class _Refusing:
    def __init__(self) -> None:
        self.refuse = True
        self.asked: list[str] = []
        self._local = LocalScheduler(QueueLimits())

    def place(self, entry: QueuedEntry,
              taken: Mapping[SlotKey, int]) -> Placement | None:
        self.asked.append(entry.run_id)
        return None if self.refuse else self._local.place(entry, taken)


async def test_a_refusing_scheduler_keeps_the_run_queued(make_service, store):
    scheduler = _Refusing()
    service = make_service(scheduler=scheduler)

    first = await _submit(service, "first")
    assert first.status == STATUS_QUEUED
    assert scheduler.asked == [first.run_id]
    # An idle device does not matter: the scheduler said no.
    await asyncio.sleep(0.1)
    assert service.is_queued(first.run_id)
    assert service.queue_position(first.run_id) == 1
    assert service.queue_snapshot() == {"cpu": [first.run_id]}
    assert service._slots == {}
    assert (await store.get_run(first.run_id)).status == STATUS_QUEUED

    # Once it agrees, the next pump starts the FRONT run first.
    scheduler.refuse = False
    second = await _submit(service, "second")
    for result in (first, second):
        assert (await _await_terminal(store, result.run_id)).status == (
            STATUS_SUCCEEDED)
    assert service._slots == {}


class _MultiSlot:
    """``cpu`` runs hold three slots; everything else follows the local rule."""

    SLOTS: tuple[SlotKey, ...] = (
        ("local", "cpu"), ("local", "cuda:0"), ("w1", "cuda:1"))

    def __init__(self) -> None:
        self._local = LocalScheduler(QueueLimits(cpu=2, gpu=1))

    def place(self, entry: QueuedEntry,
              taken: Mapping[SlotKey, int]) -> Placement | None:
        if entry.queue_key == "cpu":
            return Placement("local", ("cpu", "cuda:0"), self.SLOTS)
        return self._local.place(entry, taken)


async def test_releasing_a_multi_slot_placement_frees_every_slot(
        make_service, store):
    service = make_service(scheduler=_MultiSlot())
    release = _hold("wide")
    try:
        wide = await _submit(service, "wide", device="cpu")
        assert wide.status == STATUS_RUNNING
        await _wait_entered("wide")
        assert service._slots == {slot: 1 for slot in _MultiSlot.SLOTS}

        # ``cuda:0``'s slot is held by the wide run, so its FIFO waits.
        narrow = await _submit(service, "narrow", device="cuda:0")
        assert narrow.status == STATUS_QUEUED
    finally:
        release.set()

    assert (await _await_terminal(store, wide.run_id)).status == (
        STATUS_SUCCEEDED)
    # The release pumped cuda:0's queue too, not just the run's own.
    assert (await _await_terminal(store, narrow.run_id)).status == (
        STATUS_SUCCEEDED)
    assert service._slots == {}
