"""Device sets inside the run subprocess (#774).

A run given ``devices=["cuda:0", "cuda:2"]`` runs in a subprocess started
with ``CUDA_VISIBLE_DEVICES=0,2``, so inside it the cards are ``cuda:0`` and
``cuda:1``, in the order the set lists them. Nothing rewrites a node's
``device`` param: the CUDA runtime does the renumbering. These tests check
that chain without a GPU: :class:`FakeCudaRuntime` patches ``torch.cuda``
to behave the way the CUDA runtime does on a four-card machine, numbering
only the cards ``CUDA_VISIBLE_DEVICES`` lists.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from app.agent import child
from app.agent.runner import (
    CHILD_UNSET_ENV,
    Runner,
    child_device,
    child_environment,
    device_aliases,
    run_facts,
)
from app.core import worker_protocol as wp
from app.core.device_utils import (
    get_available_devices,
    node_target_device,
    resolve_device,
)

from . import _helpers as h

PHYSICAL_CARDS = 4


class FakeCudaRuntime:
    """``torch.cuda`` on a machine with :data:`PHYSICAL_CARDS` cards.

    Like the real runtime, it reads ``CUDA_VISIBLE_DEVICES`` and numbers the
    cards it lists ``0..N-1`` in the order listed; ``physical`` answers
    which card of the machine an in-process ``cuda:K`` is.
    """

    def visible(self) -> list[int]:
        raw = os.environ.get("CUDA_VISIBLE_DEVICES")
        if raw is None:
            return list(range(PHYSICAL_CARDS))
        return [int(part) for part in raw.split(",") if part.strip()]

    def physical(self, device: str) -> int:
        kind, _, index = device.partition(":")
        assert kind == "cuda", device
        return self.visible()[int(index or self.current_device())]

    def device_count(self) -> int:
        return len(self.visible())

    def is_available(self) -> bool:
        return self.device_count() > 0

    def current_device(self) -> int:
        if not self.visible():
            raise RuntimeError("No CUDA GPUs are available")
        return 0


@pytest.fixture
def cuda_runtime(monkeypatch):
    """A fake CUDA runtime, and ``os.environ`` restored after the test."""
    runtime = FakeCudaRuntime()
    for name in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", *CHILD_UNSET_ENV):
        # Recorded by monkeypatch so whatever apply_environment does is undone.
        monkeypatch.setenv(name, "sentinel")
        monkeypatch.delenv(name)
    monkeypatch.setattr(torch.cuda, "is_available", runtime.is_available)
    monkeypatch.setattr(torch.cuda, "device_count", runtime.device_count)
    monkeypatch.setattr(torch.cuda, "current_device", runtime.current_device)
    monkeypatch.setattr(torch.cuda, "get_device_name",
                        lambda index=0: f"Fake GPU {index}")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    get_available_devices.cache_clear()
    yield runtime
    get_available_devices.cache_clear()


def _enter_run(device: str, devices: list[str]) -> str:
    """Put the child's environment in place; the run's in-process device."""
    child.apply_environment(child_environment(device, devices))
    get_available_devices.cache_clear()
    return resolve_device(child_device(device, devices))


# ── the mapping ───────────────────────────────────────────────────────────


def test_a_single_card_run_is_cuda_0_inside_the_child(cuda_runtime):
    """Acceptance: targeted at ["cuda:1"], current_device() == 0 and
    visible_devices == "1"."""
    env = child_environment("cuda:1", ["cuda:1"])
    assert run_facts(env, ["cuda:1"]) == {
        "visible_devices": "1", "device_aliases": {"cuda:1": "cuda:0"}}
    run_device = _enter_run("cuda:1", ["cuda:1"])
    assert run_device == "cuda:0"
    assert torch.cuda.current_device() == 0
    assert cuda_runtime.physical(run_device) == 1


def test_a_node_pinned_to_cuda_1_lands_on_the_second_card_of_the_set(
        cuda_runtime):
    """Acceptance: in a ["cuda:0", "cuda:2"] run, a node's cuda:1 is
    physical card 2. The param is not rewritten; the runtime renumbers."""
    run_device = _enter_run("cuda:0", ["cuda:0", "cuda:2"])
    context = SimpleNamespace(device=run_device)
    assert cuda_runtime.physical(run_device) == 0
    params = {"device": "cuda:1"}
    node_device = node_target_device(params, context)
    assert params == {"device": "cuda:1"}
    assert node_device == "cuda:1"
    assert cuda_runtime.physical(node_device) == 2
    # 'auto' follows the run's device: the first card of the set.
    assert cuda_runtime.physical(node_target_device({"device": "auto"},
                                                    context)) == 0


def test_the_order_of_the_set_decides_the_aliases(cuda_runtime):
    devices = ["cuda:2", "cuda:0"]
    env = child_environment("cuda:2", devices)
    assert env["set"]["CUDA_VISIBLE_DEVICES"] == "2,0"
    assert device_aliases(devices) == {"cuda:2": "cuda:0", "cuda:0": "cuda:1"}
    run_device = _enter_run("cuda:2", devices)
    assert run_device == "cuda:0"
    assert cuda_runtime.physical(run_device) == 2
    context = SimpleNamespace(device=run_device)
    assert cuda_runtime.physical(
        node_target_device({"device": "cuda:1"}, context)) == 0


def test_a_physical_name_outside_the_aliases_does_not_reach_that_card(
        cuda_runtime, caplog):
    """Inside a ["cuda:0", "cuda:2"] run there is no cuda:2: the value
    degrades the way any out-of-range index does, to the current card.
    This is why the node param holds the alias, not the physical id."""
    run_device = _enter_run("cuda:0", ["cuda:0", "cuda:2"])
    context = SimpleNamespace(device=run_device)
    with caplog.at_level("WARNING"):
        node_device = node_target_device({"device": "cuda:2"}, context)
    assert node_device == "cuda:0"
    assert cuda_runtime.physical(node_device) == 0
    assert "index 2 was requested" in caplog.text


@pytest.mark.parametrize("device, devices", [
    ("cpu", ["cpu"]), ("cpu", []), ("mps", ["mps:0"]), ("mps", []),
])
def test_cpu_and_mps_runs_see_no_card(cuda_runtime, device, devices):
    env = child_environment(device, devices)
    assert run_facts(env, devices or [device]) == {
        "visible_devices": "", "device_aliases": {}}
    child.apply_environment(env)
    assert torch.cuda.device_count() == 0


def test_a_bare_cuda_with_no_set_records_the_inherited_visibility():
    env = child_environment("cuda", [])
    assert run_facts(env, ["cuda"]) == {
        "visible_devices": None, "device_aliases": {}}


def test_a_run_with_no_set_is_its_own_device():
    env = child_environment("cuda:3", [])
    assert run_facts(env, ["cuda:3"]) == {
        "visible_devices": "3", "device_aliases": {"cuda:3": "cuda:0"}}


# ── run_started carries the facts ─────────────────────────────────────────


async def test_run_started_carries_the_mapping_the_child_gets(tmp_path: Path):
    sink = h.FrameSink()
    runner = Runner(runs_dir=tmp_path / "agent-runs", send=sink,
                    offered_devices={"cpu", "cuda:0", "cuda:1", "cuda:2"},
                    child_entry=h.env_child)
    await runner.assign(wp.Assign(run_id="set1", graph={"nodes": []},
                                  device="cuda:2",
                                  devices=["cuda:2", "cuda:0"]))
    started = await sink.wait_for(lambda f: isinstance(f, wp.RunStarted))
    assert started.execution_facts == {
        "visible_devices": "2,0",
        "device_aliases": {"cuda:2": "cuda:0", "cuda:0": "cuda:1"}}
    event = await sink.wait_for(lambda f: isinstance(f, wp.Event))
    # What the subprocess actually got is what run_started says it got.
    assert (event.payload["env"]["CUDA_VISIBLE_DEVICES"]
            == started.execution_facts["visible_devices"])
    assert event.payload["device"] == "cuda:0"
    await sink.wait_for(lambda f: isinstance(f, wp.RunFinished))
    # The frame survives the wire.
    assert wp.decode_agent_frame(wp.encode_frame(started)) == started


async def test_a_cpu_run_reports_no_visible_card(tmp_path: Path):
    sink = h.FrameSink()
    runner = Runner(runs_dir=tmp_path / "agent-runs", send=sink,
                    offered_devices={"cpu"}, child_entry=h.env_child)
    await runner.assign(wp.Assign(run_id="cpu1", graph={"nodes": []}))
    started = await sink.wait_for(lambda f: isinstance(f, wp.RunStarted))
    assert started.execution_facts == {"visible_devices": "",
                                       "device_aliases": {}}
    event = await sink.wait_for(lambda f: isinstance(f, wp.Event))
    assert event.payload["env"]["CUDA_VISIBLE_DEVICES"] == ""
    await sink.wait_for(lambda f: isinstance(f, wp.RunFinished))
