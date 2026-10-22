"""``app.agent.child``: the run subprocess's side, exercised in-process (#771).

Everything here runs in the test process (no spawn): the pipe is a list,
the registry is the one ``conftest`` populated (``discover=False``).
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import threading

import pytest

from app.agent import child
from app.config import settings
from app.core.execution_context import ArtifactSignal, WarningSignal
from app.core.run_store import MetricPoint

from . import _helpers as h


def _spec(graph, options=None, **kw):
    return {"run_id": "child1", "graph": graph, "options": options or {},
            "device": "cpu", **kw}


def _run(spec) -> list[tuple]:
    sent: list[tuple] = []
    # run_assignment calls asyncio.run, so it needs a thread with no loop.
    thread = threading.Thread(target=child.run_assignment,
                              args=(spec, sent.append),
                              kwargs={"discover": False})
    thread.start()
    thread.join(60)
    assert not thread.is_alive()
    return sent


async def test_events_are_byte_identical_to_in_process_executor():
    sent = await asyncio.to_thread(
        _run, _spec(h.tensor_graph(), h.TENSOR_OPTIONS))
    assert sent[-1] == (child.MSG_FINISHED, {
        "status": "succeeded", "error": None, "peak_memory": None})
    worker = [h.event_bytes(m[1], m[2]) for m in sent
              if m[0] == child.MSG_EVENT]
    local = await h.in_process_events(h.tensor_graph(), h.TENSOR_OPTIONS)
    assert worker == local


class _ExclusiveFakeCuda:
    """Just enough of ``torch.cuda`` for the peak-memory scope (#770)."""

    def __init__(self) -> None:
        self.resets: list[int] = []

    def is_initialized(self):
        return True

    def current_device(self):
        return 0

    def device_count(self):
        return 2

    def reset_peak_memory_stats(self, card):
        self.resets.append(card)

    def max_memory_allocated(self, card):
        return 5 * 1024 ** 3 if card == 0 else 0

    def max_memory_reserved(self, card):
        return 6 * 1024 ** 3 if card == 0 else 0

    def memory_reserved(self, card):
        return self.max_memory_reserved(card)


def test_the_child_reports_an_exact_peak(monkeypatch):
    """The child's process holds this run alone: every card is reset
    unconditionally, even with a server-side scope registered on the same
    card (the module state a forked child could inherit), and the figure
    is ``exact: true`` and never ``shared``."""
    import sys
    from types import SimpleNamespace

    from app.core import executors

    cuda = _ExclusiveFakeCuda()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=cuda, backends=SimpleNamespace()))

    async def _succeeds(self, request):
        assert request.context.device == "cuda:0"
        return executors.ExecutionOutcome("succeeded")

    monkeypatch.setattr(executors.InProcessExecutor, "_execute", _succeeds)
    inherited = executors.PeakMemoryScope("cuda:0")
    inherited.card = 0
    monkeypatch.setattr(executors, "_ACTIVE_SCOPES", [inherited])

    sent = _run(_spec(h.tensor_graph(), device="cuda:0"))

    assert cuda.resets == [0, 1]
    assert sent[-1] == (child.MSG_FINISHED, {
        "status": "succeeded", "error": None,
        "peak_memory": {"cuda:0": {"allocated_bytes": 5 * 1024 ** 3,
                                   "reserved_bytes": 6 * 1024 ** 3,
                                   "exact": True}}})
    # The child's scope never registered next to the inherited one.
    assert executors._ACTIVE_SCOPES == [inherited]


async def test_a_failing_graph_reports_failed_and_never_raises():
    graph = {"nodes": [h.node("bad", "NoSuchNodeType")], "edges": []}
    sent = await asyncio.to_thread(_run, _spec(graph))
    kind, body = sent[-1]
    assert kind == child.MSG_FINISHED
    assert body["status"] == "failed" and body["error"]


async def test_a_broken_spec_still_reports():
    sent = await asyncio.to_thread(_run, {"run_id": "x"})  # no graph
    assert sent[-1][0] == child.MSG_FINISHED
    assert sent[-1][1]["status"] == "failed"
    assert "KeyError" in sent[-1][1]["error"]


def test_context_mirrors_run_service_start():
    options = {"verbose": True, "graph_id": "g1", "backward_mode": True,
               "auto_backward": True, "seed": 7, "deterministic": True,
               "weights_persistent": True}
    context = child.build_context("r1", options, "cuda:0")
    assert context.execution_id == "r1"
    assert context.device == "cuda:0"
    assert context.max_workers == settings.MAX_PARALLEL_NODES
    assert (context.verbose, context.graph_id, context.backward_mode,
            context.auto_backward, context.seed, context.deterministic) == (
        True, "g1", True, True, 7, True)
    # Isolated whatever the options say: no store reaches a worker run.
    assert context.weights_persistent is False
    assert context.node_state_store is None


async def test_pipe_sinks_write_one_message_each():
    sent: list[tuple] = []
    sinks = child._pipe_sinks(sent.append)
    await sinks.emit("node_status", {"node_id": "a", "status": "running"})
    await sinks.record_metrics([])  # the batcher poke: not forwarded
    await sinks.record_metrics([MetricPoint("loss", 0.5, 3, "a")])
    await sinks.record_artifact(ArtifactSignal("checkpoint", "/tmp/c.pt",
                                               {"epoch": 1}, "a"))
    await sinks.warn(WarningSignal("evicted", "weights dropped", "a"))
    assert sent == [
        (child.MSG_EVENT, "node_status", {"node_id": "a", "status": "running"}),
        (child.MSG_METRICS, [{"name": "loss", "value": 0.5, "step": 3,
                              "node_id": "a"}]),
        (child.MSG_ARTIFACT, {"kind": "checkpoint", "path": "/tmp/c.pt",
                              "meta": {"epoch": 1}, "node_id": "a",
                              "mount_label": None}),
        (child.MSG_WARNING, {"kind": "evicted", "detail": "weights dropped",
                             "node_id": "a"}),
    ]


@pytest.mark.parametrize("how", ["cancel", "eof"])
def test_cancel_listener_cancels_on_message_and_on_eof(how):
    context = child.build_context("r1", {}, "cpu")
    reader, writer = multiprocessing.Pipe(duplex=False)
    thread = threading.Thread(target=child._listen_for_cancel,
                              args=(reader, context), daemon=True)
    thread.start()
    if how == "cancel":
        writer.send((child.MSG_CANCEL,))
    else:
        writer.close()
    for _ in range(200):
        if context.cancelled:
            break
        threading.Event().wait(0.01)
    assert context.cancelled
    if how == "cancel":
        writer.close()
    thread.join(5)


def test_apply_environment(monkeypatch):
    monkeypatch.setenv("CODEFYUI_DB_PATH", "/x.db")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    child.apply_environment({"set": {"CUDA_VISIBLE_DEVICES": "1"},
                             "unset": ["CODEFYUI_DB_PATH"]})
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert "CODEFYUI_DB_PATH" not in os.environ
    # Not monkeypatch.delenv: that would record "1" and restore it after
    # the test, for every later test (the agent reads it, #790).
    os.environ.pop("CUDA_VISIBLE_DEVICES")


def test_safe_sender_swallows_a_closed_pipe():
    reader, writer = multiprocessing.Pipe(duplex=False)
    reader.close()
    writer.close()
    child._safe_sender(writer)(("event",))  # no exception


def test_wire_payload_applies_the_control_planes_cap_and_sanitising():
    from app.core.run_events import cap_event_payload
    from app.core.run_store import json_safe

    big = {"node_id": "a", "status": "error", "error": "x" * 50_000,
           "loss": float("nan"), "shape": (2, 3), "obj": object()}
    wired = child.wire_payload(big, 4096)
    assert wired["error"].endswith("... [truncated]")
    assert wired["loss"] is None and wired["shape"] == [2, 3]
    assert isinstance(wired["obj"], str)
    # Small payloads are what the control plane would store, unchanged.
    small = {"node_id": "a", "status": "running"}
    assert child.wire_payload(small, 4096) == cap_event_payload(
        json_safe(small), cap_bytes=4096) == small


async def test_an_unencodable_event_becomes_a_warning():
    sent: list[tuple] = []
    sinks = child._pipe_sinks(sent.append, cap_bytes=0)
    loop: dict = {"node_id": "n1"}
    loop["self"] = loop  # circular: no JSON for this
    await sinks.emit("node_status", loop)
    assert sent[0][0] == child.MSG_WARNING
    assert sent[0][1]["kind"] == "unserialisable_event"
    assert sent[0][1]["node_id"] == "n1"


def test_children_ignore_ctrl_c():
    import signal

    previous = signal.getsignal(signal.SIGINT)
    try:
        child.ignore_interrupts()
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, previous)


# ── memory budgets (#788) ─────────────────────────────────────────────────


class _CappedCuda:
    """``torch.cuda`` with 16 GiB cards and a record of the caps set."""

    def __init__(self, fail: bool = False) -> None:
        self.caps: list[tuple[float, int]] = []
        self.fail = fail

    def get_device_properties(self, card):
        from types import SimpleNamespace

        return SimpleNamespace(total_memory=16 * 1024 ** 3)

    def set_per_process_memory_fraction(self, fraction, card):
        if self.fail:
            raise RuntimeError("no CUDA here")
        self.caps.append((fraction, card))


def test_memory_budgets_cap_the_card_before_anything_else(monkeypatch):
    """The cap is set, and reported FIRST, before the graph runs: the agent
    sends ``run_started`` on that message."""
    import sys
    from types import SimpleNamespace

    from app.core import executors

    cuda = _CappedCuda()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=cuda, backends=SimpleNamespace()))

    async def _succeeds(self, request):
        assert cuda.caps == [(0.375, 0)]  # capped before the run
        return executors.ExecutionOutcome("succeeded")

    monkeypatch.setattr(executors.InProcessExecutor, "run", _succeeds)
    sent = _run(_spec(h.tensor_graph(), device="cuda:0",
                      memory_budgets={"cuda:0": 6 * 1024 ** 3}))
    assert sent[0] == (child.MSG_MEMORY_LIMIT, {"cuda:0": {
        "budget_bytes": 6 * 1024 ** 3, "total_bytes": 16 * 1024 ** 3,
        "fraction": 0.375}})
    assert sent[-1][1]["status"] == "succeeded"


def test_a_budget_over_the_card_is_clamped_and_mps_is_capped(monkeypatch):
    import sys
    from types import SimpleNamespace

    cuda = _CappedCuda()
    mps_caps: list[float] = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=cuda, mps=SimpleNamespace(
            recommended_max_memory=lambda: 8 * 1024 ** 3,
            set_per_process_memory_fraction=mps_caps.append)))
    applied = child.apply_memory_budgets({"cuda:1": 32 * 1024 ** 3,
                                          "mps:0": 2 * 1024 ** 3})
    assert cuda.caps == [(1.0, 1)]
    assert mps_caps == [0.25]
    assert applied["cuda:1"]["fraction"] == 1.0
    assert applied["mps:0"]["total_bytes"] == 8 * 1024 ** 3


def test_a_cap_that_cannot_be_set_fails_the_run_before_it_starts(
        monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=_CappedCuda(fail=True), backends=SimpleNamespace()))
    sent = _run(_spec(h.tensor_graph(), device="cuda:0",
                      memory_budgets={"cuda:0": 1024}))
    assert [message[0] for message in sent] == [child.MSG_FINISHED]
    assert sent[0][1]["status"] == "failed"
    assert "could not cap cuda:0's memory" in sent[0][1]["error"]
    with pytest.raises(RuntimeError, match="cpu has no allocator"):
        child.apply_memory_budgets({"cpu": 1024})

# ── in-run card names for the device_fallback report (#839) ───────────


def test_child_run_cards_read_the_agents_environment():
    """In-run CUDA name -> the worker's card (#839 review item 1)."""
    from app.agent.runner import child_environment

    def spec(device, devices):
        return {"env": child_environment(device, devices)}

    assert child.run_cards(spec("cuda:2", ["cuda:2", "cuda:0"])) == {
        "cuda:0": "cuda:2", "cuda:1": "cuda:0"}
    assert child.run_cards(spec("cpu", [])) == {}
    assert child.run_cards(spec("cuda", [])) is None
    assert child.run_cards({}) is None
