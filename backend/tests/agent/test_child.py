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
                              "meta": {"epoch": 1}, "node_id": "a"}),
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
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")


def test_safe_sender_swallows_a_closed_pipe():
    reader, writer = multiprocessing.Pipe(duplex=False)
    reader.close()
    writer.close()
    child._safe_sender(writer)(("event",))  # no exception
