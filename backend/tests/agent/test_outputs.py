"""``record_outputs`` on a worker, the agent's half (#789).

The run subprocess's live store and its dump (``agent.child``), the reads
the runner hands to the subprocess (``Runner.ask_child``), the outputs
helper that reads the dumps, the agent's budget, and the frames on the
connection. The control plane's half is ``tests/test_worker_record_outputs.py``.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import torch

from app.agent import child
from app.agent import outputs as outputs_module
from app.agent.outputs import (
    AgentOutputs,
    OutputsHelper,
    outputs_dir,
    prune_outputs,
)
from app.agent.runner import Runner
from app.core import output_queries as q
from app.core import worker_protocol as wp
from app.core.run_output_store import RunOutputStore

from . import _helpers as h
from .test_connection import Agent, FakeControlPlane


async def _write(store: RunOutputStore, run_id: str, directory: Any) -> Any:
    """A run's captures written as its run subprocess writes them (#789)."""
    slots = await store.snapshot(run_id)
    return q.write_slots(run_id, q.capture_slots(slots), directory)


def _dump(runs: Path, run_id: str, nbytes: int, age: float) -> Path:
    directory = outputs_dir(runs, run_id)
    directory.mkdir(parents=True)
    index = directory / q.INDEX_FILE
    index.write_text(json.dumps({"run_id": run_id, "bytes": nbytes,
                                 "slots": []}))
    stamp = time.time() - age
    os.utime(index, (stamp, stamp))
    return directory


# ── the budget ───────────────────────────────────────────────────────────


def test_prune_drops_the_oldest_dumps_first(tmp_path):
    for run_id, age in (("old", 30), ("mid", 20), ("new", 10)):
        _dump(tmp_path, run_id, 40, age)
    (tmp_path / "running" / "outputs").mkdir(parents=True)  # no index yet
    (tmp_path / "broken" / "outputs").mkdir(parents=True)
    (tmp_path / "broken" / "outputs" / q.INDEX_FILE).write_text("{")
    assert prune_outputs(tmp_path, max_bytes=100) == ["old"]
    assert not outputs_dir(tmp_path, "old").exists()
    assert prune_outputs(tmp_path, max_bytes=0, max_runs=1) == ["mid"]
    assert outputs_dir(tmp_path, "new").exists()
    assert (tmp_path / "running" / "outputs").exists()


def test_the_run_just_written_is_kept_even_over_budget(tmp_path):
    _dump(tmp_path, "older", 10, 20)
    _dump(tmp_path, "huge", 500, 30)  # oldest by time, yet the one kept
    assert prune_outputs(tmp_path, max_bytes=100, keep="huge") == ["older"]
    assert outputs_dir(tmp_path, "huge").exists()


def test_prune_without_a_runs_directory(tmp_path):
    assert prune_outputs(tmp_path / "missing", max_bytes=1) == []


# ── the child: live store, then its dump ─────────────────────────────────


class _Loop:
    """An event loop on a thread of its own, as the run's is."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever,
                                       daemon=True)
        self.thread.start()

    def run(self, coroutine: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(30)

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(10)
        self.loop.close()


def _collect(sent: list[tuple], count: int) -> list[tuple]:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        answers = [m for m in sent if m[0] == child.MSG_OUTPUT_ANSWER]
        if len(answers) >= count:
            return answers
        time.sleep(0.01)
    raise AssertionError(f"{count} answers expected, got {sent}")


def test_live_outputs_answer_from_the_store_then_from_the_files(tmp_path):
    sent: list[tuple] = []
    store = RunOutputStore(max_bytes=0)
    live = child.LiveOutputs(store, "r1", str(tmp_path / "outputs"),
                             sent.append)
    # No loop yet (the run has not started): nothing is held.
    live.answer("a0", "list", "r1", {})
    assert _collect(sent, 1)[0][1:] == (
        "a0", 404, json.dumps({"detail": "run 'r1' not found"},
                              separators=(",", ":")), False)
    looper = _Loop()
    try:
        live.loop = looper.loop
        looper.run(store.put("r1", "x", "t", torch.arange(6.0)))
        live.answer("a1", "value", "r1", {"node_id": "x", "port": "t",
                                          "slice": "0:2"})
        answer = _collect(sent, 2)[1]
        assert answer[1:3] == ("a1", 200)
        assert json.loads(answer[3])["values"] == [0.0, 1.0]
        assert answer[4] is False
        # The run ends: its captures are copied off the store (which is
        # emptied) and read from the copies, final, before any file exists.
        looper.run(live.prepare())
        assert isinstance(live.disk, q.MemoryRunReader) and live.saving
        assert not looper.run(store.has_run("r1"))
        assert not (tmp_path / "outputs").exists()
        live.answer("a2", "stats", "r1", {"node_id": "x", "port": "t"})
        answer = _collect(sent, 3)[2]
        assert (answer[2], answer[4]) == (200, True)
        assert json.loads(answer[3])["count"] == 6
        live.save()
        assert isinstance(live.disk, q.DiskRunReader) and not live.saving
        live.answer("a3", "value", "r1", {"node_id": "x", "port": "t",
                                          "slice": "4:"})
        answer = _collect(sent, 4)[3]
        assert json.loads(answer[3])["values"] == [4.0, 5.0]
        assert answer[4] is True
    finally:
        looper.close()
    # A drop after the run ended: the files are the agent's to delete, and
    # every read from now on answers 404, not 500 from a deleted directory.
    live.drop("r1")
    assert live.disk is None and live.dropped
    import shutil

    shutil.rmtree(tmp_path / "outputs")
    live.answer("a4", "list", "r1", {})
    assert _collect(sent, 5)[4][2] == 404
    live.save()  # nothing left to write


def test_a_drop_while_the_run_goes_empties_its_store(tmp_path):
    sent: list[tuple] = []
    store = RunOutputStore(max_bytes=0)
    live = child.LiveOutputs(store, "r1", str(tmp_path / "outputs"),
                             sent.append)
    live.drop("r1")  # not started: nothing to do
    looper = _Loop()
    try:
        live.loop = looper.loop
        looper.run(store.put("r1", "x", "t", 1))
        live.drop("r1")
        deadline = time.monotonic() + 10
        while looper.run(store.has_run("r1")) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not looper.run(store.has_run("r1")) and not live.dropped
    finally:
        looper.close()
    live.drop("r1")  # a closed loop: nothing to do
    live.answer("a", "list", "r1", {})
    assert _collect(sent, 1)[0][2] == 404


def test_a_drop_while_the_copies_are_written_writes_nothing(tmp_path):
    store = RunOutputStore(max_bytes=0)
    live = child.LiveOutputs(store, "r1", str(tmp_path / "outputs"), print)

    async def ended() -> None:
        await store.put("r1", "x", "t", torch.ones(3))
        await store.put("r1", "x", "u", torch.ones(3))
        await live.prepare()

    asyncio.run(ended())
    calls = {"n": 0}

    def abandoned() -> bool:
        calls["n"] += 1
        if calls["n"] == 2:
            live.drop("r1")
        return live.dropped

    entries = live._entries
    assert q.write_slots("r1", entries, tmp_path / "outputs",
                         abandoned=abandoned) is None
    assert not (tmp_path / "outputs").exists()
    # And through save(): a drop already made writes nothing at all.
    live.save()
    assert not (tmp_path / "outputs").exists()


def test_a_drop_while_the_captures_are_copied_keeps_nothing(tmp_path,
                                                            monkeypatch):
    store = RunOutputStore(max_bytes=0)
    live = child.LiveOutputs(store, "r1", str(tmp_path / "outputs"), print)
    copy = q.capture_slots

    def dropped_meanwhile(slots: Any) -> Any:
        live._drops += 1
        return copy(slots)

    monkeypatch.setattr(q, "capture_slots", dropped_meanwhile)

    async def ended() -> None:
        await store.put("r1", "x", "t", 1)
        await live.prepare()

    asyncio.run(ended())
    assert live.disk is None and not live.saving


def test_captures_that_cannot_be_copied_are_a_warning(tmp_path,
                                                     monkeypatch):
    sent: list[tuple] = []
    store = RunOutputStore(max_bytes=0)
    live = child.LiveOutputs(store, "r1", str(tmp_path / "outputs"),
                             sent.append)

    def boom(_: Any) -> Any:
        raise MemoryError("no room")

    monkeypatch.setattr(q, "capture_slots", boom)

    async def go() -> None:
        await store.put("r1", "x", "t", 1)
        await live.prepare()

    asyncio.run(go())
    assert live.disk is None and not live.saving
    [warning] = [m for m in sent if m[0] == child.MSG_WARNING]
    assert warning[1]["kind"] == "outputs_not_kept"
    assert "no room" in warning[1]["detail"]


def test_copies_that_cannot_be_written_are_logged(tmp_path):
    sent: list[tuple] = []
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    store = RunOutputStore(max_bytes=0)
    live = child.LiveOutputs(store, "r1", str(blocker / "outputs"),
                             sent.append)

    async def go() -> None:
        await store.put("r1", "x", "t", 1)
        await live.prepare()

    asyncio.run(go())
    live.save()  # after finished: logged, never raised
    assert not live.saving and sent == []
    # The copies still answer until the process exits.
    assert isinstance(live.disk, q.MemoryRunReader)


def test_nothing_recorded_writes_nothing(tmp_path):
    store = RunOutputStore(max_bytes=0)
    live = child.LiveOutputs(store, "r1", str(tmp_path / "out"), print)
    asyncio.run(live.prepare())
    assert not live.saving
    live.save()
    assert not (tmp_path / "out").exists()
    no_dir = child.LiveOutputs(store, "r1", None, print)
    asyncio.run(no_dir.prepare())


def test_a_crashing_read_is_answered_500(tmp_path, monkeypatch):
    sent: list[tuple] = []
    live = child.LiveOutputs(RunOutputStore(max_bytes=0), "r1", None,
                             sent.append)

    async def boom(*_: Any, **__: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(q, "dispatch", boom)
    live.answer("a", "list", "r1", {})
    answer = _collect(sent, 1)[0]
    assert answer[2] == 500 and "failed" in json.loads(answer[3])["detail"]


def test_the_listener_routes_reads_and_drops():
    class Pipe:
        def __init__(self, messages: list[Any]) -> None:
            self.messages = list(messages)

        def recv(self) -> Any:
            if not self.messages:
                raise EOFError
            return self.messages.pop(0)

    class Outputs:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def answer(self, *args: Any) -> None:
            self.calls.append(("answer", *args))

        def drop(self, run_id: str) -> None:
            self.calls.append(("drop", run_id))

    class Context:
        cancelled = 0

        def cancel(self) -> None:
            self.cancelled += 1

    outputs, context = Outputs(), Context()
    child._listen_for_cancel(Pipe([
        (), "junk", (child.MSG_FETCH, "a", "list", "r1", {}),
        (child.MSG_FETCH, "short"), (child.MSG_DROP_OUTPUTS, "r1"),
        (child.MSG_CANCEL,)]), context, outputs)
    assert outputs.calls == [("answer", "a", "list", "r1", {}),
                             ("drop", "r1")]
    assert context.cancelled == 2  # the cancel, then EOF


def test_a_recorded_run_reports_before_it_writes(tmp_path):
    """``run_assignment`` with ``record_outputs``: ``finished`` (with
    ``saving_outputs``) goes before the files are written, so neither a
    cancel's grace nor the run's device slot waits for the disk; the files
    follow before the process ends."""
    directory = tmp_path / "outputs"
    sent: list[tuple] = []
    spec = {"run_id": "rec1", "graph": h.tensor_graph(), "device": "cpu",
            "options": {**h.TENSOR_OPTIONS, "record_outputs": True},
            "outputs_dir": str(directory)}

    def send(message: tuple) -> None:
        if message[0] == child.MSG_FINISHED:
            sent.append(("index at finish",
                         q.read_index(directory) is not None))
        sent.append(message)

    thread = threading.Thread(target=child.run_assignment, args=(spec, send),
                              kwargs={"discover": False})
    thread.start()
    thread.join(60)
    assert ("index at finish", False) in sent
    assert sent[-1][1]["status"] == "succeeded"
    assert sent[-1][1]["saving_outputs"] is True
    assert q.read_index(directory) is not None
    reader = q.DiskRunReader(directory)
    status, body = asyncio.run(q.dispatch(
        reader, "value", "rec1", {"node_id": "scale", "port": "tensor"},
        compute=None))
    assert status == 200 and body["values"] == [[2.5] * 3] * 2


# ── the runner asks its subprocess ───────────────────────────────────────


def _assign(run_id: str, record: bool = True) -> wp.Assign:
    return wp.Assign(run_id=run_id, graph={"nodes": []},
                     options={"record_outputs": record})


async def test_ask_child_gets_the_subprocess_answer(tmp_path):
    sink = h.FrameSink()
    runner = Runner(runs_dir=tmp_path, offered_devices={"cpu"}, send=sink,
                    child_entry=h.answering_child, outputs_max_bytes=0)
    await runner.assign(_assign("r1"))
    assert runner.runs["r1"].record_outputs
    answer = await runner.ask_child("r1", "value", {"node_id": "n"})
    assert answer == (200, json.dumps({"op": "value", "run_id": "r1",
                                       "params": {"node_id": "n"}}), False)
    assert runner.child_drop("r1") is True
    await sink.wait_for(lambda f: isinstance(f, wp.Event)
                        and f.event_type == "dropped")
    # The spec names where the subprocess writes its captures.
    assert await runner.ask_child("nope", "list", {}) is None
    assert runner.child_drop("nope") is False
    handle = runner.runs["r1"]
    await runner.cancel("r1")
    await sink.wait_for(lambda f: isinstance(f, wp.RunFinished))
    await asyncio.wait_for(handle.done.wait(), 30)
    assert await runner.ask_child("r1", "list", {}) is None


async def test_a_read_the_subprocess_leaves_unanswered_goes_to_the_files(
        tmp_path):
    sink = h.FrameSink()
    runner = Runner(runs_dir=tmp_path, offered_devices={"cpu"}, send=sink,
                    child_entry=h.silent_child, outputs_max_bytes=0)
    await runner.assign(_assign("r1"))
    assert await runner.ask_child("r1", "list", {}) is None
    await sink.wait_for(lambda f: isinstance(f, wp.RunFinished))


async def test_a_run_without_record_outputs_is_never_asked(tmp_path):
    sink = h.FrameSink()
    runner = Runner(runs_dir=tmp_path, offered_devices={"cpu"}, send=sink,
                    child_entry=h.answering_child, outputs_max_bytes=0,
                    kill_after_s=2.0)
    await runner.assign(_assign("r1", record=False))
    # Stop the child only once it runs: a Windows child killed while it
    # still starts up never reports (the runner tests wait the same way).
    await sink.wait_for(lambda f: isinstance(f, wp.Event))
    assert await runner.ask_child("r1", "list", {}) is None
    assert runner.child_drop("r1") is False
    await runner.cancel("r1", grace_s=0.5)
    await sink.wait_for(lambda f: isinstance(f, wp.RunFinished), timeout=20)


async def test_a_subprocess_that_does_not_answer_in_time(tmp_path):
    sink = h.FrameSink()
    runner = Runner(runs_dir=tmp_path, offered_devices={"cpu"}, send=sink,
                    child_entry=h.deaf_child, outputs_max_bytes=0,
                    kill_after_s=2.0)
    await runner.assign(_assign("r1"))
    await sink.wait_for(lambda f: isinstance(f, wp.Event))
    status, text, final = await runner.ask_child("r1", "list", {},
                                                 timeout_s=0.1)
    assert status == 504 and "did not answer" in json.loads(text)["detail"]
    await runner.cancel("r1", grace_s=0.5)
    await sink.wait_for(lambda f: isinstance(f, wp.RunFinished), timeout=20)


def test_the_runner_budget_comes_from_the_setting(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "RUN_OUTPUT_STORE_MAX_MB", 3)
    assert Runner(runs_dir=tmp_path, offered_devices={"cpu"}
                  ).outputs_max_bytes == 3 * 1024 * 1024


def test_retiring_a_recorded_run_applies_the_budget(tmp_path):
    runner = Runner(runs_dir=tmp_path, offered_devices={"cpu"},
                    outputs_max_bytes=50)
    _dump(tmp_path, "old", 40, 20)
    _dump(tmp_path, "new", 40, 10)
    runner._retire_files(tmp_path / "new", recorded=True)
    assert not outputs_dir(tmp_path, "old").exists()
    assert outputs_dir(tmp_path, "new").exists()


# ── the helper that reads the files ──────────────────────────────────────


async def test_the_real_helper_reads_a_dump(tmp_path):
    store = RunOutputStore(max_bytes=0)
    await store.put("r1", "x", "t", torch.arange(4.0))
    await _write(store, "r1", tmp_path / "outputs")
    helper = OutputsHelper(idle_s=60)
    try:
        status, text, final = await helper.ask(
            tmp_path / "outputs", "r1", "value",
            {"node_id": "x", "port": "t"})
        assert (status, final) == (200, True)
        assert json.loads(text)["values"] == [0.0, 1.0, 2.0, 3.0]
        status, _, _ = await helper.ask(tmp_path / "outputs", "r1", "value",
                                        {"node_id": "x", "port": "gone"})
        assert status == 404
        assert helper.running
    finally:
        helper.close()


async def test_a_helper_that_dies_fails_its_reads_and_restarts(tmp_path):
    helper = OutputsHelper(entry=h.dying_helper, idle_s=60)
    status, text, _ = await helper.ask(tmp_path, "r1", "list", {})
    assert status == 500 and "exited (code 5)" in json.loads(text)["detail"]
    helper.entry = h.echo_helper
    status, text, _ = await helper.ask(tmp_path, "r1", "list", {})
    assert status == 200 and json.loads(text)["op"] == "list"
    helper.close()


async def test_a_slow_helper_times_out_and_an_idle_one_stops(tmp_path):
    helper = OutputsHelper(entry=h.echo_helper, idle_s=0.2, timeout_s=0.1)
    status, text, _ = await helper.ask(tmp_path, "r1", "slow", {})
    assert status == 504
    deadline = time.monotonic() + 15
    while helper.running and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    assert not helper.running
    helper.close()  # already stopped: harmless


async def test_a_helper_whose_pipe_is_closed_says_so(tmp_path):
    helper = OutputsHelper(entry=h.echo_helper, idle_s=60)
    helper.start()
    helper._current.requests.close()
    status, text, _ = await helper.ask(tmp_path, "r1", "list", {})
    assert status == 500 and "not reachable" in json.loads(text)["detail"]
    helper.close()


# ── the agent answers fetch_output ───────────────────────────────────────


class _Runner:
    def __init__(self, runs_dir: Path, answer: Any = None,
                 live: bool = False) -> None:
        self.runs_dir = runs_dir
        self.answer = answer
        self.live = live
        self.dropped: list[str] = []

    async def ask_child(self, run_id: str, op: str, params: Any,
                        **_: Any) -> Any:
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer

    def child_drop(self, run_id: str) -> bool:
        self.dropped.append(run_id)
        return self.live


class _Helper:
    def __init__(self) -> None:
        self.asked: list[tuple] = []
        self.closed = False

    async def ask(self, directory: Path, run_id: str, op: str,
                  params: Any, **_: Any) -> Any:
        self.asked.append((directory, run_id, op))
        return 200, "[]", True

    def close(self) -> None:
        self.closed = True


def _agent_outputs(runner: _Runner) -> tuple[AgentOutputs, list[tuple]]:
    sent: list[tuple] = []

    async def send(*args: Any) -> None:
        sent.append(args)

    return AgentOutputs(runner, send, helper=_Helper()), sent


def _fetch(op: str, run_id: str = "r1") -> wp.FetchOutput:
    return wp.FetchOutput(request_id="q1", run_id=run_id, op=op)


async def test_a_live_run_is_answered_by_its_subprocess(tmp_path):
    outputs, sent = _agent_outputs(_Runner(tmp_path, (200, "{}", False)))
    await outputs.serve(_fetch("list"))
    assert sent == [("q1", 200, "{}", False)]
    assert outputs.helper.asked == []


async def test_an_ended_run_is_answered_from_its_files(tmp_path):
    _dump(tmp_path, "r1", 1, 0)
    outputs, sent = _agent_outputs(_Runner(tmp_path))
    await outputs.serve(_fetch("value"))
    assert sent == [("q1", 200, "[]", True)]
    assert outputs.helper.asked == [(outputs_dir(tmp_path, "r1"), "r1",
                                     "value")]


@pytest.mark.parametrize("op, detail", [
    ("list", "run 'r1' not found"), ("value", "run 'r1' not found"),
    ("steps", "run 'r1' not found"), ("grads", "run 'r1' not found"),
    ("stats", q.not_captured("run 'r1' has no captured outputs"))])
async def test_no_files_answer_as_the_reads_do(tmp_path, op, detail):
    outputs, sent = _agent_outputs(_Runner(tmp_path))
    await outputs.serve(_fetch(op))
    assert sent == [("q1", 404, json.dumps({"detail": detail}), True)]
    assert outputs.helper.asked == []
    store = RunOutputStore(max_bytes=0)
    assert (await q.dispatch(store, op, "r1", {}, compute=None))[1] == {
        "detail": detail}


async def test_delete_and_drop(tmp_path):
    outputs, sent = _agent_outputs(_Runner(tmp_path))
    await outputs.serve(_fetch("delete"))
    assert sent[-1][:2] == ("q1", 404)
    _dump(tmp_path, "r1", 1, 0)
    await outputs.serve(_fetch("delete"))
    assert sent[-1] == ("q1", 200, json.dumps({"run_id": "r1",
                                               "deleted": True}), True)
    assert not outputs_dir(tmp_path, "r1").exists()
    live, sent = _agent_outputs(_Runner(tmp_path, live=True))
    await live.serve(_fetch("delete"))
    assert sent[-1][1] == 200
    _dump(tmp_path, "r2", 1, 0)
    await outputs.drop("r2")
    assert not outputs_dir(tmp_path, "r2").exists()
    await outputs.drop("r3")  # nothing there: nothing to do
    outputs.close()
    assert outputs.helper.closed


async def test_a_failing_read_is_answered_500_and_a_lost_send_is_quiet(
        tmp_path):
    outputs, sent = _agent_outputs(_Runner(tmp_path, RuntimeError("bad")))
    await outputs.serve(_fetch("list"))
    assert sent[-1][1] == 500 and "bad" in json.loads(sent[-1][2])["detail"]

    async def gone(*_: Any) -> None:
        raise ConnectionError

    outputs.send_answer = gone
    await outputs.serve(_fetch("list"))  # does not raise


# ── over the connection ──────────────────────────────────────────────────


async def test_the_connection_answers_fetch_output_in_parts(tmp_path,
                                                            monkeypatch):
    monkeypatch.setattr(wp, "OUTPUT_ANSWER_PART_CHARS", 16)
    started: list[bool] = []
    monkeypatch.setattr(outputs_module.OutputsHelper, "start",
                        lambda self: started.append(True))
    async with FakeControlPlane() as cp, Agent(
            tmp_path, cp.url, child_entry=h.answering_child) as agent:
        await cp.wait_for(lambda n, f: isinstance(f, wp.Hello))
        hello = cp.of_type(wp.Hello)[0]
        assert wp.CAPABILITY_RECORD_OUTPUTS in hello.capabilities
        await cp.send(_assign("r1"))
        await cp.wait_for(lambda n, f: isinstance(f, wp.RunStarted))
        await cp.send(wp.FetchOutput(request_id="q9", run_id="r1",
                                     op="value", params={"node_id": "n"}))
        last = await cp.wait_for(
            lambda n, f: isinstance(f, wp.OutputAnswer)
            and f.part == f.parts - 1)
        parts = [f for f in cp.of_type(wp.OutputAnswer)
                 if f.request_id == "q9"]
        assert len(parts) == last.parts > 1
        assert all(len(p.body) <= 16 for p in parts)
        body = json.loads("".join(p.body for p in parts))
        assert body == {"op": "value", "run_id": "r1",
                        "params": {"node_id": "n"}}
        await cp.send(wp.DropOutputs(run_id="r1"))
        await cp.wait_for(lambda n, f: isinstance(f, wp.Event)
                          and f.event_type == "dropped")
        handle = agent.runner.runs["r1"]
        await cp.send(wp.Cancel(run_id="r1"))
        await cp.wait_for(lambda n, f: isinstance(f, wp.RunFinished))
        await asyncio.wait_for(handle.done.wait(), 30)
        # A recorded run ending starts nothing: the torch-importing helper
        # waits for a read of the files (it starts on the first).
        assert started == []


async def test_an_empty_answer_is_one_part_and_a_closed_socket_stops(
        tmp_path, monkeypatch):
    from app.agent.connection import AgentConnection

    started: list[bool] = []
    monkeypatch.setattr(outputs_module.OutputsHelper, "start",
                        lambda self: started.append(True))

    runner = Runner(runs_dir=tmp_path, offered_devices={"cpu"},
                    outputs_max_bytes=0)
    connection = AgentConnection(
        runner=runner, name="box", version="t", control_url="http://x",
        token="t", probe_devices=list)
    await connection.send_answer("q1", 204, "", True)  # not connected

    class Ws:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, line: str) -> None:
            self.sent.append(line)

    connection._ws = Ws()
    await connection.send_answer("q1", 204, "", True)
    [line] = connection._ws.sent
    assert json.loads(line)["parts"] == 1 and json.loads(line)["body"] == ""
    assert started == []
    connection.outputs.close()


async def test_a_helper_that_cannot_start_answers_500(tmp_path, monkeypatch,
                                                      caplog):
    def broken(self: Any) -> None:
        raise OSError("no spawn")

    monkeypatch.setattr(outputs_module.OutputsHelper, "start", broken)
    _dump(tmp_path, "r1", 1, 0)
    sent: list[tuple] = []

    async def send(*args: Any) -> None:
        sent.append(args)

    outputs = AgentOutputs(_Runner(tmp_path), send)
    await outputs.serve(_fetch("list"))
    assert sent[-1][1] == 500 and "no spawn" in json.loads(sent[-1][2])[
        "detail"]


# ── review fixes (#853) ──────────────────────────────────────────────────


async def test_a_child_writing_its_outputs_reports_now_and_gets_time(
        tmp_path):
    """A cancelled recorded run reports as soon as its captures are off
    its cards: the control plane frees its slot then, and the cancel's
    grace (0.2 s, then 0.5 s to kill) does not cut the write short."""
    marker = tmp_path / "written"
    sink = h.FrameSink()
    runner = Runner(runs_dir=tmp_path / "runs", offered_devices={"cpu"},
                    send=sink, child_entry=h.saving_child,
                    outputs_max_bytes=0, kill_after_s=0.5,
                    exit_after_finished_s=0.2, save_outputs_s=60)
    await runner.assign(wp.Assign(run_id="r1", graph={"nodes": []}, options={
        "record_outputs": True, "save_s": 4, "marker": str(marker)}))
    await sink.wait_for(lambda f: isinstance(f, wp.Event))
    handle = runner.runs["r1"]
    started = time.monotonic()
    await runner.cancel("r1", grace_s=0.2)
    finished = await sink.wait_for(lambda f: isinstance(f, wp.RunFinished),
                                   timeout=20)
    assert time.monotonic() - started < 3.5  # before the 4 s write ends
    assert finished.status == "cancelled"
    assert handle.reported_early and not marker.exists()
    # Not running any more, but still answering reads of what it holds.
    assert runner.running_ids() == [] and "r1" in runner.runs
    assert await runner.ask_child("r1", "list", {}) == (
        200, json.dumps({"saving": True}), True)
    await asyncio.wait_for(handle.done.wait(), 30)
    assert marker.read_text() == "written"
    assert len([f for f in sink.frames if isinstance(f, wp.RunFinished)]) == 1


async def test_a_write_over_its_time_limit_is_stopped(tmp_path):
    marker = tmp_path / "written"
    sink = h.FrameSink()
    runner = Runner(runs_dir=tmp_path / "runs", offered_devices={"cpu"},
                    send=sink, child_entry=h.saving_child,
                    outputs_max_bytes=0, kill_after_s=0.5,
                    save_outputs_s=0.3)
    await runner.assign(wp.Assign(run_id="r1", graph={"nodes": []}, options={
        "record_outputs": True, "save_s": 60, "marker": str(marker)}))
    await sink.wait_for(lambda f: isinstance(f, wp.Event))
    handle = runner.runs["r1"]
    await runner.cancel("r1", grace_s=0.2)
    await asyncio.wait_for(handle.done.wait(), 30)
    assert not marker.exists()
    [finished] = [f for f in sink.frames if isinstance(f, wp.RunFinished)]
    assert finished.status == "cancelled"  # as reported; not re-reported


async def test_a_closed_helper_starts_anew_for_the_next_read(tmp_path):
    """``close`` then a read at once: a new helper answers it (it used to
    find the old one's pipe gone, an AttributeError answered 500)."""
    helper = OutputsHelper(entry=h.echo_helper, idle_s=60)
    helper.start()
    helper.close()
    status, text, _ = await helper.ask(tmp_path, "r1", "list", {})
    assert status == 200 and json.loads(text)["op"] == "list"
    helper.close()


async def test_an_old_helper_exiting_fails_only_its_own_reads(tmp_path):
    helper = OutputsHelper(entry=h.echo_helper, idle_s=60, timeout_s=30)
    old = helper.start()
    helper.close()  # it exits
    read = asyncio.create_task(helper.ask(tmp_path, "r1", "slow", {}))
    deadline = time.monotonic() + 15
    while old.process.is_alive() and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.5)  # its exit reaches the loop
    assert not old.process.is_alive() and old.closed
    assert not read.done()  # the new helper's read is untouched
    assert helper.running
    read.cancel()
    helper.close()


def test_the_agent_gives_up_before_the_control_plane():
    def frame(timeout_s: float | None) -> wp.FetchOutput:
        return wp.FetchOutput(request_id="q", run_id="r", op="list",
                              timeout_s=timeout_s)

    assert outputs_module.answer_timeout(frame(None)) == 55.0
    assert outputs_module.answer_timeout(frame(60)) == 55.0
    assert outputs_module.answer_timeout(frame(4)) == 2.0
    from app.agent.runner import ASK_CHILD_TIMEOUT_S

    assert ASK_CHILD_TIMEOUT_S < outputs_module.DEFAULT_FETCH_TIMEOUT_S
    assert OutputsHelper().timeout_s < outputs_module.DEFAULT_FETCH_TIMEOUT_S
    assert outputs_module.MAX_RUNS == RunOutputStore().max_runs


async def test_the_wait_rides_on_the_frame_and_cancel_frees_the_place(
        tmp_path):
    asked: list[float] = []
    gate = asyncio.Event()

    class Slow(_Runner):
        async def ask_child(self, run_id: str, op: str, params: Any,
                            timeout_s: float = 0) -> Any:
            asked.append(timeout_s)
            await gate.wait()
            return 200, "{}", False

    outputs, sent = _agent_outputs(Slow(tmp_path))
    tasks = [outputs.start(wp.FetchOutput(request_id=f"q{i}", run_id="r1",
                                          op="list", timeout_s=20))
             for i in range(outputs_module.MAX_CONCURRENT + 1)]
    await asyncio.sleep(0.05)
    assert asked == [15.0] * outputs_module.MAX_CONCURRENT  # one waits
    assert all(outputs.cancel(f"q{i}") for i in range(len(tasks)))
    await asyncio.gather(*tasks, return_exceptions=True)
    assert sent == [] and not outputs.cancel("q0")  # done: nothing to cancel
    # Every place is free again.
    gate.set()
    await outputs.start(_fetch("list"))
    assert sent == [("q1", 200, "{}", False)]
    outputs.close()


async def test_drop_outputs_does_not_hold_up_the_frame_loop(tmp_path,
                                                            monkeypatch):
    from app.agent.connection import AgentConnection

    runner = Runner(runs_dir=tmp_path, offered_devices={"cpu"},
                    outputs_max_bytes=0)
    connection = AgentConnection(
        runner=runner, name="box", version="t", control_url="http://x",
        token="t", probe_devices=list)
    gate = asyncio.Event()
    dropped: list[str] = []

    async def slow_drop(run_id: str) -> None:
        await gate.wait()
        dropped.append(run_id)

    monkeypatch.setattr(connection.outputs, "drop", slow_drop)
    await asyncio.wait_for(connection._dispatch(wp.DropOutputs(run_id="r1")),
                           1)
    assert dropped == []  # still deleting, and the next frame is read
    cancelled: list[str] = []
    monkeypatch.setattr(connection.outputs, "cancel", cancelled.append)
    await connection._dispatch(wp.CancelFetch(request_id="q1"))
    assert cancelled == ["q1"]
    gate.set()
    for _ in range(50):
        if dropped:
            break
        await asyncio.sleep(0.01)
    assert dropped == ["r1"]
    connection.outputs.close()


def test_numpy_scalars_and_primitive_subclasses_load_back(tmp_path):
    """``np.float64`` subclasses ``float`` and used to be saved as numpy's
    object, which ``weights_only`` refuses: every read of the run, the list
    included, answered 500."""
    import enum

    import numpy as np

    class Level(enum.IntEnum):
        HIGH = 2

    store = RunOutputStore(max_bytes=0)

    async def go() -> None:
        await store.put("r1", "m", "loss", np.float64(0.25))
        await store.put("r1", "m", "name", np.str_("adam"))
        await store.put("r1", "m", "flags", {"ok": np.bool_(True),
                                             "n": [np.int64(3)]})
        await store.put("r1", "m", "level", Level.HIGH)
        await store.put("r1", "m", "t", torch.ones(2, 3))
        await _write(store, "r1", tmp_path)

    asyncio.run(go())
    reader = q.DiskRunReader(tmp_path)

    def read(op: str, **params: Any) -> Any:
        return asyncio.run(q.dispatch(reader, op, "r1", params, compute=None))

    status, listed = read("list")
    assert status == 200
    assert {row["port"]: row["type"] for row in listed} == {
        "loss": "scalar", "name": "string", "flags": "dict",
        "level": "scalar", "t": "tensor"}
    assert read("value", node_id="m", port="loss")[1]["value"] == 0.25
    assert read("value", node_id="m", port="name")[1]["value"] == "adam"
    assert read("value", node_id="m", port="level")[0] == 200
    assert type(q.portable(np.float64(0.25))) is float
    assert q.portable({np.int64(1): 2})["__codefyui_stub__"] == "opaque"


def test_one_unreadable_file_fails_only_its_own_slot(tmp_path):
    store = RunOutputStore(max_bytes=0)

    async def go() -> None:
        await store.put("r1", "a", "t", torch.ones(2))
        await store.put("r1", "b", "t", torch.zeros(3))
        await _write(store, "r1", tmp_path)

    asyncio.run(go())
    index = q.read_index(tmp_path)
    broken = tmp_path / index["slots"][0]["file"]
    broken.write_bytes(b"not a torch file")
    # An index without the listed facts (an older dump) loads each slot.
    for slot in index["slots"]:
        slot.pop("type"), slot.pop("full_shape")
    (tmp_path / q.INDEX_FILE).write_text(json.dumps(index))
    reader = q.DiskRunReader(tmp_path)
    status, listed = asyncio.run(q.dispatch(reader, "list", "r1", {},
                                            compute=None))
    assert status == 200
    assert listed == [
        {"node_id": "a", "port": "t", "type": q.UNREADABLE,
         "full_shape": None},
        {"node_id": "b", "port": "t", "type": "tensor", "full_shape": [3]}]
    status, _ = asyncio.run(q.dispatch(
        reader, "value", "r1", {"node_id": "a", "port": "t"}, compute=None))
    assert status == 500


def test_a_list_reads_the_index_not_the_files(tmp_path, monkeypatch):
    store = RunOutputStore(max_bytes=0)

    async def go() -> None:
        for step in range(300):
            await store.put("r1", "loop", f"__step__{step}__x",
                            torch.ones(4, step + 1))
        await store.put("r1", "card", "out", q.CaptureAlias("loop",
                                                            "__step__0__x"))
        await store.put("r1", "card", "gone", q.CaptureAlias("nope", "x"))
        await _write(store, "r1", tmp_path)

    asyncio.run(go())
    reader = q.DiskRunReader(tmp_path)

    def no_loads(*_: Any) -> Any:
        raise AssertionError("a list must not load a value")

    monkeypatch.setattr(reader, "_load", no_loads)
    status, listed = asyncio.run(q.dispatch(reader, "list", "r1", {},
                                            compute=None))
    assert status == 200 and len(listed) == 302
    assert listed[7] == {"node_id": "loop", "port": "__step__7__x",
                         "type": "tensor", "full_shape": [4, 8]}
    assert listed[300]["full_shape"] == [4, 1]  # the alias, followed
    assert listed[301]["type"] == "NoneType"  # an alias to nothing
    # The in-memory copies answer the same list.
    slots = asyncio.run(store.snapshot("r1"))
    memory = q.MemoryRunReader("r1", q.capture_slots(slots))
    assert asyncio.run(q.dispatch(memory, "list", "r1", {},
                                  compute=None)) == (200, listed)


# ── second review (#853) ─────────────────────────────────────────────────


def test_the_copies_answer_before_the_store_is_emptied(tmp_path):
    """No read between the store being emptied and the copies being
    published may find nothing (a 404 the Inspector shows as expired)."""
    store = RunOutputStore(max_bytes=0)
    live = child.LiveOutputs(store, "r1", str(tmp_path / "outputs"), print)
    seen: list[Any] = []
    delete = store.delete_run

    async def watched(run_id: str) -> bool:
        seen.append(live.disk)
        return await delete(run_id)

    store.delete_run = watched  # type: ignore[method-assign]

    async def go() -> None:
        await store.put("r1", "x", "t", torch.ones(2))
        await live.prepare()

    asyncio.run(go())
    assert isinstance(seen[0], q.MemoryRunReader)
    assert not asyncio.run(store.has_run("r1"))


def test_a_read_after_the_run_ended_does_not_wait_on_its_loop(
        tmp_path, monkeypatch):
    sent: list[tuple] = []
    store = RunOutputStore(max_bytes=0)
    live = child.LiveOutputs(store, "r1", None, sent.append)
    # A loop that exists but never runs anything (one shutting down).
    stalled = asyncio.new_event_loop()
    try:
        live.loop = stalled
        monkeypatch.setattr(child, "LOOP_CALL_TIMEOUT_S", 0.2)
        started = time.monotonic()
        live.answer("a1", "list", "r1", {})
        answer = _collect(sent, 1)[0]
        assert answer[2] == 500 and time.monotonic() - started < 10
        assert "did not answer" in json.loads(answer[3])["detail"]
        # Once the run has ended it is not asked at all.
        live.closing = True
        live.answer("a2", "list", "r1", {})
        assert _collect(sent, 2)[1][2] == 404
    finally:
        stalled.close()


def test_run_assignment_closes_reads_before_its_loop_ends(tmp_path):
    spec = {"run_id": "r0", "graph": h.tensor_graph(), "device": "cpu",
            "options": {**h.TENSOR_OPTIONS, "record_outputs": True},
            "outputs_dir": None}
    made: list[Any] = []
    original = child.LiveOutputs

    def recording(*args: Any, **kw: Any) -> Any:
        made.append(original(*args, **kw))
        return made[-1]

    child.LiveOutputs = recording  # type: ignore[misc]
    try:
        child.run_assignment(spec, lambda message: None, discover=False)
    finally:
        child.LiveOutputs = original  # type: ignore[misc]
    assert made[0].closing


async def test_one_deadline_covers_the_whole_read(tmp_path):
    _dump(tmp_path, "r1", 1, 0)
    waits: list[float] = []

    class Late(_Runner):
        async def ask_child(self, *_: Any, timeout_s: float = 0) -> Any:
            waits.append(timeout_s)
            await asyncio.sleep(1.0)
            return None  # it exited meanwhile: the files answer

    class Helper(_Helper):
        async def ask(self, *args: Any, timeout_s: float = 0) -> Any:
            waits.append(timeout_s)
            return 200, "[]", True

    sent: list[tuple] = []

    async def send(*args: Any) -> None:
        sent.append(args)

    outputs = AgentOutputs(Late(tmp_path), send, helper=Helper())
    await outputs.serve(wp.FetchOutput(request_id="q", run_id="r1",
                                       op="list", timeout_s=10))
    assert waits[0] == 5.0
    assert 3.5 < waits[1] <= 4.05  # what is left of the same 5 s
    assert sent == [("q", 200, "[]", True)]


def test_the_helper_keeps_a_reader_while_the_index_is_unchanged(tmp_path):
    made: list[Path] = []

    def make(directory: Path) -> Any:
        made.append(directory)
        return object()

    cache = outputs_module._ReaderCache(make, size=2)
    one = _dump(tmp_path, "r1", 1, 30)
    first = cache.get(one)
    assert cache.get(one) is first and made == [one]
    # Rewritten (a new size and time): read again.
    (one / q.INDEX_FILE).write_text(json.dumps({"run_id": "r1", "bytes": 22,
                                                "slots": []}))
    assert cache.get(one) is not first and made == [one, one]
    # No index: never kept.
    missing = tmp_path / "nothing"
    cache.get(missing)
    cache.get(missing)
    assert made.count(missing) == 2
    # Bounded, least recently used first out.
    two = _dump(tmp_path, "r2", 1, 20)
    three = _dump(tmp_path, "r3", 1, 10)
    cache.get(two)
    cache.get(three)
    cache.get(one)
    assert made.count(one) == 3


def test_the_write_limit_comes_from_the_setting(tmp_path, monkeypatch):
    from app.config import settings

    assert settings.WORKER_OUTPUTS_WRITE_TIMEOUT_S == 300.0
    monkeypatch.setattr(settings, "WORKER_OUTPUTS_WRITE_TIMEOUT_S", 7.5)
    assert Runner(runs_dir=tmp_path, offered_devices={"cpu"}
                  ).save_outputs_s == 7.5
    assert Runner(runs_dir=tmp_path, offered_devices={"cpu"},
                  save_outputs_s=2).save_outputs_s == 2


async def test_a_graceful_stop_waits_for_a_write_in_progress(tmp_path):
    marker = tmp_path / "written"
    sink = h.FrameSink()
    runner = Runner(runs_dir=tmp_path / "runs", offered_devices={"cpu"},
                    send=sink, child_entry=h.saving_child,
                    outputs_max_bytes=0, kill_after_s=0.5,
                    save_outputs_s=60)
    await runner.assign(wp.Assign(run_id="r1", graph={"nodes": []}, options={
        "record_outputs": True, "save_s": 3, "marker": str(marker)}))
    await sink.wait_for(lambda f: isinstance(f, wp.Event))
    await runner.cancel("r1", grace_s=0.2)
    await sink.wait_for(lambda f: isinstance(f, wp.RunFinished), timeout=20)
    assert not marker.exists()
    await asyncio.wait_for(runner.shutdown(grace_s=0.2), 30)
    assert marker.read_text() == "written"
    assert runner.runs == {}
