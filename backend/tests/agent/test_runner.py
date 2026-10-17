"""``app.agent.runner``: one spawned subprocess per run (#771).

The subprocess tests use the cheap entry points in ``_helpers`` (no torch,
no node discovery) except the two that must run the real ``child.main``:
the torch-free/byte-identity check and the cooperative cancel (which runs
the real engine with one registered node).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from app.agent.runner import (
    FINISHED_MARKER,
    AssignmentRefused,
    Runner,
    canonical_device,
    child_device,
    child_environment,
    mark_finished,
    prune_run_dirs,
    read_tail,
)
from app.core import worker_protocol as wp

from . import _helpers as h

BACKEND_DIR = Path(__file__).resolve().parents[2]


def _runner(tmp_path: Path, sink: h.FrameSink, **kw) -> Runner:
    kw.setdefault("offered_devices", {"cpu", "mps"})
    return Runner(runs_dir=tmp_path / "agent-runs", send=sink, **kw)


def _assign(run_id: str, **kw) -> wp.Assign:
    kw.setdefault("graph", {"nodes": [], "edges": []})
    return wp.Assign(run_id=run_id, **kw)


# ── pure helpers ──────────────────────────────────────────────────────────


def test_child_device_renumbers_into_the_visible_set():
    assert child_device("cuda:5", ["cuda:2", "cuda:5"]) == "cuda:1"
    assert child_device("cuda:2", ["cuda:2", "cuda:5"]) == "cuda:0"
    # No set, a bare kind and non-CUDA devices pass through unchanged.
    assert child_device("cuda:3", []) == "cuda:3"
    assert child_device("cuda", ["cuda:1"]) == "cuda"
    assert child_device("cpu", ["cpu"]) == "cpu"
    assert child_device("mps", []) == "mps"
    with pytest.raises(AssignmentRefused):
        child_device("cuda:1", ["cuda:2"])
    with pytest.raises(AssignmentRefused):
        child_device("tpu:0", [])


def test_child_environment_pins_cards_and_drops_the_database():
    env = child_environment("cuda:3", ["cuda:3", "cuda:1"])
    assert env["set"] == {"CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                          "CUDA_VISIBLE_DEVICES": "3,1"}
    assert set(env["unset"]) == {"CODEFYUI_DB_PATH", "CODEFYUI_AGENT_TOKEN",
                                 "HIP_VISIBLE_DEVICES",
                                 "ROCR_VISIBLE_DEVICES"}
    # No set: the run's own device decides.
    assert child_environment("cuda:2", [])["set"]["CUDA_VISIBLE_DEVICES"] == "2"
    # A run that names no card sees none.
    for device, devices in (("cpu", []), ("cpu", ["cpu"]), ("mps", [])):
        assert child_environment(device, devices)["set"][
            "CUDA_VISIBLE_DEVICES"] == ""
    # A bare cuda with no set keeps what the agent inherited.
    assert "CUDA_VISIBLE_DEVICES" not in child_environment("cuda", [])["set"]


def test_canonical_device():
    assert canonical_device("MPS") == "mps:0"
    assert canonical_device("mps:0") == "mps:0"
    assert canonical_device("cuda:01") == "cuda:1"
    assert canonical_device("cuda") == "cuda"
    assert canonical_device("cpu") == "cpu"


def test_read_tail_keeps_the_last_bytes(tmp_path):
    log = tmp_path / "out.log"
    log.write_bytes(b"x" * 5000 + b"END")
    tail = read_tail(log, limit=10)
    assert tail == "xxxxxxxEND"
    assert read_tail(tmp_path / "missing.log") == ""


def test_prune_orders_by_retirement_and_skips_runs_in_flight(tmp_path):
    for index, name in enumerate(["a", "b", "c", "d"]):
        (tmp_path / name).mkdir()
        if name != "a":  # "a" is in flight: no marker
            mark_finished(tmp_path / name)
            marker = tmp_path / name / FINISHED_MARKER
            # "b" started first but retired last.
            stamp = 3000 if name == "b" else 1000 + index
            os.utime(marker, (stamp, stamp))
    (tmp_path / "stray-file").write_text("x")
    prune_run_dirs(tmp_path, keep_last=2)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "a", "b", "d", "stray-file"]
    prune_run_dirs(tmp_path, keep_last=0)  # 0 disables retention
    assert len(list(tmp_path.iterdir())) == 4
    prune_run_dirs(tmp_path / "gone", keep_last=1)  # no directory: no error


def test_a_new_runner_marks_leftover_directories(tmp_path):
    runs = tmp_path / "agent-runs"
    (runs / "old").mkdir(parents=True)
    Runner(runs_dir=runs, offered_devices={"cpu"})
    assert (runs / "old" / FINISHED_MARKER).exists()


# ── refusals: no subprocess is started ───────────────────────────────────


@pytest.mark.parametrize("assign, reason", [
    (_assign("r1", options={"record_outputs": True}), "record_outputs"),
    (_assign("r1", devices=["cuda:0"], device="cuda:0"), "not offered"),
    (_assign("r1", device="tpu:0"), "unknown device"),
    (_assign("r1", device="cuda:0"), "not offered"),
    (_assign("r1", device="cuda"), "not offered"),
    (_assign("r1", device="mps", devices=["cpu"]), "not the first device"),
    (_assign("r1", device="cpu", devices=["mps:0"]), "not the first device"),
    (_assign("r1", device="mps", devices=["mps", "mps:0"]), "twice"),
])
async def test_refused_assignments_finish_failed_at_once(tmp_path, assign,
                                                         reason):
    sink = h.FrameSink()
    runner = _runner(tmp_path, sink, child_entry=h.quick_child)
    await runner.assign(assign)
    finished = sink.finished("r1")
    assert finished is not None and finished.status == "failed"
    assert reason in finished.error and finished.seq == 1
    assert runner.runs == {}
    assert [type(f) for f in sink.frames] == [wp.RunFinished]


async def test_mps_and_mps0_are_one_device(tmp_path):
    sink = h.FrameSink()
    runner = _runner(tmp_path, sink, child_entry=h.quick_child,
                     offered_devices={"cpu", "mps:0"})
    await runner.assign(_assign("m1", device="mps", devices=["mps"]))
    finished = await sink.wait_for(lambda f: isinstance(f, wp.RunFinished))
    assert finished.status == "succeeded"


async def test_an_unexpected_error_fails_only_that_run(tmp_path, monkeypatch):
    sink = h.FrameSink()
    runner = _runner(tmp_path, sink, child_entry=h.quick_child)

    def broken(*_a, **_k):
        raise RuntimeError("no more pipes")

    monkeypatch.setattr(runner._ctx, "Pipe", broken)
    await runner.assign(_assign("x1"))
    finished = sink.finished("x1")
    assert finished.status == "failed" and "no more pipes" in finished.error
    assert runner.runs == {}


async def test_draining_refuses_new_runs(tmp_path):
    sink = h.FrameSink()
    runner = _runner(tmp_path, sink, child_entry=h.quick_child)
    runner.draining = True
    await runner.assign(_assign("r1"))
    assert "draining" in sink.finished("r1").error


# ── subprocess behaviour ─────────────────────────────────────────────────


async def test_crash_is_reported_with_exit_code_and_agent_keeps_serving(
        tmp_path):
    """Acceptance: os._exit(3) mid-run -> execution_error-style finish."""
    sink = h.FrameSink()
    runner = _runner(tmp_path, sink, child_entry=h.crash_child)
    await runner.assign(_assign("crash1"))
    finished = await sink.wait_for(
        lambda f: isinstance(f, wp.RunFinished) and f.run_id == "crash1")
    assert finished.status == "failed"
    assert finished.exit_code == 3
    assert finished.error == "run subprocess exited with code 3"
    assert h.CRASH_MARKER in finished.stderr_tail
    assert finished.last_node == "boom"
    frames = sink.of("crash1")
    assert [f.type for f in frames] == ["run_started", "event", "run_finished"]
    assert [f.seq for f in frames] == [1, 2, 3]
    await asyncio.wait_for(_until(lambda: "crash1" not in runner.runs), 10)

    # The agent keeps serving: the next run starts and finishes.
    runner.child_entry = h.quick_child
    await runner.assign(_assign("after"))
    after = await sink.wait_for(
        lambda f: isinstance(f, wp.RunFinished) and f.run_id == "after")
    assert after.status == "succeeded"


async def test_run_finished_waits_for_a_lingering_child_to_exit(tmp_path):
    """The slot is freed on run_finished, so the child must be gone first."""
    alive_at_finish: list[bool] = []
    sink = h.FrameSink()

    async def send(frame):
        if isinstance(frame, wp.RunFinished):
            alive_at_finish.append(runner.runs[frame.run_id].process.is_alive())
        await sink(frame)

    runner = Runner(runs_dir=tmp_path / "agent-runs", send=send,
                    offered_devices={"cpu"}, child_entry=h.lingering_child,
                    exit_after_finished_s=0.5, kill_after_s=2.0)
    started = time.monotonic()
    await runner.assign(_assign("linger1"))
    finished = await sink.wait_for(
        lambda f: isinstance(f, wp.RunFinished), timeout=30)
    assert alive_at_finish == [False]
    assert time.monotonic() - started >= 0.5
    # The report is the child's own, unchanged by the forced exit.
    assert finished.status == "succeeded" and finished.error is None
    assert finished.exit_code is None and finished.stderr_tail is None
    assert [f.seq for f in sink.of("linger1")] == [1, 2]


async def test_run_finished_follows_a_clean_exit(tmp_path):
    alive_at_finish: list[bool] = []
    sink = h.FrameSink()

    async def send(frame):
        if isinstance(frame, wp.RunFinished):
            alive_at_finish.append(runner.runs[frame.run_id].process.is_alive())
        await sink(frame)

    runner = Runner(runs_dir=tmp_path / "agent-runs", send=send,
                    offered_devices={"cpu"}, child_entry=h.quick_child)
    await runner.assign(_assign("q1"))
    finished = await sink.wait_for(lambda f: isinstance(f, wp.RunFinished))
    assert finished.status == "succeeded"
    assert alive_at_finish == [False]


async def test_a_child_writing_after_reporting_is_drained_not_blocked(
        tmp_path):
    """2 MB after ``finished`` would fill the pipe and block the child's
    exit; the reader keeps reading (and dropping) until it is gone."""
    sink = h.FrameSink()
    runner = _runner(tmp_path, sink, child_entry=h.chatty_child,
                     exit_after_finished_s=30.0)
    started = time.monotonic()
    await runner.assign(_assign("chatty1"))
    handle = runner.runs["chatty1"]
    finished = await sink.wait_for(
        lambda f: isinstance(f, wp.RunFinished), timeout=25)
    assert time.monotonic() - started < 25  # it exited, nobody terminated it
    assert finished.status == "succeeded"
    assert [type(f) for f in sink.of("chatty1")] == [wp.RunStarted,
                                                     wp.RunFinished]
    assert handle.discarded == 32
    assert not handle.terminated


async def test_a_cancel_while_waiting_for_exit_stops_the_process(tmp_path):
    """The child reported; the cancel must not wait out the exit grace,
    and the report it gave is still the one sent."""
    sink = h.FrameSink()
    runner = _runner(tmp_path, sink, child_entry=h.lingering_child,
                     exit_after_finished_s=60.0, kill_after_s=2.0)
    await runner.assign(_assign("lc1"))
    await asyncio.wait_for(_until(lambda: runner.runs["lc1"].finished), 20)
    handle = runner.runs["lc1"]
    assert sink.finished("lc1") is None
    started = time.monotonic()
    assert await runner.cancel("lc1") is True
    assert await runner.cancel("lc1") is True  # repeat: same escalation
    finished = await sink.wait_for(
        lambda f: isinstance(f, wp.RunFinished), timeout=15)
    assert time.monotonic() - started < 10
    assert handle.terminated and not handle.process.is_alive()
    assert finished.status == "succeeded" and finished.error is None
    assert [f.seq for f in sink.of("lc1")] == [1, 2]


async def test_cancel_is_cooperative_within_grace(tmp_path):
    """Acceptance: cancel ends a long node well inside grace_s + 5 s."""
    sink = h.FrameSink()
    runner = _runner(tmp_path, sink, child_entry=h.slow_graph_child)
    await runner.assign(_assign("slow1", graph=h.slow_graph()))
    await sink.wait_for(lambda f: isinstance(f, wp.Event)
                        and f.payload.get("node_id") == "slow"
                        and f.payload.get("status") == "running", timeout=60)
    # A second assignment of a running run is refused without a frame.
    await runner.assign(_assign("slow1", graph=h.slow_graph()))
    assert len([f for f in sink.of("slow1")
                if isinstance(f, wp.RunFinished)]) == 0
    started = time.monotonic()
    assert await runner.cancel("slow1", grace_s=20)
    finished = await sink.wait_for(
        lambda f: isinstance(f, wp.RunFinished), timeout=25)
    assert time.monotonic() - started < 20
    assert finished.status == "cancelled"
    assert not runner.runs.get("slow1") or runner.runs["slow1"].finished
    seqs = [f.seq for f in sink.of("slow1")]
    assert seqs == list(range(1, len(seqs) + 1))


async def test_cancel_escalates_to_terminate_and_reports_interrupted(tmp_path):
    sink = h.FrameSink()
    runner = _runner(tmp_path, sink, child_entry=h.deaf_child,
                     kill_after_s=2.0)
    await runner.assign(_assign("deaf1"))
    await sink.wait_for(lambda f: isinstance(f, wp.Event))
    started = time.monotonic()
    await runner.cancel("deaf1", grace_s=0.5)
    finished = await sink.wait_for(
        lambda f: isinstance(f, wp.RunFinished), timeout=0.5 + 2.0 + 10)
    assert time.monotonic() - started < 0.5 + 2.0 + 5
    assert finished.status == "interrupted"
    assert finished.last_node == "deaf"
    assert "grace period" in finished.error
    assert finished.stderr_tail is not None
    assert await runner.cancel("deaf1") is False  # gone: nothing to cancel


async def test_a_crash_while_stopping_is_still_a_crash(tmp_path):
    sink = h.FrameSink()
    runner = _runner(tmp_path, sink, child_entry=h.crash_on_cancel_child)
    await runner.assign(_assign("cc1"))
    await sink.wait_for(lambda f: isinstance(f, wp.Event))
    await runner.cancel("cc1", grace_s=20)
    finished = await sink.wait_for(lambda f: isinstance(f, wp.RunFinished),
                                   timeout=20)
    assert finished.status == "failed" and finished.exit_code == 7
    assert h.CRASH_MARKER in finished.stderr_tail


async def test_shutdown_stops_every_run(tmp_path):
    sink = h.FrameSink()
    runner = _runner(tmp_path, sink, child_entry=h.deaf_child,
                     kill_after_s=1.0)
    await runner.assign(_assign("d1"))
    await sink.wait_for(lambda f: isinstance(f, wp.Event))
    await asyncio.wait_for(runner.shutdown(grace_s=0.2), 15)
    assert runner.runs == {}
    assert sink.finished("d1").status == "interrupted"
    assert runner.draining


async def test_child_gets_the_assigned_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEFYUI_DB_PATH", str(tmp_path / "never.db"))
    monkeypatch.setenv("CODEFYUI_AGENT_TOKEN", "secret")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
    sink = h.FrameSink()
    runner = _runner(tmp_path, sink, child_entry=h.env_child,
                     offered_devices={"cpu", "cuda:2", "cuda:5"})
    await runner.assign(_assign("env1", device="cuda:5",
                                devices=["cuda:5", "cuda:2"]))
    event = await sink.wait_for(lambda f: isinstance(f, wp.Event))
    assert event.payload["env"] == {
        "CUDA_VISIBLE_DEVICES": "5,2", "HIP_VISIBLE_DEVICES": None,
        "ROCR_VISIBLE_DEVICES": None, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CODEFYUI_DB_PATH": None, "CODEFYUI_AGENT_TOKEN": None}
    assert event.payload["device"] == "cuda:0"
    assert event.payload["torch_loaded"] is False
    await sink.wait_for(lambda f: isinstance(f, wp.RunFinished))
    assert (tmp_path / "agent-runs" / "env1").is_dir()


async def _until(predicate) -> None:
    while not predicate():
        await asyncio.sleep(0.02)


# ── the agent process stays torch-free; a child's events are byte-identical


_DRIVER = textwrap.dedent("""
    import asyncio, json, sys
    from pathlib import Path
    from app.agent.runner import Runner
    from app.core import worker_protocol as wp

    async def main():
        frames, done = [], asyncio.Event()

        async def send(frame):
            frames.append(wp.encode_frame(frame))
            if isinstance(frame, wp.RunFinished):
                done.set()

        runner = Runner(runs_dir=Path(sys.argv[1]), offered_devices={"cpu"},
                        send=send)
        await runner.assign(wp.Assign.model_validate_json(sys.argv[2]))
        await asyncio.wait_for(done.wait(), 120)
        while runner.runs:
            await asyncio.sleep(0.05)
        print(json.dumps({"frames": frames,
                          "torch": "torch" in sys.modules}))

    asyncio.run(main())
""")


async def test_agent_never_imports_torch_and_child_events_match_in_process(
        tmp_path):
    """Acceptance: the agent process never imports torch, and the frames a
    real run subprocess sends are byte-identical to InProcessExecutor's."""
    assign = wp.Assign(run_id="parity1", graph=h.tensor_graph(),
                       options=h.TENSOR_OPTIONS)
    proc = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", _DRIVER, str(tmp_path / "runs"),
         assign.model_dump_json()],
        cwd=BACKEND_DIR, capture_output=True, text=True, timeout=180,
        env={**os.environ, "PYTHONPATH": str(BACKEND_DIR)})
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["torch"] is False
    frames = [wp.decode_agent_frame(raw) for raw in result["frames"]]
    assert frames[0].type == "run_started" and frames[-1].type == "run_finished"
    assert frames[-1].status == "succeeded"
    assert [f.seq for f in frames] == list(range(1, len(frames) + 1))
    worker = [h.event_bytes(f.event_type, f.payload) for f in frames
              if isinstance(f, wp.Event)]
    local = await h.in_process_events(h.tensor_graph(), h.TENSOR_OPTIONS)
    assert worker == local
    assert len(local) == 8  # running + completed for each of four nodes


# ── memory budgets (#788) ─────────────────────────────────────────────────


async def test_run_started_waits_for_the_memory_cap(tmp_path):
    """A budgeted run's ``run_started`` goes out once the subprocess says
    its cap is set, carrying what was applied; the budget reaches the child
    in the child's own numbering (its set starts at cuda:0), and the report
    comes back in the worker's (cuda:1, as placed)."""
    sink = h.FrameSink()
    runner = _runner(tmp_path, sink, child_entry=h.budget_child,
                     offered_devices={"cpu", "cuda:0", "cuda:1"})
    gib6 = 6 * 1024 ** 3
    await runner.assign(_assign("b1", device="cuda:1", devices=["cuda:1"],
                                memory_budgets={"cuda:1": gib6}))
    finished = await sink.wait_for(lambda f: isinstance(f, wp.RunFinished))
    started, end = sink.of("b1")
    assert isinstance(started, wp.RunStarted) and started.seq == 1
    assert started.execution_facts["memory_limit"] == {"cuda:1": {
        "budget_bytes": gib6, "total_bytes": 16 * 1024 ** 3,
        "fraction": 0.375, "seen_as": "cuda:0"}}
    assert started.execution_facts["visible_devices"] == "1"
    assert end is finished and finished.seq == 2
    assert finished.status == "succeeded"


async def test_a_run_whose_cap_failed_never_reports_started(tmp_path):
    sink = h.FrameSink()
    runner = _runner(tmp_path, sink, child_entry=h.uncapped_child,
                     offered_devices={"cuda:0"})
    await runner.assign(_assign("b2", device="cuda:0", devices=["cuda:0"],
                                memory_budgets={"cuda:0": 1024}))
    finished = await sink.wait_for(lambda f: isinstance(f, wp.RunFinished))
    assert sink.of("b2") == [finished]
    assert finished.status == "failed" and finished.seq == 1


async def test_a_budget_for_a_card_outside_the_set_is_refused(tmp_path):
    sink = h.FrameSink()
    runner = _runner(tmp_path, sink, child_entry=h.budget_child,
                     offered_devices={"cuda:0", "cuda:1"})
    await runner.assign(_assign("b3", device="cuda:0", devices=["cuda:0"],
                                memory_budgets={"cuda:1": 1024}))
    finished = await sink.wait_for(lambda f: isinstance(f, wp.RunFinished))
    assert finished.status == "failed"
    assert "not an assigned device" in finished.error
