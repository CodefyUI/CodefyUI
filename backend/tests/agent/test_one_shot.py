"""``cdui agent start --one-shot``, agent side (#790).

Against the fake control plane of ``test_connection``: the agent says
``one_shot`` in ``hello``, takes the first run it is assigned and no
other, exits once the control plane acknowledged that run's
``run_finished``, and maps the outcome to its exit code.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

import pytest

from app.agent import cli
from app.agent.connection import AgentConnection
from app.agent.one_shot import (
    EXIT_NO_RUN,
    EXIT_RUN_FAILED,
    EXIT_SUCCEEDED,
    OneShot,
)
from app.agent.runner import Runner
from app.core import worker_protocol as wp

from . import _helpers as h
from .test_connection import FakeControlPlane, _cpu


class OneShotControlPlane(FakeControlPlane):
    """Lists the ``one_shot`` capability (unless told not to) and acks
    every ``run_finished`` (unless told not to)."""

    def __init__(self, *, capable: bool = True, ack: bool = True,
                 **kw: Any) -> None:
        super().__init__(**kw)
        self.capable = capable
        self.ack = ack

    async def _handle(self, ws: Any, *_: Any) -> None:
        import websockets

        number = len(self.sockets)
        self.sockets.append(ws)
        try:
            self._record(number, wp.decode_agent_frame(await ws.recv()))
            await ws.send(wp.encode_frame(wp.Welcome(
                worker_id="w", heartbeat_s=self.heartbeat_s,
                grace_s=self.grace_s, resume_grace_s=self.resume_grace_s,
                capabilities=list(wp.CAPABILITIES) if self.capable
                else ["ack"])))
            async for raw in ws:
                frame = wp.decode_agent_frame(raw)
                self._record(number, frame)
                if self.ack and isinstance(frame, wp.RunFinished):
                    await ws.send(wp.encode_frame(wp.Ack(
                        run_id=frame.run_id, seq=frame.seq)))
        except websockets.exceptions.ConnectionClosed:
            pass


def _agent(tmp_path: Any, control: str, one_shot: OneShot,
           **runner_kw: Any) -> tuple[AgentConnection, asyncio.Event]:
    runner_kw.setdefault("offered_devices", {"cpu"})
    runner = Runner(runs_dir=tmp_path / "agent-runs", **runner_kw)
    connection = AgentConnection(
        runner=runner, name="slurm", version="9.9.9",
        control_url=control, token="t", probe_devices=_cpu,
        backoff_min_s=0.05, one_shot=one_shot)
    return connection, asyncio.Event()


async def _run(connection: AgentConnection, stop: asyncio.Event,
               one_shot: OneShot, timeout: float = 60) -> int:
    code = await asyncio.wait_for(connection.run(stop), timeout)
    return one_shot.exit_code(code)


# ── the state ────────────────────────────────────────────────────────────


def test_it_takes_one_run_and_only_its_run_id():
    one = OneShot()
    assert one.refusal("a") is None
    one.take("a")
    one.take("b")  # the first one stays
    assert one.taken == "a" and one.refusal("a") is None
    assert one.refusal("b") == "this one-shot agent already took run a"
    pinned = OneShot(run_id="r7")
    assert pinned.refusal("r1") == "this one-shot agent takes only run r7"
    assert pinned.refusal("r7") is None


def test_its_end_and_its_delivery_set_the_exit_code():
    one = OneShot()
    assert one.exit_code(0) == EXIT_NO_RUN
    assert one.exit_code(2) == 2  # a refusal is a refusal
    one.take("a")
    one.sent(wp.RunFinished(run_id="other", seq=1, status="succeeded"))
    assert not one.finished.is_set()
    one.acked("a")  # an ack before the end means nothing
    assert not one.delivered.is_set()
    one.sent(wp.RunFinished(run_id="a", seq=4, status="succeeded"))
    one.sent(wp.RunFinished(run_id="a", seq=5, status="failed"))
    assert one.status == "succeeded"
    assert one.exit_code(0) == EXIT_RUN_FAILED  # not recorded there yet
    one.acked("a")
    assert one.exit_code(0) == EXIT_SUCCEEDED
    for status in ("failed", "cancelled", "interrupted"):
        other = OneShot(taken="b")
        other.sent(wp.RunFinished(run_id="b", seq=1, status=status))
        other.acked("b")
        assert other.exit_code(0) == EXIT_RUN_FAILED
    started = OneShot(taken="c")  # stopped while it ran, no end reported
    assert started.exit_code(0) == EXIT_RUN_FAILED
    # Refused on a reconnect after it took its run: the run is what failed.
    assert started.exit_code(2) == EXIT_RUN_FAILED
    assert OneShot().exit_code(2) == 2


async def test_an_outcome_that_is_never_recorded_stops_after_the_window(
        monkeypatch, caplog):
    monkeypatch.setattr("app.agent.one_shot.MIN_DELIVER_WINDOW_S", 0.05)
    one = OneShot(taken="a")
    stop = asyncio.Event()
    watch = asyncio.create_task(one.wait_done(stop, lambda: 0.0))
    one.sent(wp.RunFinished(run_id="a", seq=1, status="succeeded"))
    await asyncio.wait_for(watch, 5)
    assert stop.is_set() and not one.delivered.is_set()
    assert "did not record it within" in caplog.text
    assert one.exit_code(0) == EXIT_RUN_FAILED


# ── the command line ─────────────────────────────────────────────────────


def _args(**kw: Any) -> argparse.Namespace:
    return cli.build_parser().parse_args(
        ["start", *[part for key, value in kw.items()
                    for part in ([f"--{key.replace('_', '-')}"]
                                 + ([] if value is True else [value]))]])


def test_one_shot_runs_in_the_foreground_and_run_id_needs_it():
    args = _args(one_shot=True)
    cli.check_one_shot(args)
    assert args.foreground and args.run_id is None
    args = _args(one_shot=True, run_id=" abc123 ")
    cli.check_one_shot(args)
    assert args.run_id == "abc123"
    with pytest.raises(ValueError, match="--run-id needs --one-shot"):
        cli.check_one_shot(_args(run_id="abc"))
    with pytest.raises(ValueError, match="is not a run id"):
        cli.check_one_shot(_args(one_shot=True, run_id="../etc"))
    plain = _args()
    cli.check_one_shot(plain)
    assert not plain.foreground


def test_a_bad_one_shot_flag_exits_2(capsys):
    assert cli.main(["start", "--run-id", "abc"]) == 2
    assert "--run-id needs --one-shot" in capsys.readouterr().err


# ── against a control plane ──────────────────────────────────────────────


async def test_it_says_one_shot_takes_its_run_and_exits_0(tmp_path):
    one = OneShot()
    async with OneShotControlPlane() as cp:
        connection, stop = _agent(tmp_path, cp.url, one,
                                  child_entry=h.quick_child)
        task = asyncio.create_task(_run(connection, stop, one))
        hello = await cp.wait_for(lambda n, f: isinstance(f, wp.Hello))
        assert (hello.one_shot, hello.one_shot_run_id) == (True, None)
        assert hello.instance_id == connection.instance_id
        await cp.send(wp.Assign(run_id="run1", graph={"nodes": []}))
        await cp.send(wp.Assign(run_id="run2", graph={"nodes": []}))
        refused = await cp.wait_for(lambda n, f: isinstance(
            f, wp.RunFinished) and f.run_id == "run2")
        assert refused.status == "failed"
        assert refused.error == "this one-shot agent already took run run1"
        assert await asyncio.wait_for(task, 60) == EXIT_SUCCEEDED
    finished = [f for f in cp.of_type(wp.RunFinished) if f.run_id == "run1"]
    assert [f.status for f in finished] == ["succeeded"]


async def test_a_failed_run_exits_1(tmp_path):
    one = OneShot(run_id="run9")
    async with OneShotControlPlane() as cp:
        connection, stop = _agent(tmp_path, cp.url, one,
                                  child_entry=h.crash_child)
        task = asyncio.create_task(_run(connection, stop, one))
        hello = await cp.wait_for(lambda n, f: isinstance(f, wp.Hello))
        assert hello.one_shot_run_id == "run9"
        await cp.send(wp.Assign(run_id="other", graph={"nodes": []}))
        refused = await cp.wait_for(lambda n, f: isinstance(
            f, wp.RunFinished) and f.run_id == "other")
        assert refused.error == "this one-shot agent takes only run run9"
        await cp.send(wp.Assign(run_id="run9", graph={"nodes": []}))
        assert await asyncio.wait_for(task, 60) == EXIT_RUN_FAILED
    assert [f.status for f in cp.of_type(wp.RunFinished)
            if f.run_id == "run9"] == ["failed"]


async def test_a_cancelled_run_exits_1(tmp_path):
    one = OneShot()
    async with OneShotControlPlane() as cp:
        connection, stop = _agent(tmp_path, cp.url, one,
                                  child_entry=h.deaf_child, kill_after_s=1.0)
        task = asyncio.create_task(_run(connection, stop, one))
        await cp.wait_for(lambda n, f: isinstance(f, wp.Hello))
        await cp.send(wp.Assign(run_id="run1", graph={"nodes": []}))
        await cp.wait_for(lambda n, f: isinstance(f, wp.Event))
        await cp.send(wp.Cancel(run_id="run1", grace_s=0.2))
        assert await asyncio.wait_for(task, 60) == EXIT_RUN_FAILED
    assert one.status == "interrupted"


async def test_a_dropped_connection_mid_run_is_ridden_out(tmp_path):
    one = OneShot()
    async with OneShotControlPlane() as cp:
        connection, stop = _agent(tmp_path, cp.url, one,
                                  child_entry=h.ticking_child)
        task = asyncio.create_task(_run(connection, stop, one))
        await cp.wait_for(lambda n, f: isinstance(f, wp.Hello))
        await cp.send(wp.Assign(run_id="run1", graph={"nodes": []}))
        await cp.wait_for(lambda n, f: isinstance(f, wp.Event))
        await cp.sockets[0].close()  # the network drops
        alive = await cp.wait_for(
            lambda n, f: n == 1 and isinstance(f, wp.RunAlive), timeout=15)
        assert [e.run_id for e in alive.runs] == ["run1"]
        hellos = cp.of_type(wp.Hello)
        # The same agent: the control plane lets it back in, and hears
        # which run it took (a restarted control plane knows nothing else).
        assert hellos[0].instance_id == hellos[1].instance_id
        assert (hellos[0].one_shot_taken, hellos[1].one_shot_taken) == (
            None, "run1")
        assert await asyncio.wait_for(task, 60) == EXIT_SUCCEEDED
    assert [f.status for f in cp.of_type(wp.RunFinished)] == ["succeeded"]


async def test_a_stop_before_any_run_exits_3(tmp_path):
    one = OneShot()
    async with OneShotControlPlane() as cp:
        connection, stop = _agent(tmp_path, cp.url, one)
        task = asyncio.create_task(_run(connection, stop, one))
        await cp.wait_for(lambda n, f: isinstance(f, wp.Heartbeat))
        stop.set()
        assert await asyncio.wait_for(task, 30) == EXIT_NO_RUN


async def test_a_control_plane_without_one_shot_is_refused(tmp_path):
    one = OneShot()
    async with OneShotControlPlane(capable=False) as cp:
        connection, stop = _agent(tmp_path, cp.url, one)
        assert await _run(connection, stop, one, timeout=15) == 2
        assert len(cp.of_type(wp.Hello)) == 1


async def test_a_regular_agent_is_not_one_shot(tmp_path):
    runner = Runner(runs_dir=tmp_path / "agent-runs", offered_devices={"cpu"},
                    child_entry=h.quick_child)
    async with FakeControlPlane() as cp:
        connection = AgentConnection(
            runner=runner, name="box", version="9.9.9", control_url=cp.url,
            token="t", probe_devices=_cpu, backoff_min_s=0.05)
        stop = asyncio.Event()
        task = asyncio.create_task(connection.run(stop))
        hello = await cp.wait_for(lambda n, f: isinstance(f, wp.Hello))
        assert not hello.one_shot and hello.instance_id
        for run_id in ("a", "b"):
            await cp.send(wp.Assign(run_id=run_id, graph={"nodes": []}))
        await cp.wait_for(lambda n, f: len(cp.of_type(wp.RunFinished)) == 2)
        stop.set()
        assert await asyncio.wait_for(task, 30) == 0
    assert {f.status for f in cp.of_type(wp.RunFinished)} == {"succeeded"}


# ── a batch system's own card (SLURM's CUDA_VISIBLE_DEVICES) ─────────────


def test_a_run_gets_the_card_the_batch_system_gave_the_agent(monkeypatch):
    from app.agent.runner import child_environment, inherited_card

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert inherited_card(2) == "2"
    assert child_environment("cuda:1", ["cuda:1"])["set"][
        "CUDA_VISIBLE_DEVICES"] == "1"
    # SLURM allocated physical card 3: the agent offers it as cuda:0, and
    # the run must see card 3, not card 0.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert child_environment("cuda:0", ["cuda:0"])["set"][
        "CUDA_VISIBLE_DEVICES"] == "3"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc, 5,,7")
    assert [inherited_card(i) for i in range(4)] == ["GPU-abc", "5", "2", "3"]
    assert child_environment("cuda:1", ["cuda:1", "cuda:0"])["set"][
        "CUDA_VISIBLE_DEVICES"] == "5,GPU-abc"
    # A CPU run still sees no card.
    assert child_environment("cpu", [])["set"]["CUDA_VISIBLE_DEVICES"] == ""


# ── setup failures and ROCm (review of #835) ─────────────────────────────


def test_a_one_shot_agent_that_cannot_start_exits_4(monkeypatch, capsys):
    """Exit 1 means "the run failed" for a one-shot agent, so a refused
    --mount (and a failed enrollment, through the same helper) is 4; a
    regular agent keeps 1."""
    from app.agent.one_shot import EXIT_SETUP

    monkeypatch.setattr(cli, "running_agent_pid", lambda: 4242)
    assert cli.main(["start", "--one-shot", "--mount", "data=/x"]) \
        == EXIT_SETUP == 4
    assert "--mount cannot change" in capsys.readouterr().err
    assert cli.main(["start", "--foreground", "--mount", "data=/x"]) == 1
    assert cli.setup_failed(_args(one_shot=True)) == 4
    assert cli.setup_failed(_args()) == 1


def test_a_failed_enrollment_exits_4_under_one_shot(monkeypatch, tmp_path):
    from app.agent import enroll

    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    from app.core.resources import DeviceFact

    monkeypatch.setattr(cli, "_probe_facts", lambda: [DeviceFact(
        "cpu", "cpu", None, "", None, None, "none")])

    def refuse(*_: Any, **__: Any) -> Any:
        raise enroll.EnrollError("this enrollment token was already used")

    monkeypatch.setattr(enroll, "enroll", refuse)
    argv = ["start", "--control", "http://127.0.0.1:9", "--token",
            "abcdef.0123456789abcdef", "--name", "slurm", "--devices", "cpu"]
    assert cli.main([*argv, "--one-shot"]) == 4
    assert cli.main([*argv, "--foreground"]) == 1


def test_an_agent_offers_no_card_it_cannot_see_through_nvml(monkeypatch):
    """Why ROCm visibility needs no translation in ``child_environment``:
    the agent lists GPUs through NVML only (it never imports torch), so on
    an AMD machine it offers no GPU and no ROCm run is ever assigned to it,
    whatever ``ROCR_VISIBLE_DEVICES`` / ``HIP_VISIBLE_DEVICES`` say."""
    from app.core import resources

    monkeypatch.setattr(resources, "_nvml_cards", lambda: [])
    monkeypatch.setattr(cli, "apple_silicon", lambda: False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    assert [fact.kind for fact in cli._probe_facts()] == ["cpu"]
