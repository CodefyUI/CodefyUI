"""``cdui agent start --one-shot --instance [ID]``, agent side (#845).

Several one-shot agents share one credential and one user data dir (the
tasks of a SLURM array): each says ``hello.instance``, keeps its runs in a
directory of its own, sends its ``instance_id`` with its transfer requests,
and the tasks of an array started together with one enrollment token
enroll once. The control plane side is ``tests/test_one_shot_instances.py``.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from typing import Any

import pytest

from app.agent import cli
from app.agent.connection import AgentConnection
from app.agent.one_shot import EXIT_SUCCEEDED, OneShot
from app.agent.runner import Runner
from app.agent.transfer import InputCache, TransferClient
from app.core import worker_protocol as wp

from . import _helpers as h
from .test_connection import _cpu
from .test_one_shot import OneShotControlPlane, _args, _run


# ── the instance id ──────────────────────────────────────────────────────


def test_a_slurm_array_task_is_its_array_job_and_task():
    assert cli.derive_instance({"SLURM_ARRAY_JOB_ID": "4821",
                                "SLURM_ARRAY_TASK_ID": "3",
                                "SLURM_JOB_ID": "4824"}) == "4821-3"
    # Not an array: the job id; no batch system: random.
    assert cli.derive_instance({"SLURM_JOB_ID": " 77 "}) == "77"
    assert cli.derive_instance({"SLURM_ARRAY_TASK_ID": "3",
                                "SLURM_JOB_ID": "77"}) == "77"
    first, second = cli.derive_instance({}), cli.derive_instance({})
    assert len(first) == 6 and first != second
    assert wp.INSTANCE_PATTERN.fullmatch(first)


def test_the_instance_flag_needs_one_shot_and_a_valid_id(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "991")
    monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
    bare = _args(one_shot=True, instance=True)
    assert bare.instance == cli.INSTANCE_AUTO
    cli.check_one_shot(bare)
    assert bare.instance == "991" and bare.foreground
    given = _args(one_shot=True, instance=" Task-7 ")
    cli.check_one_shot(given)
    assert given.instance == "task-7"
    with pytest.raises(ValueError, match="--instance needs --one-shot"):
        cli.check_one_shot(_args(instance="3"))
    for bad in ("a.b", "-3", "x" * 33, "a/b"):
        with pytest.raises(ValueError, match="invalid --instance"):
            cli.check_one_shot(_args(one_shot=True, instance=bad))
    plain = _args(one_shot=True)
    cli.check_one_shot(plain)
    assert plain.instance is None


def test_a_bad_instance_flag_exits_2(capsys):
    assert cli.main(["start", "--instance", "3"]) == 2
    assert "--instance needs --one-shot" in capsys.readouterr().err


def test_an_instance_keeps_its_runs_apart(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    assert cli.instance_runs_dir("4821-3") == \
        tmp_path / cli.INSTANCES_DIR_NAME / "4821-3"
    # Not under the regular agent's runs dir, whose leftovers a starting
    # agent retires.
    assert cli.runs_dir() not in cli.instance_runs_dir("x").parents


# ── enrolling once for the whole array ───────────────────────────────────


def test_only_an_instance_takes_the_enrollment_lock(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    with cli.enrollment_lock(_args(one_shot=True)) as held:
        assert held is None  # nullcontext
    lock = cli.enrollment_lock(_args(one_shot=True, instance="3"))
    with lock:
        # No fcntl on Windows: nothing is locked there, by design.
        assert lock.path.exists() or sys.platform == "win32"
    assert lock._file is None


@pytest.mark.skipif(sys.platform == "win32", reason="no flock on Windows")
def test_tasks_started_together_enroll_once(monkeypatch, tmp_path):
    from app.agent import enroll
    from app.core.resources import DeviceFact

    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "_probe_facts", lambda: [DeviceFact(
        "cpu", "cpu", None, "", None, None, "none")])
    calls: list[str] = []

    def fake_enroll(control: str, token: str, *, name: str,
                    **_: Any) -> tuple[str, str]:
        calls.append(name)
        time.sleep(0.3)  # long enough for the others to queue on the lock
        return name, "the-worker-token"

    monkeypatch.setattr(enroll, "enroll", fake_enroll)
    started: list[tuple[str, Any, Any]] = []

    def fake_run_agent(args: Any, control: Any, token: Any, *_: Any,
                       **__: Any) -> int:
        started.append((args.instance, args.name, token))
        return 0

    monkeypatch.setattr(cli, "run_agent", fake_run_agent)
    codes: list[int] = []

    def task(index: int) -> None:
        codes.append(cli.main([
            "start", "--control", "http://127.0.0.1:9", "--token",
            "abcdef.0123456789abcdef", "--name", "gpu", "--devices", "cpu",
            "--one-shot", "--instance", str(index)]))

    threads = [threading.Thread(target=task, args=(i,)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert codes == [0, 0, 0]
    assert calls == ["gpu"]  # one enrollment for the three tasks
    assert sorted(started) == [(str(i), "gpu", "the-worker-token")
                               for i in range(3)]


# ── against a control plane ──────────────────────────────────────────────


def _instance_agent(tmp_path: Any, control: str, one_shot: OneShot,
                    instance: str = "4821-3",
                    **runner_kw: Any) -> tuple[AgentConnection, asyncio.Event]:
    runner = Runner(runs_dir=tmp_path / "agent-instances" / instance,
                    offered_devices={"cpu"}, cache_dir=tmp_path / "cache",
                    **runner_kw)
    connection = AgentConnection(
        runner=runner, name="gpu", version="9.9.9", control_url=control,
        token="t", probe_devices=_cpu, backoff_min_s=0.05, one_shot=one_shot,
        instance=instance)
    return connection, asyncio.Event()


async def test_an_instance_says_so_and_takes_its_run(tmp_path):
    one = OneShot()
    async with OneShotControlPlane() as cp:
        connection, stop = _instance_agent(tmp_path, cp.url, one,
                                           child_entry=h.quick_child)
        task = asyncio.create_task(_run(connection, stop, one))
        hello = await cp.wait_for(lambda n, f: isinstance(f, wp.Hello))
        assert (hello.name, hello.instance, hello.one_shot) == (
            "gpu", "4821-3", True)
        assert hello.instance_id == connection.instance_id
        assert len(connection.instance_id) == 32
        await cp.send(wp.Assign(run_id="run1", graph={"nodes": []}))
        assert await asyncio.wait_for(task, 60) == EXIT_SUCCEEDED
    # Its transfer requests carry its instance id; its cache is the one
    # given, shared with the other instances.
    transfer = connection.runner.transfer
    assert transfer._headers()[wp.INSTANCE_HEADER] == connection.instance_id
    assert transfer.cache.root == tmp_path / "cache"


async def test_a_control_plane_without_instances_is_refused(tmp_path):
    class OldControlPlane(OneShotControlPlane):
        async def _handle(self, ws: Any, *_: Any) -> None:
            self.sockets.append(ws)
            self._record(0, wp.decode_agent_frame(await ws.recv()))
            await ws.send(wp.encode_frame(wp.Welcome(
                worker_id="w", heartbeat_s=self.heartbeat_s,
                grace_s=self.grace_s, capabilities=[
                    c for c in wp.CAPABILITIES
                    if c != wp.CAPABILITY_INSTANCES])))
            await ws.wait_closed()

    one = OneShot()
    async with OldControlPlane() as cp:
        connection, stop = _instance_agent(tmp_path, cp.url, one)
        assert await _run(connection, stop, one, timeout=15) == 2
        assert len(cp.of_type(wp.Hello)) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="no flock on Windows")
def test_a_second_agent_with_the_same_instance_touches_nothing(monkeypatch,
                                                              tmp_path):
    """Found on Meow: a second agent started with a live instance's id
    retired its spool (runner start) and deleted its runs dir (exit), so
    the live agent's run could not report its end. Now it is refused
    before it builds a runner, and leaves the directory alone."""
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    held = cli.InstanceLock("7")
    held.acquire()
    live = cli.instance_runs_dir("7") / "run1"
    live.mkdir(parents=True)
    (live / "spool.jsonl").write_text("{}\n")
    with pytest.raises(cli.InstanceBusy, match="give each concurrent job"):
        cli.InstanceLock("7").acquire()
    cli.InstanceLock("8").acquire()  # another id is another lock

    def no_runner(**_: Any) -> Any:
        raise AssertionError("a busy instance must not build a runner")

    monkeypatch.setattr("app.agent.runner.Runner", no_runner)
    args = _args(one_shot=True, instance="7")
    cli.check_one_shot(args)
    assert cli.run_agent(args, "http://127.0.0.1:9", "t", []) == 2
    assert (live / "spool.jsonl").exists()
    held.release()
    held.release()  # idempotent
    assert not held.path.exists()
    again = cli.InstanceLock("7")
    again.acquire()
    again.release()


def test_a_regular_transfer_client_sends_no_instance_header(tmp_path):
    plain = TransferClient(base_url="http://x", token="t",
                           cache=InputCache(tmp_path))
    assert plain._headers() == {"X-CodefyUI-Token": "t"}
    instance = TransferClient(base_url="http://x", token=None,
                              cache=InputCache(tmp_path), instance_id="i")
    assert instance._headers() == {wp.INSTANCE_HEADER: "i"}


def test_a_runner_caches_beside_its_runs_unless_told(tmp_path):
    assert Runner(runs_dir=tmp_path / "agent-runs",
                  offered_devices={"cpu"}).cache_dir == \
        tmp_path / "agent-cache"


def test_an_instance_removes_its_runs_dir_when_it_exits(monkeypatch,
                                                        tmp_path):
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    seen: dict[str, Any] = {}

    class FakeConnection:
        def __init__(self, *, runner: Runner, instance: Any,
                     **_: Any) -> None:
            seen["runs_dir"] = runner.runs_dir
            seen["cache_dir"] = runner.cache_dir
            seen["instance"] = instance
            (runner.runs_dir / "run1").mkdir(parents=True)

        async def run(self, stop: Any) -> int:
            return 0

    monkeypatch.setattr("app.agent.connection.AgentConnection",
                        FakeConnection)
    monkeypatch.setattr("app.agent.facts.gather",
                        lambda cancel=None: object())
    monkeypatch.setattr(cli, "_install_stop_handlers", lambda *_: None)
    args = _args(one_shot=True, instance="9", name="gpu")
    cli.check_one_shot(args)
    args.name = "gpu"
    code = cli.run_agent(args, "http://127.0.0.1:9", "t", [])
    assert code == 3  # stopped before a run: the one-shot rule
    assert seen["instance"] == "9"
    assert seen["runs_dir"] == tmp_path / cli.INSTANCES_DIR_NAME / "9"
    assert seen["cache_dir"] == tmp_path / "agent-cache"
    assert not seen["runs_dir"].exists()
