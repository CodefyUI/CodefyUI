"""SSH workers (#784): ``core.ssh_launch`` and its routes.

A fake ``ssh`` (``tests/fake_ssh.py``, behind an ``ssh`` shim on PATH)
records its argv and the token it reads from stdin, and keeps running (or
fails, or reports a failed or forbidden forward, or another worker's
stored credential). It answers the prepare one-shot with a made-up remote
directory. The remote commands themselves are run with a real ``sh``
against a fake ``cdui``.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core import ssh_launch
from app.core.auth import TOKEN_HEADER, session_token
from app.core.db import Database
from app.core.ssh_launch import SshLauncher
from app.core.worker_store import SshWorker, WorkerStore
from app.core.workers import WorkerRegistry
from app.main import app

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the fake ssh is a POSIX shell shim")

FAKE_SSH = Path(__file__).resolve().parent / "fake_ssh.py"
LOCAL_PORT = 8123
RANGE = (18000, 18009)
#: This control plane's instance id in these tests.
CP = "abcd1234"
#: The remote directory the fake ssh's prepare answers with, for slot 18000.
FAKE_DIR = f"/home/fake/.codefyui-ssh/{CP}-18000"
#: The address the agent stores its credential under, for slot 18000.
URL = f"http://cp-{CP}.invalid:18000"


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def fake_ssh(tmp_path, monkeypatch):
    """``ssh`` on PATH is the fake; returns the path of its argv log."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "ssh"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_SSH}" "$@"\n')
    shim.chmod(0o755)
    log = tmp_path / "ssh.log"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SSH_LOG", str(log))
    monkeypatch.setenv("FAKE_SSH_MODE", "hold")
    monkeypatch.delenv("FAKE_SSH_BUSY_PORTS", raising=False)
    monkeypatch.setenv("FAKE_SSH_HANG_S", "0")
    monkeypatch.setattr(ssh_launch, "BACKOFF_MIN_S", 0.05)
    monkeypatch.setattr(ssh_launch, "POLL_S", 0.05)
    monkeypatch.setattr(ssh_launch, "STOP_GRACE_S", 3.0)
    return log


class FakeRegistry:
    """What the launcher asks of ``WorkerRegistry``: ``get`` and ``drain``."""

    def __init__(self) -> None:
        self.online: set[str] = set()
        self.drained: dict[str, bool] = {}
        self.stopped_names: list[str] = []

    def get(self, name: str) -> Any:
        return SimpleNamespace(online=True) if name in self.online else None

    def drain(self, name: str, draining: bool = True) -> None:
        self.drained[name] = draining

    def stopped(self, name: str, reason: str) -> None:
        self.stopped_names.append(name)


@pytest.fixture
async def launcher(tmp_path, fake_ssh):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    store = WorkerStore(database)
    registry = FakeRegistry()
    launcher = SshLauncher(store, registry, port_range=RANGE, instance_id=CP,
                           local_port=LOCAL_PORT)
    try:
        yield launcher
    finally:
        await launcher.shutdown()
        database.close()


def _entry(name: str = "gpu-box", **fields: Any) -> SshWorker:
    return SshWorker(id=name, target=fields.pop("target", "alice@gpu-box"),
                     **fields)


def _all_calls(log: Path) -> list[list[str]]:
    """Every ssh the launcher ran: sessions and one-shots."""
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def _calls(log: Path) -> list[list[str]]:
    """The sessions (the ssh runs with a forward)."""
    return [argv for argv in _all_calls(log) if "-R" in argv]


def _oneshots(log: Path) -> list[list[str]]:
    return [argv for argv in _all_calls(log) if "-R" not in argv]


def _stdin(log: Path) -> list[str]:
    """Every line the fake ssh read on stdin, in order."""
    path = Path(str(log) + ".stdin")
    return path.read_text().splitlines() if path.exists() else []


def _tokens(log: Path) -> list[str]:
    """The tokens: the stdin lines that are not ``stop``."""
    return [line for line in _stdin(log) if line != "stop"]


async def _until(predicate: Any, what: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(what)
        await asyncio.sleep(0.02)


# ── the command line ──────────────────────────────────────────────────────


async def test_ssh_gets_the_spec_options_and_the_token_only_on_stdin(
        launcher, fake_ssh, tmp_path):
    key = tmp_path / "id_ed25519"
    key.write_text("not a real key")
    await launcher.add(_entry(identity=str(key), port=2222))
    await _until(lambda: _tokens(fake_ssh), "ssh never read a token")
    [argv] = _calls(fake_ssh)
    [token] = _tokens(fake_ssh)

    assert argv[0] == "-T"  # (the fake records its arguments only)
    options = [argv[i + 1] for i, arg in enumerate(argv) if arg == "-o"]
    assert "BatchMode=yes" in options
    assert "ExitOnForwardFailure=yes" in options
    assert "ServerAliveInterval=15" in options
    assert "StrictHostKeyChecking=accept-new" in options
    assert "-T" in argv
    assert argv[argv.index("-p") + 1] == "2222"
    assert argv[argv.index("-i") + 1] == str(key)
    # The remote end is a Unix socket in the 0700 state directory the
    # prepare one-shot made (and reported) first; never a TCP port.
    [prepare] = _oneshots(fake_ssh)
    assert "chmod 700" in prepare[-1] and 'rm -f "$d/sock"' in prepare[-1]
    assert _all_calls(fake_ssh).index(prepare) == 0
    assert argv[argv.index("-R") + 1] == (
        f"{FAKE_DIR}/sock:127.0.0.1:{LOCAL_PORT}")
    # `--`, then the target, then one command string.
    dashes = argv.index("--")
    assert argv[dashes + 1:dashes + 2] == ["alice@gpu-box"]
    command = argv[dashes + 2]
    assert len(argv) == dashes + 3
    assert command.startswith("sh -c '") and command.endswith("'")
    assert ('"$HOME"/CodefyUI/cdui agent start --uds "$d/sock" --foreground '
            f'--control {URL} --name gpu-box') in command
    # The token is an enrollment token, on stdin, never on a command line.
    assert ssh_launch.ENROLL_TTL_S <= 600
    assert all(token not in arg for arg in argv)
    [pending] = await launcher.store.list_enrollments()
    assert pending["label"] == "ssh alice@gpu-box"
    assert pending["worker"] is None and pending["state"] == "pending"
    assert launcher.describe("gpu-box")["state"] == "connecting"
    assert (await launcher.store.list_ssh_workers())[0].remote_port == 18000


async def test_strict_host_key_checking_can_require_a_known_host(
        launcher, fake_ssh, monkeypatch):
    monkeypatch.setattr(settings, "SSH_STRICT_HOST_KEY_CHECKING", "yes")
    await launcher.add(_entry())
    await _until(lambda: _calls(fake_ssh), "ssh never ran")
    assert "StrictHostKeyChecking=yes" in _calls(fake_ssh)[0]
    # Port 22 is left to ssh, so a Host alias's own Port applies.
    assert "-p" not in _calls(fake_ssh)[0]


async def test_up_once_the_worker_is_online_then_the_token_is_dropped(
        launcher, fake_ssh):
    await launcher.add(_entry())
    await _until(lambda: _tokens(fake_ssh), "ssh never read a token")
    assert len(await launcher.store.list_enrollments()) == 1
    launcher.registry.online.add("gpu-box")
    await _until(lambda: launcher.describe("gpu-box")["state"] == "up",
                 "the tunnel never came up")
    assert launcher.describe("gpu-box")["up_since"] is not None
    for _ in range(100):
        if not await launcher.store.list_enrollments():
            break
        await asyncio.sleep(0.02)
    assert await launcher.store.list_enrollments() == []
    launcher.registry.online.discard("gpu-box")
    await _until(lambda: launcher.describe("gpu-box")["state"] == "connecting",
                 "a worker that went away still reads up")


async def test_an_enrolled_worker_gets_a_token_bound_to_it(launcher,
                                                           fake_ssh):
    store = launcher.store
    created = await store.create_enrollment()
    await store.enroll(created["token"], name="gpu-box", platform={},
                       mounts={}, devices=[])
    await store.delete_enrollment(created["id"])
    await launcher.add(_entry())
    await _until(lambda: _tokens(fake_ssh), "ssh never read a token")
    [pending] = await store.list_enrollments()
    assert pending["worker"] == "gpu-box"


async def test_a_stop_ends_the_workers_waiting_runs(launcher, fake_ssh):
    """Disable (and Remove, and a shutdown) is a stop on purpose: once ssh
    is gone, whatever run is still waiting for the worker ends now, not
    after the resume grace."""
    await launcher.add(_entry())
    await _until(lambda: _tokens(fake_ssh), "ssh never ran")
    await launcher.set_enabled("gpu-box", False)
    assert launcher.registry.stopped_names == ["gpu-box"]


def test_the_stop_grace_outlasts_the_agents_own_shutdown(tmp_path):
    """ssh must still carry the tunnel while the agent stops its runs
    (their grace, then terminate, then kill) and uploads their last files
    (an interrupt checkpoint), or ``run_finished`` never arrives."""
    from app.agent.runner import KILL_AFTER_S, Runner

    runner = Runner(runs_dir=tmp_path / "runs", offered_devices={"cpu"})
    assert ssh_launch.STOP_GRACE_S >= (runner.shutdown_grace_s
                                       + KILL_AFTER_S + 30)


# ── failures and retries ──────────────────────────────────────────────────


async def test_a_failing_ssh_is_retried_with_its_error_surfaced(
        launcher, fake_ssh, monkeypatch):
    monkeypatch.setenv("FAKE_SSH_MODE", "fail")
    await launcher.add(_entry())
    await _until(lambda: len(_all_calls(fake_ssh)) >= 3
                 and launcher.describe("gpu-box")["state"] == "failed",
                 "ssh was not retried")
    view = launcher.describe("gpu-box")
    assert view["gave_up"] is False
    assert "Permission denied (publickey)" in view["last_error"]
    assert view["exit_code"] == 255
    assert view["attempts"] >= 3
    assert any("Permission denied" in line for line in view["tail"])
    # Each attempt minted a new token and deleted the unspent one before.
    assert len(await launcher.store.list_enrollments()) <= 1


def test_the_backoff_doubles_up_to_a_minute():
    assert ssh_launch.next_backoff(2.0) == 4.0
    assert ssh_launch.next_backoff(40.0) == ssh_launch.BACKOFF_MAX_S
    assert ssh_launch.next_backoff(0.0) == ssh_launch.BACKOFF_MIN_S


async def test_a_oneshot_that_hangs_is_killed_and_reaped(launcher, fake_ssh,
                                                        monkeypatch, tmp_path):
    """A one-shot ssh (prepare, stop) that does not finish within
    ``ONESHOT_TIMEOUT_S`` is killed and waited for, not left running."""
    monkeypatch.setattr(ssh_launch, "ONESHOT_TIMEOUT_S", 0.5)
    monkeypatch.setenv("FAKE_SSH_HANG_S", "30")
    await launcher.add(_entry())
    pid_file = Path(str(fake_ssh) + ".pid")
    await _until(lambda: launcher.describe("gpu-box")["state"] == "failed"
                 and pid_file.exists(), "the hanging one-shot never failed")
    pid = int(pid_file.read_text().split()[0])
    await _until(lambda: not _alive(pid), "the hanging ssh was left running")
    view = launcher.describe("gpu-box")
    assert "did not finish within" in view["last_error"]
    # No session was started without its directory.
    assert _calls(fake_ssh) == []


async def test_a_cancelled_oneshot_leaves_no_process(launcher, fake_ssh,
                                                     monkeypatch):
    """Cancelled while a one-shot ssh hangs (a Disable, a shutdown): the
    ssh is killed and reaped, not left running."""
    monkeypatch.setenv("FAKE_SSH_HANG_S", "30")
    launch = ssh_launch.SshLaunch(_entry(), launcher)
    task = asyncio.ensure_future(launch._oneshot("true", "test"))
    pid_file = Path(str(fake_ssh) + ".pid")
    await _until(pid_file.exists, "the one-shot never started")
    pid = int(pid_file.read_text().split()[0])
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not _alive(pid)


async def test_a_home_with_a_space_is_refused_with_its_reason(
        launcher, fake_ssh, monkeypatch):
    monkeypatch.setenv("FAKE_SSH_PREPARE_DIR",
                       f"/home/al ice/.codefyui-ssh/{CP}-18000")
    await launcher.add(_entry())
    await _until(lambda: launcher.describe("gpu-box")["gave_up"],
                 "a home with a space was not refused")
    error = launcher.describe("gpu-box")["last_error"]
    assert "/home/al ice/" in error and "a space" in error
    assert "AllowStreamLocalForwarding" not in error
    assert _calls(fake_ssh) == []


async def test_a_shutdown_stops_every_session_within_its_own_grace(
        launcher, fake_ssh, monkeypatch):
    graces: list[float] = []
    real = ssh_launch._end_process

    async def spy(proc, grace_s=ssh_launch.STOP_GRACE_S):
        graces.append(grace_s)
        await real(proc, grace_s)

    monkeypatch.setattr(ssh_launch, "_end_process", spy)
    await launcher.add(_entry())
    await launcher.add(_entry("two-box", target="b@two"))
    await _until(lambda: len(_tokens(fake_ssh)) == 2, "ssh never ran")
    await launcher.shutdown()
    # One per launch's stop (the loops also end their own sessions,
    # with whatever grace; stop terminates at its own).
    assert graces.count(ssh_launch.SHUTDOWN_GRACE_S) == 2
    assert sorted(launcher.registry.stopped_names) == ["gpu-box", "two-box"]


async def test_the_first_launch_on_a_new_port_forgets_and_later_ones_reuse(
        launcher, fake_ssh):
    await launcher.add(_entry())
    await _until(lambda: _tokens(fake_ssh), "ssh never ran")
    # A port this entry never used may hold another worker's credential.
    assert (f'"$HOME"/CodefyUI/cdui agent forget {URL}'
            in _calls(fake_ssh)[0][-1])
    await launcher.set_enabled("gpu-box", False)
    await launcher.set_enabled("gpu-box", True)
    await _until(lambda: len(_tokens(fake_ssh)) == 2, "never relaunched")
    # Its own port again: the credential stored there is its own.
    assert "agent forget" not in _calls(fake_ssh)[1][-1]


async def test_the_agents_refused_token_restarts_it_with_forget(
        launcher, fake_ssh, monkeypatch):
    """The agent reports that this server refused its stored token (its
    own log line, which only that agent writes): the session restarts
    once with ``cdui agent forget``. A refusal ``ws_worker`` sees on the
    listener is not reported here at all (another local client could have
    caused it)."""
    await launcher.store.add_ssh_worker(_entry(remote_port=18003))
    monkeypatch.setenv("FAKE_SSH_MODE", "refused")
    await launcher.start_all()
    await _until(lambda: len(_calls(fake_ssh)) >= 2,
                 "the refused agent was not restarted")
    assert "agent forget" not in _calls(fake_ssh)[0][-1]
    assert (f'"$HOME"/CodefyUI/cdui agent forget http://cp-{CP}.invalid:18003'
            in _calls(fake_ssh)[1][-1])
    assert not hasattr(launcher, "note_refused")


async def test_another_workers_stored_credential_restarts_with_forget(
        launcher, fake_ssh, monkeypatch):
    monkeypatch.setenv("FAKE_SSH_MODE", "mismatch")
    await launcher.store.add_ssh_worker(_entry(remote_port=18002))
    await launcher.start_all()
    await _until(lambda: len(_calls(fake_ssh)) >= 2, "not restarted")
    first, second = _calls(fake_ssh)[:2]
    assert "agent forget" not in first[-1]
    assert "agent forget http://cp-abcd1234.invalid:18002" in second[-1]
    assert "is for worker" in (launcher.describe("gpu-box")["last_error"]
                               or "")


async def test_forbidden_forwarding_is_a_final_failure_without_port_walking(
        launcher, fake_ssh, monkeypatch):
    monkeypatch.setenv("FAKE_SSH_MODE", "forbidden")
    await launcher.add(_entry())
    await _until(lambda: launcher.describe("gpu-box")["gave_up"],
                 "forbidden forwarding was retried")
    view = launcher.describe("gpu-box")
    assert view["state"] == "failed" and view["retry_at"] is None
    assert "does not allow forwarding to a Unix" in view["last_error"]
    assert len(_calls(fake_ssh)) == 1
    assert view["remote_port"] == 18000
    # Enable tries again.
    monkeypatch.setenv("FAKE_SSH_MODE", "hold")
    await launcher.set_enabled("gpu-box", True)
    await _until(lambda: _tokens(fake_ssh), "Enable did not retry")
    assert launcher.describe("gpu-box")["gave_up"] is False


async def test_failed_forwards_give_up_after_a_few(launcher, fake_ssh,
                                                  monkeypatch):
    monkeypatch.setenv("FAKE_SSH_BUSY_PORTS", "all")
    await launcher.add(_entry())
    await _until(lambda: launcher.describe("gpu-box")["gave_up"],
                 "the retries never stopped")
    calls = _calls(fake_ssh)
    assert len(calls) == ssh_launch.FORWARD_GIVE_UP
    # The same slot every time: a socket path is not a port someone took.
    assert {c[c.index("-R") + 1] for c in calls} == {
        f"{FAKE_DIR}/sock:127.0.0.1:{LOCAL_PORT}"}
    assert "times in a row" in launcher.describe("gpu-box")["last_error"]
    assert launcher.describe("gpu-box")["remote_port"] == 18000


async def test_the_remote_agent_pid_is_reported(launcher, fake_ssh):
    await launcher.add(_entry())
    await _until(lambda: launcher.describe("gpu-box")["remote_pid"] == 4242,
                 "the remote pid was not read")


async def test_disable_sends_stop_through_the_session(launcher, fake_ssh):
    await launcher.add(_entry())
    await _until(lambda: launcher.describe("gpu-box")["remote_pid"] == 4242,
                 "the session never attached")
    await launcher.set_enabled("gpu-box", False)
    assert _stdin(fake_ssh)[1:] == ["stop"]
    # A live session carried the stop: no one-shot stop.
    assert not any("stopped agent pid" in argv[-1]
                   for argv in _oneshots(fake_ssh))


async def test_disable_with_the_session_gone_stops_the_agent_over_ssh(
        launcher, fake_ssh, monkeypatch):
    await launcher.add(_entry())
    await _until(lambda: launcher.describe("gpu-box")["remote_pid"] == 4242,
                 "the session never attached")
    monkeypatch.setenv("FAKE_SSH_MODE", "fail")  # every relaunch fails now
    os.kill(launcher.describe("gpu-box")["pid"], 9)  # the session drops
    await _until(lambda: launcher.describe("gpu-box")["state"] == "failed",
                 "the drop was not seen")
    await launcher.set_enabled("gpu-box", False)
    oneshots = [argv for argv in _oneshots(fake_ssh)
                if "stopped agent pid" in argv[-1]]
    assert len(oneshots) == 1
    assert "ExitOnForwardFailure=yes" not in oneshots[0]
    assert "kill -TERM $a" in oneshots[0][-1]
    assert f".codefyui-ssh/{CP}-18000" in oneshots[0][-1]


async def test_a_dropped_session_comes_back_on_the_same_socket(
        launcher, fake_ssh):
    """Each session is prepared first (the stale socket removed), and the
    relaunch forwards the same path, where the surviving agent dials."""
    await launcher.add(_entry())
    await _until(lambda: launcher.describe("gpu-box")["remote_pid"] == 4242,
                 "the session never attached")
    os.kill(launcher.describe("gpu-box")["pid"], 9)  # the session drops
    await _until(lambda: len(_tokens(fake_ssh)) >= 2, "never relaunched")
    forwards = {argv[argv.index("-R") + 1] for argv in _calls(fake_ssh)}
    assert forwards == {f"{FAKE_DIR}/sock:127.0.0.1:{LOCAL_PORT}"}
    prepares = [argv for argv in _oneshots(fake_ssh)
                if "codefyui-ssh: dir" in argv[-1]]
    assert len(prepares) >= 2
    assert launcher.describe("gpu-box")["remote_port"] == 18000


async def test_enable_waiting_behind_a_disable_ends_running(launcher,
                                                            fake_ssh):
    await launcher.add(_entry())
    await _until(lambda: _tokens(fake_ssh), "ssh never ran")
    disable = asyncio.ensure_future(launcher.set_enabled("gpu-box", False))
    enable = asyncio.ensure_future(launcher.set_enabled("gpu-box", True))
    await asyncio.gather(disable, enable)
    assert launcher.launches["gpu-box"].running
    assert launcher.entries["gpu-box"].enabled
    assert (await launcher.store.list_ssh_workers())[0].enabled
    await _until(lambda: len(_tokens(fake_ssh)) == 2, "not relaunched")


async def test_delete_racing_enable_does_not_resurrect(launcher, fake_ssh):
    await launcher.add(_entry())
    await _until(lambda: _tokens(fake_ssh), "ssh never ran")
    remove = asyncio.ensure_future(launcher.remove("gpu-box"))
    enable = asyncio.ensure_future(launcher.set_enabled("gpu-box", True))
    removed, enabled = await asyncio.gather(remove, enable)
    assert removed is True and enabled is None
    assert "gpu-box" not in launcher.launches
    assert await launcher.store.list_ssh_workers() == []
    await asyncio.sleep(0.3)
    assert len(_calls(fake_ssh)) == 1


async def test_a_disabled_rows_worker_is_drained_when_it_connects(launcher):
    await launcher.store.add_ssh_worker(_entry(enabled=False))
    await launcher.store.add_ssh_worker(_entry("on-box", target="b@on"))
    await launcher.start_all()
    launcher.registry.drained.clear()
    launcher.worker_connected("gpu-box")
    launcher.worker_connected("on-box")
    launcher.worker_connected("unknown")
    assert launcher.registry.drained == {"gpu-box": True}


async def test_no_ssh_on_path_fails_with_a_clear_message(launcher,
                                                         monkeypatch):
    monkeypatch.setattr(ssh_launch, "ssh_executable", lambda: None)
    await launcher.add(_entry())
    await _until(lambda: launcher.describe("gpu-box")["state"] == "failed",
                 "a missing ssh was not reported")
    assert "ssh was not found" in launcher.describe("gpu-box")["last_error"]


async def test_the_port_range_can_run_out(tmp_path, fake_ssh):
    database = Database(tmp_path / "small.db")
    database.connect()
    launcher = SshLauncher(WorkerStore(database), FakeRegistry(),
                           port_range=(18000, 18000), local_port=LOCAL_PORT,
                           instance_id=CP)
    try:
        await launcher.add(_entry("one"))
        await launcher.add(_entry("two", target="bob@other"))
        await _until(lambda: launcher.describe("two")["state"] == "failed",
                     "a full range was not reported")
        assert "no free slot" in launcher.describe("two")["last_error"]
    finally:
        await launcher.shutdown()
        database.close()


def test_the_instance_id_is_read_only_when_a_launch_needs_it(
        tmp_path, monkeypatch):
    import app.config as config

    monkeypatch.setattr(config, "_user_data_root", lambda: tmp_path)
    launcher = SshLauncher(None, FakeRegistry())
    assert not (tmp_path / ssh_launch.INSTANCE_ID_FILE).exists()
    made = launcher.instance_id
    assert (tmp_path / ssh_launch.INSTANCE_ID_FILE).read_text().strip() \
        == made
    assert launcher.instance_id == made


def test_a_slot_is_not_a_port():
    """The server's own port number is a slot like any other: a slot names
    a state directory, no TCP port."""
    launcher = SshLauncher(None, FakeRegistry(), port_range=(8000, 8002),
                           local_port=8000, instance_id=CP)
    assert launcher.free_port("x") == 8000


def test_two_control_planes_never_share_a_remote_directory(tmp_path,
                                                           monkeypatch):
    """Two control planes on one remote account, both on slot 18000: their
    directories, sockets, credential addresses and agents differ."""
    other = "0123abcd"
    entry = _entry()
    for cp in (CP, other):
        where = f".codefyui-ssh/{cp}-18000"
        assert where in ssh_launch.remote_prepare_command(cp, 18000)
        assert where in ssh_launch.remote_stop_command(cp, 18000)
        assert ssh_launch.control_url(cp, 18000) in ssh_launch.remote_command(
            entry, cp, 18000)
    assert ssh_launch.control_url(CP, 18000) != ssh_launch.control_url(
        other, 18000)
    # The id is made once and kept in the user-data directory.
    first = ssh_launch.load_instance_id(tmp_path)
    assert ssh_launch.INSTANCE_ID_PATTERN.fullmatch(first)
    assert ssh_launch.load_instance_id(tmp_path) == first
    assert (tmp_path / ssh_launch.INSTANCE_ID_FILE).read_text().strip() \
        == first
    (tmp_path / ssh_launch.INSTANCE_ID_FILE).write_text("junk")
    assert ssh_launch.load_instance_id(tmp_path) != "junk"


# ── delete cleans up on the remote host (#829) ────────────────────────────


def _removes(log: Path) -> list[list[str]]:
    return [argv for argv in _oneshots(log) if "rm -rf" in argv[-1]]


async def test_delete_removes_the_slot_and_the_credential_disable_does_not(
        launcher, fake_ssh):
    await launcher.add(_entry())
    await _until(lambda: launcher.describe("gpu-box")["remote_pid"] == 4242,
                 "the session never attached")
    # Disable and Enable keep both: Enable reuses the stored credential.
    await launcher.set_enabled("gpu-box", False)
    await launcher.set_enabled("gpu-box", True)
    await _until(lambda: len(_tokens(fake_ssh)) == 2, "never relaunched")
    assert _removes(fake_ssh) == []
    assert await launcher.remove("gpu-box") is True
    [remove] = _removes(fake_ssh)
    # After the session ended (its ``stop`` went first), its own slot only.
    assert _stdin(fake_ssh)[-1] == "stop"
    assert _all_calls(fake_ssh)[-1] == remove
    assert "ExitOnForwardFailure=yes" not in remove and "-R" not in remove
    command = remove[-1]
    assert f'd="$HOME"/.codefyui-ssh/{CP}-18000;' in command
    assert f'"$HOME"/CodefyUI/cdui agent forget {URL} ' in command
    assert 'rm -rf "$d"' in command
    assert await launcher.store.list_ssh_workers() == []


async def test_delete_of_a_disabled_or_never_launched_entry(launcher,
                                                            fake_ssh):
    await launcher.store.add_ssh_worker(_entry(enabled=False,
                                               remote_port=18004))
    await launcher.store.add_ssh_worker(_entry("new-box", target="b@new",
                                               enabled=False))
    await launcher.start_all()
    assert await launcher.remove("gpu-box") is True
    [remove] = _removes(fake_ssh)
    assert f".codefyui-ssh/{CP}-18004;" in remove[-1]
    # No slot was ever taken: nothing was made on the remote host.
    assert await launcher.remove("new-box") is True
    assert len(_removes(fake_ssh)) == 1


async def test_delete_succeeds_when_the_host_is_unreachable(
        launcher, fake_ssh, monkeypatch, caplog):
    await launcher.add(_entry())
    await _until(lambda: _tokens(fake_ssh), "ssh never ran")
    monkeypatch.setenv("FAKE_SSH_MODE", "fail")
    with caplog.at_level("WARNING", logger="app.core.ssh_launch"):
        assert await launcher.remove("gpu-box") is True
    assert len(_removes(fake_ssh)) == 1
    assert await launcher.store.list_ssh_workers() == []
    [warning] = [r.getMessage() for r in caplog.records
                 if "left behind" in r.getMessage()]
    assert f"~/.codefyui-ssh/{CP}-18000/" in warning
    assert f"cdui agent forget {URL}" in warning
    assert "Permission denied" in warning


async def test_a_hanging_delete_cleanup_is_bounded(launcher, fake_ssh,
                                                   monkeypatch):
    await launcher.store.add_ssh_worker(_entry(enabled=False,
                                               remote_port=18000))
    await launcher.start_all()
    monkeypatch.setattr(ssh_launch, "REMOVE_TIMEOUT_S", 0.5)
    monkeypatch.setenv("FAKE_SSH_HANG_S", "30")
    began = time.monotonic()
    assert await launcher.remove("gpu-box") is True
    assert time.monotonic() - began < 10
    pid = int(Path(str(fake_ssh) + ".pid").read_text().split()[0])
    assert not _alive(pid)
    assert await launcher.store.list_ssh_workers() == []



# ── stop, enable, disable, restart ────────────────────────────────────────


async def test_disable_drains_then_ends_the_session_by_closing_stdin(
        launcher, fake_ssh):
    await launcher.add(_entry())
    await _until(lambda: _tokens(fake_ssh), "ssh never read a token")
    began = time.monotonic()
    view = await launcher.set_enabled("gpu-box", False)
    # The fake exits on stdin EOF: no terminate needed, so no grace wait.
    assert time.monotonic() - began < ssh_launch.STOP_GRACE_S
    assert launcher.registry.drained == {"gpu-box": True}
    assert view["enabled"] is False and view["state"] == "stopped"
    assert view["pid"] is None
    assert not (await launcher.store.list_ssh_workers())[0].enabled
    # No unspent token is left behind.
    assert await launcher.store.list_enrollments() == []

    view = await launcher.set_enabled("gpu-box", True)
    assert launcher.registry.drained == {"gpu-box": False}
    await _until(lambda: len(_tokens(fake_ssh)) == 2, "enable did not relaunch")
    assert launcher.describe("gpu-box")["state"] == "connecting"
    assert await launcher.set_enabled("nope", True) is None


async def test_restart_relaunches_enabled_rows_and_drains_disabled_ones(
        tmp_path, fake_ssh):
    database = Database(tmp_path / "boot.db")
    database.connect()
    store = WorkerStore(database)
    await store.add_ssh_worker(_entry("on-box", remote_port=18005))
    await store.add_ssh_worker(_entry("off-box", target="bob@off",
                                      enabled=False))
    registry = FakeRegistry()
    launcher = SshLauncher(store, registry, port_range=RANGE, instance_id=CP,
                           local_port=LOCAL_PORT)
    try:
        await launcher.start_all()
        await _until(lambda: _tokens(fake_ssh), "the enabled row never ran")
        [argv] = _calls(fake_ssh)
        # The stored slot is reused (and with it the credential the agent
        # keeps for that address).
        assert argv[argv.index("-R") + 1].startswith(
            f"/home/fake/.codefyui-ssh/{CP}-18005/sock:")
        assert registry.drained == {"off-box": True}
        assert [v["id"] for v in launcher.describe_all()] == ["off-box",
                                                              "on-box"]
        assert launcher.describe("off-box")["state"] == "stopped"
        assert await launcher.remove("on-box") is True
        assert await launcher.remove("on-box") is False
    finally:
        await launcher.shutdown()
        database.close()


# ── the remote command, run by a real sh ──────────────────────────────────


TOKEN = "abcdef.0123456789abcdef"


def _fake_cdui(home: Path) -> Path:
    """A ``cdui`` that records each call and, for ``agent start``, its
    token and PATH, then idles until SIGTERM (recorded in ``got_term``)."""
    script = home / "cdui"
    script.write_text(
        "#!/bin/sh\n"
        'out="$(dirname "$0")"\n'
        'printf "%s\\n" "$*" >> "$out/args"\n'
        'case "$2" in forget) exit 0;; esac\n'
        'printf "%s" "$CODEFYUI_AGENT_TOKEN" > "$out/token"\n'
        'printf "%s" "$PATH" > "$out/path"\n'
        'echo "fake agent says hello"\n'
        'trap "echo term >> \\"$out/got_term\\"; exit 0" TERM\n'
        "while :; do sleep 0.05; done\n")
    script.chmod(0o755)
    return script


def _run_remote(home: Path, command: str) -> subprocess.Popen:
    env = {**os.environ, "HOME": str(home)}
    return subprocess.Popen(["/bin/sh", "-c", command], env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)


def _wait_file(path: Path, timeout: float = 10.0, text: str = "") -> str:
    deadline = time.monotonic() + timeout
    while not path.exists() or text not in path.read_text():
        if time.monotonic() > deadline:
            raise AssertionError(f"{path} never had {text!r}")
        time.sleep(0.02)
    time.sleep(0.05)
    return path.read_text()


def _alive(pid: int) -> bool:
    out = subprocess.run(["ps", "-p", str(pid), "-o", "stat="],
                         capture_output=True, text=True).stdout.strip()
    return bool(out) and not out.startswith("Z")


@pytest.fixture
def remote_home(tmp_path):
    home = tmp_path / "remote-home"
    home.mkdir()
    _fake_cdui(home)
    yield home
    # Whatever a test left running: stop it.
    pid_file = home / ".codefyui-ssh" / f"{CP}-18000" / "pid"
    if pid_file.exists():
        subprocess.run(["kill", "-TERM", pid_file.read_text().strip()],
                       capture_output=True)


def _state(home: Path, port: int = 18000, cp: str = CP) -> Path:
    return home / ".codefyui-ssh" / f"{cp}-{port}"


def test_the_remote_agent_gets_the_token_in_its_env_and_stop_ends_it(
        remote_home):
    command = ssh_launch.remote_command(_entry(remote_cdui="~/cdui"), CP,
                                        18000)
    assert "'" not in command[len("sh -c '"):-1]
    assert "\\" not in command
    proc = _run_remote(remote_home, command)
    proc.stdin.write(f"{TOKEN}\n".encode())
    proc.stdin.flush()
    assert _wait_file(remote_home / "token") == TOKEN
    args = (remote_home / "args").read_text()
    assert args.strip() == (
        f"agent start --uds {_state(remote_home)}/sock --foreground "
        "--control http://cp-abcd1234.invalid:18000 --name gpu-box")
    assert "abcdef" not in args
    assert (_wait_file(remote_home / "path").split(":")[:2]
            == [f"{remote_home}/.local/bin", f"{remote_home}/.cargo/bin"])
    pid = int(_wait_file(_state(remote_home) / "pid"))
    assert _wait_file(_state(remote_home) / "log", text="hello")
    # Disable: a ``stop`` line, then the end of input.
    proc.stdin.write(b"stop\n")
    proc.stdin.close()
    output = proc.stdout.read().decode()
    proc.wait(timeout=10)
    assert proc.returncode == 0
    assert _wait_file(remote_home / "got_term").strip() == "term"
    assert not _alive(pid)
    assert f"codefyui-ssh: agent pid {pid}" in output
    assert "fake agent says hello" in output
    assert not (_state(remote_home) / "pid").exists()


def test_a_dropped_session_leaves_the_agent_for_the_grace_then_stops_it(
        remote_home):
    command = ssh_launch.remote_command(_entry(remote_cdui="~/cdui"), CP,
                                        18000, grace_s=2)
    proc = _run_remote(remote_home, command)
    proc.stdin.write(f"{TOKEN}\n".encode())
    proc.stdin.flush()
    pid = int(_wait_file(_state(remote_home) / "pid"))
    _wait_file(remote_home / "path")
    proc.stdin.close()  # the session drops: no ``stop``
    time.sleep(1.0)
    assert _alive(pid) and not (remote_home / "got_term").exists()
    proc.wait(timeout=10)
    assert _wait_file(remote_home / "got_term").strip() == "term"
    assert not _alive(pid)


def test_a_new_session_within_the_grace_reattaches_to_the_same_agent(
        remote_home):
    entry = _entry(remote_cdui="~/cdui")
    first = _run_remote(remote_home, ssh_launch.remote_command(
        entry, CP, 18000, grace_s=2))
    first.stdin.write(f"{TOKEN}\n".encode())
    first.stdin.flush()
    pid = int(_wait_file(_state(remote_home) / "pid"))
    _wait_file(remote_home / "path")
    first.stdin.close()  # dropped

    second = _run_remote(remote_home, ssh_launch.remote_command(
        entry, CP, 18000, grace_s=2, forget_first=True))
    second.stdin.write(f"{TOKEN}\n".encode())
    second.stdin.flush()
    first.wait(timeout=10)  # superseded: it leaves without a kill
    time.sleep(0.5)
    assert _alive(pid), "the first session's grace stopped a re-attached agent"
    assert not (remote_home / "got_term").exists()
    # One agent start ever, and no forget for an agent that is alive.
    assert (remote_home / "args").read_text().count("agent start") == 1
    assert "agent forget" not in (remote_home / "args").read_text()
    second.stdin.write(b"stop\n")
    second.stdin.close()
    output = second.stdout.read().decode()
    second.wait(timeout=10)
    assert f"codefyui-ssh: agent pid {pid} reattached" in output
    assert _wait_file(remote_home / "got_term").strip() == "term"
    assert not _alive(pid)


def test_the_remote_shell_refuses_to_start_without_a_token(remote_home):
    proc = _run_remote(remote_home, ssh_launch.remote_command(
        _entry(remote_cdui=str(remote_home / "cdui")), CP, 18000))
    proc.stdin.close()
    assert proc.wait(timeout=10) == ssh_launch.NO_TOKEN_EXIT
    assert not (remote_home / "args").exists()


def test_forget_first_runs_cdui_agent_forget_before_a_fresh_start(
        remote_home):
    proc = _run_remote(remote_home, ssh_launch.remote_command(
        _entry(remote_cdui="~/cdui"), CP, 18000, forget_first=True))
    proc.stdin.write(f"{TOKEN}\n".encode())
    proc.stdin.flush()
    _wait_file(remote_home / "token")
    lines = (remote_home / "args").read_text().splitlines()
    assert lines[0] == "agent forget http://cp-abcd1234.invalid:18000"
    assert lines[1].startswith("agent start --uds ")
    proc.stdin.write(b"stop\n")
    proc.stdin.close()
    assert proc.wait(timeout=10) == 0


def test_the_one_shot_stop_ends_a_surviving_agent(remote_home):
    proc = _run_remote(remote_home, ssh_launch.remote_command(
        _entry(remote_cdui="~/cdui"), CP, 18000, grace_s=60))
    proc.stdin.write(f"{TOKEN}\n".encode())
    proc.stdin.flush()
    pid = int(_wait_file(_state(remote_home) / "pid"))
    _wait_file(remote_home / "path")  # the agent is up, its trap set
    proc.stdin.close()  # dropped; the agent waits out its grace
    stop = subprocess.run(["/bin/sh", "-c", ssh_launch.remote_stop_command(
        CP, 18000)], env={**os.environ, "HOME": str(remote_home)},
        capture_output=True, text=True, timeout=10)
    assert f"stopped agent pid {pid}" in stop.stdout
    assert _wait_file(remote_home / "got_term").strip() == "term"
    proc.wait(timeout=10)
    # Nothing to stop: still fine.
    again = subprocess.run(["/bin/sh", "-c", ssh_launch.remote_stop_command(
        CP, 18000)], env={**os.environ, "HOME": str(remote_home)},
        capture_output=True, text=True, timeout=10)
    assert again.returncode == 0 and again.stdout == ""


def test_the_prepare_command_makes_a_private_directory_and_clears_the_socket(
        remote_home):
    command = ssh_launch.remote_prepare_command(CP, 18000)
    assert "'" not in command[len("sh -c '"):-1] and "\\" not in command
    env = {**os.environ, "HOME": str(remote_home)}
    first = subprocess.run(["/bin/sh", "-c", command], env=env,
                           capture_output=True, text=True, timeout=10)
    assert first.returncode == 0, first.stderr
    state = _state(remote_home)
    assert first.stdout.strip() == f"codefyui-ssh: dir {state}"
    assert ssh_launch.REMOTE_DIR.search(first.stdout).group(1) == str(state)
    assert state.stat().st_mode & 0o777 == 0o700
    assert state.parent.stat().st_mode & 0o777 == 0o700
    # A socket a dropped session left behind is removed.
    (state / "sock").write_text("stale")
    state.parent.chmod(0o755)
    again = subprocess.run(["/bin/sh", "-c", command], env=env,
                           capture_output=True, text=True, timeout=10)
    assert again.returncode == 0
    assert not (state / "sock").exists()
    assert state.parent.stat().st_mode & 0o777 == 0o700


def test_a_reattached_session_reads_only_new_agent_output(remote_home):
    """A "refused the token" line the agent wrote before this session
    attached is not shown to it again: the launcher would read it as news
    and restart a healthy agent with ``cdui agent forget``."""
    entry = _entry(remote_cdui="~/cdui")
    first = _run_remote(remote_home, ssh_launch.remote_command(
        entry, CP, 18000, grace_s=3))
    first.stdin.write(f"{TOKEN}\n".encode())
    first.stdin.flush()
    pid = int(_wait_file(_state(remote_home) / "pid"))
    _wait_file(remote_home / "path")
    with open(_state(remote_home) / "log", "a") as log:
        log.write(f"x ERROR {ssh_launch.TOKEN_REFUSED} (HTTP 401)\n"
                  f"the credential in y {ssh_launch.CREDENTIAL_MISMATCH} z\n")
    first.stdin.close()  # dropped
    second = _run_remote(remote_home, ssh_launch.remote_command(
        entry, CP, 18000, grace_s=3))
    second.stdin.write(f"{TOKEN}\n".encode())
    second.stdin.flush()
    first.wait(timeout=10)  # superseded: it leaves without a kill
    time.sleep(1.0)
    with open(_state(remote_home) / "log", "a") as log:
        log.write("fresh line after attaching\n")
    time.sleep(1.5)
    second.stdin.write(b"stop\n")
    second.stdin.close()
    output = second.stdout.read().decode()
    second.wait(timeout=10)
    assert f"codefyui-ssh: agent pid {pid} reattached" in output
    assert ssh_launch.TOKEN_REFUSED not in output
    assert ssh_launch.CREDENTIAL_MISMATCH not in output
    assert "fresh line after attaching" in output


def _legacy_agent(home: Path, args: list[str]) -> subprocess.Popen:
    """A process that looks like an agent of an earlier build: its command
    line is *args*, its pid in the old ``<slot>/`` layout."""
    script = home / "old-agent"
    script.write_text("#!/bin/sh\nwhile :; do sleep 0.05; done\n")
    script.chmod(0o755)
    proc = subprocess.Popen([str(script), *args], start_new_session=True)
    legacy = home / ".codefyui-ssh" / "18000"
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "pid").write_text(str(proc.pid))
    return proc


def _prepare(home: Path) -> str:
    result = subprocess.run(
        ["/bin/sh", "-c", ssh_launch.remote_prepare_command(CP, 18000)],
        env={**os.environ, "HOME": str(home)}, capture_output=True,
        text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_prepare_stops_a_detached_legacy_tcp_agent(remote_home):
    old = _legacy_agent(remote_home, [
        "agent", "start", "--foreground", "--control",
        "http://127.0.0.1:18000", "--name", "gpu-box"])
    try:
        out = _prepare(remote_home)
        assert f"stopped legacy agent pid {old.pid}" in out
        old.wait(timeout=10)
    finally:
        if old.poll() is None:
            old.kill()
            old.wait()


def test_prepare_leaves_an_attached_or_socket_agent_alone(remote_home):
    # A legacy agent whose session is still attached: reported only.
    session = subprocess.Popen(["sleep", "30"], start_new_session=True)
    old = _legacy_agent(remote_home, [
        "agent", "start", "--foreground", "--control",
        "http://127.0.0.1:18000", "--name", "gpu-box"])
    (remote_home / ".codefyui-ssh" / "18000" / "session").write_text(
        str(session.pid))
    try:
        out = _prepare(remote_home)
        assert "is still attached; left running" in out
        assert old.poll() is None
    finally:
        for proc in (old, session):
            proc.kill()
            proc.wait()
    # An agent on a socket, or one dialling another address: not touched.
    for args in (["agent", "start", "--uds", "/x/sock", "--control",
                  "http://127.0.0.1:18000"],
                 ["agent", "start", "--control", "http://127.0.0.1:18001"]):
        (remote_home / ".codefyui-ssh" / "18000" / "session").unlink(
            missing_ok=True)
        other = _legacy_agent(remote_home, args)
        try:
            out = _prepare(remote_home)
            assert "legacy" not in out and other.poll() is None
        finally:
            other.kill()
            other.wait()


def test_a_reused_pid_is_never_taken_for_the_agent(remote_home):
    """The pid file outlives its agent and the pid may be reused: a process
    that is not this directory's agent is neither re-attached to nor
    stopped."""
    state = _state(remote_home)
    state.mkdir(parents=True)
    stranger = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        (state / "pid").write_text(str(stranger.pid))
        stop = subprocess.run(
            ["/bin/sh", "-c", ssh_launch.remote_stop_command(CP, 18000)],
            env={**os.environ, "HOME": str(remote_home)},
            capture_output=True, text=True, timeout=10)
        assert stop.stdout == "" and stranger.poll() is None
        proc = _run_remote(remote_home, ssh_launch.remote_command(
            _entry(remote_cdui="~/cdui"), CP, 18000))
        proc.stdin.write(f"{TOKEN}\n".encode())
        proc.stdin.flush()
        _wait_file(remote_home / "token")
        assert "agent start --uds" in (remote_home / "args").read_text()
        assert stranger.poll() is None
        proc.stdin.write(b"stop\n")
        proc.stdin.close()
        output = proc.stdout.read().decode()
        assert proc.wait(timeout=10) == 0
        assert "reattached" not in output
    finally:
        stranger.kill()
        stranger.wait()


def test_two_control_planes_on_one_account_keep_their_own_agents(
        remote_home):
    """Same account, same slot, two control planes: each starts, stops and
    prepares only its own agent and socket."""
    other = "0123abcd"
    env = {**os.environ, "HOME": str(remote_home)}
    sessions = {}
    for cp in (CP, other):
        prepared = subprocess.run(
            ["/bin/sh", "-c", ssh_launch.remote_prepare_command(cp, 18000)],
            env=env, capture_output=True, text=True, timeout=10)
        assert prepared.returncode == 0
        (_state(remote_home, cp=cp) / "sock").write_text("live")
        proc = _run_remote(remote_home, ssh_launch.remote_command(
            _entry(remote_cdui="~/cdui"), cp, 18000, grace_s=60))
        proc.stdin.write(f"{TOKEN}\n".encode())
        proc.stdin.flush()
        sessions[cp] = (proc, int(_wait_file(_state(remote_home, cp=cp)
                                             / "pid")))
    try:
        assert sessions[CP][1] != sessions[other][1]
        # Preparing one control plane's session leaves the other's socket.
        subprocess.run(
            ["/bin/sh", "-c", ssh_launch.remote_prepare_command(CP, 18000)],
            env=env, capture_output=True, text=True, timeout=10)
        assert not (_state(remote_home) / "sock").exists()
        assert (_state(remote_home, cp=other) / "sock").read_text() == "live"
        # Stopping one stops its agent only.
        for proc, _ in sessions.values():
            proc.stdin.close()  # dropped: the agents wait out their grace
        stop = subprocess.run(
            ["/bin/sh", "-c", ssh_launch.remote_stop_command(CP, 18000)],
            env=env, capture_output=True, text=True, timeout=10)
        assert f"stopped agent pid {sessions[CP][1]}" in stop.stdout
        deadline = time.monotonic() + 10
        while _alive(sessions[CP][1]) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(sessions[CP][1])
        assert _alive(sessions[other][1])
    finally:
        for proc, pid in sessions.values():
            subprocess.run(["kill", "-TERM", str(pid)], capture_output=True)
            proc.wait(timeout=10)


def _remove(home: Path, cp: str = CP) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/bin/sh", "-c", ssh_launch.remote_remove_command(
            _entry(remote_cdui="~/cdui"), cp, 18000)],
        env={**os.environ, "HOME": str(home)}, capture_output=True,
        text=True, timeout=30)


def test_the_remove_command_cleans_only_this_control_planes_slot(
        remote_home):
    command = ssh_launch.remote_remove_command(
        _entry(remote_cdui="~/cdui"), CP, 18000)
    assert "'" not in command[len("sh -c '"):-1] and "\\" not in command
    other = "0123abcd"
    for cp in (CP, other):
        state = _state(remote_home, cp=cp)
        state.mkdir(parents=True)
        for name in ("log", "sock", "pid", "session"):
            (state / name).write_text("99999999")
    result = _remove(remote_home)
    assert result.returncode == 0, result.stderr
    assert f"codefyui-ssh: forgot http://cp-{CP}.invalid:18000" \
        in result.stdout
    assert f"codefyui-ssh: removed {_state(remote_home)}" in result.stdout
    assert not _state(remote_home).exists()
    assert (_state(remote_home, cp=other) / "log").exists()
    assert (remote_home / "args").read_text().splitlines() == [
        f"agent forget http://cp-{CP}.invalid:18000"]
    # The last one out removes ~/.codefyui-ssh too.
    result = _remove(remote_home, cp=other)
    assert result.returncode == 0
    assert not (remote_home / ".codefyui-ssh").exists()
    # Nothing there any more: still fine.
    assert _remove(remote_home).returncode == 0


def test_the_remove_command_stops_a_surviving_agent_first(remote_home):
    proc = _run_remote(remote_home, ssh_launch.remote_command(
        _entry(remote_cdui="~/cdui"), CP, 18000, grace_s=60))
    proc.stdin.write(f"{TOKEN}\n".encode())
    proc.stdin.flush()
    pid = int(_wait_file(_state(remote_home) / "pid"))
    _wait_file(remote_home / "path")
    proc.stdin.close()  # dropped; the agent waits out its grace
    try:
        assert _alive(pid)
        result = _remove(remote_home)
        assert _wait_file(remote_home / "got_term").strip() == "term"
        assert not _alive(pid)
        assert "codefyui-ssh: removed" in result.stdout
        assert not _state(remote_home).exists()
    finally:
        proc.wait(timeout=15)


def test_the_remove_command_keeps_the_directory_of_an_attached_session(
        remote_home):
    proc = _run_remote(remote_home, ssh_launch.remote_command(
        _entry(remote_cdui="~/cdui"), CP, 18000, grace_s=60))
    proc.stdin.write(f"{TOKEN}\n".encode())
    proc.stdin.flush()
    pid = int(_wait_file(_state(remote_home) / "pid"))
    _wait_file(remote_home / "path")
    try:
        # An agent that will not stop, under a session still attached.
        subprocess.run(["kill", "-STOP", str(pid)], check=True)
        result = _remove(remote_home)
        assert "still attached; kept" in result.stdout
        assert "still runs; kept" in result.stdout
        assert "codefyui-ssh: removed" not in result.stdout
        assert _state(remote_home).exists()
    finally:
        subprocess.run(["kill", "-CONT", str(pid)], capture_output=True)
        proc.stdin.write(b"stop\n")
        proc.stdin.close()
        proc.wait(timeout=15)


# ── the remote script leaves no process behind (#829) ─────────────────────


def _shells() -> list[str]:
    """The shells the remote script is run with here: ``/bin/sh`` (bash on
    macOS, dash on Debian/Ubuntu), and bash, dash and BusyBox ``sh`` where
    they are installed. bash reports a job it killed on stderr."""
    found = ["/bin/sh"]
    for name in ("bash", "dash", "busybox"):
        if shutil.which(name):
            found.append(name)
    return found


def _run_with(shell: str, home: Path, command: str) -> subprocess.Popen:
    """*command*'s script (``sh -c '<script>'``) run by *shell*, in a
    process group of its own (its id is the shell's pid)."""
    assert command.startswith("sh -c '") and command.endswith("'")
    argv = [shell, "sh"] if shell == "busybox" else [shell]
    return subprocess.Popen(
        [*argv, "-c", command[len("sh -c '"):-1]],
        env={**os.environ, "HOME": str(home)}, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True)


def _group(pgid: int) -> list[str]:
    """The live processes of process group *pgid*: ``pid command``."""
    out = subprocess.run(["ps", "-A", "-o", "pid=,pgid=,stat=,command="],
                         capture_output=True, text=True).stdout
    left = []
    for line in out.splitlines():
        fields = line.split(None, 3)
        if (len(fields) == 4 and fields[1] == str(pgid)
                and not fields[2].startswith("Z")):
            left.append(f"{fields[0]} {fields[3]}")
    return left


def _tails(pgid: int) -> list[str]:
    """The ``tail`` processes of process group *pgid*."""
    return [p for p in _group(pgid) if p.split(" ", 1)[1].startswith("tail ")]


def _settles(predicate: Any, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


@pytest.fixture
def shell_home(tmp_path):
    """A "remote" home for the process tests; whatever they leave in their
    process groups is killed by the tests themselves."""
    home = tmp_path / "rh"
    home.mkdir()
    _fake_cdui(home)
    yield home
    for pid_file in (home / ".codefyui-ssh").glob("*/pid"):
        subprocess.run(["kill", "-TERM", pid_file.read_text().strip()],
                       capture_output=True)


@pytest.mark.parametrize("shell", _shells())
def test_a_superseded_session_leaves_no_tail_once_its_ssh_is_gone(
        shell_home, shell):
    """The leak #829 found: a session drops (ssh gone: no more input, its
    output a broken pipe), a newer one re-attaches, then the agent stops.
    The first session's shell kills its grace watcher; bash reports that on
    stderr, which used to kill the shell with SIGPIPE before it ended its
    ``tail -f``, orphaned for good (and the watcher's ``sleep``)."""
    entry = _entry(remote_cdui="~/cdui")
    first = _run_with(shell, shell_home, ssh_launch.remote_command(
        entry, CP, 18000, grace_s=60))
    second = None
    try:
        first.stdin.write(f"{TOKEN}\n".encode())
        first.stdin.flush()
        pid = int(_wait_file(_state(shell_home) / "pid"))
        _wait_file(_state(shell_home) / "log", text="hello")
        assert _tails(first.pid)
        first.stdin.close()
        first.stdout.close()  # ssh is gone: whatever the shell writes, EPIPE
        second = _run_with(shell, shell_home, ssh_launch.remote_command(
            entry, CP, 18000, grace_s=60))
        second.stdin.write(f"{TOKEN}\n".encode())
        second.stdin.flush()
        _wait_file(_state(shell_home) / "session", text=str(second.pid))
        time.sleep(0.5)
        assert _alive(pid) and first.poll() is None
        second.stdin.write(b"stop\n")
        second.stdin.close()
        second.wait(timeout=15)
        first.wait(timeout=15)
        assert not _alive(pid)
        for proc in (first, second):
            assert _settles(lambda p=proc: not _group(p.pid)), (
                f"left behind: {_group(proc.pid)}")
    finally:
        for proc in (first, second):
            if proc is None:
                continue
            subprocess.run(["kill", "-9", f"-{proc.pid}"],
                           capture_output=True)
            if proc.poll() is None:
                proc.kill()
                proc.wait()


@pytest.mark.parametrize("signal_name", ["HUP", "TERM"])
@pytest.mark.parametrize("shell", _shells())
def test_a_signalled_session_shell_takes_its_tail_with_it(shell_home, shell,
                                                         signal_name):
    """The session's shell itself killed (sshd hanging up on it, a
    ``kill``): its ``tail`` goes with it. The agent stays for the resume
    grace, as on any dropped session."""
    proc = _run_with(shell, shell_home, ssh_launch.remote_command(
        _entry(remote_cdui="~/cdui"), CP, 18000, grace_s=60))
    try:
        proc.stdin.write(f"{TOKEN}\n".encode())
        proc.stdin.flush()
        pid = int(_wait_file(_state(shell_home) / "pid"))
        _wait_file(_state(shell_home) / "log", text="hello")
        subprocess.run(["kill", f"-{signal_name}", str(proc.pid)],
                       check=True)
        proc.wait(timeout=15)
        assert _settles(lambda: not _tails(proc.pid)), _group(proc.pid)
        assert _alive(pid)
    finally:
        subprocess.run(["kill", "-9", f"-{proc.pid}"], capture_output=True)
        if proc.poll() is None:
            proc.kill()
            proc.wait()


# ── validation ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("target", ["alice@gpu-box", "gpu-box",
                                    "lab.gpu_1", "a.b@host.example.org"])
def test_targets_that_are_accepted(target):
    assert ssh_launch.validate_target(f" {target} ") == target


@pytest.mark.parametrize("target", ["", "-oProxyCommand=evil", "a b",
                                    "alice@", "@host", "host;rm", "h'x",
                                    "alice@-oFoo"])
def test_targets_that_are_refused(target):
    with pytest.raises(ssh_launch.SshConfigError):
        ssh_launch.validate_target(target)


def test_remote_cdui_paths():
    assert ssh_launch.validate_remote_cdui(None) == "~/CodefyUI/cdui"
    assert ssh_launch.validate_remote_cdui(" /opt/x/cdui ") == "/opt/x/cdui"
    assert ssh_launch.validate_remote_cdui("cdui") == "cdui"
    for bad in ("-x", "a b", "~/x;y", "$(id)", "~/../etc/cdui", "x'y",
                "~user/cdui"):
        with pytest.raises(ssh_launch.SshConfigError):
            ssh_launch.validate_remote_cdui(bad)


def test_identity_files_must_exist_and_be_absolute(tmp_path, monkeypatch):
    key = tmp_path / "key"
    key.write_text("k")
    assert ssh_launch.validate_identity(str(key)) == str(key)
    assert ssh_launch.validate_identity("  ") is None
    monkeypatch.setenv("HOME", str(tmp_path))
    assert ssh_launch.validate_identity("~/key") == str(key)
    with pytest.raises(ssh_launch.SshConfigError, match="absolute"):
        ssh_launch.validate_identity("key")
    with pytest.raises(ssh_launch.SshConfigError, match="does not exist"):
        ssh_launch.validate_identity(str(tmp_path / "missing"))


def test_default_names_and_port_ranges():
    assert ssh_launch.default_worker_name("alice@GPU1.lab.example") == "gpu1"
    assert ssh_launch.default_worker_name("lab_box") == "lab-box"
    assert ssh_launch.default_worker_name("___") == "ssh-worker"
    assert ssh_launch.parse_port_range("19000-19010") == (19000, 19010)
    assert ssh_launch.parse_port_range("nope") == ssh_launch.DEFAULT_PORT_RANGE
    assert ssh_launch.parse_port_range("90-100") == ssh_launch.DEFAULT_PORT_RANGE


# ── routes ────────────────────────────────────────────────────────────────


@pytest.fixture
async def api(tmp_path, fake_ssh):
    database = Database(tmp_path / "api.db")
    database.connect()
    store = WorkerStore(database)
    registry = WorkerRegistry()
    launcher = SshLauncher(store, registry, port_range=RANGE, instance_id=CP,
                           local_port=LOCAL_PORT)
    app.state.db = database
    app.state.worker_store = store
    app.state.worker_registry = registry
    app.state.ssh_launcher = launcher
    try:
        yield launcher
    finally:
        await launcher.shutdown()
        await registry.shutdown()
        database.close()
        for attribute in ("db", "worker_store", "worker_registry",
                          "ssh_launcher"):
            if hasattr(app.state, attribute):
                delattr(app.state, attribute)


def _client(token: bool = True,
            base: str = "http://127.0.0.1:8000") -> AsyncClient:
    headers = {TOKEN_HEADER: session_token()} if token else {}
    return AsyncClient(transport=ASGITransport(app=app), base_url=base,
                       headers=headers)


async def test_the_ssh_routes_add_list_disable_enable_and_delete(api,
                                                                  fake_ssh):
    async with _client() as client:
        added = await client.post("/api/workers/ssh", json={
            "target": "alice@gpu1.lab.example"})
        assert added.status_code == 200, added.text
        body = added.json()
        assert body["id"] == "gpu1"
        assert body["remote_cdui"] == "~/CodefyUI/cdui"
        assert body["state"] == "connecting" and body["enabled"] is True
        assert body["port"] == 22 and body["identity"] is None

        listed = (await client.get("/api/workers/ssh")).json()
        assert listed["ssh_available"] is True
        assert listed["port_range"] == list(RANGE)
        assert listed["default_remote_cdui"] == "~/CodefyUI/cdui"
        assert [e["id"] for e in listed["ssh_workers"]] == ["gpu1"]

        again = await client.post("/api/workers/ssh", json={
            "target": "bob@gpu1"})
        assert again.status_code == 409

        disabled = await client.post("/api/workers/ssh/gpu1/disable")
        assert disabled.json()["state"] == "stopped"
        enabled = await client.post("/api/workers/ssh/gpu1/enable")
        assert enabled.json()["enabled"] is True
        assert (await client.post("/api/workers/ssh/nope/enable")
                ).status_code == 404

        deleted = await client.delete("/api/workers/ssh/gpu1")
        assert deleted.json() == {"id": "gpu1", "deleted": True}
        assert (await client.delete("/api/workers/ssh/gpu1")
                ).status_code == 404


async def test_listing_needs_the_session_token_and_adding_too(api):
    async with _client(token=False) as anonymous:
        assert (await anonymous.get("/api/workers/ssh")).status_code in (
            401, 403)
        assert (await anonymous.post("/api/workers/ssh", json={
            "target": "a@b"})).status_code == 403


@pytest.mark.parametrize("body, message", [
    ({"target": "-oProxyCommand=x"}, "SSH target"),
    ({"target": "a@b", "identity": "/nope/key"}, "does not exist"),
    ({"target": "a@b", "remote_cdui": "~/x; rm"}, "remote cdui"),
    ({"target": "a@b", "name": "Bad Name"}, "worker name"),
    ({"target": "a@b", "name": "local"}, "reserved"),
])
async def test_bad_ssh_workers_are_refused(api, body, message):
    async with _client() as client:
        response = await client.post("/api/workers/ssh", json=body)
    assert response.status_code == 400
    assert message in response.json()["detail"]


async def test_a_known_worker_name_is_not_taken_over(api):
    store = api.store
    created = await store.create_enrollment()
    await store.enroll(created["token"], name="gpu-box", platform={},
                       mounts={}, devices=[])
    async with _client() as client:
        response = await client.post("/api/workers/ssh", json={
            "target": "alice@gpu-box"})
    assert response.status_code == 409
    assert "exists already" in response.json()["detail"]


async def test_no_ssh_means_no_ssh_worker(api, monkeypatch):
    monkeypatch.setattr(ssh_launch, "ssh_executable", lambda: None)
    async with _client() as client:
        response = await client.post("/api/workers/ssh", json={
            "target": "a@b"})
        listed = (await client.get("/api/workers/ssh")).json()
    assert response.status_code == 400
    assert "ssh was not found" in response.json()["detail"]
    assert listed["ssh_available"] is False


async def test_without_a_launcher_the_routes_answer_503():
    async with _client() as client:
        assert (await client.get("/api/workers/ssh")).status_code == 503


async def test_removing_the_worker_removes_its_ssh_entry(api, fake_ssh):
    async with _client() as client:
        await client.post("/api/workers/ssh", json={"target": "a@ssh-only"})
        response = await client.delete("/api/workers/ssh-only")
        assert response.status_code == 200, response.text
        assert response.json()["revoked"] is False
        assert "ssh-only" not in api.entries
        assert (await client.get("/api/workers/ssh")).json()[
            "ssh_workers"] == []


async def test_no_host_on_the_main_port_counts_as_a_tunnel(api, fake_ssh):
    """With an SSH worker running, a request to the main port naming its
    slot (or anything else) as ``Host`` is just a Host the guard refuses;
    one naming this server is an ordinary request, no tunnel either."""
    async with _client() as client:
        await client.post("/api/workers/ssh", json={"target": "a@b"})
    for base in ("http://127.0.0.1:18000", "http://localhost:18000"):
        async with _client(token=False, base=base) as forged:
            assert (await forged.post("/api/workers/enroll", json={
                "token": "nope", "name": "b"})).status_code == 421
            assert (await forged.get("/api/auth/bootstrap")).status_code \
                == 421
