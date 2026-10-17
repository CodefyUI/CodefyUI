"""End to end: an SSH worker (#784) comes up through a reverse tunnel.

The server is ``test_worker_e2e``'s (uvicorn on a free loopback port, its
real lifespan, so the ``SshLauncher`` there is the one that runs).

``test_an_ssh_worker_enrolls_runs_and_survives_disable_enable`` puts the
fake ``ssh`` (``tests/fake_ssh.py``) in ``tunnel`` mode: it really listens
on the ``-R`` Unix socket and relays to the SSH worker's own listener, and
runs the remote command with ``sh -c`` -- so the command line, the token
on stdin, the
tunnel listener, the enrollment, the stored credential and the stdin
watchdog are all the real ones. The "remote" ``cdui`` is a shim that runs
``python -m app.agent`` with a user-data dir of its own.

``test_a_real_ssh_worker`` uses the system ``ssh`` against a real host and
runs only when ``CODEFYUI_SSH_E2E_TARGET=user@host`` is set (with
``CODEFYUI_SSH_E2E_REMOTE_CDUI`` / ``_PORT`` / ``_IDENTITY`` when the
defaults do not fit). CI has no ssh server, so it skips there.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.core.auth import TOKEN_HEADER, session_token

from .test_worker_e2e import _graph, _wait_terminal, live_server  # noqa: F401

BACKEND_DIR = Path(__file__).resolve().parents[1]
FAKE_SSH = Path(__file__).resolve().parent / "fake_ssh.py"
NAME = "ssh-e2e"


def _client(port: int) -> httpx.Client:
    return httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30,
                        headers={TOKEN_HEADER: session_token()})


def _ssh_entry(client: httpx.Client, name: str) -> dict[str, Any] | None:
    for entry in client.get("/api/workers/ssh").json()["ssh_workers"]:
        if entry["id"] == name:
            return entry
    return None


def _worker(client: httpx.Client, name: str) -> dict[str, Any] | None:
    for worker in client.get("/api/workers").json()["workers"]:
        if worker["name"] == name:
            return worker
    return None


def _wait(predicate: Any, what: str, client: httpx.Client, name: str,
          timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            entry = _ssh_entry(client, name) or {}
            raise AssertionError(
                f"{what}; ssh state {entry.get('state')!r}, last error "
                f"{entry.get('last_error')!r}, tail:\n"
                + "\n".join(entry.get("tail") or []))
        time.sleep(0.1)


def _cp() -> str:
    """The live server's SSH instance id (its remote directories are
    ``~/.codefyui-ssh/<id>-<slot>``)."""
    from app.main import app

    return app.state.ssh_launcher.instance_id


def _up(client: httpx.Client, name: str) -> bool:
    entry = _ssh_entry(client, name) or {}
    worker = _worker(client, name) or {}
    # Online, or degraded (#781): the test's "remote" has a custom node
    # type (the ticker) this server lacks, so its catalog differs.
    return entry.get("state") == "up" and worker.get("status") in (
        "online", "degraded")


#: Six seconds of one metric point per tenth of a second: long enough to
#: drop the ssh session in the middle (``test_worker_e2e``'s ticker, longer).
TICKER_NODE = '''
import time

from app.core.node_base import BaseNode, DataType, PortDefinition


class SshTicker(BaseNode):
    NODE_NAME = "SshTicker"
    CATEGORY = "Test"

    @classmethod
    def define_inputs(cls):
        return []

    @classmethod
    def define_outputs(cls):
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs, params, *, context=None):
        for step in range(60):
            if context is not None:
                context.log_metric("tick", float(step), step=step)
            time.sleep(0.1)
        return {"value": 60}
'''


def _pid_alive(pid: int) -> bool:
    import subprocess

    out = subprocess.run(["ps", "-p", str(pid), "-o", "stat="],
                         capture_output=True, text=True).stdout.strip()
    return bool(out) and not out.startswith("Z")


@pytest.fixture
def fake_remote(tmp_path, monkeypatch):
    """The fake ssh in ``tunnel`` mode on PATH, and a "remote" home whose
    ``cdui`` runs ``python -m app.agent`` with its own user-data dir.
    Returns ``(ssh argv log, remote home, cdui path)``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "ssh"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_SSH}" "$@"\n')
    shim.chmod(0o755)
    log = tmp_path / "ssh.log"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SSH_LOG", str(log))
    monkeypatch.setenv("FAKE_SSH_MODE", "tunnel")
    monkeypatch.delenv("CODEFYUI_AGENT_TOKEN", raising=False)
    # Short: the forward's Unix socket lives under it, and ``sun_path``
    # holds about 100 bytes (pytest's tmp_path on macOS is longer).
    remote_home = Path(tempfile.mkdtemp(prefix="cdui-ssh-", dir="/tmp"))
    monkeypatch.setenv("FAKE_SSH_HOME", str(remote_home))
    nodes_dir = tmp_path / "nodes"
    nodes_dir.mkdir()
    (nodes_dir / "ssh_ticker.py").write_text(TICKER_NODE)
    cdui = remote_home / "cdui"
    cdui.write_text(
        "#!/bin/sh\n"
        f'cd "{BACKEND_DIR}"\n'
        f'export PYTHONPATH="{BACKEND_DIR}"\n'
        f'export CODEFYUI_USER_DATA_DIR="{remote_home}"\n'
        f'export CODEFYUI_CUSTOM_NODES_DIR="{nodes_dir}"\n'
        "export CODEFYUI_LANG=en\n"
        '[ "$1" = agent ] && shift\n'
        'if [ "$1" = start ]; then set -- "$@" --devices cpu; fi\n'
        f'exec "{sys.executable}" -m app.agent "$@"\n')
    cdui.chmod(0o755)
    yield log, remote_home, cdui
    # Whatever is left on the "remote host": stop it.
    for pid_file in (remote_home / ".codefyui-ssh").glob("*/pid"):
        try:
            os.kill(int(pid_file.read_text().strip()), signal.SIGTERM)
        except (OSError, ValueError):
            pass
    shutil.rmtree(remote_home, ignore_errors=True)


@pytest.mark.skipif(sys.platform == "win32",
                    reason="the fake ssh is a POSIX shell shim")
def test_an_ssh_worker_enrolls_runs_and_survives_disable_enable(
        fake_remote, live_server):  # noqa: F811
    log, remote_home, cdui = fake_remote
    client = _client(live_server)
    added = client.post("/api/workers/ssh", json={
        "target": "tester@fake-gpu-box", "remote_cdui": str(cdui),
        "name": NAME})
    assert added.status_code == 200, added.text
    try:
        _wait(lambda: _up(client, NAME), "the SSH worker never came up",
              client, NAME)
        entry = _ssh_entry(client, NAME)
        port = entry["remote_port"]
        worker = _worker(client, NAME)
        # It authenticated with the worker token it enrolled for, not the
        # session token its loopback peer address would also allow.
        assert worker["credential"] == "worker_token"
        assert worker["enrolled"] is True
        credential_file = remote_home / "agent" / f"cp-{_cp()}.invalid-{port}.token"
        stored = json.loads(credential_file.read_text())
        assert stored["worker_id"] == NAME
        # The one-time enrollment token was spent and then cleared away.
        assert client.get("/api/workers/enrollments").json()[
            "enrollments"] == []
        # Nothing secret on ssh's command line.
        for argv in map(json.loads, log.read_text().splitlines()):
            assert stored["worker_token"] not in " ".join(argv)
            assert "--token" not in " ".join(argv)
        # The forward's far end is a socket only this account can reach,
        # and through it there is no session bootstrap, whatever the Host.
        state = remote_home / ".codefyui-ssh" / f"{_cp()}-{port}"
        assert state.stat().st_mode & 0o777 == 0o700
        assert state.parent.stat().st_mode & 0o777 == 0o700
        through = httpx.Client(transport=httpx.HTTPTransport(
            uds=str(state / "sock")), base_url="http://127.0.0.1", timeout=10)
        with through:
            for host in ("127.0.0.1", f"127.0.0.1:{live_server}",
                         f"127.0.0.1:{port}"):
                answer = through.get("/api/auth/bootstrap",
                                     headers={"host": host})
                assert answer.status_code == 404, host
                assert session_token() not in answer.text

        run = client.post("/api/runs", json={
            "graph": _graph(), "options": {"target": {"worker": NAME}}}).json()
        record = _wait_terminal(client, run["run_id"])
        assert record["status"] == "succeeded", record
        assert record["worker_id"] == NAME

        # Disable: drained, the session ends, the remote agent stops.
        agent_pid = int((remote_home / ".codefyui-ssh" / f"{_cp()}-{port}" / "pid"
                         ).read_text())
        assert _pid_alive(agent_pid)
        disabled = client.post(f"/api/workers/ssh/{NAME}/disable").json()
        assert disabled["state"] == "stopped"
        _wait(lambda: (_worker(client, NAME) or {}).get("status")
              in ("disconnected", "offline"),
              "the worker stayed connected after disable", client, NAME)
        assert _worker(client, NAME)["draining"] is True
        deadline = time.monotonic() + 15
        while _pid_alive(agent_pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not _pid_alive(agent_pid), "disable left the remote agent"

        # Enable: back up on the same credential, not a re-key.
        client.post(f"/api/workers/ssh/{NAME}/enable")
        _wait(lambda: _up(client, NAME), "enable did not bring it back",
              client, NAME)
        assert _worker(client, NAME)["draining"] is False
        assert json.loads(credential_file.read_text()) == stored
        log_text = (remote_home / "agent.log").read_text()
        assert log_text.count("Enrolling") <= 1
        assert not any("reattached" in line for line in
                       _ssh_entry(client, NAME)["tail"])
    finally:
        removed = client.delete(f"/api/workers/{NAME}")
    assert removed.status_code == 200, removed.text
    assert removed.json()["revoked"] is True
    assert _ssh_entry(client, NAME) is None
    # Remove cleaned up on the "remote host" too (#829): the credential the
    # agent stored for this control plane and the slot's directory. That
    # runs in the background, after the answer.
    deadline = time.monotonic() + 60
    while ((credential_file.exists() or state.exists())
           and time.monotonic() < deadline):
        time.sleep(0.1)
    assert not credential_file.exists()
    assert not state.exists()


@pytest.mark.skipif(sys.platform == "win32",
                    reason="the fake ssh is a POSIX shell shim")
def test_a_dropped_ssh_session_keeps_the_agent_and_its_run(
        fake_remote, live_server):  # noqa: F811
    """The laptop-sleep case: the ssh session dies mid-run (the fake ssh
    is killed, its tunnel with it). The remote agent keeps running, its run
    keeps going and spools (#783); the launcher re-creates the tunnel on
    the same remote port, the remote script re-attaches instead of
    starting a second agent, the agent reconnects by itself, and the run
    ends ``succeeded`` with every metric point."""
    log, remote_home, cdui = fake_remote
    name = "ssh-drop"
    client = _client(live_server)
    added = client.post("/api/workers/ssh", json={
        "target": "tester@fake-gpu-box", "remote_cdui": str(cdui),
        "name": name})
    assert added.status_code == 200, added.text
    try:
        _wait(lambda: _up(client, name), "the SSH worker never came up",
              client, name)
        port = _ssh_entry(client, name)["remote_port"]
        state = remote_home / ".codefyui-ssh" / f"{_cp()}-{port}"
        agent_pid = int((state / "pid").read_text())
        graph = {"nodes": [{"id": "start", "type": "Start",
                            "data": {"params": {}}},
                           {"id": "tick", "type": "SshTicker",
                            "data": {"params": {}}}],
                 "edges": [{"id": "t", "source": "start", "target": "tick",
                            "sourceHandle": "trigger", "type": "trigger"}]}
        run_id = client.post("/api/runs", json={
            "graph": graph,
            "options": {"target": {"worker": name}}}).json()["run_id"]
        deadline = time.monotonic() + 60
        while not client.get(f"/api/runs/{run_id}/metrics").json().get(
                "metrics"):
            assert time.monotonic() < deadline, "no metric ever came"
            time.sleep(0.05)

        # The session drops: ssh dies, the tunnel with it.
        ssh_pid = _ssh_entry(client, name)["pid"]
        os.kill(ssh_pid, signal.SIGKILL)

        record: dict[str, Any] = {}
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            record = client.get(f"/api/runs/{run_id}").json()
            if record["status"] in ("succeeded", "failed", "cancelled",
                                    "interrupted"):
                break
            time.sleep(0.1)
        assert record["status"] == "succeeded", record
        points = client.get(f"/api/runs/{run_id}/metrics").json()["metrics"]
        assert sorted(p["step"] for p in points
                      if p["name"] == "tick") == list(range(60))
        # The same agent all along, re-attached by a second session on the
        # same remote port.
        assert _pid_alive(agent_pid)
        assert int((state / "pid").read_text()) == agent_pid
        entry = _ssh_entry(client, name)
        assert entry["remote_port"] == port
        assert entry["reattached"] is True
        assert entry["remote_pid"] == agent_pid
        argvs = [json.loads(line) for line in log.read_text().splitlines()]
        assert len([a for a in argvs if "-R" in a]) >= 2
        assert (remote_home / ".codefyui-ssh" / f"{_cp()}-{port}" / "log"
                ).read_text().count("Enrolling") <= 1
        events = client.get(f"/api/runs/{run_id}/events").json()["events"]
        kinds = [e["payload"]["kind"] for e in events
                 if e["type"] == "run_warning"]
        assert set(kinds) <= {"worker_disconnected", "worker_reconnected"}
        cursors = [e["cursor"] for e in events]
        assert cursors == list(range(cursors[0], cursors[0] + len(cursors)))
        assert [e["type"] for e in events].count("execution_complete") == 1
        assert len([p for p in points if p["name"] == "tick"]) == 60

        # Disable stops that agent through the new session.
        client.post(f"/api/workers/ssh/{name}/disable")
        deadline = time.monotonic() + 15
        while _pid_alive(agent_pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not _pid_alive(agent_pid)
        # Both sessions' remote scripts are gone with their ``tail -f`` of
        # the agent's log, the dropped one included (#829).
        deadline = time.monotonic() + 15
        while _tails(remote_home) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert _tails(remote_home) == []
    finally:
        client.delete(f"/api/workers/{name}")


def _tails(remote_home: Path) -> list[str]:
    """``tail`` processes still following a log under *remote_home*."""
    import subprocess

    out = subprocess.run(["ps", "-A", "-o", "pid=,command="],
                         capture_output=True, text=True).stdout
    return [line for line in out.splitlines()
            if line.split(None, 1)[1:2] and str(remote_home) in line
            and line.split(None, 1)[1].startswith("tail ")]


@pytest.mark.skipif(not os.environ.get("CODEFYUI_SSH_E2E_TARGET"),
                    reason="set CODEFYUI_SSH_E2E_TARGET=user@host to run "
                           "against a real ssh server")
def test_a_real_ssh_worker(live_server):  # noqa: F811
    name = "ssh-e2e-real"
    client = _client(live_server)
    body: dict[str, Any] = {"target": os.environ["CODEFYUI_SSH_E2E_TARGET"],
                            "name": name}
    if os.environ.get("CODEFYUI_SSH_E2E_REMOTE_CDUI"):
        body["remote_cdui"] = os.environ["CODEFYUI_SSH_E2E_REMOTE_CDUI"]
    if os.environ.get("CODEFYUI_SSH_E2E_PORT"):
        body["port"] = int(os.environ["CODEFYUI_SSH_E2E_PORT"])
    if os.environ.get("CODEFYUI_SSH_E2E_IDENTITY"):
        body["identity"] = os.environ["CODEFYUI_SSH_E2E_IDENTITY"]
    added = client.post("/api/workers/ssh", json=body)
    assert added.status_code == 200, added.text
    try:
        _wait(lambda: _up(client, name), "the SSH worker never came up",
              client, name, timeout=180)
        worker = _worker(client, name)
        assert worker["credential"] == "worker_token"
        assert worker["same_host"] is False or os.environ.get(
            "CODEFYUI_SSH_E2E_SAME_HOST")
        run = client.post("/api/runs", json={
            "graph": _graph(), "options": {"target": {"worker": name}}}).json()
        record = _wait_terminal(client, run["run_id"], timeout=300)
        assert record["status"] == "succeeded", record
        assert record["worker_id"] == name
    finally:
        client.delete(f"/api/workers/{name}")


@pytest.mark.skipif(sys.platform == "win32",
                    reason="the fake ssh is a POSIX shell shim")
def test_a_transfer_run_goes_through_the_ssh_tunnel(
        fake_remote, live_server, tmp_path, monkeypatch):  # noqa: F811
    """#784 + #785: an SSH worker with no shared folder runs a graph that
    reads an uploaded CSV and writes a checkpoint. The agent fetches the
    input and uploads the checkpoint through the reverse tunnel (the SSH
    worker's own listener) with its worker token; the listener lets
    exactly those routes through."""
    from app.config import settings

    from .test_worker_e2e import _csv_and_checkpoint_graph, _transfer_events

    control = tmp_path / "control"
    (control / "files").mkdir(parents=True)
    (control / "files" / "train.csv").write_bytes(
        b"a,b,label\n1,2,0\n3,4,1\n5,6,0\n")
    monkeypatch.setattr(settings, "PROJECT_DIR", None)
    monkeypatch.setattr(settings, "MOUNT_LABEL", "data")
    for attr, sub in (("DATA_FILES_DIR", "files"), ("IMAGES_DIR", "images"),
                      ("MODELS_DIR", "models"), ("MEDIA_DIR", "media")):
        monkeypatch.setattr(settings, attr, control / sub)
    import app.api.ws_worker as ws_worker

    # Another machine, as a real GPU box is: no directory is lent to it.
    monkeypatch.setattr(ws_worker, "machine_identity",
                        lambda: "not-this-agents-machine")
    name = "ssh-transfer"
    _, remote_home, cdui = fake_remote
    client = _client(live_server)
    added = client.post("/api/workers/ssh", json={
        "target": "tester@fake-gpu-box", "remote_cdui": str(cdui),
        "name": name})
    assert added.status_code == 200, added.text
    try:
        _wait(lambda: _up(client, name), "the SSH worker never came up",
              client, name)
        assert _worker(client, name)["mounts"] == {}
        run = client.post("/api/runs", json={
            "graph": _csv_and_checkpoint_graph(),
            "options": {"target": {"worker": name}}})
        assert run.status_code in (200, 201, 202), run.text
        run_id = run.json()["run_id"]
        record = _wait_terminal(client, run_id, timeout=180)
        assert record["status"] == "succeeded", record
        progress = _transfer_events(client, run_id)
        assert progress and "train.csv" in progress[-1] \
            and "fetched" in progress[-1], progress
        artifacts = client.get(f"/api/runs/{run_id}/artifacts").json()[
            "artifacts"]
        [checkpoint] = [a for a in artifacts if a["kind"] == "checkpoint"]
        assert checkpoint["backend"] == "transfer"
        assert Path(checkpoint["path"]).is_file()
        assert Path(checkpoint["path"]).is_relative_to(control.resolve())
        assert list((remote_home / "agent-cache").glob("[0-9a-f]" * 64))
    finally:
        client.delete(f"/api/workers/{name}")
