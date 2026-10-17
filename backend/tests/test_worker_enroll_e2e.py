"""End to end: a real agent enrolls with a real server (#780).

The server is ``test_worker_e2e``'s (uvicorn on a free loopback port, its
real lifespan); the agent is ``python -m app.agent start --foreground`` in
a subprocess whose user-data dir is NOT the server's, so it cannot read the
session token file, and it is given no session token. It reaches the server
over 127.0.0.1, so the session-token path would be open to it: the test
asserts that it authenticates with the worker token it enrolled for.

Then: a restart reuses the stored credential (the single-use enrollment
token is spent, so a second enrollment would fail), a run executes on it,
and removing the worker disconnects it and refuses its reconnects.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from app.core.auth import TOKEN_HEADER, session_token

from .test_worker_e2e import _graph, _wait_terminal, live_server  # noqa: F401

BACKEND_DIR = Path(__file__).resolve().parents[1]
NAME = "e2e-enrolled"


def _start_agent(base: str, home: Path, *extra: str) -> subprocess.Popen:
    env = {key: value for key, value in os.environ.items()
           if key != "CODEFYUI_AGENT_TOKEN"}
    env.update({"PYTHONPATH": str(BACKEND_DIR),
                "CODEFYUI_USER_DATA_DIR": str(home), "CODEFYUI_LANG": "en"})
    return subprocess.Popen(
        [sys.executable, "-m", "app.agent", "start", "--foreground",
         "--control", base, "--devices", "cpu", *extra],
        cwd=BACKEND_DIR, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _stop(agent: subprocess.Popen) -> str:
    agent.terminate()
    try:
        output, _ = agent.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        agent.kill()
        output, _ = agent.communicate()
    return output


def _worker(client: httpx.Client) -> dict[str, Any] | None:
    for worker in client.get("/api/workers").json()["workers"]:
        if worker["name"] == NAME:
            return worker
    return None


#: A connected worker: ``degraded`` when its node catalog differs from the
#: test server's (#781), which depends on what earlier tests registered.
CONNECTED = ("online", "degraded")


def _wait(predicate: Any, agent: subprocess.Popen | None, what: str,
          timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(what)
        if agent is not None and agent.poll() is not None:
            raise AssertionError(f"{what}; the agent exited:\n"
                                 f"{agent.communicate()[0]}")
        time.sleep(0.1)


def test_an_agent_enrolls_reconnects_with_its_credential_and_is_revoked(
        live_server, tmp_path):  # noqa: F811 -- the imported fixture
    base = f"http://127.0.0.1:{live_server}"
    client = httpx.Client(base_url=base, timeout=30,
                          headers={TOKEN_HEADER: session_token()})
    home = tmp_path / "agent-home"
    created = client.post("/api/workers/enrollments", json={}).json()
    assert created["command"] == (
        f"cdui agent start --control {base} --token {created['token']}")

    agent = _start_agent(base, home, "--token", created["token"],
                         "--name", NAME)
    try:
        _wait(lambda: (_worker(client) or {}).get("status") in CONNECTED,
              agent, "the enrolled agent never connected")
        worker = _worker(client)
        assert worker["enrolled"] is True
        assert worker["credential"] == "worker_token"
        stored = json.loads((home / "agent" /
                             f"127.0.0.1-{live_server}.token").read_text())
        assert stored["worker_id"] == NAME
        assert stored["worker_token"] != session_token()
        if sys.platform != "win32":
            assert os.stat(home / "agent" /
                           f"127.0.0.1-{live_server}.token"
                           ).st_mode & 0o777 == 0o600
        assert not (home / "session.token").exists()
        [enrollment] = client.get(
            "/api/workers/enrollments").json()["enrollments"]
        assert enrollment["state"] == "used"
    finally:
        first_output = _stop(agent)
    assert "Enrolled" in first_output, first_output
    _wait(lambda: (_worker(client) or {}).get("status") == "disconnected",
          None, "the server never saw the agent go")

    # A restart: no --token at all. The stored credential is the only way in.
    agent = _start_agent(base, home)
    try:
        _wait(lambda: (_worker(client) or {}).get("status") in CONNECTED,
              agent, "the restarted agent never reconnected")
        assert _worker(client)["credential"] == "worker_token"
        run = client.post("/api/runs", json={
            "graph": _graph(),
            "options": {"target": {"worker": NAME}}}).json()
        record = _wait_terminal(client, run["run_id"])
        assert record["status"] == "succeeded", record
        assert record["worker_id"] == NAME

        removed = client.delete(f"/api/workers/{NAME}")
        assert removed.json() == {"id": NAME, "revoked": True,
                                  "failed_runs": []}
        assert _worker(client) is None
        # It dials again (backoff starts at 1 s) and is refused each time.
        time.sleep(4)
        assert _worker(client) is None
    finally:
        second_output = _stop(agent)
    assert "Using the stored worker credential" in second_output
    assert "Enrolling" not in second_output
    # A revoked credential is a 401 (other handshake refusals are 403).
    assert "refused the token (HTTP 401)" in second_output, second_output
    assert "cdui agent forget" in second_output
    client.close()
