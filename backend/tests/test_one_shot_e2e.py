"""End to end: a real ``cdui agent start --one-shot`` beside a real server (#790).

The way a batch job starts it: a run waits in the one-shot pool, the agent
comes up, takes that run, and exits with the run's outcome as its exit
code; the Workers list then shows it ``offline`` with the run it took.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.core.auth import TOKEN_HEADER, session_token
from app.main import app

from . import test_worker_e2e
from .test_worker_e2e import (
    BACKEND_DIR,
    CONNECT_TIMEOUT_S,
    _graph,
    _wait_terminal,
)

#: A real server on a free loopback port (a fixture).
live_server = test_worker_e2e.live_server


def _one_shot_agent(base: str, user_data: Path, name: str,
                    *extra: str) -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": str(BACKEND_DIR),
           "CODEFYUI_USER_DATA_DIR": str(user_data),
           "CODEFYUI_AGENT_TOKEN": session_token()}
    return subprocess.Popen(
        [sys.executable, "-m", "app.agent", "start", "--one-shot",
         "--control", base, "--name", name, "--devices", "cpu", *extra],
        cwd=BACKEND_DIR, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _exit(agent: subprocess.Popen, timeout: float) -> tuple[int, str]:
    try:
        output, _ = agent.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        agent.kill()
        output, _ = agent.communicate()
        raise AssertionError(f"the one-shot agent did not exit:\n{output}")
    return agent.returncode, output


def _worker(client: httpx.Client, name: str) -> dict[str, Any] | None:
    for worker in client.get("/api/workers").json()["workers"]:
        if worker["name"] == name:
            return worker
    return None


def test_a_one_shot_agent_takes_the_waiting_run_and_exits_0(live_server,
                                                           tmp_path):
    base = f"http://127.0.0.1:{live_server}"
    client = httpx.Client(base_url=base, timeout=30,
                          headers={TOKEN_HEADER: session_token()})
    first = client.post("/api/runs", json={
        "graph": _graph(),
        "options": {"target": {"pool": "one-shot"}}}).json()["run_id"]
    second = client.post("/api/runs", json={
        "graph": _graph(),
        "options": {"target": {"pool": "one-shot"}}}).json()["run_id"]
    row = client.get(f"/api/runs/{first}").json()
    assert row["status"] == "queued" and "one-shot" in row["queue_reason"]

    # The agent's empty data dir lacks plugin packs this machine may have
    # installed; a worker that lacks node types gets no pool run (as for
    # ``any``), which test_one_shot_workers covers. Not compared here.
    app.state.worker_registry._node_catalog = None
    agent = _one_shot_agent(base, tmp_path, "e2e-shot", "--run-id", second)
    code, output = _exit(agent, CONNECT_TIMEOUT_S + 90)
    assert code == 0, output
    record = _wait_terminal(client, second)
    assert (record["status"], record["worker_id"]) == ("succeeded",
                                                       "e2e-shot")
    assert client.get(f"/api/runs/{first}").json()["status"] == "queued"
    deadline = time.monotonic() + 15
    while (_worker(client, "e2e-shot") or {}).get("status") != "offline":
        assert time.monotonic() < deadline, _worker(client, "e2e-shot")
        time.sleep(0.1)
    worker = _worker(client, "e2e-shot")
    assert worker["one_shot"] == {"run_id": second, "took": second}
    assert worker["running"] == []
    client.post(f"/api/runs/{first}/cancel")
    client.close()


@pytest.mark.skipif(sys.platform == "win32",
                    reason="no SIGTERM on Windows: terminate() is "
                           "TerminateProcess, which leaves no exit code to "
                           "check")
def test_a_one_shot_agent_stopped_before_a_run_exits_3(live_server,
                                                      tmp_path):
    base = f"http://127.0.0.1:{live_server}"
    client = httpx.Client(base_url=base, timeout=30,
                          headers={TOKEN_HEADER: session_token()})
    agent = _one_shot_agent(base, tmp_path, "e2e-idle")
    deadline = time.monotonic() + CONNECT_TIMEOUT_S
    while (_worker(client, "e2e-idle") or {}).get("status") not in (
            "online", "degraded"):
        if time.monotonic() > deadline or agent.poll() is not None:
            agent.kill()
            raise AssertionError(agent.communicate()[0])
        time.sleep(0.1)
    # ``any`` does not give it a run: this one runs here.
    run_id = client.post("/api/runs", json={
        "graph": _graph(),
        "options": {"target": {"worker": "any"}}}).json()["run_id"]
    assert _wait_terminal(client, run_id)["worker_id"] == "local"
    agent.send_signal(signal.SIGTERM)
    code, output = _exit(agent, 30)
    assert code == 3, output
    deadline = time.monotonic() + 15
    while (_worker(client, "e2e-idle") or {}).get("status") != "offline":
        assert time.monotonic() < deadline, _worker(client, "e2e-idle")
        time.sleep(0.1)
    assert _worker(client, "e2e-idle")["one_shot"]["took"] is None
    client.close()
