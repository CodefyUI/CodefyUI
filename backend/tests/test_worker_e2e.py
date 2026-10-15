"""End to end: a real ``cdui agent`` beside a real server (#771 + #772).

The app runs under uvicorn on a free loopback port in a thread (its real
lifespan: instance lock, node discovery, the worker registry); the agent
runs as ``python -m app.agent start --foreground`` in a subprocess. One
tiny seeded CPU graph runs here and then on the agent; the two event logs
must agree on ``(type, node_id, status)``.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn

from app.config import settings
from app.core.auth import TOKEN_HEADER, init_allowed_hosts, session_token
from app.main import app

BACKEND_DIR = Path(__file__).resolve().parents[1]
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _graph() -> dict[str, Any]:
    def node(node_id: str, node_type: str, **params: Any) -> dict[str, Any]:
        return {"id": node_id, "type": node_type, "data": {"params": params}}

    return {
        "nodes": [node("start", "Start"),
                  node("x", "TensorCreate", shape="2,3", fill="ones",
                       value=0, requires_grad=False),
                  node("scale", "ScalarMultiply", scalar=2.5),
                  node("show", "Print", label="tensor")],
        "edges": [
            {"id": "t", "source": "start", "target": "x",
             "sourceHandle": "trigger", "type": "trigger"},
            {"id": "a", "source": "x", "sourceHandle": "tensor",
             "target": "scale", "targetHandle": "tensor"},
            {"id": "b", "source": "scale", "sourceHandle": "tensor",
             "target": "show", "targetHandle": "value"}],
        "settings": {"seed": 0},
    }


@pytest.fixture
def live_server(monkeypatch):
    port = _free_port()
    monkeypatch.setattr(settings, "PORT", port)
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning",
        lifespan="on"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 60
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("the test server did not start")
        time.sleep(0.05)
    try:
        yield port
    finally:
        server.should_exit = True
        thread.join(60)
        init_allowed_hosts(settings.HOST, 8000)


def _wait_terminal(client: httpx.Client, run_id: str,
                   timeout: float = 90) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = client.get(f"/api/runs/{run_id}").json()
        if record["status"] in TERMINAL:
            return record
        time.sleep(0.1)
    raise AssertionError(f"run {run_id} did not finish")


def _steps(client: httpx.Client, run_id: str) -> list[tuple]:
    events = client.get(f"/api/runs/{run_id}/events").json()["events"]
    return [(e["type"], (e["payload"] or {}).get("node_id"),
             (e["payload"] or {}).get("status")) for e in events
            if e["type"] != "metric"]


def test_a_graph_runs_on_a_real_agent_like_it_runs_here(live_server,
                                                       tmp_path):
    base = f"http://127.0.0.1:{live_server}"
    client = httpx.Client(base_url=base, timeout=30,
                          headers={TOKEN_HEADER: session_token()})
    local = client.post("/api/runs", json={"graph": _graph()}).json()
    local_record = _wait_terminal(client, local["run_id"])
    assert local_record["status"] == "succeeded"

    env = {**os.environ, "PYTHONPATH": str(BACKEND_DIR),
           "CODEFYUI_USER_DATA_DIR": str(tmp_path),
           "CODEFYUI_AGENT_TOKEN": session_token()}
    agent = subprocess.Popen(
        [sys.executable, "-m", "app.agent", "start", "--foreground",
         "--control", base, "--name", "e2e-box", "--devices", "cpu"],
        cwd=BACKEND_DIR, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 30
        while True:
            workers = client.get("/api/workers").json()["workers"]
            if any(w["name"] == "e2e-box" and w["status"] == "online"
                   for w in workers):
                break
            if time.monotonic() > deadline or agent.poll() is not None:
                raise AssertionError("the agent never connected")
            time.sleep(0.1)

        remote = client.post("/api/runs", json={
            "graph": _graph(),
            "options": {"target": {"worker": "e2e-box"}}}).json()
        remote_record = _wait_terminal(client, remote["run_id"])
        assert remote_record["status"] == "succeeded", remote_record
        assert remote_record["worker_id"] == "e2e-box"
        assert remote_record["queue_key"] == "e2e-box/cpu"
        assert _steps(client, remote["run_id"]) == _steps(
            client, local["run_id"])
    finally:
        agent.terminate()
        try:
            output, _ = agent.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            agent.kill()
            output, _ = agent.communicate()
    # The agent's own exit is seen by the server: the socket closed.
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        statuses = {w["name"]: w["status"]
                    for w in client.get("/api/workers").json()["workers"]}
        if statuses.get("e2e-box") == "disconnected":
            break
        time.sleep(0.1)
    assert statuses.get("e2e-box") == "disconnected", output
    client.close()
