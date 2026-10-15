"""End to end: a real ``cdui agent`` beside a real server (#771 + #772).

The app runs under uvicorn on a free loopback port in a thread (its real
lifespan: instance lock, node discovery, the worker registry); the agent
runs as ``python -m app.agent start --foreground`` in a subprocess. One
tiny seeded CPU graph runs here and then on the agent; the two event logs
must agree on ``(type, node_id, status)``.
"""

from __future__ import annotations

import asyncio
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


#: A node the agent's run subprocess discovers from
#: ``CODEFYUI_CUSTOM_NODES_DIR``: three seconds of one metric point per
#: tenth of a second, long enough to drop the connection in the middle.
TICKER_NODE = '''
import time

from app.core.node_base import BaseNode, DataType, PortDefinition


class E2ETicker(BaseNode):
    NODE_NAME = "E2ETicker"
    CATEGORY = "Test"

    @classmethod
    def define_inputs(cls):
        return []

    @classmethod
    def define_outputs(cls):
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs, params, *, context=None):
        for step in range(30):
            if context is not None:
                context.log_metric("tick", float(step), step=step)
            time.sleep(0.1)
        return {"value": 30}
'''


def test_a_dropped_connection_does_not_end_a_real_run(live_server, tmp_path):
    """#783 acceptance with a real agent and a real run subprocess: the
    server closes the worker's socket mid-run; the subprocess keeps
    running, the agent reconnects and replays its spool, and the run ends
    ``succeeded`` with every metric point, after a ``disconnected`` spell."""
    base = f"http://127.0.0.1:{live_server}"
    client = httpx.Client(base_url=base, timeout=30,
                          headers={TOKEN_HEADER: session_token()})
    nodes_dir = tmp_path / "nodes"
    nodes_dir.mkdir()
    (nodes_dir / "e2e_ticker.py").write_text(TICKER_NODE)
    env = {**os.environ, "PYTHONPATH": str(BACKEND_DIR),
           "CODEFYUI_USER_DATA_DIR": str(tmp_path / "data"),
           "CODEFYUI_CUSTOM_NODES_DIR": str(nodes_dir),
           "CODEFYUI_AGENT_TOKEN": session_token()}
    agent = subprocess.Popen(
        [sys.executable, "-m", "app.agent", "start", "--foreground",
         "--control", base, "--name", "drop-box", "--devices", "cpu"],
        cwd=BACKEND_DIR, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = ""
    try:
        deadline = time.monotonic() + 30
        while not any(w["name"] == "drop-box" and w["status"] == "online"
                      for w in client.get("/api/workers").json()["workers"]):
            if time.monotonic() > deadline or agent.poll() is not None:
                raise AssertionError("the agent never connected")
            time.sleep(0.1)
        graph = {"nodes": [{"id": "start", "type": "Start",
                            "data": {"params": {}}},
                           {"id": "tick", "type": "E2ETicker",
                            "data": {"params": {}}}],
                 "edges": [{"id": "t", "source": "start", "target": "tick",
                            "sourceHandle": "trigger", "type": "trigger"}]}
        run_id = client.post("/api/runs", json={
            "graph": graph,
            "options": {"target": {"worker": "drop-box"}}}).json()["run_id"]
        deadline = time.monotonic() + 60
        while not client.get(f"/api/runs/{run_id}/metrics").json().get(
                "metrics"):
            if time.monotonic() > deadline:
                raise AssertionError("the run never produced a metric")
            time.sleep(0.05)

        # The network drops: the server closes the worker's socket.
        registry = app.state.worker_registry
        loop = registry._watchdog.get_loop()
        worker = registry.get("drop-box")
        asyncio.run_coroutine_threadsafe(worker.close(), loop).result(10)

        seen_row = seen_status = False
        record: dict[str, Any] = {}
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            record = client.get(f"/api/runs/{run_id}").json()
            # The row follows the executor; ``worker_status`` follows the
            # registry, which learns of the reconnect first. Each is seen
            # disconnected at some point, not necessarily in one read.
            seen_row = seen_row or record["status"] == "disconnected"
            seen_status = seen_status or (
                record.get("worker_status") or {}).get("state") == (
                    "disconnected")
            if record["status"] in TERMINAL:
                break
            time.sleep(0.05)
        assert seen_row and seen_status
        assert record["status"] == "succeeded", record
        assert agent.poll() is None  # the agent never restarted
        points = client.get(f"/api/runs/{run_id}/metrics").json()["metrics"]
        assert sorted(p["step"] for p in points
                      if p["name"] == "tick") == list(range(30))
        events = client.get(f"/api/runs/{run_id}/events").json()["events"]
        kinds = [e["payload"]["kind"] for e in events
                 if e["type"] == "run_warning"]
        assert kinds == ["worker_disconnected", "worker_reconnected"]
    finally:
        agent.terminate()
        try:
            output, _ = agent.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            agent.kill()
            output, _ = agent.communicate()
        client.close()
    assert "replayed" in output, output
# ── a shared data mount (#782) ───────────────────────────────────────────


def _csv_and_checkpoint_graph() -> dict[str, Any]:
    """Reads an uploaded CSV and trains one epoch with ``checkpoint_every``:
    a run that needs the ``data`` mount on both counts."""
    def node(node_id: str, node_type: str, **params: Any) -> dict[str, Any]:
        return {"id": node_id, "type": node_type, "data": {"params": params}}

    def trigger(target: str) -> dict[str, Any]:
        return {"id": f"t-{target}", "source": "start", "target": target,
                "sourceHandle": "trigger", "type": "trigger"}

    def wire(source: str, out: str, target: str, port: str) -> dict[str, Any]:
        return {"id": f"e-{source}-{target}-{port}", "source": source,
                "sourceHandle": out, "target": target, "targetHandle": port}

    return {
        "nodes": [
            node("start", "Start"),
            node("csv", "CSVReader", path="train.csv",
                 target_column="label"),
            node("show", "Print", label="csv"),
            node("data", "SyntheticShapes", n_samples=16, image_size=28),
            node("loader", "DataLoader", batch_size=8),
            node("model", "SequentialModel"),
            node("opt", "Optimizer", type="Adam"),
            node("loss", "Loss"),
            node("train", "TrainingLoop", epochs=1, checkpoint_every=1,
                 device="cpu"),
        ],
        "edges": [
            trigger("csv"), trigger("data"), trigger("model"),
            trigger("loss"),
            wire("csv", "tensor", "show", "value"),
            wire("data", "dataset", "loader", "dataset"),
            wire("model", "model", "opt", "model"),
            wire("model", "model", "train", "model"),
            wire("opt", "optimizer", "train", "optimizer"),
            wire("loader", "dataloader", "train", "dataloader"),
            wire("loss", "loss_fn", "train", "loss_fn"),
        ],
        "settings": {"seed": 0},
    }


def test_a_run_on_a_shared_mount_reads_the_upload_and_lists_the_checkpoint(
        live_server, tmp_path, monkeypatch):
    """Acceptance (#782): this server's data directory and the agent's
    ``--mount data=`` are the same directory (here reached through a
    symlink, as an NFS mount point would be: another path, the same
    files). The run reads the CSV uploaded here and its checkpoint is
    listed at THIS server's path, stored relative to the mount."""
    shared = tmp_path / "shared"
    (shared / "files").mkdir(parents=True)
    (shared / "files" / "train.csv").write_text(
        "a,b,label\n1,2,0\n3,4,1\n5,6,0\n", encoding="utf-8")
    monkeypatch.setattr(settings, "PROJECT_DIR", None)
    monkeypatch.setattr(settings, "MOUNT_LABEL", "data")
    for name, sub in (("DATA_FILES_DIR", "files"), ("IMAGES_DIR", "images"),
                      ("MODELS_DIR", "models"), ("MEDIA_DIR", "media")):
        monkeypatch.setattr(settings, name, shared / sub)
    mount_point = tmp_path / "nas"
    mount_point.symlink_to(shared, target_is_directory=True)
    # The agent runs here, but is taken for another machine: a same-host
    # worker's files are local rows, and this test is about mount rows.
    import app.api.ws_worker as ws_worker

    monkeypatch.setattr(ws_worker, "machine_identity",
                        lambda: "not-this-agents-machine")

    base = f"http://127.0.0.1:{live_server}"
    client = httpx.Client(base_url=base, timeout=30,
                          headers={TOKEN_HEADER: session_token()})
    env = {**os.environ, "PYTHONPATH": str(BACKEND_DIR),
           "CODEFYUI_USER_DATA_DIR": str(tmp_path / "agent-data"),
           "CODEFYUI_AGENT_TOKEN": session_token()}
    agent = subprocess.Popen(
        [sys.executable, "-m", "app.agent", "start", "--foreground",
         "--control", base, "--name", "nas-box", "--devices", "cpu",
         "--mount", f"data={mount_point}"],
        cwd=BACKEND_DIR, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 30
        while True:
            workers = {w["name"]: w for w in
                       client.get("/api/workers").json()["workers"]}
            if workers.get("nas-box", {}).get("status") == "online":
                break
            if time.monotonic() > deadline or agent.poll() is not None:
                raise AssertionError("the agent never connected")
            time.sleep(0.1)
        # GET /api/workers shows the label.
        assert workers["nas-box"]["mounts"] == {"data": str(mount_point)}

        submitted = client.post("/api/runs", json={
            "graph": _csv_and_checkpoint_graph(),
            "options": {"target": {"worker": "nas-box"}}})
        assert submitted.status_code in (200, 201, 202), submitted.text
        run_id = submitted.json()["run_id"]
        record = _wait_terminal(client, run_id, timeout=180)
        assert record["status"] == "succeeded", record
        assert record["worker_id"] == "nas-box"

        artifacts = client.get(f"/api/runs/{run_id}/artifacts").json()[
            "artifacts"]
        [checkpoint] = [a for a in artifacts if a["kind"] == "checkpoint"]
        assert checkpoint["backend"] == "mount"
        assert checkpoint["mount_label"] == "data"
        assert checkpoint["worker_id"] == "nas-box"
        relative = checkpoint["relative_path"]
        assert relative.startswith("models/periodic/")
        assert checkpoint["path"] == str(shared.resolve() / relative)
        assert Path(checkpoint["path"]).is_file()
    finally:
        agent.terminate()
        try:
            agent.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            agent.kill()
            agent.communicate()
        client.close()
