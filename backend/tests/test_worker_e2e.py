"""End to end: a real ``cdui agent`` beside a real server (#771 + #772).

The app runs under uvicorn on a free loopback port in a thread (its real
lifespan: instance lock, node discovery, the worker registry); the agent
runs as ``python -m app.agent start --foreground`` in a subprocess. One
tiny seeded CPU graph runs here and then on the agent; the two event logs
must agree on ``(type, node_id, status)``. A second agent without the
``foundations`` plugin pack is ``degraded`` and refused a graph using it
(#781).
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



#: How long an agent has to connect: it first runs its facts helper
#: (#781), which imports torch and discovers every node.
CONNECT_TIMEOUT_S = 120


def _start_agent(base: str, user_data: Path, name: str) -> subprocess.Popen:
    """``cdui agent start --foreground`` with its own (empty) user data
    directory: no plugin pack is installed for it."""
    env = {**os.environ, "PYTHONPATH": str(BACKEND_DIR),
           "CODEFYUI_USER_DATA_DIR": str(user_data),
           "CODEFYUI_AGENT_TOKEN": session_token()}
    return subprocess.Popen(
        [sys.executable, "-m", "app.agent", "start", "--foreground",
         "--control", base, "--name", name, "--devices", "cpu"],
        cwd=BACKEND_DIR, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _wait_connected(client: httpx.Client, agent: subprocess.Popen,
                    name: str) -> dict[str, Any]:
    deadline = time.monotonic() + CONNECT_TIMEOUT_S
    while True:
        workers = client.get("/api/workers").json()["workers"]
        for worker in workers:
            if worker["name"] == name and worker["status"] in (
                    "online", "degraded"):
                return worker
        if time.monotonic() > deadline or agent.poll() is not None:
            raise AssertionError(f"the agent never connected: {workers}")
        time.sleep(0.1)


def _stop_agent(client: httpx.Client, agent: subprocess.Popen,
                name: str) -> None:
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
        if statuses.get(name) == "disconnected":
            break
        time.sleep(0.1)
    assert statuses.get(name) == "disconnected", output


def test_a_graph_runs_on_a_real_agent_like_it_runs_here(live_server,
                                                       tmp_path):
    base = f"http://127.0.0.1:{live_server}"
    client = httpx.Client(base_url=base, timeout=30,
                          headers={TOKEN_HEADER: session_token()})
    local = client.post("/api/runs", json={"graph": _graph()}).json()
    local_record = _wait_terminal(client, local["run_id"])
    assert local_record["status"] == "succeeded"

    agent = _start_agent(base, tmp_path, "e2e-box")
    try:
        _wait_connected(client, agent, "e2e-box")
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
        _stop_agent(client, agent, "e2e-box")
    client.close()


def _ensure_knn_registered() -> None:
    """Make this server know ``foundations:Edu-KNN``.

    The suite registers the in-repo packs, but an earlier test may have
    rediscovered from an empty lockfile and purged the plugin namespace, so
    the node module is loaded from its file and registered directly.
    """
    import importlib.util

    from app.core.node_base import BaseNode
    from app.core.node_registry import registry

    if registry.get("foundations:Edu-KNN") is not None:
        return
    path = (BACKEND_DIR.parent / "plugins" / "foundations" / "nodes"
            / "edu_knn_node.py")
    spec = importlib.util.spec_from_file_location("_e2e_edu_knn_node", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    [node] = [obj for obj in vars(module).values()
              if isinstance(obj, type) and issubclass(obj, BaseNode)
              and getattr(obj, "NODE_NAME", "") == "Edu-KNN"]
    registry.register(node, plugin_id="foundations")


def _knn_graph() -> dict[str, Any]:
    """A graph using a node of the ``foundations`` plugin pack."""
    return {
        "nodes": [{"id": "start", "type": "Start", "data": {"params": {}}},
                  {"id": "knn", "type": "foundations:Edu-KNN",
                   "data": {"params": {}}}],
        "edges": [{"id": "t", "source": "start", "target": "knn",
                   "sourceHandle": "trigger", "type": "trigger"}],
    }


def test_a_worker_without_a_plugin_is_degraded_and_refused_its_nodes(
        live_server, tmp_path):
    """#781 end to end: this server has the ``foundations`` pack (the test
    suite registers it); the agent's user data has no plugin installed, so
    its node catalog lacks the pack's types. It connects ``degraded`` with
    those types listed, a graph using one of them is refused naming it, a
    plain graph runs there, and that run records the WORKER's fingerprint.
    """
    base = f"http://127.0.0.1:{live_server}"
    client = httpx.Client(base_url=base, timeout=30,
                          headers={TOKEN_HEADER: session_token()})
    _ensure_knn_registered()
    assert "foundations:Edu-KNN" in {
        n["node_name"] for n in client.get("/api/nodes").json()}
    agent = _start_agent(base, tmp_path, "bare-box")
    try:
        worker = _wait_connected(client, agent, "bare-box")
        assert worker["status"] == "degraded"
        assert "node_catalog" in worker["degraded_reasons"]
        assert worker["node_catalog"]["status"] == "mismatch"
        assert "foundations:Edu-KNN" in worker["node_catalog"]["missing"]
        digest = worker["fingerprint_digest"]
        assert digest and len(digest) == 64
        # The digest resolves to the worker's document already.
        recorded = client.get(f"/api/system/fingerprints/{digest}")
        assert recorded.status_code == 200
        assert recorded.json()["fingerprint"]["codefyui"]["version"] \
            == worker["version"]

        refused = client.post("/api/runs", json={
            "graph": _knn_graph(),
            "options": {"target": {"worker": "bare-box"}}})
        assert refused.status_code == 400, refused.text
        assert "foundations:Edu-KNN" in refused.json()["detail"]

        plain = client.post("/api/runs", json={
            "graph": _graph(),
            "options": {"target": {"worker": "bare-box"}}}).json()
        record = _wait_terminal(client, plain["run_id"])
        assert record["status"] == "succeeded", record
        assert record["fingerprint_digest"] == digest
        assert record["execution_facts"]["worker_id"] == "bare-box"
    finally:
        _stop_agent(client, agent, "bare-box")
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
        # Degraded: its custom node directory has a node this server lacks
        # (#781); a run naming it may use that node, which it has.
        _wait_connected(client, agent, "drop-box")
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
        # Online or degraded (#781): its catalog depends on what the test
        # server has registered.
        workers = {"nas-box": _wait_connected(client, agent, "nas-box")}
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


# ── no shared filesystem: the transfer backend (#785) ─────────────────────


def _transfer_events(client: httpx.Client, run_id: str) -> list[str]:
    events = client.get(f"/api/runs/{run_id}/events").json()["events"]
    return [e["payload"]["detail"] for e in events
            if e["type"] == "run_warning"
            and e["payload"].get("kind") == "transfer_progress"]


def test_a_run_without_shared_storage_ships_its_input_and_its_checkpoint(
        live_server, tmp_path, monkeypatch):
    """Acceptance (#785): this server and the agent share no directory
    (two separate data roots, no ``--mount``). The run's uploaded CSV is
    fetched into the agent's cache and its periodic checkpoint uploaded
    here, listed as a ``transfer`` row and downloadable; a second run finds
    the CSV in the cache; pruning both runs removes what they uploaded."""
    import hashlib

    control = tmp_path / "control"
    (control / "files").mkdir(parents=True)
    csv_bytes = b"a,b,label\n1,2,0\n3,4,1\n5,6,0\n"
    (control / "files" / "train.csv").write_bytes(csv_bytes)
    monkeypatch.setattr(settings, "PROJECT_DIR", None)
    monkeypatch.setattr(settings, "MOUNT_LABEL", "data")
    for name, sub in (("DATA_FILES_DIR", "files"), ("IMAGES_DIR", "images"),
                      ("MODELS_DIR", "models"), ("MEDIA_DIR", "media")):
        monkeypatch.setattr(settings, name, control / sub)
    import app.api.ws_worker as ws_worker

    # Taken for another machine: a same-host worker borrows this server's
    # directories, and this test is about having none in common.
    monkeypatch.setattr(ws_worker, "machine_identity",
                        lambda: "not-this-agents-machine")

    base = f"http://127.0.0.1:{live_server}"
    client = httpx.Client(base_url=base, timeout=30,
                          headers={TOKEN_HEADER: session_token()})
    agent_data = tmp_path / "agent-data"
    env = {**os.environ, "PYTHONPATH": str(BACKEND_DIR),
           "CODEFYUI_USER_DATA_DIR": str(agent_data),
           "CODEFYUI_AGENT_TOKEN": session_token()}
    agent = subprocess.Popen(
        [sys.executable, "-m", "app.agent", "start", "--foreground",
         "--control", base, "--name", "bare-box", "--devices", "cpu"],
        cwd=BACKEND_DIR, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = ""
    try:
        # Online or degraded (#781): its catalog depends on what the test
        # environment has installed; the graph uses built-in nodes only.
        worker = _wait_connected(client, agent, "bare-box")
        assert worker["mounts"] == {}

        def graph() -> dict[str, Any]:
            # Plus a ModelSaver: a writer whose file lands in this
            # server's own models folder, as on a local run.
            built = _csv_and_checkpoint_graph()
            built["nodes"].append({"id": "save", "type": "ModelSaver",
                                   "data": {"params": {
                                       "path": "e2e_saved.pt"}}})
            built["edges"].append({
                "id": "e-train-save", "source": "train",
                "sourceHandle": "model", "target": "save",
                "targetHandle": "model"})
            return built

        def submit() -> str:
            submitted = client.post("/api/runs", json={
                "graph": graph(),
                "options": {"target": {"worker": "bare-box"}}})
            assert submitted.status_code in (200, 201, 202), submitted.text
            run_id = submitted.json()["run_id"]
            record = _wait_terminal(client, run_id, timeout=180)
            assert record["status"] == "succeeded", record
            assert record["worker_id"] == "bare-box"
            return run_id

        first = submit()
        progress = _transfer_events(client, first)
        assert progress and "train.csv" in progress[-1] \
            and "fetched" in progress[-1], progress
        cached = agent_data / "agent-cache" / hashlib.sha256(
            csv_bytes).hexdigest()
        assert cached.read_bytes() == csv_bytes

        artifacts = client.get(f"/api/runs/{first}/artifacts").json()[
            "artifacts"]
        [checkpoint] = [a for a in artifacts if a["kind"] == "checkpoint"]
        assert checkpoint["backend"] == "transfer"
        assert checkpoint["worker_id"] == "bare-box"
        assert checkpoint["relative_path"].startswith(
            f"runs/{first}/checkpoint/")
        stored = Path(checkpoint["path"])
        assert stored == control.resolve() / checkpoint["relative_path"]
        assert stored.is_file()
        downloaded = client.get(
            f"/api/runs/{first}/artifacts/{checkpoint['id']}/download")
        assert downloaded.status_code == 200
        assert downloaded.content == stored.read_bytes()

        [saved] = [a for a in artifacts if a["kind"] == "output"]
        assert saved["relative_path"] == "models/e2e_saved.pt"
        assert (control / "models" / "e2e_saved.pt").is_file()

        second = submit()
        progress = _transfer_events(client, second)
        assert progress and "cached" in progress[-1], progress

        # Retention: keep only the newest run, so both of these go, and
        # with them every file they uploaded here.
        assert (control / "runs" / first).is_dir()
        app.state.run_service._retention_keep_last = 1
        last = client.post("/api/runs", json={"graph": _graph()}).json()
        _wait_terminal(client, last["run_id"])
        deadline = time.monotonic() + 15
        while (control / "runs" / first).exists() \
                or (control / "runs" / second).exists():
            assert time.monotonic() < deadline, "retention kept the files"
            time.sleep(0.1)
        assert client.get(f"/api/runs/{first}").status_code == 404
        # The writer's file is the user's: retention left it.
        assert (control / "models" / "e2e_saved.pt").is_file()
    finally:
        agent.terminate()
        try:
            output, _ = agent.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            agent.kill()
            output, _ = agent.communicate()
        client.close()
    assert "Traceback" not in output, output
