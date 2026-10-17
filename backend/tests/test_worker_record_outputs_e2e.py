"""End to end: ``record_outputs`` on a real agent (#789).

A real server (uvicorn on a loopback port, its real lifespan) and a real
``cdui agent start --foreground`` subprocess. The same graph runs here and
on the agent with ``record_outputs``; every read of
``/api/execution/outputs/*`` must answer the same for both, while the run
still goes on the worker (its run subprocess answers) and after it ended
(the agent's outputs helper answers from the files). Then the worker's
budget drops an older run, a server that forgot the run (as after a
restart) finds it again from the run's record, and a worker that is gone
answers 503 naming it, at once.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from app.core.auth import TOKEN_HEADER, session_token
from app.main import app

from .test_worker_e2e import (  # noqa: F401 -- live_server is a fixture
    BACKEND_DIR,
    TERMINAL,
    _stop_agent,
    _wait_connected,
    _wait_terminal,
    live_server,
)

#: Holds the run on the worker for a few seconds after its inputs ran, so
#: the reads below find it still going there.
SLOW_NODE = '''
import time

from app.core.node_base import BaseNode, DataType, PortDefinition


class E2ESlowPass(BaseNode):
    NODE_NAME = "E2ESlowPass"
    CATEGORY = "Test"

    @classmethod
    def define_inputs(cls):
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls):
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs, params, *, context=None):
        time.sleep(float(params.get("seconds", 4)))
        return {"value": inputs.get("value")}
'''


def _node(node_id: str, node_type: str, **params: Any) -> dict[str, Any]:
    return {"id": node_id, "type": node_type, "data": {"params": params}}


def _graph(slow: bool) -> dict[str, Any]:
    """A 4x70000 tensor (over the preview limit), scaled, and a model; on
    the worker, a slow node after them keeps the run going."""
    nodes = [_node("start", "Start"),
             _node("x", "TensorCreate", shape="4,70000", fill="ones",
                   value=0, requires_grad=False),
             _node("scale", "ScalarMultiply", scalar=2.5),
             _node("model", "SequentialModel")]
    edges = [
        {"id": "t", "source": "start", "target": "x",
         "sourceHandle": "trigger", "type": "trigger"},
        {"id": "tm", "source": "start", "target": "model",
         "sourceHandle": "trigger", "type": "trigger"},
        {"id": "a", "source": "x", "sourceHandle": "tensor",
         "target": "scale", "targetHandle": "tensor"}]
    if slow:
        nodes.append(_node("slow", "E2ESlowPass", seconds=10))
        edges.append({"id": "b", "source": "scale", "sourceHandle": "tensor",
                      "target": "slow", "targetHandle": "value"})
    return {"nodes": nodes, "edges": edges, "settings": {"seed": 0}}


#: ``(path, query)`` of the reads compared between the two runs.
READS: list[tuple[str, dict[str, Any]]] = [
    ("scale/tensor", {"preview": "true"}),
    ("scale/tensor", {"slice": "1,0:5"}),
    ("scale/tensor", {"max_elements": "10"}),          # 413
    ("scale/tensor", {"slice": "9"}),                  # 400
    ("scale/tensor/stats", {}),
    ("x/tensor", {"preview": "true", "max_elements": "100"}),
    ("model/model", {}),
    ("model/model/stats", {}),
    ("scale/nope", {}),                                # 404
    ("scale/nope/stats", {}),                          # 404 with the hint
    ("scale/__steps_index", {}),
    ("scale/__grad_index", {}),
]


def _read(client: httpx.Client, run_id: str, path: str,
          query: dict[str, Any]) -> tuple[int, Any]:
    response = client.get(f"/api/execution/outputs/{run_id}/{path}",
                          params=query)
    body: Any = None
    if response.status_code != 204:
        body = response.json()
    return response.status_code, _without_run_id(body, run_id)


def _without_run_id(body: Any, run_id: str) -> Any:
    """The body with the run's own id taken out, so two runs compare."""
    if isinstance(body, dict):
        return {k: (_without_run_id(v, run_id) if k != "run_id" else "<run>")
                for k, v in body.items()}
    if isinstance(body, str):
        return body.replace(run_id, "<run>")
    if isinstance(body, list):
        return [_without_run_id(item, run_id) for item in body]
    return body


def _start(base: str, data: Path, nodes: Path, name: str,
           budget_mb: int | None = None) -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": str(BACKEND_DIR),
           "CODEFYUI_USER_DATA_DIR": str(data),
           "CODEFYUI_CUSTOM_NODES_DIR": str(nodes),
           "CODEFYUI_AGENT_TOKEN": session_token()}
    if budget_mb is not None:
        env["CODEFYUI_RUN_OUTPUT_STORE_MAX_MB"] = str(budget_mb)
    return subprocess.Popen(
        [sys.executable, "-m", "app.agent", "start", "--foreground",
         "--control", base, "--name", name, "--devices", "cpu"],
        cwd=BACKEND_DIR, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _submit(client: httpx.Client, graph: dict[str, Any],
            **options: Any) -> str:
    response = client.post("/api/runs", json={
        "graph": graph, "options": {"record_outputs": True, **options}})
    assert response.status_code in (200, 201, 202), response.text
    return response.json()["run_id"]


def _until(check: Any, timeout: float = 60, what: str = "") -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def test_a_worker_runs_outputs_read_like_a_local_runs(live_server,  # noqa: F811
                                                       tmp_path):
    base = f"http://127.0.0.1:{live_server}"
    client = httpx.Client(base_url=base, timeout=60,
                          headers={TOKEN_HEADER: session_token()})
    local = _submit(client, _graph(slow=False))
    assert _wait_terminal(client, local)["status"] == "succeeded"
    expected = {(path, tuple(sorted(query.items()))):
                _read(client, local, path, query) for path, query in READS}
    assert expected[("scale/tensor", (("preview", "true"),))][0] == 200
    assert expected[("scale/tensor", (("max_elements", "10"),))][0] == 413

    nodes = tmp_path / "nodes"
    nodes.mkdir()
    (nodes / "e2e_slow.py").write_text(SLOW_NODE)
    # A budget of 1 MiB: one run's captures (a 1.1 MB tensor twice) are
    # over it alone, so the next recorded run drops this one's.
    agent = _start(base, tmp_path / "data", nodes, "rec-box", budget_mb=1)
    try:
        _wait_connected(client, agent, "rec-box")
        remote = _submit(client, _graph(slow=True),
                         target={"worker": "rec-box"})

        # While it runs there: the run subprocess answers from its store.
        def scale_is_read() -> bool:
            status, _ = _read(client, remote, "scale/tensor",
                              {"preview": "true"})
            return status == 200

        _until(scale_is_read, what="the live read of scale.tensor")
        assert client.get(f"/api/runs/{remote}").json()["status"] \
            not in TERMINAL, "the run ended before it was read live"
        for path, query in READS:
            key = (path, tuple(sorted(query.items())))
            assert _read(client, remote, path, query) == expected[key], key
        listed = client.get(f"/api/execution/outputs/{remote}").json()
        assert {(e["node_id"], e["port"]) for e in listed} >= {
            ("x", "tensor"), ("scale", "tensor"), ("model", "model")}

        record = _wait_terminal(client, remote, timeout=120)
        assert record["status"] == "succeeded", record
        # After the run: the agent's helper answers from the files.
        for path, query in READS:
            key = (path, tuple(sorted(query.items())))
            assert _read(client, remote, path, query) == expected[key], key
        slow = _read(client, remote, "slow/value", {"preview": "true"})
        assert slow[0] == 200 and slow[1]["full_shape"] == [4, 70000]

        # Stats of an ended run are cached here: a second read does not
        # go to the worker.
        cache = app.state.port_stats_cache
        hits = cache.hits
        assert _read(client, remote, "scale/tensor/stats", {})[0] == 200
        assert cache.hits == hits + 1

        # A server that forgot the run (as after a restart) finds it on
        # its worker again from the run's record, and retains it.
        store = app.state.run_output_store
        loop = app.state.worker_registry._watchdog.get_loop()
        import asyncio

        asyncio.run_coroutine_threadsafe(store.forget_remote(remote),
                                         loop).result(10)
        assert _read(client, remote, "scale/tensor", {"slice": "0,0:3"}) \
            == (200, _read(client, local, "scale/tensor",
                           {"slice": "0,0:3"})[1])
        assert asyncio.run_coroutine_threadsafe(
            store.remote_holder(remote), loop).result(10) == "rec-box"

        # The worker's budget: the next recorded run drops this one's.
        second = _submit(client, _graph(slow=False),
                         target={"worker": "rec-box"})
        assert _wait_terminal(client, second)["status"] == "succeeded"
        assert _read(client, second, "scale/tensor",
                     {"preview": "true"})[0] == 200
        gone = _until(lambda: (lambda r: r if r[0] == 404 else None)(
            _read(client, remote, "scale/tensor", {"preview": "true"})),
            what="the budget to drop the older run")
        assert gone[1]["detail"] == "run '<run>' not found"

        # DELETE removes them on the worker.
        deleted = client.delete(f"/api/execution/outputs/{second}")
        assert deleted.status_code == 200, deleted.text
        assert deleted.json() == {"run_id": second, "deleted": True}
        assert _read(client, second, "scale/tensor", {})[0] == 404
    finally:
        _stop_agent(client, agent, "rec-box")
    # A worker that is gone: 503 at once, naming it.
    started = time.monotonic()
    response = client.get(f"/api/execution/outputs/{remote}/scale/tensor")
    assert response.status_code == 503, response.text
    assert "rec-box" in response.json()["detail"]
    assert time.monotonic() - started < 5
    client.close()
