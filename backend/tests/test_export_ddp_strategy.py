"""Export Python with a ``strategy: ddp`` TrainingLoop (#792).

The exported script keeps the single-process path: ``codegen`` is not
touched, the script carries the param like any other, and the TrainingLoop
it runs trains in one process with a warning. The export answers a
``ddp_strategy`` warning beside the script so the toolbar can say so. A
graph without ``strategy: ddp`` exports exactly as before.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.agent import _ddp

RUN_TIMEOUT_S = 300


def _graph(strategy: str | None, *, label: str | None = None) -> dict:
    graph = _ddp.training_graph(batch_size=8, epochs=2, strategy=strategy,
                                save_as=None)
    for node in graph["nodes"]:
        node.setdefault("position", {"x": 0, "y": 0})
        if label is not None and node["id"] == "train":
            node["data"]["label"] = label
    graph["name"] = "ddp-export"
    return graph


async def _export(test_client, graph: dict) -> dict:
    response = await test_client.post("/api/graph/export", json=graph)
    assert response.status_code == 200, response.text
    return response.json()


async def test_a_ddp_loop_exports_with_a_warning(test_client):
    body = await _export(test_client, _graph("ddp", label="Fine-tune"))
    assert body["warnings"] == [{
        "code": "ddp_strategy", "node_id": "train", "label": "Fine-tune",
        "param": "strategy", "value": "ddp", "containers": [],
    }]
    assert "'strategy': 'ddp'" in body["script"]


async def test_a_graph_without_ddp_exports_as_before(test_client):
    plain = await _export(test_client, _graph(None))
    assert plain["warnings"] == []
    assert "strategy" not in plain["script"]
    single = await _export(test_client, _graph("single"))
    assert single["warnings"] == []


async def test_a_ddp_loop_inside_a_block_is_named_by_its_block(test_client):
    inner = _ddp.training_graph(batch_size=8, epochs=1, save_as=None)
    graph = {
        "name": "ddp-in-block",
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
             "data": {"params": {}}},
            {"id": "blk", "type": "subgraph:stage",
             "position": {"x": 100, "y": 0}, "data": {"params": {}}},
        ],
        "edges": [{"id": "t", "source": "start", "target": "blk",
                   "sourceHandle": "trigger", "targetHandle": "",
                   "type": "trigger"}],
        "subgraphs": [{
            "id": "stage", "name": "Stage", "description": "",
            "nodes": [n for n in inner["nodes"] if n["id"] != "start"],
            "edges": [e for e in inner["edges"] if e["source"] != "start"],
            "interface": {"inputs": [], "outputs": [],
                          "triggerTargets": ["gen", "model", "loss"]},
        }],
    }
    body = await _export(test_client, graph)
    (warning,) = [w for w in body["warnings"] if w["code"] == "ddp_strategy"]
    assert warning["node_id"] == "blk/train"
    assert warning["containers"] == [{"node_id": "blk", "label": "Stage"}]


def _grader_env(home: Path) -> dict[str, str]:
    env = {name: value for name, value in os.environ.items()
           if not name.upper().startswith("CODEFYUI_")}
    env["CODEFYUI_USER_DATA_DIR"] = str(home / "user-data")
    env["CODEFYUI_DATA_FILES_DIR"] = str(home / "data" / "files")
    env["CODEFYUI_IMAGES_DIR"] = str(home / "data" / "images")
    env["CODEFYUI_MODELS_DIR"] = str(home / "data" / "models")
    env["CODEFYUI_MEDIA_DIR"] = str(home / "data" / "media")
    return env


async def test_the_exported_script_trains_a_ddp_loop_in_one_process(
        test_client, tmp_path):
    body = await _export(test_client, _graph("ddp"))
    folder = tmp_path / "grader"
    folder.mkdir()
    (folder / "ddp_graph.py").write_bytes(body["script"].encode("utf-8"))
    completed = subprocess.run(
        [sys.executable, "-I", "ddp_graph.py", "--device", "cpu"],
        cwd=folder, env=_grader_env(tmp_path / "home"),
        capture_output=True, timeout=RUN_TIMEOUT_S, check=False)
    stderr = completed.stderr.decode("utf-8", errors="replace")
    assert completed.returncode == 0, stderr
    assert "strategy=ddp trains across the CUDA cards" in stderr
    assert "trains in one process" in stderr
