"""``run_graph.py``: the graph file's ``settings.device`` is the default for
``--device``, and an explicit flag overrides it.

``execute_graph`` is replaced with a capturing fake, so the assertion is on
the ``ExecutionContext`` the CLI builds and nothing runs on a device.
"""

from __future__ import annotations

import json

import pytest

import run_graph


def _graph(device: str | None = None) -> dict:
    graph = {
        "name": "probe",
        "nodes": [
            {"id": "start", "type": "Start", "data": {"params": {}}},
            {"id": "t", "type": "TensorCreate",
             "data": {"params": {"shape": "2", "fill": "ones"}}},
        ],
        "edges": [
            {"id": "e", "source": "start", "target": "t",
             "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
        ],
    }
    if device is not None:
        graph["settings"] = {"device": device}
    return graph


@pytest.fixture
def captured(monkeypatch):
    calls: list[dict] = []

    async def fake_execute_graph(nodes, edges, **kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(run_graph, "execute_graph", fake_execute_graph)
    # Identity resolution, so the file's assignment is visible as-is.
    monkeypatch.setattr(
        "app.core.device_utils.resolve_device",
        lambda requested: (requested or "cpu").strip().lower() or "cpu")
    return calls


async def test_the_file_device_becomes_the_context_device(tmp_path, captured):
    path = tmp_path / "g.json"
    path.write_text(json.dumps(_graph("cpu")), encoding="utf-8")
    await run_graph.run(str(path))
    assert captured[0]["context"].device == "cpu"


async def test_the_flag_overrides_the_file_device(tmp_path, captured):
    path = tmp_path / "g.json"
    path.write_text(json.dumps(_graph("cuda")), encoding="utf-8")
    await run_graph.run(str(path), device="cpu")
    assert captured[0]["context"].device == "cpu"
    await run_graph.run(str(path))
    assert captured[1]["context"].device == "cuda"


async def test_no_device_anywhere_keeps_the_default_context(tmp_path, captured):
    path = tmp_path / "g.json"
    path.write_text(json.dumps(_graph()), encoding="utf-8")
    await run_graph.run(str(path))
    assert captured[0]["context"] is None
