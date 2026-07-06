"""ID5: in project mode, publish snapshots the LOGIC file's exact bytes
(positionless), and invoke runs that positionless snapshot."""

import pytest

from app.api import routes_graph


def _echo_graph():
    return {
        "name": "echo-graph", "description": "",
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
            {"id": "gi", "type": "GraphInput", "position": {"x": 9, "y": 9},
             "data": {"params": {"name": "x", "type": "string", "required": True,
                                 "default": "", "description": ""}}},
            {"id": "out", "type": "GraphOutput", "position": {"x": 5, "y": 5},
             "data": {"params": {"name": "y", "description": ""}}},
        ],
        "edges": [
            {"id": "t1", "source": "start", "target": "gi", "sourceHandle": "trigger",
             "targetHandle": "", "type": "trigger"},
            {"id": "d1", "source": "gi", "target": "out", "sourceHandle": "value",
             "targetHandle": "value", "type": "data"},
        ],
    }


@pytest.fixture
def project(monkeypatch, tmp_path):
    monkeypatch.setattr(routes_graph.settings, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(routes_graph.settings, "GRAPHS_DIR", tmp_path / "graphs")
    (tmp_path / "graphs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "layout").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


async def test_publish_snapshots_positionless_logic_bytes(test_client, app_db, project):
    await test_client.post("/api/graph/save", json=_echo_graph())
    logic_bytes = (project / "graphs" / "echo-graph.graph.json").read_text()
    assert '"position"' not in logic_bytes  # split kept positions out of logic

    r = await test_client.post("/api/apps/echo/publish",
                               json={"graph": "echo-graph", "create": True})
    assert r.status_code == 200, r.text

    def _snap(conn):
        return conn.execute(
            "SELECT graph_json FROM app_versions WHERE version = 1").fetchone()[0]

    assert await app_db.run(_snap) == logic_bytes  # snapshot == logic bytes

    # And it still invokes (execution never reads positions). Mint a real key
    # through the actual endpoint (mirrors the `api_key` fixture in
    # test_api_apps_invoke.py) rather than a raw INSERT, so the bearer token
    # actually authenticates.
    key_resp = await test_client.post("/api/keys", json={"name": "id5-test"})
    assert key_resp.status_code == 200, key_resp.text
    token = key_resp.json()["token"]

    inv = await test_client.post(
        "/api/apps/echo/invoke",
        json={"inputs": {"x": "hello"}},
        headers=_bearer(token),
    )
    assert inv.status_code == 200, inv.text
    assert inv.json()["outputs"] == {"y": "hello"}
