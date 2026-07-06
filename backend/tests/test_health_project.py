"""/api/health gains an additive `project` field (spec ID4).

Additive is scoped to PROJECT MODE ONLY: the byte-for-byte non-project
refactor guard (spec header Global Constraints) means the key must be
ABSENT -- not merely null -- when settings.PROJECT_DIR is None, so a
non-project response body stays identical to pre-Task-10 main.py.
"""

from app.api import routes_graph  # for the shared settings object


async def test_health_project_key_absent_when_unset(test_client, monkeypatch):
    monkeypatch.setattr(routes_graph.settings, "PROJECT_DIR", None)
    r = await test_client.get("/api/health")
    body = r.json()
    assert body["status"] == "ok"
    assert "project" not in body


async def test_health_project_dir_in_project_mode(test_client, monkeypatch, tmp_path):
    monkeypatch.setattr(routes_graph.settings, "PROJECT_DIR", tmp_path)
    r = await test_client.get("/api/health")
    assert r.json()["project"] == str(tmp_path)
