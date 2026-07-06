"""format_version read policy (ID8): loads warn but NEVER block."""

import json
import logging


async def test_load_newer_format_warns_never_blocks(test_client, monkeypatch, tmp_path, caplog):
    monkeypatch.setattr("app.config.settings.PROJECT_DIR", None)
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    (tmp_path / "foo.json").write_text(json.dumps(
        {"format_version": 999, "name": "foo", "nodes": [], "edges": []}))
    with caplog.at_level(logging.WARNING):
        r = await test_client.get("/api/graph/load/foo")
    assert r.status_code == 200
    assert r.json()["format_version"] == 999  # returned, never blocked
    assert any("format_version" in rec.getMessage() for rec in caplog.records)
