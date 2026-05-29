"""Tests for ``GET /api/plugins/chapter-packs``.

A "chapter pack" is a virtual palette section declared by a plugin manifest's
``[chapter_pack]`` table. The endpoint reads each enabled plugin's manifest
fresh per request and returns a list ordered by ``position`` ascending.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core import plugin_loader


@pytest.fixture
def chapter_pack_lockfile(tmp_path, monkeypatch):
    """Mark c1 + c2 installed and enabled; c3 installed but disabled.

    Only c1 has ``[chapter_pack]`` (added in the real manifest). c2/c3 don't,
    so they must be omitted from the response. c3 is disabled to exercise the
    enabled filter even if it later gains a [chapter_pack].
    """
    target_dir = tmp_path / "plugins"
    target_dir.mkdir()
    (target_dir / "installed.json").write_text(
        json.dumps({
            "schema": 1,
            "plugins": {
                "c1": {
                    "source_kind": "builtin",
                    "source": "c1",
                    "installed_at": "2026-05-29T00:00:00Z",
                    "manifest": {"id": "c1", "name": "Chapter 1", "version": "0.1.0"},
                    "trusted_modules": [],
                    "enabled": True,
                },
                "c2": {
                    "source_kind": "builtin",
                    "source": "c2",
                    "installed_at": "2026-05-29T00:00:00Z",
                    "manifest": {"id": "c2", "name": "Chapter 2", "version": "0.1.0"},
                    "trusted_modules": [],
                    "enabled": True,
                },
                "c3": {
                    "source_kind": "builtin",
                    "source": "c3",
                    "installed_at": "2026-05-29T00:00:00Z",
                    "manifest": {"id": "c3", "name": "Chapter 3", "version": "0.1.0"},
                    "trusted_modules": [],
                    "enabled": False,
                },
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(plugin_loader, "plugins_user_root", lambda: target_dir)
    yield target_dir


@pytest.fixture
def client(chapter_pack_lockfile):
    from app.config import settings
    from app.core.auth import TOKEN_HEADER, session_token
    from app.main import app
    with TestClient(app, base_url=f"http://127.0.0.1:{settings.PORT}") as c:
        c.headers[TOKEN_HEADER] = session_token()
        yield c


def test_chapter_packs_returns_list(client):
    r = client.get("/api/plugins/chapter-packs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_c1_chapter_pack_present(client):
    r = client.get("/api/plugins/chapter-packs")
    packs = r.json()
    by_id = {p["plugin_id"]: p for p in packs}
    assert "c1" in by_id, f"expected c1 in chapter packs, got {list(by_id)}"


def test_c1_pack_label_and_node_order_preserved(client):
    """Manifest order is the pedagogical sequence — must not be alphabetised."""
    by_id = {p["plugin_id"]: p for p in client.get("/api/plugins/chapter-packs").json()}
    c1 = by_id["c1"]
    assert c1["label"] == "C1 — 基本資訊"
    expected_nodes = [
        "Start",
        "CSVReader",
        "ColumnSelector",
        "Squeeze",
        "Mean",
        "Print",
        "c1:EduColumnStats",
        "c1:EduFilterRows",
    ]
    assert c1["nodes"] == expected_nodes


def test_c1_pack_has_numeric_position(client):
    by_id = {p["plugin_id"]: p for p in client.get("/api/plugins/chapter-packs").json()}
    assert isinstance(by_id["c1"]["position"], (int, float))
    assert by_id["c1"]["position"] == 100.0


def test_plugin_without_chapter_pack_section_is_omitted(client):
    """c2 has no [chapter_pack] in its manifest → must not appear in response."""
    ids = {p["plugin_id"] for p in client.get("/api/plugins/chapter-packs").json()}
    assert "c2" not in ids


def test_disabled_plugin_omitted(client):
    """c3 is disabled in lockfile → must not appear even if it had a pack."""
    ids = {p["plugin_id"] for p in client.get("/api/plugins/chapter-packs").json()}
    assert "c3" not in ids


def test_response_sorted_by_position(client):
    """Sort key (position asc, plugin_id asc as tie-breaker)."""
    packs = client.get("/api/plugins/chapter-packs").json()
    positions = [p["position"] for p in packs]
    assert positions == sorted(positions)


def test_endpoint_route_order_not_shadowed_by_plugin_id(client):
    """``/chapter-packs`` must beat the ``/{plugin_id}`` catch-all."""
    r = client.get("/api/plugins/chapter-packs")
    assert r.status_code == 200
    body = r.json()
    # If the {plugin_id} route had matched, response would be a dict with
    # a top-level "id"/"manifest" shape (or a 404). It's a list.
    assert isinstance(body, list)
