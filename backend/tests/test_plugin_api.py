"""Integration tests for the plugin HTTP surface.

Drives FastAPI TestClient so the lifespan runs, then asserts the wire
shapes of ``/api/plugins``, ``/api/nodes`` (provider field), ``/api/examples``
(plugin source tag + plugin-shipped graph loading), and ``/api/plugins/reload``.

A temp lockfile is used so the user's real install state can't leak into
the test outcome.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core import plugin_loader


_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def chapter_lockfile(tmp_path, monkeypatch):
    """Redirect plugins_user_root + write a deterministic lockfile with c2/c3/c4 builtin."""
    target_dir = tmp_path / "plugins"
    target_dir.mkdir()
    (target_dir / "installed.json").write_text(
        json.dumps({
            "schema": 1,
            "plugins": {
                "c2": {
                    "source_kind": "builtin",
                    "source": "c2",
                    "installed_at": "2026-05-16T00:00:00Z",
                    "manifest": {"id": "c2", "name": "Chapter 2 — Classical AI", "version": "0.1.0"},
                    "trusted_modules": [],
                },
                "c3": {
                    "source_kind": "builtin",
                    "source": "c3",
                    "installed_at": "2026-05-16T00:00:00Z",
                    "manifest": {"id": "c3", "name": "Chapter 3", "version": "0.1.0"},
                    "trusted_modules": [],
                },
                "c4": {
                    "source_kind": "builtin",
                    "source": "c4",
                    "installed_at": "2026-05-16T00:00:00Z",
                    "manifest": {"id": "c4", "name": "Chapter 4", "version": "0.1.0"},
                    "trusted_modules": [],
                },
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(plugin_loader, "plugins_user_root", lambda: target_dir)
    yield target_dir


@pytest.fixture
def client(chapter_lockfile):
    """TestClient with lifespan run AFTER the lockfile redirect — discovery sees c2/c3/c4."""
    from app.config import settings
    from app.core.auth import TOKEN_HEADER, session_token
    from app.main import app
    with TestClient(app, base_url=f"http://127.0.0.1:{settings.PORT}") as c:
        c.headers[TOKEN_HEADER] = session_token()
        yield c


# ── /api/plugins ───────────────────────────────────────────────────────────

def test_list_plugins_returns_all_chapter_packs(client):
    r = client.get("/api/plugins")
    assert r.status_code == 200
    data = r.json()
    ids = {p["id"] for p in data}
    assert {"c2", "c3", "c4"}.issubset(ids)


def test_list_plugins_populates_node_names(client):
    r = client.get("/api/plugins")
    by_id = {p["id"]: p for p in r.json()}
    assert "EduKNN" in by_id["c2"]["nodes"]
    assert "EduLinearRegression" in by_id["c2"]["nodes"]
    assert "EduLogisticRegression" in by_id["c2"]["nodes"]
    assert "EduCrossAttention" in by_id["c3"]["nodes"]
    assert "EduResBlock" in by_id["c3"]["nodes"]
    assert {"EduFFN", "EduMultiHeadAttention", "EduSelfAttention", "EduTokenEmbedding"} <= set(
        by_id["c4"]["nodes"]
    )


def test_get_plugin_returns_manifest(client):
    r = client.get("/api/plugins/c2")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == "c2"
    assert data["manifest"]["plugin"]["id"] == "c2"
    assert data["manifest"]["lessons"]["chapters"] == ["C2"]
    # nodes from discovery, not just manifest
    assert "EduKNN" in data["nodes"]


def test_get_plugin_returns_404_for_unknown(client):
    r = client.get("/api/plugins/does-not-exist")
    assert r.status_code == 404


def test_reload_plugins_returns_counts(client):
    r = client.post("/api/plugins/reload")
    assert r.status_code == 200
    data = r.json()
    for key in ("builtin", "custom", "plugins", "presets", "total"):
        assert key in data
    assert data["plugins"] >= 9  # 3 EduC2 + 2 EduC3 + 4 EduC4
    assert data["total"] == data["builtin"] + data["custom"] + data["plugins"]


# ── /api/plugins/{id}/enable|disable ───────────────────────────────────────

def test_list_plugins_includes_enabled_flag(client):
    """Every entry must carry the explicit enabled field for the UI."""
    r = client.get("/api/plugins")
    by_id = {p["id"]: p for p in r.json()}
    for pid in ("c2", "c3", "c4"):
        assert "enabled" in by_id[pid]
        assert by_id[pid]["enabled"] is True


def test_disable_then_enable_via_api(client):
    """Toggling drops c2 from the registry then restores it after enable."""
    r = client.post("/api/plugins/c2/disable")
    assert r.status_code == 200
    assert r.json()["enabled"] is False

    # After disable, c2 must still appear in the listing (it's installed)
    # but its enabled flag flips and its nodes drop out of /api/nodes.
    by_id = {p["id"]: p for p in client.get("/api/plugins").json()}
    assert by_id["c2"]["enabled"] is False
    assert by_id["c2"]["nodes"] == []  # not registered → none reported

    nodes_after_disable = {n["node_name"] for n in client.get("/api/nodes").json()}
    # Names are qualified ("c2:EduKNN") since the registry namespacing change —
    # confirms both the disable filter and the qualified-name surface in one go.
    assert "c2:EduKNN" not in nodes_after_disable
    # Other plugins' nodes survive — toggle is per-plugin only.
    assert "c3:EduCrossAttention" in nodes_after_disable

    # Re-enable restores everything.
    r = client.post("/api/plugins/c2/enable")
    assert r.status_code == 200
    assert r.json()["enabled"] is True

    nodes_after_enable = {n["node_name"] for n in client.get("/api/nodes").json()}
    assert "c2:EduKNN" in nodes_after_enable


def test_disable_missing_plugin_returns_404(client):
    r = client.post("/api/plugins/does-not-exist/disable")
    assert r.status_code == 404


def test_enable_missing_plugin_returns_404(client):
    r = client.post("/api/plugins/does-not-exist/enable")
    assert r.status_code == 404


def test_disabled_plugin_examples_disappear_from_list(client):
    """`/api/examples/list` must hide examples shipped by disabled plugins."""
    # Each example carries source="plugin:c2" (exact) and path="plugin:c2/...".
    def c2_examples() -> set[str]:
        return {
            e["path"]
            for e in client.get("/api/examples/list").json()
            if e.get("source") == "plugin:c2"
        }

    before = c2_examples()
    assert before, "expected some c2 plugin examples in the catalog before disable"

    client.post("/api/plugins/c2/disable")
    after = c2_examples()
    assert after == set(), f"c2 examples should be hidden when disabled, got {after}"

    # Restore so the next test (or another worker on shared state) isn't surprised.
    client.post("/api/plugins/c2/enable")
    assert c2_examples() == before


# ── /api/nodes provider field ──────────────────────────────────────────────

def test_provider_field_is_plugin_for_edu_nodes(client):
    r = client.get("/api/nodes")
    assert r.status_code == 200
    by_name = {n["node_name"]: n for n in r.json()}
    # API names are namespaced (c2:EduKNN, not bare EduKNN) after the plugin
    # node-name registry change — provider field stays a clean ``plugin:<id>``.
    assert by_name["c2:EduKNN"]["provider"] == "plugin:c2"
    assert by_name["c3:EduCrossAttention"]["provider"] == "plugin:c3"
    assert by_name["c4:EduSelfAttention"]["provider"] == "plugin:c4"


def test_provider_field_is_builtin_for_production_nodes(client):
    r = client.get("/api/nodes")
    by_name = {n["node_name"]: n for n in r.json()}
    # 'KNN' (sklearn-backed production) lives in backend/app/nodes/
    assert by_name["KNN"]["provider"] == "builtin"


# ── /api/examples picks up plugin examples ─────────────────────────────────

def test_examples_list_includes_plugin_sourced_entries(client):
    r = client.get("/api/examples/list")
    assert r.status_code == 200
    examples = r.json()
    plugin = [e for e in examples if e.get("source", "").startswith("plugin:")]
    # 2 from c2 + 2 from c3 + 2 from c4 = 6
    assert len(plugin) >= 6
    sources = {e["source"] for e in plugin}
    assert {"plugin:c2", "plugin:c3", "plugin:c4"}.issubset(sources)


def test_examples_load_resolves_plugin_path(client):
    r = client.get(
        "/api/examples/load",
        params={"path": "plugin:c2/Classical/KNN-from-Scratch"},
    )
    assert r.status_code == 200
    graph = r.json()
    assert "nodes" in graph
    # Plugin example graphs now reference nodes by their qualified type
    # ("c2:EduKNN"), so two plugins can't shadow each other and a reader
    # can tell which pack the node came from just by looking at the JSON.
    assert any(n.get("type") == "c2:EduKNN" for n in graph["nodes"])


def test_examples_load_rejects_traversal_in_plugin_path(client):
    r = client.get(
        "/api/examples/load",
        params={"path": "plugin:c2/../../etc/passwd"},
    )
    # 400 (rejected outright) or 404 (couldn't resolve) — both are safe
    assert r.status_code in (400, 404)
