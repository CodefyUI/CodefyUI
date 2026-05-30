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
def direction_lockfile(tmp_path, monkeypatch):
    """Redirect plugins_user_root + write a deterministic lockfile with the
    three direction packs (foundations / deep / rl) marked builtin."""
    target_dir = tmp_path / "plugins"
    target_dir.mkdir()
    (target_dir / "installed.json").write_text(
        json.dumps({
            "schema": 1,
            "plugins": {
                "foundations": {
                    "source_kind": "builtin",
                    "source": "foundations",
                    "installed_at": "2026-05-30T00:00:00Z",
                    "manifest": {"id": "foundations", "name": "Foundations — Data & Classic ML", "version": "0.1.0"},
                    "trusted_modules": [],
                },
                "deep": {
                    "source_kind": "builtin",
                    "source": "deep",
                    "installed_at": "2026-05-30T00:00:00Z",
                    "manifest": {"id": "deep", "name": "Deep Models", "version": "0.1.0"},
                    "trusted_modules": [],
                },
                "rl": {
                    "source_kind": "builtin",
                    "source": "rl",
                    "installed_at": "2026-05-30T00:00:00Z",
                    "manifest": {"id": "rl", "name": "Reinforcement Learning", "version": "0.1.0"},
                    "trusted_modules": [],
                },
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(plugin_loader, "plugins_user_root", lambda: target_dir)
    yield target_dir


@pytest.fixture
def client(direction_lockfile):
    """TestClient with lifespan run AFTER the lockfile redirect — discovery sees the 3 packs."""
    from app.config import settings
    from app.core.auth import TOKEN_HEADER, session_token
    from app.main import app
    with TestClient(app, base_url=f"http://127.0.0.1:{settings.PORT}") as c:
        c.headers[TOKEN_HEADER] = session_token()
        yield c


# ── /api/plugins ───────────────────────────────────────────────────────────

def test_list_plugins_returns_all_direction_packs(client):
    r = client.get("/api/plugins")
    assert r.status_code == 200
    data = r.json()
    ids = {p["id"] for p in data}
    assert {"foundations", "deep", "rl"}.issubset(ids)


def test_list_plugins_populates_node_names(client):
    r = client.get("/api/plugins")
    by_id = {p["id"]: p for p in r.json()}
    assert {
        "Edu-ColumnStats", "Edu-KNN", "Edu-LinearRegression",
        "Edu-LogisticRegression", "Edu-TokenEmbedding", "Edu-FFN",
    } <= set(by_id["foundations"]["nodes"])
    assert {
        "Edu-CrossAttention", "Edu-ResBlock", "Edu-SelfAttention",
        "Edu-MultiHeadAttention", "Edu-Patchify",
    } <= set(by_id["deep"]["nodes"])
    assert "Edu-PolicyGradient" in by_id["rl"]["nodes"]


def test_get_plugin_returns_manifest(client):
    r = client.get("/api/plugins/foundations")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == "foundations"
    assert data["manifest"]["plugin"]["id"] == "foundations"
    assert data["manifest"]["lessons"]["chapters"] == ["C1", "C2"]
    # nodes from discovery, not just manifest
    assert "Edu-KNN" in data["nodes"]


def test_get_plugin_returns_404_for_unknown(client):
    r = client.get("/api/plugins/does-not-exist")
    assert r.status_code == 404


def test_reload_plugins_returns_counts(client):
    r = client.post("/api/plugins/reload")
    assert r.status_code == 200
    data = r.json()
    for key in ("builtin", "custom", "plugins", "presets", "total"):
        assert key in data
    assert data["plugins"] >= 12  # 6 foundations + 5 deep + 1 rl
    assert data["total"] == data["builtin"] + data["custom"] + data["plugins"]


# ── /api/plugins/{id}/enable|disable ───────────────────────────────────────

def test_list_plugins_includes_enabled_flag(client):
    """Every entry must carry the explicit enabled field for the UI."""
    r = client.get("/api/plugins")
    by_id = {p["id"]: p for p in r.json()}
    for pid in ("foundations", "deep", "rl"):
        assert "enabled" in by_id[pid]
        assert by_id[pid]["enabled"] is True


def test_disable_then_enable_via_api(client):
    """Toggling drops foundations from the registry then restores it after enable."""
    r = client.post("/api/plugins/foundations/disable")
    assert r.status_code == 200
    assert r.json()["enabled"] is False

    # After disable, foundations must still appear in the listing (it's installed)
    # but its enabled flag flips and its nodes drop out of /api/nodes.
    by_id = {p["id"]: p for p in client.get("/api/plugins").json()}
    assert by_id["foundations"]["enabled"] is False
    assert by_id["foundations"]["nodes"] == []  # not registered → none reported

    nodes_after_disable = {n["node_name"] for n in client.get("/api/nodes").json()}
    # Names are qualified ("foundations:Edu-KNN") since the registry namespacing
    # change — confirms both the disable filter and the qualified-name surface.
    assert "foundations:Edu-KNN" not in nodes_after_disable
    # Other plugins' nodes survive — toggle is per-plugin only.
    assert "deep:Edu-CrossAttention" in nodes_after_disable

    # Re-enable restores everything.
    r = client.post("/api/plugins/foundations/enable")
    assert r.status_code == 200
    assert r.json()["enabled"] is True

    nodes_after_enable = {n["node_name"] for n in client.get("/api/nodes").json()}
    assert "foundations:Edu-KNN" in nodes_after_enable


def test_disable_missing_plugin_returns_404(client):
    r = client.post("/api/plugins/does-not-exist/disable")
    assert r.status_code == 404


def test_enable_missing_plugin_returns_404(client):
    r = client.post("/api/plugins/does-not-exist/enable")
    assert r.status_code == 404


def test_disabled_plugin_examples_disappear_from_list(client):
    """`/api/examples/list` must hide examples shipped by disabled plugins."""
    # Each example carries source="plugin:foundations" (exact) and path="plugin:foundations/...".
    def foundations_examples() -> set[str]:
        return {
            e["path"]
            for e in client.get("/api/examples/list").json()
            if e.get("source") == "plugin:foundations"
        }

    before = foundations_examples()
    assert before, "expected some foundations plugin examples in the catalog before disable"

    client.post("/api/plugins/foundations/disable")
    after = foundations_examples()
    assert after == set(), f"foundations examples should be hidden when disabled, got {after}"

    # Restore so the next test (or another worker on shared state) isn't surprised.
    client.post("/api/plugins/foundations/enable")
    assert foundations_examples() == before


# ── /api/nodes provider field ──────────────────────────────────────────────

def test_provider_field_is_plugin_for_edu_nodes(client):
    r = client.get("/api/nodes")
    assert r.status_code == 200
    by_name = {n["node_name"]: n for n in r.json()}
    # API names are namespaced (foundations:Edu-KNN, not bare Edu-KNN) after the
    # plugin node-name registry change — provider stays a clean ``plugin:<id>``.
    assert by_name["foundations:Edu-KNN"]["provider"] == "plugin:foundations"
    assert by_name["deep:Edu-CrossAttention"]["provider"] == "plugin:deep"
    assert by_name["deep:Edu-SelfAttention"]["provider"] == "plugin:deep"


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
    assert len(plugin) >= 6
    sources = {e["source"] for e in plugin}
    assert {"plugin:foundations", "plugin:deep", "plugin:rl"}.issubset(sources)


def test_examples_load_resolves_plugin_path(client):
    r = client.get(
        "/api/examples/load",
        params={"path": "plugin:foundations/Classical/KNN-from-Scratch"},
    )
    assert r.status_code == 200
    graph = r.json()
    assert "nodes" in graph
    # Plugin example graphs now reference nodes by their qualified type
    # ("foundations:Edu-KNN"), so two plugins can't shadow each other and a
    # reader can tell which pack the node came from just by looking at the JSON.
    assert any(n.get("type") == "foundations:Edu-KNN" for n in graph["nodes"])


def test_examples_load_rejects_traversal_in_plugin_path(client):
    r = client.get(
        "/api/examples/load",
        params={"path": "plugin:foundations/../../etc/passwd"},
    )
    # 400 (rejected outright) or 404 (couldn't resolve) — both are safe
    assert r.status_code in (400, 404)
