"""Tests for plugin frontend-extension support (manifest validation,
bundle serving, /api/plugins frontend_entry field)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.plugin_loader import frontend_entry_rel


# -- frontend_entry_rel ------------------------------------------------------

def test_entry_rel_returns_normalized_path():
    m = {"frontend": {"entry": "frontend/index.js"}}
    assert frontend_entry_rel(m) == "frontend/index.js"


def test_entry_rel_accepts_nested_path():
    m = {"frontend": {"entry": "frontend/dist/main.js"}}
    assert frontend_entry_rel(m) == "frontend/dist/main.js"


def test_entry_rel_none_when_table_missing():
    assert frontend_entry_rel({}) is None
    assert frontend_entry_rel({"plugin": {"id": "x"}}) is None


def test_entry_rel_none_when_entry_missing_or_not_string():
    assert frontend_entry_rel({"frontend": {}}) is None
    assert frontend_entry_rel({"frontend": {"entry": 3}}) is None
    assert frontend_entry_rel({"frontend": {"entry": ""}}) is None


def test_entry_rel_rejects_traversal_and_absolute():
    assert frontend_entry_rel({"frontend": {"entry": "frontend/../secrets.py"}}) is None
    assert frontend_entry_rel({"frontend": {"entry": "../frontend/index.js"}}) is None
    assert frontend_entry_rel({"frontend": {"entry": "/etc/passwd"}}) is None


def test_entry_rel_rejects_paths_outside_frontend_dir():
    assert frontend_entry_rel({"frontend": {"entry": "nodes/evil.js"}}) is None
    assert frontend_entry_rel({"frontend": {"entry": "frontend"}}) is None


def test_entry_rel_normalizes_backslashes():
    assert frontend_entry_rel({"frontend": {"entry": "frontend\\index.js"}}) == "frontend/index.js"


from app.core import plugin_loader


def _write_frontend_plugin(root: Path, plugin_id: str, *, enabled: bool = True,
                           with_entry: bool = True) -> None:
    """Create a fake installed third-party plugin with a frontend bundle."""
    pdir = root / plugin_id
    (pdir / "frontend").mkdir(parents=True)
    manifest = [
        "[plugin]",
        f'id = "{plugin_id}"',
        f'name = "{plugin_id}"',
        'version = "0.1.0"',
        "schema_version = 1",
    ]
    if with_entry:
        manifest += ["", "[frontend]", 'entry = "frontend/index.js"']
    (pdir / "cdui.plugin.toml").write_text("\n".join(manifest), encoding="utf-8")
    (pdir / "frontend" / "index.js").write_text(
        "export default function activate(api) {}", encoding="utf-8"
    )
    (pdir / "frontend" / "style.css").write_text(".x{}", encoding="utf-8")
    # A file OUTSIDE frontend/ that must never be reachable via the route.
    (pdir / "secret.txt").write_text("nope", encoding="utf-8")

    lock_path = root / "installed.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {
        "schema": 1, "plugins": {},
    }
    lock["plugins"][plugin_id] = {
        "source_kind": "github_url",
        "source": f"someone/{plugin_id}",
        "installed_at": "2026-06-11T00:00:00Z",
        "manifest": {"id": plugin_id, "name": plugin_id, "version": "0.1.0"},
        "trusted_modules": [],
        "enabled": enabled,
    }
    lock_path.write_text(json.dumps(lock), encoding="utf-8")


@pytest.fixture
def frontend_plugin_env(tmp_path, monkeypatch):
    root = tmp_path / "plugins"
    root.mkdir()
    _write_frontend_plugin(root, "fe-pack")
    _write_frontend_plugin(root, "fe-disabled", enabled=False)
    _write_frontend_plugin(root, "no-fe", with_entry=False)
    monkeypatch.setattr(plugin_loader, "plugins_user_root", lambda: root)
    yield root


@pytest.fixture
def fe_client(frontend_plugin_env):
    from app.config import settings
    from app.core.auth import TOKEN_HEADER, session_token
    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app, base_url=f"http://127.0.0.1:{settings.PORT}") as c:
        c.headers[TOKEN_HEADER] = session_token()
        yield c


# -- GET /plugins/<id>/frontend/<path> ---------------------------------------

def test_serves_frontend_js_with_module_mime(fe_client):
    r = fe_client.get("/plugins/fe-pack/frontend/index.js")
    assert r.status_code == 200
    assert "activate" in r.text
    assert r.headers["content-type"].startswith("text/javascript")


def test_frontend_bundle_sets_revalidation_cache_control(fe_client):
    # Plugin bundles ship under a fixed filename and change on
    # `cdui plugin update`, so the route must force revalidation — otherwise
    # browsers heuristically cache the JS and keep serving stale plugin code.
    r = fe_client.get("/plugins/fe-pack/frontend/index.js")
    assert r.status_code == 200
    assert "no-cache" in r.headers.get("cache-control", "")


def test_serves_css(fe_client):
    r = fe_client.get("/plugins/fe-pack/frontend/style.css")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/css")


def test_404_for_file_outside_frontend_dir(fe_client):
    # Encoded traversal — TestClient does not collapse %2e%2e.
    r = fe_client.get("/plugins/fe-pack/frontend/%2e%2e/secret.txt")
    assert r.status_code == 404


def test_404_for_disabled_plugin(fe_client):
    r = fe_client.get("/plugins/fe-disabled/frontend/index.js")
    assert r.status_code == 404


def test_404_when_manifest_has_no_frontend_table(fe_client):
    r = fe_client.get("/plugins/no-fe/frontend/index.js")
    assert r.status_code == 404


def test_404_for_unknown_plugin_and_missing_file(fe_client):
    assert fe_client.get("/plugins/ghost/frontend/index.js").status_code == 404
    assert fe_client.get("/plugins/fe-pack/frontend/missing.js").status_code == 404


# -- /api/plugins frontend_entry ---------------------------------------------

def test_list_plugins_exposes_frontend_entry(fe_client):
    by_id = {p["id"]: p for p in fe_client.get("/api/plugins").json()}
    assert by_id["fe-pack"]["frontend_entry"] == "/plugins/fe-pack/frontend/index.js"


def test_list_plugins_frontend_entry_null_when_absent_or_disabled(fe_client):
    by_id = {p["id"]: p for p in fe_client.get("/api/plugins").json()}
    assert by_id["no-fe"]["frontend_entry"] is None
    assert by_id["fe-disabled"]["frontend_entry"] is None


def test_list_plugins_frontend_entry_null_when_file_missing(frontend_plugin_env):
    # Declared in manifest but the bundle file is gone.
    (frontend_plugin_env / "fe-pack" / "frontend" / "index.js").unlink()
    from app.config import settings
    from app.core.auth import TOKEN_HEADER, session_token
    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app, base_url=f"http://127.0.0.1:{settings.PORT}") as c:
        c.headers[TOKEN_HEADER] = session_token()
        by_id = {p["id"]: p for p in c.get("/api/plugins").json()}
    assert by_id["fe-pack"]["frontend_entry"] is None
