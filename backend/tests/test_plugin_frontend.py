"""Tests for the files a plugin ships: manifest validation, bundle serving,
``assets/`` serving, and the /api/plugins frontend_entry field.

Both serving routes resolve the plugin per REQUEST, which is what the
``assets/`` tests below are really about: the URL is the one the
``StaticFiles`` mounts used to answer, and the cases that would have failed
under a mount built once at startup -- a plugin installed after the app
object exists, and a plugin disabled after it -- are the point of the change.
"""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path

import pytest

from app.api.routes_plugin_frontend import _file_under
from app.core import plugin_loader
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


#: What a pack ships in ``assets/``: a data file its nodes read and an image
#: its lesson shows. Bytes nobody parses -- the route answers on the name --
#: and written as bytes so a Windows checkout does not turn the newlines of
#: the CSV into something the assertions have to know about.
ASSET_CSV = b"a,b\n1,2\n"
ASSET_PNG = b"\x89PNG\r\n\x1a\n-not-a-real-png-"


def _write_frontend_plugin(root: Path, plugin_id: str, *, enabled: bool = True,
                           with_entry: bool = True) -> None:
    """Create a fake installed third-party plugin with a bundle and assets."""
    pdir = root / plugin_id
    (pdir / "frontend").mkdir(parents=True)
    (pdir / "assets" / "img").mkdir(parents=True)
    (pdir / "assets" / "data.csv").write_bytes(ASSET_CSV)
    (pdir / "assets" / "img" / "logo.png").write_bytes(ASSET_PNG)
    (pdir / "assets" / "blob.weird").write_bytes(b"\x00\x01\x02")
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


# -- GET /plugins/<id>/assets/<path> -----------------------------------------

def _set_enabled(root: Path, plugin_id: str, enabled: bool) -> None:
    """Flip a plugin's ``enabled`` flag in the lockfile, in place."""
    lock_path = root / "installed.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["plugins"][plugin_id]["enabled"] = enabled
    lock_path.write_text(json.dumps(lock), encoding="utf-8")


def test_serves_an_asset_with_the_type_mimetypes_gives_it(fe_client):
    r = fe_client.get("/plugins/fe-pack/assets/data.csv")
    assert r.status_code == 200
    assert r.content == ASSET_CSV
    # Not spelled "text/csv": Windows reads media types from the registry,
    # where .csv is commonly application/vnd.ms-excel. What the route
    # promises is the answer mimetypes gives on THIS machine -- and that it
    # is a real answer rather than the octet-stream fallback below.
    expected = mimetypes.guess_type("data.csv")[0]
    assert expected is not None
    assert r.headers["content-type"].split(";")[0] == expected


def test_serves_an_asset_from_a_subdirectory(fe_client):
    r = fe_client.get("/plugins/fe-pack/assets/img/logo.png")
    assert r.status_code == 200
    assert r.content == ASSET_PNG
    # Pinned in app.main for Windows, standard everywhere else.
    assert r.headers["content-type"] == "image/png"


def test_an_asset_of_an_unknown_type_downloads_rather_than_guesses(fe_client):
    """A pack may ship a .npz or a .pt as readily as a .png."""
    r = fe_client.get("/plugins/fe-pack/assets/blob.weird")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"


def test_assets_are_revalidated_like_bundles(fe_client):
    # `cdui plugin update` replaces a pack's data files under their existing
    # names, so a heuristically cached CSV outlives the update that changed it.
    r = fe_client.get("/plugins/fe-pack/assets/data.csv")
    assert "no-cache" in r.headers.get("cache-control", "")


def test_assets_need_no_manifest_declaration(fe_client):
    """Unlike the bundle: a pack either has an assets/ directory or has not,
    and a node reading its own CSV declares nothing to do it."""
    r = fe_client.get("/plugins/no-fe/assets/data.csv")
    assert r.status_code == 200


def test_404_for_unknown_plugin_and_missing_asset(fe_client):
    assert fe_client.get("/plugins/ghost/assets/data.csv").status_code == 404
    assert fe_client.get("/plugins/fe-pack/assets/missing.csv").status_code == 404


def test_404_for_a_directory_rather_than_a_listing(fe_client):
    for path in ("/plugins/fe-pack/assets/img", "/plugins/fe-pack/assets/img/"):
        r = fe_client.get(path)
        assert r.status_code == 404, path
        assert r.json()["detail"] == "Plugin asset not found", path


@pytest.mark.parametrize(
    "path",
    [
        "/plugins/fe-pack/assets/..%2Fcdui.plugin.toml",
        "/plugins/fe-pack/assets/%2e%2e/cdui.plugin.toml",
        "/plugins/fe-pack/assets/%2e%2e%2fsecret.txt",
        "/plugins/fe-pack/assets/img/%2e%2e/%2e%2e/secret.txt",
    ],
)
def test_404_for_anything_outside_the_assets_dir(fe_client, path):
    """The manifest and the pack's own files sit one level above assets/,
    and `..` has more spellings over the wire than a string check has.

    Encoded, all of them: httpx applies RFC 3986 dot-segment removal to the
    URL before it sends anything, so a plain ``../`` never reaches the app
    from this client. That spelling is proven against the check itself
    below -- a client that does not normalise can still send it."""
    r = fe_client.get(path)
    assert r.status_code == 404
    # From THIS route, not from the SPA fallback that answers 404 for
    # everything else: a refusal that arrives by accident stops being one
    # the day the route stops matching.
    assert r.json()["detail"] == "Plugin asset not found"


@pytest.mark.parametrize(
    "resource_path",
    ["../cdui.plugin.toml", "img/../../secret.txt", "../../../etc/passwd", ".."],
)
def test_the_traversal_check_refuses_a_parent_segment_unencoded(
        frontend_plugin_env, resource_path):
    """The half of the traversal contract an HTTP client normalises away."""
    assert _file_under(
        frontend_plugin_env / "fe-pack" / "assets", resource_path
    ) is None


def test_404_for_an_asset_of_a_disabled_plugin(fe_client):
    assert fe_client.get("/plugins/fe-disabled/assets/data.csv").status_code == 404


def test_disabling_a_plugin_stops_serving_its_assets_at_once(
        fe_client, frontend_plugin_env):
    """A mount created at startup kept serving a disabled plugin's files
    until a restart. The route reads the lockfile the disable just wrote."""
    assert fe_client.get("/plugins/fe-pack/assets/data.csv").status_code == 200
    _set_enabled(frontend_plugin_env, "fe-pack", False)
    assert fe_client.get("/plugins/fe-pack/assets/data.csv").status_code == 404


def test_a_plugin_installed_after_startup_serves_its_assets(
        fe_client, frontend_plugin_env):
    """The regression this route exists for. The app object -- and, under a
    mount, the whole routing table for assets -- is already built when the
    plugin arrives."""
    assert fe_client.get("/plugins/late-pack/assets/data.csv").status_code == 404
    _write_frontend_plugin(frontend_plugin_env, "late-pack")
    r = fe_client.get("/plugins/late-pack/assets/data.csv")
    assert r.status_code == 200
    assert r.content == ASSET_CSV


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
