"""``GET /api/plugins/catalog``: the one route the Plugin Center draws from.

The panel merges two documents that do not know about each other -- the
catalog this build ships and the lockfile this install wrote -- so the tests
here are mostly about the SEAM. Each state a plugin can be in gets a row of
its own in one fixture, because the states are defined by which document
mentions the plugin and how:

    foundations      in both, enabled          -> installed
    deep             in both, disabled         -> disabled
    rl               tombstoned, not installed -> removed
    stats / edu      catalog only, on disk     -> available
    self-learning    catalog only, a repo      -> available, nothing promised
    demo-external    lockfile only             -> external
    ghost-pack       lockfile only, no files   -> missing_files

Three of those are worth saying out loud. ``available`` has to name the
nodes the pack would add WITHOUT importing them -- that is the whole content
of the card, and importing is what installing means. ``missing_files`` is
the state that used to render as a normal installed row whose nodes had all
silently vanished. And ``removed`` is the #175 tombstone: "you threw this
away" and "you have never had this" are the two states an Install button
has to tell apart.

The key list is asserted as an ORDERED list rather than a set: the payload
is what a TypeScript type will be written against, and a field that quietly
changes place is a diff worth seeing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute
from httpx import ASGITransport, AsyncClient

from app.api import routes_plugins
from app.config import settings
from app.core import plugin_loader
from app.core.node_registry import registry
from app.core.plugins import listing
from app.core.plugins.catalog import CatalogEntry
from app.core.plugins.listing import catalog_listing
from app.core.plugins.reload import rediscover_now
from app.main import app

BASE_URL = f"http://127.0.0.1:{settings.PORT}"

#: Every key in a catalog row, in order. Additive is a decision: a new field
#: should break this once, on purpose.
ENTRY_KEYS = [
    "id", "name", "description", "kind", "official", "status", "source_kind",
    "source", "repo", "ref", "sha", "url", "homepage", "version",
    "installed_at", "enabled", "chapters", "lessons", "tags", "nodes",
    "node_count", "capabilities", "trusted_modules", "python_deps",
    "has_frontend", "consent_required", "frontend_entry", "job",
]

TOP_KEYS = {"entries", "active_job", "remote_install_allowed", "generation"}

#: Deliberately asks for MORE than the lockfile entry below was granted: a
#: plugin that rewrites its own manifest after install must not be able to
#: show the new list as though the user had agreed to it.
EXTERNAL_MANIFEST = """\
[plugin]
id = "demo-external"
name = "Demo External"
version = "2.0.0"
description = "Installed from a URL nobody put in the catalog."
schema_version = 1

[security]
capabilities = ["network", "filesystem"]
allowed_modules = ["requests", "pathlib"]

[python_deps]
tabulate = ">=0.9"
"""


@pytest.fixture
def center_lockfile(tmp_path, monkeypatch) -> Path:
    """A user root holding one plugin of every state the panel can draw."""
    user_root = tmp_path / "plugins"
    (user_root / "demo-external").mkdir(parents=True)
    (user_root / "demo-external" / "cdui.plugin.toml").write_text(
        EXTERNAL_MANIFEST, encoding="utf-8"
    )
    (user_root / "installed.json").write_text(
        json.dumps({
            "schema": 1,
            "plugins": {
                "foundations": {
                    "source_kind": "builtin",
                    "source": "foundations",
                    "installed_at": "2026-05-30T00:00:00Z",
                    "manifest": {"id": "foundations", "version": "0.1.0"},
                    "trusted_modules": [],
                    "capabilities": [],
                    "enabled": True,
                },
                "deep": {
                    "source_kind": "builtin",
                    "source": "deep",
                    "installed_at": "2026-05-30T00:00:00Z",
                    "manifest": {"id": "deep", "version": "0.1.0"},
                    "trusted_modules": [],
                    "capabilities": [],
                    "enabled": False,
                },
                "demo-external": {
                    "source_kind": "github_url",
                    "source": "alice/extras@v1.2.3",
                    "url": "https://github.com/alice/extras",
                    "ref": "v1.2.3",
                    "sha": "0" * 40,
                    "installed_at": "2026-06-01T00:00:00Z",
                    "manifest": {"id": "demo-external", "version": "2.0.0"},
                    "trusted_modules": ["requests"],
                    "capabilities": ["network"],
                    "enabled": True,
                },
                "ghost-pack": {
                    "source_kind": "github_url",
                    "source": "bob/ghost",
                    "url": "https://github.com/bob/ghost",
                    "ref": "",
                    "sha": "1" * 40,
                    "installed_at": "2026-06-02T00:00:00Z",
                    "manifest": {"id": "ghost-pack", "name": "Ghost Pack",
                                 "version": "9.9.9"},
                    "trusted_modules": [],
                    "capabilities": [],
                    "enabled": True,
                },
            },
            "removed": {
                "rl": {"removed_at": "2026-06-03T00:00:00Z",
                       "source_kind": "builtin"},
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(plugin_loader, "plugins_user_root", lambda: user_root)
    return user_root


@pytest.fixture
async def anon_client(center_lockfile):
    """No session token: what a browser that never bootstrapped would send."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url=BASE_URL,
    ) as http:
        yield http


async def rows(client) -> dict[str, dict]:
    response = await client.get("/api/plugins/catalog")
    assert response.status_code == 200, response.text
    return {entry["id"]: entry for entry in response.json()["entries"]}


# -- the contract ---------------------------------------------------------


async def test_the_catalog_is_an_open_get_with_the_envelope_it_promises(
        anon_client, monkeypatch):
    """No token, four top-level keys, and the gate the panel greys out on."""
    monkeypatch.setattr(settings, "HOST", "127.0.0.1")
    monkeypatch.setattr(settings, "ALLOW_REMOTE_PLUGIN_INSTALL", False)

    response = await anon_client.get("/api/plugins/catalog")
    assert response.status_code == 200, response.text
    body = response.json()

    assert set(body) == TOP_KEYS
    assert body["active_job"] is None
    assert body["remote_install_allowed"] is True
    assert body["generation"] == plugin_loader.reload_generation()
    assert isinstance(body["entries"], list) and body["entries"]


async def test_every_row_has_every_field_in_the_same_order(anon_client):
    body = (await anon_client.get("/api/plugins/catalog")).json()
    for entry in body["entries"]:
        assert list(entry) == ENTRY_KEYS, entry["id"]
        assert entry["node_count"] == len(entry["nodes"])
        assert isinstance(entry["python_deps"], dict)


async def test_the_catalog_and_the_lockfile_are_merged_once_each(anon_client):
    """The catalog in the order it is written, then the strangers, sorted."""
    body = (await anon_client.get("/api/plugins/catalog")).json()
    ids = [entry["id"] for entry in body["entries"]]

    assert len(ids) == len(set(ids)), "a plugin was listed twice"
    assert ids[:5] == ["edu", "foundations", "deep", "stats", "rl"]
    assert ids[-2:] == ["demo-external", "ghost-pack"]
    assert set(ids[5:-2]) == {"graph-copilot", "self-learning",
                              "official-template"}


# -- one row per state ----------------------------------------------------


async def test_an_installed_builtin_reports_the_nodes_it_registered(anon_client):
    row = (await rows(anon_client))["foundations"]
    assert row["status"] == "installed"
    assert row["kind"] == "builtin"
    assert row["official"] is True
    assert row["enabled"] is True
    assert row["source_kind"] == "builtin"
    assert row["installed_at"] == "2026-05-30T00:00:00Z"
    assert row["consent_required"] is False
    # From the live registry -- these are the palette entries, not a guess.
    assert "Edu-KNN" in row["nodes"]
    # From the manifest on disk, not from the catalog's prose.
    assert row["version"] == "0.1.0"
    assert row["chapters"] == ["C1", "C2"]


async def test_a_disabled_builtin_stays_listed_and_says_so(anon_client):
    row = (await rows(anon_client))["deep"]
    assert row["status"] == "disabled"
    assert row["enabled"] is False
    assert row["installed_at"] == "2026-05-30T00:00:00Z"


async def test_an_uninstalled_builtin_names_the_nodes_it_would_add(anon_client):
    """Read as text off disk: the card has to say what the pack contains
    before the user agrees to run any of it."""
    row = (await rows(anon_client))["stats"]
    assert row["status"] == "available"
    assert row["official"] is True
    assert row["enabled"] is False
    assert row["installed_at"] is None
    assert row["source_kind"] is None
    assert row["source"] == "stats"
    assert row["consent_required"] is False
    assert {"Stats-Describe", "Stats-Histogram"} <= set(row["nodes"])
    assert row["tags"] == ["statistics", "chart", "table", "eda",
                           "confusion-matrix"]
    # The manifest of a pack that ships in this release is on this disk, so
    # an available built-in row still answers "what version, what deps".
    assert row["version"] == "0.1.0"


async def test_a_pack_the_user_removed_is_not_merely_absent(anon_client):
    """#175: "you threw this away" and "you never had this" are two states."""
    row = (await rows(anon_client))["rl"]
    assert row["status"] == "removed"
    assert row["enabled"] is False
    assert row["installed_at"] is None
    assert "Edu-PolicyGradient" in row["nodes"]  # still installable


async def test_an_available_github_entry_promises_nothing_it_cannot_see(
        anon_client):
    """Nothing is fetched to draw a card, so the repository half is empty."""
    row = (await rows(anon_client))["self-learning"]
    assert row["kind"] == "github"
    assert row["official"] is True
    assert row["status"] == "available"
    assert row["consent_required"] is True
    assert row["nodes"] == []
    assert row["node_count"] == 0
    assert row["version"] is None
    assert row["python_deps"] == {}
    assert row["has_frontend"] is False
    assert row["repo"] == "CodefyUI/CodefyUI-Plugin-Self-Learning"
    assert row["url"] == "https://github.com/CodefyUI/CodefyUI-Plugin-Self-Learning"
    assert row["source"] == "CodefyUI/CodefyUI-Plugin-Self-Learning"


async def test_a_plugin_the_catalog_never_heard_of_is_external(anon_client):
    row = (await rows(anon_client))["demo-external"]
    assert row["kind"] == "external"
    assert row["official"] is False
    assert row["status"] == "installed"
    assert row["source_kind"] == "github_url"
    assert row["consent_required"] is True   # it came out of a repository
    assert row["repo"] == "alice/extras"
    assert row["url"] == "https://github.com/alice/extras"
    assert row["ref"] == "v1.2.3"
    assert row["sha"] == "0" * 40
    assert row["tags"] == []
    # The lockfile records what was consented to, and its manifest has since
    # started asking for more -- which must not show as though it had been
    # agreed to. What the plugin WOULD install is a different question, and
    # the manifest is the only place that can answer it.
    assert row["capabilities"] == ["network"]
    assert row["trusted_modules"] == ["requests"]
    assert row["python_deps"] == {"tabulate": ">=0.9"}
    assert row["name"] == "Demo External"


async def test_an_entry_whose_files_are_gone_says_missing_files(anon_client):
    """The state that used to draw as a normal row with no nodes in it."""
    row = (await rows(anon_client))["ghost-pack"]
    assert row["status"] == "missing_files"
    assert row["kind"] == "external"
    assert row["nodes"] == []
    assert row["frontend_entry"] is None
    # Nothing can be read off a directory that is not there, so the name and
    # version come from what the install wrote down.
    assert row["name"] == "Ghost Pack"
    assert row["version"] == "9.9.9"


async def test_a_running_job_marks_only_its_own_row(center_lockfile):
    """``installing`` outranks every other status, on that row alone."""
    job = {"job_id": "j1", "plugin_id": "stats", "kind": "install",
           "status": "running", "current_step": "deps"}
    listing = catalog_listing(
        plugin_loader.load_lockfile(),
        registry=registry,
        active_job=job,
        remote_install_allowed=True,
        generation=7,
    )
    by_id = {entry["id"]: entry for entry in listing["entries"]}

    assert listing["active_job"] == job
    assert listing["generation"] == 7
    assert by_id["stats"]["status"] == "installing"
    assert by_id["stats"]["job"] == {"job_id": "j1", "status": "running",
                                     "current_step": "deps"}
    assert by_id["foundations"]["status"] == "installed"
    assert by_id["foundations"]["job"] is None


async def test_what_was_installed_beats_what_the_catalog_pins(
        center_lockfile, monkeypatch):
    """A row can be backed by a catalog entry AND by an install that
    disagrees with it -- a fork, a moved owner, an older tag. The catalog
    still says whether the plugin is official; where the FILES came from is
    the install's own answer."""
    pinned = {
        plugin_id: CatalogEntry(
            id=plugin_id, name="Demo", description="", kind="github",
            repo="carol/demo", ref="v9.9.9", official=True,
        )
        for plugin_id in ("demo-external", "ghost-pack")
    }
    monkeypatch.setattr(listing, "catalog_entries", lambda: pinned)

    listed = listing.catalog_listing(
        plugin_loader.load_lockfile(),
        registry=registry,
        remote_install_allowed=True,
        generation=0,
    )
    by_id = {e["id"]: e for e in listed["entries"]}
    entry = by_id["demo-external"]

    assert entry["kind"] == "github"
    assert entry["official"] is True
    assert entry["ref"] == "v1.2.3"
    assert entry["repo"] == "alice/extras"
    assert entry["url"] == "https://github.com/alice/extras"
    # "" is the default branch, and it is what this install used -- the
    # catalog's tag is where a REinstall would go, not where these files
    # came from.
    assert by_id["ghost-pack"]["ref"] == ""


def test_a_recorded_empty_grant_is_not_a_miss():
    """"You granted this plugin nothing" is an answer. Falling through to
    the manifest there would show an ungranted capability as agreed to."""
    manifest = {"security": {"capabilities": ["network"],
                             "allowed_modules": ["requests"]}}
    assert listing.declared_capabilities({"capabilities": []}, manifest) == []
    assert listing.declared_trusted_modules({"trusted_modules": []}, manifest) == []
    # A lockfile written before either field existed still asks the manifest.
    assert listing.declared_capabilities({}, manifest) == ["network"]
    assert listing.declared_trusted_modules({}, manifest) == ["requests"]
    assert listing.declared_capabilities(None, manifest) == ["network"]


# -- the router ------------------------------------------------------------


def test_every_fixed_path_is_declared_before_the_plugin_id_route():
    """A pack cannot be called ``catalog``, but the router is not the place
    to find that out. Starlette matches in registration order, so a fixed
    path declared after ``/{plugin_id}`` is reachable only by that promise."""
    paths = [route.path for route in routes_plugins.router.routes
             if isinstance(route, APIRoute)]
    assert "/api/plugins/catalog" in paths
    first_dynamic = paths.index("/api/plugins/{plugin_id}")
    for index, path in enumerate(paths):
        if "{" not in path:
            assert index < first_dynamic, f"{path} is declared too late"


# -- the install gate ------------------------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "[::1]"])
def test_a_loopback_bind_may_install_plugins(host, monkeypatch):
    monkeypatch.setattr(settings, "HOST", host)
    monkeypatch.setattr(settings, "ALLOW_REMOTE_PLUGIN_INSTALL", False)
    assert routes_plugins.remote_plugin_install_allowed() is True


def test_a_wildcard_bind_may_not_unless_it_opts_in(monkeypatch):
    monkeypatch.setattr(settings, "HOST", "0.0.0.0")
    monkeypatch.setattr(settings, "ALLOW_REMOTE_PLUGIN_INSTALL", False)
    assert routes_plugins.remote_plugin_install_allowed() is False

    monkeypatch.setattr(settings, "ALLOW_REMOTE_PLUGIN_INSTALL", True)
    assert routes_plugins.remote_plugin_install_allowed() is True


async def test_the_gate_names_the_variable_that_lifts_it(monkeypatch):
    """A 403 that does not say how to allow the thing is a dead end."""
    monkeypatch.setattr(settings, "HOST", "192.168.1.20")
    monkeypatch.setattr(settings, "ALLOW_REMOTE_PLUGIN_INSTALL", False)

    with pytest.raises(HTTPException) as refused:
        await routes_plugins._require_local_plugin_install()
    assert refused.value.status_code == 403
    assert "CODEFYUI_ALLOW_REMOTE_PLUGIN_INSTALL" in refused.value.detail

    monkeypatch.setattr(settings, "ALLOW_REMOTE_PLUGIN_INSTALL", True)
    assert await routes_plugins._require_local_plugin_install() is None


async def test_the_catalog_says_when_installing_is_refused(anon_client,
                                                           monkeypatch):
    monkeypatch.setattr(settings, "HOST", "0.0.0.0")
    monkeypatch.setattr(settings, "ALLOW_REMOTE_PLUGIN_INSTALL", False)
    body = (await anon_client.get("/api/plugins/catalog")).json()
    assert body["remote_install_allowed"] is False


# -- the shared reload -----------------------------------------------------


async def test_rediscover_now_answers_what_the_reload_route_answers(
        test_client):
    """The route is a one-line call over it, so the two cannot disagree."""
    response = await test_client.post("/api/plugins/reload")
    assert response.status_code == 200, response.text
    over_http = response.json()
    direct = rediscover_now()

    assert set(direct) == {"builtin", "custom", "plugins", "presets", "total"}
    assert direct == over_http
    assert direct["total"] == (direct["builtin"] + direct["custom"]
                               + direct["plugins"])
