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
import os
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
from app.core.plugins.service import PluginService
from app.main import app
from app import main

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
    disagrees with it -- a fork, a moved owner, an older tag. Where the FILES
    came from is the install's own answer, and it is also what decides the
    badge: a row whose repository is not the catalog's is not official, no
    matter which id it claims."""
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
    assert entry["official"] is False   # alice/extras is not carol/demo
    assert entry["ref"] == "v1.2.3"
    assert entry["repo"] == "alice/extras"
    assert entry["url"] == "https://github.com/alice/extras"
    # "" is the default branch, and it is what this install used -- the
    # catalog's tag is where a REinstall would go, not where these files
    # came from.
    assert by_id["ghost-pack"]["ref"] == ""


def test_a_manifest_that_declares_no_chapters_shows_none(tmp_path):
    """Presence, not truthiness. ``chapters = []`` is a pack saying it
    teaches no chapter, and the catalog's list is what a row shows when the
    pack has not said -- not what it shows when the pack said "none"."""
    plugin_dir = tmp_path / "teaches-nothing"
    plugin_dir.mkdir()
    (plugin_dir / "cdui.plugin.toml").write_text(
        '[plugin]\nid = "teaches-nothing"\nversion = "1.0"\n'
        "schema_version = 1\n\n[lessons]\nchapters = []\n",
        encoding="utf-8",
    )
    row = CatalogEntry(
        id="teaches-nothing", name="T", description="", kind="github",
        repo="alice/t", chapters=("C1", "C2"),
    )

    payload = listing._entry_payload(
        "teaches-nothing",
        row=row,
        entry={"source_kind": "github_url", "enabled": True},
        plugin_dir=plugin_dir,
        tombstoned=False,
        registry=registry,
        active_job=None,
    )
    assert payload["chapters"] == []


def test_a_deleted_node_file_is_a_cache_miss(tmp_path):
    """The scan cache is keyed on the newest mtime, and deleting a file moves
    no other file's mtime -- so without the file COUNT in the key, removing a
    node from a pack would keep advertising it. Mtimes are set explicitly so
    the deleted file is provably not the newest one."""
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    (nodes / "old.py").write_text('NODE_NAME = "Old"\n', encoding="utf-8")
    (nodes / "new.py").write_text('NODE_NAME = "New"\n', encoding="utf-8")
    os.utime(nodes / "old.py", (1_000_000, 1_000_000))
    os.utime(nodes / "new.py", (2_000_000, 2_000_000))

    assert listing.scan_node_names(nodes) == ["New", "Old"]
    (nodes / "old.py").unlink()
    assert listing.scan_node_names(nodes) == ["New"]


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


def _install_claiming(
    user_root: Path, plugin_id: str, repo: str, url: str | None = None
) -> None:
    """Put *plugin_id* in the lockfile as a URL install of *repo*.

    No ``catalog_id``: this is the free-text ``cdui plugin install
    owner/repo`` path, which is the one that can claim an id the catalog also
    uses -- deliberately, since it is how the author of an official plugin
    installs their own repository.

    *url* overrides the recorded URL, for the entry a hand-edited lockfile
    can hold: one whose URL is not on GitHub at all.
    """
    plugin_dir = user_root / plugin_id
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "cdui.plugin.toml").write_text(
        f'[plugin]\nid = "{plugin_id}"\nname = "Claimed"\n'
        f'version = "0.0.1"\nschema_version = 1\n',
        encoding="utf-8",
    )
    lockfile = plugin_loader.load_lockfile()
    lockfile["plugins"][plugin_id] = {
        "source_kind": "github_url",
        "source": repo,
        "url": url if url is not None else f"https://github.com/{repo}",
        "ref": "",
        "installed_at": "2026-06-04T00:00:00Z",
        "manifest": {"id": plugin_id, "version": "0.0.1"},
        "trusted_modules": [],
        "capabilities": [],
        "enabled": True,
    }
    plugin_loader.save_lockfile(lockfile)


async def test_a_foreign_repository_cannot_wear_the_catalogs_badge(
        anon_client, center_lockfile):
    """The id of a ``github`` catalog row is NOT reserved -- it cannot be,
    because the plugin's own author installs it by repository. So the badge
    has to be earned by the repository, not by the name on the manifest."""
    _install_claiming(center_lockfile, "self-learning", "mallory/evil")

    row = (await rows(anon_client))["self-learning"]
    assert row["status"] == "installed"
    assert row["repo"] == "mallory/evil"
    assert row["official"] is False

    listed = {p["id"]: p for p in (await anon_client.get("/api/plugins")).json()}
    assert listed["self-learning"]["official"] is False
    # Nor may it borrow the catalog row's identity by another name: a
    # ``catalog_id`` is a claim that THIS is the pack that card describes,
    # and a link to a page about different code is the same lie in smaller
    # print.
    assert listed["self-learning"]["catalog_id"] is None


async def test_a_lookalike_url_on_another_host_is_not_that_repository(
        anon_client, center_lockfile):
    """``https://evil.example.com/CodefyUI/CodefyUI-Plugin-Self-Learning``
    ends in the owner and repository of a plugin CodefyUI vouches for. Any
    reading of a recorded URL that ignores the host hands the badge over for
    the price of a domain name."""
    lookalike = "https://evil.example.com/CodefyUI/CodefyUI-Plugin-Self-Learning"
    _install_claiming(center_lockfile, "self-learning", lookalike, url=lookalike)

    row = (await rows(anon_client))["self-learning"]
    assert row["status"] == "installed"
    assert row["url"] == lookalike
    assert row["official"] is False
    # Nothing recorded says which repository this is, and ``None`` is that
    # answer. Falling through to the catalog printed the official
    # ``owner/repo`` on a row whose files came off another host entirely --
    # the badge withheld in one field and handed over in the next.
    assert row["repo"] is None

    listed = {p["id"]: p for p in (await anon_client.get("/api/plugins")).json()}
    assert listed["self-learning"]["official"] is False
    assert listed["self-learning"]["catalog_id"] is None


async def test_a_local_link_under_a_catalog_id_points_at_no_repository(
        anon_client, center_lockfile, tmp_path):
    """``cdui plugin link`` records the author's own working tree, which is
    how the author of an official plugin works on it -- so a catalog id over
    a local link is an ordinary state, not an attack. It is also a state
    where nothing at all is known about a repository: a working tree is
    whatever is in it right now. The row says so in both fields rather than
    borrowing the catalog's repository and its GitHub link."""
    work = tmp_path / "work" / "self-learning"
    work.mkdir(parents=True)
    (work / "cdui.plugin.toml").write_text(
        '[plugin]\nid = "self-learning"\nname = "My Fork"\n'
        'version = "0.0.1"\nschema_version = 1\n',
        encoding="utf-8",
    )
    lockfile = plugin_loader.load_lockfile()
    lockfile["plugins"]["self-learning"] = {
        "source_kind": "local",
        "source": str(work),
        "path": str(work),
        "installed_at": "2026-06-05T00:00:00Z",
        "manifest": {"id": "self-learning", "version": "0.0.1"},
        "trusted_modules": [],
        "capabilities": [],
        "enabled": True,
    }
    plugin_loader.save_lockfile(lockfile)

    row = (await rows(anon_client))["self-learning"]
    assert row["status"] == "installed"
    assert row["source_kind"] == "local"
    assert row["repo"] is None
    assert row["url"] is None
    assert row["official"] is False


async def test_the_catalogs_own_repository_is_official_by_any_road(
        anon_client, center_lockfile):
    """Installed by name or by URL, it is the same code from the same
    repository -- so the badge must not depend on which command was typed."""
    _install_claiming(
        center_lockfile, "self-learning", "CodefyUI/CodefyUI-Plugin-Self-Learning"
    )

    row = (await rows(anon_client))["self-learning"]
    assert row["status"] == "installed"
    assert row["repo"] == "CodefyUI/CodefyUI-Plugin-Self-Learning"
    assert row["official"] is True

    listed = {p["id"]: p for p in (await anon_client.get("/api/plugins")).json()}
    assert listed["self-learning"]["official"] is True
    # And this one really is the pack that catalog row describes, even
    # though the install recorded no ``catalog_id`` to say so.
    assert listed["self-learning"]["catalog_id"] == "self-learning"


def test_official_is_a_question_about_provenance_not_about_the_id():
    """Every branch of the rule, in one place."""
    builtin = CatalogEntry(id="stats", name="Stats", description="",
                           kind="builtin", path="plugins/stats")
    official_repo = CatalogEntry(
        id="self-learning", name="SL", description="", kind="github",
        repo="CodefyUI/CodefyUI-Plugin-Self-Learning", official=True,
    )
    listed_third_party = CatalogEntry(
        id="third", name="Third", description="", kind="github",
        repo="carol/third", official=False,
    )

    # Nothing installed: the row itself is what the badge describes.
    assert listing.is_official(builtin, None) is True
    assert listing.is_official(official_repo, None) is True
    # A row the catalog merely LISTS never earns it, installed or not.
    assert listing.is_official(listed_third_party, None) is False
    assert listing.is_official(None, None) is False

    # A built-in pack can only arrive by being activated in place.
    assert listing.is_official(builtin, {"source_kind": "builtin"}) is True
    assert listing.is_official(
        builtin, {"source_kind": "local", "source": "/home/me/stats"}) is False
    assert listing.is_official(
        builtin, {"source_kind": "github_url", "source": "mallory/stats",
                  "url": "https://github.com/mallory/stats"}) is False

    # A repository row: the recorded catalog id, or the repository itself.
    assert listing.is_official(
        official_repo,
        {"source_kind": "github_url", "catalog_id": "self-learning",
         "source": "CodefyUI/CodefyUI-Plugin-Self-Learning"}) is True
    assert listing.is_official(
        official_repo,
        {"source_kind": "github_url",
         "url": "https://github.com/codefyui/codefyui-plugin-self-learning"}
    ) is True                                   # GitHub names are not case-sensitive
    assert listing.is_official(
        official_repo,
        {"source_kind": "github_url", "source": "mallory/evil",
         "url": "https://github.com/mallory/evil"}) is False
    # A linked working tree cannot be checked against anything.
    assert listing.is_official(
        official_repo,
        {"source_kind": "local", "source": "/home/me/self-learning"}) is False


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


# -- the running install ---------------------------------------------------


class _StubInstaller:
    """Stands in for ``PluginService`` on ``app.state``: one question asked.

    The route's whole job here is to ask the service what is running and pass
    the answer on, so a stub that answers is a truer test of the route than a
    real service with a scripted flow behind it -- which would be testing
    ``PluginService`` again, in the file that is about the payload.
    """

    def __init__(self, payload):
        self.payload = payload

    def active_job_payload(self):
        return self.payload


async def test_the_catalog_reports_the_install_that_is_running(
        anon_client, monkeypatch):
    """How a panel opened mid-install -- a second tab, a reload -- finds the
    job it should be following, and which row is busy."""
    job = {"job_id": "j1", "plugin_id": "stats", "kind": "install",
           "status": "running", "current_step": "deps"}
    monkeypatch.setattr(app.state, "plugin_service",
                        _StubInstaller(job), raising=False)

    body = (await anon_client.get("/api/plugins/catalog")).json()
    by_id = {entry["id"]: entry for entry in body["entries"]}

    assert body["active_job"] == job
    assert by_id["stats"]["status"] == "installing"
    assert by_id["stats"]["job"] == {"job_id": "j1", "status": "running",
                                     "current_step": "deps"}
    assert by_id["foundations"]["job"] is None


async def test_the_catalog_is_readable_with_no_installer_behind_it(
        anon_client, monkeypatch):
    """A read, not an install: a server whose installer never started can
    still say what is installed. Nothing running answers the same way."""
    monkeypatch.setattr(app.state, "plugin_service",
                        _StubInstaller(None), raising=False)
    assert (await anon_client.get("/api/plugins/catalog")).json()[
        "active_job"] is None

    monkeypatch.delattr(app.state, "plugin_service",
                        raising=False)
    body = (await anon_client.get("/api/plugins/catalog")).json()
    assert body["active_job"] is None
    assert body["entries"]


async def test_the_lifespan_wires_the_two_installers_to_each_other(
        tmp_path, monkeypatch):
    """They install into one interpreter, so each has to be able to see the
    other's running job. Built one after the other and wired through
    ``app.state``, because whichever is built first cannot hold the other."""
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path / "userdata"))
    # The real one clears the root logger's handlers, which is not something
    # a test in the middle of a suite should do to everybody else.
    monkeypatch.setattr(main, "setup_logging", lambda **kwargs: None)

    async with main.lifespan(app):
        plugins, packs = app.state.plugin_service, app.state.pack_service
        assert isinstance(plugins, PluginService)
        assert plugins.current_job_id() is None

        # Each closure must reach the OTHER service -- crossed wires would
        # have each installer refusing itself and nothing refusing anybody.
        monkeypatch.setattr(packs, "current_job_id", lambda: "pack-9")
        monkeypatch.setattr(plugins, "current_job_id", lambda: "plugin-9")
        assert plugins._busy_elsewhere() == "pack-9"
        assert packs._busy_elsewhere() == "plugin-9"

    assert app.state.plugin_service is None
    assert app.state.pack_service is None


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
