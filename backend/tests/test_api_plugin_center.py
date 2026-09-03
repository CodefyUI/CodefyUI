"""The Plugin Center's REST surface: the catalog it draws, and the install.

Part 1 is ``GET /api/plugins/catalog``, the one route the panel draws from.
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

Part 2 is the install path -- ``POST /inspect``, ``POST /install`` and the
two job routes -- and it is a different kind of test: a real ``PluginService``
on ``app.state`` with a flow driven from the test thread, and assertions
about status codes and the ``detail.code`` a panel branches on. Nothing there
installs anything or opens a socket; GitHub is faked at
``app.core.plugins.github``, one call at a time.

Part 3 is the lifecycle -- ``DELETE /api/plugins/{id}``, ``POST /{id}/update``
and the two toggles -- over the same fixtures: the uninstall really does edit
that throwaway lockfile and really does delete out of that throwaway user
root, because the answer the panel draws (what was removed, what was left
behind, what is still installed) is only worth asserting against a lockfile
something wrote. The update reads that lockfile too, which is what makes
"follow the repository the user actually installed from" testable at all.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute
from httpx import ASGITransport, AsyncClient

from app.api import routes_plugins
from app.config import settings
from app.core.auth import TOKEN_HEADER, init_allowed_hosts, session_token
from app.core import plugin_loader
from app.core.node_registry import registry
from app.core.plugins import catalog as catalog_module
from app.core.plugins import github
from app.core.plugins import inspect as inspect_module
from app.core.plugins import listing
from app.core.plugins.catalog import CatalogEntry
from app.core.plugins.errors import (
    GitHubError,
    PluginInstallError,
    PluginNeedsRestart,
    ReservedPluginId,
)
from app.core.plugins.inspect import ALLOWED_MODULES_WARNING, FRONTEND_WARNING
from app.core.plugins.listing import catalog_listing
from app.core.plugins.reload import rediscover_now
from app.core.plugins.service import PluginService
from app.main import app
from app import main

from tests.test_plugin_service import (
    ScriptedFlow,
    Sources,
    types_of,
    wait_started,
)

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

    # The install path, named rather than derived: three of its paths carry
    # no ``{`` and are covered by the loop above, and the two job routes do
    # carry one -- which is exactly why they need saying out loud. Their
    # first segment (``jobs``) is fixed, and it is only a reserved plugin id
    # that keeps a pack from claiming it.
    for path in ("/api/plugins/inspect",
                 "/api/plugins/install",
                 "/api/plugins/jobs/{job_id}/events",
                 "/api/plugins/jobs/{job_id}/cancel"):
        assert path in paths, f"{path} is not registered"
        assert paths.index(path) < first_dynamic, f"{path} is declared too late"


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
    # No project: the lifespan would otherwise load the developer's own
    # project .env INTO os.environ (a raw assignment monkeypatch cannot
    # undo) and warn about its plugin pins. ``test_main_lifespan.py`` gives
    # itself a throwaway project for that; this test is not about it.
    monkeypatch.setattr(settings, "PROJECT_DIR", None)

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
    # The lifespan re-ran init_allowed_hosts with the CORS origins added --
    # harmless, but it is process-wide state, so the conftest-seeded
    # whitelist is put back for whoever runs next.
    init_allowed_hosts(settings.HOST, settings.PORT)


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


# ==========================================================================
# part 2: the install path
# ==========================================================================
#
# A real ``PluginService`` behind the routes, because what is being tested is
# how the service's refusals look on the wire. Nothing here installs
# anything: the flow is scripted from the test thread and the two GitHub
# calls an inspection makes are answered from memory.

A_SHA = "a" * 40

#: Every key ``POST /api/plugins/inspect`` answers with, in order. The
#: TypeScript ``PluginInspection`` in ``frontend/src/api/rest.ts`` is written
#: against this list. ``owner`` and ``repo`` are deliberately absent: they
#: exist on the ``Inspection`` the service holds, for the install to download
#: from, and the wire contract does not carry them.
INSPECTION_KEYS = [
    "inspection_id", "expires_at", "kind", "mode", "plugin_id", "catalog_id",
    "official", "source", "url", "ref", "sha", "name", "version",
    "description", "homepage", "manifest", "capabilities", "allowed_modules",
    "python_deps", "has_frontend", "chapters", "lessons", "consent_required",
    "installed", "up_to_date", "capabilities_added", "allowed_modules_added",
    "warnings",
]

#: A repository plugin that asks for everything a consent screen has to draw:
#: a capability, a module the gate would otherwise refuse, a Python package
#: and browser code.
REPO_MANIFEST = """\
[plugin]
id = "extras"
name = "Extras"
version = "1.2.0"
description = "Nodes somebody wrote."
homepage = "https://example.com/extras"
schema_version = 1

[security]
capabilities = ["network"]
allowed_modules = ["requests"]

[python_deps]
tabulate = ">=0.9"

[frontend]
entry = "web/index.js"

[lessons]
chapters = ["C7"]
lessons = ["intro"]
"""


def a_manifest(plugin_id: str = "extras", **plugin_fields: object) -> str:
    """The smallest manifest this build will install, plus *plugin_fields*."""
    lines = [f'id = "{plugin_id}"', "schema_version = 1", 'version = "1.0.0"']
    lines += [f'{key} = "{value}"' for key, value in plugin_fields.items()]
    return "[plugin]\n" + "\n".join(lines) + "\n"


class FakeGitHub:
    """The two calls an INSPECTION makes, answered without a socket.

    Not the tarball: the install half of ``core.plugins.github`` is never
    reached from these tests, because the flow that would download one is
    scripted. Patched on the module rather than on ``inspect`` so the real
    ``inspect_github`` runs -- the sha, the manifest read, the validation and
    the reserved-id refusal are all the production ones.
    """

    def __init__(self, monkeypatch) -> None:
        self._monkeypatch = monkeypatch
        #: Every ``(owner, repo, ref)`` a resolve was asked for.
        self.resolved: list[tuple[str, str, str]] = []

    def answers(self, manifest: str, *, sha: str = A_SHA) -> None:
        def resolve(owner: str, repo: str, ref: str) -> str:
            self.resolved.append((owner, repo, ref))
            return sha

        self._monkeypatch.setattr(github, "resolve_sha", resolve)
        self._monkeypatch.setattr(github, "fetch_manifest_text",
                                  lambda owner, repo, at: manifest)

    def answers_bytes(self, raw: bytes, *, sha: str = A_SHA) -> None:
        """Serve *raw* as the manifest FILE, with the real decode in the path.

        ``fetch_manifest_text`` lets a ``UnicodeDecodeError`` out on purpose
        -- a manifest that is not text is a disk answer, not a network one --
        so a test about how the route maps it must not fake the function that
        raises it. Faked at ``_gh_get``, which the module's own docstring
        names as the one place every request goes through.
        """
        self._monkeypatch.setattr(github, "resolve_sha",
                                  lambda owner, repo, ref: sha)
        self._monkeypatch.setattr(github, "_gh_get",
                                  lambda url, *args, **kwargs: raw)

    def raises(self, exc: BaseException) -> None:
        def boom(*args, **kwargs):
            raise exc

        self._monkeypatch.setattr(github, "resolve_sha", boom)
        self._monkeypatch.setattr(github, "fetch_manifest_text", boom)


@pytest.fixture
def fake_github(monkeypatch) -> FakeGitHub:
    return FakeGitHub(monkeypatch)


@pytest.fixture
def flow() -> ScriptedFlow:
    return ScriptedFlow()


@pytest.fixture
async def plugin_service(flow, center_lockfile):
    """A ``PluginService`` on ``app.state``, installed and removed by hand.

    The lifespan does not run under httpx's ASGITransport (the
    ``pack_service`` precedent in ``test_api_packs.py``), so the service these
    routes reach for is put there by this fixture. ``reload`` is a stub on
    purpose: the real one clears the node registry every other test in the
    session shares, and none of these tests is about re-discovery.

    ``center_lockfile`` comes with it, so every inspection here compares
    against a throwaway lockfile rather than the developer's own.
    """
    service = PluginService(run_flow=flow, reload=lambda: {})
    previous = getattr(app.state, "plugin_service", None)
    app.state.plugin_service = service
    try:
        yield service
    finally:
        await service.shutdown()
        if previous is None:
            if hasattr(app.state, "plugin_service"):
                delattr(app.state, "plugin_service")
        else:
            app.state.plugin_service = previous


@pytest.fixture
async def client(plugin_service):
    """A client carrying the session token, over that service."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url=BASE_URL,
        headers={TOKEN_HEADER: session_token()},
    ) as http:
        yield http


@pytest.fixture
async def uninstalled_client(center_lockfile, monkeypatch):
    """A client with the token, over a server whose installer never started.

    Not merely "no service in this test": the attribute is REMOVED, which is
    the state ``app.state`` is in when the lifespan has not run at all.
    """
    monkeypatch.delattr(app.state, "plugin_service", raising=False)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url=BASE_URL,
        headers={TOKEN_HEADER: session_token()},
    ) as http:
        yield http


async def inspected(client, source: str) -> dict:
    """POST /inspect and return the body, asserting it was answered."""
    response = await client.post("/api/plugins/inspect",
                                 json={"source": source})
    assert response.status_code == 200, response.text
    return response.json()


async def start_install(client, source: str = "alice/extras",
                        **answers) -> str:
    """Inspect *source*, install what came back, return the job id.

    Both turns, because that is the only way to reach ``/install``: the id it
    takes is minted by ``/inspect`` and never by a client.
    """
    inspection = await inspected(client, source)
    response = await client.post(
        "/api/plugins/install",
        json={"inspection_id": inspection["inspection_id"], **answers})
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


async def drain(client, job_id: str, *, timeout: float = 20.0
                ) -> tuple[list[dict], str]:
    """Every event of a job over HTTP, waiting for it to finish."""
    limit = 500
    events: list[dict] = []
    cursor, status = 0, "running"
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() > deadline:
            raise AssertionError(f"job {job_id} never finished ({status})")
        response = await client.get(f"/api/plugins/jobs/{job_id}/events",
                                    params={"cursor": cursor, "wait": 1.0,
                                            "limit": limit})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["job_id"] == job_id
        events.extend(body["events"])
        cursor, status = body["cursor"], body["status"]
        # A page cut short by ``limit`` can carry a terminal status without
        # the terminal event, so both have to be true before we stop.
        if status != "running" and len(body["events"]) < limit:
            return events, status


# -- POST /api/plugins/inspect ---------------------------------------------


async def test_inspecting_a_builtin_answers_the_whole_consent_screen(client):
    """A pack that ships in this release goes through inspect too, so the
    panel has ONE flow -- and comes back with nobody to be asked about."""
    body = await inspected(client, "stats")

    assert list(body) == INSPECTION_KEYS
    assert len(body["inspection_id"]) == 32
    # ISO-8601 UTC: "expires_at: 1731.9" means nothing in a browser.
    assert body["expires_at"].endswith("+00:00")
    assert body["kind"] == "builtin"
    assert body["mode"] == "install"
    assert body["plugin_id"] == "stats"
    assert body["catalog_id"] == "stats"
    assert body["official"] is True
    assert body["source"] == "stats"
    assert body["version"] == "0.1.0"
    # Nothing was resolved and nothing was downloaded: the files are here.
    assert body["url"] is None
    assert body["ref"] is None
    assert body["sha"] is None
    assert body["consent_required"] is False
    assert body["warnings"] == []
    assert body["capabilities"] == []
    assert body["allowed_modules"] == []
    assert body["installed"] is None
    assert body["up_to_date"] is False


async def test_inspecting_a_repository_says_what_installing_would_cost(
        client, fake_github):
    """Every field the consent screen draws, from the real ``inspect_github``
    over a manifest read at a resolved sha."""
    fake_github.answers(REPO_MANIFEST)

    body = await inspected(client, "alice/extras")

    assert list(body) == INSPECTION_KEYS
    assert body["kind"] == "github"
    assert body["mode"] == "install"
    assert body["plugin_id"] == "extras"
    assert body["catalog_id"] is None
    # "official" is a claim only the catalog is entitled to make.
    assert body["official"] is False
    assert body["source"] == "alice/extras"
    assert body["url"] == "https://github.com/alice/extras"
    assert body["sha"] == A_SHA
    assert body["name"] == "Extras"
    assert body["version"] == "1.2.0"
    assert body["homepage"] == "https://example.com/extras"
    assert body["capabilities"] == ["network"]
    assert body["allowed_modules"] == ["requests"]
    assert body["python_deps"] == {"tabulate": ">=0.9"}
    assert body["has_frontend"] is True
    assert body["chapters"] == ["C7"]
    assert body["lessons"] == ["intro"]
    assert body["consent_required"] is True
    assert body["manifest"]["plugin"]["id"] == "extras"
    # Both sentences, because they are two different decisions: browser code
    # runs in the editor with everything the editor can reach, and an
    # allowed module is the AST gate switched off by name.
    assert FRONTEND_WARNING in body["warnings"]
    assert ALLOWED_MODULES_WARNING.format(modules="requests") in body["warnings"]
    # The repository is SPLIT on the inspection the service keeps, because
    # ``download_tarball`` takes an owner and a repo. The wire contract does
    # not carry it, and a response built by dumping the dataclass would.
    assert "owner" not in body
    assert "repo" not in body
    # Read AT the sha, not at the ref: what the user is shown and what an
    # install would fetch have to be the same commit.
    assert fake_github.resolved == [("alice", "extras", "")]


async def test_an_update_shows_the_difference_not_the_total(
        client, fake_github):
    """``demo-external`` is installed having been granted one capability and
    one module; the repository now asks for more. What the user has to decide
    about is the DELTA, so it travels in its own fields."""
    fake_github.answers("""\
[plugin]
id = "demo-external"
name = "Demo External"
version = "3.0.0"
schema_version = 1

[security]
capabilities = ["network", "filesystem"]
allowed_modules = ["requests", "os"]
""")

    body = await inspected(client, "alice/extras")

    assert body["mode"] == "update"
    assert body["up_to_date"] is False
    assert body["installed"] == {
        "sha": "0" * 40, "version": "2.0.0", "capabilities": ["network"],
        "trusted_modules": ["requests"], "enabled": True,
        "source_kind": "github_url",
    }
    assert body["capabilities_added"] == ["filesystem"]
    assert body["allowed_modules_added"] == ["os"]


async def test_the_same_commit_that_is_installed_is_up_to_date(
        client, fake_github):
    fake_github.answers(a_manifest("demo-external"), sha="0" * 40)

    body = await inspected(client, "alice/extras")

    assert body["mode"] == "update"
    assert body["up_to_date"] is True
    assert body["capabilities_added"] == []


@pytest.mark.parametrize("body", [
    {},
    {"source": ""},
    {"source": "   "},
    {"source": "alice/extras", "sha": "deadbeef"},
    {"source": ["alice/extras"]},
])
async def test_a_request_the_schema_will_not_take_is_422(client, body):
    """``extra="forbid"`` is what makes "the client cannot hand the server a
    sha of its own" a property of the schema rather than of the handler."""
    response = await client.post("/api/plugins/inspect", json=body)
    assert response.status_code == 422, response.text


async def test_a_pasted_source_arrives_trimmed(client, fake_github):
    fake_github.answers(REPO_MANIFEST)
    body = await inspected(client, "  alice/extras\n")
    assert body["source"] == "alice/extras"


async def test_a_name_the_catalog_does_not_have_offers_the_ones_it_does(
        client):
    """The user did type a name; what they cannot see is which names this
    build has, so the refusal carries them."""
    response = await client.post("/api/plugins/inspect",
                                 json={"source": "no-such-pack"})
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]

    assert detail["code"] == "unknown_catalog_name"
    assert "stats" in detail["known"]


async def test_a_string_that_names_nothing_installable_is_unparseable(client):
    response = await client.post(
        "/api/plugins/inspect",
        json={"source": "https://evil.example.com/alice/extras"})
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == {"code": "unparseable_source"}


async def test_a_manifest_this_build_will_not_install_is_a_400(
        client, fake_github):
    fake_github.answers('[plugin]\nid = "extras"\nschema_version = 99\n')

    response = await client.post("/api/plugins/inspect",
                                 json={"source": "alice/extras"})
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == {"code": "invalid_manifest"}


async def test_a_manifest_that_is_not_even_toml_is_a_400_not_a_500(
        client, fake_github):
    """A captive portal, a proxy's block page, or a repository whose manifest
    somebody broke: the read never gets as far as validation."""
    fake_github.answers("<!doctype html>\n<html>not a manifest</html>\n")

    response = await client.post("/api/plugins/inspect",
                                 json={"source": "alice/extras"})
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == {"code": "invalid_manifest"}


async def test_a_manifest_that_is_not_utf8_is_a_400_not_a_500(
        client, fake_github):
    """A repository whose ``cdui.plugin.toml`` is UTF-16, or a binary file
    under that name. ``fetch_manifest_text`` lets the ``UnicodeDecodeError``
    out on purpose, and it is a ``ValueError`` that is neither a
    ``ManifestError`` nor a ``SourceError`` -- so this reached the client as
    a crash until the route named it."""
    fake_github.answers_bytes("[plugin]\nid = 'extras'\n".encode("utf-16"))

    response = await client.post("/api/plugins/inspect",
                                 json={"source": "alice/extras"})
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == {"code": "invalid_manifest"}


async def test_a_builtin_whose_manifest_is_gone_is_a_400_not_a_500(
        client, tmp_path, monkeypatch):
    """A release that shipped a catalog row and lost the directory behind it,
    or a pack somebody deleted by hand. ``read_manifest`` raises
    ``FileNotFoundError`` -- an ``OSError``, unrelated to every class the
    other clauses catch -- and the disk saying the source is not there is the
    same answer as a manifest that cannot be read."""
    builtin_root = tmp_path / "builtin"
    builtin_root.mkdir()
    (builtin_root / "registry.json").write_text(
        json.dumps({"schema": 1, "plugins": {"ghost": {
            "kind": "builtin", "name": "Ghost", "path": "plugins/ghost"}}}),
        encoding="utf-8")
    monkeypatch.setattr(plugin_loader, "plugins_builtin_root",
                        lambda: builtin_root)

    response = await client.post("/api/plugins/inspect",
                                 json={"source": "ghost"})
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == {"code": "invalid_manifest"}


async def test_an_id_this_build_owns_is_refused_with_the_id(
        client, fake_github):
    """``/api/plugins/catalog`` is a route, so no plugin may be called
    ``catalog``. The panel has to say WHICH id clashed, and the exception
    does not carry one -- this is what pins the route's reading of it."""
    fake_github.answers(a_manifest("catalog"))

    response = await client.post("/api/plugins/inspect",
                                 json={"source": "alice/extras"})
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == {"code": "reserved_id",
                                         "id": "catalog"}


async def test_a_catalog_row_too_broken_to_install_is_not_a_500(
        client, monkeypatch):
    """The name parses (the raw registry has it) and the validated catalog
    does not: a row this build cannot install from. Named but unusable is
    still the source's problem, not the server's."""
    monkeypatch.setattr(catalog_module, "load_catalog",
                        lambda: {"plugins": {"broken": {"kind": "builtin"}}})
    monkeypatch.setattr(catalog_module, "catalog_entries", dict)

    response = await client.post("/api/plugins/inspect",
                                 json={"source": "broken"})
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == {"code": "invalid_manifest"}


@pytest.mark.parametrize("status, expected, code", [
    (404, 404, "not_found"),
    (403, 502, "github_rate_limited"),
    (429, 502, "github_rate_limited"),
    (500, 502, "github_unreachable"),
    (None, 502, "github_unreachable"),
])
async def test_a_github_failure_is_split_by_what_it_means(
        client, fake_github, status, expected, code):
    """A typo is the caller's to fix and travels as a 404; everything else
    happened between this server and GitHub, and "wait" and "check the
    network" are different waits."""
    fake_github.raises(GitHubError("no", status=status))

    response = await client.post("/api/plugins/inspect",
                                 json={"source": "alice/extras"})
    assert response.status_code == expected, response.text
    assert response.json()["detail"] == {"code": code}


async def test_only_one_source_is_read_at_a_time(client, monkeypatch):
    """Refused rather than queued: a panel that inspects per keystroke would
    otherwise open a socket per keystroke, and an answer that arrives after
    the person has typed something else answers nothing."""
    sources = Sources()
    monkeypatch.setattr(inspect_module, "inspect_source", sources)
    sources.answer("alice/extras", inspect_module.inspect_builtin(
        "stats", lockfile=plugin_loader.load_lockfile()))
    sources.blocked.set()

    first = asyncio.create_task(
        client.post("/api/plugins/inspect", json={"source": "alice/extras"}))
    deadline = time.monotonic() + 10.0
    while not sources.entered.is_set():
        assert time.monotonic() < deadline, "the inspection never started"
        await asyncio.sleep(0.01)

    refused = await client.post("/api/plugins/inspect",
                                json={"source": "bob/other"})
    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"] == {"code": "inspect_busy"}

    sources.blocked.clear()
    assert (await first).status_code == 200


# -- POST /api/plugins/install ---------------------------------------------


async def test_installing_runs_a_job_that_ends_by_reloading_the_editor(
        client, flow, fake_github):
    """The whole event order the panel is written against, with the registry
    re-discovery as the last step -- on the loop, after the flow's thread and
    before the terminal event, so ``job_done`` can say what is now in the
    palette."""
    fake_github.answers(a_manifest("extras"))

    job_id = await start_install(client)
    await wait_started(flow)
    flow.send({"type": "step_started", "step": "download",
               "label": "Downloading extras"})
    flow.send({"type": "step_done", "step": "download"})
    flow.finish()

    events, status = await drain(client, job_id)

    assert status == "done"
    assert types_of(events) == ["job_started", "step_started", "step_done",
                               "step_started", "step_done", "job_done"]
    started = events[0]
    assert started["plugin_id"] == "extras"
    assert started["kind"] == "github"
    assert started["mode"] == "install"
    assert started["source"] == "alice/extras"
    assert events[3]["step"] == "reload"
    assert events[4]["step"] == "reload"
    done = events[-1]
    assert done["plugin_id"] == "extras"
    assert done["sha"] == A_SHA
    # Nothing was really installed, so nothing is in the palette -- but the
    # keys are what a panel reads to know that without asking again.
    assert done["nodes"] == []
    assert done["generation"] == plugin_loader.reload_generation()


async def test_a_failed_install_carries_its_message_and_its_hint(
        client, flow, fake_github):
    """And does NOT reload: the registry is rebuilt after a flow that
    returned, never after one that raised."""
    fake_github.answers(a_manifest("extras"))
    job_id = await start_install(client)
    await wait_started(flow)
    flow.fail(PluginInstallError("extras was refused by the security scan.",
                                 hint="nodes/evil.py imports os"))

    events, status = await drain(client, job_id)

    assert status == "failed"
    assert types_of(events) == ["job_started", "job_failed"]
    assert events[-1]["message"] == "extras was refused by the security scan."
    assert events[-1]["hint"] == "nodes/evil.py imports os"


async def test_an_install_that_cannot_finish_in_here_says_what_to_run(
        client, flow, fake_github):
    fake_github.answers(a_manifest("extras"))
    job_id = await start_install(client)
    await wait_started(flow)
    flow.fail(PluginNeedsRestart(
        "extras needs packages this server cannot install into itself.",
        command="cdui plugin install alice/extras",
        hint="uv could not resolve them add-only"))

    events, status = await drain(client, job_id)

    assert status == "needs_restart"
    assert events[-1]["type"] == "needs_restart"
    assert events[-1]["command"] == "cdui plugin install alice/extras"
    assert events[-1]["hint"] == "uv could not resolve them add-only"


@pytest.mark.parametrize("body", [
    {},
    {"inspection_id": "x", "sha": "deadbeef"},
    {"inspection_id": "x", "accept_capabilities": True},
    {"inspection_id": "x", "accept_capabilities": "network"},
])
async def test_an_install_body_the_schema_will_not_take_is_422(client, body):
    """``accept_capabilities`` is a LIST, and ``true`` is not a shorter way
    of saying it: a blanket yes cannot be checked against a manifest that has
    since grown a capability."""
    response = await client.post("/api/plugins/install", json=body)
    assert response.status_code == 422, response.text


async def test_a_capability_nobody_ticked_is_refused_with_the_list(
        client, fake_github):
    fake_github.answers(REPO_MANIFEST)
    inspection = await inspected(client, "alice/extras")

    response = await client.post(
        "/api/plugins/install",
        json={"inspection_id": inspection["inspection_id"]})

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == {"code": "consent_required",
                                         "missing_capabilities": ["network"]}


async def test_trusting_the_author_is_a_second_question_with_its_own_answer(
        client, fake_github):
    """Ticking "network" is not agreeing to let the plugin import
    ``requests``: the two are different decisions with different controls, so
    they are different codes -- and ``TrustAuthorRequired`` is a
    ``ConsentRequired``, which is why the route's catch order matters."""
    fake_github.answers(REPO_MANIFEST)
    inspection = await inspected(client, "alice/extras")

    response = await client.post(
        "/api/plugins/install",
        json={"inspection_id": inspection["inspection_id"],
              "accept_capabilities": ["network"]})

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == {"code": "trust_author_required",
                                         "allowed_modules": ["requests"]}


async def test_both_halves_answered_starts_the_job(client, flow, fake_github):
    fake_github.answers(REPO_MANIFEST)

    job_id = await start_install(client, accept_capabilities=["network"],
                                 trust_author=True)
    await wait_started(flow)
    flow.finish()
    events, status = await drain(client, job_id)

    assert status == "done"
    # What consent decided, as the install itself sees it.
    assert flow.plans[0].granted_capabilities == ("network",)
    assert flow.plans[0].trust_author is True
    assert types_of(events)[0] == "job_started"


async def test_an_inspection_this_server_does_not_remember_is_a_404(client):
    """Used, evicted or timed out are one answer, because the next move is
    the same for all three: inspect the source again."""
    response = await client.post("/api/plugins/install",
                                 json={"inspection_id": "no-such-id"})

    assert response.status_code == 404, response.text
    assert response.json()["detail"] == {"code": "inspection_expired",
                                         "inspection_id": "no-such-id"}


async def test_an_inspection_is_spent_by_the_install_it_starts(
        client, flow, fake_github):
    """One consent screen cannot start two installs."""
    fake_github.answers(a_manifest("extras"))
    inspection = await inspected(client, "alice/extras")
    body = {"inspection_id": inspection["inspection_id"]}

    first = await client.post("/api/plugins/install", json=body)
    assert first.status_code == 202, first.text
    again = await client.post("/api/plugins/install", json=body)

    assert again.status_code == 404, again.text
    assert again.json()["detail"]["code"] == "inspection_expired"

    flow.finish()
    await drain(client, first.json()["job_id"])


async def test_a_plugin_that_is_already_here_is_an_offer_not_a_job(
        client, flow, fake_github):
    """``demo-external`` is in the lockfile. Replacing it is a decision the
    user makes, so the refusal names the plugin -- and leaves the inspection
    spendable, which is what makes "Reinstall" one click rather than two."""
    fake_github.answers(a_manifest("demo-external"))
    inspection = await inspected(client, "alice/extras")
    body = {"inspection_id": inspection["inspection_id"]}

    refused = await client.post("/api/plugins/install", json=body)
    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"] == {"code": "already_installed",
                                        "plugin_id": "demo-external"}

    forced = await client.post("/api/plugins/install",
                               json={**body, "force": True})
    assert forced.status_code == 202, forced.text
    flow.finish()
    await drain(client, forced.json()["job_id"])


async def test_one_install_at_a_time_and_the_refusal_names_the_job(
        client, flow, fake_github):
    fake_github.answers(a_manifest("extras"))
    job_id = await start_install(client)
    await wait_started(flow)

    fake_github.answers(a_manifest("other"))
    second = await inspected(client, "bob/other")
    refused = await client.post(
        "/api/plugins/install",
        json={"inspection_id": second["inspection_id"]})

    assert refused.status_code == 409, refused.text
    # No ``reason``: this is our own job, and the id is one the panel can
    # follow rather than something to wait for.
    assert refused.json()["detail"] == {"code": "busy", "job_id": job_id}

    flow.finish()
    await drain(client, job_id)


async def test_a_pack_install_in_the_other_center_is_its_own_refusal(
        client, plugin_service, fake_github, monkeypatch):
    """They install into one interpreter, so each installer refuses while the
    other is running -- and the code says whose job the id belongs to,
    because "follow your install" and "wait for somebody else's" are
    different offers."""
    monkeypatch.setattr(plugin_service, "_busy_elsewhere", lambda: "pack-7")
    fake_github.answers(a_manifest("extras"))
    inspection = await inspected(client, "alice/extras")

    refused = await client.post(
        "/api/plugins/install",
        json={"inspection_id": inspection["inspection_id"]})

    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"] == {"code": "pack_install_running",
                                        "job_id": "pack-7"}


async def test_the_catalog_reports_the_job_a_second_tab_would_follow(
        client, flow):
    """A built-in install, so the row and the job are the same id: how a
    panel opened mid-install finds what to follow and which row is busy."""
    job_id = await start_install(client, "stats")
    await wait_started(flow)

    body = (await client.get("/api/plugins/catalog")).json()
    by_id = {entry["id"]: entry for entry in body["entries"]}

    # ``kind`` here is the job's MODE: what the panel decides between
    # "installed" and "updated" with. A job's own kind is builtin/github.
    assert body["active_job"] == {"job_id": job_id, "plugin_id": "stats",
                                  "kind": "install", "status": "running",
                                  "current_step": None}
    assert by_id["stats"]["status"] == "installing"
    assert by_id["stats"]["job"] == {"job_id": job_id, "status": "running",
                                     "current_step": None}
    assert by_id["foundations"]["job"] is None

    flow.finish()
    await drain(client, job_id)
    # A FINISHED job is not what the panel calls active: a row still carrying
    # one would spin until the next install.
    assert (await client.get("/api/plugins/catalog")).json()["active_job"] is None


# -- the job routes --------------------------------------------------------


async def test_the_events_route_answers_at_once_when_nothing_is_waiting(
        client, flow, fake_github):
    """``wait=0`` is the poll a panel makes when it is not following: it must
    not park, and an empty tail is a normal answer."""
    fake_github.answers(a_manifest("extras"))
    job_id = await start_install(client)
    await wait_started(flow)

    started_at = time.monotonic()
    first = await client.get(f"/api/plugins/jobs/{job_id}/events",
                             params={"cursor": 0, "wait": 0})
    assert first.status_code == 200, first.text
    page = first.json()
    assert set(page) == {"job_id", "status", "events", "cursor"}
    assert page["job_id"] == job_id
    assert page["status"] == "running"
    assert types_of(page["events"]) == ["job_started"]
    assert page["cursor"] == 1

    tail = await client.get(f"/api/plugins/jobs/{job_id}/events",
                            params={"cursor": page["cursor"], "wait": 0})
    assert tail.json()["events"] == []
    assert tail.json()["cursor"] == page["cursor"], "the cursor moved backwards"
    assert time.monotonic() - started_at < 3.0, "wait=0 parked"

    flow.finish()
    await drain(client, job_id)


@pytest.mark.parametrize("params", [
    {"wait": 61}, {"wait": -1}, {"cursor": -1}, {"limit": 0}, {"limit": 2001},
])
async def test_the_events_route_bounds_are_the_packs_route_bounds(
        client, flow, fake_github, params):
    """One panel draws both centers; a long poll that took a different range
    on one of them would be a bug found only in the half nobody watches."""
    fake_github.answers(a_manifest("extras"))
    job_id = await start_install(client)
    flow.finish()
    await drain(client, job_id)

    response = await client.get(f"/api/plugins/jobs/{job_id}/events",
                                params=params)
    assert response.status_code == 422, response.text


async def test_a_job_this_server_never_had_is_a_404_on_both_job_routes(
        client):
    """Only the most recent job is kept, so "gone" and "never existed" are
    the same answer -- and the client's next move is the same either way."""
    events = await client.get("/api/plugins/jobs/nope/events")
    assert events.status_code == 404, events.text
    assert events.json()["detail"] == {"code": "unknown_job", "job_id": "nope"}

    cancel = await client.post("/api/plugins/jobs/nope/cancel")
    assert cancel.status_code == 404, cancel.text
    assert cancel.json()["detail"] == {"code": "unknown_job", "job_id": "nope"}


async def test_cancelling_ends_the_job_and_asking_twice_is_not_an_error(
        client, flow, fake_github):
    fake_github.answers(a_manifest("extras"))
    job_id = await start_install(client)
    await wait_started(flow)

    first = await client.post(f"/api/plugins/jobs/{job_id}/cancel")
    assert first.status_code == 200, first.text
    assert first.json() == {"job_id": job_id, "cancelled": True}

    events, status = await drain(client, job_id)
    assert status == "cancelled"
    assert types_of(events)[-1] == "job_cancelled"

    again = await client.post(f"/api/plugins/jobs/{job_id}/cancel")
    assert again.status_code == 200, again.text
    assert again.json() == {"job_id": job_id, "cancelled": False}


# -- what guards the install path ------------------------------------------

#: Every route of the install path that CHANGES something, with a body the
#: schema accepts. Deliberately not a walk of the router: ``/reload`` and
#: ``/{id}/enable|disable`` are mutating too and are token-only on purpose,
#: because they act on code this machine already has.
#:
#: ``/update`` names ``foundations`` -- a built-in pack -- so that the walk
#: which lifts the loopback refusal stops at ``not_updatable`` instead of
#: driving a real inspection into the real network: this file fakes GitHub
#: per test, and a walk has no fixture of its own.
MUTATING = [
    ("POST", "/api/plugins/inspect", {"source": "stats"}),
    ("POST", "/api/plugins/install", {"inspection_id": "no-such-id"}),
    ("POST", "/api/plugins/jobs/no-such-job/cancel", None),
    ("POST", "/api/plugins/foundations/update", None),
]


@pytest.mark.parametrize("method, path, body", MUTATING)
async def test_every_route_that_installs_needs_the_session_token(
        anon_client, method, path, body):
    response = await anon_client.request(method, path, json=body)
    assert response.status_code == 403, f"{method} {path}"
    assert TOKEN_HEADER in response.json()["detail"]


@pytest.mark.parametrize("method, path, body", MUTATING)
async def test_a_remote_bind_may_not_install_unless_it_opts_in(
        client, monkeypatch, method, path, body):
    """A stranger on the LAN does not get to put third-party code where this
    process will import it. A classroom or office server that deliberately
    serves the LAN opts back in with one variable, which the refusal names."""
    monkeypatch.setattr(settings, "HOST", "192.168.1.20")
    monkeypatch.setattr(settings, "ALLOW_REMOTE_PLUGIN_INSTALL", False)

    refused = await client.request(method, path, json=body)
    assert refused.status_code == 403, f"{method} {path}"
    assert refused.json()["detail"] == (
        "Installing plugins is only allowed from the computer that runs the "
        "server. Set CODEFYUI_ALLOW_REMOTE_PLUGIN_INSTALL=1 to override.")

    monkeypatch.setattr(settings, "ALLOW_REMOTE_PLUGIN_INSTALL", True)
    allowed = await client.request(method, path, json=body)
    # Past the gate: whatever these ids and bodies deserve, not a 403.
    assert allowed.status_code != 403, f"{method} {path}"


@pytest.mark.parametrize("method, path, body", MUTATING + [
    # The events route too, which is otherwise open: it is the Package
    # Center's route body verbatim, and "this server has no installer" is a
    # truer answer to a follower than "that job never existed".
    ("GET", "/api/plugins/jobs/no-such-job/events", None),
])
async def test_a_server_with_no_installer_says_so_rather_than_guessing(
        uninstalled_client, method, path, body):
    response = await uninstalled_client.request(method, path, json=body)
    assert response.status_code == 503, f"{method} {path}"
    assert response.json()["detail"] == {"code": "unavailable"}


def test_the_request_models_do_not_rename_another_routers_schema():
    """Two routers with a model of the same name is not an error -- it is
    worse. FastAPI falls back to the module-qualified component name for BOTH
    of them, so declaring an ``InstallRequest`` here would quietly rename the
    Package Center's in ``/docs`` and in every client generated from
    ``/openapi.json``. The names are chosen so nothing moves."""
    schemas = app.openapi()["components"]["schemas"]

    assert "PluginInspectRequest" in schemas
    assert "PluginInstallRequest" in schemas
    assert "InstallRequest" in schemas, "the packs model was renamed"


async def test_the_catalog_still_answers_without_an_installer(
        uninstalled_client):
    """The contrast that makes the 503 above a decision: ``/catalog`` is a
    READ, and a server whose installer failed to start can still say what is
    installed."""
    response = await uninstalled_client.get("/api/plugins/catalog")
    assert response.status_code == 200, response.text
    assert response.json()["active_job"] is None
    assert response.json()["entries"]


async def test_a_plugin_can_be_removed_without_an_installer(
        uninstalled_client, center_lockfile, forgetting):
    """The other half of that decision, and the reason the delete route asks
    ``_installer`` rather than ``_service``.

    Removing a plugin is a lockfile edit, a purge and a re-discovery; none of
    the three belongs to the installer, and no installer means no job to be
    in the way. A 503 here would refuse to uninstall a plugin BECAUSE the
    thing that installs plugins had failed to start -- which is the state you
    would most want to get out of.
    """
    response = await uninstalled_client.delete("/api/plugins/demo-external")
    assert response.status_code == 200, response.text
    assert response.json()["removed"] is True
    assert "demo-external" not in lockfile_of(center_lockfile)["plugins"]


# ==========================================================================
# part 3: the lifecycle
# ==========================================================================
#
# ``DELETE /api/plugins/{id}`` and the two toggles, over the same lockfile
# part 1 draws its rows from. WHICH files an uninstall may touch is
# ``lifecycle.uninstall_plugin``'s rule and is pinned against the CLI in
# ``test_plugin_uninstall_builtin.py``; what is tested here is the route --
# the payload the panel is written against, the refusals, and the two calls
# that make this process forget a plugin it has already imported.

#: Every key ``DELETE /api/plugins/{id}`` answers with, in order. The
#: TypeScript ``PluginUninstallResult`` in ``frontend/src/api/rest.ts`` is
#: written against this list.
UNINSTALL_KEYS = ["id", "removed", "tombstoned", "files_removed",
                  "python_deps_left", "uninstall_command", "reinstall_hint"]

#: The lifecycle routes, behind the same two gates as the install path. Its
#: own list rather than three more entries in ``MUTATING``: the third walk
#: over that one asserts 503 when no installer is on ``app.state``, and a
#: delete deliberately does not need one -- removing a plugin is a lockfile
#: edit and a re-discovery, neither of which the installer owns.
LIFECYCLE = [("DELETE", "/api/plugins/demo-external", None)]


def lockfile_of(user_root: Path) -> dict:
    """The lockfile as it is ON DISK, not as some cache remembers it."""
    return json.loads(
        (user_root / "installed.json").read_text(encoding="utf-8"))


@pytest.fixture
def forgetting(monkeypatch) -> list[str]:
    """Record the two calls that make this process forget a plugin.

    Patched rather than run: the real ``rediscover_now`` clears the node
    registry every other test in this session shares (``conftest`` repairs
    it, at a cost), and what these tests are about is that the route makes
    both calls, in one order rather than the other. One test below
    deliberately does NOT take this fixture, so the real pair is exercised
    once -- a recorder can only prove the calls happen, not that they work.
    """
    calls: list[str] = []

    def purge(plugin_id: str) -> None:
        calls.append(f"purge:{plugin_id}")

    def rediscover() -> dict[str, int]:
        calls.append("rediscover")
        return {}

    monkeypatch.setattr(plugin_loader, "purge_plugin_modules", purge)
    # Bound into the route's module at import time, so that is where it has
    # to be replaced -- and ``_set_plugin_enabled`` reads the same name, so
    # the toggles stop re-discovering too.
    monkeypatch.setattr(routes_plugins, "rediscover_now", rediscover)
    return calls


# -- DELETE /api/plugins/{id} ----------------------------------------------


async def test_deleting_a_downloaded_plugin_takes_its_files_and_its_entry(
        client, center_lockfile):
    """A ``github_url`` plugin is a copy this install downloaded, so the copy
    goes with it. Its Python packages do NOT: uninstalling those from inside
    the process that imported them is how a running server ends up half
    loaded, so the answer names them and hands over the command instead.

    The one delete here that lets the REAL purge and re-discovery run -- a
    recorder can prove the calls happen, not that they work. This is the
    plugin to do it with: nothing has ever imported ``demo-external``, so the
    purge cannot take a namespace the rest of the session shares, and the
    generation asserted below is what a real re-discovery bumps.
    """
    assert (center_lockfile / "demo-external").is_dir()
    generation = plugin_loader.reload_generation()

    response = await client.delete("/api/plugins/demo-external")
    assert response.status_code == 200, response.text
    body = response.json()

    assert list(body) == UNINSTALL_KEYS
    assert body["id"] == "demo-external"
    assert body["removed"] is True
    # Nothing re-adds a plugin nobody's catalog lists, so there is nothing
    # for a tombstone to prevent.
    assert body["tombstoned"] is False
    assert body["files_removed"] is True
    assert body["python_deps_left"] == ["tabulate"]
    assert body["uninstall_command"].startswith("uv pip uninstall --python ")
    assert body["uninstall_command"].endswith(" tabulate")
    assert body["reinstall_hint"] == "cdui plugin install demo-external"

    assert not (center_lockfile / "demo-external").exists()
    assert "demo-external" not in lockfile_of(center_lockfile)["plugins"]
    # The editor polls this counter to learn that the palette moved, and only
    # a real re-discovery bumps it.
    assert plugin_loader.reload_generation() > generation


async def test_the_plugin_is_forgotten_before_the_palette_is_rebuilt(
        client, forgetting):
    """Its modules leave ``sys.modules`` and only THEN is the registry
    re-discovered.

    Nothing else ever drops those modules: a re-discovery rebuilds the
    namespaces of the plugins the lockfile still lists, and the one just
    deleted is not among them. And the re-discovery is what bumps the
    generation the editor polls -- so purging after it would announce a new
    palette at the one moment a client is guaranteed to come back and read,
    while the pack that has just been deleted was still importable here.
    """
    assert (await client.delete("/api/plugins/demo-external")
            ).status_code == 200
    assert forgetting == ["purge:demo-external", "rediscover"]


async def test_deleting_a_builtin_pack_keeps_its_files_and_tombstones_it(
        client, center_lockfile, forgetting):
    """A pack that ships in this release is repo code, not a download: the
    directory stays exactly where the release put it, and the removal is
    RECORDED so ``cdui plugin sync`` does not put back what the user just
    threw away (#175).

    Forgetting is recorded rather than done, and here that is not only about
    speed: ``conftest`` binds ``cdui_plugins.foundations`` for the whole
    session, so a real purge of THIS id would take the namespace every later
    pack test imports through -- the trap ``test_stats_pack`` avoids by
    installing its copy under a different id.
    """
    repo_dir = plugin_loader.plugins_builtin_root() / "foundations"

    response = await client.delete("/api/plugins/foundations")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["removed"] is True
    assert body["tombstoned"] is True
    # ``None``, not ``False``: there was never a copy of ours to delete,
    # which is a different thing from having tried and failed.
    assert body["files_removed"] is None
    assert body["python_deps_left"] == []
    assert body["uninstall_command"] is None

    assert repo_dir.is_dir()
    assert (repo_dir / "cdui.plugin.toml").is_file()
    data = lockfile_of(center_lockfile)
    assert "foundations" not in data["plugins"]
    assert plugin_loader.removed_ids(data) == {"rl", "foundations"}
    assert forgetting == ["purge:foundations", "rediscover"]


async def test_unlinking_a_local_plugin_leaves_the_authors_checkout_alone(
        client, center_lockfile, tmp_path, forgetting):
    """``cdui plugin link`` records a path into somebody's working tree. The
    link goes; the tree is not ours to delete, and no tombstone is left --
    nothing re-adds a link uninvited."""
    work = tmp_path / "work" / "my-pack"
    (work / "nodes").mkdir(parents=True)
    (work / "cdui.plugin.toml").write_text(
        '[plugin]\nid = "my-pack"\nname = "Mine"\nversion = "0.1.0"\n'
        'schema_version = 1\n', encoding="utf-8")
    data = lockfile_of(center_lockfile)
    data["plugins"]["my-pack"] = {"source_kind": "local", "source": str(work),
                                  "path": str(work), "enabled": True}
    (center_lockfile / "installed.json").write_text(json.dumps(data),
                                                    encoding="utf-8")

    response = await client.delete("/api/plugins/my-pack")
    assert response.status_code == 200, response.text
    assert response.json()["files_removed"] is None
    assert response.json()["tombstoned"] is False

    assert (work / "cdui.plugin.toml").is_file()
    after = lockfile_of(center_lockfile)
    assert "my-pack" not in after["plugins"]
    assert plugin_loader.removed_ids(after) == {"rl"}


async def test_a_row_whose_files_are_already_gone_can_still_be_removed(
        client, center_lockfile, forgetting):
    """``ghost-pack`` is the ``missing_files`` state part 1 draws: an entry
    whose directory is not there any more. An absent directory is SUCCESS --
    nothing of ours is left, which is all "removed" ever meant -- so the
    panel's Remove button clears the row instead of reporting a conflict
    about files nobody can find."""
    assert not (center_lockfile / "ghost-pack").exists()

    response = await client.delete("/api/plugins/ghost-pack")
    assert response.status_code == 200, response.text
    assert response.json()["files_removed"] is True
    assert response.json()["python_deps_left"] == []
    assert "ghost-pack" not in lockfile_of(center_lockfile)["plugins"]


async def test_deleting_something_that_is_not_installed_is_a_404(
        client, forgetting):
    response = await client.delete("/api/plugins/no-such-plugin")
    assert response.status_code == 404, response.text
    assert response.json()["detail"] == {"code": "not_installed"}
    assert forgetting == []


async def test_files_that_will_not_delete_keep_the_plugin_installed(
        client, center_lockfile, monkeypatch, forgetting):
    """Windows holding one node file open is the ordinary cause. The lockfile
    entry is left alone, because a pack whose files are still there loads
    again on the next start -- calling it uninstalled would be a lie the
    lockfile then tells forever -- and this process is not told to forget it.

    ``error`` is the operating system's own sentence and ``hint`` names the
    directory that is still there, which is the half the user can act on.
    """
    def refuse(*args, **kwargs):
        raise OSError(32, "The process cannot access the file")

    # The one ``rmtree`` in the plugin system is ``lifecycle``'s, and it
    # reaches it through this module object.
    monkeypatch.setattr(shutil, "rmtree", refuse)

    response = await client.delete("/api/plugins/demo-external")
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]

    assert detail["code"] == "files_locked"
    assert "cannot access the file" in detail["error"]
    assert str(center_lockfile / "demo-external") in detail["hint"]

    assert "demo-external" in lockfile_of(center_lockfile)["plugins"]
    assert (center_lockfile / "demo-external").is_dir()
    assert forgetting == []


# -- POST /api/plugins/{id}/update -----------------------------------------
#
# ``demo-external`` is the row this section works on: a ``github_url`` entry
# recording one capability, one trusted module and the commit ``0`` * 40 at
# ``alice/extras@v1.2.3``. What the repository answers is faked per test, so
# each of the three outcomes is one manifest away from the others.

#: What the repository has grown into by the next commit, asking for exactly
#: what the lockfile already records. The ordinary update: a new sha and
#: nothing new to decide.
SAME_TERMS_MANIFEST = """\
[plugin]
id = "demo-external"
name = "Demo External"
version = "2.1.0"
schema_version = 1

[security]
capabilities = ["network"]
allowed_modules = ["requests"]
"""


async def test_the_commit_that_is_installed_leaves_nothing_to_do(
        client, flow, fake_github):
    """And the repository it asked is the one the LOCKFILE recorded, at the
    ref recorded with it -- an update follows what the user installed from,
    not something re-typed."""
    fake_github.answers(a_manifest("demo-external"), sha="0" * 40)

    response = await client.post("/api/plugins/demo-external/update")

    assert response.status_code == 200, response.text
    assert response.json() == {"status": "up_to_date", "sha": "0" * 40}
    assert fake_github.resolved == [("alice", "extras", "v1.2.3")]
    assert not flow.started.is_set()


async def test_an_update_the_user_already_consented_to_starts_a_job(
        client, flow, fake_github):
    """One click, one job. The plan carries ``force`` because pressing Update
    IS the offer to replace the copy on disk -- without it the install this
    route just started would be refused for being already installed."""
    fake_github.answers(SAME_TERMS_MANIFEST, sha="c" * 40)

    response = await client.post("/api/plugins/demo-external/update")

    assert response.status_code == 202, response.text
    assert list(response.json()) == ["job_id"]
    job_id = response.json()["job_id"]
    await wait_started(flow)
    plan = flow.plans[0]
    assert plan.mode == "update"
    assert plan.force is True
    assert plan.sha == "c" * 40
    # Granted by the previous install rather than by this request: nobody was
    # asked anything, and the capability still reaches the plan.
    assert plan.granted_capabilities == ("network",)

    flow.finish()
    events, status = await drain(client, job_id)
    assert status == "done"
    assert events[0]["mode"] == "update"


async def test_a_version_that_asks_for_more_comes_back_as_a_consent_screen(
        client, flow, fake_github):
    """The delta is the whole content of an update's dialog, so it travels
    twice: inside the inspection and beside it, from the same payload.

    Then the second turn, which is the point of storing it: the client sends
    the id and the answers to what it was asked and NOTHING else -- no force
    -- and is not refused for the replacement it already agreed to.
    """
    fake_github.answers(EXTERNAL_MANIFEST, sha="c" * 40)

    response = await client.post("/api/plugins/demo-external/update")

    assert response.status_code == 200, response.text
    body = response.json()
    assert list(body) == ["status", "inspection", "capabilities_added",
                          "allowed_modules_added"]
    assert body["status"] == "needs_consent"
    assert list(body["inspection"]) == INSPECTION_KEYS
    assert body["inspection"]["mode"] == "update"
    assert body["inspection"]["installed"]["sha"] == "0" * 40
    assert body["capabilities_added"] == ["filesystem"]
    assert body["allowed_modules_added"] == ["pathlib"]
    assert not flow.started.is_set(), "nothing may be installed unasked"

    accepted = await client.post("/api/plugins/install", json={
        "inspection_id": body["inspection"]["inspection_id"],
        "accept_capabilities": ["network", "filesystem"],
        "trust_author": True})

    assert accepted.status_code == 202, accepted.text
    await wait_started(flow)
    assert flow.plans[0].force is True
    flow.finish()
    _, status = await drain(client, accepted.json()["job_id"])
    assert status == "done"


async def test_an_update_keeps_the_catalog_row_the_install_recorded(
        client, flow, fake_github, center_lockfile):
    """``cdui plugin install <name>`` writes down which catalog row a plugin
    came from precisely so a later reader can tell the catalog's own pack
    from free text carrying the same id. An update is the same plugin seen
    again, so the row travels with it -- dropping it here made every update
    of an official plugin read as third-party."""
    data = lockfile_of(center_lockfile)
    data["plugins"]["demo-external"]["catalog_id"] = "demo-external"
    (center_lockfile / "installed.json").write_text(json.dumps(data),
                                                    encoding="utf-8")
    fake_github.answers(SAME_TERMS_MANIFEST, sha="c" * 40)

    response = await client.post("/api/plugins/demo-external/update")

    assert response.status_code == 202, response.text
    await wait_started(flow)
    assert flow.plans[0].catalog_id == "demo-external"

    flow.finish()
    await drain(client, response.json()["job_id"])


async def test_a_builtin_pack_updates_with_codefyui_itself(client):
    """It updates -- just not from here, and the hint is the only part of
    this refusal a person can act on."""
    response = await client.post("/api/plugins/foundations/update")

    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "not_updatable"
    assert "cdui update" in detail["hint"]


async def test_a_linked_directory_has_nothing_to_fetch(
        client, center_lockfile, tmp_path):
    """``cdui plugin link`` records a path into somebody's working tree: what
    is there is whatever its author saved a moment ago."""
    work = tmp_path / "work" / "my-pack"
    data = lockfile_of(center_lockfile)
    data["plugins"]["my-pack"] = {"source_kind": "local", "source": str(work),
                                  "path": str(work), "enabled": True}
    (center_lockfile / "installed.json").write_text(json.dumps(data),
                                                    encoding="utf-8")

    response = await client.post("/api/plugins/my-pack/update")

    assert response.status_code == 400, response.text
    assert response.json()["detail"]["code"] == "not_updatable"


async def test_updating_something_that_is_not_installed_is_a_404(client):
    response = await client.post("/api/plugins/no-such-plugin/update")

    assert response.status_code == 404, response.text
    assert response.json()["detail"] == {"code": "not_installed"}


async def test_an_update_waits_for_the_install_that_is_running(
        client, flow, fake_github):
    """Wider than the delete's guard, and deliberately: only one install runs
    at a time, so an update of ANY plugin waits for it. Refused before the
    network, because the round trip would end at this same 409."""
    fake_github.answers(a_manifest("extras"))
    job_id = await start_install(client)
    await wait_started(flow)
    reads = len(fake_github.resolved)

    refused = await client.post("/api/plugins/demo-external/update")

    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"] == {"code": "busy", "job_id": job_id}
    assert len(fake_github.resolved) == reads, "GitHub was asked anyway"

    flow.finish()
    await drain(client, job_id)


async def test_an_update_is_refused_while_another_source_is_being_read(
        client, monkeypatch):
    """An update IS an inspection -- of a source the server looked up instead
    of the user -- so it takes the same one-at-a-time slot."""
    sources = Sources()
    monkeypatch.setattr(inspect_module, "inspect_source", sources)
    sources.answer("alice/extras", inspect_module.inspect_builtin(
        "stats", lockfile=plugin_loader.load_lockfile()))
    sources.blocked.set()
    reading = asyncio.create_task(
        client.post("/api/plugins/inspect", json={"source": "alice/extras"}))
    deadline = time.monotonic() + 10.0
    while not sources.entered.is_set():
        assert time.monotonic() < deadline, "the inspection never started"
        await asyncio.sleep(0.01)

    refused = await client.post("/api/plugins/demo-external/update")

    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"] == {"code": "inspect_busy"}

    sources.blocked.clear()
    assert (await reading).status_code == 200


@pytest.mark.parametrize("status, expected, code", [
    (404, 404, "not_found"),
    (403, 502, "github_rate_limited"),
    (500, 502, "github_unreachable"),
])
async def test_a_github_failure_on_an_update_is_split_the_same_way(
        client, fake_github, status, expected, code):
    """The repository behind an installed plugin can be renamed, made private
    or simply be unreachable, and the three are different waits."""
    fake_github.raises(GitHubError("no", status=status))

    response = await client.post("/api/plugins/demo-external/update")

    assert response.status_code == expected, response.text
    assert response.json()["detail"] == {"code": code}


async def test_a_repository_that_now_declares_a_reserved_id_is_refused(
        client, fake_github):
    """The plugin was installed as ``demo-external`` and its repository has
    since renamed itself onto an id this build owns. Nothing is fetched, and
    the panel is told which id clashed."""
    fake_github.answers(a_manifest("catalog"), sha="c" * 40)

    response = await client.post("/api/plugins/demo-external/update")

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == {"code": "reserved_id",
                                         "id": "catalog"}


async def test_an_update_of_a_row_whose_manifest_is_not_one_is_a_400(
        client, fake_github):
    fake_github.answers("<!doctype html>\n<html>not a manifest</html>\n")

    response = await client.post("/api/plugins/demo-external/update")

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == {"code": "invalid_manifest"}


# -- the busy guard --------------------------------------------------------


async def test_a_delete_waits_for_that_plugins_own_install(
        client, flow, fake_github, forgetting):
    """An install writes the plugin's directory and the lockfile entry beside
    it; a delete removes both. Refused while that flow runs, and refused
    BEFORE "not installed" is considered -- the pack being written is not in
    the lockfile yet, and "wait for job X" is the answer to "remove it" that
    a user can act on, where "you do not have it" invites them to press the
    button again into the same race."""
    fake_github.answers(a_manifest("extras"))
    job_id = await start_install(client)
    await wait_started(flow)

    refused = await client.delete("/api/plugins/extras")
    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"] == {"code": "busy", "job_id": job_id}
    assert forgetting == []

    flow.finish()
    await drain(client, job_id)

    # The job is terminal, so the guard is open again -- and the scripted
    # flow wrote nothing, so what is left is the honest 404.
    assert (await client.delete("/api/plugins/extras")).status_code == 404


async def test_another_plugins_install_does_not_block_the_lifecycle(
        client, flow, fake_github, center_lockfile, forgetting):
    """One install at a time is a rule about the INSTALLER -- two ``uv pip
    install`` runs share one site-packages -- not about the lockfile. Two
    plugins are two directories and two keys, so a long download of one must
    not freeze every other row in the panel."""
    fake_github.answers(a_manifest("extras"))
    job_id = await start_install(client)
    await wait_started(flow)

    removed = await client.delete("/api/plugins/demo-external")
    assert removed.status_code == 200, removed.text
    assert "demo-external" not in lockfile_of(center_lockfile)["plugins"]

    toggled = await client.post("/api/plugins/deep/enable")
    assert toggled.status_code == 200, toggled.text

    flow.finish()
    await drain(client, job_id)


@pytest.mark.parametrize("action, enabled", [("enable", True),
                                             ("disable", False)])
async def test_a_toggle_waits_for_that_plugins_own_install(
        client, flow, fake_github, forgetting, action, enabled):
    """``enable`` and ``disable`` rewrite the very entry the flow is about to
    write. They stay token-only -- flipping a flag on code this machine
    already has, and the user already agreed to, is not the question the
    loopback gate asks -- but they do not get to race the install."""
    fake_github.answers(a_manifest("demo-external"), sha="b" * 40)
    job_id = await start_install(client, force=True)
    await wait_started(flow)

    refused = await client.post(f"/api/plugins/demo-external/{action}")
    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"] == {"code": "busy", "job_id": job_id}

    flow.finish()
    await drain(client, job_id)

    allowed = await client.post(f"/api/plugins/demo-external/{action}")
    assert allowed.status_code == 200, allowed.text
    assert allowed.json() == {"id": "demo-external", "enabled": enabled}


# -- what guards the lifecycle path ----------------------------------------


def test_the_delete_route_is_declared_with_the_other_plugin_id_routes():
    """The same invariant as the GET's, one method along.

    A path that matches with the wrong method is a PARTIAL match, and the
    router keeps scanning for a full one -- so what a late fixed path loses
    to ``/{plugin_id}`` here is narrower than it is for the GET: only a fixed
    path that ALSO answers DELETE would be shadowed. The order is asserted
    anyway rather than reasoned about per method, because the next route
    added below this line is the one nobody re-derives it for."""
    api_routes = [route for route in routes_plugins.router.routes
                  if isinstance(route, APIRoute)]
    deleting = [index for index, route in enumerate(api_routes)
                if "DELETE" in route.methods]

    assert [api_routes[i].path for i in deleting] == [
        "/api/plugins/{plugin_id}"]
    for index, route in enumerate(api_routes):
        if "{" not in route.path:
            assert index < deleting[0], f"{route.path} is declared too late"


@pytest.mark.parametrize("method, path, body", LIFECYCLE)
async def test_every_lifecycle_route_needs_the_session_token(
        anon_client, method, path, body):
    response = await anon_client.request(method, path, json=body)
    assert response.status_code == 403, f"{method} {path}"
    assert TOKEN_HEADER in response.json()["detail"]


@pytest.mark.parametrize("method, path, body", LIFECYCLE)
async def test_a_remote_bind_may_not_change_what_is_installed(
        client, monkeypatch, forgetting, method, path, body):
    """Taking somebody's plugin away is the same question as putting one
    there: the server only takes it from the machine it runs on, and a
    classroom or office server that deliberately serves the LAN opts back in
    with the one variable the refusal names."""
    monkeypatch.setattr(settings, "HOST", "192.168.1.20")
    monkeypatch.setattr(settings, "ALLOW_REMOTE_PLUGIN_INSTALL", False)

    refused = await client.request(method, path, json=body)
    assert refused.status_code == 403, f"{method} {path}"
    assert refused.json()["detail"] == (
        "Installing plugins is only allowed from the computer that runs the "
        "server. Set CODEFYUI_ALLOW_REMOTE_PLUGIN_INSTALL=1 to override.")

    monkeypatch.setattr(settings, "ALLOW_REMOTE_PLUGIN_INSTALL", True)
    allowed = await client.request(method, path, json=body)
    # Past the gate: whatever this route deserves, not a 403.
    assert allowed.status_code != 403, f"{method} {path}"


async def test_the_toggles_stay_token_only(client, monkeypatch, forgetting):
    """The contrast that makes the gate above a decision. Enabling a plugin
    runs code that is already here and was already consented to, so a server
    deliberately bound to a LAN keeps answering -- the same reason
    ``/reload`` carries no loopback gate either."""
    monkeypatch.setattr(settings, "HOST", "192.168.1.20")
    monkeypatch.setattr(settings, "ALLOW_REMOTE_PLUGIN_INSTALL", False)

    response = await client.post("/api/plugins/deep/enable")
    assert response.status_code == 200, response.text
    assert response.json() == {"id": "deep", "enabled": True}


# -- the reserved id, off the exception ------------------------------------


async def test_a_reserved_id_is_read_off_the_exception_not_its_message(
        client, monkeypatch):
    """The route used to recover the id with a regular expression over the
    message, which quietly made the wording of an English sentence part of
    this wire contract. Raised here with a message that names nothing, so
    only the attribute can answer -- the regex reading would fall through to
    ``invalid_manifest`` and leave the panel with no id to show."""
    def refuse(spec: str, *, lockfile: dict):
        raise ReservedPluginId("that name is taken.", plugin_id="catalog",
                               taken_by="a route under /api/plugins/")

    monkeypatch.setattr(inspect_module, "inspect_source", refuse)

    response = await client.post("/api/plugins/inspect",
                                 json={"source": "alice/extras"})
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == {"code": "reserved_id",
                                         "id": "catalog"}
