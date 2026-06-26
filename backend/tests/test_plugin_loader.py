"""Unit tests for the plugin loader: synthetic namespace, lockfile, discovery."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from app.core import plugin_loader


@pytest.fixture(autouse=True)
def _clean_synthetic_namespace():
    """Strip ``cdui_plugins.*`` from sys.modules before & after every test."""
    plugin_loader.purge_all_plugin_modules()
    yield
    plugin_loader.purge_all_plugin_modules()


@pytest.fixture
def isolated_lockfile(tmp_path, monkeypatch):
    """Point lockfile_path() at a temp dir so tests don't touch real user data."""
    target = tmp_path / "plugins"
    target.mkdir()
    monkeypatch.setattr(plugin_loader, "plugins_user_root", lambda: target)
    return target


def _write_plugin(plugin_dir: Path, plugin_id: str, with_nodes: bool = True) -> None:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "cdui.plugin.toml").write_text(
        dedent(f"""\
            [plugin]
            id = "{plugin_id}"
            name = "Test pack {plugin_id}"
            version = "0.0.1"
            description = ""
            schema_version = 1
            """),
        encoding="utf-8",
    )
    if with_nodes:
        nodes = plugin_dir / "nodes"
        nodes.mkdir()
        (nodes / "__init__.py").write_text("", encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# load_lockfile / save_lockfile
# ──────────────────────────────────────────────────────────────────────

def test_load_lockfile_returns_empty_default_when_missing(isolated_lockfile):
    data = plugin_loader.load_lockfile()
    assert data == {"schema": plugin_loader.LOCKFILE_SCHEMA, "plugins": {}}


def test_load_lockfile_returns_empty_default_when_corrupt(isolated_lockfile):
    plugin_loader.lockfile_path().write_text("not json at all{{", encoding="utf-8")
    data = plugin_loader.load_lockfile()
    assert data == {"schema": plugin_loader.LOCKFILE_SCHEMA, "plugins": {}}


def test_load_lockfile_returns_empty_default_when_missing_plugins_key(isolated_lockfile):
    plugin_loader.lockfile_path().write_text(json.dumps({"schema": 1}), encoding="utf-8")
    data = plugin_loader.load_lockfile()
    assert "plugins" in data
    assert data["plugins"] == {}


def test_save_then_load_roundtrip(isolated_lockfile):
    payload = {
        "schema": plugin_loader.LOCKFILE_SCHEMA,
        "plugins": {
            "c2": {
                "source_kind": "builtin",
                "source": "c2",
                "installed_at": "2026-05-16T00:00:00Z",
            }
        },
    }
    plugin_loader.save_lockfile(payload)
    assert plugin_loader.load_lockfile() == payload


# ──────────────────────────────────────────────────────────────────────
# install_plugin_finder
# ──────────────────────────────────────────────────────────────────────

def test_install_plugin_finder_returns_empty_when_no_plugins(tmp_path):
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    builtin.mkdir()
    user.mkdir()
    pairs = plugin_loader.install_plugin_finder(builtin, user, {"plugins": {}})
    assert pairs == []


def test_install_plugin_finder_registers_synthetic_namespace(tmp_path):
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    plugin_dir = builtin / "c2"
    user.mkdir()
    _write_plugin(plugin_dir, "c2")
    lockfile = {"plugins": {"c2": {"source_kind": "builtin", "source": "c2"}}}

    pairs = plugin_loader.install_plugin_finder(builtin, user, lockfile)

    assert "cdui_plugins" in sys.modules
    assert "cdui_plugins.c2" in sys.modules
    assert len(pairs) == 1
    nodes_dir, pkg_name = pairs[0]
    assert nodes_dir == plugin_dir / "nodes"
    assert pkg_name == "cdui_plugins.c2.nodes"


def test_install_plugin_finder_resolves_user_packs_from_user_root(tmp_path):
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    builtin.mkdir()
    plugin_dir = user / "alice-extras"
    _write_plugin(plugin_dir, "alice-extras")
    lockfile = {
        "plugins": {
            "alice-extras": {
                "source_kind": "github_url",
                "source": "https://github.com/alice/cdui-extras",
            }
        }
    }

    pairs = plugin_loader.install_plugin_finder(builtin, user, lockfile)

    # kebab-case "alice-extras" → snake-case "alice_extras" for the Python module name
    assert "cdui_plugins.alice_extras" in sys.modules
    assert len(pairs) == 1
    assert pairs[0][1] == "cdui_plugins.alice_extras.nodes"


def test_install_plugin_finder_skips_plugins_without_manifest(tmp_path):
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    user.mkdir()
    (builtin / "c2").mkdir(parents=True)  # No cdui.plugin.toml inside
    lockfile = {"plugins": {"c2": {"source_kind": "builtin", "source": "c2"}}}

    pairs = plugin_loader.install_plugin_finder(builtin, user, lockfile)
    assert pairs == []


def test_install_plugin_finder_skips_plugin_without_nodes_dir(tmp_path):
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    user.mkdir()
    _write_plugin(builtin / "c5", "c5", with_nodes=False)
    lockfile = {"plugins": {"c5": {"source_kind": "builtin", "source": "c5"}}}

    pairs = plugin_loader.install_plugin_finder(builtin, user, lockfile)
    # Namespace package still registered (so a future nodes/ becomes importable)
    assert "cdui_plugins.c5" in sys.modules
    # But no discovery pair because there's nothing to walk
    assert pairs == []


def test_install_plugin_finder_resolves_local_path_entries(tmp_path):
    """A linked (source_kind='local') plugin loads from its recorded ``path``,
    which may live outside both the builtin and user roots — e.g. the author's
    own working checkout, the way ``cdui plugin link`` registers it."""
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    builtin.mkdir()
    user.mkdir()
    work = tmp_path / "work" / "my-plugin"  # outside both roots
    _write_plugin(work, "my-plugin")
    lockfile = {
        "plugins": {
            "my-plugin": {
                "source_kind": "local",
                "source": str(work),
                "path": str(work),
            }
        }
    }

    pairs = plugin_loader.install_plugin_finder(builtin, user, lockfile)

    assert "cdui_plugins.my_plugin" in sys.modules
    assert len(pairs) == 1
    nodes_dir, pkg_name = pairs[0]
    assert nodes_dir == work / "nodes"
    assert pkg_name == "cdui_plugins.my_plugin.nodes"


def test_install_plugin_finder_local_without_path_is_skipped(tmp_path):
    """A malformed local entry missing ``path`` must not crash discovery —
    it falls back to a location with no manifest and is skipped silently."""
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    builtin.mkdir()
    user.mkdir()
    lockfile = {"plugins": {"broken": {"source_kind": "local"}}}

    pairs = plugin_loader.install_plugin_finder(builtin, user, lockfile)
    assert pairs == []


# ──────────────────────────────────────────────────────────────────────
# purge_plugin_modules
# ──────────────────────────────────────────────────────────────────────

def test_purge_plugin_modules_removes_synthetic_namespace(tmp_path):
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    _write_plugin(builtin / "c2", "c2")
    user.mkdir()
    lockfile = {"plugins": {"c2": {"source_kind": "builtin", "source": "c2"}}}
    plugin_loader.install_plugin_finder(builtin, user, lockfile)
    assert "cdui_plugins.c2" in sys.modules

    plugin_loader.purge_plugin_modules("c2")
    assert "cdui_plugins.c2" not in sys.modules
    # Submodules under the prefix are removed too
    for name in sys.modules:
        assert not name.startswith("cdui_plugins.c2")


def test_purge_plugin_modules_handles_kebab_case(tmp_path):
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    builtin.mkdir()
    _write_plugin(user / "alice-extras", "alice-extras")
    lockfile = {"plugins": {"alice-extras": {"source_kind": "github_url", "source": "x"}}}
    plugin_loader.install_plugin_finder(builtin, user, lockfile)
    assert "cdui_plugins.alice_extras" in sys.modules

    plugin_loader.purge_plugin_modules("alice-extras")
    assert "cdui_plugins.alice_extras" not in sys.modules


# ──────────────────────────────────────────────────────────────────────
# iter_plugin_dirs
# ──────────────────────────────────────────────────────────────────────

def test_iter_plugin_dirs_yields_only_plugins_with_manifest(tmp_path):
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    user.mkdir()
    _write_plugin(builtin / "c2", "c2")
    (builtin / "ghost").mkdir(parents=True)  # in lockfile but no manifest
    lockfile = {
        "plugins": {
            "c2": {"source_kind": "builtin", "source": "c2"},
            "ghost": {"source_kind": "builtin", "source": "ghost"},
        }
    }

    out = plugin_loader.iter_plugin_dirs(builtin, user, lockfile)
    ids = [pid for pid, _ in out]
    assert ids == ["c2"]


def test_iter_plugin_dirs_resolves_local_path_entries(tmp_path):
    """iter_plugin_dirs (used by examples / assets / presets / the plugin list)
    resolves a linked plugin from its recorded ``path``."""
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    builtin.mkdir()
    user.mkdir()
    work = tmp_path / "elsewhere" / "loc"
    _write_plugin(work, "loc")
    lockfile = {"plugins": {"loc": {"source_kind": "local", "path": str(work)}}}

    out = plugin_loader.iter_plugin_dirs(builtin, user, lockfile)
    assert out == [("loc", work)]
