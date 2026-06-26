"""Unit tests for scripts/plugins.py — source parsing, manifest validation, lockfile ops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent

import pytest

import plugins as plugin_cli
from app.core import plugin_loader
from app.core.plugin_validator import PluginValidationError


# ── parse_source ───────────────────────────────────────────────────────────

def test_parse_source_catalog_case_insensitive():
    # 'foundations' / 'deep' / 'rl' live in plugins/registry.json
    assert plugin_cli.parse_source("foundations") == ("catalog", "foundations", "", "")
    assert plugin_cli.parse_source("Foundations") == ("catalog", "foundations", "", "")
    assert plugin_cli.parse_source("RL") == ("catalog", "rl", "", "")


def test_parse_source_github_short_no_ref():
    assert plugin_cli.parse_source("alice/extras") == ("github", "alice", "extras", "")


def test_parse_source_github_short_with_ref():
    assert plugin_cli.parse_source("alice/extras@v1.2.3") == (
        "github", "alice", "extras", "v1.2.3",
    )


def test_parse_source_github_full_url():
    assert plugin_cli.parse_source("https://github.com/alice/extras") == (
        "github", "alice", "extras", "",
    )
    assert plugin_cli.parse_source("https://github.com/alice/extras.git") == (
        "github", "alice", "extras", "",
    )
    assert plugin_cli.parse_source("http://www.github.com/alice/extras") == (
        "github", "alice", "extras", "",
    )


def test_parse_source_rejects_garbage():
    with pytest.raises(ValueError):
        plugin_cli.parse_source("not a valid source spec")


# ── validate_manifest ──────────────────────────────────────────────────────

def _good_manifest(plugin_id: str = "test-pack") -> dict:
    return {"plugin": {"id": plugin_id, "name": "Test", "version": "0.0.1", "schema_version": 1}}


def test_validate_manifest_accepts_well_formed():
    plugin_cli.validate_manifest(_good_manifest())


def test_validate_manifest_rejects_missing_plugin_table():
    with pytest.raises(ValueError, match="\\[plugin\\]"):
        plugin_cli.validate_manifest({"content": {"nodes_dir": "nodes"}})


def test_validate_manifest_rejects_unknown_schema():
    bad = _good_manifest()
    bad["plugin"]["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        plugin_cli.validate_manifest(bad)


def test_validate_manifest_rejects_uppercase_id():
    bad = _good_manifest("CapsId")
    with pytest.raises(ValueError, match="Invalid plugin id"):
        plugin_cli.validate_manifest(bad)


def test_validate_manifest_rejects_trailing_dash_id():
    bad = _good_manifest("trailing-")
    with pytest.raises(ValueError, match="Invalid plugin id"):
        plugin_cli.validate_manifest(bad)


# ── validate_nodes_dir uses the AST validator on every .py ────────────────

def test_validate_nodes_dir_passes_clean_code(tmp_path):
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    (nodes / "__init__.py").write_text("", encoding="utf-8")
    (nodes / "ok.py").write_text(
        "from app.core.node_base import BaseNode\n"
        "class X(BaseNode):\n"
        "    NODE_NAME = 'X'\n"
        "    CATEGORY = 'Test'\n"
        "    DESCRIPTION = ''\n",
        encoding="utf-8",
    )
    plugin_cli.validate_nodes_dir(nodes, allowed_modules=[])


def test_validate_nodes_dir_rejects_dangerous_import(tmp_path):
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    (nodes / "bad.py").write_text("import os\nos.system('whoami')\n", encoding="utf-8")
    with pytest.raises(PluginValidationError):
        plugin_cli.validate_nodes_dir(nodes, allowed_modules=[])


def test_validate_nodes_dir_honours_allowed_modules(tmp_path):
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    (nodes / "ok_with_pathlib.py").write_text(
        "from pathlib import Path\np = Path('/tmp')\n", encoding="utf-8"
    )
    # pathlib is in the default blocklist; allow_modules opens it back up.
    plugin_cli.validate_nodes_dir(nodes, allowed_modules=["pathlib"])


def test_validate_nodes_dir_allows_getattr_with_literal(tmp_path):
    """`getattr(obj, "literal")` is the common idiom for optional attrs —
    refining the AST gate so plugins that just want `getattr(context,
    "verbose", False)` aren't false-positived."""
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    (nodes / "verbose_check.py").write_text(
        "def f(context):\n"
        "    return getattr(context, 'verbose', False)\n",
        encoding="utf-8",
    )
    plugin_cli.validate_nodes_dir(nodes, allowed_modules=[])


def test_validate_nodes_dir_rejects_dynamic_getattr(tmp_path):
    """Dynamic attribute names — the actual sandbox-bypass shape — stay blocked."""
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    (nodes / "bad.py").write_text(
        "def f(name):\n"
        "    return getattr(__builtins__, name)\n",
        encoding="utf-8",
    )
    with pytest.raises(PluginValidationError):
        plugin_cli.validate_nodes_dir(nodes, allowed_modules=[])


# ── load_catalog ────────────────────────────────────────────────────────────

def test_load_catalog_returns_three_direction_packs():
    catalog = plugin_cli.load_catalog()
    plugins = catalog.get("plugins", {})
    assert set(plugins.keys()) >= {"foundations", "deep", "rl"}
    for pid in ("foundations", "deep", "rl"):
        assert plugins[pid].get("kind") == "builtin"
        assert plugins[pid].get("path") == f"plugins/{pid}"


# ── _install_deps spec construction ───────────────────────────────────────

def test_install_deps_builds_correct_pip_specs(monkeypatch):
    captured: list[list[str]] = []

    def fake_run(cmd, check=False):
        captured.append(cmd)
        class _R:
            returncode = 0
        return _R()

    import subprocess
    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = plugin_cli._install_deps({
        "foo": ">=1.0",         # version constraint passes through
        "bar": "==2.3.4",       # explicit equality
        "baz": "1.0.0",         # bare version → coerced to ==1.0.0
        "qux": "",              # no constraint
    })
    assert rc == 0
    assert captured, "uv pip install should have been invoked"
    cmd = captured[0]
    assert cmd[:3] == ["uv", "pip", "install"]
    specs = cmd[3:]
    assert "foo>=1.0" in specs
    assert "bar==2.3.4" in specs
    assert "baz==1.0.0" in specs
    assert "qux" in specs


# ── _manifest_has_frontend ────────────────────────────────────────────────

def test_manifest_has_frontend_detection():
    assert plugin_cli._manifest_has_frontend({"frontend": {"entry": "frontend/index.js"}}) is True
    assert plugin_cli._manifest_has_frontend({}) is False
    assert plugin_cli._manifest_has_frontend({"frontend": {}}) is False
    assert plugin_cli._manifest_has_frontend({"frontend": {"entry": ""}}) is False


# ── link / unlink / reload (local dev loop) ────────────────────────────────

@pytest.fixture
def isolated_lockfile(tmp_path, monkeypatch):
    """Redirect the lockfile to a temp dir and stub the server hot-reload so
    CLI tests never touch real user data or a running server."""
    target = tmp_path / "plugins"
    target.mkdir()
    monkeypatch.setattr(plugin_loader, "plugins_user_root", lambda: target)
    monkeypatch.setattr(plugin_cli, "_backend_reload", lambda: False)
    return target


def _write_plugin_dir(root: Path, plugin_id: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "cdui.plugin.toml").write_text(
        dedent(f"""\
            [plugin]
            id = "{plugin_id}"
            name = "Local {plugin_id}"
            version = "0.1.0"
            schema_version = 1
            """),
        encoding="utf-8",
    )
    nodes = root / "nodes"
    nodes.mkdir(exist_ok=True)
    (nodes / "__init__.py").write_text("", encoding="utf-8")


def test_cmd_link_writes_local_lockfile_entry(isolated_lockfile, tmp_path):
    work = tmp_path / "work" / "my-dev-plugin"
    _write_plugin_dir(work, "my-dev-plugin")

    rc = plugin_cli.cmd_link(argparse.Namespace(path=str(work), force=False))
    assert rc == 0

    entry = plugin_loader.load_lockfile()["plugins"]["my-dev-plugin"]
    assert entry["source_kind"] == "local"
    assert Path(entry["path"]) == work.resolve()
    assert entry["enabled"] is True


def test_cmd_link_rejects_dir_without_manifest(isolated_lockfile, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = plugin_cli.cmd_link(argparse.Namespace(path=str(empty), force=False))
    assert rc == 1
    assert plugin_loader.load_lockfile()["plugins"] == {}


def test_cmd_link_rejects_catalog_id_collision(isolated_lockfile, tmp_path):
    work = tmp_path / "shadow"
    _write_plugin_dir(work, "foundations")  # a real built-in catalog id
    rc = plugin_cli.cmd_link(argparse.Namespace(path=str(work), force=False))
    assert rc == 1
    assert "foundations" not in plugin_loader.load_lockfile()["plugins"]


def test_cmd_link_existing_id_requires_force(isolated_lockfile, tmp_path):
    work = tmp_path / "dup"
    _write_plugin_dir(work, "dup-plugin")
    assert plugin_cli.cmd_link(argparse.Namespace(path=str(work), force=False)) == 0
    # Re-linking without --force is rejected; with --force it succeeds.
    assert plugin_cli.cmd_link(argparse.Namespace(path=str(work), force=False)) == 1
    assert plugin_cli.cmd_link(argparse.Namespace(path=str(work), force=True)) == 0


def test_cmd_unlink_removes_entry_without_deleting_files(isolated_lockfile, tmp_path):
    work = tmp_path / "work" / "linked"
    _write_plugin_dir(work, "linked")
    plugin_cli.cmd_link(argparse.Namespace(path=str(work), force=False))

    rc = plugin_cli.cmd_unlink(argparse.Namespace(plugin_id="linked"))
    assert rc == 0
    assert "linked" not in plugin_loader.load_lockfile()["plugins"]
    # The author's working tree is untouched.
    assert (work / "cdui.plugin.toml").exists()


def test_cmd_unlink_refuses_non_local_entry(isolated_lockfile):
    lockfile = plugin_loader.load_lockfile()
    lockfile.setdefault("plugins", {})["deep"] = {
        "source_kind": "builtin", "source": "deep", "enabled": True,
    }
    plugin_loader.save_lockfile(lockfile)

    rc = plugin_cli.cmd_unlink(argparse.Namespace(plugin_id="deep"))
    assert rc == 1
    assert "deep" in plugin_loader.load_lockfile()["plugins"]  # refused to drop it


def test_cmd_unlink_missing_plugin_errors(isolated_lockfile):
    assert plugin_cli.cmd_unlink(argparse.Namespace(plugin_id="nope")) == 1


def test_cmd_reload_no_server_returns_zero(isolated_lockfile):
    # _backend_reload stubbed to False (no server) — reload is a graceful no-op.
    assert plugin_cli.cmd_reload(argparse.Namespace()) == 0


def test_link_unlink_reload_parser_wired():
    parser = plugin_cli.build_parser()
    a = parser.parse_args(["link", "/some/path"])
    assert a._func is plugin_cli.cmd_link and a.path == "/some/path" and a.force is False
    a = parser.parse_args(["link", "/p", "--force"])
    assert a.force is True
    a = parser.parse_args(["unlink", "foo"])
    assert a._func is plugin_cli.cmd_unlink and a.plugin_id == "foo"
    a = parser.parse_args(["reload"])
    assert a._func is plugin_cli.cmd_reload
