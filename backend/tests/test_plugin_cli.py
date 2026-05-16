"""Unit tests for scripts/plugins.py — source parsing, manifest validation, lockfile ops."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

import plugins as plugin_cli
from app.core import plugin_loader
from app.core.plugin_validator import PluginValidationError


# ── parse_source ───────────────────────────────────────────────────────────

def test_parse_source_catalog_case_insensitive():
    # 'C2' / 'c2' / 'C3' / etc. live in plugins/registry.json
    assert plugin_cli.parse_source("C2") == ("catalog", "c2", "", "")
    assert plugin_cli.parse_source("c2") == ("catalog", "c2", "", "")
    assert plugin_cli.parse_source("c6") == ("catalog", "c6", "", "")


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


# ── load_catalog ────────────────────────────────────────────────────────────

def test_load_catalog_returns_six_chapter_packs():
    catalog = plugin_cli.load_catalog()
    plugins = catalog.get("plugins", {})
    assert set(plugins.keys()) >= {"c1", "c2", "c3", "c4", "c5", "c6"}
    for pid in ("c1", "c2", "c3", "c4", "c5", "c6"):
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
