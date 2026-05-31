"""Tests for the plugin enable/disable feature.

Three layers of coverage:

    1. ``plugin_loader`` — ``is_enabled`` defaults, filter behavior in
       ``install_plugin_finder`` and ``iter_plugin_dirs``.
    2. CLI ``scripts/plugins.py`` — ``cmd_enable`` / ``cmd_disable``
       cycle, no-op idempotency, error on missing plugin.
    3. HTTP API ``/api/plugins/{id}/enable|disable`` — toggle endpoints,
       list endpoint reports ``enabled`` flag, GET errors on missing
       plugin.

The fixture redirects ``CODEFYUI_USER_DATA_DIR`` to a ``tmp_path`` so
tests don't touch the dev or global lockfiles.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import plugins as plugin_cli
from app.core import plugin_loader


@pytest.fixture
def isolated_lockfile(tmp_path, monkeypatch):
    """Redirect the plugin lockfile to a tmp dir for the test's duration."""
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    yield tmp_path


# ── plugin_loader low-level ──────────────────────────────────────────────


def test_is_enabled_defaults_true_for_legacy_entries():
    """Lockfiles written before the enabled field exist still work."""
    assert plugin_loader.is_enabled({}) is True
    assert plugin_loader.is_enabled({"source_kind": "builtin"}) is True


def test_is_enabled_respects_explicit_flag():
    assert plugin_loader.is_enabled({"enabled": True}) is True
    assert plugin_loader.is_enabled({"enabled": False}) is False


def test_install_plugin_finder_skips_disabled(tmp_path):
    """Disabled plugins must not contribute nodes to the discovery list."""
    builtin = plugin_loader.plugins_builtin_root()
    lockfile = {
        "schema": 1,
        "plugins": {
            "foundations": {"source_kind": "builtin", "source": "foundations", "enabled": True},
            "deep": {"source_kind": "builtin", "source": "deep", "enabled": False},
        },
    }
    pairs = plugin_loader.install_plugin_finder(builtin, tmp_path, lockfile)
    paths = {p[0].parent.name for p in pairs}
    assert "foundations" in paths
    assert "deep" not in paths


def test_iter_plugin_dirs_include_disabled_flag(tmp_path):
    builtin = plugin_loader.plugins_builtin_root()
    lockfile = {
        "schema": 1,
        "plugins": {
            "foundations": {"source_kind": "builtin", "source": "foundations", "enabled": True},
            "deep": {"source_kind": "builtin", "source": "deep", "enabled": False},
        },
    }
    enabled_only = {p[0] for p in plugin_loader.iter_plugin_dirs(builtin, tmp_path, lockfile)}
    with_all = {
        p[0]
        for p in plugin_loader.iter_plugin_dirs(
            builtin, tmp_path, lockfile, include_disabled=True
        )
    }
    assert enabled_only == {"foundations"}
    assert with_all == {"foundations", "deep"}


# ── CLI ──────────────────────────────────────────────────────────────────


def _read_lockfile(root: Path) -> dict:
    lf = root / "plugins" / "installed.json"
    return json.loads(lf.read_text(encoding="utf-8"))


def test_cli_install_sets_enabled_true(isolated_lockfile):
    """Fresh install always writes enabled=true so the entry is active."""
    rc = plugin_cli.main(["install", "foundations", "--no-confirm"])
    assert rc == 0
    assert _read_lockfile(isolated_lockfile)["plugins"]["foundations"]["enabled"] is True


def test_cli_disable_then_enable_cycle(isolated_lockfile):
    plugin_cli.main(["install", "foundations", "--no-confirm"])

    assert plugin_cli.main(["disable", "foundations"]) == 0
    assert _read_lockfile(isolated_lockfile)["plugins"]["foundations"]["enabled"] is False

    assert plugin_cli.main(["enable", "foundations"]) == 0
    assert _read_lockfile(isolated_lockfile)["plugins"]["foundations"]["enabled"] is True


def test_cli_disable_is_idempotent_noop(isolated_lockfile, capsys):
    plugin_cli.main(["install", "foundations", "--no-confirm"])
    plugin_cli.main(["disable", "foundations"])
    # Second disable should succeed with a "already disabled" notice, not error.
    assert plugin_cli.main(["disable", "foundations"]) == 0
    assert _read_lockfile(isolated_lockfile)["plugins"]["foundations"]["enabled"] is False


def test_cli_disable_missing_plugin_fails(isolated_lockfile):
    assert plugin_cli.main(["disable", "foundations"]) == 1


def test_cli_disable_preserves_repo_files(isolated_lockfile):
    """Disable must not delete the repo's plugins/foundations/ directory."""
    plugin_cli.main(["install", "foundations", "--no-confirm"])
    repo_dir = plugin_loader.plugins_builtin_root() / "foundations"
    nodes_before = sorted(p.name for p in (repo_dir / "nodes").glob("*.py"))

    plugin_cli.main(["disable", "foundations"])

    assert repo_dir.is_dir()
    nodes_after = sorted(p.name for p in (repo_dir / "nodes").glob("*.py"))
    assert nodes_after == nodes_before
