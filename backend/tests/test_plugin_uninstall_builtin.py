"""Verify official chapter packs (c1–c6) can be uninstalled and re-installed.

The c1–c6 packs live inside the repo at ``plugins/c<N>/`` and are activated
by writing a lockfile entry at ``<USER_DATA>/plugins/installed.json`` — no
file copy. Uninstall should:

    1. Remove the lockfile entry (deactivation).
    2. **Leave the repo directory intact** — it's checked-in code, not user
       data, and a follow-up ``install`` must be able to re-activate it.

A regression here would either (a) refuse to uninstall builtins as "you
can't remove what you didn't download", or (b) accidentally try to
``rmtree`` the repo's ``plugins/c1/`` directory. Both have happened in
plugin systems before; this test pins the contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import plugins as plugin_cli
from app.core import plugin_loader


@pytest.fixture
def isolated_lockfile(tmp_path, monkeypatch):
    """Redirect lockfile + downloaded-packs root to a tmp dir.

    Honored by plugin_loader.plugins_user_root() via CODEFYUI_USER_DATA_DIR
    (see test_plugin_dev_env.py for the env-var contract).
    """
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    yield tmp_path


def _read_lockfile(root: Path) -> dict:
    lf = root / "plugins" / "installed.json"
    if not lf.exists():
        return {"plugins": {}}
    return json.loads(lf.read_text(encoding="utf-8"))


def test_install_then_uninstall_builtin_pack(isolated_lockfile, capsys):
    """Full uninstall/reinstall cycle for c1 — the canonical builtin pack."""
    # 1. Fresh tmp lockfile → nothing installed.
    assert "c1" not in _read_lockfile(isolated_lockfile).get("plugins", {})

    # 2. Install c1 from the in-repo catalog.
    rc = plugin_cli.main(["install", "c1", "--no-confirm"])
    assert rc == 0
    locked = _read_lockfile(isolated_lockfile)["plugins"]
    assert "c1" in locked
    assert locked["c1"]["source_kind"] == "builtin"
    assert locked["c1"]["source"] == "c1"

    # 3. Builtin install MUST NOT touch the repo's plugins/c1/ — it's the
    #    source of truth, not a user download.
    repo_plugin_dir = plugin_loader.plugins_builtin_root() / "c1"
    assert repo_plugin_dir.is_dir()
    assert (repo_plugin_dir / "cdui.plugin.toml").is_file()
    nodes_before = sorted(p.name for p in (repo_plugin_dir / "nodes").glob("*.py"))
    assert nodes_before, "repo plugins/c1/nodes/ should contain Edu node files"

    # 4. Uninstall c1 — should succeed for builtin packs (the bug we're
    #    guarding against: a check that refuses to uninstall source_kind=builtin).
    rc = plugin_cli.main(["uninstall", "c1"])
    assert rc == 0
    assert "c1" not in _read_lockfile(isolated_lockfile)["plugins"]

    # 5. Repo dir still untouched after uninstall — uninstall must not rm
    #    the repo's plugins/c1/ even though source_kind=builtin pointed at it.
    assert repo_plugin_dir.is_dir()
    nodes_after = sorted(p.name for p in (repo_plugin_dir / "nodes").glob("*.py"))
    assert nodes_after == nodes_before, "uninstall should not delete repo plugin files"

    # 6. Re-install works — proves builtin uninstall is fully reversible.
    rc = plugin_cli.main(["install", "c1", "--no-confirm"])
    assert rc == 0
    assert "c1" in _read_lockfile(isolated_lockfile)["plugins"]


def test_uninstall_missing_plugin_fails_cleanly(isolated_lockfile):
    """Uninstalling something that's not in the lockfile should return non-zero."""
    rc = plugin_cli.main(["uninstall", "c1"])
    assert rc == 1


def test_uninstall_does_not_touch_other_builtins(isolated_lockfile):
    """Uninstalling c1 must not affect c2..c6."""
    for pid in ("c1", "c2", "c3"):
        rc = plugin_cli.main(["install", pid, "--no-confirm"])
        assert rc == 0

    rc = plugin_cli.main(["uninstall", "c2"])
    assert rc == 0

    locked = _read_lockfile(isolated_lockfile)["plugins"]
    assert set(locked.keys()) == {"c1", "c3"}
    # And every repo dir still in place.
    for pid in ("c1", "c2", "c3"):
        assert (plugin_loader.plugins_builtin_root() / pid).is_dir()
