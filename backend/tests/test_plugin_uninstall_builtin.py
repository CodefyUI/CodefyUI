"""Verify official direction packs (foundations / deep / rl) can be uninstalled and re-installed.

The packs live inside the repo at ``plugins/<id>/`` and are activated by
writing a lockfile entry at ``<USER_DATA>/plugins/installed.json`` — no
file copy. Uninstall should:

    1. Remove the lockfile entry (deactivation).
    2. **Leave the repo directory intact** — it's checked-in code, not user
       data, and a follow-up ``install`` must be able to re-activate it.
    3. **Record that the removal happened** (#175) — a tombstone in the
       lockfile's ``removed`` map. Popping the entry alone made "this install
       never heard of the pack" and "the user threw it away" the same state,
       and those are precisely the two states ``cdui plugin sync`` has to tell
       apart. The tombstone lives beside ``plugins`` rather than inside it so
       every existing reader of ``plugins`` keeps its exact meaning.

A regression here would either (a) refuse to uninstall builtins as "you
can't remove what you didn't download", or (b) accidentally try to
``rmtree`` the repo's ``plugins/foundations/`` directory. Both have happened
in plugin systems before; this test pins the contract.
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
    """Full uninstall/reinstall cycle for foundations — the canonical builtin pack."""
    # 1. Fresh tmp lockfile → nothing installed.
    assert "foundations" not in _read_lockfile(isolated_lockfile).get("plugins", {})

    # 2. Install foundations from the in-repo catalog.
    rc = plugin_cli.main(["install", "foundations", "--no-confirm"])
    assert rc == 0
    locked = _read_lockfile(isolated_lockfile)["plugins"]
    assert "foundations" in locked
    assert locked["foundations"]["source_kind"] == "builtin"
    assert locked["foundations"]["source"] == "foundations"

    # 3. Builtin install MUST NOT touch the repo's plugins/foundations/ — it's
    #    the source of truth, not a user download.
    repo_plugin_dir = plugin_loader.plugins_builtin_root() / "foundations"
    assert repo_plugin_dir.is_dir()
    assert (repo_plugin_dir / "cdui.plugin.toml").is_file()
    nodes_before = sorted(p.name for p in (repo_plugin_dir / "nodes").glob("*.py"))
    assert nodes_before, "repo plugins/foundations/nodes/ should contain Edu node files"

    # 4. Uninstall foundations — should succeed for builtin packs (the bug we're
    #    guarding against: a check that refuses to uninstall source_kind=builtin).
    rc = plugin_cli.main(["uninstall", "foundations"])
    assert rc == 0
    assert "foundations" not in _read_lockfile(isolated_lockfile)["plugins"]

    # 5. Repo dir still untouched after uninstall — uninstall must not rm the
    #    repo's plugins/foundations/ even though source_kind=builtin pointed at it.
    assert repo_plugin_dir.is_dir()
    nodes_after = sorted(p.name for p in (repo_plugin_dir / "nodes").glob("*.py"))
    assert nodes_after == nodes_before, "uninstall should not delete repo plugin files"

    # 6. Re-install works — proves builtin uninstall is fully reversible.
    rc = plugin_cli.main(["install", "foundations", "--no-confirm"])
    assert rc == 0
    assert "foundations" in _read_lockfile(isolated_lockfile)["plugins"]


def test_uninstall_missing_plugin_fails_cleanly(isolated_lockfile):
    """Uninstalling something that's not in the lockfile should return non-zero."""
    rc = plugin_cli.main(["uninstall", "foundations"])
    assert rc == 1


def test_uninstall_does_not_touch_other_builtins(isolated_lockfile):
    """Uninstalling deep must not affect foundations / rl."""
    for pid in ("foundations", "deep", "rl"):
        rc = plugin_cli.main(["install", pid, "--no-confirm"])
        assert rc == 0

    rc = plugin_cli.main(["uninstall", "deep"])
    assert rc == 0

    locked = _read_lockfile(isolated_lockfile)["plugins"]
    assert set(locked.keys()) == {"foundations", "rl"}
    # And every repo dir still in place.
    for pid in ("foundations", "deep", "rl"):
        assert (plugin_loader.plugins_builtin_root() / pid).is_dir()


# ── the uninstall tombstone (#175, option E) ─────────────────────────────────

def test_uninstall_records_the_removal(isolated_lockfile, capsys):
    assert plugin_cli.main(["install", "rl", "--no-confirm"]) == 0
    assert plugin_cli.main(["uninstall", "rl"]) == 0

    data = _read_lockfile(isolated_lockfile)
    assert "rl" not in data["plugins"]          # gone from the active set
    assert plugin_loader.removed_ids(data) == {"rl"}  # but the decision is kept
    entry = data["removed"]["rl"]
    assert entry["source_kind"] == "builtin"
    assert entry["removed_at"]                 # when, so the record is auditable

    # And the user is told what the tombstone means, since nothing else shows it.
    out = capsys.readouterr().out
    assert "cdui plugin sync" in out or "cdui plugin sync" in out.lower()
    assert "cdui plugin install rl" in out


def test_a_reinstall_clears_the_tombstone(isolated_lockfile):
    assert plugin_cli.main(["install", "rl", "--no-confirm"]) == 0
    assert plugin_cli.main(["uninstall", "rl"]) == 0
    assert plugin_cli.main(["install", "rl", "--no-confirm"]) == 0

    data = _read_lockfile(isolated_lockfile)
    assert "rl" in data["plugins"]
    # Cleared entirely, not left as an empty map: a lockfile that records no
    # removals should look exactly like one written before this field existed.
    assert plugin_loader.removed_ids(data) == set()
    assert "removed" not in data


def test_a_tombstone_records_only_the_pack_that_was_removed(isolated_lockfile):
    for pid in ("foundations", "deep", "rl"):
        assert plugin_cli.main(["install", pid, "--no-confirm"]) == 0
    assert plugin_cli.main(["uninstall", "deep"]) == 0

    data = _read_lockfile(isolated_lockfile)
    assert plugin_loader.removed_ids(data) == {"deep"}
    assert set(data["plugins"]) == {"foundations", "rl"}


def test_unlinking_a_local_plugin_leaves_no_tombstone(isolated_lockfile, tmp_path):
    """Only built-in packs need one — nothing re-adds a local link uninvited."""
    work = tmp_path / "work" / "my-local-pack"
    (work / "nodes").mkdir(parents=True)
    (work / "nodes" / "__init__.py").write_text("", encoding="utf-8")
    (work / "cdui.plugin.toml").write_text(
        '[plugin]\nid = "my-local-pack"\nname = "Local"\n'
        'version = "0.1.0"\nschema_version = 1\n',
        encoding="utf-8",
    )
    assert plugin_cli.main(["link", str(work)]) == 0
    assert plugin_cli.main(["uninstall", "my-local-pack"]) == 0

    data = _read_lockfile(isolated_lockfile)
    assert "my-local-pack" not in data["plugins"]
    assert plugin_loader.removed_ids(data) == set()


def test_a_pre_175_lockfile_loads_and_gains_a_tombstone_in_place(isolated_lockfile):
    """Backward compat: a lockfile with no ``removed`` key at all is untouched
    until something is actually removed, and every other field survives."""
    lockfile_dir = isolated_lockfile / "plugins"
    lockfile_dir.mkdir(parents=True, exist_ok=True)
    legacy = {
        "schema": 1,
        "plugins": {
            "rl": {"source_kind": "builtin", "source": "rl"},  # no `enabled` either
        },
    }
    (lockfile_dir / "installed.json").write_text(json.dumps(legacy), encoding="utf-8")

    loaded = plugin_loader.load_lockfile()
    assert loaded == legacy                      # read back verbatim
    assert plugin_loader.removed_ids(loaded) == set()
    assert plugin_loader.is_enabled(loaded["plugins"]["rl"]) is True

    assert plugin_cli.main(["uninstall", "rl"]) == 0
    data = _read_lockfile(isolated_lockfile)
    assert data["schema"] == 1
    assert plugin_loader.removed_ids(data) == {"rl"}
