"""A built-in pack that shipped but was never installed must be visible.

The failure this covers (#175): a release adds a pack, `cdui update` puts its
files on disk, but the server loads only what the lockfile records and nothing
re-syncs it. The pack is fully installable and completely invisible — the
class updates as instructed, the new chapter's nodes are not in the palette,
and nothing anywhere says why. `stats` shipped exactly this way.

Discoverability first: nothing is installed or enabled behind the user's back.
Running code a user did not ask for because a release shipped it is a consent
decision, and it stays theirs — which is why the cure is a verb they type,
`cdui plugin sync`, and why a pack they uninstalled is remembered as removed
instead of offered again on the next start.
"""

from __future__ import annotations

import argparse
import json
from textwrap import dedent
from types import SimpleNamespace

import pytest

import dev  # scripts/dev.py -- conftest puts scripts/ on sys.path
import plugins as plugin_cli  # scripts/plugins.py
from app.core import plugin_loader


@pytest.fixture(autouse=True)
def _english_cli_messages(monkeypatch):
    """Pin the CLI's language so the message assertions below are deterministic.

    `plugins.py` picks zh from `LANG`/`LC_ALL`, so a maintainer whose shell is
    Traditional Chinese would otherwise see these tests fail on strings that are
    perfectly correct. Same trick as test_plugin_dx.py's subprocess env.
    """
    monkeypatch.setenv("CODEFYUI_LANG", "en")


@pytest.fixture
def catalog_of(monkeypatch):
    """Install a fake catalog + lockfile pair.

    ``removed`` is passed as a plain id list for brevity and stored in the
    lockfile shape the CLI actually writes (a map of id -> record), so these
    tests exercise the same read path as a real tombstone.
    """

    def _apply(catalog: dict, installed: dict, removed: list[str] | None = None):
        lockfile: dict = {"plugins": installed}
        if removed is not None:
            lockfile["removed"] = {pid: {"removed_at": "2026-08-12T00:00:00+00:00"}
                                   for pid in removed}
        monkeypatch.setattr(plugin_cli, "load_catalog", lambda: {"plugins": catalog})
        monkeypatch.setattr(plugin_cli, "load_lockfile", lambda: lockfile)

    return _apply


BUILTIN_A = {"kind": "builtin", "name": "Pack A", "path": "plugins/a"}
BUILTIN_B = {"kind": "builtin", "name": "Pack B", "path": "plugins/b"}
FROM_GITHUB = {"kind": "github", "name": "Third Party", "path": "plugins/x"}


def test_lists_a_builtin_pack_that_was_never_installed(catalog_of):
    catalog_of({"a": BUILTIN_A}, {})
    assert plugin_cli.available_builtin_packs() == [("a", "Pack A")]


def test_omits_packs_that_are_already_installed(catalog_of):
    catalog_of({"a": BUILTIN_A, "b": BUILTIN_B}, {"a": {"enabled": True}})
    assert plugin_cli.available_builtin_packs() == [("b", "Pack B")]


def test_a_disabled_pack_still_counts_as_installed(catalog_of):
    """Disabling is a decision the user already made -- do not nag."""
    catalog_of({"a": BUILTIN_A}, {"a": {"enabled": False}})
    assert plugin_cli.available_builtin_packs() == []


def test_ignores_non_builtin_catalog_entries(catalog_of):
    catalog_of({"x": FROM_GITHUB}, {})
    assert plugin_cli.available_builtin_packs() == []


def test_falls_back_to_the_id_when_an_entry_has_no_name(catalog_of):
    catalog_of({"a": {"kind": "builtin", "path": "plugins/a"}}, {})
    assert plugin_cli.available_builtin_packs() == [("a", "a")]


def test_result_is_sorted(catalog_of):
    catalog_of({"z": BUILTIN_B, "a": BUILTIN_A}, {})
    assert [pid for pid, _ in plugin_cli.available_builtin_packs()] == ["a", "z"]


def test_a_pack_the_user_uninstalled_is_not_offered_again(catalog_of):
    """The #175 consent crux: "removed on purpose" is not "never seen"."""
    catalog_of({"a": BUILTIN_A, "b": BUILTIN_B}, {}, removed=["a"])
    assert plugin_cli.available_builtin_packs() == [("b", "Pack B")]


def test_a_legacy_lockfile_without_the_removed_field_is_read_unchanged(catalog_of):
    """Backward compat: pre-#175 lockfiles have no ``removed`` key at all."""
    catalog_of({"a": BUILTIN_A}, {})  # no `removed` passed -> key absent
    assert plugin_cli.available_builtin_packs() == [("a", "Pack A")]


def test_a_malformed_removed_field_is_read_as_no_tombstones(monkeypatch):
    """A hand-edited lockfile must not hide every pack behind a typo."""
    monkeypatch.setattr(plugin_cli, "load_catalog", lambda: {"plugins": {"a": BUILTIN_A}})
    monkeypatch.setattr(
        plugin_cli, "load_lockfile", lambda: {"plugins": {}, "removed": ["a"]}
    )
    assert plugin_cli.available_builtin_packs() == [("a", "Pack A")]


def test_a_broken_lockfile_yields_no_notice_rather_than_raising(monkeypatch):
    def boom():
        raise OSError("lockfile is a directory")

    monkeypatch.setattr(plugin_cli, "load_lockfile", boom)
    assert plugin_cli.available_builtin_packs() == []


# ── `cdui plugin list` ───────────────────────────────────────────────────────

def test_plugin_list_names_available_packs_when_none_are_installed(
    catalog_of, capsys
):
    catalog_of({"stats": {"kind": "builtin", "name": "Stats", "path": "p"}}, {})
    assert plugin_cli.cmd_list(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert "stats" in out
    # The notice points at the verb, not at an id list that grows per release.
    assert "cdui plugin sync" in out


def test_plugin_list_names_available_packs_alongside_installed_ones(
    catalog_of, capsys
):
    catalog_of(
        {"a": BUILTIN_A, "b": BUILTIN_B},
        {"a": {"enabled": True, "source_kind": "builtin", "source": "a",
               "manifest": {"name": "Pack A", "version": "0.1.0"}}},
    )
    assert plugin_cli.cmd_list(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert "Pack A" in out            # the installed one
    assert "Pack B" in out            # and the one that is not
    assert "cdui plugin sync" in out  # with the one command that fixes it


def test_plugin_list_says_nothing_extra_when_everything_is_installed(
    catalog_of, capsys
):
    catalog_of(
        {"a": BUILTIN_A},
        {"a": {"enabled": True, "source_kind": "builtin", "source": "a",
               "manifest": {"name": "Pack A"}}},
    )
    plugin_cli.cmd_list(argparse.Namespace())
    out = capsys.readouterr().out
    assert "cdui plugin install" not in out
    assert "cdui plugin sync" not in out


# ── the `cdui start` banner ──────────────────────────────────────────────────

def test_start_banner_names_available_packs(monkeypatch, capsys):
    monkeypatch.setattr(
        plugin_cli, "available_builtin_packs", lambda: [("stats", "Stats")]
    )
    dev._print_uninstalled_builtin_packs()
    out = capsys.readouterr().out
    assert "stats (Stats)" in out
    assert "cdui plugin sync" in out


def test_start_banner_is_silent_when_nothing_is_pending(monkeypatch, capsys):
    monkeypatch.setattr(plugin_cli, "available_builtin_packs", lambda: [])
    dev._print_uninstalled_builtin_packs()
    assert capsys.readouterr().out == ""


def test_start_banner_never_breaks_startup(monkeypatch, capsys):
    """A notice must not be the reason a server fails to start."""

    def boom():
        raise RuntimeError("plugin subsystem exploded")

    monkeypatch.setattr(plugin_cli, "available_builtin_packs", boom)
    dev._print_uninstalled_builtin_packs()  # must not raise
    assert capsys.readouterr().out == ""


# ── `cdui plugin sync` (#175, option D) ──────────────────────────────────────

def _write_pack(root, pack_id: str) -> None:
    nodes = root / pack_id / "nodes"
    nodes.mkdir(parents=True, exist_ok=True)
    (nodes / "__init__.py").write_text("", encoding="utf-8")
    (root / pack_id / "cdui.plugin.toml").write_text(
        dedent(f"""\
            [plugin]
            id = "{pack_id}"
            name = "Pack {pack_id}"
            version = "0.1.0"
            schema_version = 1
            """),
        encoding="utf-8",
    )


@pytest.fixture
def fake_packs(tmp_path, monkeypatch):
    """A fake built-in root (registry.json + three pack dirs) and a tmp lockfile.

    The real catalog cannot be used for the install path: `edu` declares
    `python_deps`, so syncing it would shell out to `uv pip install` from a unit
    test. Everything else is real — `cmd_sync` calls the same
    `_install_catalog` the CLI does, and the lockfile is written and re-read
    from disk, so the tombstone round-trips through JSON exactly as it will in
    a user's `installed.json`.
    """
    user_root = tmp_path / "user" / "plugins"
    user_root.mkdir(parents=True)
    builtin_root = tmp_path / "repo" / "plugins"
    builtin_root.mkdir(parents=True)

    for pack_id in ("alpha", "beta", "gamma"):
        _write_pack(builtin_root, pack_id)
    (builtin_root / "registry.json").write_text(
        json.dumps({
            "schema": 1,
            "plugins": {
                pack_id: {
                    "kind": "builtin",
                    "name": f"Pack {pack_id}",
                    "path": f"plugins/{pack_id}",
                }
                for pack_id in ("alpha", "beta", "gamma")
            },
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(plugin_loader, "plugins_user_root", lambda: user_root)
    monkeypatch.setattr(plugin_cli, "plugins_user_root", lambda: user_root)
    monkeypatch.setattr(plugin_cli, "plugins_builtin_root", lambda: builtin_root)
    monkeypatch.setattr(plugin_cli, "_backend_reload", lambda: False)
    return SimpleNamespace(user_root=user_root, builtin_root=builtin_root)


def _locked() -> dict:
    return plugin_loader.load_lockfile()


def _installed_ids() -> set[str]:
    return set(_locked().get("plugins", {}))


def test_sync_installs_every_pending_pack(fake_packs):
    assert plugin_cli.main(["sync", "--yes"]) == 0
    assert _installed_ids() == {"alpha", "beta", "gamma"}


def test_sync_installs_exactly_the_missing_ones(fake_packs):
    assert plugin_cli.main(["install", "beta", "--no-confirm"]) == 0
    installed_at = _locked()["plugins"]["beta"]["installed_at"]

    assert plugin_cli.main(["sync", "--yes"]) == 0
    assert _installed_ids() == {"alpha", "beta", "gamma"}
    # beta was not reinstalled on top of itself -- sync skips, it does not force.
    assert _locked()["plugins"]["beta"]["installed_at"] == installed_at


def test_sync_is_a_no_op_once_everything_is_decided(fake_packs, capsys):
    assert plugin_cli.main(["sync", "--yes"]) == 0
    capsys.readouterr()
    assert plugin_cli.main(["sync", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "nothing to sync" in out


def test_sync_never_re_adds_a_pack_the_user_uninstalled(fake_packs, capsys):
    """The consent crux (#175, option E): sync respects a removal."""
    assert plugin_cli.main(["install", "alpha", "--no-confirm"]) == 0
    assert plugin_cli.main(["uninstall", "alpha"]) == 0
    capsys.readouterr()

    assert plugin_cli.main(["sync", "--yes"]) == 0
    assert _installed_ids() == {"beta", "gamma"}
    out = capsys.readouterr().out
    assert "Skipping packs you removed: alpha" in out


def test_installing_by_name_clears_the_tombstone(fake_packs):
    assert plugin_cli.main(["install", "alpha", "--no-confirm"]) == 0
    assert plugin_cli.main(["uninstall", "alpha"]) == 0
    assert plugin_loader.removed_ids(_locked()) == {"alpha"}

    assert plugin_cli.main(["install", "alpha", "--no-confirm"]) == 0
    assert plugin_loader.removed_ids(_locked()) == set()
    # And a pack whose tombstone is gone counts as decided again, not pending.
    assert plugin_cli.available_builtin_packs() == [("beta", "Pack beta"),
                                                   ("gamma", "Pack gamma")]


def test_dry_run_lists_the_pending_packs_and_installs_nothing(fake_packs, capsys):
    assert plugin_cli.main(["sync", "--dry-run"]) == 0
    assert _installed_ids() == set()
    out = capsys.readouterr().out
    for pack_id in ("alpha", "beta", "gamma"):
        assert pack_id in out
    assert "--dry-run" in out


def test_dry_run_omits_a_pack_the_user_uninstalled(fake_packs, capsys):
    assert plugin_cli.main(["install", "alpha", "--no-confirm"]) == 0
    assert plugin_cli.main(["uninstall", "alpha"]) == 0
    capsys.readouterr()

    assert plugin_cli.main(["sync", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Skipping packs you removed: alpha" in out
    # alpha appears only in that skip line, never in the would-install list.
    assert out.count("alpha") == 1


def test_one_pack_that_fails_does_not_stop_the_others(fake_packs, monkeypatch, capsys):
    """An offline `python_deps` download must cost one pack, not the batch."""
    real_install = plugin_cli._install_catalog

    def flaky(pack_id, args, lockfile):
        if pack_id == "beta":
            plugin_cli.err("下載相依套件失敗", "dependency download failed")
            return 1
        return real_install(pack_id, args, lockfile)

    monkeypatch.setattr(plugin_cli, "_install_catalog", flaky)

    assert plugin_cli.main(["sync", "--yes"]) == 1  # the failure is reported
    assert _installed_ids() == {"alpha", "gamma"}   # and the rest still landed
    captured = capsys.readouterr()
    assert "beta failed" in captured.out
    assert "1 failed (beta)" in captured.err


def test_sync_refuses_to_install_without_a_terminal_or_yes(fake_packs, capsys):
    """Fails closed like the capability gate: never install on a guess."""
    assert plugin_cli.main(["sync"]) == 1
    assert _installed_ids() == set()
    assert "--yes" in capsys.readouterr().err


def test_sync_fails_when_stdin_closes_under_a_terminal(fake_packs, monkeypatch, capsys):
    """`cdui plugin sync < /dev/null` on Windows: isatty() says yes, EOF anyway.

    Found in the #175 smoke run. The first version asked `_stdin_is_interactive()`
    a second time to pick the exit code, so this path printed "pass --yes" and
    then exited 0 — an unanswerable question reported as a completed command.
    """
    monkeypatch.setattr(plugin_cli, "_stdin_is_interactive", lambda: True)

    def _eof(_prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)
    assert plugin_cli.main(["sync"]) == 1
    assert _installed_ids() == set()
    assert "--yes" in capsys.readouterr().err


def test_an_interactive_no_installs_nothing_and_is_not_an_error(
    fake_packs, monkeypatch, capsys
):
    monkeypatch.setattr(plugin_cli, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _p: "n")
    assert plugin_cli.main(["sync"]) == 0  # a decision, not a failure
    assert _installed_ids() == set()
    assert "Cancelled" in capsys.readouterr().out


@pytest.mark.parametrize("answer", ["y", "Y", "yes", " YES "])
def test_an_interactive_yes_installs(fake_packs, monkeypatch, answer):
    monkeypatch.setattr(plugin_cli, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _p: answer)
    assert plugin_cli.main(["sync"]) == 0
    assert _installed_ids() == {"alpha", "beta", "gamma"}


def test_yes_never_prompts(fake_packs, monkeypatch):
    def _explode(*_a, **_k):  # pragma: no cover - only runs on a bug
        raise AssertionError("--yes must not prompt")

    monkeypatch.setattr("builtins.input", _explode)
    assert plugin_cli.main(["sync", "--yes"]) == 0
    assert _installed_ids() == {"alpha", "beta", "gamma"}


def test_dry_run_never_prompts(fake_packs, monkeypatch):
    def _explode(*_a, **_k):  # pragma: no cover - only runs on a bug
        raise AssertionError("--dry-run must not prompt")

    monkeypatch.setattr(plugin_cli, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", _explode)
    assert plugin_cli.main(["sync", "--dry-run"]) == 0
    assert _installed_ids() == set()


def test_sync_prunes_only_orphaned_builtin_keys(fake_packs, capsys):
    assert plugin_cli.main(["sync", "--yes"]) == 0
    lockfile = _locked()
    # A pack a past release shipped and this one does not: still in the
    # lockfile, silently skipped by discovery, invisible to every command.
    lockfile["plugins"]["ghost"] = {"source_kind": "builtin", "source": "ghost",
                                   "enabled": True}
    # Two entries prune must NOT touch: a local link whose checkout may just be
    # unmounted, and a downloaded pack whose files are the user's own data.
    lockfile["plugins"]["mylink"] = {"source_kind": "local",
                                     "path": str(fake_packs.user_root / "nope"),
                                     "enabled": True}
    lockfile["plugins"]["third"] = {"source_kind": "github_url", "source": "a/b",
                                    "enabled": True}
    lockfile.setdefault("removed", {})["vanished"] = {"removed_at": "2026-01-01T00:00:00+00:00"}
    plugin_loader.save_lockfile(lockfile)
    capsys.readouterr()

    assert plugin_cli.main(["sync", "--prune"]) == 0
    after = _locked()
    assert set(after["plugins"]) == {"alpha", "beta", "gamma", "mylink", "third"}
    assert plugin_loader.removed_ids(after) == set()
    out = capsys.readouterr().out
    assert "ghost" in out and "vanished" in out
    assert "mylink" not in out and "third" not in out


def test_prune_under_dry_run_changes_nothing(fake_packs, capsys):
    """`--dry-run` promises to change nothing, including with `--prune`."""
    lockfile = plugin_loader.empty_lockfile()
    lockfile["plugins"]["ghost"] = {"source_kind": "builtin", "source": "ghost"}
    plugin_loader.save_lockfile(lockfile)

    assert plugin_cli.main(["sync", "--prune", "--dry-run"]) == 0
    assert "ghost" in _installed_ids()  # still there
    out = capsys.readouterr().out
    assert "would prune" in out


def test_sync_without_prune_leaves_a_stale_key_alone(fake_packs):
    lockfile = plugin_loader.empty_lockfile()
    lockfile["plugins"]["ghost"] = {"source_kind": "builtin", "source": "ghost"}
    plugin_loader.save_lockfile(lockfile)

    assert plugin_cli.main(["sync", "--yes"]) == 0
    assert "ghost" in _installed_ids()


def test_sync_parser_wiring():
    parser = plugin_cli.build_parser()
    a = parser.parse_args(["sync"])
    assert a._func is plugin_cli.cmd_sync
    assert a.dry_run is False and a.yes is False and a.prune is False
    a = parser.parse_args(["sync", "--dry-run", "-y", "--prune"])
    assert a.dry_run is True and a.yes is True and a.prune is True
