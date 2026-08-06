"""A built-in pack that shipped but was never installed must be visible.

The failure this covers (#175): a release adds a pack, `cdui update` puts its
files on disk, but the server loads only what the lockfile records and nothing
re-syncs it. The pack is fully installable and completely invisible — the
class updates as instructed, the new chapter's nodes are not in the palette,
and nothing anywhere says why. `stats` shipped exactly this way.

Discoverability only: nothing here installs or enables a pack. Running code a
user did not ask for because a release shipped it is a consent decision, and
it stays theirs.
"""

from __future__ import annotations

import argparse

import pytest

import dev  # scripts/dev.py -- conftest puts scripts/ on sys.path
import plugins as plugin_cli  # scripts/plugins.py


@pytest.fixture
def catalog_of(monkeypatch):
    """Install a fake catalog + lockfile pair."""

    def _apply(catalog: dict, installed: dict):
        monkeypatch.setattr(plugin_cli, "load_catalog", lambda: {"plugins": catalog})
        monkeypatch.setattr(
            plugin_cli, "load_lockfile", lambda: {"plugins": installed}
        )

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
    assert "cdui plugin install stats" in out


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
    assert "Pack A" in out           # the installed one
    assert "cdui plugin install b" in out  # and the one that is not


def test_plugin_list_says_nothing_extra_when_everything_is_installed(
    catalog_of, capsys
):
    catalog_of(
        {"a": BUILTIN_A},
        {"a": {"enabled": True, "source_kind": "builtin", "source": "a",
               "manifest": {"name": "Pack A"}}},
    )
    plugin_cli.cmd_list(argparse.Namespace())
    assert "cdui plugin install" not in capsys.readouterr().out


# ── the `cdui start` banner ──────────────────────────────────────────────────

def test_start_banner_names_available_packs(monkeypatch, capsys):
    monkeypatch.setattr(
        plugin_cli, "available_builtin_packs", lambda: [("stats", "Stats")]
    )
    dev._print_uninstalled_builtin_packs()
    out = capsys.readouterr().out
    assert "stats (Stats)" in out
    assert "cdui plugin install stats" in out


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
