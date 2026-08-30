"""``cdui cache`` -- what the app leaves on disk, and how to get it back (#306).

``LMTokenizedDataset`` writes one packed ``.pt`` per distinct (corpus,
tokenizer, seq_len, append_eos, max_tokens) key under
``<data root>/cache/lm_blocks/``, at 8 bytes per token. Nothing evicted it and
nothing named it, so a learner sweeping ``seq_len`` over three values silently
left three ~800 MB copies of the same corpus behind. ``cdui cache list`` names
them and ``cdui cache prune`` deletes them.

What these tests pin, beyond the obvious delete:

* the inventory is DERIVED caches only -- a list that grows by accident is how
  a prune command ends up deleting a downloaded model or somebody's run
  outputs, so the tuple is asserted whole;
* ``prune`` refuses while a server started from this checkout is alive,
  because a run in it may be reading a block file it is about to lose;
* ``--older-than`` keeps young entries, and ``--yes`` is the only way past the
  prompt -- with no terminal the command refuses rather than deciding yes.

Every test drives a tmp directory: ``derived_cache_root`` is patched, so
nothing here can reach the real ``backend/data``.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

import cache  # scripts/cache.py -- conftest puts scripts/ on sys.path
import dev  # scripts/dev.py -- same


@pytest.fixture(autouse=True)
def _english(monkeypatch):
    """Pin the CLI's language so assertions are about behaviour, not locale."""
    monkeypatch.setenv("CODEFYUI_LANG", "en")


@pytest.fixture(autouse=True)
def _no_server(monkeypatch):
    """No server is running unless a test says one is.

    Patched at ``dev``'s end rather than at ``cache``'s, so every test here
    still goes through the real delegation: the wrapper is what production
    calls, and a stub over it would hide a wrapper that had stopped asking.
    The real check shells out to ``tasklist``, which would otherwise make
    these tests depend on whatever is running on the machine.
    """
    monkeypatch.setattr(dev, "_running_server_pid", lambda: None)


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    """Point the whole command at a tmp ``cache/`` directory."""
    root = tmp_path / "cache"
    monkeypatch.setattr(cache, "derived_cache_root", lambda: root)
    return root


def _blocks(cache_root: Path, *names: str, size: int = 1024) -> list[Path]:
    """Write *names* into the ``lm_blocks`` cache and hand back the paths."""
    directory = cache_root / "lm_blocks"
    directory.mkdir(parents=True, exist_ok=True)
    made = []
    for name in names:
        path = directory / name
        path.write_bytes(b"x" * size)
        made.append(path)
    return made


def _age(path: Path, days: float) -> None:
    """Backdate *path* so ``--older-than`` sees it as *days* old."""
    when = time.time() - days * 86400
    os.utime(path, (when, when))


# ── dev.py dispatch ───────────────────────────────────────────────────────


def test_cache_dispatch_is_registered():
    """``cdui cache ...`` routes like plugin, project and packs do."""
    assert dev._subcommand_group(["cdui", "cache", "list"]) is \
        dev._dispatch_cache_subcommand
    assert dev._subcommand_group(["cdui", "cache", "prune"]) is \
        dev._dispatch_cache_subcommand


def test_cache_is_listed_in_the_help_text():
    """A subcommand nobody can find is one nobody types."""
    assert "cache <subcmd>" in dev.__doc__


def test_dispatch_enters_the_venv_but_does_not_go_looking_for_uv(monkeypatch):
    """The venv hop is needed; the uv bootstrap is not.

    ``cache.py`` reads ``app.core.data_paths`` for the data root, so the
    interpreter has to be the venv's first. It spawns no package manager at
    all, though, and ``_ensure_uv`` DOWNLOADS uv from the network when it is
    missing -- which is not a thing listing or deleting a directory should
    ever wait on. The other three groups call it because they install things.
    """
    calls: list[str] = []
    monkeypatch.setattr(dev, "_exec_into_venv_if_available",
                        lambda: calls.append("venv"))
    monkeypatch.setattr(dev, "_ensure_uv", lambda: calls.append("uv"))
    monkeypatch.setattr(dev, "_apply_dev_env", lambda: calls.append("env"))
    monkeypatch.setattr(cache, "main",
                        lambda argv: calls.append(f"main:{argv}") or 5)
    monkeypatch.setattr(sys, "argv", ["cdui", "cache", "prune", "--yes"])

    assert dev._dispatch_cache_subcommand() == 5
    assert calls == ["venv", "env", "main:['prune', '--yes']"]


# ── the inventory ─────────────────────────────────────────────────────────


def test_the_inventory_is_exactly_the_derived_caches():
    """Asserted whole, because this list decides what ``prune`` deletes.

    Everything named here is rebuildable from inputs that are still on disk.
    Nothing else under any cache directory belongs in it: the pack model
    cache and the downloaded assets are under the USER data root, not the
    data root, and they are downloads rather than derivations -- deleting one
    costs bandwidth, not CPU.
    """
    assert tuple(name for name, _label in cache.DERIVED_CACHES) == ("lm_blocks",)


def test_the_cache_root_is_the_data_root_s_cache_directory(monkeypatch,
                                                           tmp_path):
    """The same directory ``LMTokenizedDataset`` writes into.

    Derived from ``settings`` on every call rather than captured at import,
    so project mode (which repoints ``MODELS_DIR``) moves the cache and this
    command follows it.
    """
    from app.config import settings

    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "assets" / "models")
    assert cache.derived_cache_root() == (tmp_path / "assets" / "cache").resolve()


def test_the_server_check_is_the_one_dev_py_already_makes(monkeypatch):
    """Not a second reading of the pidfile: ``dev`` owns that question, and
    its answer also clears a stale pidfile as it goes."""
    monkeypatch.setattr(dev, "_running_server_pid", lambda: 4321)
    assert cache._server_pid() == 4321


# ── cdui cache list ───────────────────────────────────────────────────────


def test_list_names_each_cache_with_its_count_and_size(cache_root, capsys):
    _blocks(cache_root, "a.pt", "b.pt", size=2048)

    assert cache.main(["list"]) == 0

    out = capsys.readouterr().out
    assert "lm_blocks" in out
    assert "2 entries" in out
    assert "4.0 KB" in out
    # The path, because the next thing somebody does is go and look at it.
    assert str(cache_root) in out


def test_list_counts_a_nested_cache_dir_as_one_entry(cache_root, capsys):
    """``cache_dir=my_corpus`` nests a DIRECTORY under ``lm_blocks``.

    One experiment's worth of blocks is one thing to delete, so it is counted
    and measured as one entry rather than as the files inside it.
    """
    nested = cache_root / "lm_blocks" / "my_corpus"
    nested.mkdir(parents=True)
    (nested / "one.pt").write_bytes(b"x" * 1024)
    (nested / "two.pt").write_bytes(b"x" * 1024)

    assert cache.main(["list"]) == 0

    out = capsys.readouterr().out
    assert "1 entry" in out
    assert "2.0 KB" in out


def test_list_says_when_there_is_nothing_cached(cache_root, capsys):
    """An absent directory is "nothing cached", not an error -- a fresh
    install has never run ``LMTokenizedDataset``."""
    assert cache.main(["list"]) == 0

    out = capsys.readouterr().out
    assert "0 entries" in out
    assert "cdui cache prune" not in out, \
        "nothing to prune, so nothing should be suggested"


def test_list_points_at_prune_when_there_is_something_to_delete(cache_root,
                                                               capsys):
    _blocks(cache_root, "a.pt")
    assert cache.main(["list"]) == 0
    assert "cdui cache prune" in capsys.readouterr().out


def test_the_columns_are_padded_by_display_width_not_character_count():
    """A Chinese character is two terminal columns wide. Padded by character
    count, the zh-TW rendering of the row drifts and the size column stops
    lining up -- which on a one-row table is the whole table."""
    assert cache._pad("項目", 6) == "項目  "
    assert cache._pad("ab", 6) == "ab    "


# ── cdui cache prune ──────────────────────────────────────────────────────


def test_prune_deletes_the_entries_and_reports_what_it_freed(cache_root,
                                                             capsys):
    made = _blocks(cache_root, "a.pt", "b.pt", size=2048)

    assert cache.main(["prune", "--yes"]) == 0

    assert [p for p in made if p.exists()] == []
    out = capsys.readouterr().out
    assert "deleted 2 entries" in out
    assert "4.0 KB" in out


def test_prune_keeps_the_cache_directory_itself(cache_root):
    """Only the entries go. An absent ``lm_blocks/`` would be recreated on the
    next run anyway, but deleting it turns a prune into a directory the user
    did not ask about."""
    _blocks(cache_root, "a.pt")
    assert cache.main(["prune", "--yes"]) == 0
    assert (cache_root / "lm_blocks").is_dir()


def test_prune_removes_a_nested_cache_dir_whole(cache_root):
    nested = cache_root / "lm_blocks" / "my_corpus"
    nested.mkdir(parents=True)
    (nested / "one.pt").write_bytes(b"x" * 512)

    assert cache.main(["prune", "--yes"]) == 0

    assert not nested.exists()


def test_prune_asks_before_deleting_and_a_no_keeps_everything(
        cache_root, monkeypatch, capsys):
    (block,) = _blocks(cache_root, "a.pt")
    asked: list[str] = []
    monkeypatch.setattr(cache, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input",
                        lambda prompt: asked.append(prompt) or "n")

    assert cache.main(["prune"]) == 1

    assert block.exists()
    assert asked and "[y/N]" in asked[0]
    assert "Cancelled" in capsys.readouterr().out


def test_yes_skips_the_prompt(cache_root, monkeypatch):
    """``--yes`` is for CI and for a person who has already decided."""
    _blocks(cache_root, "a.pt")
    monkeypatch.setattr(cache, "_stdin_is_interactive", lambda: True)

    def _never(prompt):
        raise AssertionError("--yes still stopped to ask")

    monkeypatch.setattr("builtins.input", _never)
    assert cache.main(["prune", "--yes"]) == 0


def test_with_no_terminal_prune_refuses_rather_than_deciding_yes(
        cache_root, monkeypatch, capsys):
    """Fails closed: a CI job must not block on a prompt nobody will answer,
    and must not delete somebody's cache on their behalf either."""
    (block,) = _blocks(cache_root, "a.pt")
    monkeypatch.setattr(cache, "_stdin_is_interactive", lambda: False)

    assert cache.main(["prune"]) == 2

    assert block.exists()
    assert "--yes" in capsys.readouterr().err


def test_older_than_keeps_the_young_entries(cache_root, capsys):
    old, young = _blocks(cache_root, "old.pt", "young.pt")
    _age(old, 40)
    _age(young, 1)

    assert cache.main(["prune", "--older-than", "30", "--yes"]) == 0

    assert not old.exists()
    assert young.exists()
    # Said out loud: a prune that silently kept half of what somebody asked
    # it to delete looks like a prune that did not work.
    assert "1 entry newer than 30 days kept" in capsys.readouterr().out


def test_older_than_measures_the_newest_file_inside_a_nested_dir(cache_root):
    """A directory refreshed yesterday is not a month-old entry, even though
    the directory's own mtime may never have moved."""
    nested = cache_root / "lm_blocks" / "my_corpus"
    nested.mkdir(parents=True)
    stale = nested / "old.pt"
    fresh = nested / "new.pt"
    stale.write_bytes(b"x")
    fresh.write_bytes(b"x")
    _age(stale, 90)
    _age(nested, 90)
    _age(fresh, 1)

    assert cache.main(["prune", "--older-than", "30", "--yes"]) == 0

    assert nested.exists()


def test_prune_says_when_there_is_nothing_to_delete(cache_root, capsys):
    assert cache.main(["prune", "--yes"]) == 0
    assert "Nothing to delete" in capsys.readouterr().out


def test_prune_refuses_while_a_server_is_running(cache_root, monkeypatch,
                                                 capsys):
    """A graph in that server may be part-way through reading a block file;
    losing it mid-run is a failed training run rather than a slow one."""
    (block,) = _blocks(cache_root, "a.pt")
    monkeypatch.setattr(dev, "_running_server_pid", lambda: 2468)

    # `--yes` and all: the refusal is not the confirmation prompt.
    assert cache.main(["prune", "--yes"]) == 3

    assert block.exists()
    captured = capsys.readouterr()
    assert "2468" in captured.err
    assert "cdui stop" in captured.out


def test_a_negative_older_than_is_refused(cache_root, capsys):
    """Reads as "younger than", which would delete exactly what was meant to
    be kept -- so it is a refusal, not a clamp."""
    (block,) = _blocks(cache_root, "a.pt")

    assert cache.main(["prune", "--older-than", "-1", "--yes"]) == 2

    assert block.exists()
    assert "--older-than" in capsys.readouterr().err


def test_one_undeletable_entry_does_not_stop_the_others(cache_root,
                                                        monkeypatch, capsys):
    """A file another process is holding open costs that file, not the run --
    and the exit code still says something went wrong."""
    stuck, other = _blocks(cache_root, "a.pt", "b.pt")
    real_unlink = Path.unlink

    def _unlink(self, *args, **kwargs):
        if self.name == "a.pt":
            raise OSError(13, "held open by another process")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _unlink)

    assert cache.main(["prune", "--yes"]) == 1

    assert stuck.exists()
    assert not other.exists()
    captured = capsys.readouterr()
    assert "held open by another process" in captured.err
    assert "deleted 1 entry" in captured.out
