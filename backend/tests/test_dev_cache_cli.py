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


# ── project mode ──────────────────────────────────────────────────────────


@pytest.fixture
def _restore_project_env():
    """``_activate_project`` writes raw ``os.environ``, and monkeypatch
    cannot undo a write it never saw -- snapshot the one key it touches.
    Same fixture ``test_dev_cli.py`` keeps, for the same reason."""
    before = os.environ.get("CODEFYUI_PROJECT_DIR")
    yield
    if before is None:
        os.environ.pop("CODEFYUI_PROJECT_DIR", None)
    else:
        os.environ["CODEFYUI_PROJECT_DIR"] = before


@pytest.fixture
def live_data_root(monkeypatch, _restore_project_env):
    """Rebuild the data root from a fresh ``Settings`` on every call.

    In production this command imports ``app`` for the FIRST time inside
    ``derived_cache_root()``, so ``Settings()`` -- and the project-mode
    derivation in its validator -- runs AFTER ``--project`` has exported
    ``CODEFYUI_PROJECT_DIR``. Under pytest ``app.config`` was imported long
    before any of these tests ran and its module-level ``settings`` froze
    the non-project roots, which would make every assertion below pass for
    the wrong reason. One ``Settings`` per call restores the production
    ordering without spawning a subprocess, and leaves the real
    ``derived_cache_root`` -- the thing under test -- in place.
    """
    import app.core.data_paths as data_paths
    from app.config import Settings

    monkeypatch.delenv("CODEFYUI_PROJECT_DIR", raising=False)
    monkeypatch.setattr(data_paths, "data_root",
                        lambda: Settings().MODELS_DIR.parent.resolve())


def _project(tmp_path: Path, name: str = "lab") -> Path:
    """A directory ``cdui start --project`` would accept."""
    proj = tmp_path / name
    proj.mkdir()
    (proj / "codefyui.project.toml").write_text(
        f'[project]\nname = "{name}"\n', encoding="utf-8")
    return proj


def test_both_subcommands_take_the_project_flag():
    """On both, because the answer and the delete are the same question
    asked twice: a flag only ``list`` accepts is one somebody types on
    ``prune`` and watches argparse reject."""
    parser = cache.build_parser()
    assert parser.parse_args(["list", "--project", "lab"]).project == "lab"
    assert parser.parse_args(["prune", "--project=lab"]).project == "lab"
    assert parser.parse_args(["list"]).project is None


def test_list_reports_the_project_s_cache_when_given_the_project(
        tmp_path, capsys, live_data_root):
    """``cdui start --project <dir>`` moves the cache to
    ``<dir>/assets/cache`` and exports that only inside its own process, so
    a ``cdui cache list`` typed in any shell answered about
    ``backend/data/cache`` -- ``0 entries`` -- while the copies filling the
    disk sat in the project. The flag is the one the server was started
    with (#306).
    """
    proj = _project(tmp_path)
    blocks = proj / "assets" / "cache" / "lm_blocks"
    blocks.mkdir(parents=True)
    (blocks / "a.pt").write_bytes(b"x" * 2048)

    assert cache.main(["list", "--project", str(proj)]) == 0

    out = capsys.readouterr().out
    assert str((proj / "assets" / "cache").resolve()) in out
    assert "1 entry" in out
    assert "2.0 KB" in out


def test_prune_with_a_project_deletes_there_and_not_from_the_default_root(
        tmp_path, monkeypatch, capsys, live_data_root):
    """The flag moves what is deleted, not just what is counted."""
    default_blocks = tmp_path / "data" / "cache" / "lm_blocks"
    default_blocks.mkdir(parents=True)
    outside = default_blocks / "keep.pt"
    outside.write_bytes(b"x" * 1024)
    monkeypatch.setenv("CODEFYUI_MODELS_DIR", str(tmp_path / "data" / "models"))

    # Control: with no flag, THAT is the directory the command works on.
    assert cache.main(["list"]) == 0
    assert "1 entry" in capsys.readouterr().out

    # An explicitly-set root beats project derivation (`_derive_project_roots`
    # in app/config.py checks `model_fields_set`), here exactly as it does on
    # the server -- so the variable goes away and `--project` is what moves
    # the root, rather than the two of them arguing.
    monkeypatch.delenv("CODEFYUI_MODELS_DIR")
    proj = _project(tmp_path)
    project_blocks = proj / "assets" / "cache" / "lm_blocks"
    project_blocks.mkdir(parents=True)
    doomed = project_blocks / "a.pt"
    doomed.write_bytes(b"x" * 1024)

    assert cache.main(["prune", "--project", str(proj), "--yes"]) == 0

    # With CODEFYUI_MODELS_DIR gone, a `--project` that stopped taking effect
    # would point the root at this checkout's real backend/data -- and the
    # prune above would have run there. Pin the root before trusting the two
    # existence checks below.
    assert cache.derived_cache_root().is_relative_to(proj), (
        "prune measured a root outside the project it was given")
    assert not doomed.exists()
    assert outside.exists(), "prune reached outside the project it was given"


def test_the_project_is_exported_the_way_cdui_start_exports_it(
        tmp_path, monkeypatch, capsys, live_data_root):
    """Absolute, and through ``dev``'s own helper.

    A relative ``--project ./lab`` has to resolve, because the variable is
    read by ``app.config`` long after anything remembers which directory it
    was typed in; and reading the manifest twice, in two files, is two
    places to fix the day project mode changes shape.
    """
    proj = _project(tmp_path)
    monkeypatch.chdir(tmp_path)

    assert cache.main(["list", "--project", "lab"]) == 0

    assert os.environ["CODEFYUI_PROJECT_DIR"] == str(proj.resolve())
    assert str(proj.resolve()) in capsys.readouterr().out


def test_a_project_with_no_manifest_is_refused_like_cdui_start(
        tmp_path, capsys, live_data_root):
    """Same check, same exit code, same sentence. A typo in the path must
    not quietly list this install's cache instead -- the reader would take
    ``0 entries`` for an answer about their project."""
    with pytest.raises(SystemExit) as exc:
        cache.main(["list", "--project", str(tmp_path / "nope")])

    assert exc.value.code == 1
    assert "manifest" in capsys.readouterr().err.lower()
    assert "CODEFYUI_PROJECT_DIR" not in os.environ


def test_an_empty_project_is_refused_not_swallowed(
        tmp_path, monkeypatch, capsys, live_data_root):
    """``--project ""`` is what ``--project "$UNSET"`` expands to in a script.
    ``cdui start`` tests the flag with ``is not None`` and refuses it (the
    manifest check runs against the working directory and fails); a truthiness
    test here would read it as "no project" and answer about this install's
    own cache -- or prune it."""
    monkeypatch.chdir(tmp_path)  # no manifest here
    with pytest.raises(SystemExit) as exc:
        cache.main(["list", "--project", ""])

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "manifest" in captured.err.lower()
    assert "entr" not in captured.out.lower(), "the listing ran anyway"


# ── what the refusal can and cannot see ───────────────────────────────────


def test_prune_help_says_a_foreground_server_must_be_stopped_by_hand(capsys):
    """``_running_server_pid`` reads ``server.pid``, which only the
    BACKGROUND branch of ``start`` writes: a ``cdui dev`` or ``cdui start
    -f`` session is invisible to the refusal, and the docs said "a server
    started from this install", which is more than the check delivers."""
    with pytest.raises(SystemExit) as exc:
        cache.main(["prune", "--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out.lower()
    assert "background" in out
    assert "foreground" in out
    assert "cdui start -f" in out
