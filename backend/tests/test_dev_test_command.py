"""`cdui test` has to mean the whole project (core#245 item 3).

It used to be three lines -- ``pytest`` in ``backend/``, nothing else -- while
138 frontend test files went unrun. A contributor could see a green
``cdui test``, push, and fail ``pnpm test`` in CI, and CONTRIBUTING.md points
people at exactly this command.

Two properties are load-bearing and both are pinned below:

1. **Both halves run, and neither hides the other.** A red backend must not
   stop the frontend from running, because the point of one command is
   learning about both in one pass.
2. **A missing pnpm is a skip, not a failure, and it SAYS so.** CodefyUI has a
   deliberate no-Node install path (``frontend-dist.tar.gz`` from the release),
   so "no pnpm on this machine" is a healthy install, not a broken one. What
   would be dangerous is reporting that as a pass.

Repo gotcha this file obeys: stubbing ``run`` alone is not enough when testing
``dev.py`` -- ``ROOT``, ``VENV``, ``DIST_DIR`` and ``FRONTEND_DIR`` are read at
call time and any ``shutil.rmtree`` in the command under test executes for
real. Every test here redirects the directory constants at ``tmp_path`` and
asserts at the end that the developer's real ``frontend/`` is untouched.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

import dev  # scripts/dev.py -- conftest puts scripts/ on sys.path


REAL_FRONTEND = Path(dev.__file__).resolve().parent.parent / "frontend"


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point every path constant `test()` can reach at a throwaway tree.

    Returns a recorder: ``calls`` is the list of (cmd, cwd) pairs the command
    would have run, and ``codes`` lets a test make a specific command fail.
    """
    frontend = tmp_path / "frontend"
    (frontend / "node_modules").mkdir(parents=True)
    (frontend / "package.json").write_text('{"name": "x"}', encoding="utf-8")
    (frontend / "dist").mkdir()
    backend = tmp_path / "backend"
    backend.mkdir()

    monkeypatch.setattr(dev, "ROOT", tmp_path)
    monkeypatch.setattr(dev, "BACKEND_DIR", backend)
    monkeypatch.setattr(dev, "FRONTEND_DIR", frontend)
    monkeypatch.setattr(dev, "DIST_DIR", frontend / "dist")
    monkeypatch.setattr(dev, "VENV", tmp_path / "venv")
    monkeypatch.setattr(dev, "_require_venv_tool", lambda name: f"/fake/{name}")
    monkeypatch.setattr(sys, "argv", ["cdui", "test"])
    # `t()` reads this global at call time, and it is derived from the
    # developer's locale. Pin it so the assertions below are about the
    # command's behaviour rather than about which machine ran them.
    monkeypatch.setattr(dev, "LANG", "en")

    class Recorder:
        def __init__(self):
            self.calls: list[tuple[list[str], Path]] = []
            self.codes: dict[str, int] = {}

        def cmds(self) -> list[list[str]]:
            return [c for c, _ in self.calls]

        def __call__(self, cmd, cwd):
            self.calls.append((list(cmd), cwd))
            return self.codes.get(" ".join(str(p) for p in cmd), 0)

    rec = Recorder()
    monkeypatch.setattr(dev, "_run_status", rec)
    rec.frontend = frontend
    rec.backend = backend
    return rec


@pytest.fixture(autouse=True)
def _pnpm_present(monkeypatch):
    """Default to a machine WITH pnpm; the no-Node tests override this."""
    monkeypatch.setattr(dev.shutil, "which", lambda name: f"/usr/bin/{name}")


@pytest.fixture(autouse=True)
def _real_frontend_survives():
    """The rmtree trap, as an assertion rather than a comment.

    ``dev.py`` commands delete ``frontend/dist`` and ``frontend/node_modules``
    for real. If a future edit to ``test()`` ever grows a cleanup step, this
    notices before it eats the developer's build.
    """
    before = REAL_FRONTEND.exists() and sorted(p.name for p in REAL_FRONTEND.iterdir())
    yield
    after = REAL_FRONTEND.exists() and sorted(p.name for p in REAL_FRONTEND.iterdir())
    assert before == after, "cdui test touched the real frontend/ tree"


def _run(expect_exit: int | None = None):
    """Call `dev.test()`, asserting on the SystemExit it did or did not raise."""
    if expect_exit is None:
        dev.test()          # a clean run returns rather than exiting
        return
    with pytest.raises(SystemExit) as exc:
        dev.test()
    assert exc.value.code == expect_exit


# ── both halves run ────────────────────────────────────────────────────────


def test_it_runs_the_backend_and_the_frontend(sandbox, capsys):
    _run()
    assert sandbox.cmds() == [["/fake/pytest"], ["pnpm", "test"]]
    assert [cwd for _, cwd in sandbox.calls] == [sandbox.backend, sandbox.frontend]


def test_the_summary_reports_both_halves_as_run(sandbox, capsys):
    _run()
    out = capsys.readouterr().out
    assert "backend" in out and "frontend" in out
    assert "SKIPPED" not in out


def test_a_red_backend_does_not_stop_the_frontend(sandbox):
    """The regression this command exists to prevent, in reverse: the whole
    point of running both is finding out about both in ONE pass."""
    sandbox.codes["/fake/pytest"] = 1
    _run(expect_exit=1)
    assert ["pnpm", "test"] in sandbox.cmds(), "frontend was skipped after a red backend"


def test_a_red_frontend_fails_the_command(sandbox, capsys):
    sandbox.codes["pnpm test"] = 1
    _run(expect_exit=1)
    assert "FAIL" in capsys.readouterr().out


def test_node_modules_are_installed_on_first_run(sandbox):
    shutil.rmtree(sandbox.frontend / "node_modules")
    _run()
    assert sandbox.cmds() == [["/fake/pytest"], ["pnpm", "install"], ["pnpm", "test"]]


def test_a_failed_pnpm_install_does_not_pretend_the_tests_ran(sandbox, capsys):
    """`run()` would have raised CalledProcessError here and killed the whole
    command mid-way, with the backend result already printed and no summary."""
    shutil.rmtree(sandbox.frontend / "node_modules")
    sandbox.codes["pnpm install"] = 1
    _run(expect_exit=1)
    assert ["pnpm", "test"] not in sandbox.cmds()
    assert "FAIL" in capsys.readouterr().out


# ── the no-Node install path ───────────────────────────────────────────────


def test_without_pnpm_the_backend_still_runs_and_the_command_succeeds(
        sandbox, monkeypatch):
    """A release install has no Node at all. That is a supported, healthy
    machine -- `cdui test` must not fail on it."""
    monkeypatch.setattr(dev.shutil, "which", lambda name: None)
    _run()
    assert sandbox.cmds() == [["/fake/pytest"]]


def test_without_pnpm_the_summary_says_skipped_not_passed(sandbox, monkeypatch, capsys):
    """The dangerous outcome is not "the frontend did not run" -- it is a
    green summary that does not admit it."""
    monkeypatch.setattr(dev.shutil, "which", lambda name: None)
    _run()
    out, errout = capsys.readouterr()
    frontend_line = next(ln for ln in out.splitlines() if "frontend" in ln)
    assert "SKIPPED" in frontend_line
    assert "PASS" not in frontend_line
    # ... and it says why, and how to fix it.
    assert "pnpm" in errout and "frontend-build.yml" in errout


def test_a_checkout_without_a_frontend_is_a_skip_not_a_crash(sandbox, capsys):
    (sandbox.frontend / "package.json").unlink()
    _run()
    assert sandbox.cmds() == [["/fake/pytest"]]
    assert "SKIPPED" in capsys.readouterr().out


# ── selectors ──────────────────────────────────────────────────────────────


def test_backend_flag_skips_the_frontend_even_when_pnpm_exists(sandbox, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cdui", "test", "--backend"])
    _run()
    assert sandbox.cmds() == [["/fake/pytest"]]


def test_frontend_flag_skips_the_backend(sandbox, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cdui", "test", "--frontend"])
    _run()
    assert sandbox.cmds() == [["pnpm", "test"]]


def test_both_selectors_together_mean_run_everything(sandbox, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cdui", "test", "--backend", "--frontend"])
    _run()
    assert sandbox.cmds() == [["/fake/pytest"], ["pnpm", "test"]]


def test_an_unknown_argument_is_refused_rather_than_ignored(sandbox, monkeypatch, capsys):
    """`cdui test -k foo` used to run the entire suite and look like it had
    filtered. Exiting 2 is the difference between a typo and a lie."""
    monkeypatch.setattr(sys, "argv", ["cdui", "test", "-k", "foo"])
    _run(expect_exit=2)
    assert sandbox.cmds() == []
    assert "-k" in capsys.readouterr().err


# ── the summary formatter ──────────────────────────────────────────────────


def test_the_three_states_are_distinguishable(monkeypatch):
    """PASS / FAIL / SKIPPED must not collapse into each other: `_SKIPPED` is
    ``None`` and ``0`` is a pass, and `0 == False` in Python."""
    monkeypatch.setattr(dev, "LANG", "en")
    passed = dev._test_summary_line("backend", 0)
    failed = dev._test_summary_line("backend", 1)
    skipped = dev._test_summary_line("backend", dev._SKIPPED)
    assert "PASS" in passed and "FAIL" not in passed and "SKIPPED" not in passed
    assert "FAIL" in failed and "exit 1" in failed
    assert "SKIPPED" in skipped and "PASS" not in skipped


def test_the_three_states_are_distinguishable_in_chinese_too(monkeypatch):
    """The summary is the only place the command reports what it did, so it
    has to be readable in both locales -- and the three states must stay
    distinct in both."""
    monkeypatch.setattr(dev, "LANG", "zh")
    rendered = [dev._test_summary_line("backend", code)
                for code in (0, 1, dev._SKIPPED)]
    assert len({r for r in rendered}) == 3
    assert "通過" in rendered[0]
    assert "失敗" in rendered[1]
    assert "略過" in rendered[2]
