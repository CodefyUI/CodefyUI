"""git provenance + stale-pin detection used by the startup transparency log
(spec 7.4)."""

import shutil
import subprocess

import pytest

from app.core.project import (
    check_stale_pins,
    check_stale_pins_from_manifest,
    git_provenance,
)


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=True)


def test_git_provenance_non_repo(tmp_path):
    assert git_provenance(tmp_path) == (None, None)


def test_git_provenance_repo_no_commits(tmp_path):
    """`cdui project init` runs `git init` and deliberately makes NO initial
    commit (scripts/project.py cmd_init) -- so a freshly-scaffolded project is
    a git repo with an unborn HEAD. That is the common post-init state, not
    an edge case, and must degrade the same as a non-repo dir: (None, None)."""
    if not shutil.which("git"):
        pytest.skip("git not installed")
    _git("init", cwd=tmp_path)
    assert git_provenance(tmp_path) == (None, None)


def test_git_provenance_clean_then_dirty(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git not installed")
    _git("init", cwd=tmp_path)
    _git("config", "user.email", "t@t.t", cwd=tmp_path)
    _git("config", "user.name", "t", cwd=tmp_path)
    (tmp_path / "a.txt").write_text("1")
    _git("add", "-A", cwd=tmp_path)
    _git("commit", "-m", "init", cwd=tmp_path)
    commit, dirty = git_provenance(tmp_path)
    assert commit and len(commit) == 40
    assert dirty is False
    (tmp_path / "a.txt").write_text("2")
    _commit, dirty2 = git_provenance(tmp_path)
    assert dirty2 is True


def test_git_provenance_dirty_with_unicode_filename(tmp_path):
    """A non-ASCII untracked filename must not crash the STATUS call. This is
    a regression guard for forcing utf-8 decoding of git's output rather than
    the platform-default encoding, which on this project's actual Windows
    deployments (Traditional Chinese locale) is cp950 and cannot decode
    arbitrary UTF-8 bytes -- exactly the crash class called out in project
    memory for other subprocess/console output."""
    if not shutil.which("git"):
        pytest.skip("git not installed")
    _git("init", cwd=tmp_path)
    _git("config", "user.email", "t@t.t", cwd=tmp_path)
    _git("config", "user.name", "t", cwd=tmp_path)
    (tmp_path / "a.txt").write_text("1")
    _git("add", "-A", cwd=tmp_path)
    _git("commit", "-m", "init", cwd=tmp_path)
    # Built from chr() codepoints (0x65e5 0x672c 0x8a9e) so this source file
    # stays ASCII-only; the resulting filename is still non-ASCII (CJK) at
    # runtime, which is what this test targets.
    cjk_name = chr(0x65E5) + chr(0x672C) + chr(0x8A9E) + ".txt"
    (tmp_path / cjk_name).write_text("x", encoding="utf-8")
    commit, dirty = git_provenance(tmp_path)
    assert commit and len(commit) == 40
    assert dirty is True


def test_git_provenance_status_call_failure_keeps_commit(tmp_path, monkeypatch):
    """If the STATUS call fails (timeout / spawn error) after a successful
    rev-parse, degrade to dirty=None instead of losing the already-resolved
    commit -- the original status call was unguarded, asymmetric with the
    try/except already wrapping the rev-parse call."""
    if not shutil.which("git"):
        pytest.skip("git not installed")
    _git("init", cwd=tmp_path)
    _git("config", "user.email", "t@t.t", cwd=tmp_path)
    _git("config", "user.name", "t", cwd=tmp_path)
    (tmp_path / "a.txt").write_text("1")
    _git("add", "-A", cwd=tmp_path)
    _git("commit", "-m", "init", cwd=tmp_path)

    real_run = subprocess.run

    def _flaky_run(cmd, *a, **k):
        if "status" in cmd:
            raise subprocess.TimeoutExpired(cmd, 5)
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(subprocess, "run", _flaky_run)
    commit, dirty = git_provenance(tmp_path)
    assert commit and len(commit) == 40
    assert dirty is None


def test_check_stale_pins():
    manifest_dir = None  # unused; pins read from a written manifest
    lockfile = {"plugins": {"pack-a": {"sha": "a" * 40}}}
    # pack-a present + matching -> not stale; pack-b missing -> stale;
    # pack-a mismatched sha -> stale.
    ok_pins = {"plugins": {"pack-a": {"sha": "a" * 40}}}
    assert check_stale_pins_from_manifest(ok_pins, lockfile) == []
    missing = {"plugins": {"pack-b": {"sha": "b" * 40}}}
    assert check_stale_pins_from_manifest(missing, lockfile) == ["pack-b"]
    mism = {"plugins": {"pack-a": {"sha": "z" * 40}}}
    assert check_stale_pins_from_manifest(mism, lockfile) == ["pack-a"]


def test_check_stale_pins_missing_or_bad_manifest_returns_empty(tmp_path):
    """check_stale_pins (the disk-reading wrapper) must never crash startup:
    no manifest at all, and an unparseable one, both resolve to []."""
    lockfile = {"plugins": {"pack-a": {"sha": "a" * 40}}}
    assert check_stale_pins(tmp_path, lockfile) == []
    (tmp_path / "codefyui.project.toml").write_text("not [ valid toml",
                                                     encoding="utf-8")
    assert check_stale_pins(tmp_path, lockfile) == []


def test_check_stale_pins_reads_manifest_from_disk(tmp_path):
    lockfile = {"plugins": {"pack-a": {"sha": "a" * 40}}}
    (tmp_path / "codefyui.project.toml").write_text(
        '[plugins]\npack-a = { sha = "' + "z" * 40 + '" }\n', encoding="utf-8")
    assert check_stale_pins(tmp_path, lockfile) == ["pack-a"]
