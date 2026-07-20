"""Build-stamp dist staleness check (`cdui start`).

Regression suite for the stale-dist false positive on release installs:
git checkout stamps frontend/src with "now" while tar extraction restores
the CI build's mtimes, so a pure mtime comparison flags every end-user
install as stale. The check now trusts frontend/dist/build-info.json
(written by release-build.yml and local `cdui build`) and only falls back
to mtimes for dirty/legacy developer checkouts.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

import pytest

import dev  # scripts/dev.py — conftest puts scripts/ on sys.path

HEAD = "a" * 40
OTHER = "b" * 40


def _mk_frontend(tmp_path, monkeypatch, *, src_newer=True):
    """End-user shape: full checkout with src/, prebuilt dist/index.html.

    src_newer=True mirrors a release install (checkout "now", dist built
    hours earlier by CI).
    """
    frontend = tmp_path / "frontend"
    src = frontend / "src"
    src.mkdir(parents=True)
    dist = frontend / "dist"
    dist.mkdir()
    index = dist / "index.html"
    index.write_text("<html></html>", encoding="utf-8")
    srcfile = src / "App.tsx"
    srcfile.write_text("export {}", encoding="utf-8")
    now = time.time()
    if src_newer:
        os.utime(index, (now - 41 * 3600, now - 41 * 3600))
        os.utime(srcfile, (now, now))
    else:
        os.utime(index, (now, now))
        os.utime(srcfile, (now - 3600, now - 3600))
    monkeypatch.setattr(dev, "FRONTEND_DIR", frontend)
    monkeypatch.setattr(dev, "DIST_DIR", dist)
    monkeypatch.setattr(dev, "DIST_INDEX", index)
    return dist


def _stamp(dist, commit=HEAD, tag="1.4.1", source="release-build"):
    (dist / "build-info.json").write_text(
        json.dumps(
            {"tag": tag, "commit": commit, "built_at": "2026-07-18T10:25:00Z", "source": source}
        ),
        encoding="utf-8",
    )


def _fake_git(monkeypatch, head=HEAD, dirty=False, unchanged=False):
    monkeypatch.setattr(dev, "_git_head_commit", lambda: head)
    monkeypatch.setattr(dev, "_git_frontend_src_dirty", lambda: dirty)
    monkeypatch.setattr(dev, "_git_frontend_unchanged_since", lambda commit: unchanged)


def _fake_pnpm(monkeypatch, present):
    monkeypatch.setattr(
        dev.shutil, "which", lambda cmd: ("C:/pnpm.cmd" if present else None)
    )


# ── stamp matches HEAD ────────────────────────────────────────────────────────


def test_release_install_matching_stamp_is_silent(tmp_path, monkeypatch, capsys):
    """The bug fix: pristine checkout + matching stamp -> no warning even
    though src mtimes are hours newer than the extracted dist."""
    dist = _mk_frontend(tmp_path, monkeypatch, src_newer=True)
    _stamp(dist, commit=HEAD)
    _fake_git(monkeypatch, head=HEAD, dirty=False)
    _fake_pnpm(monkeypatch, present=False)
    dev._warn_if_dist_stale()
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("pnpm_present,advice", [(False, "cdui update"), (True, "cdui build")])
def test_matching_stamp_dirty_src_newer_warns(tmp_path, monkeypatch, capsys, pnpm_present, advice):
    dist = _mk_frontend(tmp_path, monkeypatch, src_newer=True)
    _stamp(dist, commit=HEAD)
    _fake_git(monkeypatch, head=HEAD, dirty=True)
    _fake_pnpm(monkeypatch, present=pnpm_present)
    dev._warn_if_dist_stale()
    err = capsys.readouterr().err
    assert "dist" in err
    assert advice in err


def test_matching_stamp_dirty_src_older_is_silent(tmp_path, monkeypatch, capsys):
    dist = _mk_frontend(tmp_path, monkeypatch, src_newer=False)
    _stamp(dist, commit=HEAD)
    _fake_git(monkeypatch, head=HEAD, dirty=True)
    _fake_pnpm(monkeypatch, present=True)
    dev._warn_if_dist_stale()
    assert capsys.readouterr().err == ""


def test_matching_stamp_dirty_unknown_is_silent(tmp_path, monkeypatch, capsys):
    """git status failing (None) counts as not-dirty -> quiet."""
    dist = _mk_frontend(tmp_path, monkeypatch, src_newer=True)
    _stamp(dist, commit=HEAD)
    monkeypatch.setattr(dev, "_git_head_commit", lambda: HEAD)
    monkeypatch.setattr(dev, "_git_frontend_src_dirty", lambda: None)
    _fake_pnpm(monkeypatch, present=True)
    dev._warn_if_dist_stale()
    assert capsys.readouterr().err == ""


# ── stamp mismatch ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pnpm_present,advice", [(False, "cdui update"), (True, "cdui build")])
def test_stamp_commit_mismatch_warns_regardless_of_mtime(
    tmp_path, monkeypatch, capsys, pnpm_present, advice
):
    """src deliberately OLDER than dist: the mismatch warning must not be
    mtime-driven."""
    dist = _mk_frontend(tmp_path, monkeypatch, src_newer=False)
    _stamp(dist, commit=OTHER, tag="1.4.0")
    _fake_git(monkeypatch, head=HEAD, dirty=False)
    _fake_pnpm(monkeypatch, present=pnpm_present)
    dev._warn_if_dist_stale()
    err = capsys.readouterr().err
    assert OTHER[:12] in err
    assert HEAD[:12] in err
    assert "1.4.0" in err  # the stamped tag names the version the dist came from
    assert advice in err


def test_mismatch_backend_only_commit_is_silent(tmp_path, monkeypatch, capsys):
    """Committing backend-only work after a build must not nag: frontend/
    unchanged between the stamped commit and HEAD counts as in sync."""
    dist = _mk_frontend(tmp_path, monkeypatch, src_newer=True)
    _stamp(dist, commit=OTHER)
    _fake_git(monkeypatch, head=HEAD, dirty=False, unchanged=True)
    _fake_pnpm(monkeypatch, present=True)
    dev._warn_if_dist_stale()
    assert capsys.readouterr().err == ""


def test_mismatch_equivalent_but_dirty_falls_back_to_mtime(tmp_path, monkeypatch, capsys):
    dist = _mk_frontend(tmp_path, monkeypatch, src_newer=True)
    _stamp(dist, commit=OTHER)
    _fake_git(monkeypatch, head=HEAD, dirty=True, unchanged=True)
    _fake_pnpm(monkeypatch, present=True)
    dev._warn_if_dist_stale()
    err = capsys.readouterr().err
    assert "dist mtime" in err
    assert "cdui build" in err


def test_mismatch_undecidable_equivalence_warns(tmp_path, monkeypatch, capsys):
    """Shallow clone that lost the stamped commit (None) must still warn."""
    dist = _mk_frontend(tmp_path, monkeypatch, src_newer=False)
    _stamp(dist, commit=OTHER)
    _fake_git(monkeypatch, head=HEAD, dirty=False, unchanged=None)
    _fake_pnpm(monkeypatch, present=False)
    dev._warn_if_dist_stale()
    err = capsys.readouterr().err
    assert OTHER[:12] in err
    assert "cdui update" in err


def test_git_helpers_survive_none_stdout(monkeypatch):
    """A dead pipe reader (locale decode failure) yields stdout=None; the
    helpers must degrade to unknown instead of raising."""

    class _Out:
        returncode = 0
        stdout = None

    monkeypatch.setattr(dev.subprocess, "run", lambda *a, **k: _Out())
    assert dev._git_head_commit() is None
    assert dev._git_exact_tag() is None
    assert dev._git_frontend_src_dirty() is None


def test_stamp_present_but_head_unknown_is_silent(tmp_path, monkeypatch, capsys):
    dist = _mk_frontend(tmp_path, monkeypatch, src_newer=True)
    _stamp(dist, commit=HEAD)
    monkeypatch.setattr(dev, "_git_head_commit", lambda: None)
    monkeypatch.setattr(dev, "_git_frontend_src_dirty", lambda: True)
    _fake_pnpm(monkeypatch, present=True)
    dev._warn_if_dist_stale()
    assert capsys.readouterr().err == ""


# ── no stamp (legacy dist, pre-1.4.1) ────────────────────────────────────────


def test_no_stamp_with_pnpm_keeps_legacy_mtime_warning(tmp_path, monkeypatch, capsys):
    _mk_frontend(tmp_path, monkeypatch, src_newer=True)
    _fake_git(monkeypatch, head=HEAD, dirty=False)
    _fake_pnpm(monkeypatch, present=True)
    dev._warn_if_dist_stale()
    err = capsys.readouterr().err
    assert "dist" in err
    assert "cdui build" in err


def test_no_stamp_without_pnpm_is_silent(tmp_path, monkeypatch, capsys):
    _mk_frontend(tmp_path, monkeypatch, src_newer=True)
    _fake_git(monkeypatch, head=HEAD, dirty=False)
    _fake_pnpm(monkeypatch, present=False)
    dev._warn_if_dist_stale()
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("payload", ["{not json", "[1, 2]", '{"commit": 123}'])
def test_corrupt_stamp_treated_as_no_stamp(tmp_path, monkeypatch, capsys, payload):
    """Corrupt/foreign build-info.json falls back to the legacy branch:
    warns when pnpm is present, silent when it is not."""
    dist = _mk_frontend(tmp_path, monkeypatch, src_newer=True)
    (dist / "build-info.json").write_text(payload, encoding="utf-8")
    _fake_git(monkeypatch, head=HEAD, dirty=False)
    _fake_pnpm(monkeypatch, present=True)
    dev._warn_if_dist_stale()
    assert "cdui build" in capsys.readouterr().err
    _fake_pnpm(monkeypatch, present=False)
    dev._warn_if_dist_stale()
    assert capsys.readouterr().err == ""


# ── stamp writing ────────────────────────────────────────────────────────────


def test_write_build_stamp_schema(tmp_path, monkeypatch):
    dist = _mk_frontend(tmp_path, monkeypatch)
    monkeypatch.setattr(dev, "_git_head_commit", lambda: HEAD)
    monkeypatch.setattr(dev, "_git_exact_tag", lambda: "1.4.1")
    dev._write_build_stamp("local-build")
    stamp = json.loads((dist / "build-info.json").read_text(encoding="utf-8"))
    assert stamp["commit"] == HEAD
    assert stamp["tag"] == "1.4.1"
    assert stamp["source"] == "local-build"
    assert "built_at" in stamp


def test_write_build_stamp_swallows_oserror(tmp_path, monkeypatch):
    monkeypatch.setattr(dev, "DIST_DIR", tmp_path / "missing" / "dist")
    monkeypatch.setattr(dev, "_git_head_commit", lambda: None)
    monkeypatch.setattr(dev, "_git_exact_tag", lambda: None)
    dev._write_build_stamp("local-build")  # must not raise


def test_build_writes_stamp(tmp_path, monkeypatch):
    dist = _mk_frontend(tmp_path, monkeypatch)
    (tmp_path / "frontend" / "node_modules").mkdir()
    calls = []
    monkeypatch.setattr(dev, "run", lambda cmd, cwd=None: calls.append(cmd))
    monkeypatch.setattr(dev.shutil, "which", lambda cmd: "C:/pnpm.cmd")
    monkeypatch.setattr(dev, "_git_head_commit", lambda: HEAD)
    monkeypatch.setattr(dev, "_git_exact_tag", lambda: None)
    dev.build()
    assert ["pnpm", "build"] in calls
    stamp = json.loads((dist / "build-info.json").read_text(encoding="utf-8"))
    assert stamp["source"] == "local-build"
    assert stamp["commit"] == HEAD
    assert stamp["tag"] is None


def test_build_without_pnpm_exits_1(tmp_path, monkeypatch):
    _mk_frontend(tmp_path, monkeypatch)
    monkeypatch.setattr(dev.shutil, "which", lambda cmd: None)
    with pytest.raises(SystemExit) as exc:
        dev.build()
    assert exc.value.code == 1


# ── real-git integration for the helpers ─────────────────────────────────────


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=True)


def test_git_helpers_against_real_repo(tmp_path, monkeypatch):
    if not shutil.which("git"):
        pytest.skip("git not installed")
    _git("init", cwd=tmp_path)
    _git("config", "user.email", "t@example.com", cwd=tmp_path)
    _git("config", "user.name", "t", cwd=tmp_path)
    src = tmp_path / "frontend" / "src"
    src.mkdir(parents=True)
    (src / "a.ts").write_text("1", encoding="utf-8")
    _git("add", "-A", cwd=tmp_path)
    _git("commit", "-m", "init", cwd=tmp_path)
    monkeypatch.setattr(dev, "ROOT", tmp_path)

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert dev._git_head_commit() == head
    assert dev._git_frontend_src_dirty() is False

    (src / "a.ts").write_text("2", encoding="utf-8")
    assert dev._git_frontend_src_dirty() is True

    # frontend edit committed, then a backend-only commit on top: the dist
    # stamped at the frontend commit stays equivalent to HEAD.
    _git("add", "-A", cwd=tmp_path)
    _git("commit", "-m", "src edit", cwd=tmp_path)
    frontend_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True
    ).stdout.strip()
    (tmp_path / "backend.py").write_text("x", encoding="utf-8")
    _git("add", "-A", cwd=tmp_path)
    _git("commit", "-m", "backend only", cwd=tmp_path)
    assert dev._git_frontend_unchanged_since(frontend_head) is True
    assert dev._git_frontend_unchanged_since(head) is False
    assert dev._git_frontend_unchanged_since("0" * 40) is None


def test_git_head_commit_none_outside_repo(tmp_path, monkeypatch):
    if not shutil.which("git"):
        pytest.skip("git not installed")
    lonely = tmp_path / "not-a-repo"
    lonely.mkdir()
    monkeypatch.setattr(dev, "ROOT", lonely)
    assert dev._git_head_commit() is None
