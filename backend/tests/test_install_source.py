"""Guards on where an install and an update fetch code from.

Four independent copies of the repo path exist -- ``install.sh``,
``install.ps1``, ``scripts/dev.py`` and the README one-liners -- in three
languages that cannot share a constant, because the shell installers run
before Python exists. So the single source of truth has to be a test.

The failure this prevents is not untidiness. A GitHub rename redirect keeps
an old ``owner/repo`` path working only until somebody creates a new repo at
that path, and the old owner name stays claimable forever. Since
``dev.fetch_release_dist`` verifies no signature or checksum, a stale path
that gets claimed turns every install and every ``cdui update`` into code
execution from a repo the maintainer does not control.
"""

from __future__ import annotations

import re
import subprocess

import pytest

import dev  # scripts/dev.py -- conftest puts scripts/ on sys.path

ROOT = dev.ROOT
EXPECTED_REPO = "CodefyUI/CodefyUI"
OLD_OWNER = "treeleaves30760"


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


# ── The four copies must agree ───────────────────────────────────────────────

def test_dev_py_release_repo():
    assert dev.RELEASE_REPO == EXPECTED_REPO


def test_install_sh_names_the_same_repo():
    text = _read("install.sh")
    assert f'RELEASE_REPO="{EXPECTED_REPO}"' in text
    assert f'REPO="https://github.com/{EXPECTED_REPO}.git"' in text


def test_install_ps1_names_the_same_repo():
    text = _read("install.ps1")
    assert f"$ReleaseRepo = '{EXPECTED_REPO}'" in text
    assert f"$Repo = 'https://github.com/{EXPECTED_REPO}.git'" in text


def test_readme_one_liners_fetch_from_the_same_repo():
    text = _read("README.md")
    for script in ("install.sh", "install.ps1"):
        url = f"https://raw.githubusercontent.com/{EXPECTED_REPO}/main/{script}"
        assert url in text, f"README does not point {script} at {EXPECTED_REPO}"


def test_every_entry_point_agrees_on_one_owner():
    """Whatever the repo is called, all four must say the same thing."""
    pattern = re.compile(r"github(?:usercontent)?\.com/([\w.-]+)/CodefyUI\b")
    owners = set()
    for rel in ("install.sh", "install.ps1", "scripts/dev.py", "README.md"):
        owners.update(pattern.findall(_read(rel)))
    owners.add(dev.RELEASE_REPO.split("/")[0])
    assert owners == {EXPECTED_REPO.split("/")[0]}, (
        f"install entry points disagree on the owner: {sorted(owners)}"
    )


# ── Nothing anywhere may still point at the pre-rename owner ─────────────────

def test_no_tracked_file_references_the_old_owner():
    # This file is excluded from its own scan: it has to name the forbidden
    # string in order to forbid it. Everything else in the tree is fair game.
    self_path = ":!backend/tests/test_install_source.py"
    try:
        out = subprocess.run(
            ["git", "grep", "-l", OLD_OWNER, "--", ".", self_path],
            cwd=ROOT, capture_output=True, text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        pytest.skip(f"git unavailable: {exc}")
    if out.returncode not in (0, 1):  # pragma: no cover
        pytest.skip(f"git grep unavailable here: {out.stderr.strip()}")
    stale = [line for line in out.stdout.splitlines() if line.strip()]
    assert not stale, (
        f"these files still reference the pre-rename owner {OLD_OWNER!r}, "
        f"which survives only on a GitHub redirect: {stale}"
    )


# ── The uv bootstrap must not be able to hang forever ────────────────────────

def test_uv_install_timeout_default(monkeypatch):
    monkeypatch.delenv("CODEFYUI_UV_INSTALL_TIMEOUT", raising=False)
    assert dev._uv_install_timeout() == 180


def test_uv_install_timeout_env_override(monkeypatch):
    monkeypatch.setenv("CODEFYUI_UV_INSTALL_TIMEOUT", "600")
    assert dev._uv_install_timeout() == 600


def test_uv_install_timeout_zero_disables_the_limit(monkeypatch):
    monkeypatch.setenv("CODEFYUI_UV_INSTALL_TIMEOUT", "0")
    assert dev._uv_install_timeout() == 0


def test_uv_install_timeout_rejects_nonsense(monkeypatch):
    monkeypatch.setenv("CODEFYUI_UV_INSTALL_TIMEOUT", "soon")
    assert dev._uv_install_timeout() == 180
    monkeypatch.setenv("CODEFYUI_UV_INSTALL_TIMEOUT", "-5")
    assert dev._uv_install_timeout() == 0


def test_ensure_uv_passes_a_timeout_to_the_installer(monkeypatch):
    monkeypatch.setattr(dev.shutil, "which", lambda _name: None)
    monkeypatch.setattr(dev, "_reexec", lambda *_a, **_k: None)
    monkeypatch.setenv("CODEFYUI_UV_INSTALL_TIMEOUT", "45")
    seen = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(dev.subprocess, "run", fake_run)
    dev._ensure_uv()
    assert seen.get("timeout") == 45


def test_ensure_uv_exits_with_advice_when_the_download_hangs(monkeypatch, capsys):
    monkeypatch.setattr(dev.shutil, "which", lambda _name: None)
    monkeypatch.setattr(dev, "_reexec", lambda *_a, **_k: None)

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="uv-install", timeout=180)

    monkeypatch.setattr(dev.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        dev._ensure_uv()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    # Must name the escape hatch, not just report failure.
    assert "CODEFYUI_UV_INSTALL_TIMEOUT" in err
    assert "docs.astral.sh" in err


def test_ensure_uv_is_a_noop_when_uv_is_already_installed(monkeypatch):
    monkeypatch.setattr(dev.shutil, "which", lambda _name: "/usr/bin/uv")

    def explode(*_a, **_k):  # pragma: no cover - must never run
        raise AssertionError("_ensure_uv shelled out despite uv being present")

    monkeypatch.setattr(dev.subprocess, "run", explode)
    dev._ensure_uv()
