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

import os
import re
import subprocess
import sys

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
    # Two files may name the old owner, for the same reason: they are talking
    # ABOUT it rather than pointing at it. This guard has to name the string in
    # order to forbid it, and the changelog has to name it to record the fix.
    # Everything else in the tree is fair game.
    exclusions = [
        ":!backend/tests/test_install_source.py",
        ":!CHANGELOG.md",
    ]
    try:
        out = subprocess.run(
            ["git", "grep", "-l", OLD_OWNER, "--", ".", *exclusions],
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


@pytest.fixture(autouse=True)
def _no_real_uv_install_dirs(monkeypatch, tmp_path):
    """`_ensure_uv` looks in the installer's directories under HOME; point
    them at an empty fake home so the machine running the tests (which has uv
    in ~/.local/bin more often than not) cannot answer for the test."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for name in ("XDG_BIN_HOME", "UV_INSTALL_DIR", dev._UV_BOOTSTRAP_ENV):
        monkeypatch.delenv(name, raising=False)
    return home


def _fake_uv(directory):
    directory.mkdir(parents=True, exist_ok=True)
    exe = directory / ("uv.exe" if sys.platform == "win32" else "uv")
    exe.write_text("#!/bin/sh\necho fake-uv\n", encoding="utf-8")
    exe.chmod(0o755)
    return exe


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


# ── uv installed but not on PATH must not reinstall forever (#809) ──────────

def test_ensure_uv_finds_uv_in_local_bin_without_downloading(
    monkeypatch, _no_real_uv_install_dirs,
):
    """`ssh host 'cdui ...'` reads no rc file, so the installer's ~/.local/bin
    is installed but off PATH. Found there, it goes on PATH; nothing downloads."""
    local_bin = _no_real_uv_install_dirs / ".local" / "bin"
    _fake_uv(local_bin)
    monkeypatch.setenv("PATH", "/usr/bin" + os.pathsep + "/bin")
    monkeypatch.setenv(dev._UV_BOOTSTRAP_ENV, "1")

    def explode(*_a, **_k):  # pragma: no cover - must never run
        raise AssertionError("_ensure_uv downloaded uv that was already installed")

    monkeypatch.setattr(dev.subprocess, "run", explode)
    monkeypatch.setattr(dev, "_reexec", explode)
    dev._ensure_uv()
    assert os.environ["PATH"].split(os.pathsep)[0] == str(local_bin)
    assert dev.shutil.which("uv") is not None
    assert dev._UV_BOOTSTRAP_ENV not in os.environ


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX installer locations")
def test_ensure_uv_honours_xdg_bin_home(monkeypatch, tmp_path):
    xdg = tmp_path / "xdg-bin"
    _fake_uv(xdg)
    monkeypatch.setenv("XDG_BIN_HOME", str(xdg))
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setattr(dev.subprocess, "run", lambda *_a, **_k: 1 / 0)
    dev._ensure_uv()
    assert os.environ["PATH"].split(os.pathsep)[0] == str(xdg)


def test_ensure_uv_puts_the_installed_dir_on_path_before_relaunching(
    monkeypatch, _no_real_uv_install_dirs,
):
    local_bin = _no_real_uv_install_dirs / ".local" / "bin"
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setattr(dev.shutil, "which", lambda _name: None)

    def installer(*args, **kwargs):
        _fake_uv(local_bin)
        return subprocess.CompletedProcess(args, 0)

    relaunched = {}

    def fake_reexec(*_a, **_k):
        relaunched["path"] = os.environ["PATH"]
        relaunched["marker"] = os.environ.get(dev._UV_BOOTSTRAP_ENV)

    monkeypatch.setattr(dev.subprocess, "run", installer)
    monkeypatch.setattr(dev, "_reexec", fake_reexec)
    dev._ensure_uv()
    assert relaunched["path"].split(os.pathsep)[0] == str(local_bin)
    assert relaunched["marker"] == "1"


def test_ensure_uv_installs_once_then_fails_when_uv_never_appears(monkeypatch, capsys):
    """An installer that leaves no uv anywhere this process looks: one
    download, one relaunch, then the bootstrap error and exit 1. The relaunch
    is simulated by calling `_ensure_uv` again with the environment the real
    `_reexec` would have handed down."""
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setattr(dev.shutil, "which", lambda _name: None)
    downloads = []
    relaunches = []
    monkeypatch.setattr(
        dev.subprocess, "run",
        lambda *a, **_k: downloads.append(a) or subprocess.CompletedProcess(a, 0),
    )
    monkeypatch.setattr(dev, "_reexec", lambda *_a, **_k: relaunches.append(1))

    dev._ensure_uv()
    assert (len(downloads), len(relaunches)) == (1, 1)
    with pytest.raises(SystemExit) as exc:
        dev._ensure_uv()  # the relaunched process
    assert exc.value.code == 1
    assert (len(downloads), len(relaunches)) == (1, 1)
    err = capsys.readouterr().err
    assert "docs.astral.sh" in err
    assert "PATH" in err


@pytest.mark.skipif(sys.platform == "win32", reason="the bash launcher")
def test_bash_launcher_finds_uv_in_local_bin(tmp_path):
    """`cdui` itself checks `command -v uv`; with uv only in ~/.local/bin it
    must still hand off to uv rather than fall back to (or miss) a system
    Python. The stub uv records that it ran."""
    home = tmp_path / "home"
    stub = home / ".local" / "bin" / "uv"
    stub.parent.mkdir(parents=True)
    marker = tmp_path / "ran"
    stub.write_text(f"#!/bin/sh\necho \"$@\" > {marker}\nexit 7\n", encoding="utf-8")
    stub.chmod(0o755)
    proc = subprocess.run(
        ["/bin/bash", str(ROOT / "cdui"), "--version"],
        env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=30,
    )
    assert marker.exists(), proc.stderr
    assert marker.read_text().startswith("run --python 3.11")
