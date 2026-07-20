"""`cdui update` must not re-run the interactive installer.

Regression suite for the bare-`cdui update` prompt: update() resolved its
options through the same helper as `cdui install`, so from a terminal with no
flags it dropped into the "CodefyUI installer" menu — and whatever you
answered, install() then force-reinstalled a multi-GB torch wheel (`auto`
could even switch you off a deliberately chosen variant). Update now reuses
whatever the venv already has, and the forced reinstall is reserved for an
actual variant switch.
"""

from __future__ import annotations

import sys

import pytest

import dev  # scripts/dev.py — conftest puts scripts/ on sys.path


def _fake_venv(tmp_path, monkeypatch, *, torch_version=None, pytest_installed=False):
    """A venv skeleton with an optional torch/version.py + pytest dist-info.

    "Lib" (capital L) is the Windows layout _get_installed_torch_version()
    probes first; the literal path also resolves on Linux CI.
    """
    site = tmp_path / "venv" / "Lib" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    if torch_version is not None:
        tdir = site / "torch"
        tdir.mkdir(exist_ok=True)
        (tdir / "version.py").write_text(
            f"__version__ = {torch_version!r}\ndebug = False\n", encoding="utf-8"
        )
    if pytest_installed:
        (site / "pytest-8.3.4.dist-info").mkdir(exist_ok=True)
    monkeypatch.setattr(dev, "VENV", tmp_path / "venv")
    return site


# ── installed torch variant ────────────────────────────────────────────


@pytest.mark.parametrize("version,expected", [
    ("2.11.0+cu128", "cu128"),
    ("2.4.0+cu118", "cu118"),
    ("2.6.0+cpu", "cpu"),
    ("2.5.1+rocm6.2", "rocm6.2"),
])
def test_installed_torch_variant_reads_local_build_tag(
        tmp_path, monkeypatch, version, expected):
    _fake_venv(tmp_path, monkeypatch, torch_version=version)
    assert dev._installed_torch_variant() == expected


def test_installed_torch_variant_none_when_torch_absent(tmp_path, monkeypatch):
    _fake_venv(tmp_path, monkeypatch, torch_version=None)
    assert dev._installed_torch_variant() is None


def test_installed_torch_variant_untagged_wheel_is_skip(tmp_path, monkeypatch):
    """Plain PyPI wheel (Apple Silicon, or a hand-rolled install) — we can't
    name the index it came from, so the only safe action is to leave it be."""
    _fake_venv(tmp_path, monkeypatch, torch_version="2.11.0")
    assert dev._installed_torch_variant() == "skip"


def test_installed_torch_variant_unknown_tag_is_skip(tmp_path, monkeypatch):
    _fake_venv(tmp_path, monkeypatch, torch_version="2.11.0+cu999")
    assert dev._installed_torch_variant() == "skip"


def test_installed_torch_variant_handles_suffixed_tag(tmp_path, monkeypatch):
    _fake_venv(tmp_path, monkeypatch, torch_version="2.6.0+cpu.cxx11.abi")
    assert dev._installed_torch_variant() == "cpu"


# ── installed dev extra ────────────────────────────────────────────────


def test_venv_has_dev_extra_when_pytest_present(tmp_path, monkeypatch):
    _fake_venv(tmp_path, monkeypatch, pytest_installed=True)
    assert dev._venv_has_dev_extra() is True


def test_venv_has_dev_extra_false_without_pytest(tmp_path, monkeypatch):
    _fake_venv(tmp_path, monkeypatch, pytest_installed=False)
    assert dev._venv_has_dev_extra() is False


# ── update option resolution ───────────────────────────────────────────


@pytest.fixture
def tty_no_prompt(monkeypatch):
    """A real terminal, a clean env — and any installer menu fails the test."""
    def _boom(*a, **kw):
        raise AssertionError("cdui update must never open the installer menu")

    class _Tty:
        def isatty(self):
            return True

    monkeypatch.setattr(dev, "_prompt_install_options", _boom)
    monkeypatch.setattr(
        dev, "detect_gpu", lambda: ("NVIDIA GeForce RTX 4080 (driver 610.74)", "cu128"))
    monkeypatch.setattr(sys, "stdin", _Tty())
    for var in ("CODEFYUI_GPU", "CODEFYUI_DEV"):
        monkeypatch.delenv(var, raising=False)


def test_update_reuses_installed_variant_without_prompting(
        tmp_path, monkeypatch, tty_no_prompt):
    _fake_venv(tmp_path, monkeypatch, torch_version="2.11.0+cu128")
    assert dev._resolve_update_options([]) == ("cu128", False)


def test_update_preserves_dev_extra(tmp_path, monkeypatch, tty_no_prompt):
    _fake_venv(tmp_path, monkeypatch,
               torch_version="2.6.0+cpu", pytest_installed=True)
    assert dev._resolve_update_options([]) == ("cpu", True)


def test_update_leaves_untagged_torch_alone(tmp_path, monkeypatch, tty_no_prompt):
    """The regression that matters most: a user on a custom wheel keeps it,
    instead of being silently switched to the auto-detected cu128."""
    _fake_venv(tmp_path, monkeypatch, torch_version="2.11.0")
    assert dev._resolve_update_options([])[0] == "skip"


def test_update_explicit_gpu_flag_wins(tmp_path, monkeypatch, tty_no_prompt):
    _fake_venv(tmp_path, monkeypatch, torch_version="2.11.0+cu128")
    assert dev._resolve_update_options(["--gpu", "cpu"])[0] == "cpu"


def test_update_explicit_no_dev_flag_wins(tmp_path, monkeypatch, tty_no_prompt):
    _fake_venv(tmp_path, monkeypatch,
               torch_version="2.11.0+cu128", pytest_installed=True)
    assert dev._resolve_update_options(["--no-dev"])[1] is False


def test_update_env_vars_win(tmp_path, monkeypatch, tty_no_prompt):
    _fake_venv(tmp_path, monkeypatch, torch_version="2.11.0+cu128")
    monkeypatch.setenv("CODEFYUI_GPU", "cpu")
    monkeypatch.setenv("CODEFYUI_DEV", "1")
    assert dev._resolve_update_options([]) == ("cpu", True)


def test_update_falls_back_to_detection_when_torch_missing(
        tmp_path, monkeypatch, tty_no_prompt):
    """Half-built venv — update still has to be able to repair it."""
    _fake_venv(tmp_path, monkeypatch, torch_version=None)
    assert dev._resolve_update_options([])[0] == "cu128"


def test_update_gpu_auto_resolves_through_detection(
        tmp_path, monkeypatch, tty_no_prompt):
    _fake_venv(tmp_path, monkeypatch, torch_version="2.6.0+cpu")
    assert dev._resolve_update_options(["--gpu", "auto"])[0] == "cu128"


# ── install(): forced reinstall only on a real variant switch ──────────


def _capture_install(tmp_path, monkeypatch):
    """Run install() with subprocesses captured and the frontend short-circuited."""
    calls: list[list[str]] = []
    monkeypatch.setattr(dev, "run", lambda cmd, **kw: calls.append(list(cmd)))
    monkeypatch.setattr(dev, "_print_post_install_summary", lambda gpu, dev: None)
    dist_index = tmp_path / "dist" / "index.html"
    dist_index.parent.mkdir(parents=True, exist_ok=True)
    dist_index.write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(dev, "DIST_INDEX", dist_index)
    monkeypatch.delenv("CODEFYUI_FORCE_BUILD", raising=False)
    return calls


def test_install_forces_reinstall_when_switching_variants(tmp_path, monkeypatch):
    _fake_venv(tmp_path, monkeypatch, torch_version="2.11.0+cu128")
    calls = _capture_install(tmp_path, monkeypatch)
    dev.install(gpu="cpu", dev=False)
    torch_cmd = next(c for c in calls if "torch" in c)
    assert "--reinstall-package" in torch_cmd
    assert "https://download.pytorch.org/whl/cpu" in torch_cmd


def test_install_skips_forced_reinstall_when_variant_matches(tmp_path, monkeypatch):
    """Same variant: still resolved against the cu128 index, so a raised torch
    floor upgrades from the right place — but no multi-GB re-download."""
    _fake_venv(tmp_path, monkeypatch, torch_version="2.11.0+cu128")
    calls = _capture_install(tmp_path, monkeypatch)
    dev.install(gpu="cu128", dev=False)
    torch_cmd = next(c for c in calls if "torch" in c)
    assert "--reinstall-package" not in torch_cmd
    assert "https://download.pytorch.org/whl/cu128" in torch_cmd


def test_install_forces_reinstall_when_torch_absent(tmp_path, monkeypatch):
    _fake_venv(tmp_path, monkeypatch, torch_version=None)
    calls = _capture_install(tmp_path, monkeypatch)
    dev.install(gpu="cu128", dev=False)
    torch_cmd = next(c for c in calls if "torch" in c)
    assert "--reinstall-package" in torch_cmd


# ── help text ──────────────────────────────────────────────────────────


def test_update_help_names_the_update_command(capsys):
    with pytest.raises(SystemExit):
        dev._parse_install_args(["--help"], prog="cdui update")
    assert "cdui update" in capsys.readouterr().out


def test_install_help_still_names_install(capsys):
    with pytest.raises(SystemExit):
        dev._parse_install_args(["--help"])
    assert "cdui install" in capsys.readouterr().out


def test_update_help_does_not_touch_the_checkout(tmp_path, monkeypatch, capsys):
    """`--help` has to exit before the git realign — update() hard-resets the
    working tree to FETCH_HEAD, which is not something a help flag may do."""
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(dev, "ROOT", tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(dev, "run", lambda cmd, **kw: calls.append(list(cmd)))
    monkeypatch.setattr(dev, "install", lambda **kw: calls.append(["install"]))
    monkeypatch.setattr(sys, "argv", ["cdui", "update", "--help"])

    with pytest.raises(SystemExit):
        dev.update()

    assert calls == [], f"--help ran commands before exiting: {calls}"
    assert "cdui update" in capsys.readouterr().out
