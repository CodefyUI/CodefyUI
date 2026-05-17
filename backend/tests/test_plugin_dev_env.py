"""Tests for the CODEFYUI_USER_DATA_DIR dev-mode override.

The override lets a clone of CodefyUI keep its own plugin lockfile inside
the repo (``.codefyui_dev/``) instead of sharing the global user-data dir
with every other clone on the machine. The override is read by:

    - ``plugin_loader.plugins_user_root`` (lockfile + downloaded packs)
    - ``auth._token_dir`` (session token file)

Both should fall back to ``platformdirs.user_data_dir`` when the env var
is unset, so non-dev installs keep working untouched.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.core import auth, plugin_loader


@pytest.fixture
def clean_env(monkeypatch):
    """Make sure the env var doesn't leak between tests."""
    monkeypatch.delenv("CODEFYUI_USER_DATA_DIR", raising=False)


def test_default_uses_platformdirs(clean_env):
    """Without the env var, the path lives under the OS user-data dir."""
    root = plugin_loader.plugins_user_root()
    # Should resolve under the platformdirs-managed location, not the cwd.
    assert root.name == "plugins"
    assert "codefyui" in str(root).lower()


def test_override_redirects_plugin_root(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    assert plugin_loader.plugins_user_root() == tmp_path / "plugins"
    assert plugin_loader.lockfile_path() == tmp_path / "plugins" / "installed.json"


def test_override_redirects_session_token(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    assert auth.token_file_path() == tmp_path / "session.token"


def test_override_is_per_process(clean_env, tmp_path, monkeypatch):
    """Setting the env var in one test mustn't leak — the fixture proves it."""
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    redirected = plugin_loader.plugins_user_root()
    monkeypatch.delenv("CODEFYUI_USER_DATA_DIR")
    default = plugin_loader.plugins_user_root()
    assert redirected != default
