"""Phase 4A: a project's .env can be loaded and taken back out at run time,
and a run-time load never applies the blocked names."""

import os

import pytest

from app.core import dotenv
from app.core.dotenv import (
    apply_project_env,
    is_blocked,
    needs_restart,
    read_dotenv,
    unload_project_env,
)


@pytest.fixture(autouse=True)
def _isolate_environ():
    snapshot = dict(os.environ)
    dotenv._APPLIED.clear()
    yield
    dotenv._APPLIED.clear()
    os.environ.clear()
    os.environ.update(snapshot)


def test_apply_then_unload_takes_back_exactly_what_was_applied(monkeypatch):
    monkeypatch.setenv("ALREADY_SET", "outer")
    report = apply_project_env({"P4_KEY": "a", "ALREADY_SET": "inner"}, enforce_blocklist=True)
    assert report["applied"] == ["P4_KEY"]
    assert report["kept"] == ["ALREADY_SET"]
    assert os.environ["P4_KEY"] == "a"
    assert os.environ["ALREADY_SET"] == "outer"
    assert unload_project_env() == ["P4_KEY"]
    assert "P4_KEY" not in os.environ
    assert os.environ["ALREADY_SET"] == "outer"


def test_a_value_changed_since_is_not_removed():
    apply_project_env({"P4_KEY": "a"}, enforce_blocklist=True)
    os.environ["P4_KEY"] = "changed by someone else"
    assert unload_project_env() == []
    assert os.environ["P4_KEY"] == "changed by someone else"
    assert dotenv.applied_names() == []


def test_switching_projects_replaces_the_first_projects_values():
    apply_project_env({"OPENAI_API_KEY": "project-a"}, enforce_blocklist=True)
    unload_project_env()
    apply_project_env({"OPENAI_API_KEY": "project-b"}, enforce_blocklist=True)
    assert os.environ["OPENAI_API_KEY"] == "project-b"


@pytest.mark.parametrize("name", [
    "PATH", "PYTHONPATH", "PYTHONSTARTUP", "LD_PRELOAD", "DYLD_INSERT_LIBRARIES",
    "GIT_SSH_COMMAND", "SSH_AUTH_SOCK", "CODEFYUI_ALLOW_REMOTE_SKILL_INSTALL",
    "HOME", "USERPROFILE", "SHELL", "COMSPEC",
])
def test_blocked_names_are_never_applied_at_run_time(name):
    os.environ.pop(name, None)
    report = apply_project_env({name: "x"}, enforce_blocklist=True)
    assert report["blocked"] == [name]
    assert name not in os.environ
    assert is_blocked(name)


def test_the_start_up_project_keeps_todays_behaviour():
    os.environ.pop("CODEFYUI_SOMETHING", None)
    report = apply_project_env({"CODEFYUI_SOMETHING": "1"}, enforce_blocklist=False)
    assert report["applied"] == ["CODEFYUI_SOMETHING"]


@pytest.mark.parametrize("name", ["OPENAI_API_KEY", "MY_SECRET", "WANDB_API_KEY", "PATHS_EXTRA"])
def test_ordinary_names_are_not_blocked(name):
    assert not is_blocked(name)


def test_restart_only_names_are_reported():
    for name in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "TORCH_HOME", "OMP_NUM_THREADS"):
        os.environ.pop(name, None)
    report = apply_project_env({"HF_HOME": "/x", "CUDA_VISIBLE_DEVICES": "0", "API_KEY": "k"},
                               enforce_blocklist=True)
    assert report["restart"] == ["CUDA_VISIBLE_DEVICES", "HF_HOME"]
    assert needs_restart("TORCH_HOME") and not needs_restart("API_KEY")


def test_read_dotenv_absent_is_none(tmp_path):
    assert read_dotenv(tmp_path / ".env") is None
    (tmp_path / ".env").write_text("A=1\n", encoding="utf-8")
    assert read_dotenv(tmp_path / ".env") == {"A": "1"}
