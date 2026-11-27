"""Phase 4A: a project's .env can be loaded and taken back out at run time,
and a run-time load never applies the blocked names."""

import os

import pytest

from app.core import dotenv
from app.core.dotenv import (
    apply_project_env,
    is_blocked,
    is_endpoint,
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
    # Review round 1: what a consented .env could still reach.
    "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "UV_INDEX_URL", "CONDA_PKGS_DIRS",
    "BASH_ENV", "ENV", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "NODE_OPTIONS",
    "NODE_PATH", "PERL5OPT", "RUBYOPT", "JAVA_TOOL_OPTIONS", "_JAVA_OPTIONS",
    "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "ALL_PROXY", "all_proxy",
    "NO_PROXY", "no_proxy", "SSL_CERT_FILE", "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "TMPDIR", "TEMP", "TMP",
    "EDITOR", "VISUAL", "PAGER", "BROWSER", "IFS",
    # Any letter case, on every platform.
    "Path", "pythonpath", "ld_preload", "git_ssh_command", "Pip_Index_Url",
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


@pytest.mark.parametrize("name", [
    "OPENAI_API_KEY", "MY_SECRET", "WANDB_API_KEY", "PATHS_EXTRA",
    "ENVIRONMENT", "MY_ENV", "TEMPERATURE", "PIPELINE_NAME", "UVICORN_WORKERS",
])
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


def test_xdg_cache_home_is_blocked_not_restart_only():
    assert is_blocked("XDG_CACHE_HOME") and not needs_restart("XDG_CACHE_HOME")


@pytest.mark.parametrize("name, endpoint", [
    ("OPENAI_BASE_URL", True), ("ANTHROPIC_API_BASE", True), ("OLLAMA_HOST", True),
    ("AZURE_OPENAI_ENDPOINT", True), ("MY_SERVICE_BASE", True), ("openai_base_url", True),
    ("OPENAI_API_KEY", False), ("HOSTNAME_SUFFIX", False), ("MY_SECRET", False),
])
def test_names_that_choose_a_server_are_pointed_out(name, endpoint):
    assert is_endpoint(name) is endpoint
