"""Project .env loader (spec 7.3): stdlib parse, setdefault semantics, and the
scoping guarantee that CODEFYUI_* CONFIG keys never reconfigure the already-
materialized settings singleton.

load_dotenv_file writes via a raw os.environ[...] = val (setdefault
semantics), not monkeypatch.setenv, so monkeypatch's own undo stack cannot
see or revert those writes. The autouse _isolate_environ fixture below
snapshots/restores os.environ around every test in this module so nothing a
test causes to be written (e.g. CODEFYUI_OPENAI_API_KEY, CODEFYUI_PORT) leaks
into later tests in this process.
"""

import os

import pytest

from app.core.dotenv import load_dotenv_file, parse_dotenv


@pytest.fixture(autouse=True)
def _isolate_environ():
    """Snapshot/restore os.environ around each test in this module. Needed
    because load_dotenv_file's raw os.environ writes are invisible to
    monkeypatch's undo stack (see module docstring)."""
    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


def test_parse_basic_and_comments():
    text = "# comment\n\nOPENAI_API_KEY=sk-123\nexport FOO=bar\nQUOTED=\"a b\"\n"
    parsed = parse_dotenv(text)
    assert parsed["OPENAI_API_KEY"] == "sk-123"
    assert parsed["FOO"] == "bar"       # `export ` prefix tolerated
    assert parsed["QUOTED"] == "a b"    # surrounding quotes stripped
    assert "comment" not in parsed


def test_absent_file_is_fine(tmp_path):
    assert load_dotenv_file(tmp_path / "nope.env") == 0


def test_setdefault_existing_env_wins(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("MY_SECRET=new\n")
    monkeypatch.setenv("MY_SECRET", "keep")
    load_dotenv_file(env)
    assert os.environ["MY_SECRET"] == "keep"  # already set -> not overwritten


def test_secret_reaches_environ(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("CODEFYUI_OPENAI_API_KEY=sk-xyz\n")
    monkeypatch.delenv("CODEFYUI_OPENAI_API_KEY", raising=False)
    load_dotenv_file(env)
    # Execution-time secret (read by llm_chat at execute time) is available.
    assert os.environ.get("CODEFYUI_OPENAI_API_KEY") == "sk-xyz"


def test_config_keys_do_not_reconfigure_running_settings(tmp_path, monkeypatch):
    from app.config import settings
    original_port = settings.PORT
    env = tmp_path / ".env"
    env.write_text("CODEFYUI_PORT=59999\n")
    monkeypatch.delenv("CODEFYUI_PORT", raising=False)  # teardown restores absent
    load_dotenv_file(env)
    # A CODEFYUI_* CONFIG key in .env never changes the running server: the
    # settings singleton materialized at import, before .env loads (spec 7.3).
    assert settings.PORT == original_port


def test_bom_prefixed_env_key_loads_clean(tmp_path, monkeypatch):
    """A UTF-8 BOM (b"\\xef\\xbb\\xbf") at the start of the file must not
    corrupt the first key: utf-8-sig strips it, plain utf-8 would not."""
    env = tmp_path / ".env"
    env.write_bytes(b"\xef\xbb\xbfBOM_KEY=clean\n")
    monkeypatch.delenv("BOM_KEY", raising=False)
    load_dotenv_file(env)
    assert os.environ.get("BOM_KEY") == "clean"
    assert "\ufeffBOM_KEY" not in os.environ


def test_no_env_leak_after_module():
    # Regression guard for _isolate_environ above: by the time this test runs
    # (last, in file order), CODEFYUI_OPENAI_API_KEY and CODEFYUI_PORT --
    # written directly into os.environ by earlier tests in this file via
    # load_dotenv_file -- must already be restored to absent.
    assert "CODEFYUI_OPENAI_API_KEY" not in os.environ
    assert "CODEFYUI_PORT" not in os.environ
