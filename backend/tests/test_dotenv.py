"""Project .env loader (spec 7.3): stdlib parse, setdefault semantics, and the
scoping guarantee that CODEFYUI_* CONFIG keys never reconfigure the already-
materialized settings singleton."""

import os

from app.core.dotenv import load_dotenv_file, parse_dotenv


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
