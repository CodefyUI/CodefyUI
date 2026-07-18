"""ASGI-lifespan integration test for app.main's project wiring (issue #88):
the project .env loads BEFORE discovery, the project provenance line logs,
and the stale-pin warning fires -- in that order.

httpx's ASGITransport never runs the lifespan (see conftest), so this drives
``async with lifespan(app)`` directly, with no extra dependency. Hermeticity:

- CODEFYUI_USER_DATA_DIR -> tmp: the session-token file AND the plugin
  lockfile both resolve under tmp at call time, so the stale-pin check sees
  an empty install (the manifest's ghost pin is genuinely stale) and zero
  plugin asset mounts happen.
- app.main.setup_logging -> no-op: the real one clears the root logger's
  handlers, which would detach caplog's capture handler mid-test.
- The project .env write goes RAW into os.environ (load_dotenv_file uses a
  direct assignment that monkeypatch's undo stack cannot see), so the
  fixture pops the key itself in teardown -- same rationale as
  test_dotenv.py's _isolate_environ.
"""

import logging
import os

import pytest

import app.main as main_mod
from app.config import settings
from app.core.auth import init_allowed_hosts

_SECRET_KEY = "CDUI_TEST_LIFESPAN_SECRET"


@pytest.fixture
def lifespan_project(tmp_path, monkeypatch):
    """A project dir (NOT a git repo -> 'not a repo' provenance line) with a
    .env secret and a manifest pinning a plugin that is not installed."""
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".env").write_text(f"{_SECRET_KEY}=from-dotenv\n",
                               encoding="utf-8")
    (proj / "codefyui.project.toml").write_text(
        '[project]\nname = "svc"\nformat_version = 1\n\n'
        '[plugins]\nghost-pack = { url = "https://github.com/x/y", '
        'ref = "v1", sha = "' + "0" * 40 + '" }\n', encoding="utf-8")
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path / "userdata"))
    monkeypatch.delenv(_SECRET_KEY, raising=False)
    monkeypatch.setattr(settings, "PROJECT_DIR", proj)
    monkeypatch.setattr(settings, "DB_PATH",
                        tmp_path / "db" / "lifespan-test.db")
    monkeypatch.setattr(main_mod, "setup_logging", lambda **kwargs: None)
    yield proj
    os.environ.pop(_SECRET_KEY, None)  # raw os.environ write; see docstring
    # The lifespan re-ran init_allowed_hosts with CORS origins added --
    # harmless, but restore the exact conftest-seeded whitelist anyway.
    init_allowed_hosts(settings.HOST, settings.PORT)


async def test_lifespan_wires_dotenv_provenance_and_pin_warning(
        lifespan_project, caplog):
    with caplog.at_level(logging.INFO, logger="app.main"):
        async with main_mod.lifespan(main_mod.app):
            # The .env secret reached os.environ while the server "runs"
            # (spec 7.3: loaded before node/plugin discovery, setdefault
            # semantics).
            assert os.environ.get(_SECRET_KEY) == "from-dotenv"

    messages = [r.getMessage() for r in caplog.records if r.name == "app.main"]
    # Key on stable tokens, not exact phrasing (#99/#100 reworded lines).
    dotenv_idx = next(
        i for i, m in enumerate(messages) if "project .env" in m)
    project_idx = next(
        i for i, m in enumerate(messages) if "not a repo" in m)
    pin_idx = next(
        i for i, m in enumerate(messages)
        if "ghost-pack" in m and "restore" in m)
    # Spec 7.3/7.4 wiring order: .env applies before discovery; project
    # provenance logs after discovery; the ONE stale-pin warning follows it.
    assert dotenv_idx < project_idx < pin_idx

    pin_record = next(
        r for r in caplog.records
        if r.name == "app.main" and "ghost-pack" in r.getMessage())
    assert pin_record.levelno == logging.WARNING
    # The loader logs the COUNT only -- a secret VALUE must never be logged.
    assert all("from-dotenv" not in m for m in messages)
