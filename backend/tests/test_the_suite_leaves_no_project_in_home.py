"""Phase 4A: nothing in the test suite may create a project in the user's
home directory. The session has its own PROJECTS_ROOT and the default
project is off (conftest); a lifespan run here must respect both."""

import os
from pathlib import Path

from app.config import settings
from app.main import app, lifespan


def test_the_session_points_projects_at_a_temporary_root():
    assert os.environ["CODEFYUI_DEFAULT_PROJECT"] == "0"
    root = Path(os.environ["CODEFYUI_PROJECTS_ROOT"])
    assert root != Path.home() / "CodefyUI-Projects"
    assert settings.PROJECTS_ROOT == root
    assert settings.DEFAULT_PROJECT is False


async def test_a_lifespan_creates_no_project(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "db" / "t.db")
    root = Path(os.environ["CODEFYUI_PROJECTS_ROOT"])
    before = sorted(root.iterdir()) if root.exists() else []
    home = Path.home() / "CodefyUI-Projects"
    existed = home.exists()
    async with lifespan(app):
        pass
    assert (sorted(root.iterdir()) if root.exists() else []) == before
    assert home.exists() == existed
