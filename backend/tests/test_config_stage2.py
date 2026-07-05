"""Stage-2 settings: defaults + CODEFYUI_ env overrides (spec Section 8)."""

from __future__ import annotations

from pathlib import Path

import app.config
from app.config import Settings, settings


def test_stage2_defaults():
    backend_dir = Path(app.config.__file__).resolve().parent.parent
    assert settings.DB_PATH.resolve() == (
        backend_dir / "data" / "codefyui.db"
    ).resolve()
    assert settings.MAX_IMAGE_PIXELS == 25_000_000
    assert settings.RUN_IO_CAP_BYTES == 64 * 1024
    assert settings.RUNS_RETENTION_DAYS == 0        # 0 = keep forever
    assert settings.EXTRA_ALLOWED_HOSTS == ""


def test_db_path_is_sibling_of_graphs_dir():
    # data/codefyui.db lives next to data/graphs (spec Section 5); the
    # repo-relative default isolates dev clones from a global install.
    assert settings.DB_PATH.name == "codefyui.db"
    assert settings.DB_PATH.parent.name == "data"


def test_stage2_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEFYUI_DB_PATH", str(tmp_path / "custom.db"))
    monkeypatch.setenv("CODEFYUI_MAX_IMAGE_PIXELS", "123456")
    monkeypatch.setenv("CODEFYUI_RUNS_RETENTION_DAYS", "14")
    monkeypatch.setenv("CODEFYUI_RUN_IO_CAP_BYTES", "1024")
    monkeypatch.setenv(
        "CODEFYUI_EXTRA_ALLOWED_HOSTS", "192.168.1.20:8000,mybox:8000",
    )
    fresh = Settings()
    assert fresh.DB_PATH == tmp_path / "custom.db"
    assert fresh.MAX_IMAGE_PIXELS == 123456
    assert fresh.RUNS_RETENTION_DAYS == 14
    assert fresh.RUN_IO_CAP_BYTES == 1024
    assert fresh.EXTRA_ALLOWED_HOSTS == "192.168.1.20:8000,mybox:8000"
