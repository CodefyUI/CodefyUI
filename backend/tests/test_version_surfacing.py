"""The running version must be discoverable, and the three copies must agree.

Before this, nothing reported a version: the FastAPI app declared none,
`/api/health` returned none, there was no `cdui --version`, and no
`__version__` existed anywhere. A bug report from a classroom could not state
which version it came from.

The version also lives in three files that are bumped by hand -- there is no
bump command -- so `test_all_version_fields_agree` is the only thing standing
between a release and a frontend that claims one version while the backend
claims another.
"""

from __future__ import annotations

import json
import re

import pytest

import dev  # scripts/dev.py -- conftest puts scripts/ on sys.path
from app.core.version import UNKNOWN_VERSION, get_version
from app.main import app

ROOT = dev.ROOT
SEMVER = re.compile(r"^\d+\.\d+\.\d+")


def _pyproject_version() -> str:
    text = (ROOT / "backend" / "pyproject.toml").read_text(encoding="utf-8")
    return re.search(r'^version\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE).group(1)


def _package_json_version() -> str:
    data = json.loads((ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
    return data["version"]


def _uv_lock_version() -> str | None:
    """The codefyui-backend entry in backend/uv.lock, if present."""
    text = (ROOT / "backend" / "uv.lock").read_text(encoding="utf-8")
    m = re.search(
        r'\[\[package\]\]\s*\nname\s*=\s*"codefyui-backend"\s*\nversion\s*=\s*"([^"]+)"',
        text,
    )
    return m.group(1) if m else None


# ── The three copies must agree ──────────────────────────────────────────────

def test_all_version_fields_agree():
    """Nothing else asserts this, and every bump edits all three by hand."""
    fields = {
        "backend/pyproject.toml": _pyproject_version(),
        "frontend/package.json": _package_json_version(),
    }
    lock = _uv_lock_version()
    if lock is not None:
        fields["backend/uv.lock"] = lock
    assert len(set(fields.values())) == 1, f"version fields disagree: {fields}"


def test_the_version_looks_like_a_version():
    assert SEMVER.match(_pyproject_version())


# ── The library helper ───────────────────────────────────────────────────────

def _installed_version() -> str | None:
    """What `importlib.metadata` reports, or None if not pip-installed."""
    from importlib.metadata import PackageNotFoundError, version as pkg_version

    try:
        return pkg_version("codefyui-backend")
    except PackageNotFoundError:
        return None


def test_get_version_reports_what_is_installed():
    """Its contract is the INSTALLED version, not the checkout's.

    Comparing it to pyproject.toml would be testing the wrong thing: the two
    legitimately differ right after a version bump, until the editable install
    is refreshed -- and that gap is precisely what the helper exists to
    report honestly. A CI job installs from the bumped pyproject, so there
    they agree; a maintainer's stale local venv is not a failure.
    """
    installed = _installed_version()
    assert get_version() == (installed if installed is not None else _pyproject_version())


def test_get_version_falls_back_to_pyproject_without_installed_metadata(monkeypatch):
    """A source checkout that was never pip-installed still reports a number."""
    from importlib.metadata import PackageNotFoundError

    import app.core.version as version_mod

    def boom(_name):
        raise PackageNotFoundError(_name)

    monkeypatch.setattr(version_mod, "_package_version", boom)
    version_mod.get_version.cache_clear()
    try:
        assert version_mod.get_version() == _pyproject_version()
    finally:
        version_mod.get_version.cache_clear()


def test_unreadable_everything_reports_an_obviously_wrong_version(monkeypatch):
    """A plausible-looking wrong number in a bug report is worse than 'unknown'."""
    from importlib.metadata import PackageNotFoundError

    import app.core.version as version_mod

    def boom(_name):
        raise PackageNotFoundError(_name)

    monkeypatch.setattr(version_mod, "_package_version", boom)
    monkeypatch.setattr(version_mod, "_version_from_pyproject", lambda: None)
    version_mod.get_version.cache_clear()
    try:
        assert version_mod.get_version() == UNKNOWN_VERSION
    finally:
        version_mod.get_version.cache_clear()


# ── The three surfaces ───────────────────────────────────────────────────────

async def test_health_reports_the_version(test_client):
    body = (await test_client.get("/api/health")).json()
    assert body["version"] == get_version()


async def test_health_reports_the_version_outside_project_mode(test_client, monkeypatch):
    """Unlike `project`, this key is unconditional -- see main.py's comment."""
    from app.api import routes_graph

    monkeypatch.setattr(routes_graph.settings, "PROJECT_DIR", None)
    body = (await test_client.get("/api/health")).json()
    assert "version" in body


def test_the_fastapi_app_declares_it():
    assert app.version == get_version()


def test_openapi_carries_it():
    """This is what an external caller of a published graph reads."""
    assert app.openapi()["info"]["version"] == get_version()


# ── `cdui --version` ─────────────────────────────────────────────────────────

def test_cdui_version_helper_agrees():
    assert dev._codefyui_version() == _pyproject_version()


def test_cdui_version_does_not_need_the_venv(monkeypatch):
    """It must answer on a half-finished install; that is when it is asked."""
    called = []
    monkeypatch.setattr(dev, "_ensure_uv", lambda: called.append("uv"))
    dev._codefyui_version()
    assert called == []


@pytest.mark.parametrize("flag", ["--version", "-V", "version"])
def test_all_three_spellings_are_accepted(flag):
    """Pinned against the dispatch block in dev.py's __main__."""
    source = (ROOT / "scripts" / "dev.py").read_text(encoding="utf-8")
    assert f'"{flag}"' in source
