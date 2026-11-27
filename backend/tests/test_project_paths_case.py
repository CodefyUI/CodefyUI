"""Phase 4A review round 1: one folder, one spelling.

On a case-insensitive filesystem (macOS and Windows by default) a path in
another letter case names the same folder. It must not get a project past
``refused_location``, and it must not make the same project a second one
(another ``project_id``, another registry and trust key, another run root).
The tests that need such a filesystem are skipped on Linux.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.config import settings
from app.core import projects
from app.core.fs_paths import canonical_path, is_within
from app.core.project_switch import refused_location

from .test_api_projects import _project, http  # noqa: F401  (fixture)
from .test_projects_startup import isolated  # noqa: F401  (fixture)


def _swapcase_tail(path: Path, parts: int = 1) -> Path:
    """*path* with its last *parts* components in the other letter case."""
    head = Path(*path.parts[:-parts]) if parts < len(path.parts) else Path(path.anchor)
    return head.joinpath(*(p.swapcase() for p in path.parts[-parts:]))


def _symlink(link: Path, target: Path) -> None:
    """A directory symlink, or skip: Windows needs a privilege for one."""
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this platform does not allow creating symlinks here")


def _case_insensitive(directory: Path) -> bool:
    probe = directory / "CaseProbe"
    probe.mkdir(exist_ok=True)
    try:
        return (directory / "cASEpROBE").exists()
    finally:
        probe.rmdir()


@pytest.fixture
def insensitive(tmp_path):
    if not _case_insensitive(tmp_path):
        pytest.skip("the filesystem is case-sensitive")
    return tmp_path


def test_canonical_path_keeps_a_spelling_that_is_listed(tmp_path):
    (tmp_path / "Alpha" / "beta").mkdir(parents=True)
    assert canonical_path(tmp_path / "Alpha" / "beta") == (tmp_path / "Alpha" / "beta").resolve()


def test_canonical_path_keeps_a_missing_tail_as_given(tmp_path):
    (tmp_path / "Alpha").mkdir()
    got = canonical_path(tmp_path / "Alpha" / "New" / "Deeper")
    assert got == (tmp_path / "Alpha").resolve() / "New" / "Deeper"


def test_canonical_path_takes_the_disk_spelling(insensitive):
    real = insensitive / "Projects" / "MyLab"
    real.mkdir(parents=True)
    assert canonical_path(insensitive / "PROJECTS" / "mylab") == real.resolve()
    assert canonical_path(insensitive / "projects" / "MYLAB" / "not-yet") \
        == real.resolve() / "not-yet"


def test_is_within_sees_through_case_and_symlinks(insensitive):
    root = insensitive / "Root"
    (root / "inner").mkdir(parents=True)
    assert is_within(insensitive / "ROOT" / "INNER" / "new", root)
    link = insensitive / "link"
    _symlink(link, root)
    assert is_within(link / "inner", root)
    assert not is_within(insensitive / "Other", root)


def test_is_within_by_identity_when_spelling_cannot_tell(tmp_path):
    """A second name for the same folder (a hard-to-spell alias stands in
    for an 8.3 short name or another Unicode form) is caught by identity."""
    root = tmp_path / "root"
    root.mkdir()
    alias = tmp_path / "alias"
    _symlink(alias, root)
    # resolve() would follow the link; refuse to rely on it here.
    assert is_within(alias / "x", root)


def test_refused_location_in_another_case(insensitive, isolated, monkeypatch):  # noqa: F811
    user = Path(os.environ["CODEFYUI_USER_DATA_DIR"])
    user.mkdir(parents=True, exist_ok=True)
    assert refused_location(user / "p") is not None
    assert refused_location(_swapcase_tail(user) / "p") is not None
    from app.core import project_switch

    install = insensitive / "Install"
    (install / "backend").mkdir(parents=True)
    monkeypatch.setattr(project_switch, "_install_dir", lambda: canonical_path(install))
    assert refused_location(insensitive / "INSTALL" / "BACKEND" / "p") is not None
    assert refused_location(insensitive / "elsewhere" / "p") is None


async def test_create_in_the_user_data_dir_in_another_case_is_refused(
        http, isolated, insensitive):  # noqa: F811
    user = Path(os.environ["CODEFYUI_USER_DATA_DIR"])
    user.mkdir(parents=True, exist_ok=True)
    resp = await http.post("/api/projects/create",
                           json={"name": "x2", "parent": str(_swapcase_tail(user))})
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"]["code"] == "path_refused"
    assert not (user / "x2").exists()


async def test_create_in_the_install_dir_in_another_case_is_refused(
        http, isolated, insensitive, monkeypatch):  # noqa: F811
    from app.core import project_switch

    install = insensitive / "wt"
    (install / "backend").mkdir(parents=True)
    monkeypatch.setattr(project_switch, "_install_dir", lambda: canonical_path(install))
    resp = await http.post("/api/projects/create", json={
        "name": "x2", "parent": str(insensitive / "WT" / "BACKEND")})
    assert resp.status_code == 403, resp.text
    assert not (install / "backend" / "x2").exists()


async def test_one_project_in_two_spellings_is_one_project(http, isolated, insensitive):  # noqa: F811
    target = _project(isolated / "ext" / "Plain")
    first = await http.post("/api/projects/open", json={"path": str(target)})
    assert first.status_code == 200, first.text
    other = _swapcase_tail(target, 2)
    assert str(other) != str(target)
    again = await http.post("/api/projects/open", json={"path": str(other)})
    assert again.status_code == 200, again.text
    assert again.json()["unchanged"] is True
    assert again.json()["project_id"] == first.json()["project_id"]
    assert projects.project_id(other) == projects.project_id(target)
    assert projects._key(other) == projects._key(target)
    assert settings.PROJECT_DIR == canonical_path(target)
    recent = (await http.get("/api/projects")).json()["recent"]
    assert [r["path"] for r in recent] == [str(canonical_path(target))]


async def test_a_run_made_under_one_spelling_belongs_to_the_other(
        isolated, insensitive, tmp_path):  # noqa: F811
    from app.core.db import Database
    from app.core.run_store import RunProvenance, RunStore, run_project_refusal

    target = _project(isolated / "ext" / "Lab")
    projects.activate(_swapcase_tail(target))
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    try:
        store = RunStore(database)
        record = await store.create_run(graph_snapshot={"nodes": [], "edges": []},
                                        provenance=RunProvenance())
        assert record.project_dir == str(canonical_path(target))
        projects.activate(target)
        assert run_project_refusal(await store.get_run(record.id)) is None
    finally:
        database.close()
