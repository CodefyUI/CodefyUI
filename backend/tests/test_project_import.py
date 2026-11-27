"""Phase 4A, R9: importing the flat-mode graphs and the files they use."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core import projects
from app.core.auth import TOKEN_HEADER, session_token
from app.core.project import scaffold_project
from app.core.project_import import plan_import
from app.main import app

from .test_projects_startup import isolated  # noqa: F401  (fixture)

BASE_URL = f"http://127.0.0.1:{settings.PORT}"


def _graph(name: str, params: dict) -> dict:
    return {"name": name, "nodes": [
        {"id": "n1", "type": "ModelLoader", "position": {"x": 0, "y": 0},
         "data": {"params": params}}], "edges": []}


@pytest.fixture
def flat(isolated):  # noqa: F811
    """A flat data folder: two graphs, referenced and unreferenced files."""
    root = isolated  # flat_graphs_dir() is <tmp>/flat-graphs; its parent is root
    graphs = root / "flat-graphs"
    graphs.mkdir()
    (root / "models").mkdir()
    (root / "models" / "weights.pt").write_bytes(b"w" * 1000)
    (root / "models" / "unused.pt").write_bytes(b"u" * 50)
    (root / "images" / "cats").mkdir(parents=True)
    (root / "images" / "cats" / "a.png").write_bytes(b"a" * 30)
    (root / "images" / "cats" / "b.png").write_bytes(b"b" * 20)
    (root / "media").mkdir()
    (root / "media" / "clip.wav").write_bytes(b"c" * 7)
    (root / "secret.txt").write_text("outside the asset folders",
                                     encoding="utf-8")
    (graphs / "train.json").write_text(json.dumps(_graph("train", {
        "path": "weights.pt",            # relative to models/
        "folder": "images/cats",         # relative to the data folder
        "missing": "nope.pt",            # does not exist: not a reference
        "kind": "models",                # the folder itself: not a reference
        "escape": "../secret.txt",       # leaves models/: not a reference
    })), encoding="utf-8")
    (graphs / "listen.json").write_text(json.dumps(_graph("listen", {
        "audio": str(root / "media" / "clip.wav"),  # absolute
        "again": "weights.pt",                      # counted once
        "nested": {"deep": ["images/cats/a.png"]},  # inside a listed folder
    })), encoding="utf-8")
    return root


def test_the_plan_lists_each_referenced_file_and_the_total(flat):
    plan = plan_import().as_dict()
    assert [g["name"] for g in plan["graphs"]] == ["listen", "train"]
    assert [(a["path"], a["kind"], a["size"], a["files"]) for a in plan["assets"]] == [
        ("images/cats", "images", 50, 2),
        ("media/clip.wav", "media", 7, 1),
        ("models/weights.pt", "models", 1000, 1),
    ]
    assert plan["assets_bytes"] == 1057
    assert plan["graphs_bytes"] == sum(g["size"] for g in plan["graphs"])


@pytest.fixture
async def http(flat):
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL,
                           headers={TOKEN_HEADER: session_token()}) as client:
        yield client


def _listing(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


async def test_import_into_a_new_project_copies_and_opens_it(http, flat):
    before = _listing(flat)
    assert (await http.get("/api/projects/import")).json()["assets_bytes"] == 1057

    resp = await http.post("/api/projects/import", json={"name": "imported"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    target = (flat / "projects" / "imported").resolve()
    assert body["created"] == str(target)
    assert sorted(body["imported"]) == ["listen", "train"]
    assert sorted(body["copied"]) == ["images/cats/a.png", "images/cats/b.png",
                                      "media/clip.wav", "models/weights.pt"]
    assert (target / "assets" / "models" / "weights.pt").read_bytes() == b"w" * 1000
    assert not (target / "assets" / "models" / "unused.pt").exists()
    assert (target / "graphs" / "train.graph.json").is_file()
    assert settings.PROJECT_DIR == target
    assert projects.env_trust_state(target) in ("none", "trusted")
    # Nothing in the flat folder moved or changed.
    after = {k: v for k, v in _listing(flat).items()
             if not k.startswith(("projects/", "user/"))}
    assert after == {k: v for k, v in before.items()
                     if not k.startswith(("projects/", "user/"))}


async def test_assets_can_be_left_out(http, flat):
    resp = await http.post("/api/projects/import",
                           json={"name": "graphs-only", "include_assets": False})
    assert resp.status_code == 200, resp.text
    assert resp.json()["copied"] == []
    assert not (flat / "projects" / "graphs-only" / "assets" / "models"
                / "weights.pt").exists()


async def test_import_into_an_existing_project_keeps_what_it_has(http, flat):
    existing = flat / "projects" / "mine"
    scaffold_project(existing)
    (existing / "graphs" / "train.graph.json").write_text('{"mine": true}\n',
                                                         encoding="utf-8")
    (existing / "assets" / "models" / "weights.pt").write_bytes(b"theirs")
    resp = await http.post("/api/projects/import", json={"path": str(existing)})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["imported"] == ["listen"]
    assert body["skipped"][0]["file"] == "train.json"
    assert "models/weights.pt" in body["kept"]
    assert (existing / "graphs" / "train.graph.json").read_text(
        encoding="utf-8") == '{"mine": true}\n'
    assert (existing / "assets" / "models" / "weights.pt").read_bytes() == b"theirs"
    assert settings.PROJECT_DIR == existing.resolve()


async def test_import_needs_a_target_in_flat_mode(http):
    resp = await http.post("/api/projects/import", json={})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "no_target"
