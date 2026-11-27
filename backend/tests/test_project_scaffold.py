"""Phase 4A: the project scaffold and adoption live in app.core.project, so
the server can create projects and import flat graphs the way `cdui project
init [--adopt]` does."""

import json

import pytest

from app.core.project import (
    PROJECT_DIRS,
    PROJECT_MANIFEST,
    GraphAmbiguityError,
    adopt_graphs,
    read_project_manifest,
    scaffold_project,
)


def test_scaffold_writes_the_starter_files_once(tmp_path):
    target = tmp_path / "proj"
    written = scaffold_project(target)
    assert set(written) == {PROJECT_MANIFEST, ".gitignore", ".gitattributes", ".env.example", "README.md"}
    for rel in PROJECT_DIRS:
        assert (target / rel / ".gitkeep").is_file()
    assert read_project_manifest(target)["project"]["name"] == "proj"
    (target / "README.md").write_text("mine", encoding="utf-8")
    assert scaffold_project(target) == []
    assert (target / "README.md").read_text(encoding="utf-8") == "mine"


def test_scaffold_takes_a_display_name_and_writes_lf(tmp_path):
    scaffold_project(tmp_path / "p", name="My Project")
    raw = (tmp_path / "p" / PROJECT_MANIFEST).read_bytes()
    assert b'name = "My Project"' in raw and b"\r\n" not in raw


def test_adopt_copies_and_never_touches_the_source(tmp_path):
    src = tmp_path / "flat"
    src.mkdir()
    graph = {"name": "g", "nodes": [{"id": "a", "type": "Start", "position": {"x": 3, "y": 4},
                                     "data": {"params": {}}}], "edges": []}
    (src / "g.json").write_text(json.dumps(graph), encoding="utf-8")
    (src / "broken.json").write_text("{", encoding="utf-8")
    before = {p.name: p.read_bytes() for p in src.iterdir()}
    target = tmp_path / "proj"
    scaffold_project(target)
    result = adopt_graphs(src, target)
    assert result.adopted == ["g"]
    assert [name for name, _ in result.skipped] == ["broken.json"]
    assert (target / "graphs" / "g.graph.json").is_file()
    assert (target / "layout" / "g.layout.json").is_file()
    assert not (target / "graphs" / ".gitkeep").exists()
    assert {p.name: p.read_bytes() for p in src.iterdir()} == before


def test_adopt_refuses_an_ambiguous_source(tmp_path):
    src = tmp_path / "flat"
    src.mkdir()
    for name in ("x.json", "x.graph.json"):
        (src / name).write_text('{"nodes": [], "edges": []}', encoding="utf-8")
    scaffold_project(tmp_path / "proj")
    with pytest.raises(GraphAmbiguityError):
        adopt_graphs(src, tmp_path / "proj")
