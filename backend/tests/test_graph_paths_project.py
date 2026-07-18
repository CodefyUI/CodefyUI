"""Project-aware graph path resolution, reserved names, legacy acceptance,
and the both-files-exist ambiguity guard (spec ID2/ID7)."""

import json

import pytest

from app.api import routes_graph
from app.api.routes_graph import (
    GraphAmbiguityError,
    _graph_layout_path,
    _graph_logic_path,
    _graph_path,
    _reserved_graph_name,
)
from app.core import project as core_project
from app.core.project import collect_graph_files, resolve_graph_file


def _project(monkeypatch, tmp_path):
    """Repoint settings into a fresh project dir for the duration of a test."""
    monkeypatch.setattr(routes_graph.settings, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(routes_graph.settings, "GRAPHS_DIR", tmp_path / "graphs")
    (tmp_path / "graphs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "layout").mkdir(parents=True, exist_ok=True)


def test_non_project_path_is_legacy_single_file(monkeypatch, tmp_path):
    monkeypatch.setattr(routes_graph.settings, "PROJECT_DIR", None)
    monkeypatch.setattr(routes_graph.settings, "GRAPHS_DIR", tmp_path)
    assert _graph_path("foo") == tmp_path / "foo.json"
    assert _graph_logic_path("foo") == tmp_path / "foo.json"
    assert _graph_layout_path("foo") is None


def test_project_logic_and_layout_paths(monkeypatch, tmp_path):
    _project(monkeypatch, tmp_path)
    assert _graph_logic_path("foo") == tmp_path / "graphs" / "foo.graph.json"
    assert _graph_layout_path("foo") == tmp_path / "layout" / "foo.layout.json"


def test_project_prefers_canonical_over_missing(monkeypatch, tmp_path):
    _project(monkeypatch, tmp_path)
    # Neither exists -> canonical (non-existent) so callers' .exists() -> 404.
    assert _graph_path("foo") == tmp_path / "graphs" / "foo.graph.json"


def test_project_accepts_legacy_when_only_legacy_exists(monkeypatch, tmp_path):
    _project(monkeypatch, tmp_path)
    legacy = tmp_path / "graphs" / "foo.json"
    legacy.write_text(json.dumps({"nodes": [], "edges": []}))
    assert _graph_path("foo") == legacy


def test_project_ambiguity_raises_naming_both(monkeypatch, tmp_path):
    _project(monkeypatch, tmp_path)
    canonical = tmp_path / "graphs" / "foo.graph.json"
    legacy = tmp_path / "graphs" / "foo.json"
    canonical.write_text("{}")
    legacy.write_text("{}")
    with pytest.raises(GraphAmbiguityError) as ei:
        _graph_path("foo")
    assert ei.value.canonical == canonical
    assert ei.value.legacy == legacy
    assert "foo.graph.json" in str(ei.value)
    assert "foo.json" in str(ei.value)


def test_reserved_names():
    assert _reserved_graph_name("my.graph") is True
    assert _reserved_graph_name("my.layout") is True
    assert _reserved_graph_name("my-graph") is False
    assert _reserved_graph_name("classifier") is False


# -- The shared canonical-vs-legacy rule itself (issue #85): ONE source in
# -- app.core.project consumed by _graph_path, /list, and the project CLI.


def test_routes_ambiguity_error_is_the_core_class():
    """routes_graph re-exports the single shared exception -- catching either
    import path must catch the same class."""
    assert routes_graph.GraphAmbiguityError is core_project.GraphAmbiguityError


def test_resolve_graph_file_four_branches(tmp_path):
    # Neither exists -> canonical (non-existent) so callers can 404.
    assert resolve_graph_file(tmp_path, "foo") == tmp_path / "foo.graph.json"
    # Only legacy -> legacy accepted.
    legacy = tmp_path / "foo.json"
    legacy.write_text("{}")
    assert resolve_graph_file(tmp_path, "foo") == legacy
    # Both -> ambiguity error carrying both paths.
    canonical = tmp_path / "foo.graph.json"
    canonical.write_text("{}")
    with pytest.raises(GraphAmbiguityError) as ei:
        resolve_graph_file(tmp_path, "foo")
    assert ei.value.canonical == canonical
    assert ei.value.legacy == legacy
    # Only canonical -> canonical.
    legacy.unlink()
    assert resolve_graph_file(tmp_path, "foo") == canonical


def test_resolve_graph_file_display_name_decorates_error(tmp_path):
    (tmp_path / "weird_name.graph.json").write_text("{}")
    (tmp_path / "weird_name.json").write_text("{}")
    with pytest.raises(GraphAmbiguityError) as ei:
        resolve_graph_file(tmp_path, "weird_name", display_name="weird name")
    assert ei.value.name == "weird name"
    assert "weird name" in str(ei.value)


def test_collect_graph_files_mixed_dir_sorted_and_layout_skipped(tmp_path):
    (tmp_path / "b_canon.graph.json").write_text("{}")
    (tmp_path / "a_legacy.json").write_text("{}")
    (tmp_path / "stray.layout.json").write_text("{}")
    got = collect_graph_files(tmp_path)
    assert got == [
        ("a_legacy", tmp_path / "a_legacy.json"),
        ("b_canon", tmp_path / "b_canon.graph.json"),
    ]


def test_collect_graph_files_ambiguity_raises_naming_both(tmp_path):
    (tmp_path / "dup.graph.json").write_text("{}")
    (tmp_path / "dup.json").write_text("{}")
    with pytest.raises(GraphAmbiguityError) as ei:
        collect_graph_files(tmp_path)
    assert "dup.graph.json" in str(ei.value)
    assert "dup.json" in str(ei.value)


def test_collect_graph_files_missing_dir_is_empty(tmp_path):
    assert collect_graph_files(tmp_path / "nope") == []
