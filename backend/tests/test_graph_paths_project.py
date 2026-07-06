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
