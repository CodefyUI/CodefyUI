"""Phase 4B: `cdui project migrate` moves graphs between the single-file and
folder forms, verifying each one before the old form is removed."""

import argparse
import json

import pytest

import project
from app.core import graph_storage as gs
from app.core.graph_storage import GraphStore

from .test_graph_storage_dir import LONG_CODE


def _init(tmp_path, fmt=None):
    target = tmp_path / "svc"
    project.cmd_init(argparse.Namespace(dir=str(target), adopt=None, force=False))
    if fmt is not None:
        project._set_manifest_graph_format(target / "codefyui.project.toml", fmt)
    return target


def _graph(name="g"):
    return {
        "name": name, "description": "d", "settings": {"seed": 3},
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": 1, "y": 2},
             "data": {"params": {}}},
            {"id": "py", "type": "PythonScript", "position": {"x": 30, "y": 40},
             "data": {"params": {"code": LONG_CODE}}},
        ],
        "edges": [{"id": "t", "source": "start", "target": "py",
                   "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"}],
        "presets": [], "subgraphs": [],
    }


def _margs(proj, to="dir", graph=None):
    return argparse.Namespace(dir=str(proj), to=to, graph=graph)


def _merged(proj, stem):
    data = GraphStore.for_project(proj).read(stem)
    data.pop("layout_missing", None)
    return data


def test_file_to_dir_and_back_keeps_the_graph(tmp_path, capsys):
    proj = _init(tmp_path, "file")
    GraphStore.for_project(proj).write("g", _graph())
    assert (proj / "graphs" / "g.graph.json").is_file()
    before = _merged(proj, "g")
    layout_bytes = (proj / "layout" / "g.layout.json").read_bytes()

    assert project.cmd_migrate(_margs(proj, "dir")) == 0
    assert not (proj / "graphs" / "g.graph.json").exists()
    assert (proj / "graphs" / "g" / "nodes" / "py.code.py").read_text() == LONG_CODE
    assert _merged(proj, "g") == before
    assert (proj / "layout" / "g.layout.json").read_bytes() == layout_bytes
    out = capsys.readouterr().out
    assert "git log --follow" in out
    assert 'graph_format = "dir"' in (proj / "codefyui.project.toml").read_text()

    assert project.cmd_migrate(_margs(proj, "file")) == 0
    assert not (proj / "graphs" / "g").exists()
    assert (proj / "graphs" / "g.graph.json").is_file()
    assert _merged(proj, "g") == before
    assert 'graph_format = "file"' in (proj / "codefyui.project.toml").read_text()


def test_a_legacy_single_file_migrates_with_its_positions(tmp_path):
    proj = _init(tmp_path, "file")
    (proj / "graphs" / "old.json").write_text(json.dumps(_graph("old")))
    assert project.cmd_migrate(_margs(proj, "dir")) == 0
    assert not (proj / "graphs" / "old.json").exists()
    merged = _merged(proj, "old")
    assert merged["nodes"][1]["position"] == {"x": 30, "y": 40}


def test_only_the_named_graph_moves_and_the_manifest_stays(tmp_path):
    proj = _init(tmp_path, "file")
    store = GraphStore.for_project(proj)
    store.write("a", _graph("a"))
    store.write("b", _graph("b"))
    assert project.cmd_migrate(_margs(proj, "dir", graph=["a"])) == 0
    assert (proj / "graphs" / "a").is_dir() and (proj / "graphs" / "b.graph.json").is_file()
    assert 'graph_format = "file"' in (proj / "codefyui.project.toml").read_text()


def test_a_failed_check_leaves_the_graph_exactly_as_it_was(tmp_path, monkeypatch):
    proj = _init(tmp_path, "file")
    GraphStore.for_project(proj).write("g", _graph())
    snapshot = {p.relative_to(proj).as_posix(): p.read_bytes()
                for p in (proj / "graphs").rglob("*") if p.is_file()}
    real = gs.implode

    def lossy(files):
        logic, repairs = real(files)
        logic["nodes"] = []
        return logic, repairs

    monkeypatch.setattr(gs, "implode", lossy)
    assert project.cmd_migrate(_margs(proj, "dir")) == 1
    after = {p.relative_to(proj).as_posix(): p.read_bytes()
             for p in (proj / "graphs").rglob("*") if p.is_file()}
    assert after == snapshot
    assert not (proj / "graphs" / "g").exists()
    assert 'graph_format = "file"' in (proj / "codefyui.project.toml").read_text()


def test_a_folder_that_needs_repairs_is_refused(tmp_path, capsys):
    proj = _init(tmp_path)
    GraphStore.for_project(proj).write("g", _graph())
    (proj / "graphs" / "g" / "nodes" / "start.json").unlink()
    assert project.cmd_migrate(_margs(proj, "file")) == 1
    captured = capsys.readouterr()
    assert "repair" in captured.out + captured.err
    assert (proj / "graphs" / "g").is_dir() and not (proj / "graphs" / "g.graph.json").exists()


def test_wires_a_folder_cannot_keep_are_refused(tmp_path, capsys):
    proj = _init(tmp_path, "file")
    graph = _graph()
    graph["edges"].append({"id": "ghost", "source": "nowhere", "target": "py",
                           "sourceHandle": "v", "targetHandle": "x", "type": "data"})
    (proj / "graphs" / "g.graph.json").write_text(json.dumps(graph))
    assert project.cmd_migrate(_margs(proj, "dir")) == 1
    assert (proj / "graphs" / "g.graph.json").is_file() and not (proj / "graphs" / "g").exists()


@pytest.mark.parametrize("to", ["dir", "file"])
def test_migrating_twice_is_a_no_op(tmp_path, to):
    proj = _init(tmp_path, to)
    GraphStore.for_project(proj).write("g", _graph())
    assert project.cmd_migrate(_margs(proj, to)) == 0
