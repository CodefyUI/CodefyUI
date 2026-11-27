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
    # The same layout; a folder graph writes its id-keyed maps sorted by id.
    assert json.loads((proj / "layout" / "g.layout.json").read_bytes()) == json.loads(layout_bytes)
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


def test_git_setup_defines_the_driver_and_adds_the_attribute_lines(tmp_path):
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("no git")
    proj = _init(tmp_path)
    subprocess.run(["git", "-C", str(proj), "init", "-q"], check=True)
    # A project from before the merge lines.
    (proj / ".gitattributes").write_text("*.json text eol=lf\n")
    assert project.cmd_git_setup(argparse.Namespace(dir=str(proj))) == 0
    driver = subprocess.run(["git", "-C", str(proj), "config", "--local", "merge.codefyui.driver"],
                            capture_output=True, text=True).stdout
    assert "merge_driver.py" in driver and "%O %A %B %P" in driver and " -I " in driver
    attrs = (proj / ".gitattributes").read_text()
    assert "graphs/**/edges.jsonl merge=codefyui" in attrs
    assert attrs.startswith("*.json text eol=lf\n")
    # Running it again adds nothing twice.
    assert project.cmd_git_setup(argparse.Namespace(dir=str(proj))) == 0
    assert (proj / ".gitattributes").read_text() == attrs


def test_every_form_is_written_with_lf_and_a_crlf_file_comes_back_lf(tmp_path):
    """The server writes LF on every platform (Windows CI once saw a v1 save
    write CRLF through text mode). A single file that HAS CRLF -- edited on
    Windows without the attribute -- is read as the same graph and written
    back LF by migrate, like by any save: line endings are not content."""
    proj = _init(tmp_path, "file")
    store = GraphStore.for_project(proj)
    store.write("g", _graph())
    for path in ((proj / "graphs" / "g.graph.json"), (proj / "layout" / "g.layout.json")):
        assert b"\r" not in path.read_bytes(), path
    crlf = (proj / "graphs" / "g.graph.json").read_bytes().replace(b"\n", b"\r\n")
    (proj / "graphs" / "g.graph.json").write_bytes(crlf)
    before = _merged(proj, "g")
    assert project.cmd_migrate(_margs(proj, "dir")) == 0
    assert project.cmd_migrate(_margs(proj, "file")) == 0
    assert _merged(proj, "g") == before
    for path in (proj / "graphs" / "g.graph.json", proj / "layout" / "g.layout.json"):
        assert b"\r" not in path.read_bytes(), path
