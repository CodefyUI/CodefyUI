"""cdui project init scaffolds a self-contained project dir and (via --adopt)
splits legacy graphs. Called in-process (conftest puts scripts/ on sys.path)."""

import argparse
import json
import shutil
import subprocess

import pytest

import project


def _args(**kw):
    ns = argparse.Namespace(dir=None, adopt=None, force=False)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_init_scaffolds_all_files(tmp_path):
    target = tmp_path / "svc"
    assert project.cmd_init(_args(dir=str(target))) == 0
    for rel in (
        "codefyui.project.toml", "graphs", "layout",
        "assets/images", "assets/models", "assets/data",
        ".env.example", ".gitignore", ".gitattributes", "README.md",
    ):
        assert (target / rel).exists(), rel
    toml_text = (target / "codefyui.project.toml").read_text(encoding="utf-8")
    assert 'name = "svc"' in toml_text
    assert "format_version = 1" in toml_text
    gi = (target / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gi and "*.safetensors" in gi
    # Controller-mandated addition (from the Task 3 review, carried forward
    # as a binding requirement on Task 7): the server's atomic two-file graph
    # writes (app.core.project._atomic_write) create `<name>.tmp-<suffix>`
    # temp files; an interrupted write can orphan one, so committed projects
    # must not track them.
    assert "*.tmp-*" in gi
    ga = (target / ".gitattributes").read_text(encoding="utf-8")
    assert "layout/*.layout.json linguist-generated=true" in ga


def test_init_refuses_nonempty_dir(tmp_path):
    target = tmp_path / "svc"
    target.mkdir()
    (target / "junk.txt").write_text("x", encoding="utf-8")
    assert project.cmd_init(_args(dir=str(target))) == 1


def test_init_adopt_splits_legacy_json(tmp_path):
    src = tmp_path / "old"
    src.mkdir()
    (src / "classifier.json").write_text(json.dumps({
        "name": "classifier",
        "nodes": [{"id": "a", "type": "Dataset",
                   "position": {"x": 3, "y": 4}, "data": {"params": {}}}],
        "edges": [],
    }), encoding="utf-8")
    target = tmp_path / "svc"
    assert project.cmd_init(_args(dir=str(target), adopt=str(src))) == 0
    assert (target / "graphs" / "classifier.graph.json").exists()
    layout = json.loads(
        (target / "layout" / "classifier.layout.json").read_text(encoding="utf-8"))
    assert layout["positions"]["a"] == {"x": 3, "y": 4}


def test_init_git_init_no_commit(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git not installed")
    target = tmp_path / "svc"
    project.cmd_init(_args(dir=str(target)))
    assert (target / ".git").is_dir()
    head = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "--verify", "HEAD"],
        capture_output=True)
    assert head.returncode != 0  # scaffold made NO initial commit


def test_dev_parse_project_flag():
    dev = pytest.importorskip("dev")
    assert dev._parse_project(["--project", "/tmp/svc"]) == "/tmp/svc"
    assert dev._parse_project(["--project=/tmp/svc"]) == "/tmp/svc"
    assert dev._parse_project(["--host", "0.0.0.0"]) is None
