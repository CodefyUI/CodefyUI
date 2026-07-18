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


def test_init_adopt_canonical_source_keeps_base(tmp_path):
    """A source file already named `<base>.graph.json` adopts under `<base>`
    (never a doubled `mix.graph.graph.json`)."""
    src = tmp_path / "old"
    src.mkdir()
    (src / "mix.graph.json").write_text(json.dumps(
        {"name": "mix", "nodes": [], "edges": []}), encoding="utf-8")
    target = tmp_path / "svc"
    assert project.cmd_init(_args(dir=str(target), adopt=str(src))) == 0
    assert (target / "graphs" / "mix.graph.json").exists()
    assert not (target / "graphs" / "mix.graph.graph.json").exists()


def test_init_adopt_skips_layout_files(tmp_path):
    """`*.layout.json` in the source dir is layout, not a graph -- never
    adopted as a graph named '<base>.layout'."""
    src = tmp_path / "old"
    src.mkdir()
    (src / "keep.json").write_text(json.dumps(
        {"name": "keep", "nodes": [], "edges": []}), encoding="utf-8")
    (src / "stray.layout.json").write_text(json.dumps(
        {"format_version": 1, "positions": {}}), encoding="utf-8")
    target = tmp_path / "svc"
    assert project.cmd_init(_args(dir=str(target), adopt=str(src))) == 0
    adopted = sorted(p.name for p in (target / "graphs").glob("*.graph.json"))
    assert adopted == ["keep.graph.json"]


def test_init_adopt_ambiguous_pair_fails_naming_both(tmp_path, capsys):
    """CONVERGED RULE (issue #85): a source dir holding BOTH `dup.json` and
    `dup.graph.json` is the same ambiguity the server 409s on and validate
    fails on -- adopt must surface it identically (it previously wrote both
    payloads to the same target, so whichever sorted last silently won)."""
    src = tmp_path / "old"
    src.mkdir()
    (src / "dup.json").write_text(json.dumps(
        {"name": "legacy wins?", "nodes": [], "edges": []}), encoding="utf-8")
    (src / "dup.graph.json").write_text(json.dumps(
        {"name": "canonical wins?", "nodes": [], "edges": []}), encoding="utf-8")
    target = tmp_path / "svc"
    assert project.cmd_init(_args(dir=str(target), adopt=str(src))) == 1
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "dup.graph.json" in out
    assert "dup.json" in out
    # Nothing half-adopted: the ambiguity aborts before any split is written.
    assert list((target / "graphs").glob("*.json")) == []


def test_init_adopt_removes_stale_gitkeep(tmp_path):
    src = tmp_path / "old"
    src.mkdir()
    (src / "g.json").write_text(json.dumps({
        "name": "g", "nodes": [], "edges": [],
    }), encoding="utf-8")
    target = tmp_path / "svc"
    assert project.cmd_init(_args(dir=str(target), adopt=str(src))) == 0
    # Real files landed, so the scaffold placeholders must be gone ...
    assert (target / "graphs" / "g.graph.json").exists()
    assert not (target / "graphs" / ".gitkeep").exists()
    assert not (target / "layout" / ".gitkeep").exists()
    # ... while still-empty assets dirs keep theirs.
    assert (target / "assets" / "images" / ".gitkeep").exists()


def test_init_adopt_empty_src_keeps_gitkeep(tmp_path):
    src = tmp_path / "old"
    src.mkdir()
    target = tmp_path / "svc"
    assert project.cmd_init(_args(dir=str(target), adopt=str(src))) == 0
    assert (target / "graphs" / ".gitkeep").exists()
    assert (target / "layout" / ".gitkeep").exists()


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
