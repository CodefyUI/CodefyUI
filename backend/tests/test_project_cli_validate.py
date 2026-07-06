"""cdui project validate mirrors the publish six-gate pre-flight per graph,
plus project-level .env-tracked and plugin-pin checks (spec ID3)."""

import argparse
import json
import shutil
import subprocess

import pytest

import project
from app.core.plugin_loader import tomllib
from app.core.secret_params import secret_param_names


def _init(tmp_path):
    target = tmp_path / "svc"
    project.cmd_init(argparse.Namespace(dir=str(target), adopt=None, force=False))
    return target


def _echo_graph(name="echo"):
    return {
        "format_version": 1, "name": name, "description": "",
        "nodes": [
            {"id": "start", "type": "Start", "data": {"params": {}}},
            {"id": "gi", "type": "GraphInput", "data": {"params": {
                "name": "x", "type": "string", "required": True,
                "default": "", "description": ""}}},
            {"id": "out", "type": "GraphOutput",
             "data": {"params": {"name": "y", "description": ""}}},
        ],
        "edges": [
            {"id": "t1", "source": "start", "target": "gi",
             "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
            {"id": "d1", "source": "gi", "target": "out",
             "sourceHandle": "value", "targetHandle": "value", "type": "data"},
        ],
        "presets": [],
    }


def _write_graph(proj, graph, base=None):
    base = base or graph["name"]
    (proj / "graphs" / f"{base}.graph.json").write_text(json.dumps(graph))


def _vargs(proj, strict=False):
    return argparse.Namespace(dir=str(proj), strict=strict)


def test_valid_contract_graph_passes(tmp_path):
    proj = _init(tmp_path)
    _write_graph(proj, _echo_graph())
    assert project.cmd_validate(_vargs(proj)) == 0


def test_no_entry_points_fails(tmp_path):
    proj = _init(tmp_path)
    g = _echo_graph()
    g["nodes"] = [n for n in g["nodes"] if n["type"] != "Start"]
    g["edges"] = [e for e in g["edges"] if e.get("type") != "trigger"]
    _write_graph(proj, g)
    assert project.cmd_validate(_vargs(proj)) == 1


def test_unknown_node_type_specific_message(tmp_path, capsys):
    proj = _init(tmp_path)
    g = _echo_graph()
    g["nodes"].append({"id": "ghost", "type": "GhostPluginNode",
                       "data": {"params": {}}})
    _write_graph(proj, g)
    assert project.cmd_validate(_vargs(proj)) == 1
    # FAIL lines print via err() (stderr, per this CLI's convention -- see
    # plugins.py) rather than stdout; check both streams combined so this
    # assertion reflects what a CI log actually shows.
    captured = capsys.readouterr()
    out = (captured.out + captured.err).lower()
    assert "unknown node type" in out and "restore" in out


def test_secret_in_graph_fails(tmp_path):
    proj = _init(tmp_path)
    names = secret_param_names("LLMChat")
    assert names, "LLMChat must declare a SECRET param"
    secret = sorted(names)[0]
    g = _echo_graph()
    g["nodes"].append({"id": "llm", "type": "LLMChat",
                       "data": {"params": {secret: "sk-leak"}}})
    _write_graph(proj, g)
    assert project.cmd_validate(_vargs(proj)) == 1


def test_embedded_preset_graph_passes(tmp_path):
    proj = _init(tmp_path)
    g = _echo_graph()
    g["nodes"].append({"id": "p", "type": "preset:EmbeddedPr", "data": {}})
    g["presets"] = [{
        "preset_name": "EmbeddedPr", "category": "Custom", "description": "",
        "tags": [],
        "nodes": [{"id": "inner", "type": "Print", "params": {"label": "x"}}],
        "edges": [],
        "exposed_inputs": [{"name": "in", "internal_node": "inner",
                            "internal_port": "value", "data_type": "ANY",
                            "description": ""}],
        "exposed_outputs": [], "exposed_params": [],
    }]
    _write_graph(proj, g)
    assert project.cmd_validate(_vargs(proj)) == 0
    # Without the embedded definition it fails (Unknown preset).
    g["presets"] = []
    _write_graph(proj, g)
    assert project.cmd_validate(_vargs(proj)) == 1


def test_env_tracked_fails(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git not installed")
    proj = _init(tmp_path)
    _write_graph(proj, _echo_graph())
    (proj / ".env").write_text("SECRET=1")
    subprocess.run(["git", "-C", str(proj), "add", "-f", ".env"],
                   capture_output=True)
    assert project.cmd_validate(_vargs(proj)) == 1


def test_strict_pin_missing(tmp_path, monkeypatch):
    proj = _init(tmp_path)
    (proj / "codefyui.project.toml").write_text(
        '[project]\nname = "svc"\nformat_version = 1\n\n'
        '[plugins]\nghostpack = { url = "https://github.com/x/y", '
        'ref = "v1", sha = "' + "0" * 40 + '" }\n')
    monkeypatch.setattr(project, "load_lockfile",
                        lambda: {"schema": 1, "plugins": {}})
    assert project.cmd_validate(_vargs(proj, strict=False)) == 0  # warn only
    assert project.cmd_validate(_vargs(proj, strict=True)) == 1   # strict error
