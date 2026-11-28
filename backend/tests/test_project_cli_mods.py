"""Phase 4C CLI: `cdui project mods [--trust [--yes] [--full]] [--write]`,
`freeze --used-only` and the manifest's [mods], edited in place; the CLI
loads mods only on a trust record that matches the tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import project
from app.core import mods
from app.core.node_registry import registry
from app.core.plugin_loader import tomllib

from .test_projects_startup import isolated  # noqa: F401  (fixture)

NODE = (
    "from app.core.node_base import BaseNode\n\n\n"
    "class Focal(BaseNode):\n"
    "    NODE_NAME = 'FocalLoss'\n"
    "    CATEGORY = 'Loss'\n"
    "    @classmethod\n"
    "    def define_inputs(cls):\n"
    "        return []\n"
    "    @classmethod\n"
    "    def define_outputs(cls):\n"
    "        return []\n"
    "    def execute(self, inputs, params):\n"
    "        return {}\n"
)

MANIFEST = (
    "# my service -- this comment must survive\n"
    "[project]\n"
    'name = "svc"  # trailing comment\n'
    "format_version = 1\n"
    "\n"
    "# pins below\n"
    "[plugins]\n"
    'stale = { url = "https://github.com/o/stale", ref = "v0", sha = "' + "d" * 40 + '" }\n'
    "\n"
    "# notes about publishing\n"
    "[publish]\n"
    'graph = "main"\n'
)


def _write(path: Path, data: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode("utf-8"))
    return path


def _project(root: Path, files: dict[str, str]) -> Path:
    project.cmd_init(argparse.Namespace(dir=str(root), adopt=None, force=False))
    _write(root / "codefyui.project.toml", MANIFEST)
    for rel, data in files.items():
        _write(root / "mods" / rel, data)
    return root.resolve()


def _mods_args(proj: Path, **kw) -> argparse.Namespace:
    return argparse.Namespace(dir=str(proj), trust=kw.get("trust", False),
                              yes=kw.get("yes", False), full=kw.get("full", False),
                              write=kw.get("write", False))


@pytest.fixture(autouse=True)
def _unloaded():
    mods.unload_project_mods()
    yield
    mods.unload_project_mods()


# ── the in-place table editor ────────────────────────────────────────────

def test_a_table_is_replaced_in_place_and_comments_stay():
    text = MANIFEST
    out = project._set_toml_table(text, "plugins", ["a = 1"])
    assert "# my service -- this comment must survive" in out
    assert 'name = "svc"  # trailing comment' in out
    assert "# pins below" in out and "# notes about publishing" in out
    assert tomllib.loads(out)["plugins"] == {"a": 1}
    assert tomllib.loads(out)["publish"] == {"graph": "main"}
    assert project._set_toml_table(out, "plugins", ["a = 1"]) == out   # idempotent
    added = project._set_toml_table(out, "mods", ['"x" = 1'])
    assert tomllib.loads(added)["mods"] == {"x": 1}
    removed = project._set_toml_table(added, "mods", None)
    assert removed == out


# ── cdui project mods ────────────────────────────────────────────────────

def test_mods_shows_the_scan_and_records_trust_with_yes(isolated, capsys):  # noqa: F811
    p = _project(isolated / "svc", {"nodes/loss.py": NODE})
    assert project.cmd_mods(_mods_args(p)) == 0
    shown = capsys.readouterr().out
    assert "mods/nodes/loss.py" in shown
    assert "project:FocalLoss" in shown
    assert mods.mods_digest(mods.walk_mods(p)) in shown
    assert "trust: untrusted" in shown
    assert "a hint, not a sandbox" in shown
    assert project.cmd_mods(_mods_args(p, trust=True, yes=True)) == 0
    assert mods.mods_trust_state(p)["state"] == "trusted"


def test_mods_trust_without_a_terminal_or_yes_records_nothing(isolated, monkeypatch):  # noqa: F811
    p = _project(isolated / "svc", {"nodes/loss.py": NODE})
    monkeypatch.setattr(project, "_stdin_is_interactive", lambda: False)
    assert project.cmd_mods(_mods_args(p, trust=True)) == 1
    assert mods.mods_trust_state(p)["state"] == "untrusted"
    # In a terminal, "n" records nothing either.
    monkeypatch.setattr(project, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    assert project.cmd_mods(_mods_args(p, trust=True)) == 1
    assert mods.mods_trust_state(p)["state"] == "untrusted"
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    assert project.cmd_mods(_mods_args(p, trust=True)) == 0
    assert mods.mods_trust_state(p)["state"] == "trusted"


def test_a_blocked_tree_cannot_be_trusted(isolated):  # noqa: F811
    p = _project(isolated / "svc", {"nodes/loss.py": NODE,
                                    "nodes/__pycache__/loss.cpython-310.pyc": "junk"})
    assert project.cmd_mods(_mods_args(p, trust=True, yes=True)) == 1
    assert mods.mods_trust_state(p)["state"] == "blocked"


# ── the CLI loads mods only on a matching record ─────────────────────────

def test_validate_loads_mods_only_when_the_record_matches(isolated):  # noqa: F811
    p = _project(isolated / "svc", {"nodes/loss.py": NODE})
    args = argparse.Namespace(dir=str(p), graph=None, strict=False)
    project.cmd_validate(args)
    assert registry.get("project:FocalLoss") is None
    project.cmd_mods(_mods_args(p, trust=True, yes=True))
    project.cmd_validate(args)
    assert registry.get("project:FocalLoss") is not None
    # A change after the record: the CLI does not load it.
    _write(p / "mods" / "nodes" / "loss.py", NODE + "# changed\n")
    project.cmd_validate(args)
    assert registry.get("project:FocalLoss") is None


def test_an_owned_project_is_not_auto_trusted_outside_the_server(isolated, monkeypatch):  # noqa: F811
    from app.core import projects

    # Pinned: a test earlier in the run may have started the app's lifespan
    # in this process, which marks it the server.
    monkeypatch.setattr(mods, "_server_process", False)
    p = _project(isolated / "svc", {})
    projects.mark_owned(p)                     # created here, no mods yet
    _write(p / "mods" / "nodes" / "loss.py", NODE)   # the owner adds one
    assert mods.load_project_mods(p)["loaded"] is False   # not the server process
    assert mods._entry(p).get("mods_sha256") is None      # nothing recorded for it


def test_an_owned_project_loads_and_records_in_the_server(isolated, monkeypatch):  # noqa: F811
    from app.core import projects

    monkeypatch.setattr(mods, "_server_process", True)
    p = _project(isolated / "svc", {"nodes/loss.py": NODE})
    projects.mark_owned(p)
    assert mods.load_project_mods(p)["nodes"] == ["project:FocalLoss"]
    assert mods._entry(p)["mods_sha256"] == mods.mods_digest(mods.walk_mods(p))


# ── [mods], freeze --used-only ───────────────────────────────────────────

def test_mods_write_and_the_validate_warning(isolated, capsys):  # noqa: F811
    p = _project(isolated / "svc", {
        "nodes/loss.py": NODE,
        "presets/aug.json": json.dumps({"preset_name": "Strong Aug"})})
    manifest = p / "codefyui.project.toml"
    assert project._check_mods_block(p, tomllib.loads(manifest.read_text())) is True
    assert project.cmd_mods(_mods_args(p, write=True)) == 0
    text = manifest.read_text(encoding="utf-8")
    assert "# my service -- this comment must survive" in text
    block = tomllib.loads(text)["mods"]
    assert block["nodes/loss.py"]["nodes"] == ["FocalLoss"]
    assert block["presets/aug.json"]["preset"] == "Strong Aug"
    assert len(block["nodes/loss.py"]["sha256"]) == 64
    assert project._check_mods_block(p, tomllib.loads(text)) is False
    # [mods] is never trust: writing it trusted nothing.
    assert mods.mods_trust_state(p)["state"] == "untrusted"
    _write(p / "mods" / "nodes" / "loss.py", NODE + "# edit\n")
    capsys.readouterr()
    assert project._check_mods_block(p, tomllib.loads(text)) is True
    assert "cdui project mods --write" in capsys.readouterr().out


def _graph(types: list[str]) -> str:
    return json.dumps({"name": "g", "nodes": [
        {"id": f"n{i}", "type": t, "data": {"params": {}}} for i, t in enumerate(types)],
        "edges": []})


def test_freeze_used_only_pins_what_the_graphs_use(isolated, monkeypatch):  # noqa: F811
    p = _project(isolated / "svc", {"nodes/loss.py": NODE})
    _write(p / "graphs" / "main.graph.json",
           _graph(["Start", "pack-a:Embed", "gone:X", "project:FocalLoss"]))
    lock = {"schema": 1, "plugins": {
        "pack-a": {"source_kind": "github_url", "url": "https://github.com/o/pack-a",
                   "ref": "v1", "sha": "a" * 40},
        "pack-b": {"source_kind": "github_url", "url": "https://github.com/o/pack-b",
                   "ref": "v1", "sha": "b" * 40}}}
    monkeypatch.setattr(project, "load_lockfile", lambda: lock)
    assert project.cmd_freeze(argparse.Namespace(dir=str(p), used_only=True)) == 0
    text = (p / "codefyui.project.toml").read_text(encoding="utf-8")
    manifest = tomllib.loads(text)
    assert set(manifest["plugins"]) == {"pack-a"}            # pack-b unused, gone not installed
    assert "nodes/loss.py" in manifest["mods"]               # freeze writes [mods] too
    assert "# pins below" in text and "# notes about publishing" in text
    # Without the flag: every pinnable plugin, as before.
    assert project.cmd_freeze(argparse.Namespace(dir=str(p), used_only=False)) == 0
    manifest = tomllib.loads((p / "codefyui.project.toml").read_text(encoding="utf-8"))
    assert set(manifest["plugins"]) == {"pack-a", "pack-b"}


def test_freeze_used_only_fails_on_an_unreadable_graph(isolated, monkeypatch):  # noqa: F811
    p = _project(isolated / "svc", {})
    _write(p / "graphs" / "bad.graph.json", "{ not json")
    monkeypatch.setattr(project, "load_lockfile", lambda: {"schema": 1, "plugins": {}})
    assert project.cmd_freeze(argparse.Namespace(dir=str(p), used_only=True)) == 1
