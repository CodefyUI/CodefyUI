"""Phase 4B review round 1, item 5: the keyed merge driver for folder graphs.

Pure merges (random disjoint changes merge to the set semantics, overlapping
ones are refused), the git entry point (a fallback is git's own text merge
plus a non-zero exit), and real merges: through ``run_git`` (which defines
the driver per command) and plain ``git merge`` without it (git falls back to
its line merge, which conflicts but never merges wrong).
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from app.core.git import merge_driver as md
from app.core.git.runner import T_LOCAL, merge_driver_config, run_git
from app.core.graph_storage import GraphStore, explode, implode
from app.core.project import PROJECT_GITATTRIBUTES

from .test_git_service import isolated_git  # noqa: F401  (autouse fixture)

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="the host has no git")


def _uid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def _graph(rng: random.Random, n: int = 20) -> dict:
    ids = [_uid(rng) for _ in range(n)]
    return {
        "format_version": 1, "name": "g", "description": "",
        "nodes": [{"id": i, "type": "X", "data": {"params": {"k": 1}}} for i in ids],
        "edges": [{"id": _uid(rng), "source": ids[k], "target": ids[k + 1],
                   "sourceHandle": "o", "targetHandle": "i", "type": "data"}
                  for k in range(n - 1)],
        "presets": [], "subgraphs": [],
    }


def _edge_set(graph: dict) -> set:
    return {(e["source"], e["sourceHandle"], e["target"], e["targetHandle"]) for e in graph["edges"]}


def _change(rng: random.Random, graph: dict, taken: set) -> dict:
    """Add 2 wires and drop 1, touching no wire key in *taken* (so two
    calls make disjoint changes)."""
    g = json.loads(json.dumps(graph))
    ids = [n["id"] for n in g["nodes"]]
    free = [e for e in g["edges"] if (e["source"], e["sourceHandle"], e["target"], e["targetHandle"]) not in taken]
    gone = rng.choice(free)
    g["edges"].remove(gone)
    taken.add((gone["source"], gone["sourceHandle"], gone["target"], gone["targetHandle"]))
    for _ in range(2):
        while True:
            key = (rng.choice(ids), "o", rng.choice(ids), f"h{rng.randint(0, 9)}")
            if key not in taken and key not in _edge_set(g):
                break
        taken.add(key)
        g["edges"].append({"id": _uid(rng), "source": key[0], "target": key[2],
                           "sourceHandle": "o", "targetHandle": key[3], "type": "data"})
    return g


@pytest.mark.parametrize("seed", range(60))
def test_disjoint_edge_changes_merge_to_the_set_semantics(seed):
    rng = random.Random(seed)
    base = _graph(rng)
    taken: set = set()
    ours, theirs = _change(rng, base, taken), _change(rng, base, taken)
    # Each branch saves over the base's files, as the editor does: existing
    # wires keep their `o`.
    files = {"b": explode(base)}
    files["o"] = explode(ours, files["b"])
    files["t"] = explode(theirs, files["b"])
    merged = md.merge_edges(*(files[k]["edges.jsonl"].decode() for k in "bot"))
    want = (_edge_set(ours) & _edge_set(theirs)) | (_edge_set(ours) - _edge_set(base)) \
        | (_edge_set(theirs) - _edge_set(base))
    result = dict(files["o"])
    result["edges.jsonl"] = merged.encode()
    logic, repairs = implode(result)
    assert repairs == []
    assert _edge_set(logic) == want
    # Written in the storage's order: a later save rewrites nothing for it.
    assert merged.splitlines() == sorted(merged.splitlines(),
                                         key=lambda line: md._edge_key(json.loads(line)))


def test_one_wire_changed_differently_on_both_sides_is_a_conflict():
    base = '{"source": "a", "sourceHandle": "o", "target": "b", "targetHandle": "i", "type": "data", "id": "e"}\n'
    ours = base.replace('"data"', '"trigger"')
    theirs = base.replace('"id": "e"', '"id": "f"')
    with pytest.raises(md.Unmergeable):
        md.merge_edges(base, ours, theirs)
    # Both adding the same wire the same way is not a conflict.
    assert md.merge_edges("", base, base) == base


def test_layout_merges_by_id_and_keeps_a_sorted_side_sorted():
    def doc(positions):
        return json.dumps({"format_version": 1, "positions": positions, "notes": {},
                           "segmentGroups": [], "subgraphPositions": {}}, indent=2) + "\n"
    base = doc({"a": {"x": 0, "y": 0}, "c": {"x": 2, "y": 2}})
    ours = doc({"a": {"x": 5, "y": 0}, "b": {"x": 1, "y": 1}, "c": {"x": 2, "y": 2}})
    theirs = doc({"a": {"x": 0, "y": 0}, "c": {"x": 2, "y": 2}, "d": {"x": 3, "y": 3}})
    merged = json.loads(md.merge_layout(base, ours, theirs))
    assert list(merged["positions"]) == ["a", "b", "c", "d"]
    assert merged["positions"]["a"] == {"x": 5, "y": 0}
    with pytest.raises(md.Unmergeable):
        md.merge_layout(base, ours, doc({"a": {"x": 9, "y": 0}, "c": {"x": 2, "y": 2}}))


def test_index_merges_requires_as_sets():
    def doc(nodes, desc="d"):
        return json.dumps({"storage_version": 2, "format_version": 1, "name": "g",
                           "description": desc,
                           "requires": {"codefyui": ">=3.4", "plugins": {"p": {"nodes": nodes}}}})
    merged = json.loads(md.merge_index(doc(["p:A", "p:B"]), doc(["p:A", "p:B", "p:C"]),
                                       doc(["p:B"], desc="new")))
    assert merged["requires"]["plugins"]["p"]["nodes"] == ["p:B", "p:C"]
    assert merged["description"] == "new"
    assert list(merged)[:4] == ["storage_version", "format_version", "name", "description"]


@needs_git
def test_the_entry_point_falls_back_to_a_text_merge_and_says_conflict(tmp_path):
    base, ours, theirs = (tmp_path / n for n in ("O", "A", "B"))
    base.write_text("not json\n")
    ours.write_text("ours\n")
    theirs.write_text("theirs\n")
    assert md.main([str(base), str(ours), str(theirs), "graphs/g/edges.jsonl"]) == 1
    text = ours.read_text()
    assert "<<<<<<<" in text and "ours" in text and "theirs" in text


# ── Real merges ──────────────────────────────────────────────────────────

def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if check:
        assert proc.returncode == 0, proc.stderr
    return proc


def _two_branches(tmp_path: Path, seed: int) -> tuple[Path, dict, dict]:
    """A project with a folder graph on main, branch `a` and branch `b`
    each adding two nodes and three wires; `b` is checked out."""
    rng = random.Random(seed)
    root = tmp_path / f"repo{seed}"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    (root / "codefyui.project.toml").write_text('[project]\nname = "p"\n[storage]\ngraph_format = "dir"\n')
    (root / ".gitattributes").write_text(PROJECT_GITATTRIBUTES)
    store = GraphStore.for_project(root)
    base = _graph(rng, 30)
    for node in base["nodes"]:
        node["position"] = {"x": rng.randint(0, 999), "y": rng.randint(0, 999)}
    store.write("g", base)
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    ids = [n["id"] for n in base["nodes"]]

    def branch(name: str) -> dict:
        g = json.loads(json.dumps(base))
        new = [_uid(rng), _uid(rng)]
        g["nodes"] += [{"id": i, "type": "X", "position": {"x": 1, "y": 2},
                        "data": {"params": {}}} for i in new]
        g["edges"] += [
            {"id": _uid(rng), "source": rng.choice(ids), "target": new[0], "sourceHandle": "o",
             "targetHandle": "i", "type": "data"},
            {"id": _uid(rng), "source": new[0], "target": new[1], "sourceHandle": "o",
             "targetHandle": "i", "type": "data"},
            {"id": _uid(rng), "source": new[1], "target": rng.choice(ids), "sourceHandle": "o",
             "targetHandle": "aux", "type": "data"},
        ]
        _git(root, "checkout", "-q", "-B", name, "main")
        store.write("g", g)
        _git(root, "add", "-A")
        _git(root, "commit", "-qm", name)
        return g

    a, b = branch("a"), branch("b")
    return root, a, b


@needs_git
@pytest.mark.parametrize("seed", range(12))
def test_both_adding_nodes_and_wires_merge_cleanly_through_the_app(tmp_path, seed):
    root, a, b = _two_branches(tmp_path, seed)
    result = run_git(["merge", "--no-edit", "a"], cwd=root, timeout=T_LOCAL, ok_codes=(0, 1))
    assert result.returncode == 0, result.err + result.out
    got = GraphStore.for_project(root).read("g")
    assert "storage_repairs" not in got
    assert {n["id"] for n in got["nodes"]} == {n["id"] for n in a["nodes"]} | {n["id"] for n in b["nodes"]}
    assert {e["id"] for e in got["edges"]} == {e["id"] for e in a["edges"]} | {e["id"] for e in b["edges"]}
    assert all("position" in n for n in got["nodes"])


@needs_git
def test_without_the_driver_git_falls_back_to_its_line_merge(tmp_path):
    """The `merge=codefyui` attribute with no driver defined (a terminal
    merge before `cdui project git-setup`): git uses its text merge. Some
    seed conflicts on this layout; none merges wrong."""
    conflicted = 0
    for seed in range(6):
        root, a, b = _two_branches(tmp_path, seed)
        proc = _git(root, "merge", "--no-edit", "a", check=False)
        if proc.returncode != 0:
            conflicted += 1
            assert "CONFLICT" in proc.stdout + proc.stderr
            continue
        got = GraphStore.for_project(root).read("g")
        assert {e["id"] for e in got["edges"]} == \
            {e["id"] for e in a["edges"]} | {e["id"] for e in b["edges"]}
    assert conflicted > 0


@needs_git
def test_git_setup_defines_the_driver_for_terminal_merges(tmp_path):
    root, a, b = _two_branches(tmp_path, 3)
    for setting in merge_driver_config()[1::2]:
        key, _, value = setting.partition("=")
        _git(root, "config", "--local", key, value)
    proc = _git(root, "merge", "--no-edit", "a", check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_the_driver_rides_only_on_commands_that_merge(tmp_path):
    from app.core.git.runner import _argv

    def has(args):
        return any("merge.codefyui.driver=" in a for a in _argv("git", tmp_path, args))

    for args in (["merge", "a"], ["pull"], ["stash", "pop", "stash@{0}"],
                 ["stash", "apply", "stash@{0}"], ["cherry-pick", "x"], ["checkout", "b"]):
        assert has(args), args
    for args in (["status"], ["stash", "push"], ["add", "-A"], ["commit", "--file=-"]):
        assert not has(args), args
    command = merge_driver_config()[3]
    # The server's own interpreter and file, quoted; nothing from the repo.
    assert " -I " in command and command.endswith("%O %A %B %P")
