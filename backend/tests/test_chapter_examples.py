"""Smoke-test every chapter plugin example by actually executing its graph.

Each chapter plugin has at least one ``plugins/<plugin>/examples/**/graph.json``
that the textbook references. A broken graph wastes the student's first
attempt at running the example, so this test asserts each graph parses,
validates, and executes end-to-end without error.

Parametrised by glob so adding a new example file is enough — no test
code changes required.

A second check runs over the same graphs: every node ``type`` must be an
EXACT key in the node registry. ``validate_graph`` is deliberately lenient
about that and the canvas is not — see
``test_chapter_graph_node_types_match_the_palette_exactly``.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from app.config import settings
from app.core.graph_engine import execute_graph, validate_graph
from app.core.node_registry import NodeRegistry
from app.core.plugin_loader import install_plugin_finder

from tests._example_graphs import graph_nodes

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLUGIN_ROOT = _REPO_ROOT / "plugins"


def _discover_chapter_graphs() -> list[Path]:
    return sorted(_PLUGIN_ROOT.glob("*/examples/**/graph.json"))


_GRAPHS = _discover_chapter_graphs()
assert _GRAPHS, "chapter graph smoke suite discovered no plugin examples"

# Node types that pull a real dataset, train for multiple epochs, download
# weights, or otherwise take longer than a few seconds. These graphs ship as
# fully-working teaching examples but are skipped in the fast smoke test —
# students run them manually. The unit tests for the underlying nodes already
# cover correctness.
_SLOW_NODE_TYPES = {
    "Dataset",
    "DataLoader",
    "TrainingLoop",
    "preset:Training Pipeline",
    "ModelLoader",
    "ModelSaver",
    "HuggingFaceDataset",
    "KaggleDataset",
    "Inference",
    # Needs the sentence-embeddings pack; graphs are still validated. Listed
    # even though no chapter example uses it yet -- CI has no pack cache, so
    # a pack example added later would fail at the gate rather than run, and
    # a machine that DOES have the pack would load half a gigabyte of weights
    # inside the fast smoke suite.
    "TextEmbedding",
}


def _is_slow(payload: dict) -> bool:
    return any(n.get("type") in _SLOW_NODE_TYPES for n in payload.get("nodes", []))


@pytest.mark.parametrize(
    "graph_path",
    _GRAPHS,
    ids=[p.relative_to(_PLUGIN_ROOT).as_posix() for p in _GRAPHS],
)
def test_chapter_graph_executes(graph_path: Path):
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = payload["nodes"]
    edges = payload["edges"]

    errors = validate_graph(nodes, edges)
    assert not errors, f"validate_graph errors for {graph_path}: {errors}"

    if _is_slow(payload):
        pytest.skip(
            "Graph pulls a real dataset / trains a model — validated structurally, "
            "manual run required for full execution."
        )

    # Some example graphs reference data files via paths relative to the
    # backend cwd (CSVReader's default is ``data/samples/iris.csv``).
    # ``cdui test`` always launches pytest from backend/, but a contributor
    # running ``pytest`` directly from the repo root would otherwise hit a
    # FileNotFoundError. Hop cwd just for this test to keep both invocations
    # working — restore afterwards so other tests aren't affected.
    backend_dir = Path(__file__).resolve().parents[1]
    prev_cwd = Path.cwd()
    os.chdir(backend_dir)
    try:
        # execute_graph is async; pytest-asyncio is set up for async fixtures
        # but plain ``asyncio.run`` is the simplest pattern when we only need
        # a one-shot execution per graph.
        asyncio.run(execute_graph(nodes, edges, error_mode="fail_fast"))
    finally:
        os.chdir(prev_cwd)


# ── every example type is a type the palette actually has ─────────────────

#: Container node types that are not registry entries and are resolved
#: elsewhere: presets come from ``preset_registry`` (with a graph-embedded
#: fallback for presets the server has never seen), subgraph instances are
#: expanded from the graph's own ``subgraphs`` list.
_NON_REGISTRY_PREFIXES = ("preset:", "subgraph:")

#: Every pack in the repo, by manifest directory — the same auto-discovery
#: the graph glob above uses, so a new pack is covered without editing this
#: file.
_IN_REPO_PACKS = sorted(m.parent.name for m in _PLUGIN_ROOT.glob("*/cdui.plugin.toml"))


@pytest.fixture(scope="module")
def palette() -> set[str]:
    """The registry keys an install of the in-repo packs would offer.

    Built here rather than read off the session-wide registry singleton on
    purpose. The singleton's contents depend on what earlier tests did to it
    — ``test_plugin_api`` runs the app lifespan against a lockfile naming
    three packs, which drops the others — and a check about whether examples
    match the palette must not be able to pass or fail on collection order.

    It is also the point of the check: the keys are produced by the real
    discovery path, ``install_plugin_finder`` → ``discover``, so they are
    exactly the strings ``/api/nodes`` would publish. Built-in nodes are
    included because pack examples wire pack nodes to them; user custom nodes
    are not, because a shipped example must never depend on one.
    """
    reg = NodeRegistry()
    reg.discover(settings.NODES_DIR, "app.nodes")
    lockfile = {
        "schema": 1,
        "plugins": {
            pack: {"source_kind": "builtin", "source": pack, "enabled": True}
            for pack in _IN_REPO_PACKS
        },
    }
    for ns in install_plugin_finder(
        _PLUGIN_ROOT,
        _REPO_ROOT / "_phantom_user_root_for_tests",  # never read
        lockfile,
    ):
        reg.discover(ns.nodes_dir, ns.package_name, plugin_id=ns.plugin_id)

    keys = set(reg.nodes)
    assert _IN_REPO_PACKS, "found no packs under plugins/ -- the scan is broken"
    assert any(":" in k for k in keys), (
        "no pack node was discovered -- this check would pass vacuously"
    )
    return keys


@pytest.mark.parametrize(
    "graph_path",
    _GRAPHS,
    ids=[p.relative_to(_PLUGIN_ROOT).as_posix() for p in _GRAPHS],
)
def test_chapter_graph_node_types_match_the_palette_exactly(
    graph_path: Path, palette: set[str]
):
    """Every node type in a pack example is a key the palette actually has.

    Read through ``graph_nodes``, so a note the example carries to explain
    itself is not asked to be a registry key -- it is an annotation, and
    the canvas renders it from the node list without consulting a
    definition.

    ``test_chapter_graph_executes`` above does NOT cover this.
    ``registry.get`` falls back to a suffix scan, so a graph asking for a
    bare ``Edu-KNN`` while the registry holds ``foundations:Edu-KNN``
    validates and executes server-side and looks entirely healthy from here.

    The canvas has no such fallback. ``resolveSerializedNodes`` does an exact
    ``Map.get(node.type)`` over the definitions from ``/api/nodes`` and, on a
    miss, substitutes an EMPTY definition: the node renders as a blank box
    badged "Utility" with no ports at all, and because the handles do not
    exist every edge touching it is silently dropped. Nothing is logged and
    nothing fails — the student just gets a broken-looking example.

    So this guards the whole class rather than one spelling of it: any pack
    whose registry key stops agreeing with the id its examples were written
    against — a rename, a typo, or an id that the loader spells one way and
    the manifest another — fails here instead of on a student's screen.
    """
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    unresolved = sorted(
        {
            node_type
            for node in graph_nodes(payload)
            for node_type in [str(node.get("type", ""))]
            if not node_type.startswith(_NON_REGISTRY_PREFIXES)
            and node_type not in palette
        }
    )
    assert not unresolved, (
        f"{graph_path} uses node types that are not registry keys: "
        f"{unresolved}. The canvas resolves a node type by exact match, so "
        f"each of these renders as an empty box with no ports. Plugin nodes "
        f"must be written qualified, as \"<plugin-id>:<NodeName>\" — the "
        f"plugin id exactly as its cdui.plugin.toml spells it."
    )


# ── a pack trainer reports a number, on the split it did not train on ─────

#: The deep pack's MNIST trainer. It trains inside
#: ``preset:Training Pipeline``, so ``_SLOW_NODE_TYPES`` keeps it out of the
#: smoke run above and structure is the only thing a test can see here --
#: which is also the thing that rots. A ``split`` flipped to ``train``
#: scores the model on the images it memorised, and a dropped ``accuracy``
#: edge takes the number off the screen again; both still validate, and
#: both still train.
_LENET_TRAINER = (_PLUGIN_ROOT / "deep" / "examples" / "C3-1"
                  / "LeNet-MNIST-Training" / "graph.json")


def test_the_lenet_pack_trainer_scores_the_split_it_did_not_train_on():
    """Train on ``train``, report ``test``, and put the number on screen.

    The dataset the example trains on is read off the preset's own
    ``internalParams`` rather than named here, so the two halves cannot
    drift apart: an example repointed at another dataset has to repoint its
    evaluation with it. The built-in quick-start trainer holds the same
    contract in ``test_builtin_examples.py``.
    """
    payload = json.loads(_LENET_TRAINER.read_text(encoding="utf-8"))
    edges = payload["edges"]
    by_id = {node["id"]: node for node in graph_nodes(payload)}

    def feeding(target: str, handle: str) -> dict:
        sources = [e["source"] for e in edges
                   if e.get("type", "data") == "data"
                   and e["target"] == target and e.get("targetHandle") == handle]
        assert len(sources) == 1, f"{target}.{handle} is fed by {sources}"
        return by_id[sources[0]]

    pipeline = feeding("eval", "model")
    assert pipeline["type"] == "preset:Training Pipeline", pipeline["type"]
    trained_on = pipeline["data"]["internalParams"]["dataset"]
    assert trained_on["split"] == "train", trained_on

    scored_on = feeding("eval", "dataset")
    assert scored_on["type"] == "Dataset", scored_on["type"]
    assert scored_on["data"]["params"]["name"] == trained_on["name"], (
        f"the example trains on {trained_on['name']} and scores itself on "
        f"{scored_on['data']['params']['name']}")
    assert scored_on["data"]["params"]["split"] == "test", (
        f"{scored_on['id']} reads the {scored_on['data']['params']['split']!r} "
        f"split -- an accuracy measured on the images the loop trained on is "
        f"not an accuracy")

    shown = {by_id[e["target"]]["type"] for e in edges
             if e.get("type", "data") == "data" and e["source"] == "eval"
             and e.get("sourceHandle") == "accuracy"}
    assert "Print" in shown, (
        f"EvaluateModel.accuracy reaches {sorted(shown)} -- the one number "
        f"the evaluation tail exists to produce has to reach a Print")


#: The foundations pack's MNIST trainer, the C2 chapter's flagship. Same
#: contract as the LeNet one above and the built-in quick-start trainer,
#: checked separately because it is a separate file a separate edit can
#: break. It carries one extra pin: this graph pins its training device to
#: ``cpu`` instead of ``auto``, and an evaluation left on another device is
#: the bug the ResNet baseline shipped with -- a graph that trains on one
#: device and scores on another.
_MLP_TRAINER = (_PLUGIN_ROOT / "foundations" / "examples" / "C2-5"
                / "MLP-MNIST-Training" / "graph.json")


def test_the_foundations_pack_trainer_scores_the_split_it_did_not_train_on():
    """Train on ``train``, report ``test``, on the training device."""
    payload = json.loads(_MLP_TRAINER.read_text(encoding="utf-8"))
    edges = payload["edges"]
    by_id = {node["id"]: node for node in graph_nodes(payload)}

    def feeding(target: str, handle: str) -> dict:
        sources = [e["source"] for e in edges
                   if e.get("type", "data") == "data"
                   and e["target"] == target and e.get("targetHandle") == handle]
        assert len(sources) == 1, f"{target}.{handle} is fed by {sources}"
        return by_id[sources[0]]

    evaluator = by_id["eval"]
    pipeline = feeding("eval", "model")
    assert pipeline["type"] == "preset:Training Pipeline", pipeline["type"]
    trained_on = pipeline["data"]["internalParams"]["dataset"]
    assert trained_on["split"] == "train", trained_on

    scored_on = feeding("eval", "dataset")
    assert scored_on["type"] == "Dataset", scored_on["type"]
    assert scored_on["data"]["params"]["name"] == trained_on["name"], (
        f"the example trains on {trained_on['name']} and scores itself on "
        f"{scored_on['data']['params']['name']}")
    assert scored_on["data"]["params"]["split"] == "test", (
        f"{scored_on['id']} reads the {scored_on['data']['params']['split']!r} "
        f"split -- an accuracy measured on the images the loop trained on is "
        f"not an accuracy")

    training_device = pipeline["data"]["internalParams"]["train_loop"]["device"]
    assert evaluator["data"]["params"]["device"] == training_device, (
        f"the loop trains on {training_device!r} and the evaluation runs on "
        f"{evaluator['data']['params']['device']!r}; a graph that pins its "
        f"training device has to pin the same one for the pass that scores it")

    shown = {by_id[e["target"]]["type"] for e in edges
             if e.get("type", "data") == "data" and e["source"] == "eval"
             and e.get("sourceHandle") == "accuracy"}
    assert "Print" in shown, (
        f"EvaluateModel.accuracy reaches {sorted(shown)} -- the one number "
        f"the evaluation tail exists to produce has to reach a Print")


# ── a note that quotes a number still quotes the number the graph gives ───

#: ``(example, note id, quoted text, node id, output port)`` for the pack
#: examples whose notes tell the reader what a run comes out at.
#:
#: A number in a note is the part of it that rots: a seed, a param or a
#: ``values`` list is edited in a second, everything still validates and
#: still runs, and the note goes on quoting what the example used to
#: produce. That is worse than no note, so each quote is pinned to the port
#: it was read off -- to the decimals it is written to, because that is how
#: a reader compares it with the Print.
#:
#: The foundations accuracies are all k/n over a fixed test split, so a
#: prediction has to flip before one of them moves -- which is exactly the
#: change worth failing on. Two numbers those notes also quote are left out
#: deliberately: the sklearn MLP training losses (0.004 and 0.008) are the
#: one place where another machine's BLAS could move a digit without
#: anything being wrong, and their accuracies already guard the same graphs.
#: The deep pack has nothing to pin here -- its notes quote shapes, its one
#: measured accuracy belongs to an MNIST graph this suite does not execute,
#: and C4-1 says in the note itself that its numbers are unseeded.
_QUOTED_FROM_A_PORT = [
    ("foundations/C2-1/Supervised-Learning-101", "note-overview", "1.0",
     "acc", "accuracy"),
    ("foundations/C2-2/Concentric-Circles-Failure", "note-overview", "0.45",
     "acc", "accuracy"),
    ("foundations/C2-3/Decision-Tree-Iris", "note-overview", "0.9667",
     "acc", "accuracy"),
    ("foundations/C2-3/SVM-RBF-Beats-Circles", "note-overview", "1.0",
     "acc", "accuracy"),
    ("foundations/C2-4/MLP-Solves-Circles", "note-overview", "1.0",
     "acc", "accuracy"),
    ("foundations/C2-4/MLP-Without-Activation", "note-overview", "0.4",
     "acc", "accuracy"),
    ("foundations/C2-5/MLP-Inline-Demo", "note-overview", "1.0",
     "acc", "accuracy"),
    ("rl/C5-1/RL-Trajectory-Mockup", "note-overview", "0.35", "return", "tensor"),
    ("rl/C5-1/RL-Trajectory-Mockup", "note-return", "0.35", "return", "tensor"),
    ("rl/C5-3/RLHF-Reward-Model", "note-overview", "0.041", "rm_chosen", "rewards"),
    ("rl/C5-3/RLHF-Reward-Model", "note-overview", "0.177", "rm_reject", "rewards"),
    ("rl/C5-4/GRPO-Group-Advantage", "note-overview", "0.4875", "mean", "tensor"),
    ("rl/RL/Policy-Gradient-101", "note-overview", "-0.128", "pg", "loss"),
    ("rl/RL/Policy-Gradient-101", "note-pg", "-0.1283", "pg", "loss"),
    ("stats/Stats/Confusion-Matrix-Heatmap", "note-cm", "0.9778", "cm", "accuracy"),
]

#: The same idea where the number is a cell of a table rather than a port of
#: its own: the quote has to appear in the text ``Stats-TableView`` renders,
#: which is the surface the note sends the reader to look at.
#:
#: ``Column-Stats-101`` fits neither list. Its note quotes one element of a
#: three-column tensor, which the single-number reader below rejects, and
#: that graph has no table view to read a cell out of; a pin for it would
#: have to index a tensor, which is a third shape for two numbers.
_QUOTED_FROM_A_TABLE = [
    ("stats/Stats/Iris-Describe-Table", "note-overview", "5.8433", "view"),
    ("stats/Stats/Iris-Describe-Table", "note-overview", "3.7580", "view"),
    ("stats/Stats/Iris-GroupBy-Chart", "note-overview", "1.462", "view"),
    ("stats/Stats/Iris-GroupBy-Chart", "note-overview", "4.260", "view"),
    ("stats/Stats/Iris-GroupBy-Chart", "note-overview", "5.552", "view"),
]


def _note_content(payload: dict, note_id: str) -> str:
    note = next((n for n in payload["nodes"]
                 if n.get("id") == note_id and n.get("type") == "note"), None)
    assert note is not None, f"{note_id} is not a note in this example"
    return str(note["data"]["noteContent"])


def _one_number(value) -> float:
    """The single number behind a port, tensor or plain float."""
    flatten = getattr(value, "flatten", None)
    if flatten is None:
        return float(value)
    flat = flatten()
    assert len(flat) == 1, f"expected one number, got {len(flat)}"
    return float(flat[0])


def test_the_pack_notes_quote_what_their_graphs_produce():
    """Run each example and check its notes against the real output.

    Cheap: none of these graphs holds a slow node type, which is why the
    smoke test above executes them all anyway. That one asks whether they
    run; this one asks whether they still say what they do.
    """
    backend_dir = Path(__file__).resolve().parents[1]
    payloads: dict[str, dict] = {}
    results: dict[str, dict] = {}

    prev_cwd = Path.cwd()
    os.chdir(backend_dir)  # CSVReader reads data/samples/iris.csv from here
    try:
        for example in sorted({row[0] for row
                               in _QUOTED_FROM_A_PORT + _QUOTED_FROM_A_TABLE}):
            pack, _, rest = example.partition("/")
            path = _PLUGIN_ROOT / pack / "examples" / rest / "graph.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payloads[example] = payload
            results[example] = asyncio.run(
                execute_graph(payload["nodes"], payload["edges"],
                              error_mode="fail_fast")
            )
    finally:
        os.chdir(prev_cwd)

    wrong: list[str] = []

    def quoted_at_all(example: str, note_id: str, quoted: str) -> None:
        if quoted not in _note_content(payloads[example], note_id):
            wrong.append(f"{example} {note_id}: does not say {quoted} at all "
                         f"-- the pin, not the note, is what went stale")

    for example, note_id, quoted, node_id, port in _QUOTED_FROM_A_PORT:
        produced = _one_number(results[example][node_id][port])
        rendered = f"{produced:.{len(quoted.partition('.')[2])}f}"
        if rendered != quoted:
            wrong.append(f"{example} {note_id}: says {quoted}, but "
                         f"{node_id}.{port} comes out {rendered}")
        quoted_at_all(example, note_id, quoted)

    for example, note_id, quoted, node_id in _QUOTED_FROM_A_TABLE:
        if quoted not in str(results[example][node_id]["text"]):
            wrong.append(f"{example} {note_id}: says {quoted}, which is "
                         f"nowhere in the table {node_id} rendered")
        quoted_at_all(example, note_id, quoted)

    assert not wrong, (
        "a note tells the reader what a run comes out at, and the run no "
        f"longer comes out at that: {wrong}")
