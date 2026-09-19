"""The Concepts section ends on something the reader can check.

The eight graphs in the gallery's ``concepts`` section each exist to show
one idea. Three of them used to stop just short of showing it: the sklearn
KNN example fitted a classifier and printed its raw predictions without
ever scoring them, the tabular pipeline ended on a 120x4 tensor dump, and
the forward-diffusion example blended Gaussian noise into a ``randn``
tensor -- noise into noise, so the picture it drew could not show anything
dissolving.

``test_builtin_examples.py`` already runs every one of these graphs and
asserts it does not raise. That is the other axis, and deliberately not
repeated here: this file pins what each graph now *lands on*, structurally
and, where the overview notes quote a number, by executing it and measuring
the number back. A note that contradicts its graph is worse than no note,
so the claims the notes make are held here rather than trusted.

The note geometry is pinned for the same reason. A note's ``position`` is
what the canvas draws before anything is dragged, and its ``boundOffset``
is what the canvas re-derives that position from on an auto-layout, so a
file whose two disagree renders one way and re-renders another.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
import torch

from app.core.graph_engine import execute_graph, is_note_node

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _REPO_ROOT / "examples"
_BACKEND_DIR = _REPO_ROOT / "backend"

#: Every example the gallery files under ``concepts``, by its API path.
_CONCEPT_EXAMPLES = (
    "Classical/Tabular-Iris-Pipeline",
    "Classical/Iris-Sklearn-KNN",
    "RNN/RNN-OneStep",
    "Transformer/MoE-TopK-Routing",
    "Diffusion/Forward-Process",
    "Diffusion/Toy-Sampling",
    "Diffusion/Mini-UNet-Compact",
    "RL/RLHF-Reward-and-KL",
)


def _graph(key: str) -> dict:
    return json.loads(
        (_EXAMPLES_ROOT / key / "graph.json").read_text(encoding="utf-8"))


def _nodes_by_id(graph: dict) -> dict[str, dict]:
    return {n["id"]: n for n in graph["nodes"]}


def _params(graph: dict, node_id: str) -> dict:
    node = _nodes_by_id(graph)[node_id]
    return node["data"]["params"]


def _has_edge(graph: dict, source: str, out_port: str,
              target: str, in_port: str) -> bool:
    return any(
        e.get("source") == source and e.get("sourceHandle") == out_port
        and e.get("target") == target and e.get("targetHandle") == in_port
        for e in graph["edges"])


def _feeds(graph: dict, target: str, in_port: str) -> tuple[str, str] | None:
    """The ``(node id, output port)`` wired into ``target.in_port``."""
    for edge in graph["edges"]:
        if edge.get("target") == target and edge.get("targetHandle") == in_port:
            return edge.get("source"), edge.get("sourceHandle")
    return None


def _run(key: str) -> dict:
    """Execute a concept graph the way the smoke suite does.

    From ``backend/`` -- ``CSVReader``'s path is relative to the working
    directory of the process running the graph, and the server's is
    ``backend/``.
    """
    graph = _graph(key)
    prev_cwd = Path.cwd()
    os.chdir(_BACKEND_DIR)
    try:
        return asyncio.run(
            execute_graph(graph["nodes"], graph["edges"],
                          error_mode="fail_fast"))
    finally:
        os.chdir(prev_cwd)


# ── Iris + sklearn KNN: the classifier now scores itself ──────────────────

def test_the_knn_example_scores_its_own_predictions():
    """It fitted a classifier and printed a list of 30 species names.

    Whether any of them were right was left to the reader to work out by
    eye, against a ``y_test`` the graph never showed.
    """
    graph = _graph("Classical/Iris-Sklearn-KNN")

    assert _nodes_by_id(graph)["acc"]["type"] == "Accuracy"
    assert _feeds(graph, "acc", "predictions") == ("knn", "predictions")
    assert _feeds(graph, "acc", "labels") == ("split", "y_test")

    printed = [e for e in graph["edges"]
               if e.get("source") == "acc" and e.get("sourceHandle") == "accuracy"]
    assert printed, "the accuracy is computed and never shown"
    assert _nodes_by_id(graph)[printed[0]["target"]]["type"] == "Print"


def test_the_knn_example_lands_on_an_accuracy_worth_quoting():
    """The overview note says "about 97%"; this is where that comes from."""
    results = _run("Classical/Iris-Sklearn-KNN")

    assert results["acc"]["total"] == 30, "30 = the 20% test split of 150 rows"
    assert results["acc"]["accuracy"] >= 0.9, results["acc"]


# ── Tabular Iris: a terminal value a reader can check ─────────────────────

def test_the_tabular_pipeline_prints_a_number_and_not_a_tensor_dump():
    graph = _graph("Classical/Tabular-Iris-Pipeline")

    assert _nodes_by_id(graph)["mean-cols"]["type"] == "Mean"
    # dim 0 reduces over rows, so what comes out is one value per feature.
    assert _params(graph, "mean-cols")["dim"] == "0"
    assert _feeds(graph, "mean-cols", "tensor") == ("split", "x_train")
    assert _feeds(graph, "print", "value") == ("mean-cols", "tensor")


def test_the_tabular_pipeline_means_come_out_near_zero():
    """z-scoring per column puts the whole table's mean at 0.

    The 120-row training split is a subset, so its means are near zero
    rather than at it -- which is exactly the thing the graph is asking the
    reader to check, and the overview note quotes the range.
    """
    results = _run("Classical/Tabular-Iris-Pipeline")

    means = results["mean-cols"]["tensor"]
    assert tuple(means.shape) == (4,), "one mean per Iris feature"
    assert float(means.abs().max()) < 0.2, means.tolist()


# ── Forward diffusion: a real image under the noise ───────────────────────

def test_the_forward_process_noises_a_real_image():
    """``randn`` blended into ``randn`` is noise mixed into noise."""
    graph = _graph("Diffusion/Forward-Process")

    x0 = _nodes_by_id(graph)["x0"]
    assert x0["type"] == "ImageReader"
    # The one image tracked in the repo, so the graph runs on a fresh clone.
    from app.config import settings
    assert (settings.IMAGES_DIR / _params(graph, "x0")["path"]).is_file()

    # The noise follows the image's shape rather than a hard-coded one, so
    # swapping in a different picture needs no second edit.
    assert _feeds(graph, "noise", "shape_ref") == ("x0", "tensor")


def test_the_forward_process_leaves_the_digit_visible():
    """What "dissolving" means, as a number.

    Measured against the file on disk rather than against whatever the
    graph put on that port, so the assertion is "the picture still holds
    that digit" and not the tautology "the blend resembles its own input".
    The blend has to keep enough of the image for the 7 to read through the
    noise: at the shipped alpha the correlation is about 0.62, at the 0.4
    this example used to carry it is about 0.17.
    """
    from PIL import Image
    from torchvision import transforms

    from app.config import settings

    graph = _graph("Diffusion/Forward-Process")
    reader = _params(graph, "x0")
    image = Image.open(settings.IMAGES_DIR / reader["path"])
    image.load()
    clean = transforms.ToTensor()(image.convert(reader["mode"]))

    results = _run("Diffusion/Forward-Process")
    assert torch.allclose(results["x0"]["tensor"], clean), (
        "x_0 is not the image the graph names")

    xt = results["lerp"]["tensor"]
    assert xt.shape == clean.shape
    correlation = float(torch.corrcoef(
        torch.stack([clean.flatten().double(), xt.flatten().double()]))[0, 1])
    assert correlation > 0.5, correlation


# ── RNN unrolled: one set of weights, applied three times ─────────────────

def test_the_unrolled_rnn_applies_one_set_of_weights():
    """The name says "one set of weights", so the params have to say it too.

    Each cell builds its own module from ``seed``, so three cells with the
    same four structural params hold three identical copies -- which is what
    makes the unrolled chain equal to one cell applied three times. Change
    one seed and the graph stops being an RNN, silently.
    """
    graph = _graph("RNN/RNN-OneStep")
    cells = [n for n in graph["nodes"] if n.get("type") == "RNNCell"]

    assert [c["id"] for c in cells] == ["cell1", "cell2", "cell3"]
    shapes = {c["id"]: {k: c["data"]["params"][k] for k in
                        ("input_size", "hidden_size", "nonlinearity", "seed")}
              for c in cells}
    assert len(set(map(str, shapes.values()))) == 1, shapes

    # ...and the chain that makes the three one sequence.
    assert _has_edge(graph, "cell1", "hidden", "cell2", "hidden")
    assert _has_edge(graph, "cell2", "hidden", "cell3", "hidden")


# ── the notes ─────────────────────────────────────────────────────────────

def _notes(graph: dict) -> list[dict]:
    return [n for n in graph["nodes"] if is_note_node(n)]


@pytest.mark.parametrize("key", _CONCEPT_EXAMPLES)
def test_every_concept_example_opens_with_an_overview_note(key: str):
    graph = _graph(key)
    notes = _notes(graph)

    ids = [n["id"] for n in notes]
    assert len(ids) == len(set(ids)), f"{key} repeats a note id: {ids}"

    overview = [n for n in notes if n["id"] == "note-overview"]
    assert overview, f"{key} has no overview note"

    entry = [n for n in graph["nodes"] if n.get("type") == "Start"]
    assert overview[0]["data"]["boundToNodeId"] == entry[0]["id"]


@pytest.mark.parametrize("key", _CONCEPT_EXAMPLES)
def test_every_concept_note_sits_where_its_binding_says(key: str):
    graph = _graph(key)
    by_id = _nodes_by_id(graph)

    for note in _notes(graph):
        data = note["data"]
        bound = data.get("boundToNodeId")
        assert bound in by_id, f"{key}: {note['id']} is bound to {bound!r}"
        assert not is_note_node(by_id[bound]), (
            f"{key}: {note['id']} is bound to another note")

        offset = data["boundOffset"]
        anchor = by_id[bound]["position"]
        assert note["position"] == {"x": anchor["x"] + offset["x"],
                                    "y": anchor["y"] + offset["y"]}, (
            f"{key}: {note['id']} draws somewhere its offset does not put it")
        assert data["noteWidth"] > 0


@pytest.mark.parametrize("key", _CONCEPT_EXAMPLES)
def test_every_concept_note_is_written_in_both_languages(key: str):
    """English paragraph, blank line, the same thing in Chinese.

    There is no translation mechanism for note text -- the note itself is
    the bilingual surface -- so a note with only one half of it is a note
    half the readers cannot use.
    """
    notes = _notes(_graph(key))
    assert notes, f"{key} carries no notes at all"

    for note in notes:
        english, _, chinese = note["data"]["noteContent"].partition("\n\n")
        assert english.strip(), f"{key}: {note['id']} has no English"
        assert chinese.strip(), f"{key}: {note['id']} has no Chinese"
        assert any("一" <= ch <= "鿿" for ch in chinese), (
            f"{key}: {note['id']}'s second half is not Chinese")
        assert not any("一" <= ch <= "鿿" for ch in english), (
            f"{key}: {note['id']} mixes Chinese into its English half")
