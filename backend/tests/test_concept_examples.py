"""The Concepts section ends on something the reader can check.

The graphs in the gallery's ``concepts`` section each exist to show one
idea. Three of them used to stop just short of showing it: the sklearn
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
    "Classical/Classifier-Showdown-Moons",
    "Transformer/Causal-Attention-From-Scratch",
    "RL/Reward-Hacking-Shortcut",
    "RL/GRPO-Step-GridWorld",
    "Classical/KFold-Cross-Validation",
    "Others/Reduce-And-Switch",
    "Others/Tensor-Ops-Tour",
)

#: Every example the gallery files under ``papers``, by its API path. They
#: follow the same note rules as the concept examples; what each one claims
#: is executed in its own test file (``test_paper_examples.py`` for the
#: optimisation ones, ``test_paper_sequence_examples.py`` for the sequence
#: and regression ones, ``test_papers_rl_examples.py`` for the RL ones).
_PAPER_EXAMPLES = (
    "Papers/Dropout-Srivastava-2014",
    "Papers/BatchNorm-Ioffe-2015",
    "Papers/ResNet-Degradation-He-2016",
    "Papers/Adam-Kingma-2015",
    "Papers/Label-Smoothing-Szegedy-2016",
    "Papers/SGDR-Loshchilov-2017",
    "Papers/Vanishing-Gradients-Glorot-2010",
    "Papers/Gradient-Clipping-RNN",
    "Papers/Autoencoder-MNIST-2D",
    "Papers/LSTM-Forecast-Sine",
    "Papers/Transformer-Sort",
    "Papers/Weight-Decay-Regression",
    "Papers/Early-Stopping-Regression",
    "Papers/REINFORCE-CartPole",
    "Papers/DQN-CartPole",
    "Papers/PPO-CartPole",
    "Papers/QLearning-GridWorld",
)

#: The examples the note rules below hold over.
_NOTED_EXAMPLES = _CONCEPT_EXAMPLES + _PAPER_EXAMPLES


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
    assert _feeds(graph, "mean-cols", "tensor") == ("norm-test", "tensor")
    assert _feeds(graph, "print", "value") == ("mean-cols", "tensor")


def test_the_tabular_pipeline_test_means_come_out_near_zero():
    """z-scoring fitted on the training rows puts THEIR means at 0 exactly.

    The 30 test rows are scaled with those statistics, so their means are
    near zero rather than at it -- what the graph asks the reader to check,
    and what the overview note quotes as "the largest about 0.10 away".
    It measures 0.1007; the bounds hold the note's claim, and the lower one
    fails if the test rows were z-scored on their own statistics (#713).
    """
    results = _run("Classical/Tabular-Iris-Pipeline")

    train_means = results["norm"]["tensor"].mean(dim=0)
    assert float(train_means.abs().max()) < 1e-5, train_means.tolist()

    means = results["mean-cols"]["tensor"]
    assert tuple(means.shape) == (4,), "one mean per Iris feature"
    assert 0.08 < float(means.abs().max()) < 0.12, means.tolist()


# ── #713: every Iris example fits Normalize on the training split only ────

_PLUGIN_EXAMPLES_ROOT = _REPO_ROOT / "plugins" / "foundations" / "examples"

_SPLIT_THEN_NORMALIZE = (
    (_EXAMPLES_ROOT, "Classical/Tabular-Iris-Pipeline"),
    (_EXAMPLES_ROOT, "Classical/Iris-Sklearn-KNN"),
    (_PLUGIN_EXAMPLES_ROOT, "Classical/KNN-from-Scratch"),
)


@pytest.mark.parametrize("root,key", _SPLIT_THEN_NORMALIZE,
                         ids=[k for _, k in _SPLIT_THEN_NORMALIZE])
def test_normalize_is_fitted_on_the_training_split_and_reused_on_test(root, key):
    graph = json.loads((root / key / "graph.json").read_text(encoding="utf-8"))
    nodes = _nodes_by_id(graph)

    assert nodes["norm"]["type"] == nodes["norm-test"]["type"] == "Normalize"
    assert _feeds(graph, "split", "features") == ("select", "tensor")
    assert _feeds(graph, "norm", "tensor") == ("split", "x_train")
    assert _feeds(graph, "norm-test", "tensor") == ("split", "x_test")
    assert _feeds(graph, "norm-test", "stats") == ("norm", "stats")
    # Nothing between the reader and the split normalises the whole table.
    assert not any(n["type"] == "Normalize" for n in graph["nodes"]
                   if n["id"] not in ("norm", "norm-test"))
    if "knn" in nodes:
        assert _feeds(graph, "knn", "x_train") == ("norm", "tensor")
        assert _feeds(graph, "knn", "x_query") == ("norm-test", "tensor")


def test_knn_from_scratch_gets_29_of_30_as_its_note_says():
    graph = json.loads((_PLUGIN_EXAMPLES_ROOT / "Classical/KNN-from-Scratch"
                        / "graph.json").read_text(encoding="utf-8"))
    prev_cwd = Path.cwd()
    os.chdir(_BACKEND_DIR)
    try:
        results = asyncio.run(execute_graph(
            graph["nodes"], graph["edges"], error_mode="fail_fast"))
    finally:
        os.chdir(prev_cwd)

    preds = results["knn"]["predictions"]
    truth = results["split"]["y_test"]
    assert sum(p == t for p, t in zip(preds, truth)) == 29
    assert len(truth) == 30


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
    """The description says "one set of weights", so the params say it too.

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


def _run_with_its_seed(key: str) -> dict:
    """Execute a concept graph under the run seed its file stores.

    The canvas runs an opened example with ``settings.seed``, so a number a
    note quotes off a seeded example is the number this run gives.
    """
    from app.core.execution_context import ExecutionContext

    graph = _graph(key)
    seed = graph["settings"]["seed"]
    prev_cwd = Path.cwd()
    os.chdir(_BACKEND_DIR)
    try:
        return asyncio.run(execute_graph(
            graph["nodes"], graph["edges"], error_mode="fail_fast",
            context=ExecutionContext(device="cpu", seed=seed)))
    finally:
        os.chdir(prev_cwd)


# ── Six classifiers on two moons: the numbers the overview quotes ─────────

def test_the_showdown_scores_every_classifier_on_the_same_split():
    graph = _graph("Classical/Classifier-Showdown-Moons")
    classifiers = ("logreg", "knn", "tree", "forest", "svm", "mlp")
    for clf in classifiers:
        assert _feeds(graph, clf, "x_train") == ("split", "x_train")
        assert _feeds(graph, clf, "x_query") == ("split", "x_test")
        assert _feeds(graph, f"acc-{clf}", "predictions") == (clf, "predictions")
        assert _feeds(graph, f"acc-{clf}", "labels") == ("split", "y_test")


def test_the_showdown_lands_where_its_note_says():
    """0.82 for the straight line, 0.83 for the depth-5 tree, 0.86 to 0.90
    for the four that bend."""
    results = _run_with_its_seed("Classical/Classifier-Showdown-Moons")
    acc = {clf: results[f"acc-{clf}"]["accuracy"]
           for clf in ("logreg", "knn", "tree", "forest", "svm", "mlp")}
    assert results["acc-logreg"]["total"] == 120, "the 30% split of 400 points"
    assert f"{acc['logreg']:.2f}" == "0.82", acc
    assert f"{acc['tree']:.2f}" == "0.83", acc
    for clf in ("knn", "forest", "svm", "mlp"):
        assert 0.86 <= round(acc[clf], 2) <= 0.90, acc
        assert acc[clf] > acc["logreg"], acc


# ── Causal attention from scratch: the mask shows in the weights ──────────

def test_causal_attention_weights_are_lower_triangular_rows_of_one():
    results = _run_with_its_seed("Transformer/Causal-Attention-From-Scratch")
    weights = results["softmax"]["tensor"]
    assert tuple(weights.shape) == (6, 6)
    assert torch.equal(torch.triu(weights, diagonal=1), torch.zeros(6, 6)), (
        "a position reads a later one")
    assert torch.allclose(weights.sum(-1), torch.ones(6))
    assert float(weights[0, 0]) == pytest.approx(1.0), (
        "position 0 can read only itself")
    assert tuple(results["out"]["tensor"].shape) == (6, 8)


def test_causal_attention_scales_by_one_over_root_d():
    graph = _graph("Transformer/Causal-Attention-From-Scratch")
    d = int(_params(graph, "k")["shape"].split(",")[1])
    assert _params(graph, "scale")["scalar"] == pytest.approx(d ** -0.5, abs=1e-6)


# ── Reward hacking: the gap the overview quotes ───────────────────────────

def test_reward_hacking_hides_from_the_training_score():
    """Both arms fit training to 1.0; holdout ends near 0.78 with the
    shortcut and about 0.96 without it."""
    results = _run_with_its_seed("RL/Reward-Hacking-Shortcut")
    hack, ctrl = results["rm-hack"], results["rm-ctrl"]
    assert float(hack["train_accuracy"][-1]) == pytest.approx(1.0)
    assert float(ctrl["train_accuracy"][-1]) == pytest.approx(1.0)
    assert float(hack["holdout_accuracy"][-1]) == pytest.approx(0.78, abs=0.01)
    assert float(ctrl["holdout_accuracy"][-1]) == pytest.approx(0.96, abs=0.01)


# ── One GRPO step: 3 goals and 5 traps under the stored seed ──────────────

def test_the_grpo_rollouts_are_the_ones_the_note_counts():
    results = _run_with_its_seed("RL/GRPO-Step-GridWorld")
    returns = results["rollout"]["returns"].tolist()
    assert len(returns) == 8
    assert returns.count(1.0) == 3, returns
    assert returns.count(-1.0) == 5, returns

    advantages = results["adv"]["advantages"]
    assert float(advantages.sum()) == pytest.approx(0.0, abs=1e-5)
    for ret, adv in zip(returns, advantages.tolist()):
        assert (adv > 0) == (ret > sum(returns) / len(returns)), (returns, adv)


def test_the_grpo_step_feeds_one_advantage_per_step_into_the_clip():
    results = _run_with_its_seed("RL/GRPO-Step-GridWorld")
    steps = int(results["rollout"]["episode_lengths"].sum())
    assert tuple(results["adv"]["advantages_expanded"].shape) == (steps,)
    # old == new log-probs: every ratio is 1, nothing is clipped.
    assert torch.allclose(results["ppo"]["ratio"], torch.ones(steps))
    assert results["ppo"]["clip_fraction"] == 0


# ── the notes ─────────────────────────────────────────────────────────────

def _notes(graph: dict) -> list[dict]:
    return [n for n in graph["nodes"] if is_note_node(n)]


@pytest.mark.parametrize("key", _NOTED_EXAMPLES)
def test_every_concept_example_opens_with_an_overview_note(key: str):
    graph = _graph(key)
    notes = _notes(graph)

    ids = [n["id"] for n in notes]
    assert len(ids) == len(set(ids)), f"{key} repeats a note id: {ids}"

    overview = [n for n in notes if n["id"] == "note-overview"]
    assert overview, f"{key} has no overview note"

    entry = [n for n in graph["nodes"] if n.get("type") == "Start"]
    assert overview[0]["data"]["boundToNodeId"] == entry[0]["id"]


@pytest.mark.parametrize("key", _NOTED_EXAMPLES)
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


@pytest.mark.parametrize("key", _NOTED_EXAMPLES)
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


# ── Offline training examples: what their notes can check without training ─

def test_the_segmentation_note_quotes_the_all_background_baseline():
    """The note says a model answering background everywhere scores 0.89;
    that is the background share of the held-out masks."""
    from app.nodes.data.synthetic_segmentation_node import SyntheticSegmentationNode

    graph = _graph("Usage_Example/Segmentation-Synthetic-Shapes")
    params = {k: v for k, v in _params(graph, "seg-test").items()}
    dataset = SyntheticSegmentationNode().execute({}, params)["dataset"]
    masks = torch.stack([dataset[i][1] for i in range(len(dataset))])
    assert f"{float((masks == 0).float().mean()):.2f}" == "0.89"
    note = next(n for n in graph["nodes"] if n["id"] == "note-overview")
    assert "0.89" in note["data"]["noteContent"]


def test_the_lstm_example_tests_on_sequences_it_never_trained_on():
    graph = _graph("RNN/LSTM-Recall-First-Token")
    train, test = _params(graph, "seq-train"), _params(graph, "seq-test")
    assert train["seed"] != test["seed"]
    for key in ("kind", "seq_len", "n_classes", "n_distractors"):
        assert train[key] == test[key], key
    assert _feeds(graph, "eval", "dataset") == ("seq-test", "dataset")


# ── K-fold cross-validation: the three errors and their mean ──────────────

def test_the_three_folds_differ_only_in_which_fold_they_hold_out():
    graph = _graph("Classical/KFold-Cross-Validation")
    params = [_params(graph, f"fold-{k}") for k in range(3)]
    assert [p["fold"] for p in params] == [0, 1, 2]
    for p in params:
        assert {k: v for k, v in p.items() if k != "fold"} == {
            k: v for k, v in params[0].items() if k != "fold"}
        assert p["n_folds"] == 3
    for k in range(3):
        assert _feeds(graph, f"reg-{k}", "x_query") == (f"fold-{k}", "x_val")
        assert _feeds(graph, f"metrics-{k}", "targets") == (f"fold-{k}", "y_val")


def test_the_kfold_errors_are_the_ones_the_note_quotes():
    """0.050, 0.030 and 0.033 per fold, 0.038 for their mean, and the
    worst fold about 1.7 times the best."""
    results = _run_with_its_seed("Classical/KFold-Cross-Validation")
    mse = [results[f"metrics-{k}"]["mse"] for k in range(3)]
    assert [f"{m:.3f}" for m in mse] == ["0.050", "0.030", "0.033"], mse
    assert f"{max(mse) / min(mse):.1f}" == "1.7", mse
    mean = float(results["print-mean"]["value"])
    assert mean == pytest.approx(sum(mse) / 3)
    assert f"{mean:.3f}" == "0.038"
    # Every row held out exactly once, in folds of 50.
    held_out = [len(results[f"fold-{k}"]["y_val"]) for k in range(3)]
    assert held_out == [50, 50, 50]


# ── Reduce and Switch: what each selector reports ─────────────────────────

def _run_reduce_switch(selector: int) -> dict:
    from app.core.execution_context import ExecutionContext

    graph = _graph("Others/Reduce-And-Switch")
    _nodes_by_id(graph)["switch"]["data"]["params"]["selector"] = selector
    prev_cwd = Path.cwd()
    os.chdir(_BACKEND_DIR)
    try:
        return asyncio.run(execute_graph(
            graph["nodes"], graph["edges"], error_mode="fail_fast",
            context=ExecutionContext(device="cpu", seed=graph["settings"]["seed"])))
    finally:
        os.chdir(prev_cwd)


@pytest.mark.parametrize("selector,expected,skipped", [
    (0, ("3.758", "1.0", "6.9"), "csv-sepal"),
    (1, ("5.843", "4.3", "7.9"), "csv-petal"),
])
def test_the_switch_picks_the_column_the_note_says(selector, expected, skipped):
    graph = _graph("Others/Reduce-And-Switch")
    assert _params(graph, "switch")["selector"] == 0, "the note starts on petal length"
    results = _run_reduce_switch(selector)
    got = (f"{float(results['mean']['result']):.3f}",
           f"{float(results['min']['result']):.1f}",
           f"{float(results['max']['result']):.1f}")
    assert got == expected
    assert tuple(results["stack"]["result"].shape) == (150,)
    # The unselected reader is left out of the run, as note-switch says.
    assert skipped not in results


# ── Tensor ops tour: the shapes and statistics the notes quote ────────────

def test_the_tensor_tour_lands_on_the_numbers_its_notes_quote():
    results = _run_with_its_seed("Others/Tensor-Ops-Tour")
    grid = results["stats-grid"]
    assert grid["shape"] == "(4, 6)"
    assert grid["mean"] == pytest.approx(11.5)
    assert f"{grid['std']:.4f}" == "6.9222"

    stacked = results["stats-stack"]
    assert stacked["shape"] == "(2, 2, 6)"
    assert stacked["std"] == pytest.approx(grid["std"])

    rows = results["rows"]["tensor"]
    assert rows[:, 0].tolist() == [18.0, 0.0], "rows 3 and 0, in that order"

    pooled = results["pool"]["tensor"]
    assert pooled.tolist() == [[[[3.5, 5.5, 7.5], [15.5, 17.5, 19.5]]]]
    assert results["stats-pool"]["mean"] == pytest.approx(11.5)
    assert f"{results['stats-pool']['std']:.4f}" == "6.2183"


# ── Training section: augmentation and resuming ───────────────────────────

def _run_training_example(key: str) -> dict:
    """``_run_with_its_seed`` for a graph outside ``_CONCEPT_EXAMPLES``."""
    return _run_with_its_seed(key)


def test_augmentation_goes_on_the_training_split_only():
    from app.nodes.data.transforms._base import pipeline_is_random

    graph = _graph("Usage_Example/Augment-Then-Train")
    assert _feeds(graph, "ds-train", "train_transform") == ("normalize", "transform")
    assert _feeds(graph, "ds-test", "eval_transform") == ("eval-normalize", "transform")
    assert _params(graph, "ds-test")["split"] == "test"
    assert _feeds(graph, "eval", "dataset") == ("subset-test", "train")

    results = _run_training_example("Usage_Example/Augment-Then-Train")
    assert pipeline_is_random(results["normalize"]["transform"])
    assert not pipeline_is_random(results["eval-normalize"]["transform"])
    # The overview note says "about 0.9"; 0.895 on this Mac.
    assert results["eval"]["total"] == 1000
    assert results["eval"]["accuracy"] >= 0.85, results["eval"]


def test_the_resumed_loop_picks_up_at_the_epoch_the_checkpoint_stores(
        tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "models")
    graph = _graph("Usage_Example/Resume-From-Checkpoint")
    assert _feeds(graph, "save", "epoch") == ("train-1", "epoch")
    assert _feeds(graph, "save", "optimizer") == ("train-1", "optimizer")
    assert _feeds(graph, "load", "model") == ("save", "model")
    assert _feeds(graph, "train-2", "start_epoch") == ("load", "epoch")
    assert _params(graph, "save")["path"] == _params(graph, "load")["path"]

    results = _run_training_example("Usage_Example/Resume-From-Checkpoint")
    assert (tmp_path / "models" / _params(graph, "save")["path"]).is_file()
    assert int(results["load"]["epoch"]) == 10
    assert int(results["train-2"]["epoch"]) == 20
    assert tuple(results["train-2"]["losses"].shape) == (10,)
    assert tuple(results["curve"]["tensor"].shape) == (20,)
    assert torch.equal(results["load"]["losses"], results["train-1"]["losses"])
