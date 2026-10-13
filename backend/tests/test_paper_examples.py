"""The paper reproductions: each graph runs, and lands where its notes say.

Every graph under ``examples/Papers/`` reproduces one paper's central claim
on a MNIST subset, and its notes quote the numbers a CPU run reaches under
the seed stored in the file (``settings.seed``). Each test below executes
one graph the way the canvas runs it -- ``ExecutionContext(device="cpu",
seed=<settings.seed>)`` -- checks that every number it pins is written in
the graph's notes, and asserts the paper's claim on the result.

Tolerances. A seeded CPU run is reproducible on one machine, but CI runs
other CPUs, another BLAS and a newer torch, and a few hundred SGD steps
amplify the last-bit differences. So each quoted number is checked against
a band about as wide as the spread the same graph showed over five other
run seeds on the authoring machine (narrow for the stable quantities, wide
for the plain nets that sit on a plateau), and every claim is asserted as
an inequality with a margin that held under all of those seeds.

The graphs all read MNIST, which the repository ships under
``backend/data``, so nothing here downloads. The seven together run in
about 15 seconds on an M3 laptop.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
from pathlib import Path

import pytest

from app.core.execution_context import ExecutionContext
from app.core.graph_engine import execute_graph, is_note_node

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PAPERS = _REPO_ROOT / "examples" / "Papers"
_BACKEND_DIR = _REPO_ROOT / "backend"


class _RecordingContext(ExecutionContext):
    """The run context, keeping every metric point instead of queueing it.

    ``TrainingLoop`` sends ``grad_norm`` to the run's metric store, which a
    bare ``execute_graph`` call has no consumer for. Each node gets a shallow
    copy of the context, so ``recorded`` is one list shared by all of them.
    """

    def log_metric(self, name, value, step, node_id=None):
        self.recorded.append((node_id or self.current_node_id, name, float(value), int(step)))


def _graph(name: str) -> dict:
    return json.loads((_PAPERS / name / "graph.json").read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=None)
def _run(name: str) -> tuple[dict, list]:
    """Execute one paper graph under its stored seed, once per session."""
    graph = _graph(name)
    context = _RecordingContext(device="cpu", seed=graph["settings"]["seed"])
    context.recorded = []
    prev_cwd = Path.cwd()
    os.chdir(_BACKEND_DIR)  # Dataset's data_dir is relative to backend/
    try:
        results = asyncio.run(execute_graph(
            graph["nodes"], graph["edges"], error_mode="fail_fast", context=context))
    finally:
        os.chdir(prev_cwd)
    return results, context.recorded


def _notes_text(name: str) -> str:
    return "\n".join(n["data"]["noteContent"] for n in _graph(name)["nodes"] if is_note_node(n))


def _quoted(name: str, *numbers: str) -> None:
    """Every number a test pins is a number the notes print."""
    text = _notes_text(name)
    missing = [n for n in numbers if n not in text]
    assert not missing, f"{name}: the notes no longer quote {missing}"


def _acc(results: dict, node: str) -> float:
    return float(results[node]["accuracy"])


def _losses(results: dict, node: str) -> list[float]:
    return [float(x) for x in results[node]["losses"].tolist()]


def test_every_paper_graph_is_tested_here():
    """A new reproduction gets a test of its claim, not only its note rules."""
    tested = {
        "Dropout-Srivastava-2014", "BatchNorm-Ioffe-2015", "ResNet-Degradation-He-2016",
        "Adam-Kingma-2015", "Label-Smoothing-Szegedy-2016", "SGDR-Loshchilov-2017",
        "Vanishing-Gradients-Glorot-2010",
    }
    assert {p.parent.name for p in _PAPERS.glob("*/graph.json")} == tested


def test_every_arm_trains_its_own_model():
    """One SequentialModel feeding two TrainingLoops shares one set of weights.

    The second loop would then continue from the first, and the comparison
    would be between a model and a longer-trained copy of itself.
    """
    for path in sorted(_PAPERS.glob("*/graph.json")):
        graph = json.loads(path.read_text(encoding="utf-8"))
        trainers_per_model: dict[str, int] = {}
        for edge in graph["edges"]:
            if edge.get("targetHandle") == "model" and edge["target"].startswith("train-"):
                trainers_per_model[edge["source"]] = trainers_per_model.get(edge["source"], 0) + 1
        assert trainers_per_model and set(trainers_per_model.values()) == {1}, path


# ── Dropout (Srivastava et al., JMLR 2014) ─────────────────────────────────

def test_dropout_narrows_the_gap_between_training_and_test_accuracy():
    name = "Dropout-Srivastava-2014"
    _quoted(name, "training accuracy of 1.00", "plain net scores 0.91", "dropout net 0.93",
            "shrinks from 0.09 to 0.07", "down to 0.005", "ends at 0.06")
    r, _ = _run(name)
    train_plain, test_plain = _acc(r, "eval-train-plain"), _acc(r, "eval-test-plain")
    train_drop, test_drop = _acc(r, "eval-train-dropout"), _acc(r, "eval-test-dropout")

    assert train_plain >= 0.99 and train_drop >= 0.99, (train_plain, train_drop)
    assert test_plain == pytest.approx(0.91, abs=0.02)
    assert test_drop == pytest.approx(0.93, abs=0.02)
    assert test_drop >= test_plain + 0.005, (test_drop, test_plain)
    assert train_drop - test_drop < train_plain - test_plain
    assert _losses(r, "train-plain")[-1] < 0.01
    assert 0.03 < _losses(r, "train-dropout")[-1] < 0.10


# ── Batch Normalization (Ioffe & Szegedy, ICML 2015) ───────────────────────

def test_batchnorm_trains_at_a_rate_the_plain_net_cannot_use():
    name = "BatchNorm-Ioffe-2015"
    _quoted(name, "falls from 0.43 to 0.11", "scores 0.90", "still at 2.11", "scores 0.18")
    r, _ = _run(name)
    bn, plain = _losses(r, "train-bn"), _losses(r, "train-plain")

    assert bn[0] == pytest.approx(0.43, abs=0.06)
    assert 0.08 < bn[-1] < 0.15, bn
    assert _acc(r, "eval-test-bn") == pytest.approx(0.90, abs=0.05)
    assert plain[-1] == pytest.approx(2.11, abs=0.2)
    assert _acc(r, "eval-test-plain") < 0.35
    assert plain[-1] > 10 * bn[-1]


# ── Deep Residual Learning (He et al., CVPR 2016) ──────────────────────────

def test_the_plain_deep_net_fits_its_training_set_worse_than_the_residual_one():
    name = "ResNet-Degradation-He-2016"
    _quoted(name, "training accuracy of 0.95 (test 0.91)", "reaches 0.77 (test 0.76)",
            "ends at 0.79", "at 0.27")
    r, _ = _run(name)
    res_train, plain_train = _acc(r, "eval-train-residual"), _acc(r, "eval-train-plain")

    assert res_train == pytest.approx(0.95, abs=0.03)
    assert _acc(r, "eval-test-residual") == pytest.approx(0.91, abs=0.03)
    assert 0.55 < plain_train < 0.85, plain_train
    assert 0.55 < _acc(r, "eval-test-plain") < 0.85
    assert res_train >= plain_train + 0.1
    assert 0.6 < _losses(r, "train-plain")[-1] < 1.1
    assert 0.2 < _losses(r, "train-residual")[-1] < 0.4


def test_the_residual_model_uses_the_layer_editor_add_merge():
    graph = _graph("ResNet-Degradation-He-2016")
    by_id = {n["id"]: n for n in graph["nodes"]}
    plain = json.loads(by_id["model-plain"]["data"]["params"]["layers"])
    residual = json.loads(by_id["model-residual"]["data"]["params"]["layers"])
    assert residual["version"] == 2
    adds = [n for n in residual["nodes"] if n["type"] == "Add"]
    assert len(adds) == 19
    for add in adds:
        assert sum(e["target"] == add["id"] for e in residual["edges"]) == 2
    linears = [n for n in plain["nodes"] if n["type"] == "Linear"]
    assert len(linears) == 21  # 20 hidden layers and the classifier
    assert not [n for n in plain["nodes"] if n["type"] == "Add"]


# ── Adam (Kingma & Ba, ICLR 2015) ──────────────────────────────────────────

def test_adam_reaches_the_lowest_training_loss():
    name = "Adam-Kingma-2015"
    _quoted(name, "0.79 for SGD", "0.27 for SGD with momentum", "0.21 for Adam",
            "average loss is 0.80 against 2.20")
    r, _ = _run(name)
    sgd, mom, adam = (_losses(r, f"train-{k}") for k in ("sgd", "momentum", "adam"))

    assert 0.70 < sgd[-1] < 0.90, sgd
    assert 0.24 < mom[-1] < 0.31, mom
    assert 0.18 < adam[-1] < 0.24, adam
    assert adam[-1] < mom[-1] - 0.03 < sgd[-1]
    assert adam[0] == pytest.approx(0.80, abs=0.08)
    assert sgd[0] == pytest.approx(2.20, abs=0.08)


# ── Label smoothing (Szegedy et al., CVPR 2016; Müller et al., NeurIPS 2019) ─

def test_label_smoothing_keeps_accuracy_and_lowers_confidence():
    name = "Label-Smoothing-Szegedy-2016"
    _quoted(name, "0.92 with hard labels", "0.94 with smoothing", "from 0.96 to 0.80",
            "levels off at 0.52", "loss to 0.01")
    r, _ = _run(name)
    acc_hard, acc_smooth = _acc(r, "eval-test-hard"), _acc(r, "eval-test-smooth")
    conf_hard = float(r["confidence-hard"]["out1"])
    conf_smooth = float(r["confidence-smooth"]["out1"])

    assert acc_hard == pytest.approx(0.92, abs=0.02)
    assert acc_smooth == pytest.approx(0.94, abs=0.02)
    assert acc_smooth >= acc_hard
    assert conf_hard == pytest.approx(0.96, abs=0.02)
    assert conf_smooth == pytest.approx(0.80, abs=0.03)
    assert conf_hard > acc_hard, "hard labels make the model over-confident"
    assert conf_hard - conf_smooth > 0.1
    # The floor the note derives: the entropy of (0.91, nine 0.01s) is 0.50.
    assert 0.50 < _losses(r, "train-smooth")[-1] == pytest.approx(0.52, abs=0.02)
    assert _losses(r, "train-hard")[-1] < 0.03


# ── SGDR (Loshchilov & Hutter, ICLR 2017) ──────────────────────────────────

def test_warm_restarts_show_in_the_learning_rate_and_end_lower():
    name = "SGDR-Loshchilov-2017"
    _quoted(name, "0.015 with restarts", "0.13 at a constant rate", "0.95 and 0.94",
            "0.05, 0.025, back to 0.05", "down to 0.0019", "rises from 0.10 to 0.18")
    r, _ = _run(name)
    lr = [float(x) for x in r["lr-history"]["out1"]]
    sgdr, constant = _losses(r, "train-sgdr"), _losses(r, "train-constant")

    # The schedule is arithmetic, so it is exact on every machine.
    assert lr[:3] == pytest.approx([0.05, 0.025, 0.05])
    assert lr[6] == pytest.approx(0.05) and lr[-1] == pytest.approx(0.0019, abs=1e-4)
    assert all(lr[i + 1] < lr[i] for i in (2, 3, 4) + tuple(range(6, 13)))
    assert r["train-constant"]["metrics"]["lr_history"] == pytest.approx([0.05] * 14)

    assert sgdr[-1] < 0.03 and 0.05 < constant[-1] < 0.2, (sgdr[-1], constant[-1])
    assert sgdr[-1] < constant[-1] / 2
    assert sgdr[6] > sgdr[5] + 0.03, "the restart at epoch 7 shows in the loss"
    assert sgdr[5] == pytest.approx(0.10, abs=0.05) and sgdr[6] == pytest.approx(0.18, abs=0.1)
    assert _acc(r, "eval-test-sgdr") == pytest.approx(0.95, abs=0.02)
    assert _acc(r, "eval-test-constant") == pytest.approx(0.94, abs=0.03)


# ── Vanishing gradients (Glorot & Bengio, AISTATS 2010) ────────────────────

def test_the_deep_sigmoid_net_stalls_with_a_small_gradient():
    name = "Vanishing-Gradients-Glorot-2010"
    _quoted(name, "from 2.35 to 2.30", "test accuracy 0.11", "averages 0.21",
            "from 1.66 to 0.20", "test accuracy 0.91", "gradient norm of 1.30")
    r, recorded = _run(name)

    def late_grad_norm(node):
        series = [v for (n, metric, v, _step) in recorded if n == node and metric == "grad_norm"]
        assert len(series) > 50, f"{node} recorded {len(series)} grad_norm points"
        half = series[len(series) // 2:]
        return sum(half) / len(half)

    sig, relu = _losses(r, "train-sigmoid"), _losses(r, "train-relu")
    gn_sig, gn_relu = late_grad_norm("train-sigmoid"), late_grad_norm("train-relu")

    assert sig[0] == pytest.approx(2.35, abs=0.03)
    assert sig[-1] == pytest.approx(2.30, abs=0.01)
    assert _acc(r, "eval-test-sigmoid") < 0.2
    assert gn_sig == pytest.approx(0.21, abs=0.05)
    assert relu[0] == pytest.approx(1.66, abs=0.3)
    assert 0.12 < relu[-1] < 0.35, relu
    assert _acc(r, "eval-test-relu") == pytest.approx(0.91, abs=0.04)
    assert 0.9 < gn_relu < 1.9, gn_relu
    assert gn_relu > 3 * gn_sig
