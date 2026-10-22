"""The paper reproductions: each graph runs, and lands where its notes say.

Every graph under ``examples/Papers/`` reproduces one paper's central claim
on a MNIST subset, and its notes quote the numbers a CPU run reaches under
the seed stored in the file (``settings.seed``). Each test below executes
one graph the way the canvas runs it -- ``ExecutionContext(device="cpu",
seed=<settings.seed>)`` -- checks that every number it pins is written in
the graph's notes, and asserts the paper's claim on the result.

Tolerances. A seeded CPU run is reproducible on one machine, but CI runs
other CPUs, another BLAS and a newer torch, and a few hundred SGD steps
amplify the last-bit differences. A number from a stable regime (a net
that trains smoothly, or the schedule) is checked against a band about as
wide as the spread the graph showed over five other run seeds. A number
from a regime a different CPU can move further than that (a 20-layer
narrow net, the no-BatchNorm net at lr 0.5, the stalled sigmoid net) is
checked only as the claim: an inequality with a margin that held under
all of those seeds and on CI. The 20-layer residual net is the measured
case: 0.95 training accuracy on an Apple M3, 0.92 on the Linux runner.

The graphs all read MNIST, which the repository ships under
``backend/data``, so nothing here downloads. The eight together run in
about 30 seconds on an M3 laptop; GroupNorm's two batch-2 arms take half.
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


#: The reproductions whose claims this file executes. The RL ones live in
#: ``test_papers_rl_examples.py`` and the sequence ones in
#: ``test_paper_sequence_examples.py``; ``test_concept_examples._PAPER_EXAMPLES``
#: is the registry every file under ``examples/Papers`` must appear in.
_TESTED_HERE = (
    "Dropout-Srivastava-2014", "BatchNorm-Ioffe-2015", "ResNet-Degradation-He-2016",
    "Adam-Kingma-2015", "Label-Smoothing-Szegedy-2016", "SGDR-Loshchilov-2017",
    "Vanishing-Gradients-Glorot-2010", "GroupNorm-Wu-2018",
)


def test_every_paper_graph_is_registered_and_the_ones_here_exist():
    """A new reproduction joins the registry, so its note rules run; the
    eight this file executes are on disk under the names used below."""
    from tests.test_concept_examples import _PAPER_EXAMPLES

    registered = {key.split("/", 1)[1] for key in _PAPER_EXAMPLES}
    on_disk = {p.parent.name for p in _PAPERS.glob("*/graph.json")}
    assert on_disk == registered, (
        f"unregistered: {sorted(on_disk - registered)}; "
        f"registered but missing: {sorted(registered - on_disk)}")
    assert set(_TESTED_HERE) <= on_disk


def test_every_arm_trains_its_own_model():
    """One SequentialModel feeding two TrainingLoops shares one set of weights.

    The second loop would then continue from the first, and the comparison
    would be between a model and a longer-trained copy of itself.
    """
    for name in _TESTED_HERE:
        path = _PAPERS / name / "graph.json"
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

    assert bn[0] < 0.6 and bn[-1] < 0.2, bn
    assert _acc(r, "eval-test-bn") >= 0.85
    # lr 0.5 without BatchNorm is chaotic, so only the claim is pinned:
    # the plain net is still far from trained after the same 3 epochs.
    assert plain[-1] > 1.5, plain
    assert _acc(r, "eval-test-plain") < 0.5
    assert plain[-1] > 10 * bn[-1]


# ── Deep Residual Learning (He et al., CVPR 2016) ──────────────────────────

def test_the_plain_deep_net_fits_its_training_set_worse_than_the_residual_one():
    name = "ResNet-Degradation-He-2016"
    _quoted(name, "training accuracy of 0.95 on an Apple M3", "0.92 on the Linux CI runner",
            "at 0.77 on the M3 (test 0.76)", "ends at 0.79", "at 0.27")
    r, _ = _run(name)
    res_train, plain_train = _acc(r, "eval-train-residual"), _acc(r, "eval-train-plain")
    res_test, plain_test = _acc(r, "eval-test-residual"), _acc(r, "eval-test-plain")
    res_loss, plain_loss = _losses(r, "train-residual")[-1], _losses(r, "train-plain")[-1]
    measured = (res_train, plain_train, res_test, plain_test, res_loss, plain_loss)

    # A 20-layer net of width 32 moves with the CPU's rounding (0.95 on the
    # M3, 0.92 on the Linux runner), so the claim is pinned, not the value.
    assert res_train >= 0.88, measured
    assert res_train >= plain_train + 0.10, measured
    assert plain_train < 0.85, measured
    assert res_test >= 0.85 and res_test >= plain_test + 0.08, measured
    assert res_loss < 0.45 and plain_loss > 1.8 * res_loss, measured


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

    measured = (sig[0], sig[-1], gn_sig, relu[0], relu[-1], gn_relu)
    # The stalled sigmoid net is pinned as the claim: its loss stays on the
    # plateau of a uniform guess (ln 10 = 2.30) with a small gradient.
    assert sig[-1] > 2.25 and sig[0] - sig[-1] < 0.15, measured
    assert _acc(r, "eval-test-sigmoid") < 0.2
    assert gn_sig < 0.35, measured
    assert relu[-1] < 0.4 and relu[0] - relu[-1] > 1.0, measured
    assert _acc(r, "eval-test-relu") >= 0.85
    assert gn_relu > 3 * gn_sig, measured


# ── Group Normalization (Wu & He, ECCV 2018) ───────────────────────────────

def test_groupnorm_trains_at_batch_2_where_batchnorm_does_not():
    name = "GroupNorm-Wu-2018"
    _quoted(name, "scores 0.96 on the test set", "training loss of 1.70",
            "and scores 0.70", "ends at 0.18 and scores 0.95")
    r, _ = _run(name)
    bn, gn, bn32 = (_acc(r, f"eval-test-{k}") for k in ("bn", "gn", "bn32"))
    bn_loss, gn_loss = _losses(r, "train-bn")[-1], _losses(r, "train-gn")[-1]
    measured = (bn, gn, bn32, bn_loss, gn_loss)

    assert bn32 == pytest.approx(0.96, abs=0.02), measured
    assert gn == pytest.approx(0.95, abs=0.03), measured
    assert gn_loss == pytest.approx(0.18, abs=0.06), measured
    # BatchNorm1d over two samples maps every feature to +1 or -1, so the
    # batch-2 BatchNorm net is pinned as the claim: well below both others.
    assert bn == pytest.approx(0.70, abs=0.08), measured
    assert bn_loss > 1.4, measured
    assert gn > bn + 0.15 and bn32 > bn + 0.15, measured


def test_the_groupnorm_arms_differ_only_in_their_norm_layers_and_batch():
    graph = _graph("GroupNorm-Wu-2018")
    by_id = {n["id"]: n for n in graph["nodes"]}

    def types(arm):
        spec = json.loads(by_id[f"model-{arm}"]["data"]["params"]["layers"])
        return [n["type"] for n in spec["nodes"]]

    norms = {"BatchNorm2d", "BatchNorm1d", "GroupNorm"}
    assert types("bn") == types("bn32")
    assert [t for t in types("bn") if t in norms] == ["BatchNorm2d", "BatchNorm2d", "BatchNorm1d"]
    assert [t for t in types("gn") if t in norms] == ["GroupNorm"] * 3
    assert [t for t in types("bn") if t not in norms] == [t for t in types("gn") if t not in norms]

    loaders = {e["target"]: e["source"] for e in graph["edges"]
               if e.get("targetHandle") == "dataloader"}
    assert loaders == {"train-bn": "dl-train", "train-gn": "dl-train",
                       "train-bn32": "dl-train-32"}
    assert by_id["dl-train"]["data"]["params"]["batch_size"] == 2
    assert by_id["dl-train-32"]["data"]["params"]["batch_size"] == 32
