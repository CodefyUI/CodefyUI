"""The paper reproductions in ``examples/Papers/`` land where their notes say.

Six graphs, gallery orders 11 to 16, each reproducing one paper's central
claim on a CPU in about 20 seconds or less: gradient clipping (Pascanu et
al. 2013), the 2-D autoencoder (Hinton & Salakhutdinov 2006), LSTM
forecasting (Hochreiter & Schmidhuber 1997), a transformer that sorts
(Vaswani et al. 2017), weight decay (Krogh & Hertz 1992) and early stopping
(Prechelt 1998).

``test_builtin_examples.py`` validates these graphs and skips executing them
(they train). This file executes each one under the seed its file stores, on
the CPU, the way the canvas runs an opened example, and holds two things:

* the claim, as an inequality with a margin (clipping beats no clipping,
  the autoencoder beats PCA, and so on), which is what the example exists
  to show;
* the numbers the overview notes quote that stay put across run seeds,
  within +-0.03 on an accuracy or R² and 10% on an MSE. Two runs sit in a
  chaotic regime (the clipping RNN at the edge of divergence, the undecayed
  MLP fitting noise); their numbers move with the CPU, so they are held as
  bands and their notes quote both the M3 and the Linux CI value.

The numbers were measured on an Apple M3 (torch 2.11). Training from a
small random initialisation amplifies float rounding, so another CPU can
land a little off them; the claim assertions are sized from the spread over
run seeds 0 to 4 and are the ones that must never fail.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path

import pytest

from app.core.execution_context import ExecutionContext
from app.core.graph_engine import execute_graph

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PAPERS = _REPO_ROOT / "examples" / "Papers"
_BACKEND_DIR = _REPO_ROOT / "backend"


def _graph(name: str) -> dict:
    return json.loads((_PAPERS / name / "graph.json").read_text(encoding="utf-8"))


def _overview(graph: dict) -> str:
    return next(n for n in graph["nodes"]
                if n["id"] == "note-overview")["data"]["noteContent"]


def _run(name: str) -> tuple[dict, dict, list[tuple[str, str, int, float]]]:
    """Execute *name* under its stored seed; return graph, results, metrics."""
    graph = _graph(name)
    metrics: list[tuple[str, str, int, float]] = []

    def on_signal(signal):
        if getattr(signal, "name", None) is not None:
            metrics.append((signal.node_id, signal.name, signal.step, signal.value))

    prev_cwd = Path.cwd()
    os.chdir(_BACKEND_DIR)  # Dataset's data_dir is relative to backend/
    try:
        results = asyncio.run(execute_graph(
            graph["nodes"], graph["edges"], error_mode="fail_fast",
            context=ExecutionContext(device="cpu", seed=graph["settings"]["seed"]),
            on_signal=on_signal))
    finally:
        os.chdir(prev_cwd)
    return graph, results, metrics


def _series(metrics, node_id: str, name: str) -> list[float]:
    return [v for (nid, n, _, v) in metrics if nid == node_id and n == name]


# ── 11: gradient clipping ─────────────────────────────────────────────────

def test_gradient_clipping_trains_the_rnn_that_explodes_without_it():
    graph, res, metrics = _run("Gradient-Clipping-RNN")
    note = _overview(graph)
    plain, clipped = res["eval-plain"]["accuracy"], res["eval-clip"]["accuracy"]

    # The claim: same RNN, same data, only grad_clip_norm differs. This run
    # sits near the edge of divergence, so another CPU's float rounding moves
    # its numbers (clipped: 0.886 on an Apple M3, 1.0 on the Linux CI
    # runner). The bands below are the claim; the note quotes both values.
    assert clipped >= 0.8, clipped
    assert plain <= 0.3, "the unclipped RNN should stay near chance (0.10)"
    assert "0.89 test accuracy on an Apple M3 and 1.0 on the Linux CI runner" in note

    # log_grad_norm put the pre-clip norm in the run's metrics, and the
    # clipped run's applied norm never passed the threshold.
    raw = [v for v in _series(metrics, "train-plain", "grad_norm") if math.isfinite(v)]
    applied = _series(metrics, "train-clip", "grad_norm_clipped")
    assert raw and applied
    assert max(applied) <= 1.0 + 1e-4
    median = sorted(raw)[len(raw) // 2]
    assert max(raw) > 20 * median, "the unclipped run should show a spike"

    # note-chart: the clipped loss leaves the ln 10 plateau, and the unclipped
    # one ends back on it.
    assert res["train-clip"]["losses"].tolist()[-1] < 1.0
    assert res["train-plain"]["losses"].tolist()[-1] > 2.2


# ── 12: autoencoder vs PCA ────────────────────────────────────────────────

def test_the_autoencoder_keeps_more_of_the_digit_than_pca():
    graph, res, _ = _run("Autoencoder-MNIST-2D")
    note = _overview(graph)
    mse_ae, mse_pca = res["mse-ae"]["mse"], res["mse-pca"]["mse"]
    knn_ae, knn_pca = res["encode"]["out3"], res["pca"]["out4"]

    assert mse_ae < 0.9 * mse_pca, (mse_ae, mse_pca)
    assert knn_ae > knn_pca + 0.08, (knn_ae, knn_pca)
    assert tuple(res["encode"]["out1"].shape) == (1000, 2), "2-D codes"

    assert mse_ae == pytest.approx(0.0438, rel=0.1)
    assert mse_pca == pytest.approx(0.0547, rel=0.05)
    assert knn_ae == pytest.approx(0.59, abs=0.05)
    assert knn_pca == pytest.approx(0.45, abs=0.03)
    for quoted in ("0.0438", "0.0547", "0.59", "0.45"):
        assert quoted in note, quoted


# ── 13: LSTM forecasting ──────────────────────────────────────────────────

def test_the_lstm_forecasts_held_out_windows_with_r2_above_0_9():
    graph, res, _ = _run("LSTM-Forecast-Sine")
    note = _overview(graph)
    r2, r2_base = res["metrics"]["r2"], res["baseline"]["r2"]
    held_out = res["gen"]["out4"]
    cap = 1 - 0.15 ** 2 / float(held_out.var())

    assert r2 > 0.9
    assert r2 > r2_base + 0.04, (r2, r2_base)
    assert r2 <= cap + 0.01, "nothing beats the noise floor"

    assert r2 == pytest.approx(0.95, abs=0.03)
    assert r2_base == pytest.approx(0.87, abs=0.01)
    assert cap == pytest.approx(0.97, abs=0.01)
    assert "R² 0.95" in note and "0.87" in note and "0.97" in note


def test_the_lstm_example_tests_on_the_future():
    """The test windows all come after the training ones in time."""
    graph = _graph("LSTM-Forecast-Sine")
    edges = {(e["source"], e.get("sourceHandle"), e["target"], e.get("targetHandle"))
             for e in graph["edges"]}
    assert ("gen", "out1", "ds", "features") in edges
    assert ("gen", "out3", "infer", "input") in edges
    assert ("gen", "out4", "metrics", "targets") in edges


# ── 14: a transformer that sorts ──────────────────────────────────────────

def test_the_transformer_sorts_fresh_sequences():
    graph, res, _ = _run("Transformer-Sort")
    note = _overview(graph)
    exact, token = res["score"]["out1"], res["score"]["out2"]

    assert exact == pytest.approx(1.0, abs=0.03)
    assert token == pytest.approx(1.0, abs=0.01)
    assert "exact-match accuracy 1.00" in note
    assert "per-position accuracy 1.000" in note

    # Scored on sequences generated after, and apart from, the training set.
    x_test = res["gen"]["out2"]
    assert tuple(x_test.shape) == (1000, 10)
    assert len(res["gen"]["out1"]) == 5000


# ── 15: weight decay ──────────────────────────────────────────────────────

def test_weight_decay_gives_the_lower_test_error():
    graph, res, _ = _run("Weight-Decay-Regression")
    note = _overview(graph)
    test_plain, test_decay = res["mse-plain"]["mse"], res["mse-decay"]["mse"]
    train_plain = float(res["train-plain"]["losses"][-1])
    train_decay = float(res["train-decay"]["losses"][-1])

    # The undecayed net fits the noise: lower training error, higher test.
    # Its numbers come from fitting noise, so a different CPU's rounding moves
    # them (training MSE 0.0012 on an Apple M3, 0.003 on the Linux CI runner);
    # the bands below are the claim. The decayed run is the stable one.
    assert train_plain < 0.01, train_plain
    assert train_plain < train_decay
    assert test_decay < test_plain / 2, (test_decay, test_plain)

    assert test_decay == pytest.approx(0.033, rel=0.1)
    assert train_decay == pytest.approx(0.056, rel=0.1)
    for quoted in ("about 0.001 to 0.003", "0.269", "0.056", "0.033"):
        assert quoted in note, quoted


# ── 16: early stopping ────────────────────────────────────────────────────

def test_early_stopping_restores_the_better_model():
    graph, res, _ = _run("Early-Stopping-Regression")
    note = _overview(graph)
    metrics = res["train-early"]["metrics"]
    test_full, test_early = res["mse-full"]["mse"], res["mse-early"]["mse"]

    assert metrics["stopped_early"] is True
    assert metrics["total_epochs_run"] == metrics["best_epoch"] + 100, (
        "patience 100: it stops 100 epochs after the best one")
    assert res["train-full"]["metrics"]["total_epochs_run"] == 1000
    assert test_early < test_full / 2, (test_early, test_full)

    assert metrics["best_epoch"] == pytest.approx(121, abs=15)
    assert test_early == pytest.approx(0.033, rel=0.1)
    assert test_full == pytest.approx(0.095, rel=0.1)
    assert res["curves"]["out3"] == metrics["best_epoch"]
    for quoted in ("after epoch 221", "restores epoch 121", "0.033", "0.095"):
        assert quoted in note, quoted
