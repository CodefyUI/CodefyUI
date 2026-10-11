"""Train a neural network on a CSV, end to end (#712).

Before #712 nothing turned tensors into a DATASET, CSVReader's target was
always a list of strings, and no node measured a regression. This module
runs the path the issue asks for through the real engine:

    CSVReader(target_type=number) → TrainTestSplit → TensorDataset → DataLoader
        → TrainingLoop → Inference → RegressionMetrics

on a generated CSV whose target is a known linear function of its features,
and checks the numbers against numpy. The node-level tests below pin the
edges the e2e run does not reach.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from app.core.execution_context import ExecutionContext, MetricSignal
from app.core.graph_engine import execute_graph
from app.nodes.classical.regression_metrics_node import RegressionMetricsNode
from app.nodes.data.csv_reader_node import CSVReaderNode
from app.nodes.data.tensor_dataset_node import TensorDatasetNode
from app.nodes.rl.gridworld_env_node import GridWorldEnvNode
from app.nodes.rl.policy_rollout_node import PolicyRolloutNode

_ROWS = 240
_FEATURES = 3
_WEIGHTS = np.array([2.0, -3.0, 0.5])

_MLP = (
    '{"version":2,"nodes":['
    '{"id":"in","type":"Input","ports":[{"id":"p_x","name":"x"}]},'
    f'{{"id":"l1","type":"Linear","params":{{"in_features":{_FEATURES},"out_features":16}}}},'
    '{"id":"r1","type":"ReLU"},'
    '{"id":"l2","type":"Linear","params":{"in_features":16,"out_features":1}},'
    '{"id":"out","type":"Output","ports":[{"id":"p_y","name":"y"}]}],'
    '"edges":['
    '{"id":"e1","source":"in","sourceHandle":"p_x","target":"l1"},'
    '{"id":"e2","source":"l1","target":"r1"},'
    '{"id":"e3","source":"r1","target":"l2"},'
    '{"id":"e4","source":"l2","target":"out","targetHandle":"p_y"}]}'
)


def _write_csv(path: Path) -> Path:
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=(_ROWS, _FEATURES))
    y = x @ _WEIGHTS + 0.05 * rng.standard_normal(_ROWS)
    lines = ["a,b,c,target"] + [
        ",".join(f"{v:.6f}" for v in (*row, t)) for row, t in zip(x, y)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _edge(source: str, source_handle: str, target: str, target_handle: str) -> dict:
    return {"id": f"{source}.{source_handle}->{target}.{target_handle}",
            "source": source, "sourceHandle": source_handle,
            "target": target, "targetHandle": target_handle}


def _graph(csv_path: Path) -> tuple[list[dict], list[dict]]:
    def node(node_id: str, node_type: str, **params) -> dict:
        return {"id": node_id, "type": node_type, "data": {"params": params}}

    nodes = [
        node("start", "Start"),
        node("csv", "CSVReader", path=str(csv_path), target_column="target",
             target_type="number"),
        node("split", "TrainTestSplit", test_size=0.25, seed=0),
        node("train_ds", "TensorDataset", target_type="number"),
        node("test_ds", "TensorDataset", target_type="number"),
        node("train_dl", "DataLoader", batch_size=16, shuffle=True),
        node("test_dl", "DataLoader", batch_size=64, shuffle=False),
        node("model", "SequentialModel", layers=_MLP),
        node("opt", "Optimizer", type="Adam", lr=0.01),
        node("loss", "Loss", type="MSELoss"),
        node("train", "TrainingLoop", epochs=40, device="cpu"),
        node("predict", "Inference", device="cpu"),
        node("metrics", "RegressionMetrics"),
    ]
    trigger = [{"id": f"t-{t}", "source": "start", "target": t,
                "sourceHandle": "trigger", "type": "trigger"}
               for t in ("csv", "model", "loss")]
    edges = trigger + [
        _edge("csv", "tensor", "split", "features"),
        _edge("csv", "labels", "split", "labels"),
        _edge("split", "x_train", "train_ds", "features"),
        _edge("split", "y_train", "train_ds", "targets"),
        _edge("split", "x_test", "test_ds", "features"),
        _edge("split", "y_test", "test_ds", "targets"),
        _edge("train_ds", "dataset", "train_dl", "dataset"),
        _edge("test_ds", "dataset", "test_dl", "dataset"),
        _edge("model", "model", "opt", "model"),
        _edge("model", "model", "train", "model"),
        _edge("opt", "optimizer", "train", "optimizer"),
        _edge("loss", "loss_fn", "train", "loss_fn"),
        _edge("train_dl", "dataloader", "train", "dataloader"),
        _edge("test_dl", "dataloader", "train", "val_dataloader"),
        _edge("train", "model", "predict", "model"),
        _edge("split", "x_test", "predict", "input"),
        _edge("predict", "output", "metrics", "predictions"),
        _edge("split", "y_test", "metrics", "targets"),
    ]
    return nodes, edges


async def test_an_mlp_regressor_trains_on_a_csv_and_its_metrics_match_numpy(tmp_path):
    nodes, edges = _graph(_write_csv(tmp_path / "linear.csv"))
    logged: dict[tuple[str, str], float] = {}

    class _RecordingContext(ExecutionContext):
        def log_metric(self, name, value, step, node_id=None):
            logged[(str(name), node_id or self.current_node_id)] = float(value)

    context = _RecordingContext(device="cpu", seed=0)
    outputs = await execute_graph(nodes, edges, context=context)

    losses = outputs["train"]["losses"].tolist()
    assert len(losses) == 40
    # The target's variance is about 4.4; an MLP that learned the linear map
    # ends far below it, and a broadcast [B, 1] vs [B] loss would not.
    assert losses[-1] < 0.1 * losses[0], losses

    y_test = np.asarray(outputs["split"]["y_test"], dtype=np.float64)
    y_hat = outputs["predict"]["output"].detach().numpy().astype(np.float64).reshape(-1)
    error = y_hat - y_test
    mse = np.mean(error ** 2)
    mae = np.mean(np.abs(error))
    r2 = 1.0 - np.sum(error ** 2) / np.sum((y_test - y_test.mean()) ** 2)

    metrics = outputs["metrics"]
    assert metrics["mse"] == pytest.approx(mse, rel=1e-9)
    assert metrics["mae"] == pytest.approx(mae, rel=1e-9)
    assert metrics["r2"] == pytest.approx(r2, rel=1e-9)
    assert metrics["r2"] > 0.95

    # Recorded as run metrics too, so a sweep can rank a regression (#711).
    assert logged[("mse", "metrics")] == pytest.approx(mse, rel=1e-9)
    assert logged[("r2", "metrics")] == pytest.approx(r2, rel=1e-9)
    assert ("val_loss", "train") in logged


# ── the node edges ────────────────────────────────────────────────────────


def test_csv_reader_numeric_target_names_the_row_that_is_not_a_number(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("x,y\n1,2.5\n2,n/a?\n3,4\n")
    params = {"path": str(path), "target_column": "y", "target_type": "number"}
    with pytest.raises(ValueError, match=r"data rows 2 \('n/a\?'\)"):
        CSVReaderNode().execute({}, params)

    path.write_text("x,y\n1,2.5\n2,3\n")
    assert CSVReaderNode().execute({}, params)["labels"] == [2.5, 3.0]
    # The default is unchanged: string class labels.
    params.pop("target_type")
    assert CSVReaderNode().execute({}, params)["labels"] == ["2.5", "3.0"]


def test_tensor_dataset_number_targets_are_a_column_for_a_one_output_model():
    dataset = TensorDatasetNode().execute(
        {"features": torch.zeros(3, 2), "targets": [1.5, -2, 0]},
        {"target_type": "number"})["dataset"]
    x, y = dataset.tensors
    assert y.dtype == torch.float32
    assert y.tolist() == [[1.5], [-2.0], [0.0]]


def test_tensor_dataset_class_targets_number_labels_in_sorted_order():
    out = TensorDatasetNode().execute(
        {"features": torch.zeros(4, 2), "targets": ["10", "9", "10", "2"]},
        {"target_type": "class"})
    assert out["classes"] == ["2", "9", "10"]
    assert out["dataset"].tensors[1].tolist() == [2, 1, 2, 0]
    assert out["dataset"].tensors[1].dtype == torch.long


def test_tensor_dataset_refuses_mismatched_rows_and_string_numbers():
    with pytest.raises(ValueError, match="3 rows but there are 2 targets"):
        TensorDatasetNode().execute(
            {"features": torch.zeros(3, 2), "targets": [1.0, 2.0]},
            {"target_type": "number"})
    with pytest.raises(ValueError, match="target_type to 'class'"):
        TensorDatasetNode().execute(
            {"features": torch.zeros(2, 2), "targets": ["setosa", "virginica"]},
            {"target_type": "number"})


def test_regression_metrics_r2_is_nan_for_a_constant_target_and_refuses_lengths():
    out = RegressionMetricsNode().execute(
        {"predictions": torch.tensor([1.0, 3.0]), "targets": [2.0, 2.0]}, {})
    assert out["mse"] == pytest.approx(1.0)
    assert out["mae"] == pytest.approx(1.0)
    assert np.isnan(out["r2"])
    with pytest.raises(ValueError, match="3 predictions vs 2 targets"):
        RegressionMetricsNode().execute(
            {"predictions": torch.zeros(3, 1), "targets": [1.0, 2.0]}, {})


def test_policy_rollout_records_its_success_rate_and_mean_return():
    """The RL rollout node logs what it returns (#711)."""
    env = GridWorldEnvNode().execute({}, {"size": 3, "max_steps": 6})["env"]
    model = torch.nn.Sequential(torch.nn.Linear(9, 4))
    context = ExecutionContext(device="cpu")
    out = PolicyRolloutNode().execute(
        {"model": model, "env": env}, {"episodes": 6, "seed": 3},
        context=context)
    signals, _dropped = context.outbox.drain()
    logged = {s.name: s.value for s in signals if isinstance(s, MetricSignal)}
    assert logged["success_rate"] == pytest.approx(out["success_rate"])
    assert logged["mean_return"] == pytest.approx(float(out["returns"].mean()))
