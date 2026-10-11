"""#696: nodes handed one live model take turns, and a shared arm is named.

The engine runs the nodes of a level concurrently. When a model output
feeds a TrainingLoop and an Inference (a sanity check), or two
TrainingLoops (an "AdamW vs SGD" comparison), they all hold one
``nn.Module``: Inference's ``eval()`` switched Dropout and BatchNorm off in
the middle of the training epoch, and two loops stepping one module failed
autograd's in-place check. The graphs below are the issue's, on synthetic
MNIST-shaped data, unseeded (a seeded run is serial already).

The model records, on every forward pass that builds a graph (training),
whether it was in train mode and how many such passes were in flight. It
sleeps briefly in those passes and before switching to eval mode, so that
on an engine that runs the siblings together the overlap happens on every
run.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
import torch
from torch import nn

from app.core.execution_context import ExecutionContext
from app.core.graph_engine import (
    execute_graph,
    shared_handle_groups,
    shared_model_training_warnings,
)
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import registry

FIXTURE_DATASET = "_SharedHandleDataset696"
MODEL_SOURCE = "_SharedHandleModel696"

_SAMPLES = 32
_CLASSES = 4


class _RecordingNet(nn.Module):
    """Flatten -> Linear -> BatchNorm, with a record of its training passes."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.body = nn.Sequential(
            nn.Flatten(), nn.Linear(28 * 28, _CLASSES),
            nn.BatchNorm1d(_CLASSES))
        self.train_mode_seen: list[bool] = []
        self.max_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    def train(self, mode: bool = True) -> "_RecordingNet":
        # ``eval()`` lands late, so that on an engine that lets a sibling
        # run alongside the TrainingLoop it lands mid-epoch every time
        # rather than only when the scheduler happens to line it up.
        if not mode:
            time.sleep(0.06)
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return self.body(x)
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            self.train_mode_seen.append(self.training)
        try:
            time.sleep(0.008)
            return self.body(x)
        finally:
            with self._lock:
                self._in_flight -= 1


_BUILT: list[_RecordingNet] = []


class _ModelSourceNode(BaseNode):
    NODE_NAME = MODEL_SOURCE
    CATEGORY = "Test"
    DESCRIPTION = "A small model that records how it was trained."
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="model", data_type=DataType.MODEL)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        model = _RecordingNet()
        _BUILT.append(model)
        return {"model": model}


class _DatasetNode(BaseNode):
    NODE_NAME = FIXTURE_DATASET
    CATEGORY = "Test"
    DESCRIPTION = "Fixed MNIST-shaped tensors."

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="dataset", data_type=DataType.DATASET)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(1)
        images = torch.rand(_SAMPLES, 1, 28, 28, generator=generator)
        targets = torch.arange(_SAMPLES) % _CLASSES
        return {"dataset": torch.utils.data.TensorDataset(images, targets)}


@pytest.fixture(autouse=True)
def _fixture_nodes():
    registry._nodes[FIXTURE_DATASET] = _DatasetNode
    registry._nodes[MODEL_SOURCE] = _ModelSourceNode
    _BUILT.clear()
    yield
    registry._nodes.pop(FIXTURE_DATASET, None)
    registry._nodes.pop(MODEL_SOURCE, None)
    _BUILT.clear()


def _edge(edge_id: str, source: str, src: str, target: str, tgt: str) -> dict:
    return {"id": edge_id, "source": source, "target": target,
            "sourceHandle": src, "targetHandle": tgt}


def _trigger(*targets: str) -> list[dict]:
    return [{"id": f"trig{i}", "source": "start", "target": target,
             "sourceHandle": "trigger", "type": "trigger"}
            for i, target in enumerate(targets)]


def _training_graph(*, with_inference: bool) -> tuple[list[dict], list[dict]]:
    """SequentialModel -> Optimizer -> TrainingLoop, plus a sanity Inference.

    ``DatasetBatch`` puts Inference one level down from the model, in the
    same level as the TrainingLoop, as in the issue's graph.
    """
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "model", "type": MODEL_SOURCE, "data": {"params": {}}},
        {"id": "data", "type": FIXTURE_DATASET, "data": {"params": {}}},
        {"id": "loss", "type": "Loss",
         "data": {"params": {"type": "CrossEntropyLoss"}}},
        {"id": "dl", "type": "DataLoader",
         "data": {"params": {"batch_size": 4, "shuffle": False,
                             "num_workers": 0}}},
        {"id": "opt", "type": "Optimizer",
         "data": {"params": {"type": "SGD", "lr": 0.1}}},
        {"id": "train", "type": "TrainingLoop",
         "data": {"params": {"epochs": 3, "device": "cpu"}}},
    ]
    edges = _trigger("model", "data", "loss") + [
        _edge("e1", "data", "dataset", "dl", "dataset"),
        _edge("e2", "model", "model", "opt", "model"),
        _edge("e3", "model", "model", "train", "model"),
        _edge("e4", "opt", "optimizer", "train", "optimizer"),
        _edge("e5", "dl", "dataloader", "train", "dataloader"),
        _edge("e6", "loss", "loss_fn", "train", "loss_fn"),
    ]
    if with_inference:
        nodes += [
            {"id": "sample", "type": "DatasetBatch",
             "data": {"params": {"batch_size": 2}}},
            {"id": "infer", "type": "Inference",
             "data": {"params": {"device": "cpu"}}},
        ]
        edges += [
            _edge("e7", "data", "dataset", "sample", "dataset"),
            _edge("e8", "sample", "images", "infer", "input"),
            _edge("e9", "model", "model", "infer", "model"),
        ]
    return nodes, edges


@pytest.mark.asyncio
async def test_a_sanity_inference_does_not_change_the_training_loss():
    """The issue's graph A: loss with the Inference branch == loss without.

    Before the fix, Inference ran during the first epoch and its ``eval()``
    left BatchNorm in eval mode for the rest of it, so the training saw
    eval-mode passes and a different loss curve (0.627 vs 1.936 at MNIST
    scale with Dropout).
    """
    nodes, edges = _training_graph(with_inference=False)
    plain = await execute_graph(nodes, edges,
                                context=ExecutionContext(device="cpu"))
    nodes, edges = _training_graph(with_inference=True)
    checked = await execute_graph(nodes, edges,
                                  context=ExecutionContext(device="cpu"))

    alone, with_sanity_check = _BUILT
    assert all(alone.train_mode_seen)
    assert all(with_sanity_check.train_mode_seen), (
        "a training forward pass ran in eval mode: "
        f"{with_sanity_check.train_mode_seen}")
    torch.testing.assert_close(checked["train"]["losses"],
                               plain["train"]["losses"])
    # The sanity check still ran on the model and produced its prediction.
    assert checked["infer"]["output"].shape == (2, _CLASSES)


def _two_arm_graph() -> tuple[list[dict], list[dict]]:
    """The issue's graph B: one model, an Adam arm and an SGD arm."""
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "model", "type": MODEL_SOURCE, "data": {"params": {}}},
        {"id": "data", "type": FIXTURE_DATASET, "data": {"params": {}}},
        {"id": "loss", "type": "Loss",
         "data": {"params": {"type": "CrossEntropyLoss"}}},
        {"id": "dl", "type": "DataLoader",
         "data": {"params": {"batch_size": 4, "shuffle": False,
                             "num_workers": 0}}},
        {"id": "opt_adam", "type": "Optimizer",
         "data": {"params": {"type": "Adam", "lr": 0.01}}},
        {"id": "opt_sgd", "type": "Optimizer",
         "data": {"params": {"type": "SGD", "lr": 0.1}}},
        {"id": "train_adam", "type": "TrainingLoop",
         "data": {"params": {"epochs": 2, "device": "cpu"}}},
        {"id": "train_sgd", "type": "TrainingLoop",
         "data": {"params": {"epochs": 2, "device": "cpu"}}},
    ]
    edges = _trigger("model", "data", "loss") + [
        _edge("e1", "data", "dataset", "dl", "dataset"),
        _edge("e2", "model", "model", "opt_adam", "model"),
        _edge("e3", "model", "model", "opt_sgd", "model"),
        _edge("e4", "model", "model", "train_adam", "model"),
        _edge("e5", "model", "model", "train_sgd", "model"),
        _edge("e6", "opt_adam", "optimizer", "train_adam", "optimizer"),
        _edge("e7", "opt_sgd", "optimizer", "train_sgd", "optimizer"),
        _edge("e8", "dl", "dataloader", "train_adam", "dataloader"),
        _edge("e9", "dl", "dataloader", "train_sgd", "dataloader"),
        _edge("e10", "loss", "loss_fn", "train_adam", "loss_fn"),
        _edge("e11", "loss", "loss_fn", "train_sgd", "loss_fn"),
    ]
    return nodes, edges


class _WarningContext(ExecutionContext):
    """Keeps what the run said through ``log_warning``."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.warnings: list[tuple[str, str]] = []

    def log_warning(self, kind, detail, node_id=None):
        self.warnings.append((kind, detail))


@pytest.mark.asyncio
async def test_two_training_loops_on_one_model_take_turns_and_are_warned():
    """Graph B, unseeded: it runs, one loop at a time, and says why.

    Before the fix the two loops stepped the module at the same time
    (two training passes in flight) and the run failed with "one of the
    variables needed for gradient computation has been modified by an
    inplace operation".
    """
    nodes, edges = _two_arm_graph()
    context = _WarningContext(device="cpu")

    outputs = await execute_graph(nodes, edges, context=context)

    (model,) = _BUILT
    assert model.max_in_flight == 1
    assert len(model.train_mode_seen) == 2 * 2 * (_SAMPLES // 4)
    assert outputs["train_adam"]["losses"].shape == (2,)
    assert outputs["train_sgd"]["losses"].shape == (2,)
    kinds = [kind for kind, _ in context.warnings]
    assert kinds.count("shared_model_training") == 1
    (detail,) = [d for k, d in context.warnings if k == "shared_model_training"]
    assert "train_adam" in detail and "train_sgd" in detail


def test_the_shared_model_warning_names_the_arms_and_spares_one_arm():
    """The /validate warning: two arms on one model, and nothing for one."""
    nodes, edges = _two_arm_graph()
    (warning,) = shared_model_training_warnings(nodes, edges)
    assert warning.code == "shared_model_training"
    assert warning.node_id == "model"
    assert warning.params["trainers"] == ["train_adam", "train_sgd"]

    nodes, edges = _training_graph(with_inference=True)
    assert shared_model_training_warnings(nodes, edges) == []


def test_nodes_with_their_own_models_stay_in_separate_groups():
    """Only a shared handle serialises; independent arms keep running together."""
    first, second = nn.Linear(2, 2), nn.Linear(2, 2)
    optimizer = torch.optim.SGD(first.parameters(), lr=0.1)
    inputs = {
        "train_a": [first, optimizer],
        "infer_a": [first, torch.zeros(1)],
        "train_b": [second],
        "plot": [torch.zeros(1)],
        "sched_a": [optimizer],
    }
    groups = shared_handle_groups(list(inputs), lambda nid: inputs[nid])
    assert groups == [["train_a", "infer_a", "sched_a"], ["train_b"], ["plot"]]



@pytest.mark.asyncio
async def test_validate_returns_the_shared_model_warning_and_stays_valid(
    test_client,
):
    """``POST /api/graph/validate`` lists it under ``warnings``, not ``errors``."""
    nodes, edges = _two_arm_graph()
    for node in nodes:
        node.setdefault("position", {"x": 0, "y": 0})
    response = await test_client.post("/api/graph/validate",
                                      json={"nodes": nodes, "edges": edges})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["valid"] is True
    assert body["errors"] == []
    (warning,) = body["warnings"]
    assert warning["code"] == "shared_model_training"
    assert warning["node_id"] == "model"
    assert warning["params"]["trainers"] == ["train_adam", "train_sgd"]
