"""#697: a Transform leaves the dataset it was handed alone.

Every consumer of a ``Dataset`` output holds the same object. ``Transform``
used to install its pipeline on that object, so a sibling branch wired to
the untransformed dataset saw the transform too, and two Transforms on one
dataset (train and eval) ended up sharing whichever pipeline was installed
last. These graphs are the issue's, at MNIST shape on synthetic samples.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from app.core.execution_context import ExecutionContext
from app.core.graph_engine import execute_graph
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import registry

FIXTURE_DATASET = "_TransformSiblingDataset697"


class _ImageLikeDataset(torch.utils.data.Dataset):
    """MNIST-shaped tensors that honour ``transform`` like torchvision."""

    def __init__(self) -> None:
        self.images = torch.rand(4, 1, 28, 28, generator=torch.Generator().manual_seed(3))
        self.transform = None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, index


class _FixtureDatasetNode(BaseNode):
    NODE_NAME = FIXTURE_DATASET
    CATEGORY = "Test"
    DESCRIPTION = "Four MNIST-shaped images."

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="dataset", data_type=DataType.DATASET)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        return {"dataset": _ImageLikeDataset()}


@pytest.fixture(autouse=True)
def _fixture_dataset():
    registry._nodes[FIXTURE_DATASET] = _FixtureDatasetNode
    yield
    registry._nodes.pop(FIXTURE_DATASET, None)


def _resize(node_id: str, size: int) -> dict:
    return {"id": node_id, "type": "Transform",
            "data": {"params": {"resize": size, "to_tensor": False,
                                "normalize": False}}}


def _batch(node_id: str) -> dict:
    return {"id": node_id, "type": "DatasetBatch",
            "data": {"params": {"batch_size": 2}}}


def _wire(edge_id: str, source: str, target: str, src: str = "dataset",
          tgt: str = "dataset") -> dict:
    return {"id": edge_id, "source": source, "target": target,
            "sourceHandle": src, "targetHandle": tgt}


_START = [{"id": "start", "type": "Start", "data": {"params": {}}}]
_TRIGGER = [{"id": "trig", "source": "start", "target": "data",
             "sourceHandle": "trigger", "type": "trigger"}]


@pytest.mark.asyncio
async def test_a_sibling_wired_to_the_raw_dataset_gets_raw_samples():
    """``Dataset -> Transform(resize=14) -> DatasetBatch A`` and
    ``Dataset -> DatasetBatch B``: B is ``[2, 1, 28, 28]``.

    Seeded, so the level runs in node order and the Transform runs before
    B every time: before the fix B came back ``[2, 1, 14, 14]`` on every
    run, and an unseeded run gave either shape depending on scheduling.
    """
    nodes = _START + [
        {"id": "data", "type": FIXTURE_DATASET, "data": {"params": {}}},
        _resize("tf", 14),
        _batch("raw"),
        _batch("small"),
    ]
    edges = _TRIGGER + [
        _wire("e1", "data", "tf"),
        _wire("e2", "data", "raw"),
        _wire("e3", "tf", "small"),
    ]

    outputs = await execute_graph(
        nodes, edges, context=ExecutionContext(device="cpu", seed=0))

    assert list(outputs["raw"]["images"].shape) == [2, 1, 28, 28]
    assert list(outputs["small"]["images"].shape) == [2, 1, 14, 14]
    assert outputs["data"]["dataset"].transform is None


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [None, 0])
async def test_two_transforms_on_one_dataset_each_keep_their_own(seed):
    """A train and an eval Transform on one dataset (resize 14 and 7).

    Both DatasetBatches run a level after both Transforms, so before the fix
    both read whichever pipeline was installed last and one of them always
    had the other's shape.
    """
    nodes = _START + [
        {"id": "data", "type": FIXTURE_DATASET, "data": {"params": {}}},
        _resize("train_tf", 14),
        _resize("eval_tf", 7),
        _batch("train_batch"),
        _batch("eval_batch"),
    ]
    edges = _TRIGGER + [
        _wire("e1", "data", "train_tf"),
        _wire("e2", "data", "eval_tf"),
        _wire("e3", "train_tf", "train_batch"),
        _wire("e4", "eval_tf", "eval_batch"),
    ]

    outputs = await execute_graph(
        nodes, edges, context=ExecutionContext(device="cpu", seed=seed))

    assert list(outputs["train_batch"]["images"].shape) == [2, 1, 14, 14]
    assert list(outputs["eval_batch"]["images"].shape) == [2, 1, 7, 7]
