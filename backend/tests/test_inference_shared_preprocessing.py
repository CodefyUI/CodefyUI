"""#702: an inference graph reuses the training preprocessing.

One ``ToTensorTransform -> NormalizeTransform(CIFAR-10)`` chain feeds the
training dataset's ``eval_transform`` and the two image readers, the way an
inference graph is meant to be wired. What the readers hand the model must be
exactly what the dataset produced for the same files, and a single image
must reach a BatchNorm model without a manual reshape.
"""

from __future__ import annotations

import json

import pytest
import torch
from PIL import Image

from app.config import settings
from app.core.execution_context import ExecutionContext
from app.core.graph_engine import execute_graph

SIZE = 8

_MODEL_SPEC = {
    "version": 2,
    "nodes": [
        {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
        {"id": "c", "type": "Conv2d", "params": {"in_channels": 3, "out_channels": 4, "kernel_size": 3, "padding": 1}},
        {"id": "bn", "type": "BatchNorm2d", "params": {"num_features": 4}},
        {"id": "r", "type": "ReLU"},
        {"id": "f", "type": "Flatten"},
        {"id": "l", "type": "Linear", "params": {"in_features": 4 * SIZE * SIZE, "out_features": 3}},
        {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
    ],
    "edges": [
        {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "c"},
        {"id": "e2", "source": "c", "target": "bn"},
        {"id": "e3", "source": "bn", "target": "r"},
        {"id": "e4", "source": "r", "target": "f"},
        {"id": "e5", "source": "f", "target": "l"},
        {"id": "e6", "source": "l", "target": "out", "targetHandle": "p_y"},
    ],
}


@pytest.fixture
def image_root(monkeypatch, tmp_path):
    """``<data>/pets/{cat,dog}/*.png`` of seeded noise, inside the data root
    ImageBatchReader is confined to."""
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "data" / "models")
    root = settings.MODELS_DIR.parent / "pets"
    gen = torch.Generator().manual_seed(702)
    for cls in ("cat", "dog"):
        (root / cls).mkdir(parents=True)
        for i in range(3):
            pixels = torch.randint(0, 256, (SIZE, SIZE, 3), generator=gen, dtype=torch.uint8)
            Image.fromarray(pixels.numpy(), "RGB").save(root / cls / f"{cls}_{i}.png")
    return root


def _node(node_id, node_type, **params):
    return {"id": node_id, "type": node_type, "data": {"params": params}}


def _edge(edge_id, source, src, target, tgt):
    return {"id": edge_id, "source": source, "sourceHandle": src, "target": target, "targetHandle": tgt}


def _trigger(edge_id, target):
    return {"id": edge_id, "source": "start", "target": target, "sourceHandle": "trigger", "type": "trigger"}


async def _run_shared_preprocessing_graph(root):
    nodes = [
        _node("start", "Start"),
        _node("tt", "ToTensorTransform"),
        _node("norm", "NormalizeTransform", preset="CIFAR-10"),
        _node("train_ds", "ImageFolderDataset", path=str(root), split="(none)"),
        _node("one", "ImageReader", path=str(root / "dog" / "dog_1.png"), mode="RGB", resize=0, add_batch_dim=True),
        _node("many", "ImageBatchReader", directory=str(root / "cat"), pattern="*.png", resize=224, max_images=0, mode="RGB"),
        _node("model", "SequentialModel", layers=json.dumps(_MODEL_SPEC)),
        _node("infer_one", "Inference", device="cpu"),
        _node("infer_many", "Inference", device="cpu"),
    ]
    edges = [
        _trigger("t1", "tt"),
        _trigger("t2", "model"),
        _edge("e1", "tt", "transform", "norm", "transform"),
        _edge("e2", "norm", "transform", "train_ds", "eval_transform"),
        _edge("e3", "norm", "transform", "one", "transform"),
        _edge("e4", "norm", "transform", "many", "transform"),
        _edge("e5", "model", "model", "infer_one", "model"),
        _edge("e6", "one", "image", "infer_one", "input"),
        _edge("e7", "model", "model", "infer_many", "model"),
        _edge("e8", "many", "images", "infer_many", "input"),
    ]
    return await execute_graph(nodes, edges, context=ExecutionContext(device="cpu", seed=0))


def _training_sample(dataset, file_name):
    index = next(i for i, (p, _) in enumerate(dataset.samples) if p.endswith(file_name))
    return dataset[index][0]


@pytest.mark.asyncio
async def test_new_images_get_exactly_the_preprocessing_training_saw(image_root):
    out = await _run_shared_preprocessing_graph(image_root)
    dataset = out["train_ds"]["dataset"]

    # A single image: same tensor as the training sample, with a batch dim.
    one = out["one"]["image"]
    assert one.shape == (1, 3, SIZE, SIZE)
    torch.testing.assert_close(one[0], _training_sample(dataset, "dog_1.png"), atol=0.0, rtol=0.0)

    # A directory: the stacked training samples for those files, in order.
    expected = torch.stack([_training_sample(dataset, f"cat_{i}.png") for i in range(3)])
    torch.testing.assert_close(out["many"]["images"], expected, atol=0.0, rtol=0.0)

    # The normalisation really ran: CIFAR-10 statistics push values below 0,
    # which a bare ToTensor in [0, 1] never produces.
    assert out["many"]["images"].min() < 0

    # And the model's answers are the ones it gives on the training tensors.
    model = out["infer_many"]["model"].eval()
    with torch.no_grad():
        torch.testing.assert_close(out["infer_many"]["output"], model(expected))
        torch.testing.assert_close(
            out["infer_one"]["output"], model(_training_sample(dataset, "dog_1.png").unsqueeze(0)),
        )


def test_a_chain_without_to_tensor_says_what_to_add(image_root):
    from torchvision import transforms

    from app.nodes.io.image_reader_node import ImageReaderNode

    with pytest.raises(ValueError, match="Add ToTensorTransform"):
        ImageReaderNode().execute(
            {"transform": transforms.Compose([transforms.Resize((4, 4))])},
            {"path": str(image_root / "cat" / "cat_0.png")},
        )


def test_a_chain_that_leaves_sizes_unequal_names_the_files(image_root):
    from torchvision import transforms

    from app.nodes.io.image_batch_reader_node import ImageBatchReaderNode

    Image.new("RGB", (SIZE * 2, SIZE)).save(image_root / "cat" / "cat_wide.png")
    params = {"directory": str(image_root / "cat"), "pattern": "*.png"}
    with pytest.raises(ValueError, match=r"cat_wide\.png is \(3, 8, 16\).*ResizeTransform"):
        ImageBatchReaderNode().execute({"transform": transforms.Compose([transforms.ToTensor()])}, params)

    resized = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor()])
    assert ImageBatchReaderNode().execute({"transform": resized}, params)["images"].shape == (4, 3, SIZE, SIZE)


def test_an_unwired_reader_keeps_its_own_resize_and_shape(image_root):
    """Guard: without a chain both readers behave as before."""
    from app.nodes.io.image_batch_reader_node import ImageBatchReaderNode
    from app.nodes.io.image_reader_node import ImageReaderNode

    one = ImageReaderNode().execute({}, {"path": str(image_root / "cat" / "cat_0.png"), "resize": 4})["image"]
    assert one.shape == (3, 4, 4) and one.min() >= 0 and one.max() <= 1
    many = ImageBatchReaderNode().execute({}, {"directory": str(image_root / "cat"), "resize": 4})["images"]
    assert many.shape == (3, 3, 4, 4)
