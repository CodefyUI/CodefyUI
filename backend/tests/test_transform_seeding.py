"""Seeded augmentation is reproducible (core#136, on top of #134's run seed).

Augmentation is the first randomness in CodefyUI that runs INSIDE another
node: the transform fires when the DataLoader pulls a sample, which happens
in the middle of ``TrainingLoop.execute``. #134's per-node ``seed_rngs``
covers that path only in the sense that the whole run is deterministic --
the numbers the crops draw still depend on everything the model drew first,
and in a multi-worker run they are re-seeded to the same value at the start
of every epoch, which would freeze the augmentation after epoch 1.

So the properties under test are, in the order they matter:

1. Same run seed, same augmented samples. Twice, from a cold start.
2. Different run seed, different augmented samples.
3. Augmentation still VARIES -- across epochs, and between two samples of
   one epoch. A wrapper that made every sample identical would pass (1) and
   (2) and destroy the feature.
4. The stream is isolated: what the rest of the graph drew first does not
   change what the crops draw.
5. No seed, no wrapper. An unseeded run keeps torch's own entropy.
"""

from __future__ import annotations

import pickle

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from app.core.execution_context import ExecutionContext
from app.nodes.data.transforms._base import (
    SeededAugmentation,
    attach_transform,
    seed_pipeline,
)
from app.nodes.data.transforms.random_crop_node import RandomCropNode
from app.nodes.data.transforms.random_horizontal_flip_node import (
    RandomHorizontalFlipNode,
)
from app.nodes.data.transforms.to_tensor_transform_node import (
    ToTensorTransformNode,
)
from app.nodes.data.transform_node import TransformNode


class _ImageDataset(Dataset):
    """A handful of fixed images with a mutable ``transform``, like torchvision."""

    def __init__(self, count: int = 4) -> None:
        generator = torch.Generator().manual_seed(11)
        self._images = [
            Image.fromarray(
                torch.randint(0, 256, (16, 16, 3), generator=generator,
                              dtype=torch.uint8).numpy(), mode="RGB")
            for _ in range(count)
        ]
        self.transform = None

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int):
        image = self._images[index]
        return (self.transform(image) if self.transform is not None
                else image), 0


def _augmenting_chain():
    chain = RandomCropNode().execute({}, {"size": 16, "padding": 4})
    chain = RandomHorizontalFlipNode().execute(
        {"transform": chain["transform"]}, {"p": 0.5})
    chain = ToTensorTransformNode().execute({"transform": chain["transform"]}, {})
    return chain["transform"]


def _deterministic_chain():
    return ToTensorTransformNode().execute({}, {})["transform"]


def _context(seed: int | None, node_id: str = "tf1") -> ExecutionContext:
    context = ExecutionContext(seed=seed)
    context.current_node_id = node_id
    return context


def _epoch(dataset, batch_size: int = 2) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return torch.cat([batch for batch, _ in loader])


def _augmented(seed: int | None, node_id: str = "tf1",
               warmup: int = 0) -> torch.Tensor:
    """One epoch's worth of augmented samples, from a cold start.

    ``warmup`` draws that many numbers from the global RNG first, standing
    in for whatever the rest of the graph did before the loop started.
    """
    dataset = _ImageDataset()
    torch.manual_seed(999)
    if warmup:
        torch.rand(warmup)
    attach_transform(dataset, _augmenting_chain(), _context(seed, node_id))
    return _epoch(dataset)


# ── the wrapper is installed exactly when it should be ───────────────────


def test_a_seeded_run_wraps_a_random_pipeline():
    wrapped = seed_pipeline(_augmenting_chain(), _context(1234), "tf1")
    assert isinstance(wrapped, SeededAugmentation)


def test_an_unseeded_run_leaves_the_pipeline_alone():
    pipeline = _augmenting_chain()
    assert seed_pipeline(pipeline, _context(None), "tf1") is pipeline


def test_a_deterministic_pipeline_is_never_wrapped():
    """Nothing to isolate, so nothing pays the per-sample cost."""
    pipeline = _deterministic_chain()
    assert seed_pipeline(pipeline, _context(1234), "tf1") is pipeline


def test_no_context_leaves_the_pipeline_alone():
    pipeline = _augmenting_chain()
    assert seed_pipeline(pipeline, None, "tf1") is pipeline


def test_attach_transform_installs_it_on_the_dataset():
    dataset = _ImageDataset()
    returned = attach_transform(dataset, _augmenting_chain(), _context(7))
    assert returned is dataset
    assert isinstance(dataset.transform, SeededAugmentation)


def test_transform_node_wraps_a_wired_chain():
    dataset = _ImageDataset()
    TransformNode().execute(
        {"dataset": dataset, "transform": _augmenting_chain()}, {},
        context=_context(7))
    assert isinstance(dataset.transform, SeededAugmentation)


def test_transform_node_params_path_is_never_wrapped():
    """The three built-in steps contain no randomness."""
    dataset = _ImageDataset()
    TransformNode().execute(
        {"dataset": dataset},
        {"resize": 0, "normalize": True, "to_tensor": True},
        context=_context(7))
    assert not isinstance(dataset.transform, SeededAugmentation)


# ── the properties that make it worth having ─────────────────────────────


def test_the_same_run_seed_reproduces_the_same_augmentations():
    assert torch.equal(_augmented(4242), _augmented(4242))


def test_a_different_run_seed_augments_differently():
    assert not torch.equal(_augmented(4242), _augmented(4243))


def test_a_different_node_id_augments_differently():
    """Two Transform nodes in one graph must not walk the same stream."""
    assert not torch.equal(_augmented(4242, "tf1"), _augmented(4242, "tf2"))


def test_the_augmentation_still_varies_between_samples():
    """Reproducible must not mean 'the same picture every time'."""
    batch = _augmented(4242)
    assert not torch.equal(batch[0], batch[1])


def test_the_augmentation_still_varies_between_epochs():
    dataset = _ImageDataset()
    attach_transform(dataset, _augmenting_chain(), _context(4242))
    first, second = _epoch(dataset), _epoch(dataset)
    assert first.shape == second.shape
    assert not torch.equal(first, second)


def test_the_stream_does_not_depend_on_what_the_graph_drew_first():
    """The isolation property, and the reason the wrapper exists at all.

    Without it the crops come out of the global RNG that the training loop
    is also drawing dropout masks from, so adding a layer to the model
    would change the augmentations.
    """
    assert torch.equal(_augmented(4242, warmup=0),
                       _augmented(4242, warmup=137))


def test_the_wrapper_hands_the_callers_rng_back_untouched():
    """A fork, not a reseed: the caller's own stream must not be disturbed.

    Called directly rather than through a DataLoader, which draws its own
    ``_base_seed`` from the global RNG on every iterator and would put the
    loader's consumption into the measurement.
    """
    dataset = _ImageDataset()
    attach_transform(dataset, _augmenting_chain(), _context(4242))

    torch.manual_seed(5)
    expected = torch.rand(3)
    torch.manual_seed(5)
    for _ in range(4):
        dataset.transform(dataset._images[0])
    assert torch.equal(torch.rand(3), expected)


def test_an_unseeded_run_is_not_pinned_to_one_sequence():
    """The unseeded path keeps torch's entropy, which is what 'no seed' means."""
    torch.manual_seed(1)
    first = _augmented(None)
    torch.manual_seed(2)
    second = _augmented(None)
    # ``_augmented`` reseeds to a fixed value itself, so this only shows the
    # unwrapped path follows the global RNG rather than a derived stream.
    assert torch.equal(first, second)
    assert not isinstance(_ImageDataset().transform, SeededAugmentation)


# ── worker processes ─────────────────────────────────────────────────────


class _FakeWorkerInfo:
    def __init__(self, worker_id: int, seed: int) -> None:
        self.id = worker_id
        self.seed = seed


def _as_worker(monkeypatch, info):
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: info)


def test_each_worker_gets_its_own_stream(monkeypatch):
    """Two workers drawing at the same index must not draw the same numbers."""
    wrapper = SeededAugmentation(_augmenting_chain(), 4242)
    image = _ImageDataset()._images[0]

    _as_worker(monkeypatch, _FakeWorkerInfo(0, 1000))
    first = wrapper(image)
    _as_worker(monkeypatch, _FakeWorkerInfo(1, 1001))
    second = wrapper(image)
    assert not torch.equal(first, second)


def test_a_worker_restarted_for_the_next_epoch_augments_differently(monkeypatch):
    """The bug augmentation is the first feature to expose.

    ``worker_init_fn`` re-seeds each worker to a FIXED value at every
    iterator creation, so without mixing in torch's own per-iterator worker
    seed every epoch would replay epoch 1's crops. The wrapper's counter
    resets too (the object is pickled fresh into the worker), so the
    per-iterator seed is the ONLY thing distinguishing them.
    """
    image = _ImageDataset()._images[0]

    _as_worker(monkeypatch, _FakeWorkerInfo(0, 1000))
    epoch1 = SeededAugmentation(_augmenting_chain(), 4242)(image)
    _as_worker(monkeypatch, _FakeWorkerInfo(0, 2000))
    epoch2 = SeededAugmentation(_augmenting_chain(), 4242)(image)
    assert not torch.equal(epoch1, epoch2)


def test_a_worker_replayed_with_the_same_seeds_reproduces(monkeypatch):
    image = _ImageDataset()._images[0]
    _as_worker(monkeypatch, _FakeWorkerInfo(0, 1000))
    first = SeededAugmentation(_augmenting_chain(), 4242)(image)
    second = SeededAugmentation(_augmenting_chain(), 4242)(image)
    assert torch.equal(first, second)


def test_the_wrapper_survives_pickling():
    """Spawn-started DataLoader workers pickle the dataset and its transform."""
    wrapper = SeededAugmentation(_augmenting_chain(), 4242)
    image = _ImageDataset()._images[0]
    restored = pickle.loads(pickle.dumps(wrapper))
    assert restored.seed == wrapper.seed
    assert torch.equal(restored(image), wrapper(image))


# ── through the engine ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_two_seeded_runs_of_the_same_graph_augment_identically():
    """End to end: the run seed reaches the transform through the engine.

    Dataset -> RandomCrop -> Flip -> ToTensor -> Transform, then the batch
    the loader would hand a training loop. Nothing here mocks the seeding;
    the only input that differs between the two runs is nothing at all.
    """
    from app.core.graph_engine import execute_graph

    def graph():
        nodes = [
            {"id": "start", "type": "Start", "data": {"params": {}}},
            {"id": "crop", "type": "RandomCrop",
             "data": {"params": {"size": 16, "padding": 4}}},
            {"id": "flip", "type": "RandomHorizontalFlip",
             "data": {"params": {"p": 0.5}}},
            {"id": "tensor", "type": "ToTensorTransform", "data": {"params": {}}},
            {"id": "src", "type": "_SeedFixtureDataset", "data": {"params": {}}},
            {"id": "apply", "type": "Transform", "data": {"params": {}}},
            {"id": "sink", "type": "_SeedFixtureEpoch", "data": {"params": {}}},
        ]
        edges = [
            {"id": "t1", "source": "start", "target": "crop",
             "sourceHandle": "trigger", "type": "trigger"},
            {"id": "t2", "source": "start", "target": "src",
             "sourceHandle": "trigger", "type": "trigger"},
            {"source": "crop", "sourceHandle": "transform",
             "target": "flip", "targetHandle": "transform"},
            {"source": "flip", "sourceHandle": "transform",
             "target": "tensor", "targetHandle": "transform"},
            {"source": "tensor", "sourceHandle": "transform",
             "target": "apply", "targetHandle": "transform"},
            {"source": "src", "sourceHandle": "dataset",
             "target": "apply", "targetHandle": "dataset"},
            {"source": "apply", "sourceHandle": "dataset",
             "target": "sink", "targetHandle": "dataset"},
        ]
        return nodes, edges

    async def run(seed: int) -> torch.Tensor:
        nodes, edges = graph()
        result = await execute_graph(
            nodes, edges, context=ExecutionContext(seed=seed))
        return result["sink"]["batch"]

    first, second = await run(31337), await run(31337)
    assert torch.equal(first, second)
    assert not torch.equal(first, await run(31338))


# Fixture nodes for the engine test. Registered by a FIXTURE, never at
# import time: pytest imports every test module during collection, and
# conftest's session-scoped ``registry_with_nodes`` only discovers the
# builtins and the PRESETS when it finds the registry still empty. A
# module-level ``register`` call runs first and silently skips preset
# discovery for the whole session. Node names starting with an underscore
# are reserved for tests by convention (see test_log_metric.py).


from app.core.node_base import (  # noqa: E402
    BaseNode,
    DataType,
    PortDefinition,
)
from app.core.node_registry import registry  # noqa: E402


class _SeedFixtureDatasetNode(BaseNode):
    NODE_NAME = "_SeedFixtureDataset"
    CATEGORY = "Test"
    DESCRIPTION = "Fixed images with a writable transform attribute"
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="dataset", data_type=DataType.DATASET)]

    def execute(self, inputs, params, progress_callback=None, *, context=None):
        return {"dataset": _ImageDataset()}


class _SeedFixtureEpochNode(BaseNode):
    NODE_NAME = "_SeedFixtureEpoch"
    CATEGORY = "Test"
    DESCRIPTION = "Pulls one epoch through the dataset's transform"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="dataset", data_type=DataType.DATASET)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="batch", data_type=DataType.TENSOR)]

    def execute(self, inputs, params, progress_callback=None, *, context=None):
        return {"batch": _epoch(inputs["dataset"])}


_TEST_NODES = {
    "_SeedFixtureDataset": _SeedFixtureDatasetNode,
    "_SeedFixtureEpoch": _SeedFixtureEpochNode,
}


@pytest.fixture(autouse=True)
def _register_test_nodes():
    registry._nodes.update(_TEST_NODES)
    yield
    for name in _TEST_NODES:
        registry._nodes.pop(name, None)
