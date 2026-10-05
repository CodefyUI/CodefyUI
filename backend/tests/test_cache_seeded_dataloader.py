"""A seeded DataLoader shuffles the same way on every Run, cached or not.

Found by browser e2e on the textbook's I3-2 graph (seed 0, Persist weights
off): three Runs printed 0.94944 / 0.94992 / 0.94918 while the exported
script printed 0.94944 every time. Run 1 matched the export; no later Run
did.

``DataLoader`` hands its loader a ``torch.Generator`` seeded from (run seed,
node id), so the shuffle order depends on the seed alone (#134). That
generator is state, and iterating the loader spends it: every epoch draws a
base seed when its iterator is created and then that epoch's permutation.
The node was cacheable and the canvas lends one ``ExecutionCache`` to every
Run, so Run 2 was handed Run 1's loader object, its generator already
``epochs`` permutations along. Same seed, same cache key -- carrying the
seed in the key cannot tell the two apart -- yet a different shuffle and a
different model. The exported script builds a fresh loader every time,
which is why it always agreed with Run 1.

The fixture dataset's only feature is the sample's own index, so a batch
says exactly which samples it holds and in what order.

The last test pins the same bug one node upstream, NOT fixed here: a cached
``Dataset`` carries its augmentation wrapper's call counters from one Run
into the next.
"""

from __future__ import annotations

import copy
import struct
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import TensorDataset

from app.core.cache import ExecutionCache
from app.core.execution_context import ExecutionContext
from app.core.graph_engine import execute_graph
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import registry

INDEX_DATASET = "_IndexFixtureDataset"
FEATURE_DATASET = "_FeatureFixtureDataset"
ORDER_RECORDER = "_BatchOrderRecorder"
SUM_RECORDER = "_SampleSumRecorder"

_SAMPLES = 12
_BATCH = 4
_EPOCHS = 3
_FEATURES = 4
_CLASSES = 3


class _IndexDatasetNode(BaseNode):
    """``TensorDataset(arange(N))``: every sample is its own index.

    Cacheable and input-free, like ``SyntheticSegmentation`` in the graph
    that found the bug, so Runs 2 and 3 serve it from the cache and the
    loader downstream of it is what decides the outcome.
    """

    NODE_NAME = INDEX_DATASET
    CATEGORY = "Test"
    DESCRIPTION = "Dataset whose samples are their own indices."

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="dataset", data_type=DataType.DATASET)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        return {"dataset": TensorDataset(torch.arange(_SAMPLES))}


class _BatchOrderRecorderNode(BaseNode):
    """Iterates the loader it is handed and reports the order, per epoch.

    Stands in for ``TrainingLoop``: non-cacheable like it, so it runs on
    every Run and reads whatever loader object the engine hands it.
    """

    NODE_NAME = ORDER_RECORDER
    CATEGORY = "Test"
    DESCRIPTION = "Records the sample order a DataLoader yields."
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="dataloader", data_type=DataType.DATALOADER)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="order", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        loader = inputs["dataloader"]
        return {"order": [[int(i) for (batch,) in loader for i in batch]
                          for _ in range(_EPOCHS)]}


class _FeatureDatasetNode(BaseNode):
    """A small supervised dataset with no randomness of its own."""

    NODE_NAME = FEATURE_DATASET
    CATEGORY = "Test"
    DESCRIPTION = "Fixed tensor dataset for the training-graph test."

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="dataset", data_type=DataType.DATASET)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        features = (torch.arange(_SAMPLES * _FEATURES, dtype=torch.float32)
                    .reshape(_SAMPLES, _FEATURES) / (_SAMPLES * _FEATURES))
        return {"dataset": TensorDataset(features,
                                         torch.arange(_SAMPLES) % _CLASSES)}


class _SampleSumRecorderNode(BaseNode):
    """Like the order recorder, but reports each sample's pixel sum.

    A random crop changes the sum, so two epochs, or two Runs, that cropped
    differently report different numbers.
    """

    NODE_NAME = SUM_RECORDER
    CATEGORY = "Test"
    DESCRIPTION = "Records the pixel sum of every sample a DataLoader yields."
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="dataloader", data_type=DataType.DATALOADER)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="sums", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        loader = inputs["dataloader"]
        return {"sums": [[float(image.sum()) for images, _ in loader
                          for image in images]
                         for _ in range(2)]}


_FIXTURE_NODES = {
    INDEX_DATASET: _IndexDatasetNode,
    FEATURE_DATASET: _FeatureDatasetNode,
    ORDER_RECORDER: _BatchOrderRecorderNode,
    SUM_RECORDER: _SampleSumRecorderNode,
}


@pytest.fixture(autouse=True)
def _fixture_nodes():
    registry._nodes.update(_FIXTURE_NODES)
    yield
    for name in _FIXTURE_NODES:
        registry._nodes.pop(name, None)


def _trigger(target: str) -> dict:
    return {"id": f"t_{target}", "source": "start", "target": target,
            "sourceHandle": "trigger", "targetHandle": "__trigger",
            "type": "trigger"}


def _edge(source: str, source_handle: str, target: str, target_handle: str) -> dict:
    return {"id": f"{source}.{source_handle}->{target}.{target_handle}",
            "source": source, "sourceHandle": source_handle,
            "target": target, "targetHandle": target_handle}


def _loader(shuffle: bool = True) -> dict:
    return {"id": "loader", "type": "DataLoader",
            "data": {"params": {"batch_size": _BATCH, "shuffle": shuffle,
                                "num_workers": 0}}}


def _order_graph() -> tuple[list[dict], list[dict]]:
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "data", "type": INDEX_DATASET, "data": {"params": {}}},
        _loader(),
        {"id": "record", "type": ORDER_RECORDER, "data": {"params": {}}},
    ]
    edges = [
        _trigger("data"),
        _edge("data", "dataset", "loader", "dataset"),
        _edge("loader", "dataloader", "record", "dataloader"),
    ]
    return nodes, edges


def _context(seed: int | None) -> ExecutionContext:
    # Persist weights off, as in the report: every Run builds its model
    # from scratch, so nothing but the loader can carry one Run into the next.
    return ExecutionContext(device="cpu", seed=seed, weights_persistent=False)


async def _run(graph, seed, cache=None, statuses=None) -> dict[str, Any]:
    nodes, edges = graph

    async def track(node_id, status, data):
        if statuses is not None and status in ("completed", "cached"):
            statuses.setdefault(node_id, []).append(status)

    return await execute_graph(copy.deepcopy(nodes), copy.deepcopy(edges),
                               context=_context(seed), cache=cache,
                               on_progress=track)


async def _order(seed, cache=None, statuses=None) -> list[list[int]]:
    outputs = await _run(_order_graph(), seed, cache, statuses)
    return outputs["record"]["order"]


# ── the bug ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_every_cached_rerun_shuffles_like_a_fresh_run():
    """Runs 2 and 3 on a shared cache must yield Run 1's order.

    A fresh run, with no cache, is what the exported script does: it builds
    every object anew from the seed. Every canvas Run has to match it.
    """
    fresh = await _order(0)
    cache = ExecutionCache()
    runs = [await _order(0, cache) for _ in range(3)]

    assert runs == [fresh, fresh, fresh], (
        "a seeded Run on a shared cache must shuffle exactly like a fresh "
        f"seeded run.\n  fresh: {fresh}\n" + "".join(
            f"  run {i}: {order}\n" for i, order in enumerate(runs, 1))
        + "A run that differs from the fresh one was handed a cached "
        "DataLoader whose generator earlier Runs had already advanced.")


# ── the controls: what has to stay true for the test above to mean much ──

@pytest.mark.asyncio
async def test_the_shuffle_still_depends_on_the_seed_and_moves_every_epoch():
    """The order is a function of the seed, and each epoch reshuffles.

    Without this, a loader that stopped shuffling, or one reseeded at every
    epoch, would pass the test above.
    """
    order = await _order(0)

    assert order != await _order(1), "seeds 0 and 1 gave the same order"
    assert len({tuple(epoch) for epoch in order}) == _EPOCHS, (
        f"every epoch must get its own permutation; got {order}")
    assert all(sorted(epoch) == list(range(_SAMPLES)) for epoch in order)


@pytest.mark.asyncio
async def test_the_dataset_upstream_is_still_served_from_cache():
    """The cache is really in play, and still serves what it safely can.

    The dataset is a cache hit on Runs 2 and 3 -- that is the path the bug
    lived on -- and the loader is rebuilt on every Run.
    """
    cache = ExecutionCache()
    statuses: dict[str, list[str]] = {}
    for _ in range(3):
        await _order(0, cache, statuses)

    assert statuses["data"] == ["completed", "cached", "cached"]
    assert statuses["loader"] == ["completed", "completed", "completed"], (
        "the DataLoader must be rebuilt on every Run; a 'cached' entry is a "
        f"Run that got an earlier Run's loader (statuses: {statuses})")


def test_no_cacheable_node_hands_out_a_dataloader(registry_with_nodes):
    """#254's port rule, for the DATALOADER port.

    ``test_cache_live_handle_nodes.py`` holds every node with a MODEL or an
    OPTIMIZER output to it. A loader is the same kind of object: whoever
    receives it iterates it and so spends its generator. Over the whole
    registry, built-ins and in-repo packs, so a pack's own loader node is
    caught here too.
    """
    offenders = []
    for name, node_cls in sorted(registry._nodes.items()):
        if (getattr(node_cls, "CATEGORY", None) == "Test"
                or not getattr(node_cls, "cacheable", True)):
            continue
        try:
            outputs = node_cls.define_outputs()
        except Exception:  # noqa: BLE001 - not this test's problem
            continue
        if any(port.data_type == DataType.DATALOADER for port in outputs):
            offenders.append(name)

    assert offenders == [], (
        f"cacheable nodes with a DATALOADER output: {offenders}. A cache hit "
        "replays a loader whose generator earlier Runs already advanced. "
        "Declare `cacheable = False`.")


# ── the case the e2e found, through the real training nodes ──────────────

_MODEL_SPEC = (
    '{"version":2,'
    '"nodes":['
    '{"id":"in","type":"Input","ports":[{"id":"p_x","name":"x"}]},'
    f'{{"id":"l1","type":"Linear","params":{{"in_features":{_FEATURES},"out_features":8}}}},'
    '{"id":"r1","type":"ReLU"},'
    f'{{"id":"l2","type":"Linear","params":{{"in_features":8,"out_features":{_CLASSES}}}}},'
    '{"id":"out","type":"Output","ports":[{"id":"p_y","name":"y"}]}'
    '],'
    '"edges":['
    '{"id":"e1","source":"in","sourceHandle":"p_x","target":"l1"},'
    '{"id":"e2","source":"l1","target":"r1"},'
    '{"id":"e3","source":"r1","target":"l2"},'
    '{"id":"e4","source":"l2","target":"out","targetHandle":"p_y"}'
    ']}'
)


def _training_graph() -> tuple[list[dict], list[dict]]:
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "data", "type": FEATURE_DATASET, "data": {"params": {}}},
        _loader(),
        {"id": "model", "type": "SequentialModel",
         "data": {"params": {"layers": _MODEL_SPEC}}},
        {"id": "opt", "type": "Optimizer",
         "data": {"params": {"type": "SGD", "lr": 0.1}}},
        {"id": "loss", "type": "Loss",
         "data": {"params": {"type": "CrossEntropyLoss"}}},
        {"id": "train", "type": "TrainingLoop",
         "data": {"params": {"epochs": _EPOCHS, "device": "cpu"}}},
    ]
    edges = [
        _trigger("data"), _trigger("model"), _trigger("loss"),
        _edge("data", "dataset", "loader", "dataset"),
        _edge("loader", "dataloader", "train", "dataloader"),
        _edge("model", "model", "opt", "model"),
        _edge("model", "model", "train", "model"),
        _edge("opt", "optimizer", "train", "optimizer"),
        _edge("loss", "loss_fn", "train", "loss_fn"),
    ]
    return nodes, edges


@pytest.mark.asyncio
async def test_a_training_graph_gives_the_same_loss_curve_on_every_run():
    """The reported symptom: the same seed, a different result on Run 2.

    Floats compared with ``==``: a seeded CPU run is bitwise reproducible
    (#134), so anything short of equal is the bug.
    """
    async def losses(cache=None) -> list[float]:
        outputs = await _run(_training_graph(), 0, cache)
        return outputs["train"]["losses"].tolist()

    fresh = await losses()
    cache = ExecutionCache()
    runs = [await losses(cache) for _ in range(3)]

    assert runs == [fresh, fresh, fresh], (
        "every Run of a seeded training graph must reproduce the fresh "
        f"run's loss curve.\n  fresh: {fresh}\n" + "".join(
            f"  run {i}: {curve}\n" for i, curve in enumerate(runs, 1)))


# ── the same bug one node upstream: pinned, not fixed here ───────────────

def _write_offline_mnist(root: Path, n_train: int = 8, n_test: int = 4) -> None:
    """IDX files torchvision's MNIST loader reads, so nothing is downloaded.

    The same trick as ``test_cache_dataset_fingerprint_scope.py``: with the
    four decompressed files present, ``download=True`` is a no-op.
    """
    raw = root / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for prefix, count in (("train", n_train), ("t10k", n_test)):
        pixels = bytes((i * 7 + 3) % 256 for i in range(count * 28 * 28))
        (raw / f"{prefix}-images-idx3-ubyte").write_bytes(
            struct.pack(">IIII", 2051, count, 28, 28) + pixels)
        (raw / f"{prefix}-labels-idx1-ubyte").write_bytes(
            struct.pack(">II", 2049, count) + bytes(i % 10 for i in range(count)))


def _augmentation_graph(data_dir: Path) -> tuple[list[dict], list[dict]]:
    """MNIST through RandomCrop -> ToTensor, an unshuffled loader, sums out."""
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "crop", "type": "RandomCrop",
         "data": {"params": {"size": 24, "padding": 0}}},
        {"id": "to_tensor", "type": "ToTensorTransform",
         "data": {"params": {}}},
        {"id": "data", "type": "Dataset",
         "data": {"params": {"name": "MNIST", "split": "train",
                             "data_dir": str(data_dir)}}},
        _loader(shuffle=False),
        {"id": "record", "type": SUM_RECORDER, "data": {"params": {}}},
    ]
    edges = [
        _trigger("crop"),
        _edge("crop", "transform", "to_tensor", "transform"),
        _edge("to_tensor", "transform", "data", "train_transform"),
        _edge("data", "dataset", "loader", "dataset"),
        _edge("loader", "dataloader", "record", "dataloader"),
    ]
    return nodes, edges


async def _sums(graph, cache=None) -> list[list[float]]:
    outputs = await _run(graph, 0, cache)
    return outputs["record"]["sums"]


@pytest.mark.asyncio
async def test_seeded_augmentation_reproduces_without_a_cache(tmp_path):
    """The control for the test below: the graph crops, and reproducibly."""
    _write_offline_mnist(tmp_path)
    graph = _augmentation_graph(tmp_path)
    fresh = await _sums(graph)

    assert fresh[0] != fresh[1], f"the two epochs cropped alike: {fresh}"
    assert await _sums(graph) == fresh


@pytest.mark.xfail(strict=True, reason=(
    "a cached Dataset/ImageFolderDataset keeps its SeededAugmentation call "
    "counters across Runs; cacheable = False would undo #144/#259, so the fix "
    "needs its own decision"))
@pytest.mark.asyncio
async def test_a_cached_dataset_augments_like_a_fresh_run(tmp_path):
    """Seeded augmentation must not depend on how many Runs came before.

    On a seeded run ``Dataset`` wraps a random chain in ``SeededAugmentation``
    (``transforms._base``), which numbers its calls per stream; in the main
    process (``num_workers=0``) the counter just keeps going. ``Dataset`` is
    cacheable (#144, pinned by #259), so Run 2 gets Run 1's dataset object
    with the counter where Run 1's epochs left it, and crops differently from
    a fresh run and from the exported script. ``ImageFolderDataset`` installs
    the same wrapper and is cacheable too. Strict, so the fix that makes this
    pass has to remove the marker.
    """
    _write_offline_mnist(tmp_path)
    graph = _augmentation_graph(tmp_path)
    fresh = await _sums(graph)
    cache = ExecutionCache()
    runs = [await _sums(graph, cache) for _ in range(3)]

    assert runs == [fresh, fresh, fresh], (
        "every Run must crop like a fresh seeded run.\n"
        f"  fresh: {fresh}\n" + "".join(
            f"  run {i}: {run}\n" for i, run in enumerate(runs, 1)))
