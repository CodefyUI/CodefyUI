"""Regression tests for #253 - the second Run of a training graph must train.

A cache hit returns the RECORDED outputs dict without calling ``execute``
again. For ``TrainingLoop`` that means the weights are never updated and the
per-epoch metrics are never written, while the node still reports success:
the second Run of a shipped teaching example did no training at all and the
canvas said ``completed``.

Two things were wrong and either one alone keeps the bug alive.

* ``TrainingLoop`` inherited ``cacheable = True`` from ``BaseNode``. Training
  is a side effect on objects the node was handed, so the recorded outputs
  do not describe it -- the same shape as the writer nodes in #143.
  ``DiffusionTrainingLoop`` had already declared ``cacheable = False``; the
  generic loop had not.
* ``SequentialModel`` owns the weights, declared no inputs, and was
  cacheable. A cacheable node with no inputs is a ROOT that nothing can ever
  invalidate, so run 2 was handed run 1's *already trained* module straight
  out of the cache. Making only ``TrainingLoop`` non-cacheable therefore
  trades "no training at all" for "every run silently continues from the
  last run's weights", which is not obviously better. It now carries
  ``StatefulModuleMixin`` like every other weight-owning node, so the
  existing ``Persist weights between runs`` setting decides which of the two
  happens and the node says in its log which one it did.

Why these tests count real ``execute()`` calls instead of reading statuses:
the shipped examples wrap the training path in a preset, and a preset whose
internals were ALL cache hits still reports ``completed`` (#260). Status is
precisely the thing that lied here, so it is never the assertion.

Why the substituted dataset does not weaken the end-to-end test: on the
shipped graph the only thing standing between ``TrainingLoop`` and a cache
hit today is an accident -- ``ModelSaver`` may only write under ``./data``,
and ``DatasetNode.cache_fingerprint`` walks all of ``./data``, so the write
dirties the dataset's key. #259 is filed to scope that walk, which would
remove the accident. The fixture dataset below has no external state at all,
so it is *more* cacheable than the shipped ``Dataset``, never less: this
graph is fully cacheable end to end and the fix has to hold on its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.core.cache import ExecutionCache
from app.core.execution_context import ExecutionContext
from app.core.graph_engine import execute_graph, expand_presets
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import registry
from app.core.node_state_store import NodeStateStore
from app.nodes.training.diffusion_training_loop_node import DiffusionTrainingLoopNode
from app.nodes.training.training_loop_node import TrainingLoopNode
from app.nodes.utility.sequential_node import SequentialModelNode

#: The example #253 was finally measured on: a plain "train an MLP and plot
#: the loss curve" graph from the teaching pack, with no writer node anywhere
#: and the training path inside a preset. Four of the six shipped graphs that
#: train have that shape.
_SHIPPED_EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "plugins" / "foundations" / "examples"
    / "C2-5" / "MLP-MNIST-Training" / "graph.json"
)

FIXTURE_NODE = "_MnistShapedFixtureDataset253"

#: MNIST's shape and label range exactly, so the shipped 784 -> 128 -> 64 ->
#: 10 layer spec runs verbatim. Tiny enough that one epoch is milliseconds.
_SAMPLES = 32
_CLASSES = 10


class _MnistShapedFixtureDatasetNode(BaseNode):
    """A deterministic stand-in for the MNIST download.

    Emits ``[N, 1, 28, 28]`` float images and integer labels -- the shapes
    the shipped graph's ``Flatten`` -> ``Linear(784, 128)`` and
    ``CrossEntropyLoss`` already expect, so nothing downstream is adjusted
    for it. No file access and no ``cache_fingerprint``, which is the point:
    it cannot accidentally invalidate anything, so the graph it feeds is
    cacheable from end to end.
    """

    NODE_NAME = FIXTURE_NODE
    CATEGORY = "Data"
    DESCRIPTION = "Fixed MNIST-shaped tensor dataset for cache tests."

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="dataset", data_type=DataType.DATASET)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        images = (
            torch.arange(_SAMPLES * 28 * 28, dtype=torch.float32)
            .reshape(_SAMPLES, 1, 28, 28)
            / (_SAMPLES * 28 * 28)
        )
        targets = torch.arange(_SAMPLES) % _CLASSES
        return {"dataset": torch.utils.data.TensorDataset(images, targets)}


MODEL_SOURCE_NODE = "_CacheableModelSource253"


class _CacheableModelSourceNode(BaseNode):
    """A legitimately cacheable node that hands out a model.

    Stands in for ``ModelLoader`` / ``CheckpointLoader``, which #144
    deliberately keeps ``cacheable = True`` (with a content fingerprint over
    the file they read) so that resuming from a checkpoint does not re-read
    it every run. A test node rather than the real ones because those need a
    weights file on disk, and the file is not what is being tested -- only
    that ``TrainingLoop`` still runs when its model arrives from a node that
    is a cache hit.
    """

    NODE_NAME = MODEL_SOURCE_NODE
    CATEGORY = "Test"
    DESCRIPTION = "Cacheable source of a small model."
    cacheable = True

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="model", data_type=DataType.MODEL)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch.nn as nn

        return {"model": nn.Sequential(nn.Flatten(), nn.Linear(784, _CLASSES))}


@pytest.fixture(autouse=True)
def _register_fixture_dataset():
    registry._nodes[FIXTURE_NODE] = _MnistShapedFixtureDatasetNode
    registry._nodes[MODEL_SOURCE_NODE] = _CacheableModelSourceNode
    yield
    registry._nodes.pop(FIXTURE_NODE, None)
    registry._nodes.pop(MODEL_SOURCE_NODE, None)


def _node_name(node_cls: type[BaseNode]) -> str:
    return node_cls.NODE_NAME


def _shipped_graph_without_the_download() -> tuple[list[dict], list[dict]]:
    """The shipped example, preset expanded, MNIST swapped for tensors.

    Everything that #253 is about executes exactly as shipped: the real
    ``SequentialModel`` with the file's own 784 -> 128 -> 64 -> 10 layer
    JSON, the real ``DataLoader``, ``Optimizer``, ``Loss``, ``TrainingLoop``,
    ``Visualize`` and ``Print``, wired by the file's own edges and the
    preset's own edges.

    Two rewrites, both on the data side:

    * the preset is expanded through the production :func:`expand_presets`,
      because the ``Dataset`` node to swap lives inside it. #253's own
      measurements confirmed the wrapped and expanded forms behave
      identically -- the wrapper hides the cache decision (#260), it does not
      change it.
    * ``Dataset`` becomes the fixture node above, and one epoch is enough.
    """
    payload = json.loads(_SHIPPED_EXAMPLE.read_text(encoding="utf-8"))
    nodes, edges, _ = expand_presets(payload["nodes"], payload["edges"])

    swapped = False
    shrunk = False
    for node in nodes:
        if node["type"] == "Dataset":
            node["type"] = FIXTURE_NODE
            node["data"]["params"] = {}
            swapped = True
        elif node["type"] == "TrainingLoop":
            node["data"]["params"].update(epochs=1, device="cpu")
            shrunk = True
        elif node["type"] == "DataLoader":
            node["data"]["params"].update(batch_size=8, num_workers=0)

    # The shipped file is the fixture. If it stops containing the nodes this
    # test rewrites, the rewrite silently became a no-op and everything below
    # would still pass while testing a different graph.
    assert swapped, f"{_SHIPPED_EXAMPLE} no longer contains a Dataset node"
    assert shrunk, f"{_SHIPPED_EXAMPLE} no longer contains a TrainingLoop node"
    assert any(n["type"] == "SequentialModel" for n in nodes), (
        f"{_SHIPPED_EXAMPLE} no longer contains a SequentialModel node")
    return nodes, edges


def _only(nodes: list[dict], node_type: str) -> str:
    ids = [n["id"] for n in nodes if n["type"] == node_type]
    assert len(ids) == 1, f"expected exactly one {node_type}, found {ids}"
    return ids[0]


@pytest.mark.parametrize(
    "node_cls",
    [TrainingLoopNode, DiffusionTrainingLoopNode, SequentialModelNode],
    ids=_node_name,
)
def test_training_and_weight_owning_nodes_are_not_cacheable(
    node_cls: type[BaseNode],
) -> None:
    """Necessary but nowhere near sufficient -- see the end-to-end test.

    ``TrainingLoop`` mutates the model and optimizer it is handed and writes
    a metric series; ``SequentialModel`` owns the weights that get mutated.
    Neither is described by its recorded outputs, so neither may be replayed
    from the cache.
    """
    assert node_cls.cacheable is False, (
        f"{node_cls.NODE_NAME} owns or mutates trainable parameters, so a "
        "cache hit that skips execute() skips the part that mattered. It "
        "must opt out with `cacheable = False`."
    )


@pytest.mark.asyncio
async def test_shipped_example_trains_on_every_run() -> None:
    """The #253 repro: run the shipped graph twice against one cache.

    Counts real ``TrainingLoopNode.execute`` calls. Before the fix this was
    ``1`` then ``0`` -- run 2 did no training and reported success.
    """
    nodes, edges = _shipped_graph_without_the_download()

    train_calls: list[int] = []
    model_calls: list[int] = []
    counted = {"train": 0, "model": 0}

    real_train = TrainingLoopNode.execute
    real_model = SequentialModelNode.execute

    def counting_train(self, *args, **kwargs):
        counted["train"] += 1
        return real_train(self, *args, **kwargs)

    def counting_model(self, *args, **kwargs):
        counted["model"] += 1
        return real_model(self, *args, **kwargs)

    cache = ExecutionCache()
    statuses: dict[str, str] = {}

    async def track(node_id, status, data):
        if status in ("completed", "cached", "skipped", "error"):
            statuses[node_id] = status

    TrainingLoopNode.execute = counting_train
    SequentialModelNode.execute = counting_model
    try:
        for _ in range(2):
            before = counted["train"], counted["model"]
            await execute_graph(nodes, edges, on_progress=track, cache=cache)
            train_calls.append(counted["train"] - before[0])
            model_calls.append(counted["model"] - before[1])
    finally:
        TrainingLoopNode.execute = real_train
        SequentialModelNode.execute = real_model

    assert train_calls == [1, 1], (
        "every Run of a training graph must actually train. TrainingLoop "
        f"executed {train_calls} time(s) per run; a 0 means that run was "
        "served from the cache and silently skipped training."
    )
    assert model_calls == [1, 1], (
        "SequentialModel owns the weights TrainingLoop mutates, so it may "
        f"not be replayed from the cache either (executed {model_calls} "
        "time(s) per run). A cached hand-back is run 1's trained module."
    )

    # Without this the test would pass just as well in a world where caching
    # had stopped working entirely and every node always re-executed. The
    # Loss node is pure, cacheable and unchanged between the two runs, so it
    # MUST be a cache hit on run 2 if the cache is really discriminating --
    # which is the property #253 depends on. It also pins that the fixture
    # dataset really did leave this graph cacheable end to end, i.e. that
    # the assertions above are not riding on an invalidation somewhere
    # upstream (the accident #259 is filed to remove).
    loss_id = _only(nodes, "Loss")
    dataset_id = _only(nodes, FIXTURE_NODE)
    assert statuses[loss_id] == "cached", (
        "the pure Loss node must still hit the cache on the second run -- "
        "otherwise this test cannot tell '#253 fixed' apart from 'caching "
        f"stopped working entirely' (status was {statuses[loss_id]!r})"
    )
    assert statuses[dataset_id] == "cached", (
        "the fixture dataset must hit the cache on the second run, proving "
        "TrainingLoop re-ran on its own merits rather than because an "
        f"upstream node invalidated it (status was {statuses[dataset_id]!r})"
    )


@pytest.mark.asyncio
async def test_training_still_runs_when_the_model_source_is_cacheable() -> None:
    """``TrainingLoop`` must not rely on its upstream being non-cacheable.

    ``SequentialModel`` going non-cacheable propagates down to the loop for
    the graphs in the shipped examples, which is enough to hide whether the
    loop's OWN ``cacheable`` flag matters. It does: ``ModelLoader`` and
    ``CheckpointLoader`` are deliberately cacheable (#144, so that resuming
    from a checkpoint does not re-read the file every run), and a graph that
    trains a model loaded from disk is a fully cacheable chain from end to
    end. ``_CacheableModelSource`` below stands in for those two -- what
    matters is only that the model arrives from a cacheable node.
    """
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "src", "type": MODEL_SOURCE_NODE, "data": {"params": {}}},
        {"id": "data", "type": FIXTURE_NODE, "data": {"params": {}}},
        {"id": "loader", "type": "DataLoader",
         "data": {"params": {"batch_size": 8, "shuffle": False,
                             "num_workers": 0}}},
        {"id": "opt", "type": "Optimizer",
         "data": {"params": {"type": "SGD", "lr": 0.05}}},
        {"id": "loss", "type": "Loss",
         "data": {"params": {"type": "CrossEntropyLoss"}}},
        {"id": "train", "type": "TrainingLoop",
         "data": {"params": {"epochs": 1, "device": "cpu"}}},
    ]
    edges = [
        {"id": "t1", "source": "start", "target": "src",
         "sourceHandle": "trigger", "type": "trigger"},
        {"id": "t2", "source": "start", "target": "data",
         "sourceHandle": "trigger", "type": "trigger"},
        {"id": "t3", "source": "start", "target": "loss",
         "sourceHandle": "trigger", "type": "trigger"},
        {"id": "e1", "source": "data", "target": "loader",
         "sourceHandle": "dataset", "targetHandle": "dataset"},
        {"id": "e2", "source": "loader", "target": "train",
         "sourceHandle": "dataloader", "targetHandle": "dataloader"},
        {"id": "e3", "source": "src", "target": "opt",
         "sourceHandle": "model", "targetHandle": "model"},
        {"id": "e4", "source": "src", "target": "train",
         "sourceHandle": "model", "targetHandle": "model"},
        {"id": "e5", "source": "opt", "target": "train",
         "sourceHandle": "optimizer", "targetHandle": "optimizer"},
        {"id": "e6", "source": "loss", "target": "train",
         "sourceHandle": "loss_fn", "targetHandle": "loss_fn"},
    ]

    counted = {"train": 0}
    per_run: list[int] = []
    real_train = TrainingLoopNode.execute

    def counting_train(self, *args, **kwargs):
        counted["train"] += 1
        return real_train(self, *args, **kwargs)

    cache = ExecutionCache()
    statuses: dict[str, str] = {}

    async def track(node_id, status, data):
        if status in ("completed", "cached", "skipped", "error"):
            statuses[node_id] = status

    TrainingLoopNode.execute = counting_train
    try:
        for _ in range(2):
            before = counted["train"]
            await execute_graph(nodes, edges, on_progress=track, cache=cache)
            per_run.append(counted["train"] - before)
    finally:
        TrainingLoopNode.execute = real_train

    assert per_run == [1, 1], (
        "TrainingLoop must execute on every run even when every node feeding "
        f"it is cacheable; it ran {per_run} time(s) per run."
    )
    assert statuses["src"] == "cached", (
        "the cacheable model source must still be a cache hit on run 2, or "
        "this test is not exercising the case it exists for "
        f"(status was {statuses['src']!r})"
    )


@pytest.mark.asyncio
async def test_concurrent_roots_persist_under_their_own_node_ids() -> None:
    """Weights must be filed under the node that owns them.

    A prerequisite for the fix above rather than a separate feature. Nodes
    at one topological level run concurrently on an UNSEEDED run (a seeded
    run is serial, #134), and the engine used to hand every one of them the
    same ``ExecutionContext`` after setting ``current_node_id`` on it. With
    an ``await`` between that assignment and the worker thread's read, the
    node that acquired the semaphore next won, and ``StatefulModuleMixin``
    persisted one node's weights under another node's id -- so 'Reset all
    weights now' could not reach them and two nodes could collide.

    Latent until now because the stateful nodes that existed were mid-chain,
    one per level. ``SequentialModel`` is a root, and several models on one
    canvas all start at level 0.
    """
    spec = json.loads(
        json.loads(_SHIPPED_EXAMPLE.read_text(encoding="utf-8"))
        ["nodes"][1]["data"]["params"]["layers"]
    )
    model_ids = [f"model-{i}" for i in range(4)]
    nodes = [{"id": "start", "type": "Start", "data": {"params": {}}}] + [
        {"id": mid, "type": "SequentialModel",
         # A different width per node, so a swap cannot be masked by the
         # two modules happening to hash the same.
         "data": {"params": {"layers": json.dumps(_widen(spec, 16 + i))}}}
        for i, mid in enumerate(model_ids)
    ]
    edges = [
        {"id": f"t{i}", "source": "start", "target": mid,
         "sourceHandle": "trigger", "type": "trigger"}
        for i, mid in enumerate(model_ids)
    ]

    store = NodeStateStore()
    context = ExecutionContext(
        graph_id="test-253-race",
        device="cpu",
        weights_persistent=True,
        node_state_store=store,
        # No seed on purpose: that is what lets the engine run these four
        # roots concurrently, which is the whole point of the test.
        seed=None,
    )
    await execute_graph(nodes, edges, context=context)

    stored_ids = sorted(key[1] for key in store._store)
    assert stored_ids == sorted(model_ids), (
        "each model's weights must be persisted under its own node id; got "
        f"{stored_ids}. A missing or duplicated id means one node's context "
        "was overwritten by a concurrent sibling."
    )


def _widen(spec: dict, hidden: int) -> dict:
    """The shipped layer spec with a different hidden width."""
    out = json.loads(json.dumps(spec))
    for node in out["nodes"]:
        params = node.get("params") or {}
        if "out_features" in params and params["out_features"] != 10:
            params["out_features"] = hidden
        if "in_features" in params and params["in_features"] != 784:
            params["in_features"] = hidden
    return out


@pytest.mark.asyncio
@pytest.mark.parametrize("persistent", [True, False], ids=["persist", "fresh"])
async def test_rerun_says_whether_it_continued_or_started_over(
    persistent: bool,
) -> None:
    """Whichever a rerun does, the user has to be able to tell.

    "Run again" continuing from the last run's weights is not wrong, but it
    is wrong to do it silently -- the losses look plausible either way, which
    is what made #253 hard to see. ``SequentialModel`` now routes the choice
    through the existing ``Persist weights between runs`` setting and reports
    in its ``__log__`` which of the two happened.
    """
    nodes, edges = _shipped_graph_without_the_download()
    model_id = _only(nodes, "SequentialModel")

    store = NodeStateStore()
    cache = ExecutionCache()
    notes: list[str] = []

    for _ in range(2):
        logs: dict[str, str] = {}

        async def track(node_id, status, data, _logs=logs):
            if isinstance(data, dict) and "__log__" in data:
                _logs[node_id] = str(data["__log__"])

        context = ExecutionContext(
            graph_id="test-253",
            device="cpu",
            weights_persistent=persistent,
            node_state_store=store,
        )
        await execute_graph(
            nodes, edges, on_progress=track, context=context, cache=cache)
        assert model_id in logs, (
            "SequentialModel must report which weights it used; nothing "
            "reached the node's log")
        notes.append(logs[model_id])

    assert notes[0].startswith("Fresh initialisation"), notes[0]
    if persistent:
        assert notes[1].startswith("Continuing from weights"), (
            "with weight persistence on, the second Run continues training "
            "from the first Run's weights and has to say so, or the loss "
            f"curve is unexplainable (log line was {notes[1]!r})")
        assert len(store) == 1, (
            "the persisted module should be in the node state store, where "
            "'Reset all weights now' can reach it")
    else:
        assert notes[1].startswith("Fresh initialisation"), (
            "with weight persistence off, every Run must re-initialise, and "
            f"say so (log line was {notes[1]!r})")
        assert len(store) == 0, (
            "nothing may be persisted when the setting is off")
