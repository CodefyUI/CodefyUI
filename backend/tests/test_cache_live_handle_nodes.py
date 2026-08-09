"""Regression tests for #254 - a cache hit must not skip a mutation.

A cache hit returns a node's RECORDED OUTPUTS without calling ``execute``
at all. #143 established that this is wrong for a node whose product is a
file write, and #253 that it is wrong for a node whose product is a change
to the weights it was handed. #254 is the same defect one layer over: five
nodes that are handed a live ``model`` / ``optimizer``, write into it, and
hand it back. The return value looks like a complete description of what
they did and is not -- the mutation is what mattered, and on a hit it does
not happen.

Two rules come out of it, and both are asserted registry-wide below so the
next node to break them fails here rather than in someone's loss curve.

**Live handles.** A node that hands out a ``MODEL`` or an ``OPTIMIZER``
cannot be replayed from its recorded output. The cache key describes how
that object was BUILT -- architecture, hyperparameters, upstream keys --
and stops describing it the moment training mutates it in place. That is
the rule #253 settled for weight-owning nodes ("a node that owns trainable
parameters is not cacheable"), stated in terms of the port type so it also
covers nodes that merely pass the handle along. ``LRScheduler`` is the same
kind of object (``TrainingLoop`` steps it once per epoch, so ``last_epoch``
advances during the run) but its port is typed ``ANY``, so it is named
explicitly rather than caught by the port rule.

**Side effects.** A node whose ``execute`` records a metric or writes a
checkpoint must not be cacheable, because that call is not in the return
value. #143 covered the file writers by hand; #254 asked for the invariant
instead, and ``EvaluateModel`` is what it catches -- its
``eval_accuracy`` point vanished on a hit, so the chart was empty on a run
the user was told succeeded.

Why the end-to-end tests use a purpose-built cacheable model source: the
engine refuses to cache any node whose upstream is non-cacheable, so after
this change no in-tree graph can reach these hits at all -- which is the
point, not a reason to stop testing them. Before this change the registry
had three such sources (``DQN``, ``PPO``, ``RewardModel``: weight-owning,
no required input, cacheable -- ``SequentialModel``'s bug one package
over), they are fixed here too, and a plugin can add another tomorrow. The
stand-in keeps the five nodes' own flags under test rather than resting on
an invariant that lives somewhere else.

None of these tests read a node's status. Status is what lied in #253: a
preset whose internals were all cache hits still reported ``completed``
(#260), and a node served from cache reports ``cached``, which is only a
bug if you know it should not have been.
"""

from __future__ import annotations

import functools
import inspect
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from app.config import settings
from app.core.cache import ExecutionCache
from app.core.execution_context import ExecutionContext
from app.core.graph_engine import execute_graph
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import registry
from app.nodes.io.checkpoint_node import CheckpointLoaderNode
from app.nodes.io.inference_node import InferenceNode
from app.nodes.io.model_loader_node import ModelLoaderNode
from app.nodes.rl.dqn_node import DQNNode
from app.nodes.rl.ppo_node import PPONode
from app.nodes.rl.reward_model_node import RewardModelNode
from app.nodes.training.evaluate_model_node import EvaluateModelNode
from app.nodes.training.lr_scheduler_node import LRSchedulerNode
from app.nodes.training.optimizer_node import OptimizerNode

#: Objects that are references to mutable state rather than values. A node
#: handing one of these out is describing something the cache key cannot
#: see; see this module's docstring.
LIVE_HANDLE_TYPES = frozenset({DataType.MODEL, DataType.OPTIMIZER})

#: Calls whose effect is invisible in the return value, so a hit skips it.
#: Matched against the node CLASS's own source, which means a side effect
#: reached through a helper defined outside the class is not seen -- this
#: is a tripwire for the next ``EvaluateModel``, not a proof of purity.
SIDE_EFFECT_CALLS = ("log_metric(", "log_artifact(", "write_checkpoint(")

#: The five nodes #254 names, plus the four that made their hits reachable.
NODES_254 = [
    ModelLoaderNode, CheckpointLoaderNode, EvaluateModelNode, InferenceNode,
    LRSchedulerNode,
]
LIVE_HANDLE_SOURCES = [OptimizerNode, DQNNode, PPONode, RewardModelNode]

_SAMPLES = 32
_CLASSES = 10

FIXTURE_DATASET = "_LiveHandleFixtureDataset254"
MODEL_SOURCE = "_CacheableModelSource254"
OPTIMIZER_SOURCE = "_CacheableOptimizerSource254"

#: The module the stand-in source last built, so a test can read the object
#: the graph is really working on rather than a copy of it.
_LAST_MODEL: dict[str, Any] = {}


def _node_name(node_cls: type[BaseNode]) -> str:
    return node_cls.NODE_NAME


class _FixtureDatasetNode(BaseNode):
    """MNIST-shaped tensors, no file access, no fingerprint.

    Deliberately MORE cacheable than the real ``Dataset``: it cannot
    invalidate anything, so a graph built on it is cacheable end to end and
    the nodes under test have nothing upstream to hide behind.
    """

    NODE_NAME = FIXTURE_DATASET
    CATEGORY = "Test"
    DESCRIPTION = "Fixed MNIST-shaped tensor dataset for cache tests."

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="dataset", data_type=DataType.DATASET)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        images = (
            torch.arange(_SAMPLES * 28 * 28, dtype=torch.float32)
            .reshape(_SAMPLES, 1, 28, 28) / (_SAMPLES * 28 * 28)
        )
        targets = torch.arange(_SAMPLES) % _CLASSES
        return {"dataset": torch.utils.data.TensorDataset(images, targets)}


class _CacheableModelSourceNode(BaseNode):
    """A cacheable node that hands out a model -- see the module docstring."""

    NODE_NAME = MODEL_SOURCE
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
        torch.manual_seed(0)
        model = nn.Sequential(nn.Flatten(), nn.Linear(784, _CLASSES))
        _LAST_MODEL["model"] = model
        return {"model": model}


class _CacheableOptimizerSourceNode(BaseNode):
    """A cacheable node that hands out an optimizer.

    Needed for the same reason as the model source, one port over:
    ``LRScheduler`` and ``CheckpointLoader`` both take an ``optimizer``,
    and the real ``Optimizer`` node is non-cacheable as of #254 -- which
    would mask their OWN flags behind the engine's propagation rule. With
    this upstream they are the only thing standing between the graph and a
    cache hit, which is what makes their tests mean something.
    """

    NODE_NAME = OPTIMIZER_SOURCE
    CATEGORY = "Test"
    DESCRIPTION = "Cacheable source of an optimizer."
    cacheable = True

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="model", data_type=DataType.MODEL)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="optimizer", data_type=DataType.OPTIMIZER)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        return {"optimizer": torch.optim.SGD(inputs["model"].parameters(),
                                             lr=0.1)}


@pytest.fixture
def live_handle_nodes():
    """NOT autouse: the registry-wide invariants below must not see a
    deliberately-cacheable model source sitting in the registry."""
    registry._nodes[FIXTURE_DATASET] = _FixtureDatasetNode
    registry._nodes[MODEL_SOURCE] = _CacheableModelSourceNode
    registry._nodes[OPTIMIZER_SOURCE] = _CacheableOptimizerSourceNode
    yield
    registry._nodes.pop(FIXTURE_DATASET, None)
    registry._nodes.pop(MODEL_SOURCE, None)
    registry._nodes.pop(OPTIMIZER_SOURCE, None)


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    d = tmp_path / "data" / "models"
    d.mkdir(parents=True)
    monkeypatch.setattr(settings, "MODELS_DIR", d)
    return d


class _CallCounter:
    """Counts real ``execute`` calls without hiding the method's signature.

    ``functools.wraps`` is load-bearing, not tidiness: ``run_node`` decides
    whether to pass ``context`` and ``progress_callback`` by inspecting
    ``execute``'s signature, so a bare ``*args, **kwargs`` wrapper silently
    strips them and the node under test runs in a different mode than the
    one being measured.
    """

    def __init__(self, node_cls: type[BaseNode]) -> None:
        self.node_cls = node_cls
        self._real = node_cls.execute
        self.calls = 0

    def __enter__(self) -> "_CallCounter":
        real, box = self._real, self

        @functools.wraps(real)
        def counted(inner_self, *args, **kwargs):
            box.calls += 1
            return real(inner_self, *args, **kwargs)

        self.node_cls.execute = counted
        return self

    def __exit__(self, *exc) -> None:
        self.node_cls.execute = self._real


def _trigger(*targets: str) -> list[dict]:
    return [{"id": f"trig{i}", "source": "start", "target": target,
             "sourceHandle": "trigger", "type": "trigger"}
            for i, target in enumerate(targets)]


async def _run_three_times(nodes, edges, counters, context_factory=None):
    """Run the graph three times against ONE cache; return per-run counts."""
    cache = ExecutionCache()
    per_run: list[tuple[int, ...]] = []
    for _ in range(3):
        before = tuple(c.calls for c in counters)
        context = context_factory() if context_factory is not None else None
        await execute_graph(nodes, edges, cache=cache, context=context)
        per_run.append(tuple(c.calls - b for c, b in zip(counters, before)))
    return per_run


def _first_weight(model) -> float:
    return round(float(next(model.parameters()).flatten()[0].detach()), 6)


# ─────────────────────────────────────────────────────────────────────────
# 1. Registry-wide invariants -- the guard #254 asked for
# ─────────────────────────────────────────────────────────────────────────

def _production_nodes() -> list[tuple[str, type[BaseNode]]]:
    """Every registered node except the synthetic ones tests register.

    Test nodes are excluded by ``CATEGORY == "Test"`` rather than by name:
    several of them exist precisely to be a cacheable model source (this
    file's own, and ``test_cache_training_nodes.py``'s), so an invariant
    that included them would assert the opposite of what they are for.
    """
    return [(name, cls) for name, cls in sorted(registry._nodes.items())
            if getattr(cls, "CATEGORY", None) != "Test"]


def test_no_cacheable_node_hands_out_a_live_model_or_optimizer(
    registry_with_nodes,
) -> None:
    """The structural half of #254, over the whole registry.

    A hand audit found five nodes; this is what would have found them, and
    what will find the sixth. It runs over every built-in AND every in-repo
    plugin pack, which is how ``DQN``/``PPO``/``RewardModel`` -- not named
    in the issue -- came to be part of the same fix.
    """
    offenders = []
    for name, node_cls in _production_nodes():
        if not getattr(node_cls, "cacheable", True):
            continue
        try:
            outputs = node_cls.define_outputs()
        except Exception:  # noqa: BLE001 - a node that cannot describe
            continue      # itself is a different test's problem
        live = [p.name for p in outputs if p.data_type in LIVE_HANDLE_TYPES]
        if live:
            offenders.append(f"{name} (outputs {', '.join(live)})")

    assert offenders == [], (
        "these nodes are cacheable AND hand out a live model/optimizer: "
        + "; ".join(offenders)
        + ". A cache hit replays the recorded handle, which by then points "
        "at an object training has mutated -- run 2 gets run 1's trained "
        "network. Declare `cacheable = False` (#253, #254)."
    )


def test_no_cacheable_node_has_a_side_effect_its_outputs_do_not_carry(
    registry_with_nodes,
) -> None:
    """The behavioural half: a metric or a file is not in the return value.

    #143 caught the file writers by auditing three names by hand and #254
    asked for this instead, having found ``EvaluateModel`` -- whose
    ``eval_accuracy`` point simply stopped being written on a cache hit,
    leaving an empty chart on a run reported as successful.
    """
    offenders = []
    for name, node_cls in _production_nodes():
        if not getattr(node_cls, "cacheable", True):
            continue
        try:
            source = inspect.getsource(node_cls)
        except (OSError, TypeError):
            continue
        hits = [call for call in SIDE_EFFECT_CALLS if call in source]
        if hits:
            offenders.append(f"{name} (calls {', '.join(hits)})")

    assert offenders == [], (
        "these nodes are cacheable AND record something no output carries: "
        + "; ".join(offenders)
        + ". A cache hit skips execute() entirely, so the metric is never "
        "logged and the file is never written while the node reports "
        "success. Declare `cacheable = False` (#143, #254)."
    )


@pytest.mark.parametrize("node_cls", NODES_254 + LIVE_HANDLE_SOURCES,
                         ids=_node_name)
def test_the_named_nodes_are_not_cacheable(node_cls: type[BaseNode]) -> None:
    """Cheap, necessary, and nowhere near sufficient -- see section 2.

    Named one by one as well as caught by the invariants above because
    ``LRScheduler``'s output port is typed ``ANY``: no port-type rule can
    see it, and it is the node with the loudest measured failure.
    """
    assert node_cls.cacheable is False, (
        f"{node_cls.NODE_NAME} either mutates a live object it was handed "
        "or hands one out. A cache hit replays its recorded outputs without "
        "calling execute(), which skips exactly the part that mattered "
        "(#254)."
    )


# ─────────────────────────────────────────────────────────────────────────
# 2. End to end: three runs, one cache, count real execute() calls
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_model_loader_reloads_the_file_on_every_run(
    live_handle_nodes, models_dir,
) -> None:
    """The realistic shape: load pretrained weights, then fine-tune.

    Run 1 trains the module in place. Run 2 must put the FILE's weights
    back before training, not continue from where run 1 stopped. Measured
    before the fix: 1 / 0 / 0 execute() calls and a starting weight of
    0.05 -> 0.085 -> 0.157 -- three runs of "fine-tune from the checkpoint"
    that quietly became one long run.
    """
    pristine = nn.Sequential(nn.Flatten(), nn.Linear(784, _CLASSES))
    nn.init.constant_(pristine[1].weight, 0.05)
    nn.init.constant_(pristine[1].bias, 0.0)
    torch.save(pristine.state_dict(), models_dir / "pretrained.pt")

    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "src", "type": MODEL_SOURCE, "data": {"params": {}}},
        {"id": "loader", "type": "ModelLoader",
         "data": {"params": {"path": "pretrained.pt",
                             "load_mode": "state_dict", "device": "cpu"}}},
        {"id": "data", "type": FIXTURE_DATASET, "data": {"params": {}}},
        {"id": "dl", "type": "DataLoader",
         "data": {"params": {"batch_size": 8, "shuffle": False,
                             "num_workers": 0}}},
        {"id": "opt", "type": "Optimizer",
         "data": {"params": {"type": "SGD", "lr": 0.5}}},
        {"id": "loss", "type": "Loss",
         "data": {"params": {"type": "CrossEntropyLoss"}}},
        {"id": "train", "type": "TrainingLoop",
         "data": {"params": {"epochs": 1, "device": "cpu"}}},
    ]
    edges = _trigger("src", "data", "loss") + [
        {"id": "e0", "source": "src", "target": "loader",
         "sourceHandle": "model", "targetHandle": "model"},
        {"id": "e1", "source": "data", "target": "dl",
         "sourceHandle": "dataset", "targetHandle": "dataset"},
        {"id": "e2", "source": "dl", "target": "train",
         "sourceHandle": "dataloader", "targetHandle": "dataloader"},
        {"id": "e3", "source": "loader", "target": "opt",
         "sourceHandle": "model", "targetHandle": "model"},
        {"id": "e4", "source": "loader", "target": "train",
         "sourceHandle": "model", "targetHandle": "model"},
        {"id": "e5", "source": "opt", "target": "train",
         "sourceHandle": "optimizer", "targetHandle": "optimizer"},
        {"id": "e6", "source": "loss", "target": "train",
         "sourceHandle": "loss_fn", "targetHandle": "loss_fn"},
    ]

    from app.nodes.training.training_loop_node import TrainingLoopNode

    starting_weights: list[float] = []
    real_train = TrainingLoopNode.execute

    @functools.wraps(real_train)
    def watching_train(self, inputs, params, *args, **kwargs):
        starting_weights.append(_first_weight(inputs["model"]))
        return real_train(self, inputs, params, *args, **kwargs)

    TrainingLoopNode.execute = watching_train
    try:
        with _CallCounter(ModelLoaderNode) as loader_calls:
            per_run = await _run_three_times(nodes, edges, [loader_calls])
    finally:
        TrainingLoopNode.execute = real_train

    assert per_run == [(1,), (1,), (1,)], (
        f"ModelLoader ran {per_run} time(s) per run. A 0 means the load into "
        "the model was skipped, so that run trained from whatever the "
        "previous one left behind instead of from the file."
    )
    assert starting_weights == [0.05, 0.05, 0.05], (
        "every run must start from the weights in the file. Got "
        f"{starting_weights}; anything drifting away from 0.05 is run N-1's "
        "training silently carrying over."
    )


@pytest.mark.asyncio
async def test_checkpoint_loader_restores_on_every_run(
    live_handle_nodes, models_dir,
) -> None:
    """Resuming from a checkpoint is a mutation, and a hit skips it.

    The model is deliberately vandalised between runs, the way training
    would change it. Whatever happened, the run after it must come back to
    the checkpoint's value. Measured before the fix: 1 / 0 / 0 execute()
    calls and the vandalised weight surviving to the end.
    """
    from app.core.checkpoints import write_checkpoint

    saved = nn.Sequential(nn.Flatten(), nn.Linear(784, _CLASSES))
    nn.init.constant_(saved[1].weight, 0.05)
    write_checkpoint("resume.pt", saved,
                     torch.optim.SGD(saved.parameters(), lr=0.1), epoch=3)

    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "src", "type": MODEL_SOURCE, "data": {"params": {}}},
        # A cacheable optimizer source rather than the real Optimizer node:
        # that one is non-cacheable now, which would leave this graph
        # uncacheable for a reason that has nothing to do with the node
        # under test.
        {"id": "opt", "type": OPTIMIZER_SOURCE, "data": {"params": {}}},
        {"id": "ck", "type": "CheckpointLoader",
         "data": {"params": {"path": "resume.pt", "device": "cpu"}}},
    ]
    edges = _trigger("src") + [
        {"id": "e1", "source": "src", "target": "opt",
         "sourceHandle": "model", "targetHandle": "model"},
        {"id": "e2", "source": "src", "target": "ck",
         "sourceHandle": "model", "targetHandle": "model"},
        {"id": "e3", "source": "opt", "target": "ck",
         "sourceHandle": "optimizer", "targetHandle": "optimizer"},
    ]

    cache = ExecutionCache()
    per_run, restored = [], []
    with _CallCounter(CheckpointLoaderNode) as ck_calls:
        for _ in range(3):
            before = ck_calls.calls
            await execute_graph(nodes, edges, cache=cache)
            per_run.append(ck_calls.calls - before)
            restored.append(_first_weight(_LAST_MODEL["model"]))
            # Stand in for what a training loop would do to the same object.
            with torch.no_grad():
                _LAST_MODEL["model"][1].weight.fill_(99.0)

    assert per_run == [1, 1, 1], (
        f"CheckpointLoader ran {per_run} time(s) per run. A 0 means the "
        "restore did not happen, so that run continued from whatever state "
        "the model was already in."
    )
    assert restored == [0.05, 0.05, 0.05], (
        "every run must come back to the checkpoint's weights. Got "
        f"{restored}; a 99.0 is the previous run's leftover state surviving "
        "a resume that reported success."
    )


@pytest.mark.asyncio
async def test_lr_scheduler_starts_each_run_at_the_top_of_its_schedule(
    live_handle_nodes,
) -> None:
    """The loudest of the five, and the quietest to the user.

    ``TrainingLoop`` steps the scheduler once per epoch, so a replayed
    scheduler arrives already at the end of the schedule it ran last time.
    Measured before the fix with StepLR(step_size=1, gamma=0.1) over two
    epochs, on the real ``Optimizer`` node: 1 / 0 / 0 execute() calls and
    the loop receiving (last_epoch=0, lr=0.1), then (2, 0.001), then
    (4, 1e-05). Run 3 trained four orders of magnitude below the learning
    rate on the node.

    The graph below swaps that real ``Optimizer`` for a cacheable stand-in,
    because ``Optimizer`` is non-cacheable now and would otherwise be the
    reason this graph misses -- masking the flag under test. The assertion
    is therefore on ``last_epoch`` alone: the learning rate the loop sees
    also depends on the optimizer being rebuilt, which is
    ``test_the_optimizer_starts_each_run_without_the_last_one_s_momentum``'s
    job.
    """
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "src", "type": MODEL_SOURCE, "data": {"params": {}}},
        {"id": "data", "type": FIXTURE_DATASET, "data": {"params": {}}},
        {"id": "dl", "type": "DataLoader",
         "data": {"params": {"batch_size": 8, "shuffle": False,
                             "num_workers": 0}}},
        {"id": "opt", "type": OPTIMIZER_SOURCE, "data": {"params": {}}},
        {"id": "sched", "type": "LRScheduler",
         "data": {"params": {"type": "StepLR", "step_size": 1,
                             "gamma": 0.1}}},
        {"id": "loss", "type": "Loss",
         "data": {"params": {"type": "CrossEntropyLoss"}}},
        {"id": "train", "type": "TrainingLoop",
         "data": {"params": {"epochs": 2, "device": "cpu"}}},
    ]
    edges = _trigger("src", "data", "loss") + [
        {"id": "e1", "source": "data", "target": "dl",
         "sourceHandle": "dataset", "targetHandle": "dataset"},
        {"id": "e2", "source": "dl", "target": "train",
         "sourceHandle": "dataloader", "targetHandle": "dataloader"},
        {"id": "e3", "source": "src", "target": "opt",
         "sourceHandle": "model", "targetHandle": "model"},
        {"id": "e4", "source": "src", "target": "train",
         "sourceHandle": "model", "targetHandle": "model"},
        {"id": "e5", "source": "opt", "target": "sched",
         "sourceHandle": "optimizer", "targetHandle": "optimizer"},
        {"id": "e6", "source": "opt", "target": "train",
         "sourceHandle": "optimizer", "targetHandle": "optimizer"},
        {"id": "e7", "source": "sched", "target": "train",
         "sourceHandle": "scheduler", "targetHandle": "lr_scheduler"},
        {"id": "e8", "source": "loss", "target": "train",
         "sourceHandle": "loss_fn", "targetHandle": "loss_fn"},
    ]

    from app.nodes.training.training_loop_node import TrainingLoopNode

    as_received: list[Any] = []
    real_train = TrainingLoopNode.execute

    @functools.wraps(real_train)
    def watching_train(self, inputs, params, *args, **kwargs):
        as_received.append(
            getattr(inputs.get("lr_scheduler"), "last_epoch", None))
        return real_train(self, inputs, params, *args, **kwargs)

    TrainingLoopNode.execute = watching_train
    try:
        with _CallCounter(LRSchedulerNode) as sched_calls:
            per_run = await _run_three_times(nodes, edges, [sched_calls])
    finally:
        TrainingLoopNode.execute = real_train

    assert per_run == [(1,), (1,), (1,)], (
        f"LRScheduler ran {per_run} time(s) per run; a 0 means that run was "
        "handed the previous run's already-exhausted schedule."
    )
    assert as_received == [0, 0, 0], (
        "every run must begin at the top of its schedule; the loop received "
        f"last_epoch={as_received}. A non-zero is the previous run's "
        "position, i.e. a run that trains at an already-decayed rate."
    )


@pytest.mark.asyncio
async def test_the_optimizer_starts_each_run_without_the_last_one_s_momentum(
    live_handle_nodes,
) -> None:
    """An optimizer is a live handle too, and its state is the evidence.

    Momentum buffers, Adam's running averages and the step counts all live
    in ``optimizer.state``, which ``TrainingLoop`` fills in as it trains.
    Replaying the recorded handle hands the next run an optimizer that is
    already mid-flight, and its learning rate is wherever the last run's
    schedule left it.

    Not reachable through any in-tree graph today -- every node that hands
    out a model is non-cacheable, so ``Optimizer`` never gets the chance --
    which is why this test supplies the cacheable model source itself. The
    flag is what keeps it unreachable when a plugin adds one.
    """
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "src", "type": MODEL_SOURCE, "data": {"params": {}}},
        {"id": "data", "type": FIXTURE_DATASET, "data": {"params": {}}},
        {"id": "dl", "type": "DataLoader",
         "data": {"params": {"batch_size": 8, "shuffle": False,
                             "num_workers": 0}}},
        {"id": "opt", "type": "Optimizer",
         "data": {"params": {"type": "SGD", "lr": 0.1, "momentum": 0.9}}},
        {"id": "loss", "type": "Loss",
         "data": {"params": {"type": "CrossEntropyLoss"}}},
        {"id": "train", "type": "TrainingLoop",
         "data": {"params": {"epochs": 1, "device": "cpu"}}},
    ]
    edges = _trigger("src", "data", "loss") + [
        {"id": "e1", "source": "data", "target": "dl",
         "sourceHandle": "dataset", "targetHandle": "dataset"},
        {"id": "e2", "source": "dl", "target": "train",
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

    from app.nodes.training.training_loop_node import TrainingLoopNode

    as_received: list[tuple[int, float]] = []
    real_train = TrainingLoopNode.execute

    @functools.wraps(real_train)
    def watching_train(self, inputs, params, *args, **kwargs):
        optimizer = inputs["optimizer"]
        as_received.append((len(optimizer.state),
                            round(optimizer.param_groups[0]["lr"], 10)))
        return real_train(self, inputs, params, *args, **kwargs)

    TrainingLoopNode.execute = watching_train
    try:
        with _CallCounter(OptimizerNode) as opt_calls:
            per_run = await _run_three_times(nodes, edges, [opt_calls])
    finally:
        TrainingLoopNode.execute = real_train

    assert per_run == [(1,), (1,), (1,)], (
        f"Optimizer ran {per_run} time(s) per run; a 0 hands the run an "
        "optimizer the previous run was still using."
    )
    assert as_received == [(0, 0.1), (0, 0.1), (0, 0.1)], (
        "every run must start with a fresh optimizer at the configured "
        f"learning rate; the loop received (state entries, lr)={as_received}. "
        "A non-empty state is the previous run's momentum."
    )


@pytest.mark.asyncio
async def test_evaluate_model_logs_its_metric_on_every_run(
    live_handle_nodes,
) -> None:
    """The accuracy chart must have a point per run, and a current one.

    ``EvaluateModel``'s ``eval_accuracy`` goes out through
    ``context.log_metric``, which no output carries, so a cache hit drops
    it silently. The model is trained by a loop in the same graph, so the
    accuracy also has to be re-measured rather than replayed. Measured
    before the fix: 1 / 0 / 0 execute() calls and ONE point across three
    runs.
    """
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "src", "type": MODEL_SOURCE, "data": {"params": {}}},
        {"id": "data", "type": FIXTURE_DATASET, "data": {"params": {}}},
        {"id": "dl", "type": "DataLoader",
         "data": {"params": {"batch_size": 8, "shuffle": False,
                             "num_workers": 0}}},
        {"id": "opt", "type": "Optimizer",
         "data": {"params": {"type": "SGD", "lr": 0.5}}},
        {"id": "loss", "type": "Loss",
         "data": {"params": {"type": "CrossEntropyLoss"}}},
        {"id": "train", "type": "TrainingLoop",
         "data": {"params": {"epochs": 1, "device": "cpu"}}},
        {"id": "ev", "type": "EvaluateModel",
         "data": {"params": {"batch_size": 8, "device": "cpu"}}},
    ]
    edges = _trigger("src", "data", "loss") + [
        {"id": "e1", "source": "data", "target": "dl",
         "sourceHandle": "dataset", "targetHandle": "dataset"},
        {"id": "e2", "source": "dl", "target": "train",
         "sourceHandle": "dataloader", "targetHandle": "dataloader"},
        {"id": "e3", "source": "src", "target": "opt",
         "sourceHandle": "model", "targetHandle": "model"},
        {"id": "e4", "source": "src", "target": "train",
         "sourceHandle": "model", "targetHandle": "model"},
        {"id": "e5", "source": "opt", "target": "train",
         "sourceHandle": "optimizer", "targetHandle": "optimizer"},
        {"id": "e6", "source": "loss", "target": "train",
         "sourceHandle": "loss_fn", "targetHandle": "loss_fn"},
        # Wired from the SOURCE, not from the loop, so nothing upstream of
        # EvaluateModel is non-cacheable: its own flag is what is on test.
        {"id": "e7", "source": "src", "target": "ev",
         "sourceHandle": "model", "targetHandle": "model"},
        {"id": "e8", "source": "data", "target": "ev",
         "sourceHandle": "dataset", "targetHandle": "dataset"},
    ]

    logged: list[tuple[str, int]] = []

    class _RecordingContext(ExecutionContext):
        def log_metric(self, name, value, step, node_id=None):
            logged.append((str(name), int(step)))

    with _CallCounter(EvaluateModelNode) as eval_calls:
        per_run = await _run_three_times(
            nodes, edges, [eval_calls],
            context_factory=lambda: _RecordingContext(
                graph_id="test-254", device="cpu"),
        )

    assert per_run == [(1,), (1,), (1,)], (
        f"EvaluateModel ran {per_run} time(s) per run; a 0 is a run whose "
        "accuracy was never measured and never logged."
    )
    assert [name for name, _ in logged].count("eval_accuracy") == 3, (
        "each run must contribute one eval_accuracy point to the chart; got "
        f"{logged}. A missing point is a run the canvas called successful "
        "with nothing to show for it."
    )


@pytest.mark.asyncio
async def test_inference_reruns_when_the_model_it_reads_has_moved(
    live_handle_nodes,
) -> None:
    """A forward pass is a function of weights the cache key cannot see."""
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "src", "type": MODEL_SOURCE, "data": {"params": {}}},
        {"id": "data", "type": FIXTURE_DATASET, "data": {"params": {}}},
        {"id": "dl", "type": "DataLoader",
         "data": {"params": {"batch_size": 8, "shuffle": False,
                             "num_workers": 0}}},
        {"id": "opt", "type": "Optimizer",
         "data": {"params": {"type": "SGD", "lr": 0.5}}},
        {"id": "loss", "type": "Loss",
         "data": {"params": {"type": "CrossEntropyLoss"}}},
        {"id": "train", "type": "TrainingLoop",
         "data": {"params": {"epochs": 1, "device": "cpu"}}},
        {"id": "ti", "type": "TensorInput",
         "data": {"params": {"shape": "2,1,28,28", "fill": "zeros"}}},
        {"id": "inf", "type": "Inference", "data": {"params": {"device": "cpu"}}},
    ]
    edges = _trigger("src", "data", "loss", "ti") + [
        {"id": "e1", "source": "data", "target": "dl",
         "sourceHandle": "dataset", "targetHandle": "dataset"},
        {"id": "e2", "source": "dl", "target": "train",
         "sourceHandle": "dataloader", "targetHandle": "dataloader"},
        {"id": "e3", "source": "src", "target": "opt",
         "sourceHandle": "model", "targetHandle": "model"},
        {"id": "e4", "source": "src", "target": "train",
         "sourceHandle": "model", "targetHandle": "model"},
        {"id": "e5", "source": "opt", "target": "train",
         "sourceHandle": "optimizer", "targetHandle": "optimizer"},
        {"id": "e6", "source": "loss", "target": "train",
         "sourceHandle": "loss_fn", "targetHandle": "loss_fn"},
        {"id": "e7", "source": "src", "target": "inf",
         "sourceHandle": "model", "targetHandle": "model"},
        {"id": "e8", "source": "ti", "target": "inf",
         "sourceHandle": "tensor", "targetHandle": "input"},
    ]

    with _CallCounter(InferenceNode) as inf_calls:
        per_run = await _run_three_times(nodes, edges, [inf_calls])

    assert per_run == [(1,), (1,), (1,)], (
        f"Inference ran {per_run} time(s) per run. A 0 hands back the "
        "prediction of a network that has since been trained further -- "
        "measured before the fix, the true output moved 0.0937 -> -0.0309 "
        "-> 0.2178 while runs 2 and 3 replayed the first number."
    )


@pytest.mark.parametrize(
    "node_type, params",
    [("DQN", {"state_dim": 4, "action_dim": 2, "hidden_dim": 8}),
     ("PPO", {"state_dim": 4, "action_dim": 2, "hidden_dim": 8}),
     ("RewardModel", {"input_dim": 8, "hidden_dim": 4, "seed": 7})],
)
@pytest.mark.asyncio
async def test_a_network_constructor_hands_out_a_fresh_network_every_run(
    node_type: str, params: dict,
) -> None:
    """#253's bug, in the three RL constructors it was never applied to.

    Each of these builds an ``nn.Module`` and has no required input, so as
    a cacheable node it was a root nothing could invalidate: run 2 got run
    1's object back, weights and all. The vandalism between runs stands in
    for what a training loop does to the same object.

    Not in #254's list. They are here because they are the only nodes in
    the registry that made #254's cache hits reachable, so leaving them
    would have fixed the five consumers and left the source open.
    """
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "net", "type": node_type, "data": {"params": params}},
    ]
    edges = _trigger("net")

    cache = ExecutionCache()
    identities, weights = [], []
    for _ in range(3):
        outputs = await execute_graph(nodes, edges, cache=cache)
        model = outputs["net"]["model"]
        identities.append(id(model))
        weights.append(_first_weight(model))
        with torch.no_grad():
            next(model.parameters()).fill_(99.0)

    assert len(set(identities)) == 3, (
        f"{node_type} handed back the same module object on "
        f"{4 - len(set(identities))} of 3 runs. A cached network is the "
        "previous run's network, including whatever training did to it."
    )
    assert 99.0 not in weights, (
        f"{node_type} handed back a network carrying the previous run's "
        f"weights: {weights}")


# ─────────────────────────────────────────────────────────────────────────
# 3. The control: caching still works
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_pure_node_in_the_same_graph_is_still_cached(
    live_handle_nodes,
) -> None:
    """Every assertion above would also hold if caching were simply broken.

    ``Loss`` is pure, cacheable and unchanged between runs, so it MUST be a
    cache hit on runs 2 and 3. Without this, "the fix engaged" and "nothing
    is ever cached any more" are the same measurement.
    """
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "loss", "type": "Loss",
         "data": {"params": {"type": "CrossEntropyLoss"}}},
    ]
    edges = _trigger("loss")

    cache = ExecutionCache()
    statuses: list[str] = []

    async def track(node_id, status, data):
        if node_id == "loss" and status in ("completed", "cached"):
            statuses.append(status)

    for _ in range(3):
        await execute_graph(nodes, edges, on_progress=track, cache=cache)

    assert statuses == ["completed", "cached", "cached"], (
        f"the pure Loss node's statuses were {statuses}; caching itself has "
        "to still discriminate, or none of the counts above mean anything."
    )


def test_the_engine_still_refuses_to_cache_downstream_of_a_miss() -> None:
    """The propagation this fix leans on, stated as a fact about the engine.

    ``ModelLoader``/``CheckpointLoader`` giving up ``cacheable = True`` costs
    almost nothing PRECISELY because of this rule: a node with any
    non-cacheable upstream is never cached, and both of them take a model
    from a weight-owning node. If this comment in the engine ever stops
    being true, the cost of the trade changes and this test says so.
    """
    source = Path(inspect.getfile(execute_graph)).read_text(encoding="utf-8")
    assert "upstream_all_cached" in source and (
        "if cache is not None and node_cacheable and upstream_all_cached:"
        in source
    ), (
        "the engine no longer gates caching on every upstream being cached; "
        "re-measure what #254's non-cacheable flags cost before trusting the "
        "reasoning in this module's docstring."
    )
