"""The edu MLP nodes give the same numbers on every run, canvas and export alike.

The textbook's I2-3 graph stacks ``FFNLayer`` / ``ActivationLayer`` three
times into ``TrainAndEvaluate``. Measured on it before this change:

* Five edu nodes hand out a live ``MODEL`` and were cacheable, which #254
  forbids, and ``TrainAndEvaluate`` trained the layers it was handed IN
  PLACE. A rerun where the FFN chain was a cache hit and the trainer was not
  (``epochs`` edited) trained the already-trained layers again: 400 then 50
  epochs on one cache gave loss 0.0199, where a fresh 50-epoch run -- what
  the exported script prints -- gives 0.1238.
* With no run seed, the layers' initial weights came from whatever state the
  process RNG happened to be in, so the canvas varied run to run (accuracy
  1.0, then 0.95) and an exported script, a fresh process, printed something
  else again. With no seed they now come from a fixed seed per node: the
  numbers a seed-0 run gives. With a seed they follow it exactly as before.

The pack is registered here if the suite has not done it already.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from app.config import settings
from app.core.cache import ExecutionCache
from app.core.execution_context import ExecutionContext
from app.core.graph_engine import execute_graph
from app.core.node_base import DataType
from app.core.node_registry import registry
from app.core.plugin_loader import discover_plugin_nodes
from app.core.seeding import derive_seed

#: Short enough to keep a run in milliseconds, long enough that a stack
#: trained twice ends somewhere else than one trained once.
EPOCHS = 60

#: The edu nodes with a ``MODEL`` output.
MODEL_NODES = {"edu:FFNLayer", "edu:ActivationLayer", "edu:TrainAndEvaluate",
               "edu:Classifier", "edu:AdvancedClassifier"}

_EDU_LOCKFILE = {"schema": 1, "plugins": {
    "edu": {"source_kind": "builtin", "source": "edu", "enabled": True}}}


@pytest.fixture(scope="module", autouse=True)
def edu_pack(tmp_path_factory):
    """Register the bundled edu pack; take back only what this added."""
    before = set(registry.nodes)
    discover_plugin_nodes(
        registry, settings.PLUGINS_BUILTIN_DIR,
        # Never read: a builtin entry resolves inside the built-in root.
        tmp_path_factory.mktemp("phantom-user-root"),
        _EDU_LOCKFILE,
    )
    yield
    for key in set(registry.nodes) - before:
        if key.startswith("edu:"):
            registry._nodes.pop(key, None)


def _node(node_id: str, node_type: str, **params) -> dict:
    return {"id": node_id, "type": node_type, "data": {"params": params}}


def _edge(source: str, source_handle: str, target: str, target_handle: str) -> dict:
    return {"id": f"{source}.{source_handle}->{target}.{target_handle}",
            "source": source, "target": target,
            "sourceHandle": source_handle, "targetHandle": target_handle}


def _trigger(target: str) -> dict:
    return {"id": f"trigger->{target}", "source": "start", "target": target,
            "sourceHandle": "trigger", "targetHandle": "__trigger",
            "type": "trigger"}


def _i23(epochs: int = EPOCHS, lr: float = 0.05) -> tuple[list[dict], list[dict]]:
    """The textbook I2-3 complete graph without its two plots.

    A second ``Print`` shows the per-epoch losses: four decimals for every
    epoch tell two runs apart far more sharply than one accuracy does.
    """
    nodes = [
        _node("start", "Start"),
        _node("data", "SyntheticDataset", kind="circles", n_samples=200,
              noise=0.15, factor=0.5, seed=42),
        _node("split", "TrainTestSplit", test_size=0.2, seed=42, stratify=True),
        _node("ffn1", "edu:FFNLayer", in_features=2, out_features=16),
        _node("act1", "edu:ActivationLayer", function="relu"),
        _node("ffn2", "edu:FFNLayer", out_features=16),
        _node("act2", "edu:ActivationLayer", function="relu"),
        _node("ffn3", "edu:FFNLayer", out_features=16),
        _node("act3", "edu:ActivationLayer", function="relu"),
        _node("train", "edu:TrainAndEvaluate", epochs=epochs, lr=lr),
        _node("acc", "Accuracy"),
        _node("print", "Print", label=""),
        _node("print_losses", "Print", label="losses"),
    ]
    edges = [
        _trigger("data"),
        _trigger("ffn1"),
        _edge("data", "tensor", "split", "features"),
        _edge("data", "labels", "split", "labels"),
        _edge("ffn1", "model", "act1", "model"),
        _edge("act1", "model", "ffn2", "model"),
        _edge("ffn2", "model", "act2", "model"),
        _edge("act2", "model", "ffn3", "model"),
        _edge("ffn3", "model", "act3", "model"),
        _edge("act3", "model", "train", "model"),
        _edge("split", "x_train", "train", "x_train"),
        _edge("split", "y_train", "train", "y_train"),
        _edge("split", "x_test", "train", "x_query"),
        _edge("train", "predictions", "acc", "predictions"),
        _edge("split", "y_test", "acc", "labels"),
        _edge("acc", "accuracy", "print", "value"),
        _edge("train", "losses", "print_losses", "value"),
    ]
    return nodes, edges


async def _run(seed: int | None, epochs: int = EPOCHS, cache=None, lr: float = 0.05) -> dict:
    nodes, edges = _i23(epochs, lr)
    return await execute_graph(nodes, edges, context=ExecutionContext(seed=seed),
                               cache=cache)


def _losses(outputs: dict) -> list[float]:
    return outputs["train"]["losses"].tolist()


def _run_exported_script(script: str, tmp_path: Path, *args: str):
    """Run an export as a grader does: fresh ``-I`` process, empty user data.

    Its lockfile names ``edu``, which keeps this valid whether or not an
    exported script loads the bundled packs it needs by itself.
    """
    script_path = tmp_path / "exported_graph.py"
    script_path.write_text(script, encoding="utf-8")
    user_data_dir = tmp_path / "user-data"
    plugins_dir = user_data_dir / "plugins"
    plugins_dir.mkdir(parents=True)
    (plugins_dir / "installed.json").write_text(
        json.dumps(_EDU_LOCKFILE), encoding="utf-8")
    env = os.environ.copy()
    env["CODEFYUI_USER_DATA_DIR"] = str(user_data_dir)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-I", str(script_path), *args],
        cwd=tmp_path, env=env, capture_output=True, text=True,
        encoding="utf-8", timeout=120, check=False,
    )


# ── the cache ────────────────────────────────────────────────────────────

def test_no_edu_node_that_hands_out_a_model_is_cacheable():
    live = {
        name: node_cls for name, node_cls in registry.nodes.items()
        if name.startswith("edu:") and any(
            port.data_type in (DataType.MODEL, DataType.OPTIMIZER)
            for port in node_cls.define_outputs())
    }
    assert MODEL_NODES <= set(live), "the edu pack is not registered"

    cacheable = sorted(name for name, node_cls in live.items()
                       if getattr(node_cls, "cacheable", True))
    assert cacheable == [], (
        f"{cacheable} hand out a live model and are cacheable: a cache hit "
        "replays the recorded object, which by then may have been trained "
        "(#254). Declare `cacheable = False`.")


@pytest.mark.asyncio
async def test_a_rerun_with_fewer_epochs_trains_from_scratch():
    cache = ExecutionCache()
    await _run(0, epochs=EPOCHS, cache=cache)

    rerun = await _run(0, epochs=EPOCHS // 2, cache=cache)

    assert _losses(rerun) == _losses(await _run(0, epochs=EPOCHS // 2)), (
        "the second run trained layers the first run had already trained")


@pytest.mark.asyncio
async def test_a_seed_change_on_one_cache_gives_the_fresh_numbers():
    cache = ExecutionCache()
    await _run(0, cache=cache)

    changed = await _run(1, cache=cache)

    assert _losses(changed) == _losses(await _run(1))


def test_training_leaves_the_stack_it_was_handed_untouched():
    """The upstream stack, and what the Inspector recorded for it, stays put."""
    stack = registry.get("edu:FFNLayer")().execute(
        {}, {"in_features": 2, "out_features": 8})["model"]
    stack = registry.get("edu:ActivationLayer")().execute(
        {"model": stack}, {"function": "relu"})["model"]
    before = {name: value.clone() for name, value in stack.state_dict().items()}

    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    registry.get("edu:TrainAndEvaluate")().execute(
        {"model": stack, "x_train": x, "y_train": ["a", "b", "b", "a"],
         "x_query": x},
        {"epochs": 20, "lr": 0.1})

    after = stack.state_dict()
    moved = [name for name in before if not torch.equal(before[name], after[name])]
    assert moved == [], f"TrainAndEvaluate trained its input in place: {moved}"


# ── seeds ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_same_seed_repeats_and_another_seed_does_not():
    first = await _run(0)

    assert _losses(await _run(0)) == _losses(first)
    assert _losses(await _run(1)) != _losses(first)


@pytest.mark.asyncio
async def test_an_unseeded_run_repeats_whatever_the_process_rng_holds():
    """What differs between the canvas and an exported script is the
    process RNG's state, so that is what this varies."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        first = await _run(None)
        torch.manual_seed(2)
        second = await _run(None)

    assert _losses(second) == _losses(first)


@pytest.mark.asyncio
async def test_an_unseeded_run_gives_the_seed_0_numbers():
    unseeded = await _run(None)
    seeded = await _run(0)

    assert _losses(unseeded) == _losses(seeded)
    assert unseeded["acc"]["accuracy"] == seeded["acc"]["accuracy"]


def test_a_layer_built_without_a_context_keeps_its_constructor_draw():
    """A direct ``execute()`` has no node id to derive a seed from, and one
    shared fallback would give every such layer the same weights."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        expected = nn.Linear(2, 16)
        torch.manual_seed(5)
        built = registry.get("edu:FFNLayer")().execute(
            {}, {"in_features": 2, "out_features": 16})["model"][0]

    assert torch.equal(built.weight, expected.weight)
    assert torch.equal(built.bias, expected.bias)


@pytest.mark.parametrize("run_seed, seeded_as", [(7, 7), (None, 0)],
                         ids=["seed-7", "no-seed"])
@pytest.mark.asyncio
async def test_every_new_layer_starts_from_its_derived_seed(run_seed, seeded_as):
    """Bit for bit what a seeded run built before this change.

    The engine seeded the global RNG with ``derive_seed(seed, node id)``
    right before each node, and a new ``nn.Linear`` was the node's first
    draw -- so the reference is ``torch.manual_seed`` of that value followed
    by the constructor. A run with no seed gets the same with seed 0. At
    ``lr=0`` Adam leaves the trainer's output layer where it started.
    """
    outputs = await _run(run_seed, epochs=1, lr=0.0)

    def reference(node_id: str, fan_in: int, fan_out: int) -> nn.Linear:
        torch.manual_seed(derive_seed(seeded_as, node_id))
        return nn.Linear(fan_in, fan_out)

    with torch.random.fork_rng(devices=[]):
        expected = {
            "ffn1": reference("ffn1", 2, 16),
            "ffn2": reference("ffn2", 16, 16),
            "ffn3": reference("ffn3", 16, 16),
            "train": reference("train", 16, 2),
        }

    stack = outputs["act3"]["model"]
    built = {"ffn1": stack[0], "ffn2": stack[2], "ffn3": stack[4],
             "train": outputs["train"]["model"].net[-1]}
    differing = [node_id for node_id, layer in built.items()
                 if not (torch.equal(layer.weight, expected[node_id].weight)
                         and torch.equal(layer.bias, expected[node_id].bias))]
    assert differing == []


# ── canvas == export ─────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", [0, None], ids=["seed-0", "no-seed"])
@pytest.mark.asyncio
async def test_the_exported_script_prints_what_the_canvas_shows(seed, tmp_path):
    from app.core.codegen import generate_python

    nodes, edges = _i23()
    canvas = await execute_graph(copy.deepcopy(nodes), copy.deepcopy(edges),
                                 context=ExecutionContext(seed=seed))
    script = generate_python(copy.deepcopy(nodes), copy.deepcopy(edges),
                             name="i23", seed=seed)

    completed = _run_exported_script(script, tmp_path, "--device", "cpu")

    assert completed.returncode == 0, completed.stderr
    for print_id in ("print", "print_losses"):
        assert canvas[print_id]["__log__"] in completed.stdout, (
            f"the export printed {completed.stdout!r}; the canvas showed "
            f"{canvas[print_id]['__log__']!r}")
