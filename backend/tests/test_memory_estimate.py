"""Memory estimates for a run's default budget (#794): the structural hash,
the static estimate, the budget source and ``POST /api/runs/estimate``.

The flow through a shared card (a run that declares no budget gets the
static estimate, ``source: "estimate"``) is in ``test_memory_admission.py``
next to #788's acceptance flow, which it reuses.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

import pytest

from app.core import memory_admission as mem
from app.core import memory_estimate as me

GIB = mem.GIB


def _layers(in_features: int = 8, hidden: int = 16, out: int = 4, *,
            x: int = 0) -> str:
    """A two-layer MLP as SequentialModel's ``layers`` (v2 spec). *x* moves
    the boxes, which the hash must not see."""
    return json.dumps({"version": 2, "nodes": [
        {"id": "in", "type": "Input", "position": {"x": x, "y": 0},
         "ports": [{"id": "p_x", "name": "x"}]},
        {"id": "l1", "type": "Linear", "position": {"x": x + 1, "y": 0},
         "params": {"in_features": in_features, "out_features": hidden}},
        {"id": "r1", "type": "ReLU"},
        {"id": "l2", "type": "Linear",
         "params": {"in_features": hidden, "out_features": out}},
        {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
    ], "edges": [
        {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "l1"},
        {"id": "e2", "source": "l1", "target": "r1"},
        {"id": "e3", "source": "r1", "target": "l2"},
        {"id": "e4", "source": "l2", "target": "out", "targetHandle": "p_y"},
    ]})


def _node(node_id: str, node_type: str, **params: Any) -> dict[str, Any]:
    return {"id": node_id, "type": node_type, "position": {"x": 0, "y": 0},
            "data": {"params": params}}


def _training_graph(*, optimizer: dict[str, Any] | None = None,
                    batch: int = 64, layers: str | None = None,
                    ids: str = "", seed: int = 0, epochs: int = 3,
                    extra: list[dict[str, Any]] | None = None
                    ) -> dict[str, Any]:
    """Start, SequentialModel + Optimizer + DataLoader(Dataset) + Loss into
    a TrainingLoop: a graph that prepares. *ids* prefixes every node id (a
    copy-paste)."""
    p = ids
    nodes = [
        _node(f"{p}start", "Start"),
        _node(f"{p}model", "SequentialModel", layers=layers or _layers()),
        _node(f"{p}opt", "Optimizer", **(optimizer or {"type": "Adam"})),
        _node(f"{p}data", "Dataset", name="MNIST"),
        _node(f"{p}loader", "DataLoader", batch_size=batch),
        _node(f"{p}loss", "Loss", type="MSELoss"),
        _node(f"{p}train", "TrainingLoop", epochs=epochs, seed=seed),
        *(extra or []),
    ]
    edges = [
        *({"id": f"{p}t{root}", "source": f"{p}start", "target": f"{p}{root}",
           "sourceHandle": "trigger", "type": "trigger"}
          for root in ("model", "data", "loss")),
        # Extra nodes are roots too: an unreachable node is pruned.
        *({"id": f"{p}t{node['id']}", "source": f"{p}start",
           "target": node["id"], "sourceHandle": "trigger", "type": "trigger"}
          for node in extra or ()),
        {"id": f"{p}a", "source": f"{p}model", "sourceHandle": "model",
         "target": f"{p}opt", "targetHandle": "model"},
        {"id": f"{p}b", "source": f"{p}data", "sourceHandle": "dataset",
         "target": f"{p}loader", "targetHandle": "dataset"},
        {"id": f"{p}c", "source": f"{p}model", "sourceHandle": "model",
         "target": f"{p}train", "targetHandle": "model"},
        {"id": f"{p}d", "source": f"{p}loader", "sourceHandle": "dataloader",
         "target": f"{p}train", "targetHandle": "dataloader"},
        {"id": f"{p}e", "source": f"{p}opt", "sourceHandle": "optimizer",
         "target": f"{p}train", "targetHandle": "optimizer"},
        {"id": f"{p}f", "source": f"{p}loss", "sourceHandle": "loss_fn",
         "target": f"{p}train", "targetHandle": "loss_fn"},
    ]
    return {"nodes": nodes, "edges": edges}


# ── the structural hash ──────────────────────────────────────────────────


def test_the_hash_ignores_ids_layout_seeds_and_run_length():
    graph = _training_graph()
    nodes, _ = me.executable(graph)
    # It prepares: the TrainingLoop has its defaults filled in.
    assert "precision" in next(n for n in nodes if n["type"] == "TrainingLoop"
                               )["data"]["params"]
    same = _training_graph(ids="copy-", seed=7, epochs=50,
                           layers=_layers(x=300))
    # Host-side loader workers do not change what the device holds.
    next(n for n in same["nodes"] if n["type"] == "DataLoader")["data"][
        "params"].update(num_workers=4, persistent_workers=True)
    assert me.structural_hash(graph) == me.structural_hash(same)
    assert mem.graph_hash(graph) == me.structural_hash(graph)


def test_the_hash_sees_what_shapes_memory():
    graph = _training_graph()
    for other in (_training_graph(batch=128),
                  _training_graph(layers=_layers(hidden=32)),
                  _training_graph(optimizer={"type": "SGD"})):
        assert me.structural_hash(other) != me.structural_hash(graph)
    # The device kind and count stay part of the key (#788 round 2).
    assert me.structural_hash(graph, {"device": "cuda:0"}) \
        != me.structural_hash(graph, {"device": "cuda:0", "target": {
            "devices": ["cuda:0", "cuda:1"]}})


def test_the_hash_reads_a_preset_as_it_executes():
    """A preset card hashes as its expanded nodes; one whose params differ
    is a different graph."""
    card = {"id": "pipe", "type": "preset:Training Pipeline",
            "data": {"params": {}, "internalParams": {
                "dataloader": {"batch_size": 64}}}}
    model = _node("model", "SequentialModel", layers=_layers())
    edges = [{"id": f"m{handle}", "source": "model", "sourceHandle": "model",
              "target": "pipe", "targetHandle": handle}
             for handle in ("model", "model_for_opt")]
    start = _node("start", "Start")
    edges.append({"id": "t", "source": "start", "target": "model",
                  "sourceHandle": "trigger", "type": "trigger"})
    graph = {"nodes": [start, model, card], "edges": edges}
    bigger = json.loads(json.dumps(graph))
    bigger["nodes"][2]["data"]["internalParams"]["dataloader"][
        "batch_size"] = 128
    assert me.structural_hash(graph) != me.structural_hash(bigger)
    nodes, _ = me.executable(graph)
    assert any(node["type"] == "TrainingLoop" for node in nodes)
    # Read as drawn (it does not prepare), the inner params still count.
    for drawn in (graph, bigger):
        drawn["edges"] = drawn["edges"][1:]
    assert me.structural_hash(graph) != me.structural_hash(bigger)


def test_an_unreadable_graph_is_hashed_as_drawn():
    graph = {"nodes": [{"id": "a", "type": "NoSuchNode",
                        "data": {"params": {"n": 1}}}], "edges": []}
    assert me.structural_hash(graph) == me.structural_hash(graph)
    assert me.structural_hash(graph) != me.structural_hash(
        {"nodes": [{"id": "a", "type": "NoSuchNode",
                    "data": {"params": {"n": 2}}}], "edges": []})


# ── the static estimate ──────────────────────────────────────────────────


def _real_count(layers: str) -> int:
    from app.nodes.utility.sequential_node import SequentialModelNode

    module = SequentialModelNode().build_module({"layers": layers})
    return sum(p.numel() for p in module.parameters())


@pytest.mark.parametrize(("optimizer", "states"), [
    ({"type": "Adam"}, 2), ({"type": "AdamW"}, 2),
    ({"type": "Adam", "amsgrad": True}, 3),
    ({"type": "SGD", "momentum": 0.9}, 1), ({"type": "SGD"}, 0),
    ({"type": "RMSprop"}, 1)])
def test_the_estimate_counts_weights_gradients_and_optimizer_state(
        optimizer, states):
    estimate = me.estimate_static(_training_graph(optimizer=optimizer))
    assert estimate is not None
    assert estimate.parameters == _real_count(_layers()) == 212
    assert estimate.trainable_parameters == 212
    assert estimate.training is True
    assert estimate.multiplier == 2 + states
    assert estimate.weights_bytes == 212 * 4 * (2 + states)
    assert estimate.batch_size == 64
    assert estimate.confidence == me.CONFIDENCE_MEDIUM
    assert estimate.notes == [] and estimate.uncounted == []
    # One sample is (8,): the first Linear's in_features.
    assert estimate.models[0].probe_input == [[8]]
    activations = (16 + 16 + 4) * 4 * me.TRAIN_ACTIVATION_FACTOR \
        + 4 * 4 * me.LOSS_COPIES
    assert estimate.activation_bytes == int(64 * activations)
    assert estimate.bytes == int(-(-(
        (estimate.weights_bytes + 64 * activations) * me.RESERVE_FACTOR
        + me.WORKSPACE_BYTES) // 1))


def test_the_activation_term_grows_with_the_batch():
    small = me.estimate_static(_training_graph(batch=8))
    large = me.estimate_static(_training_graph(batch=512))
    assert small is not None and large is not None
    assert large.activation_bytes == 64 * small.activation_bytes
    assert large.weights_bytes == small.weights_bytes
    assert large.activation_phase == "training"


def test_an_evaluation_at_a_larger_batch_can_be_the_peak():
    """An EvaluateModel at batch 4096 next to training at 64: no
    gradients, but 64 times the samples, so it sets the activation term and
    the breakdown says which pass it is."""
    graph = _training_graph(extra=[_node("eval", "EvaluateModel",
                                         batch_size=4096)])
    estimate = me.estimate_static(graph)
    assert estimate is not None
    assert (estimate.batch_size, estimate.eval_batch_size) == (64, 4096)
    assert estimate.activation_phase == "evaluation"
    per_sample = (16 + 16 + 4) * 4 * me.EVAL_ACTIVATION_FACTOR + 4 * 4
    assert estimate.activation_bytes == int(4096 * per_sample)
    assert estimate.describe()["activation_phase"] == "evaluation"


def test_a_conv_model_takes_its_image_side_from_the_data_nodes():
    """The first Conv2d says the channels; a 16-px image from the graph's
    SyntheticShapes is tried before the default sides."""
    layers = json.dumps({"version": 2, "nodes": [
        {"id": "in", "type": "Input", "ports": [{"id": "p", "name": "x"}]},
        {"id": "c", "type": "Conv2d", "params": {
            "in_channels": 1, "out_channels": 4, "kernel_size": 3,
            "padding": 1}},
        {"id": "f", "type": "Flatten"},
        {"id": "l", "type": "Linear", "params": {
            "in_features": 4 * 16 * 16, "out_features": 2}},
        {"id": "out", "type": "Output", "ports": [{"id": "o", "name": "y"}]},
    ], "edges": [
        {"id": "1", "source": "in", "sourceHandle": "p", "target": "c"},
        {"id": "2", "source": "c", "target": "f"},
        {"id": "3", "source": "f", "target": "l"},
        {"id": "4", "source": "l", "target": "out", "targetHandle": "o"},
    ]})
    estimate = me.estimate_static(_training_graph(
        layers=layers, extra=[_node("shapes", "SyntheticShapes",
                                    image_size=16)]))
    assert estimate is not None
    assert estimate.models[0].probe_input == [[1, 16, 16]]
    assert estimate.confidence == me.CONFIDENCE_MEDIUM


def _attention_layers(causal: bool) -> str:
    return json.dumps({"version": 2, "nodes": [
        {"id": "in", "type": "Input", "ports": [{"id": "p", "name": "x"}]},
        {"id": "a", "type": "MultiHeadAttention", "params": {
            "embed_dim": 16, "num_heads": 2, "causal": causal}},
        {"id": "out", "type": "Output", "ports": [{"id": "o", "name": "y"}]},
    ], "edges": [
        {"id": "1", "source": "in", "sourceHandle": "p", "target": "a"},
        {"id": "2", "source": "a", "target": "out", "targetHandle": "o"}]})


def test_attention_weights_count_only_where_they_are_materialised():
    """A plain MultiheadAttention call returns (and keeps) its batch x heads
    x T x T weights; the causal block asks ``need_weights=False`` (the fused
    kernel), so its estimate has no such term."""
    graph = _training_graph(extra=[_node("seq", "SyntheticSequence",
                                         seq_len=32)])
    plain = me.estimate_static({**graph, "nodes": [
        _node("model", "SequentialModel", layers=_attention_layers(False))
        if n["id"] == "model" else n for n in graph["nodes"]]})
    fused = me.estimate_static({**graph, "nodes": [
        _node("model", "SequentialModel", layers=_attention_layers(True))
        if n["id"] == "model" else n for n in graph["nodes"]]})
    assert plain is not None and fused is not None
    assert plain.models[0].probe_input == [[32, 16]]
    # Per sample: the output (32 x 16) in both; in the plain one also the
    # head-averaged weights it returns (32 x 32) and the per-head weights
    # 2 x 32 x 32 twice (the probabilities and what backward keeps).
    assert fused.models[0].activation_elements == 32 * 16
    assert plain.models[0].activation_elements == \
        32 * 16 + 32 * 32 + 2 * 2 * 32 * 32


def test_mixed_precision_halves_activations_but_not_weights():
    fp32 = me.estimate_static(_training_graph())
    graph = _training_graph()
    next(node for node in graph["nodes"] if node["type"] == "TrainingLoop"
         )["data"]["params"]["precision"] = "bf16"
    bf16 = me.estimate_static(graph)
    assert fp32 is not None and bf16 is not None
    assert bf16.precision == "bf16"
    assert bf16.weights_bytes == fp32.weights_bytes + 212 * 2
    assert bf16.activation_bytes < fp32.activation_bytes


def test_an_inference_graph_holds_the_weights_once():
    graph = {"nodes": [_node("model", "SequentialModel", layers=_layers())],
             "edges": []}
    estimate = me.estimate_static(graph)
    assert estimate is not None
    assert estimate.training is False and estimate.multiplier == 1
    assert estimate.weights_bytes == 212 * 4


def test_the_rl_models_are_counted_by_formula():
    """DQN, PPO and RewardModel build their nets in ``execute``; the
    formulas match what they build."""
    from app.nodes.rl.dqn_node import DQNNode
    from app.nodes.rl.ppo_node import PPONode
    from app.nodes.rl.reward_model_node import RewardModelNode

    cases = [(DQNNode, "DQN", {"state_dim": 6, "action_dim": 3,
                               "hidden_dim": 32}),
             (PPONode, "PPO", {"state_dim": 5, "action_dim": 2,
                               "hidden_dim": 16}),
             (RewardModelNode, "RewardModel", {"input_dim": 12,
                                               "hidden_dim": 7})]
    for node_cls, node_type, params in cases:
        full = {d.name: d.default for d in node_cls.define_params()}
        full.update(params)
        model = node_cls().execute({}, full)["model"]
        real = sum(p.numel() for p in model.parameters())
        estimate = me.estimate_static(
            {"nodes": [_node("m", node_type, **params)], "edges": []})
        assert estimate is not None, node_type
        assert estimate.parameters == real, node_type
        # No forward probe for these: no activation term, and it says so.
        assert estimate.activation_bytes == 0
        assert estimate.activation_phase == "none"


def test_a_causal_lm_is_counted_without_allocating_it():
    """A 1.4B-parameter configuration counts on the meta device: no
    storage, well under a second, and the probe reads the dataset's
    sequence length."""
    params = {"vocab_size": 50257, "d_model": 2048, "n_layers": 24,
              "n_heads": 16, "d_ff": 8192, "max_seq_len": 2048}
    graph = _training_graph(extra=[
        _node("lm", "CausalLMModel", **params),
        _node("tok", "LMTokenizedDataset", seq_len=512)])
    started = time.monotonic()
    estimate = me.estimate_static(graph)
    assert time.monotonic() - started < 20
    assert estimate is not None
    lm = next(m for m in estimate.models if m.node_type == "CausalLMModel")
    assert lm.parameters > 1_000_000_000
    assert lm.probe_input == [[512]]


def test_the_vla_and_diffusion_models_are_probed_with_their_own_inputs():
    """VLAModel takes one (image, tokens, actions) tuple and DiffusionUNet
    an image and a timestep: each has its recipe, so both get an
    activation term."""
    graph = _training_graph(extra=[
        _node("vla", "VLAModel", d_model=64, n_layers=1, n_heads=2,
              expert_layers=1, image_size=16, max_text_len=8),
        _node("unet", "DiffusionUNet", in_channels=1, channel_mult="1,2"),
        _node("shapes", "SyntheticShapes", image_size=16)])
    estimate = me.estimate_static(graph)
    assert estimate is not None
    probes = {m.node_type: m for m in estimate.models}
    assert probes["VLAModel"].probe_input == [[3, 16, 16], [8], [8, 2]]
    # The largest side the graph names (its MNIST Dataset: 28) first.
    assert probes["DiffusionUNet"].probe_input == [[1, 28, 28], []]
    assert all(m.activation_elements for m in probes.values())
    assert estimate.confidence == me.CONFIDENCE_MEDIUM


def test_a_model_that_cannot_be_counted_makes_the_estimate_low():
    """BradleyTerryTrain builds its reward net inside ``execute`` from the
    data's width: nothing to count it by."""
    graph = _training_graph(extra=[_node("bt", "BradleyTerryTrain")])
    estimate = me.estimate_static(graph)
    assert estimate is not None
    assert estimate.uncounted == ["BradleyTerryTrain"]
    assert estimate.confidence == me.CONFIDENCE_LOW
    assert any("BradleyTerryTrain" in note for note in estimate.notes)


def test_a_model_without_an_input_recipe_has_no_activation_term():
    """A layer graph whose input reaches no shaped layer (a PReLU has one
    parameter and no input size): the weights are counted, the activations
    are not, and the estimate says so."""
    layers = json.dumps({"version": 2, "nodes": [
        {"id": "in", "type": "Input", "ports": [{"id": "p", "name": "x"}]},
        {"id": "r", "type": "PReLU"},
        {"id": "out", "type": "Output", "ports": [{"id": "o", "name": "y"}]},
    ], "edges": [
        {"id": "1", "source": "in", "sourceHandle": "p", "target": "r"},
        {"id": "2", "source": "r", "target": "out", "targetHandle": "o"}]})
    estimate = me.estimate_static(_training_graph(layers=layers))
    assert estimate is not None
    assert estimate.parameters == 1
    assert estimate.models[0].activation_elements is None
    assert estimate.confidence == me.CONFIDENCE_LOW
    assert estimate.notes == [
        "SequentialModel: no input shape for an activation estimate"]


def test_a_graph_without_a_model_has_no_estimate():
    assert me.estimate_static({"nodes": [_node("x", "TensorCreate")],
                               "edges": []}) is None


def test_no_batch_size_is_assumed_and_says_so():
    graph = _training_graph()
    graph["nodes"] = [n for n in graph["nodes"] if n["id"] != "loader"]
    estimate = me.estimate_static(graph)
    assert estimate is not None
    assert estimate.batch_size == me.DEFAULT_BATCH
    assert estimate.batch_assumed is True
    assert estimate.describe()["batch_assumed"] is True
    assert estimate.confidence == me.CONFIDENCE_LOW


def test_the_static_estimate_runs_off_the_event_loop(monkeypatch):
    seen: list[str] = []
    real = me.estimate_static

    def recording(graph):
        seen.append(threading.current_thread().name)
        return real(graph)

    monkeypatch.setattr(me, "estimate_static", recording)
    estimate = asyncio.run(me.estimate_static_async(_training_graph()))
    assert estimate is not None
    assert seen and seen[0].startswith("memory-estimate")


def test_a_slow_or_failing_estimate_offers_none(monkeypatch):
    release = threading.Event()

    def slow(graph):
        release.wait(5)
        return None

    monkeypatch.setattr(me, "estimate_static", slow)
    try:
        assert asyncio.run(me.estimate_static_async(
            _training_graph(), timeout=0.05)) is None
    finally:
        release.set()

    def broken(graph):
        raise RuntimeError("boom")

    monkeypatch.setattr(me, "estimate_static", broken)
    assert asyncio.run(me.estimate_static_async(_training_graph())) is None


def test_a_build_that_fails_on_the_meta_device_is_not_counted(monkeypatch):
    from app.nodes.utility.sequential_node import SequentialModelNode

    def broken(self, params):
        raise RuntimeError("needs real storage")

    monkeypatch.setattr(SequentialModelNode, "build_module", broken)
    me._CACHE.clear()
    try:
        estimate = me.estimate_static(_training_graph(layers=_layers(9)))
    finally:
        me._CACHE.clear()
    assert estimate is not None
    assert estimate.uncounted == ["SequentialModel"]


# ── the budget source ────────────────────────────────────────────────────


class _History:
    def __init__(self, suggestion: mem.Suggestion | None) -> None:
        self.suggestion = suggestion

    async def suggest(self, graph, options):
        return self.suggestion


def _static(estimate: me.StaticEstimate | None):
    async def static(graph):
        return estimate
    return static


def _estimate(confidence: str, nbytes: int = 3 * GIB) -> me.StaticEstimate:
    return me.StaticEstimate(
        bytes=nbytes, confidence=confidence, parameters=1,
        trainable_parameters=1, training=True, optimizer="Adam",
        multiplier=4, batch_size=1, eval_batch_size=1, precision="fp32",
        weights_bytes=1, activation_bytes=1, workspace_bytes=1)


async def test_history_comes_first_then_a_usable_static_estimate():
    measured = mem.Suggestion(mem.Budget(2 * GIB, mem.SOURCE_HISTORY,
                                         from_run="r1"), peak_bytes=GIB)
    rows = [("r1", "2026-10-18T00:00:00Z", {"cuda:0": {
        "reserved_bytes": GIB}}), ("r0", "2026-10-17T00:00:00Z",
                                   {"mps": {"held_bytes": 5}})]
    seen: list[tuple[str, int]] = []

    async def peaks(digest, limit):
        seen.append((digest, limit))
        return rows

    source = me.EstimateBudgetSource(_History(measured), peaks,
                                     static=_static(_estimate("medium")))
    assert await source.suggest({}, {}) is measured
    explained = await source.explain({"nodes": []}, {"device": "cuda:0"})
    assert explained.suggestion is measured
    assert explained.confidence == me.CONFIDENCE_MEASURED
    # The MPS row is no peak (#788 review 4): left out.
    assert explained.history == [{"run_id": "r1", "peak_bytes": GIB,
                                  "finished_at": "2026-10-18T00:00:00Z"}]
    assert seen == [(mem.graph_hash({"nodes": []}, {"device": "cuda:0"}),
                     me.HISTORY_ROWS)]

    source = me.EstimateBudgetSource(_History(None), None,
                                     static=_static(_estimate("medium")))
    suggestion = await source.suggest({}, {})
    assert suggestion is not None
    assert suggestion.budget == mem.Budget(3 * GIB, mem.SOURCE_ESTIMATE)
    assert suggestion.peak_bytes is None
    explained = await source.explain({}, {})
    assert explained.confidence == me.CONFIDENCE_MEDIUM
    assert explained.history == []


async def test_a_low_estimate_is_reported_but_never_applied():
    source = me.EstimateBudgetSource(_History(None), None,
                                     static=_static(_estimate("low")))
    assert await source.suggest({}, {}) is None
    explained = await source.explain({}, {})
    assert explained.suggestion is None
    assert explained.static is not None
    assert explained.confidence == me.CONFIDENCE_LOW
    empty = me.EstimateBudgetSource(_History(None), None,
                                    static=_static(None))
    assert (await empty.explain({}, {})).confidence is None


async def test_a_history_read_that_fails_lists_nothing():
    async def broken(digest, limit):
        raise RuntimeError("database is locked")

    source = me.EstimateBudgetSource(_History(None), broken,
                                     static=_static(None))
    assert (await source.explain({}, {})).history == []


def test_the_breakdown_is_what_the_api_reports():
    estimate = me.estimate_static(_training_graph())
    assert estimate is not None
    described = estimate.describe()
    assert described["bytes"] == estimate.bytes
    assert described["confidence"] == "medium"
    assert described["models"] == [{
        "type": "SequentialModel", "parameters": 212,
        "activations_known": True, "probe_input": [[8]]}]
    json.dumps(described)
