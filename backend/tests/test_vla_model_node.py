"""VLAModel / VLARollout / VLAActionEval (#312), and the first REAL user of
TrainingLoop's nested-batch device path.

The training integration test at the bottom is a regression lock on a
contract nothing exercised before: ``((image, tokens, actions), target)``
batches move to the device through ``to_device``'s recursion, through all
of TrainingLoop's batch sites, with ``model(data)`` receiving the tuple as
one argument.
"""

import pickle

import pytest
import torch

from app.core.execution_context import ExecutionContext
from app.core.node_state_store import NodeStateStore
from app.nodes.training.training_loop_node import TrainingLoopNode
from app.nodes.vla.pushworld_demos_node import PushWorldDemosNode, encode_instruction
from app.nodes.vla.pushworld_env_node import PushWorldFactory
from app.nodes.vla.vla_action_eval_node import VLAActionEvalNode
from app.nodes.vla.vla_model_node import VLABehaviorLoss, VLAModelNode, VLAModule
from app.nodes.vla.vla_rollout_node import VLARolloutNode

_SIZE = 32  # divisible by the conv stem's 8 and quick to render

TINY = {
    "image_size": _SIZE,
    "d_model": 64,
    "n_layers": 1,
    "n_heads": 4,
    "expert_layers": 1,
    "chunk": 4,
    "action_dim": 2,
    "max_text_len": 48,
    "seed": 0,
}


def _build(**overrides):
    return VLAModelNode().execute({}, {**TINY, **overrides})


def _batch(batch_size=3, chunk=4):
    image = torch.rand(batch_size, 3, _SIZE, _SIZE)
    tokens = torch.stack(
        [encode_instruction("push the red puck to the blue target")]
        * batch_size)
    actions = torch.rand(batch_size, chunk, 2) * 2 - 1
    return image, tokens, actions


# ── forward contracts ───────────────────────────────────────────────────


def test_regression_forward_predicts_the_chunk_shape():
    model = _build(head_type="regression")["model"]
    image, tokens, actions = _batch()
    out = model((image, tokens, actions))
    assert out.shape == (3, 4, 2)
    assert torch.isfinite(out).all()


def test_flow_forward_returns_a_residual_of_chunk_shape():
    model = _build(head_type="flow_matching")["model"]
    image, tokens, actions = _batch()
    out = model((image, tokens, actions))
    assert out.shape == (3, 4, 2)
    assert torch.isfinite(out).all()


def test_regression_tolerates_a_two_tuple_and_flow_refuses_it():
    image, tokens, actions = _batch()
    regression = _build(head_type="regression")["model"]
    assert regression((image, tokens)).shape == (3, 4, 2)
    flow = _build(head_type="flow_matching")["model"]
    with pytest.raises(ValueError, match="action chunk inside data"):
        flow((image, tokens))


def test_forward_refuses_a_bare_tensor():
    model = _build()["model"]
    with pytest.raises(TypeError, match="one tuple"):
        model(torch.rand(3, 3, _SIZE, _SIZE))


def test_wrong_image_shape_is_a_clear_error():
    model = _build()["model"]
    _, tokens, actions = _batch()
    with pytest.raises(ValueError, match="image must be"):
        model((torch.rand(3, 1, _SIZE, _SIZE), tokens, actions))


# ── the loss the node emits ─────────────────────────────────────────────


def test_loss_fn_matches_the_head_and_the_flow_loss_is_the_residual_msq():
    result = _build(head_type="flow_matching")
    loss_fn = result["loss_fn"]
    assert isinstance(loss_fn, VLABehaviorLoss)
    assert loss_fn.head_type == "flow_matching"
    residual = torch.tensor([[1.0, -1.0], [3.0, 0.0]])
    ignored_targets = torch.zeros_like(residual)
    assert loss_fn(residual, ignored_targets).item() == pytest.approx(2.75)

    regression = _build(head_type="regression")["loss_fn"]
    outputs = torch.tensor([[1.0, 0.0]])
    targets = torch.tensor([[0.0, 0.0]])
    assert regression(outputs, targets).item() == pytest.approx(0.5)
    # never the classification gate's type
    assert not isinstance(
        regression, (torch.nn.CrossEntropyLoss, torch.nn.NLLLoss))


def test_flow_training_step_reduces_the_flow_loss():
    torch.manual_seed(0)
    model = _build(head_type="flow_matching")["model"]
    loss_fn = VLABehaviorLoss("flow_matching")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    image, tokens, actions = _batch(batch_size=16)
    losses = []
    for _ in range(30):
        out = model((image, tokens, actions))
        loss = loss_fn(out, actions)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.8


# ── predict_action ──────────────────────────────────────────────────────


def test_predict_action_shapes_for_both_heads():
    image, tokens, _ = _batch()
    for head in ("regression", "flow_matching"):
        model = _build(head_type=head)["model"]
        out = model.predict_action(image, tokens)
        assert out.shape == (3, 4, 2)
        assert torch.isfinite(out).all()


def test_flow_prediction_is_reproducible_under_a_seed():
    model = _build(head_type="flow_matching")["model"]
    image, tokens, _ = _batch(batch_size=1)
    torch.manual_seed(7)
    first = model.predict_action(image, tokens)
    torch.manual_seed(7)
    second = model.predict_action(image, tokens)
    assert torch.equal(first, second)


def test_predict_action_restores_training_mode():
    model = _build()["model"]
    model.train()
    image, tokens, _ = _batch(batch_size=1)
    model.predict_action(image, tokens)
    assert model.training


def test_flow_steps_refresh_without_a_rebuild():
    node, context = VLAModelNode(), _persisting_context()
    params = {**TINY, "head_type": "flow_matching", "flow_steps": 10}
    first = node.execute({}, params, context=context)["model"]
    second = node.execute(
        {}, {**params, "flow_steps": 3}, context=context)["model"]
    assert second is first  # runtime knob: same module...
    assert second.flow_steps == 3  # ...new behavior


# ── module hygiene ──────────────────────────────────────────────────────


def test_the_module_pickles_whole():
    """#283's lesson: module-scope classes or full_model saving breaks."""
    model = _build()["model"]
    clone = pickle.loads(pickle.dumps(model))
    assert isinstance(clone, VLAModule)
    image, tokens, actions = _batch(batch_size=1)
    ours = model((image, tokens, actions) if model.head_type != "regression"
                 else (image, tokens))
    del ours  # the point is that forward runs on the clone at all
    out = clone.predict_action(image, tokens)
    assert out.shape == (1, 4, 2)


def _persisting_context(**kwargs) -> ExecutionContext:
    return ExecutionContext(
        graph_id="g",
        weights_persistent=True,
        node_state_store=NodeStateStore(),
        current_node_id="vla",
        **kwargs,
    )


def test_persisted_weights_survive_and_structure_edits_drop_them():
    node, context = VLAModelNode(), _persisting_context()
    first = node.execute({}, dict(TINY), context=context)["model"]
    with torch.no_grad():
        first.head.weight.fill_(0.123)
    again = node.execute({}, dict(TINY), context=context)["model"]
    assert again is first
    wider = node.execute(
        {}, {**TINY, "d_model": 128, "n_heads": 4}, context=context)["model"]
    assert wider is not first


def test_patchify_stem_is_available_and_counts_differently():
    conv = _build(vision_stem="conv")["param_count"]
    patchify = _build(vision_stem="patchify", patch_size=8)["param_count"]
    assert conv != patchify
    model = _build(vision_stem="patchify", patch_size=8)["model"]
    image, tokens, actions = _batch()
    assert model((image, tokens, actions)).shape == (3, 4, 2)


def test_bad_geometry_is_refused():
    with pytest.raises(ValueError, match="divisible"):
        _build(image_size=30)  # conv stem needs %8
    with pytest.raises(ValueError, match="n_heads"):
        _build(d_model=65, n_heads=4)


# ── rollout + open-loop eval on a fresh (untrained) policy ──────────────


def _factory(**overrides):
    config = {"image_size": _SIZE, "n_distractors": 1, "max_steps": 30}
    config.update(overrides)
    return PushWorldFactory(**config)


def test_rollout_runs_and_reports(monkeypatch):
    model = _build()["model"]
    result = VLARolloutNode().execute(
        {"model": model, "env": _factory()},
        {"episodes": 3, "execute_k": 2, "max_steps": 15,
         "instruction_mode": "normal", "record_episodes": 2, "seed": 0,
         "device": "cpu"},
    )
    assert 0.0 <= result["success_rate"] <= 1.0
    assert result["avg_steps"] > 0
    frames = result["frames"]
    assert frames.dtype == torch.uint8
    assert frames.dim() == 4 and frames.shape[1:] == (3, _SIZE, _SIZE)
    assert frames.shape[0] > 0
    assert len(result["report"].splitlines()) == 3


def test_rollout_swapped_needs_a_distractor():
    model = _build()["model"]
    with pytest.raises(ValueError, match="distractor"):
        VLARolloutNode().execute(
            {"model": model, "env": _factory(n_distractors=0)},
            {"episodes": 1, "instruction_mode": "swapped", "device": "cpu"},
        )


def test_rollout_refuses_a_model_without_predict_action():
    with pytest.raises(TypeError, match="predict_action"):
        VLARolloutNode().execute(
            {"model": torch.nn.Linear(2, 2), "env": _factory()},
            {"episodes": 1, "device": "cpu"},
        )


def _tiny_demos(chunk=4):
    return PushWorldDemosNode().execute(
        {"env": _factory()},
        {"episodes": 2, "chunk": chunk, "demo_noise": 0.0,
         "holdout_episodes": 2, "video_episodes": 0, "seed": 0},
    )


def test_action_eval_reports_a_reproducible_mse():
    model = _build()["model"]
    holdout = _tiny_demos()["holdout"]
    params = {"max_samples": 16, "batch_size": 8, "seed": 3, "device": "cpu"}
    first = VLAActionEvalNode().execute(
        {"model": model, "dataset": holdout}, params)
    second = VLAActionEvalNode().execute(
        {"model": model, "dataset": holdout}, params)
    assert first["evaluated"] == 16
    assert first["action_mse"] > 0
    assert first["action_mse"] == pytest.approx(second["action_mse"])


def test_action_eval_refuses_an_empty_dataset():
    model = _build()["model"]
    empty = _tiny_demos()["holdout"]
    empty.images_u8 = empty.images_u8[:0]
    with pytest.raises(ValueError, match="empty"):
        VLAActionEvalNode().execute(
            {"model": model, "dataset": empty}, {"device": "cpu"})


# ── the nested-batch TrainingLoop contract, exercised for real ──────────


@pytest.mark.parametrize("head", ["regression", "flow_matching"])
def test_training_loop_trains_a_vla_on_nested_batches(head):
    """((image, tokens, actions), target) through the stock TrainingLoop:
    device recursion, model(data)-as-one-tuple, the emitted loss, and a
    val loader on the same nested shape. Fails before this wave if the
    recursion contract regresses."""
    result = _build(head_type=head)
    model, loss_fn = result["model"], result["loss_fn"]
    demos = _tiny_demos()
    loader = torch.utils.data.DataLoader(
        demos["dataset"], batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        demos["holdout"], batch_size=8, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.head.weight.detach().clone()

    out = TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": loader,
            "val_dataloader": val_loader,
            "optimizer": optimizer,
            "loss_fn": loss_fn,
        },
        {"epochs": 1, "device": "cpu"},
    )
    losses = out["losses"]
    assert len(losses) == 1 and all(
        torch.isfinite(torch.as_tensor(float(value))) for value in losses)
    assert out["val_losses"], "the nested val pass must produce a loss"
    assert not torch.equal(model.head.weight.detach(), before)


# ── seeds that hold beside other nodes (2026-10-13 audit) ───────────────


def _no_global_reseed(monkeypatch):
    """Make any reseed of the process-wide RNG fail the test.

    An unseeded run executes a level's nodes concurrently, so a node that
    reseeds the shared generator rewinds the stream under its neighbours,
    and two of them take turns drawing from it.
    """
    def refuse(*args, **kwargs):
        raise AssertionError("reseeded the process-wide RNG")

    monkeypatch.setattr(torch, "manual_seed", refuse)


def test_seeded_init_matches_the_old_draw_order_bit_for_bit():
    # The weights a seed gave before the local generator are unchanged, so
    # every number quoted off a VLAModel still holds.
    from app.nodes.vla.vla_model_node import _resolve_config

    for overrides in ({}, {"vision_stem": "patchify", "patch_size": 8},
                      {"head_type": "regression", "seed": 3}):
        params = {**TINY, **overrides}
        torch.manual_seed(params["seed"])
        reference = VLAModule(**_resolve_config(params))
        built = VLAModelNode().build_module(params)
        for (name, ref), got in zip(reference.state_dict().items(),
                                    built.state_dict().values()):
            assert torch.equal(ref, got), name


def test_build_never_reseeds_the_global_rng(monkeypatch):
    _no_global_reseed(monkeypatch)
    first = VLAModelNode().build_module(dict(TINY))
    torch.rand(1000)  # a neighbour drawing from the shared stream
    second = VLAModelNode().build_module(dict(TINY))
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)


def test_two_seeded_vla_models_on_one_level_of_an_unseeded_run_match():
    import asyncio

    from app.core.graph_engine import execute_graph
    from app.core.runtime import initialize_runtime

    initialize_runtime()
    nodes = [{"id": "start", "type": "Start", "data": {"params": {}}}]
    nodes += [{"id": x, "type": "VLAModel", "data": {"params": dict(TINY)}}
              for x in "ab"]
    edges = [{"id": "t" + x, "source": "start", "target": x,
              "sourceHandle": "trigger", "type": "trigger"} for x in "ab"]
    context = ExecutionContext(
        graph_id="vla-race", weights_persistent=False,
        node_state_store=NodeStateStore(), device="cpu", seed=None)
    out = asyncio.run(execute_graph(nodes, edges, context=context))
    for a, b in zip(out["a"]["model"].parameters(),
                    out["b"]["model"].parameters()):
        assert torch.equal(a, b)


def test_rollout_seed_fixes_the_flow_heads_sampling_noise():
    model = _build(head_type="flow_matching")["model"]
    params = {"episodes": 2, "execute_k": 2, "max_steps": 12,
              "record_episodes": 2, "seed": 5, "device": "cpu"}
    torch.manual_seed(123)
    first = VLARolloutNode().execute(
        {"model": model, "env": _factory()}, params)
    torch.manual_seed(456)
    second = VLARolloutNode().execute(
        {"model": model, "env": _factory()}, params)
    assert torch.equal(first["frames"], second["frames"])
    assert first["report"] == second["report"]


def test_predict_action_draws_flow_noise_from_the_given_generator():
    model = _build(head_type="flow_matching")["model"]
    image, tokens, _ = _batch(batch_size=2)
    torch.manual_seed(1)
    a = model.predict_action(
        image, tokens, generator=torch.Generator().manual_seed(9))
    torch.manual_seed(2)
    b = model.predict_action(
        image, tokens, generator=torch.Generator().manual_seed(9))
    assert torch.equal(a, b)


def test_action_eval_uses_a_local_generator(monkeypatch):
    model = _build()["model"]
    holdout = _tiny_demos()["holdout"]
    params = {"max_samples": 16, "batch_size": 8, "seed": 3, "device": "cpu"}
    # The number the node reported while it reseeded the global RNG: the
    # local CPU generator walks the same stream, so it is unchanged.
    torch.manual_seed(3)
    images = torch.stack([holdout[i][0][0] for i in range(16)])
    tokens = torch.stack([holdout[i][0][1] for i in range(16)])
    targets = torch.stack([holdout[i][1] for i in range(16)])
    expected = 0.0
    for start in (0, 8):
        predicted = model.predict_action(
            images[start:start + 8], tokens[start:start + 8])
        expected += float((predicted - targets[start:start + 8]).pow(2).sum())
    expected /= 16 * targets[0].numel()

    _no_global_reseed(monkeypatch)
    result = VLAActionEvalNode().execute(
        {"model": model, "dataset": holdout}, params)
    assert result["action_mse"] == pytest.approx(expected, rel=1e-6)
