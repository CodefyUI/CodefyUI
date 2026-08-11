"""Tests for CausalLMModelNode."""

from __future__ import annotations

import pickle

import pytest
import torch

from app.nodes.llm.causal_lm_model_node import CausalLMModelNode

TINY = {
    "vocab_size": 256,
    "d_model": 64,
    "n_layers": 2,
    "n_heads": 4,
    "d_ff": 128,
    "max_seq_len": 32,
    "seed": 7,
}


def build(**overrides):
    params = {**TINY, **overrides}
    return CausalLMModelNode().execute({}, params), params


def test_node_metadata():
    assert CausalLMModelNode.NODE_NAME == "CausalLMModel"
    assert CausalLMModelNode.CATEGORY == "LLM"
    assert CausalLMModelNode.define_inputs() == []
    assert CausalLMModelNode.cacheable is False


def test_param_count_matches_analytic_formula_tied():
    res, p = build()
    d, layers, ff, vocab, seq = (
        p["d_model"], p["n_layers"], p["d_ff"], p["vocab_size"], p["max_seq_len"],
    )
    per_block = (
        (3 * d * d + 3 * d)      # qkv
        + (d * d + d)            # attn out proj
        + (d * ff + ff)          # mlp in
        + (ff * d + d)           # mlp out
        + 4 * d                  # two LayerNorms (weight + bias each)
    )
    expected = (
        vocab * d                # token embedding (tied with lm_head)
        + seq * d                # learned positions
        + layers * per_block
        + 2 * d                  # final LayerNorm
    )
    assert res["param_count"] == expected
    assert res["param_count"] == sum(
        t.numel() for t in res["model"].parameters() if t.requires_grad
    )


def test_untied_head_adds_vocab_by_d_model():
    tied, p = build()
    untied, _ = build(tie_embeddings=False)
    assert untied["param_count"] - tied["param_count"] == p["vocab_size"] * p["d_model"]
    assert untied["model"].lm_head.weight is not untied["model"].tok_emb.weight
    assert tied["model"].lm_head.weight is tied["model"].tok_emb.weight


def test_forward_shape_contract():
    res, p = build()
    ids = torch.randint(0, p["vocab_size"], (3, 20), dtype=torch.int64)
    logits = res["model"](ids)
    assert logits.shape == (3, 20, p["vocab_size"])
    assert logits.dtype == torch.float32


def test_forward_is_causal():
    # Changing tokens after position t must not change logits at <= t.
    res, p = build()
    model = res["model"].eval()
    base = torch.randint(0, p["vocab_size"], (1, 16), dtype=torch.int64)
    changed = base.clone()
    changed[0, 10:] = (changed[0, 10:] + 1) % p["vocab_size"]
    with torch.no_grad():
        out_base = model(base)
        out_changed = model(changed)
    assert torch.allclose(out_base[0, :10], out_changed[0, :10], atol=1e-5)
    assert not torch.allclose(out_base[0, 10:], out_changed[0, 10:], atol=1e-3)


def test_same_seed_same_weights_different_seed_differs():
    a, _ = build()
    b, _ = build()
    c, _ = build(seed=8)
    for pa, pb in zip(a["model"].parameters(), b["model"].parameters()):
        assert torch.equal(pa, pb)
    assert any(
        not torch.equal(pa, pc)
        for pa, pc in zip(a["model"].parameters(), c["model"].parameters())
    )


def test_build_does_not_disturb_global_rng():
    torch.manual_seed(1234)
    before = torch.rand(3)
    torch.manual_seed(1234)
    build()
    after = torch.rand(3)
    assert torch.equal(before, after)


def test_invalid_head_split_names_the_params():
    with pytest.raises(ValueError, match="d_model.*n_heads"):
        build(n_heads=5)


def test_sequence_longer_than_max_seq_len_is_refused():
    res, p = build()
    ids = torch.zeros((1, p["max_seq_len"] + 1), dtype=torch.int64)
    with pytest.raises(ValueError, match="max_seq_len"):
        res["model"](ids)


@pytest.mark.parametrize("positional", ["learned", "sinusoidal", "rope"])
@pytest.mark.parametrize("norm", ["layernorm", "rmsnorm"])
def test_variants_forward_and_backward(positional, norm):
    res, p = build(positional=positional, norm=norm, activation="silu")
    ids = torch.randint(0, p["vocab_size"], (2, 12), dtype=torch.int64)
    logits = res["model"](ids)
    logits.sum().backward()
    grads = [t.grad for t in res["model"].parameters() if t.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_gradient_checkpointing_trains():
    res, p = build(gradient_checkpointing=True)
    model = res["model"].train()
    ids = torch.randint(0, p["vocab_size"], (2, 12), dtype=torch.int64)
    model(ids).sum().backward()
    assert model.tok_emb.weight.grad is not None


def test_full_model_pickle_roundtrip():
    # ModelSaver's full_model mode pickles the module; function-local
    # classes would fail here (#283).
    res, p = build()
    clone = pickle.loads(pickle.dumps(res["model"]))
    ids = torch.randint(0, p["vocab_size"], (1, 8), dtype=torch.int64)
    with torch.no_grad():
        assert torch.allclose(res["model"].eval()(ids), clone.eval()(ids), atol=1e-6)


def test_tiny_model_overfits_through_real_training_loop():
    from torch.utils.data import DataLoader, TensorDataset

    from app.nodes.llm.lm_cross_entropy_loss_node import LMCrossEntropyLossNode
    from app.nodes.training.training_loop_node import TrainingLoopNode

    res, p = build(d_model=32, d_ff=64, n_heads=2)
    model = res["model"]
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randint(
        0, p["vocab_size"], (8, 16), dtype=torch.int64, generator=generator,
    )
    labels = torch.roll(inputs, shifts=-1, dims=1)
    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    loss_fn = LMCrossEntropyLossNode().execute({}, {})["loss_fn"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    out = TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": loader,
            "optimizer": optimizer,
            "loss_fn": loss_fn,
        },
        {"epochs": 60, "device": "cpu"},
    )
    losses = out["losses"]
    first = float(losses[0])
    last = float(losses[-1])
    # ln(256) ~ 5.55 at init; memorizing 8 repeated sequences must cut the
    # loss to a small fraction of that.
    assert first > 3.0
    assert last < first * 0.35
