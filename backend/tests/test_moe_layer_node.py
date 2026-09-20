"""Tests for MoELayerNode."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from app.nodes.transformer import moe_layer_node
from app.nodes.transformer.moe_layer_node import MoELayerNode, _MoELayer


def _run(x, **params):
    p = {
        "num_experts": 4,
        "top_k": 2,
        "hidden_dim": 8,
        "expert_hidden_dim": 16,
        "seed": 42,
    }
    p.update(params)
    return MoELayerNode().execute({"x": x}, p)


def test_node_metadata():
    assert MoELayerNode.NODE_NAME == "MoELayer"
    assert MoELayerNode.CATEGORY == "Transformer"
    out_names = [p.name for p in MoELayerNode.define_outputs()]
    assert "output" in out_names
    assert "routing_weights" in out_names
    assert "expert_indices" in out_names


def test_output_preserves_shape():
    """[B, T, H] in → [B, T, H] out."""
    x = torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(0))
    res = _run(x)
    assert res["output"].shape == (2, 5, 8)


def test_routing_weights_top_k():
    """routing_weights should have shape [B, T, top_k] and sum to 1 per token."""
    x = torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(0))
    res = _run(x, top_k=2)
    rw = res["routing_weights"]
    assert rw.shape == (2, 5, 2)
    sums = rw.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_expert_indices_in_range():
    x = torch.randn(2, 3, 8, generator=torch.Generator().manual_seed(0))
    res = _run(x, num_experts=4, top_k=2)
    idx = res["expert_indices"]
    assert idx.shape == (2, 3, 2)
    assert int(idx.min()) >= 0
    assert int(idx.max()) < 4


def test_top_k_one_uses_single_expert():
    x = torch.randn(1, 4, 8, generator=torch.Generator().manual_seed(0))
    res = _run(x, top_k=1)
    assert res["routing_weights"].shape == (1, 4, 1)
    # Each token should be assigned exactly one expert; weights all == 1.
    assert torch.allclose(res["routing_weights"], torch.ones(1, 4, 1), atol=1e-5)


def test_seed_reproducible():
    x = torch.randn(2, 3, 8, generator=torch.Generator().manual_seed(0))
    a = _run(x, seed=42)
    b = _run(x, seed=42)
    assert torch.allclose(a["output"], b["output"])
    c = _run(x, seed=99)
    assert not torch.allclose(a["output"], c["output"])


def test_top_k_clamped_to_num_experts():
    x = torch.randn(1, 2, 8, generator=torch.Generator().manual_seed(0))
    # top_k > num_experts should be clamped — no crash
    res = _run(x, num_experts=3, top_k=10)
    assert res["routing_weights"].shape[-1] == 3


def test_dim_mismatch_raises():
    x = torch.randn(1, 2, 16)  # H=16 but hidden_dim=8
    with pytest.raises(RuntimeError):
        _run(x, hidden_dim=8)


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        MoELayerNode().execute(
            {},
            {"num_experts": 4, "top_k": 2, "hidden_dim": 8, "expert_hidden_dim": 16, "seed": 42},
        )


# ── the seed alone decides the weights ────────────────────────────────────


def _draw_beside_every_linear(monkeypatch) -> list[torch.Tensor]:
    """Make another node draw from the global RNG in the middle of a build.

    A deterministic stand-in for the engine running a level's nodes on
    worker threads: every ``nn.Linear`` the build constructs is preceded by
    one draw from the process-global generator, which is what a neighbouring
    node's draw landing mid-construction amounts to. Returns what that
    neighbour drew, in order.
    """
    drawn: list[torch.Tensor] = []
    original = nn.Linear.reset_parameters

    def interleaved(self):
        drawn.append(torch.rand(1))
        return original(self)

    monkeypatch.setattr(nn.Linear, "reset_parameters", interleaved)
    return drawn


def _layer_the_node_built(x, **params) -> nn.Module:
    """The ``_MoELayer`` behind one execute; the node hands out outputs only."""
    built: list[nn.Module] = []

    class _Recorded(_MoELayer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(moe_layer_node, "_MoELayer", _Recorded)
        _run(x, **params)
    assert len(built) == 1, f"expected one layer per execute, got {len(built)}"
    return built[0]


def _differing(a: dict, b: dict) -> list[str]:
    assert list(a) == list(b)
    return [key for key in a if not torch.equal(a[key], b[key])]


#: The params both MoE examples ship: examples/Transformer/MoE-TopK-Routing
#: and plugins/deep/examples/C6-4/MoE-Routing.
_EXAMPLE_PARAMS = {"num_experts": 4, "top_k": 2, "hidden_dim": 32,
                   "expert_hidden_dim": 64, "seed": 42}


def test_same_seed_gives_the_same_weights_while_another_node_draws(monkeypatch):
    """Two layers from one seed on one level of an unseeded run.

    Each used to seed the process-global generator and build from it, so a
    node drawing from it in between moved the gate and every expert.
    """
    x = torch.randn(1, 8, 32, generator=torch.Generator().manual_seed(0))
    alone = _layer_the_node_built(x, **_EXAMPLE_PARAMS).state_dict()

    _draw_beside_every_linear(monkeypatch)
    beside = _layer_the_node_built(x, **_EXAMPLE_PARAMS).state_dict()

    assert not _differing(alone, beside), (
        f"{_differing(alone, beside)} moved because another node drew from "
        f"the global RNG while the layer was built")


@pytest.mark.parametrize(
    ("num_experts", "top_k", "hidden_dim", "expert_hidden_dim", "seed"), [
        (4, 2, 32, 64, 42),   # both MoE examples
        (4, 2, 8, 16, 42),    # this file's defaults
        (3, 10, 8, 16, 7),    # top_k clamped to num_experts
        (1, 1, 8, 16, -3),    # the seed param has no minimum
    ])
def test_the_weights_are_the_ones_the_seeded_constructor_gave(
        num_experts, top_k, hidden_dim, expert_hidden_dim, seed):
    """Bit for bit what ``torch.manual_seed(seed)`` + the constructor gave.

    That is how the node built its layer before it stopped touching the
    global RNG, so every routing an example prints is unchanged.
    """
    torch.manual_seed(seed)
    expected = _MoELayer(num_experts, top_k, hidden_dim,
                         expert_hidden_dim).state_dict()

    got = _layer_the_node_built(
        torch.zeros(1, 2, hidden_dim), num_experts=num_experts, top_k=top_k,
        hidden_dim=hidden_dim, expert_hidden_dim=expert_hidden_dim,
        seed=seed).state_dict()

    assert not _differing(expected, got)


def test_building_neither_reseeds_nor_rewinds_the_global_rng(monkeypatch):
    """The process-global generator belongs to the run, not to this node.

    Reseeding it hands a node drawing beside this one the seed-42 stream;
    restoring it afterwards (``fork_rng``, or get/set_rng_state) rewinds
    the stream under that node, so the same numbers are handed out twice.
    The constructor's own draws are consumption, which any unseeded node
    does, so they are allowed.
    """
    x = torch.zeros(1, 2, 8)
    torch.manual_seed(1234)
    first = torch.rand(1)
    torch.manual_seed(1234)
    drawn = _draw_beside_every_linear(monkeypatch)

    _run(x)
    after = torch.rand(1)

    assert torch.equal(drawn[0], first), (
        "the node reseeded the global RNG: a draw made beside it came from "
        "another stream than the run's")
    assert not any(torch.equal(after, value) for value in drawn), (
        "the node rewound the global RNG: a value drawn beside it came out "
        "again after it finished")
