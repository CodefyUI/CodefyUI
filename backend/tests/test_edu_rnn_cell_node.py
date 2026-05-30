"""Tests for EduRNNCellNode (lesson I4-1)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cdui_plugins.deep.nodes.edu_rnn_cell_node import EduRNNCellNode


def _run(x_seq, *, h0=None, **params):
    p = {"input_size": 4, "hidden_size": 8, "activation": "tanh", "seed": 0}
    p.update(params)
    inputs: dict = {"x_seq": x_seq}
    if h0 is not None:
        inputs["h0"] = h0
    return EduRNNCellNode().execute(inputs, p)


class _VerboseCtx:
    verbose = True


def test_node_metadata():
    assert EduRNNCellNode.NODE_NAME == "Edu-RNNCell"
    assert EduRNNCellNode.CATEGORY == "RNN"
    out_names = [p.name for p in EduRNNCellNode.define_outputs()]
    assert out_names == ["h_seq", "h_last", "weights"]


def test_output_shape_2d():
    res = _run(torch.randn(6, 4), input_size=4, hidden_size=8)
    assert res["h_seq"].shape == (6, 8)
    assert res["h_last"].shape == (8,)
    # weights output is W_ih, [hidden, input].
    assert res["weights"].shape == (8, 4)


def test_output_shape_3d_batch():
    res = _run(torch.randn(3, 6, 4), input_size=4, hidden_size=8)
    assert res["h_seq"].shape == (3, 6, 8)
    assert res["h_last"].shape == (3, 8)
    assert res["weights"].shape == (8, 4)


def test_h_last_is_final_hidden_state():
    res = _run(torch.randn(5, 4))
    assert torch.allclose(res["h_last"], res["h_seq"][-1])

    resb = _run(torch.randn(2, 5, 4))
    assert torch.allclose(resb["h_last"], resb["h_seq"][:, -1, :])


def test_default_h0_is_zeros():
    """With zero h0, the first hidden state depends only on x_0 (Wh = b_hh)."""
    input_size, hidden_size, seed = 4, 8, 0
    x_seq = torch.randn(5, input_size)

    res_default = _run(x_seq, input_size=input_size, hidden_size=hidden_size, seed=seed)
    res_zeros = _run(
        x_seq,
        h0=torch.zeros(hidden_size),
        input_size=input_size,
        hidden_size=hidden_size,
        seed=seed,
    )
    assert torch.allclose(res_default["h_seq"], res_zeros["h_seq"])

    # Sanity: the first step with zero h0 equals activation(W_ih x_0 + b_ih + b_hh).
    W_ih, W_hh, b_ih, b_hh = EduRNNCellNode._build_params(input_size, hidden_size, seed)
    expected_h0 = torch.tanh(x_seq[0] @ W_ih.T + b_ih + b_hh)
    assert torch.allclose(res_default["h_seq"][0], expected_h0, atol=1e-6)


def test_deterministic_given_seed():
    x = torch.randn(5, 4)
    a = _run(x, seed=7)
    b = _run(x, seed=7)
    assert torch.allclose(a["h_seq"], b["h_seq"])
    assert torch.allclose(a["weights"], b["weights"])
    # Different seed → different weights.
    c = _run(x, seed=8)
    assert not torch.allclose(a["weights"], c["weights"])


def test_weights_output_matches_recomputed_params():
    """The display `weights` output is exactly the seed-built W_ih."""
    input_size, hidden_size, seed = 4, 8, 3
    res = _run(torch.randn(4, input_size), input_size=input_size,
               hidden_size=hidden_size, seed=seed)
    W_ih, _, _, _ = EduRNNCellNode._build_params(input_size, hidden_size, seed)
    assert torch.allclose(res["weights"], W_ih)


@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_equivalence_to_nn_rnncell_2d(activation):
    """Unrolled node must match torch.nn.RNNCell stepped over the same sequence."""
    input_size, hidden_size, seed = 4, 8, 5
    seq = 7
    x_seq = torch.randn(seq, input_size, generator=torch.Generator().manual_seed(11))

    res = _run(x_seq, input_size=input_size, hidden_size=hidden_size,
               activation=activation, seed=seed)

    # Recompute the four parameters with the same seed and load them into a
    # reference nn.RNNCell (identical layout: weight_ih/weight_hh/bias_ih/bias_hh).
    W_ih, W_hh, b_ih, b_hh = EduRNNCellNode._build_params(input_size, hidden_size, seed)
    # The display `weights` output is W_ih — confirm it lines up with what we feed
    # the reference, so the equivalence really uses the node's own parameters.
    assert torch.allclose(res["weights"], W_ih)

    ref = nn.RNNCell(input_size, hidden_size, nonlinearity=activation)
    with torch.no_grad():
        ref.weight_ih.copy_(W_ih)
        ref.weight_hh.copy_(W_hh)
        ref.bias_ih.copy_(b_ih)
        ref.bias_hh.copy_(b_hh)

    h = torch.zeros(1, hidden_size)
    ref_states = []
    for t in range(seq):
        h = ref(x_seq[t].unsqueeze(0), h)
        ref_states.append(h.squeeze(0))
    reference = torch.stack(ref_states, dim=0)  # [seq, hidden]

    assert torch.allclose(res["h_seq"], reference, atol=1e-6)
    assert torch.allclose(res["h_last"], reference[-1], atol=1e-6)


def test_equivalence_to_nn_rnncell_3d_batch():
    """Batched [N, seq, input] unroll matches nn.RNNCell run per sequence."""
    input_size, hidden_size, seed = 4, 8, 2
    N, seq = 3, 6
    x = torch.randn(N, seq, input_size, generator=torch.Generator().manual_seed(13))

    res = _run(x, input_size=input_size, hidden_size=hidden_size, seed=seed)

    W_ih, W_hh, b_ih, b_hh = EduRNNCellNode._build_params(input_size, hidden_size, seed)
    ref = nn.RNNCell(input_size, hidden_size, nonlinearity="tanh")
    with torch.no_grad():
        ref.weight_ih.copy_(W_ih)
        ref.weight_hh.copy_(W_hh)
        ref.bias_ih.copy_(b_ih)
        ref.bias_hh.copy_(b_hh)

    h = torch.zeros(N, hidden_size)
    states = []
    for t in range(seq):
        h = ref(x[:, t, :], h)
        states.append(h)
    reference = torch.stack(states, dim=1)  # [N, seq, hidden]

    assert torch.allclose(res["h_seq"], reference, atol=1e-6)


def test_custom_h0_is_used():
    """A non-zero h0 must flow into the first step's memory contribution."""
    input_size, hidden_size, seed = 4, 8, 0
    x_seq = torch.randn(4, input_size)
    h0 = torch.randn(hidden_size)

    res = _run(x_seq, h0=h0, input_size=input_size, hidden_size=hidden_size, seed=seed)

    W_ih, W_hh, b_ih, b_hh = EduRNNCellNode._build_params(input_size, hidden_size, seed)
    expected_h0 = torch.tanh(x_seq[0] @ W_ih.T + b_ih + h0 @ W_hh.T + b_hh)
    assert torch.allclose(res["h_seq"][0], expected_h0, atol=1e-6)


def test_h0_batch_broadcast_from_1d():
    """A 1-D h0 should broadcast across a 3-D batch input."""
    res = _run(torch.randn(2, 5, 4), h0=torch.randn(8), input_size=4, hidden_size=8)
    assert res["h_seq"].shape == (2, 5, 8)


def test_relu_activation_is_nonnegative():
    res = _run(torch.randn(6, 4), activation="relu")
    assert torch.all(res["h_seq"] >= 0)


def test_input_size_mismatch_raises():
    with pytest.raises(ValueError, match="input_size"):
        _run(torch.randn(5, 3), input_size=4)


def test_bad_ndim_raises():
    with pytest.raises(ValueError, match=r"\[seq, input_size\]"):
        _run(torch.randn(4))  # 1-D is invalid


def test_h0_wrong_shape_raises():
    with pytest.raises(ValueError, match="h0"):
        _run(torch.randn(5, 4), h0=torch.randn(3), input_size=4, hidden_size=8)


def test_unknown_activation_raises():
    with pytest.raises(ValueError, match="activation"):
        _run(torch.randn(5, 4), activation="sigmoid")


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        EduRNNCellNode().execute({}, {"input_size": 4, "hidden_size": 8})


def test_steps_absent_without_verbose():
    res = _run(torch.randn(5, 4))
    assert "__steps__" not in res


def test_steps_present_when_verbose():
    res = EduRNNCellNode().execute(
        {"x_seq": torch.randn(5, 4)},
        {"input_size": 4, "hidden_size": 8, "activation": "tanh", "seed": 0},
        context=_VerboseCtx(),
    )
    assert "__steps__" in res
    step_names = [s.name for s in res["__steps__"]]
    # First timestep, last timestep, and the stacked-states summary all present.
    assert "t=0" in step_names
    assert "h_seq" in step_names
    # A recorded timestep exposes x_t, Wx, Wh, h_t.
    first = next(s for s in res["__steps__"] if s.name == "t=0")
    assert set(first.tensors) == {"x_t", "Wx", "Wh", "h_t"}


def test_verbose_steps_capped_for_long_sequence():
    """A long sequence samples timesteps rather than recording every one."""
    seq = 20
    res = EduRNNCellNode().execute(
        {"x_seq": torch.randn(seq, 4)},
        {"input_size": 4, "hidden_size": 8, "activation": "tanh", "seed": 0},
        context=_VerboseCtx(),
    )
    timestep_steps = [s for s in res["__steps__"] if s.name.startswith("t=")]
    assert len(timestep_steps) <= 6
    # First and last timesteps are always included.
    names = {s.name for s in timestep_steps}
    assert "t=0" in names
    assert f"t={seq - 1}" in names
