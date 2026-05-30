"""Tests for EduLSTMCellNode (lesson I4-1)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cdui_plugins.deep.nodes.edu_lstm_cell_node import EduLSTMCellNode


def _run(x_seq, *, h0=None, c0=None, **params):
    p = {"input_size": 4, "hidden_size": 8, "seed": 0}
    p.update(params)
    inputs: dict = {"x_seq": x_seq}
    if h0 is not None:
        inputs["h0"] = h0
    if c0 is not None:
        inputs["c0"] = c0
    return EduLSTMCellNode().execute(inputs, p)


def _reference_lstm_cell(res, input_size, hidden_size):
    """Build an nn.LSTMCell seeded with the node's four parameter tensors."""
    cell = nn.LSTMCell(input_size, hidden_size)
    with torch.no_grad():
        cell.weight_ih.copy_(res["weight_ih"])
        cell.weight_hh.copy_(res["weight_hh"])
        cell.bias_ih.copy_(res["bias_ih"])
        cell.bias_hh.copy_(res["bias_hh"])
    return cell


def test_node_metadata():
    assert EduLSTMCellNode.NODE_NAME == "Edu-LSTMCell"
    assert EduLSTMCellNode.CATEGORY == "RNN"
    out_names = [p.name for p in EduLSTMCellNode.define_outputs()]
    assert out_names == [
        "h_seq",
        "h_last",
        "c_last",
        "weight_ih",
        "weight_hh",
        "bias_ih",
        "bias_hh",
    ]


def test_param_tensor_shapes():
    res = _run(torch.zeros(3, 4), input_size=4, hidden_size=8)
    assert res["weight_ih"].shape == (32, 4)
    assert res["weight_hh"].shape == (32, 8)
    assert res["bias_ih"].shape == (32,)
    assert res["bias_hh"].shape == (32,)


def test_output_shape_2d():
    res = _run(torch.randn(5, 4), input_size=4, hidden_size=8)
    assert res["h_seq"].shape == (5, 8)
    assert res["h_last"].shape == (8,)
    assert res["c_last"].shape == (8,)
    # h_last is exactly the final row of h_seq.
    assert torch.allclose(res["h_seq"][-1], res["h_last"])


def test_output_shape_3d_batch():
    res = _run(torch.randn(3, 5, 4), input_size=4, hidden_size=8)  # [N, seq, input]
    assert res["h_seq"].shape == (3, 5, 8)
    assert res["h_last"].shape == (3, 8)
    assert res["c_last"].shape == (3, 8)
    assert torch.allclose(res["h_seq"][:, -1, :], res["h_last"])


def test_equivalence_to_torch_lstmcell_2d():
    """Stepping the same sequence through nn.LSTMCell must match h_seq / c_last."""
    input_size, hidden_size = 4, 8
    x = torch.randn(7, input_size, generator=torch.Generator().manual_seed(3))
    res = _run(x, input_size=input_size, hidden_size=hidden_size, seed=11)

    cell = _reference_lstm_cell(res, input_size, hidden_size)
    h = torch.zeros(1, hidden_size)
    c = torch.zeros(1, hidden_size)
    ref_h_states = []
    for t in range(x.shape[0]):
        h, c = cell(x[t].unsqueeze(0), (h, c))
        ref_h_states.append(h.squeeze(0))
    ref_h_seq = torch.stack(ref_h_states, dim=0)

    assert torch.allclose(res["h_seq"], ref_h_seq, atol=1e-6)
    assert torch.allclose(res["h_last"], ref_h_seq[-1], atol=1e-6)
    assert torch.allclose(res["c_last"], c.squeeze(0), atol=1e-6)


def test_equivalence_to_torch_lstmcell_3d_batch():
    """Batched path must also match nn.LSTMCell stepped over the batch."""
    input_size, hidden_size = 4, 8
    x = torch.randn(2, 6, input_size, generator=torch.Generator().manual_seed(5))
    res = _run(x, input_size=input_size, hidden_size=hidden_size, seed=7)

    cell = _reference_lstm_cell(res, input_size, hidden_size)
    h = torch.zeros(2, hidden_size)
    c = torch.zeros(2, hidden_size)
    ref_h_states = []
    for t in range(x.shape[1]):
        h, c = cell(x[:, t, :], (h, c))
        ref_h_states.append(h)
    ref_h_seq = torch.stack(ref_h_states, dim=1)

    assert torch.allclose(res["h_seq"], ref_h_seq, atol=1e-6)
    assert torch.allclose(res["c_last"], c, atol=1e-6)


def test_nonzero_initial_state_matches_reference():
    """Custom h0/c0 must feed the recurrence the same way nn.LSTMCell does."""
    input_size, hidden_size = 4, 8
    x = torch.randn(4, input_size, generator=torch.Generator().manual_seed(1))
    h0 = torch.randn(hidden_size, generator=torch.Generator().manual_seed(2))
    c0 = torch.randn(hidden_size, generator=torch.Generator().manual_seed(3))
    res = _run(x, h0=h0, c0=c0, input_size=input_size, hidden_size=hidden_size, seed=9)

    cell = _reference_lstm_cell(res, input_size, hidden_size)
    h = h0.unsqueeze(0)
    c = c0.unsqueeze(0)
    for t in range(x.shape[0]):
        h, c = cell(x[t].unsqueeze(0), (h, c))

    assert torch.allclose(res["h_last"], h.squeeze(0), atol=1e-6)
    assert torch.allclose(res["c_last"], c.squeeze(0), atol=1e-6)


def test_default_initial_state_is_zeros():
    """Omitting h0/c0 must behave identically to passing explicit zeros."""
    x = torch.randn(4, 4, generator=torch.Generator().manual_seed(8))
    default = _run(x, seed=4)
    explicit = _run(x, h0=torch.zeros(8), c0=torch.zeros(8), seed=4)
    assert torch.allclose(default["h_seq"], explicit["h_seq"])
    assert torch.allclose(default["c_last"], explicit["c_last"])


def test_deterministic_given_seed():
    x = torch.randn(5, 4, generator=torch.Generator().manual_seed(0))
    a = _run(x, seed=123)
    b = _run(x, seed=123)
    assert torch.allclose(a["h_seq"], b["h_seq"])
    assert torch.allclose(a["weight_ih"], b["weight_ih"])
    # Different seed → different weights.
    c = _run(x, seed=124)
    assert not torch.allclose(a["weight_ih"], c["weight_ih"])


def test_input_size_mismatch_raises():
    with pytest.raises(ValueError, match="input_size"):
        _run(torch.zeros(3, 5), input_size=4)


def test_rejects_4d_input():
    with pytest.raises(ValueError, match="x_seq"):
        _run(torch.zeros(2, 3, 4, 4))


def test_h0_wrong_length_raises():
    with pytest.raises(ValueError, match="h0"):
        _run(torch.zeros(3, 4), h0=torch.zeros(5), input_size=4, hidden_size=8)


def test_c0_batch_mismatch_raises():
    with pytest.raises(ValueError, match="c0"):
        # x_seq batch is 2, c0 batch is 3.
        _run(torch.zeros(2, 3, 4), c0=torch.zeros(3, 8), input_size=4, hidden_size=8)


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        EduLSTMCellNode().execute({}, {"input_size": 4, "hidden_size": 8, "seed": 0})


def test_no_steps_when_not_verbose():
    res = _run(torch.randn(4, 4))
    assert "__steps__" not in res


def test_step_trace_emitted_when_verbose():
    class _Ctx:
        verbose = True

    res = EduLSTMCellNode().execute(
        {"x_seq": torch.randn(3, 4)},
        {"input_size": 4, "hidden_size": 8, "seed": 0},
        context=_Ctx(),
    )
    assert "__steps__" in res
    step_names = [s.name for s in res["__steps__"]]
    # Per-timestep gate steps plus the final stacking step.
    assert "timestep_0" in step_names
    assert "h_seq" in step_names
    # Each timestep step exposes the four gates and the new cell/hidden state.
    first = res["__steps__"][0]
    assert {"i", "f", "g", "o", "c_t", "h_t"} <= set(first.tensors.keys())


def test_verbose_step_count_capped_for_long_sequence():
    """A long sequence must not emit one step per timestep."""
    class _Ctx:
        verbose = True

    seq = 50
    res = EduLSTMCellNode().execute(
        {"x_seq": torch.randn(seq, 4)},
        {"input_size": 4, "hidden_size": 8, "seed": 0},
        context=_Ctx(),
    )
    timestep_steps = [s for s in res["__steps__"] if s.name.startswith("timestep_")]
    assert len(timestep_steps) <= 6
