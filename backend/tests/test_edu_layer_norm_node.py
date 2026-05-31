"""Tests for EduLayerNormNode (lesson I4-3)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from cdui_plugins.deep.nodes.edu_layer_norm_node import EduLayerNormNode

D = 7


def _run(x, *, gamma=None, beta=None, context=None, **params):
    p = {"normalized_dim": 0, "eps": 1e-5, "elementwise_affine": True}
    p.update(params)
    inputs: dict = {"x": x}
    if gamma is not None:
        inputs["gamma"] = gamma
    if beta is not None:
        inputs["beta"] = beta
    return EduLayerNormNode().execute(inputs, p, context=context)


class _Ctx:
    verbose = True
    weights_persistent = False
    node_state_store = None
    current_node_id = None


def test_node_metadata():
    assert EduLayerNormNode.NODE_NAME == "Edu-LayerNorm"
    assert EduLayerNormNode.CATEGORY == "Transformer"
    out_names = [p.name for p in EduLayerNormNode.define_outputs()]
    assert out_names == ["y", "mean", "var"]


def test_matches_f_layer_norm_no_affine():
    torch.manual_seed(0)
    x = torch.randn(3, 5, D)
    res = _run(x, elementwise_affine=False)
    expected = F.layer_norm(x, (D,), None, None, 1e-5)
    assert torch.allclose(res["y"], expected, atol=1e-6)


def test_matches_f_layer_norm_with_affine():
    torch.manual_seed(1)
    x = torch.randn(3, 5, D)
    gamma = torch.randn(D)
    beta = torch.randn(D)
    res = _run(x, gamma=gamma, beta=beta)
    expected = F.layer_norm(x, (D,), gamma, beta, 1e-5)
    assert torch.allclose(res["y"], expected, atol=1e-6)


def test_output_zero_mean_unit_var_no_affine():
    torch.manual_seed(2)
    x = torch.randn(3, 5, D) * 4.0 + 2.0  # non-trivial scale/shift
    res = _run(x, elementwise_affine=False)
    y = res["y"]
    mean = y.mean(dim=-1)
    var = y.var(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-3)


def test_mean_var_outputs_shape_and_values():
    torch.manual_seed(3)
    x = torch.randn(4, D)
    res = _run(x)
    assert res["mean"].shape == (4, 1)
    assert res["var"].shape == (4, 1)
    assert torch.allclose(res["mean"], x.mean(dim=-1, keepdim=True), atol=1e-6)
    assert torch.allclose(
        res["var"], x.var(dim=-1, unbiased=False, keepdim=True), atol=1e-6
    )


def test_affine_applied_correctly():
    # With xhat ~ standardised, y should equal gamma * xhat + beta exactly.
    torch.manual_seed(4)
    x = torch.randn(2, D)
    gamma = torch.full((D,), 3.0)
    beta = torch.full((D,), -1.0)
    res = _run(x, gamma=gamma, beta=beta)
    xhat = _run(x, elementwise_affine=False)["y"]
    assert torch.allclose(res["y"], gamma * xhat + beta, atol=1e-6)


def test_default_gamma_beta_is_identity_normalize():
    # Omitting gamma/beta (defaults ones/zeros) must equal the no-affine xhat.
    torch.manual_seed(5)
    x = torch.randn(3, D)
    with_default_affine = _run(x, elementwise_affine=True)["y"]
    just_xhat = _run(x, elementwise_affine=False)["y"]
    assert torch.allclose(with_default_affine, just_xhat, atol=1e-6)


def test_y_same_shape_as_x():
    res = _run(torch.zeros(2, 3, D))
    assert res["y"].shape == (2, 3, D)


def test_normalized_dim_inference_and_match():
    torch.manual_seed(6)
    x = torch.randn(3, D)
    # Explicit correct normalized_dim works the same as inferring it.
    inferred = _run(x, normalized_dim=0)["y"]
    explicit = _run(x, normalized_dim=D)["y"]
    assert torch.allclose(inferred, explicit, atol=1e-6)


def test_wrong_gamma_length_raises():
    x = torch.randn(3, D)
    with pytest.raises(ValueError, match="gamma"):
        _run(x, gamma=torch.ones(D + 1))


def test_wrong_beta_length_raises():
    x = torch.randn(3, D)
    with pytest.raises(ValueError, match="beta"):
        _run(x, beta=torch.ones(D - 1))


def test_normalized_dim_mismatch_raises():
    x = torch.randn(3, D)
    with pytest.raises(ValueError, match="normalized_dim"):
        _run(x, normalized_dim=D + 2)


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        EduLayerNormNode().execute({}, {})


def test_scalar_input_rejected():
    with pytest.raises(ValueError, match="1-D"):
        _run(torch.tensor(3.0))


def test_steps_present_when_verbose():
    res = _run(torch.randn(3, D), context=_Ctx())
    assert "__steps__" in res
    step_names = [s.name for s in res["__steps__"]]
    assert step_names == ["mean", "var", "normalize", "affine"]


def test_steps_absent_without_verbose():
    res = _run(torch.randn(3, D))
    assert "__steps__" not in res


def test_step_mean_carries_scalars():
    res = _run(torch.randn(2, D), context=_Ctx())
    mean_step = next(s for s in res["__steps__"] if s.name == "mean")
    assert mean_step.scalars["D"] == float(D)
    assert "eps" in mean_step.scalars
