"""Tests for EduSVMNode (chapter pack I2-2: linear soft-margin SVM)."""

from __future__ import annotations

import pytest
import torch

from cdui_plugins.foundations.nodes.edu_svm_node import EduSVMNode


class _Ctx:
    """Minimal ExecutionContext stand-in: just carries a verbose flag."""

    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose


def _run(x_train, y_train, x_query, *, context=None, **params):
    p = {"C": 1.0, "lr": 0.01, "epochs": 100}
    p.update(params)
    return EduSVMNode().execute(
        {"x_train": x_train, "y_train": y_train, "x_query": x_query},
        p,
        context=context,
    )


def _separable_blobs(seed: int = 0, label_neg=0, label_pos=1, n_per: int = 40):
    """Two well-separated 2-D Gaussian blobs with labels {label_neg, label_pos}."""
    torch.manual_seed(seed)
    neg = torch.randn(n_per, 2) * 0.4 + torch.tensor([-4.0, -4.0])
    pos = torch.randn(n_per, 2) * 0.4 + torch.tensor([4.0, 4.0])
    x = torch.cat([neg, pos], dim=0)
    y = torch.tensor([label_neg] * n_per + [label_pos] * n_per)
    return x, y


def test_node_metadata():
    assert EduSVMNode.NODE_NAME == "Edu-SVM"
    assert EduSVMNode.CATEGORY == "Classical"
    out_names = [p.name for p in EduSVMNode.define_outputs()]
    assert out_names == [
        "predictions",
        "weights",
        "bias",
        "support_vectors",
        "decision_values",
    ]


def test_input_and_param_definitions():
    in_names = [p.name for p in EduSVMNode.define_inputs()]
    assert in_names == ["x_train", "y_train", "x_query"]
    param_names = [p.name for p in EduSVMNode.define_params()]
    assert param_names == ["C", "lr", "epochs"]


def test_separable_blobs_perfectly_classified():
    """Two far-apart blobs → held-out query points classified ~perfectly."""
    x_train, y_train = _separable_blobs(seed=0)
    # Held-out query points clearly on each side of the gap.
    x_query = torch.tensor(
        [
            [-3.5, -3.5],
            [-5.0, -4.0],
            [3.5, 3.5],
            [5.0, 4.0],
        ]
    )
    expected = torch.tensor([0, 0, 1, 1])
    res = _run(x_train, y_train, x_query, epochs=300, lr=0.05)

    preds = res["predictions"]
    acc = (preds == expected).float().mean().item()
    assert acc >= 0.9
    # A real separating direction was learned.
    assert res["weights"].norm().item() > 0.0
    # decision_values agree in sign with the predicted side.
    assert torch.equal(res["decision_values"] >= 0, expected.bool())


def test_train_set_accuracy_high_on_separable_data():
    x_train, y_train = _separable_blobs(seed=1)
    res = _run(x_train, y_train, x_train, epochs=300, lr=0.05)
    acc = (res["predictions"] == y_train).float().mean().item()
    assert acc >= 0.9


def test_original_label_space_respected():
    """Labels {3, 7} → predictions contain only 3 or 7 (never ±1 internals)."""
    x_train, y_train = _separable_blobs(seed=2, label_neg=3, label_pos=7)
    x_query = torch.tensor([[-4.0, -4.0], [4.0, 4.0], [-3.0, -3.5]])
    res = _run(x_train, y_train, x_query, epochs=300, lr=0.05)
    preds = res["predictions"]
    unique_preds = set(preds.tolist())
    assert unique_preds.issubset({3, 7})
    # The lower class (3) maps to -1 side, higher (7) to +1 side.
    assert preds[0].item() == 3
    assert preds[1].item() == 7


def test_support_vector_mask_shape_and_dtype():
    x_train, y_train = _separable_blobs(seed=3)
    res = _run(x_train, y_train, x_train, epochs=50)
    sv = res["support_vectors"]
    assert sv.shape == (x_train.shape[0],)
    # Mask must be a boolean (or integer) per-sample indicator.
    assert sv.dtype in (torch.bool, torch.int8, torch.int32, torch.int64)


def test_output_shapes():
    x_train, y_train = _separable_blobs(seed=4)
    x_query = torch.randn(6, 2)
    res = _run(x_train, y_train, x_query, epochs=20)
    assert res["predictions"].shape == (6,)
    assert res["weights"].shape == (2,)
    assert res["bias"].ndim == 0  # scalar
    assert res["decision_values"].shape == (6,)


def test_decision_values_equal_w_dot_x_plus_b():
    x_train, y_train = _separable_blobs(seed=5)
    x_query = torch.randn(5, 2)
    res = _run(x_train, y_train, x_query, epochs=30)
    expected = x_query @ res["weights"] + res["bias"]
    assert torch.allclose(res["decision_values"], expected, atol=1e-5)


def test_non_binary_labels_raise():
    x = torch.randn(9, 2)
    y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])  # three classes
    with pytest.raises(ValueError, match="2 distinct classes"):
        _run(x, y, torch.randn(2, 2))


def test_label_count_mismatch_raises():
    with pytest.raises(ValueError, match="mismatch"):
        _run(torch.zeros(5, 2), torch.tensor([0, 1, 0]), torch.zeros(1, 2))


def test_feature_dim_mismatch_raises():
    x_train = torch.zeros(6, 2)
    y_train = torch.tensor([0, 0, 0, 1, 1, 1])
    x_query = torch.zeros(2, 3)  # 3 features vs 2
    with pytest.raises(ValueError, match="features"):
        _run(x_train, y_train, x_query)


def test_negative_C_raises():
    x_train, y_train = _separable_blobs(seed=6)
    with pytest.raises(ValueError, match="C"):
        _run(x_train, y_train, x_train, C=-1.0)


def test_zero_epochs_raises():
    x_train, y_train = _separable_blobs(seed=7)
    with pytest.raises(ValueError, match="epochs"):
        _run(x_train, y_train, x_train, epochs=0)


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        EduSVMNode().execute(
            {"x_train": torch.zeros(5, 2)},
            {"C": 1.0, "lr": 0.01, "epochs": 10},
        )


def test_verbose_yields_steps():
    x_train, y_train = _separable_blobs(seed=8)
    res = _run(
        x_train, y_train, x_train, epochs=100, context=_Ctx(verbose=True)
    )
    assert "__steps__" in res
    steps = res["__steps__"]
    assert len(steps) > 0
    names = [s.name for s in steps]
    # init + capped epoch snapshots + final decision step.
    assert names[0] == "init"
    assert names[-1] == "decision"
    # Epoch snapshots are capped to ~8 even though we ran 100 epochs.
    epoch_steps = [s for s in steps if s.name.startswith("epoch_")]
    assert 0 < len(epoch_steps) <= 8
    # Each epoch snapshot records the documented scalars and the weight tensor.
    for s in epoch_steps:
        assert {"epoch", "loss", "num_support_vectors"} <= set(s.scalars)
        assert "w" in s.tensors


def test_non_verbose_yields_no_steps():
    x_train, y_train = _separable_blobs(seed=9)
    res = _run(x_train, y_train, x_train, epochs=50)
    assert "__steps__" not in res
    # Also true when an explicit non-verbose context is supplied.
    res2 = _run(
        x_train, y_train, x_train, epochs=50, context=_Ctx(verbose=False)
    )
    assert "__steps__" not in res2
