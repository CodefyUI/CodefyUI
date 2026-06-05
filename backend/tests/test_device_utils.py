"""Tests for core.device_utils — device resolution and MPS-safe tensor moves.

The float64→float32 coercion only matters on Apple MPS (which has no float64).
CPU-path behaviour is asserted unconditionally so this file is still meaningful
on Linux CI; the actual MPS downcast assertions are gated behind availability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from app.core.device_utils import (
    context_device,
    describe_accelerator,
    get_available_devices,
    is_mps_device,
    mlx_available,
    resolve_device,
    resolve_node_device,
    to_device,
)

mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
requires_mps = pytest.mark.skipif(not mps_available, reason="MPS not available")


# ── resolve_device ──────────────────────────────────────────────────

def test_resolve_device_cpu_passthrough():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_none_defaults_cpu():
    assert resolve_device(None) == "cpu"
    assert resolve_device("") == "cpu"


def test_resolve_device_cuda_falls_back_when_unavailable():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("cuda") == expected


def test_resolve_device_mps_matches_availability():
    expected = "mps" if mps_available else "cpu"
    assert resolve_device("mps") == expected


# ── is_mps_device ───────────────────────────────────────────────────

def test_is_mps_device_recognises_strings_and_torch_device():
    assert is_mps_device("mps") is True
    assert is_mps_device("mps:0") is True
    assert is_mps_device("cpu") is False
    assert is_mps_device("cuda") is False
    assert is_mps_device(torch.device("cpu")) is False


# ── to_device: device-agnostic behaviour (always runs) ──────────────

def test_to_device_cpu_is_noop_for_tensor():
    x = torch.randn(2, 3, dtype=torch.float64)
    y = to_device(x, "cpu")
    # No MPS → no coercion: dtype is preserved.
    assert y.dtype == torch.float64
    assert y.device.type == "cpu"


def test_to_device_passes_through_non_tensor_leaves():
    assert to_device(None, "mps") is None
    assert to_device(7, "mps") == 7
    assert to_device("hello", "mps") == "hello"


def test_to_device_recurses_into_tuple_and_dict_on_cpu():
    x = torch.randn(2, 2)
    t = torch.randint(0, 5, (2,))
    moved = to_device((x, t), "cpu")
    assert isinstance(moved, tuple) and len(moved) == 2
    d = to_device({"a": x, "b": 1}, "cpu")
    assert d["b"] == 1 and d["a"].shape == (2, 2)


# ── to_device: MPS float64 coercion (gated) ─────────────────────────

@requires_mps
def test_to_device_downcasts_float64_tensor_on_mps():
    x = torch.randn(2, 3, dtype=torch.float64)
    y = to_device(x, "mps")
    assert y.dtype == torch.float32
    assert y.device.type == "mps"


@requires_mps
def test_to_device_preserves_int64_on_mps():
    # CrossEntropy targets are int64 — MPS supports it, must not be touched.
    t = torch.randint(0, 10, (4,), dtype=torch.int64)
    ty = to_device(t, "mps")
    assert ty.dtype == torch.int64
    assert ty.device.type == "mps"


@requires_mps
def test_to_device_downcasts_float64_module_on_mps():
    model = nn.Linear(3, 2).double()  # all params float64
    moved = to_device(model, "mps")
    param = next(moved.parameters())
    assert param.dtype == torch.float32
    assert param.device.type == "mps"
    # And it can actually run a forward pass (input also routed through
    # to_device so the float64 source is downcast before the move).
    out = moved(to_device(torch.randn(1, 3, dtype=torch.float64), "mps"))
    assert out.shape == (1, 2)


@requires_mps
def test_inference_node_accepts_float64_input_on_mps():
    """The original bug: float64 tensor → Inference(mps) raised
    'Cannot convert a MPS Tensor to float64'. to_device now downcasts it."""
    from app.nodes.io.inference_node import InferenceNode

    model = nn.Linear(4, 2)  # float32 weights
    x = torch.randn(1, 4, dtype=torch.float64)  # float64 input
    res = InferenceNode().execute({"model": model, "input": x}, {"device": "mps"})
    assert res["output"].shape == (1, 2)
    assert res["output"].device.type == "mps"


def test_mps_listed_in_available_devices_when_present():
    devices = get_available_devices()
    assert "cpu" in devices
    assert ("mps" in devices) == mps_available


# ── resolve_device strictness ───────────────────────────────────────

def test_resolve_device_unknown_value_falls_back_to_cpu():
    # "auto" is meaningful only as a per-node param, never a global device.
    assert resolve_device("auto") == "cpu"
    assert resolve_device("bogus") == "cpu"


# ── context_device ──────────────────────────────────────────────────

class _Ctx:
    def __init__(self, device):
        self.device = device


def test_context_device_reads_device_off_context():
    assert context_device(_Ctx("mps")) == "mps"


def test_context_device_falls_back_without_context():
    assert context_device(None) == "cpu"
    assert context_device(_Ctx(None)) == "cpu"
    assert context_device(None, fallback="cuda") == "cuda"


# ── resolve_node_device ─────────────────────────────────────────────

def test_resolve_node_device_auto_follows_context():
    assert resolve_node_device("auto", _Ctx("mps" if mps_available else "cpu")) == (
        "mps" if mps_available else "cpu"
    )
    assert resolve_node_device(None, _Ctx("cpu")) == "cpu"
    assert resolve_node_device("", _Ctx("cpu")) == "cpu"


def test_resolve_node_device_explicit_overrides_global():
    # An explicit device wins over the global one (and is availability-checked).
    assert resolve_node_device("cpu", _Ctx("mps")) == "cpu"


# ── describe_accelerator ────────────────────────────────────────────

def test_describe_accelerator_shape_and_cpu_always_present():
    info = describe_accelerator()
    assert "default" in info and "devices" in info
    values = {d["value"] for d in info["devices"]}
    assert "cpu" in values
    # Every entry has the documented keys.
    for d in info["devices"]:
        assert {"value", "label", "detail", "available"} <= set(d)
    # default must be one of the listed devices.
    assert info["default"] in values


@requires_mps
def test_describe_accelerator_labels_mps_and_defaults_to_it():
    info = describe_accelerator()
    mps_entry = next(d for d in info["devices"] if d["value"] == "mps")
    assert mps_entry["label"] == "Apple MPS"
    # With only CPU + MPS available, MPS is the best default.
    if not torch.cuda.is_available():
        assert info["default"] == "mps"


def test_describe_accelerator_never_lists_mlx_as_a_torch_device():
    # MLX is not a torch backend — it must not appear in the device selector.
    info = describe_accelerator()
    assert "mlx" not in {d["value"] for d in info["devices"]}


# ── mlx_available (Phase 3 spike detection) ─────────────────────────

def test_mlx_available_returns_bool_matching_import():
    import importlib.util

    expected = importlib.util.find_spec("mlx") is not None
    assert mlx_available() is expected
