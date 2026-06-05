"""Utilities for detecting and targeting PyTorch devices at runtime."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_available_devices() -> list[str]:
    """Return the list of available PyTorch devices.

    Always includes "cpu". Adds "cuda" if torch.cuda.is_available(),
    "mps" if torch.backends.mps.is_available(). If torch is not installed,
    only "cpu" is returned.
    """
    devices = ["cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            devices.append("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append("mps")
    except ImportError:
        pass
    return devices


def resolve_device(requested: str | None) -> str:
    """Resolve a requested device string to one that is actually available.

    Falls back to "cpu" (with a warning) when "cuda"/"mps" is requested but
    unavailable. Centralizes the availability check that the device-aware
    "sink" nodes (Training/Inference/Checkpoint/ModelLoader) used to each
    duplicate inline.
    """
    device = (requested or "cpu").strip().lower() or "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"

    if device == "cpu":
        return "cpu"
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return "cpu"
        return device
    if device.startswith("mps"):
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("MPS not available, falling back to CPU")
            return "cpu"
        return device
    # Unknown value (e.g. "auto" sent as a global device) — never hand an
    # invalid string to torch; degrade to CPU.
    logger.warning("Unknown device %r, falling back to CPU", device)
    return "cpu"


def is_mps_device(device: Any) -> bool:
    """True when `device` names the Apple MPS backend (str or torch.device)."""
    try:
        import torch
    except ImportError:
        return False
    if isinstance(device, torch.device):
        return device.type == "mps"
    return isinstance(device, str) and device.startswith("mps")


def resolve_node_device(param_value: str | None, context: Any) -> str:
    """Resolve a sink node's ``device`` param against the global run device.

    ``"auto"`` (or empty) means "follow the global device" (``context.device``,
    already resolved). An explicit ``"cpu"/"cuda"/"mps"`` overrides the global
    setting and is availability-checked via :func:`resolve_device`. This lets a
    saved graph pin a node to a device while fresh nodes default to ``"auto"``
    and ride the global selector.
    """
    value = (param_value or "auto").strip().lower() or "auto"
    if value == "auto":
        return context_device(context)
    return resolve_device(value)


def context_device(context: Any, fallback: str = "cpu") -> str:
    """Read the resolved global device off an ExecutionContext.

    Returns ``fallback`` when there is no context or no device set (e.g. the
    CLI runner passes ``context=None``), so device-aware nodes degrade to CPU.
    The value stored on the context is already ``resolve_device``-d at the
    execution entry point, so it is safe to use directly.
    """
    dev = getattr(context, "device", None)
    return dev or fallback


def mlx_available() -> bool:
    """True when Apple's native **MLX** framework is importable.

    MLX is a *separate* array framework, not a PyTorch backend — there is no
    ``tensor.to("mlx")``. It is surfaced for the inference-subset spike
    (see ``scripts/mlx_spike.py``), not as a selectable torch execution device.
    Apple acceleration in the graph engine is provided by PyTorch **MPS**.
    """
    import importlib.util

    return importlib.util.find_spec("mlx") is not None


def describe_accelerator() -> dict[str, Any]:
    """Describe the compute devices available, with human-friendly labels.

    Distinguishes **AMD ROCm** from **NVIDIA CUDA** (both surface through the
    ``torch.cuda`` API — tell them apart via ``torch.version.hip``) and labels
    Apple **MPS**. Shape::

        {
          "default": "mps",                      # best available
          "devices": [
            {"value": "cpu", "label": "CPU", "detail": "...", "available": true},
            {"value": "mps", "label": "Apple MPS", "detail": "Metal", ...},
          ],
        }

    Falls back to CPU-only when torch is missing.
    """
    cpu_entry = {"value": "cpu", "label": "CPU", "detail": "", "available": True}
    devices: list[dict[str, Any]] = [cpu_entry]
    default = "cpu"

    try:
        import torch
    except ImportError:
        return {"default": default, "devices": devices}

    if torch.cuda.is_available():
        is_rocm = getattr(torch.version, "hip", None) is not None
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001 — name lookup is best-effort
            name = ""
        devices.append({
            "value": "cuda",
            "label": "AMD ROCm" if is_rocm else "NVIDIA CUDA",
            "detail": name,
            "available": True,
        })
        default = "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append({
            "value": "mps",
            "label": "Apple MPS",
            "detail": "Metal Performance Shaders",
            "available": True,
        })
        if default == "cpu":
            default = "mps"

    return {"default": default, "devices": devices}


def _downcast_float64_module(module: "torch.nn.Module") -> None:
    """In-place downcast a module's float64 params/buffers to float32.

    MPS rejects float64, so any double-precision parameter or buffer must be
    converted before the module is moved to an MPS device. Other floating
    dtypes (float16/bfloat16) and integer buffers are left untouched.
    """
    import torch

    for mod in module.modules():
        for param in mod.parameters(recurse=False):
            if param.dtype == torch.float64:
                param.data = param.data.to(torch.float32)
                if param.grad is not None and param.grad.dtype == torch.float64:
                    param.grad.data = param.grad.data.to(torch.float32)
        for name, buf in list(mod._buffers.items()):
            if buf is not None and buf.dtype == torch.float64:
                mod._buffers[name] = buf.to(torch.float32)


def to_device(obj: Any, device: Any) -> Any:
    """Move a tensor / module / (nested) collection to `device`.

    When targeting MPS, float64 values are downcast to float32 first, because
    MPS raises "Cannot convert a MPS Tensor to float64 dtype as the MPS
    framework doesn't support float64." Tensors of other dtypes (e.g. int64
    targets) and non-tensor leaves pass through unchanged. Lists/tuples/dicts
    are mapped element-wise so a ``(data, targets)`` batch can be moved in one
    call.
    """
    import torch

    if obj is None:
        return obj

    mps = is_mps_device(device)

    if isinstance(obj, torch.Tensor):
        if mps and obj.dtype == torch.float64:
            obj = obj.to(torch.float32)
        return obj.to(device)

    if isinstance(obj, torch.nn.Module):
        if mps:
            _downcast_float64_module(obj)
        return obj.to(device)

    if isinstance(obj, (list, tuple)):
        moved = [to_device(v, device) for v in obj]
        return type(obj)(moved)

    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}

    # Unknown leaf (int, str, sklearn model, ...) — best effort .to(), else pass.
    return obj
