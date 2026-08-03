"""Utilities for detecting and targeting PyTorch devices at runtime."""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

#: Every device string torch will accept from this application, and the
#: only shapes :func:`resolve_device` passes through. Anything else is a
#: typo, an unexpanded shell variable, or a hand-edited graph -- and
#: reaching torch with it produces ``RuntimeError: Invalid device string``,
#: an error that names neither the graph nor the parameter it came from.
#:
#: Deliberately stricter than "does it start with cuda": ``cuda:``,
#: ``cuda:abc``, ``cuda:0:1``, ``cuda:1e3`` and ``cuda: 0`` all read as
#: CUDA to a ``startswith`` check and none of them is addressable. Note
#: that ``int(" 0")`` is 0 in Python, so the space in the last one survives
#: a parse-and-check approach and only a syntax check catches it.
#:
#: Mirrors ``run_service.DEVICE_PATTERN``, which validates the same
#: vocabulary at submit time; this one is the last line, covering the
#: entry points that never pass through a run submission -- the exported
#: script's ``--device``, a hand-edited SELECT param (nothing validates
#: option values at runtime), and a direct ``execute_graph``.
DEVICE_SYNTAX = re.compile(r"^(cpu|cuda|mps)(?::(\d+))?$")


def _current_cuda_index() -> int:
    """The index a bare ``cuda`` means in THIS process.

    ``torch.cuda.current_device()``, because that is the index torch itself
    would pick; hardcoding 0 would be wrong in a process that changed it.
    ``run_service.canonical_queue_key`` imports this rather than keeping a
    second copy, so the queue and the runtime always agree on which card a
    bare ``cuda`` means.
    """
    try:
        import torch

        return int(torch.cuda.current_device())
    except Exception:  # noqa: BLE001 - only reached with CUDA already checked
        logger.debug("could not read the current CUDA device", exc_info=True)
        return 0


def _mps_available() -> bool:
    """True when Apple's MPS backend is present AND usable."""
    try:
        import torch

        return bool(hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001 - no torch, no backend
        return False


def cuda_device_count() -> int:
    """How many CUDA devices this process can see. 0 when there is no CUDA.

    Not ``lru_cache``d, unlike :func:`get_available_devices`: a test that
    monkeypatches ``torch.cuda.device_count`` to pretend there are four
    cards must see its own answer, and the call is a cheap read of a value
    torch itself caches after the first driver query.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001 - no torch, no driver, no devices
        return 0


def split_device(device: str) -> tuple[str, int | None]:
    """``"cuda:1"`` -> ``("cuda", 1)``; ``"cuda"`` -> ``("cuda", None)``.

    ``None`` for the index means "whichever one torch is currently pointed
    at", which is a different statement from ``0`` and has to stay
    distinguishable.

    **A lenient parser, not a validator.** An unreadable suffix
    (``"cuda:abc"``) also yields ``None``, which is the same answer it
    gives for no suffix at all -- so this cannot be used to decide whether
    a string is addressable. :data:`DEVICE_SYNTAX` and
    :func:`resolve_device` are what answer that question; this exists for
    callers that only want the KIND, and for the fallback path that has
    already established the string is malformed.
    """
    kind, separator, index = (device or "").partition(":")
    if not separator:
        return kind, None
    try:
        return kind, int(index)
    except ValueError:
        return kind, None


@lru_cache(maxsize=1)
def get_available_devices() -> list[str]:
    """Return the list of available PyTorch devices.

    Always includes "cpu". Adds "cuda" if torch.cuda.is_available(),
    "mps" if torch.backends.mps.is_available(). If torch is not installed,
    only "cpu" is returned.

    On a machine with more than one CUDA device the per-index forms
    (``cuda:0``, ``cuda:1``, ...) are listed as well, so a node can be
    pinned to a specific card (core#135). A SINGLE-GPU box deliberately
    lists only the bare ``cuda``: there ``cuda`` and ``cuda:0`` name the
    same piece of hardware, and offering both is a choice with no meaning
    behind it.
    """
    devices = ["cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            devices.append("cuda")
            count = cuda_device_count()
            if count > 1:
                devices.extend(f"cuda:{i}" for i in range(count))
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

    **Out-of-range index (core#135).** ``cuda:3`` on a two-card box, or on
    the laptop a colleague opens the saved graph on, degrades to the
    CURRENT cuda device rather than to the CPU. Both are guesses, and this
    is the one that respects what the user asked for: they said "train on a
    GPU", and answering that with a forty-minute CPU run is a worse
    surprise than answering it with the only GPU present. The warning names
    the count so the substitution is visible in the log. ``mps`` has no
    index vocabulary beyond ``mps:0``, and is normalised the same way.

    **Malformed index.** ``cuda:``, ``cuda:abc``, ``cuda:0:1``, ``cuda:1e3``
    and ``cuda: 0`` are all rejected by :data:`DEVICE_SYNTAX` and treated
    exactly like an out-of-range one: the ``cuda`` prefix is still an
    unambiguous request for a GPU, so the fallback respects it and the
    warning names what was wrong. Before this they were returned VERBATIM
    and torch answered with ``RuntimeError: Invalid device string``, naming
    neither the graph nor the parameter the string came from. They reach
    here through the exported script's free-form ``--device`` and through a
    hand-edited SELECT param, neither of which passes a run submission's
    validation.

    The value returned is always REBUILT from what was understood, never
    echoed. That is what makes ``cuda: 0`` safe: the space survives
    ``int()``, so a parse-and-check would accept it and hand torch a string
    it rejects.

    Never raises. Every caller -- node params, run options, the exported
    script, the CLI -- treats this as a total function that always yields a
    string torch will accept.
    """
    device = (requested or "cpu").strip().lower() or "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"

    if device == "cpu":
        return "cpu"

    syntax = DEVICE_SYNTAX.match(device)
    if syntax is None:
        # The kind is stripped so ``"cuda : 0"`` degrades the same way
        # ``"cuda: 0"`` does; both are the same mistake.
        kind, _ = split_device(device)
        kind = kind.strip()
        if kind == "cuda" and torch.cuda.is_available():
            current = _current_cuda_index()
            logger.warning(
                "%r is not a usable device string (expected cuda or "
                "cuda:N); using cuda:%d instead.", device, current,
            )
            # The SAME landing place as an out-of-range index, deliberately:
            # both are "you asked for a GPU and named it wrong", and two
            # spellings of one recovery would be two behaviours to learn.
            return f"cuda:{current}"
        if kind == "mps" and _mps_available():
            logger.warning(
                "%r is not a usable device string (expected mps or mps:0); "
                "using mps instead.", device,
            )
            return "mps"
        logger.warning("Unknown device %r, falling back to CPU", device)
        return "cpu"

    kind = syntax.group(1)
    index = int(syntax.group(2)) if syntax.group(2) is not None else None
    if kind == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return "cpu"
        count = cuda_device_count()
        if index is None:
            return "cuda"
        if not (0 <= index < count):
            current = _current_cuda_index()
            logger.warning(
                "CUDA device index %d was requested but this machine has %d "
                "(valid indices 0..%d); using cuda:%d instead.",
                index, count, max(count - 1, 0), current,
            )
            return f"cuda:{current}"
        return f"cuda:{index}"
    if kind == "mps":
        if not _mps_available():
            logger.warning("MPS not available, falling back to CPU")
            return "cpu"
        if index is not None and index != 0:
            logger.warning(
                "MPS has a single device; mps:%d is not addressable, using "
                "mps instead.", index,
            )
            return "mps"
        return device
    # Unknown value (e.g. "auto" sent as a global device) — never hand an
    # invalid string to torch; degrade to CPU.
    logger.warning("Unknown device %r, falling back to CPU", device)
    return "cpu"


def device_options(param_name: str, options: list[str]) -> list[str]:
    """The ``device`` SELECT vocabulary this machine can actually offer.

    Two jobs, in this order:

    * drop backends that are not present -- a CUDA option on a laptop with
      no CUDA is an invitation to a run that silently lands on the CPU;
    * expand ``cuda`` into the per-index forms when there is more than one
      card (core#135), so a node can be pinned to ``cuda:1`` without the
      user hand-editing the graph JSON.

    ``"auto"`` (follow the global device selector) is not a backend and
    bypasses both. A node that declares no device options, or a param that
    is not called ``device``, is returned untouched.
    """
    if param_name != "device" or not options:
        return options
    available = set(get_available_devices())
    filtered: list[str] = []
    for option in options:
        if option != "auto" and option not in available:
            continue
        filtered.append(option)
        if option == "cuda":
            # Generated from the count rather than sorted out of
            # ``available``: a lexical sort puts cuda:10 before cuda:2.
            filtered.extend(
                f"cuda:{i}" for i in range(cuda_device_count())
                if f"cuda:{i}" in available
            )
    return filtered if filtered else ["cpu"]


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
        label = "AMD ROCm" if is_rocm else "NVIDIA CUDA"
        count = cuda_device_count()

        def _name(index: int) -> str:
            try:
                return torch.cuda.get_device_name(index)
            except Exception:  # noqa: BLE001 — name lookup is best-effort
                return ""

        devices.append({
            "value": "cuda",
            "label": label,
            "detail": _name(_current_cuda_index()),
            "available": True,
        })
        # Per-card entries only when there is a choice to make (core#135).
        # On a single-GPU box ``cuda`` and ``cuda:0`` are the same hardware,
        # and a selector offering both makes the user pick between two
        # spellings of one answer.
        if count > 1:
            devices.extend({
                "value": f"cuda:{index}",
                "label": f"{label} #{index}",
                "detail": _name(index),
                "available": True,
            } for index in range(count))
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
