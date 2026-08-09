import logging
import re
from typing import TYPE_CHECKING, Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

if TYPE_CHECKING:
    # `pathlib` is imported lazily inside the methods below (startup cost);
    # this makes the "Path" annotations resolvable to a type checker without
    # paying that cost at import time.
    from pathlib import Path

logger = logging.getLogger(__name__)

#: Cache for :func:`torch_nn_layer_globals`. The walk is cheap but not free,
#: and its answer cannot change while the interpreter is running.
_LAYER_GLOBALS: list[type] | None = None

#: How torch names the class its restricted unpickler stopped on.
_UNSUPPORTED_GLOBAL = re.compile(r"Unsupported global: GLOBAL ([\w.]+)")


def torch_nn_layer_globals() -> list[type]:
    """Every ``nn.Module`` subclass that ``torch.nn`` itself defines.

    This is the allowlist that makes ``load_mode="full_model"`` possible at
    all (#222). ``torch.load(..., weights_only=True)`` restricts unpickling
    to tensors and a small set of primitives, and a full-model file is a
    pickled ``nn.Module`` -- so the two contradict each other and the mode
    failed for every input it was written to accept.
    ``torch.serialization.safe_globals`` widens the restricted unpickler to
    named classes without turning it off, which resolves the contradiction
    without giving up the guarantee: a payload that pickles ``os.system``
    is still refused, because ``os.system`` is not a torch layer.

    The standing objection to an allowlist is that it is a list somebody has
    to maintain. This one is DERIVED -- a walk over the loaded subclasses of
    ``nn.Module``, filtered to the ones torch defines -- so there is nothing
    written down to fall out of date, and it tracks whatever torch the user
    installed rather than whatever torch was current when this was written.

    Scope, deliberately: layer CLASSES, and only torch's own. Reconstructing
    one rebuilds a ``__dict__``; it does not call user code. Widening this to
    ``torch.nn.functional`` would admit a second category -- callables the
    unpickler may INVOKE -- whose boundary is genuinely fuzzy (``F.relu`` is
    arithmetic, ``handle_torch_function`` in the same namespace dispatches to
    an arbitrary object's ``__torch_function__``), so it is left out and its
    absence is reported by name. The visible cost is that a
    ``TransformerEncoderLayer`` stores its activation as ``F.relu`` and so
    does not load; CodefyUI's own transformer block is built from a
    function-local class and cannot be ``torch.save``d in the first place.
    """
    global _LAYER_GLOBALS
    if _LAYER_GLOBALS is not None:
        return _LAYER_GLOBALS

    import torch.nn as nn

    # ``import torch`` already imports every ``torch.nn.modules`` submodule,
    # so the walk sees classes the top-level namespace does not re-export --
    # ``MultiheadAttention``'s ``out_proj`` is a
    # ``NonDynamicallyQuantizableLinear``, reachable only as
    # ``torch.nn.modules.linear.NonDynamicallyQuantizableLinear``.
    allowed: dict[str, type] = {"torch.nn.modules.module.Module": nn.Module}
    pending: list[type] = [nn.Module]
    seen: set[type] = set()
    while pending:
        for subclass in pending.pop().__subclasses__():
            if subclass in seen:
                continue
            seen.add(subclass)
            pending.append(subclass)
            # Keyed on where the class says it is DEFINED, not on what it is
            # called: a plugin's ``MyPack.Linear`` is not torch's, and a
            # subclass of a torch layer defined anywhere else is somebody
            # else's code. ``__module__`` is self-reported and writable, so
            # this is not a boundary against code already running in the
            # process -- which needs no pickle to do anything. It is a
            # boundary against the FILE, which is the thing being read.
            if getattr(subclass, "__module__", "").startswith("torch.nn."):
                allowed[f"{subclass.__module__}.{subclass.__qualname__}"] = subclass

    _LAYER_GLOBALS = list(allowed.values())
    return _LAYER_GLOBALS


def _full_model_refusal(p: "Path", exc: Exception) -> str:
    """The message a user gets when a full-model file will not load.

    The complaint in #222 was not only that the mode failed, it was that it
    failed unreadably: a legitimate-looking dropdown entry produced a raw
    unpickler traceback with no hint that a safe default was involved, what
    it stopped on, or what to do instead. All three go in the message,
    because the engine surfaces ``str(exception)`` and keeps the traceback
    only under DEBUG.
    """
    refused = _UNSUPPORTED_GLOBAL.search(str(exc))
    if refused is not None:
        detail = (
            f"it contains {refused.group(1)}, which is not one of the "
            f"torch.nn layer classes CodefyUI will reconstruct"
        )
    else:
        detail = f"{type(exc).__name__}: {exc}"

    return (
        f"full_model mode could not load {p.name}: {detail}.\n"
        "\n"
        "A full-model file is a pickle, so loading one runs whatever is in "
        "it. CodefyUI therefore reads it under torch's restricted unpickler "
        "(weights_only=True) and does not turn that off; the restriction is "
        "widened just far enough to rebuild models made of stock torch.nn "
        "layers, and everything else is refused rather than executed.\n"
        "\n"
        "Two ways forward:\n"
        "  1. Use a state_dict, which is the supported path and this node's "
        "default. Save with ModelSaver (save_mode=state_dict), then wire the "
        "architecture into this node's 'model' input and set "
        "load_mode=state_dict.\n"
        "  2. If you trust this file, convert it once outside CodefyUI:\n"
        "     torch.save(torch.load(PATH, weights_only=False).state_dict(), "
        "NEW_PATH)\n"
        "     and load NEW_PATH with load_mode=state_dict. Only do this for "
        "a file you produced or got from a source you trust -- that "
        "weights_only=False is the arbitrary-code-execution step, which is "
        "why it is not something CodefyUI will do for you."
    )


class ModelLoaderNode(BaseNode):
    NODE_NAME = "ModelLoader"
    CATEGORY = "IO"
    DESCRIPTION = "Load model weights from a .pt/.pth file into a model, or load a full saved model"

    # #254. In ``state_dict`` mode this node's product is a MUTATION of the
    # model it was handed -- ``load_state_dict`` writes in place -- and a
    # cache hit returns the recorded outputs without calling execute(), so
    # the load simply does not happen. Measured on the realistic shape
    # (pretrained weights -> fine-tune), fed by a cacheable model source and
    # run three times against one ExecutionCache: 1 / 0 / 0 real execute()
    # calls, and TrainingLoop saw the file's weight only on run 1
    # (0.05 -> 0.085 -> 0.157) -- every later run silently continued from
    # where the last one stopped instead of restarting from the file.
    #
    # In ``full_model`` mode there is no input to mutate, but the module it
    # returns is a live handle that downstream training mutates in place;
    # replaying it hands run 2 run 1's already-trained network. That is
    # exactly the ``SequentialModel`` bug #253 fixed, one node over.
    #
    # This DOES undo the ``cacheable = True`` #144 gave it -- measured, and
    # smaller than it sounds. The engine refuses to cache a node with any
    # non-cacheable upstream, and ``state_dict`` mode's model comes from a
    # weight-owning node, every one of which is non-cacheable; on the
    # shipped shape it measured 1 / 1 / 1 execute() calls across three runs
    # BEFORE this change, i.e. the hit #144 re-enabled was already
    # unreachable there. #144's fingerprint mechanism is untouched and
    # still serves the reader nodes it was built for (Dataset, CSVReader,
    # ImageReader ...).
    cacheable = False

    @staticmethod
    def _resolve_path(path: str) -> "Path":
        from pathlib import Path

        from ...config import settings

        p = Path(path)
        if not p.is_absolute():
            p = settings.MODELS_DIR / p
        return p.resolve()

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="model",
                data_type=DataType.MODEL,
                description="Model architecture to load weights into (required for state_dict mode)",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Model with loaded weights"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="path",
                param_type=ParamType.MODEL_FILE,
                default="",
                description="Path to the weights file (.pt, .pth, .safetensors)",
            ),
            ParamDefinition(
                name="load_mode",
                param_type=ParamType.SELECT,
                default="state_dict",
                description=(
                    "Load mode: state_dict (requires model input) or full_model. "
                    "full_model rebuilds the saved module itself and is read under "
                    "torch's restricted unpickler, so it accepts models made of "
                    "stock torch.nn layers and refuses anything else"
                ),
                options=["state_dict", "full_model"],
            ),
            ParamDefinition(
                name="device",
                param_type=ParamType.SELECT,
                default="auto",
                description="Device to load weights onto ('auto' follows the global device)",
                options=["auto", "cpu", "cuda", "mps"],
            ),
            ParamDefinition(
                name="strict",
                param_type=ParamType.BOOL,
                default=True,
                description="Whether to strictly enforce that the keys in state_dict match (state_dict mode only)",
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        import torch

        from ...config import settings

        from ...core.device_utils import is_mps_device, resolve_node_device, to_device

        path = params.get("path", "model_weights.pt")
        load_mode = params.get("load_mode", "state_dict")
        device = resolve_node_device(params.get("device"), context)
        strict = params.get("strict", True)

        # MPS can't receive float64 via map_location, so stage doubles on CPU
        # and let to_device downcast them on the way to the device.
        load_device = "cpu" if is_mps_device(device) else device

        p = self._resolve_path(path)

        # Restrict reads to project data directory
        data_root = settings.MODELS_DIR.parent.resolve()
        if not p.is_relative_to(data_root):
            raise ValueError("Weights file path must be within the project data directory")

        if not p.exists():
            raise FileNotFoundError(f"Weights file not found: {p}")

        is_safetensors = p.suffix == ".safetensors"

        if load_mode == "state_dict":
            model = inputs.get("model")
            if model is None:
                raise ValueError(
                    "state_dict mode requires a model input. "
                    "Connect a SequentialModel or other model node, or use full_model mode."
                )
            if is_safetensors:
                from safetensors.torch import load_file
                state_dict = load_file(str(p), device=load_device)
            else:
                state_dict = torch.load(str(p), map_location=load_device, weights_only=True)
            model.load_state_dict(state_dict, strict=strict)
            model = to_device(model, device)
            param_count = sum(p_.numel() for p_ in model.parameters())
            logger.info("Loaded state_dict from %s (%s parameters, strict=%s)", p, f"{param_count:,}", strict)
        else:
            if is_safetensors:
                raise ValueError("safetensors format only supports state_dict mode, not full_model")
            model = self._load_full_model(p, load_device)
            model = to_device(model, device)
            logger.info("Loaded full model from %s (%s)", p, type(model).__name__)

        return {"model": model}

    @staticmethod
    def _load_full_model(p: "Path", load_device: Any) -> Any:
        """Unpickle a whole ``nn.Module``, without unpickling anything else.

        ``weights_only=True`` stays on. It is widened -- for this call only,
        which is what the context manager is for -- to the layer classes in
        :func:`torch_nn_layer_globals`, and to nothing else.
        """
        import torch

        try:
            with torch.serialization.safe_globals(torch_nn_layer_globals()):
                return torch.load(str(p), map_location=load_device, weights_only=True)
        except Exception as exc:
            raise ValueError(_full_model_refusal(p, exc)) from exc
