import logging
import pickle
import sys
from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

logger = logging.getLogger(__name__)


def _is_nameable(cls: type) -> bool:
    """True when ``pickle`` can write a reference to *cls*.

    Pickle stores a class by NAME -- ``__module__`` plus ``__qualname__`` --
    and reconstructs it by importing the module and walking the attribute
    path. This mirrors that walk, so it answers the same question pickle is
    about to ask, and a class it cannot reach is one pickle cannot either.
    The usual reason is a class defined inside a function, whose qualname
    carries a literal ``<locals>`` segment that is not an attribute of
    anything.
    """
    obj: Any = sys.modules.get(getattr(cls, "__module__", "") or "")
    if obj is None:
        return False
    for part in getattr(cls, "__qualname__", "").split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return False
    return obj is cls


def _unnameable_module_classes(model: Any) -> list[type]:
    """Every submodule class in *model* that ``torch.save`` cannot name.

    Returned in encounter order and de-duplicated, so a model with forty
    copies of one bad layer reports it once.
    """
    modules = getattr(model, "modules", None)
    candidates = list(modules()) if callable(modules) else [model]

    offenders: list[type] = []
    for module in candidates:
        cls = type(module)
        if cls in offenders or _is_nameable(cls):
            continue
        offenders.append(cls)
    return offenders


def _full_model_save_refusal(offenders: list[type]) -> str:
    """The message a user gets when a model cannot be pickled whole (#283).

    #222 was the same complaint at the other end of the round trip: a
    legitimate-looking dropdown option produced a raw internal error naming
    an implementation detail. Before #283 this one read

        AttributeError: Can't pickle local object 'Reshape.__new__.<locals>.Mod'

    which tells a user nothing about what they did or what to do instead. The
    three wrappers that produced it are ordinary module-scope classes now, so
    this fires only for a custom node or plugin that builds its module inside
    a function -- but when it fires it has to say which class, why, and where
    to go.
    """
    named = ", ".join(sorted(f"{cls.__module__}.{cls.__qualname__}" for cls in offenders))
    plural = "es" if len(offenders) > 1 else ""

    return (
        f"full_model mode cannot save this model: the class{plural} "
        f"{named} {'are' if len(offenders) > 1 else 'is'} defined inside a "
        f"function, not at the top level of a module.\n"
        "\n"
        "A full-model file is a pickle, and pickle stores a class by name -- "
        "it writes down where to import the class from and expects to find "
        "it there again. A class created inside a function has no such name, "
        "so there is nothing to write and nothing to import back.\n"
        "\n"
        "Two ways forward:\n"
        "  1. Use save_mode=state_dict, this node's default and the supported "
        "path. It stores tensors rather than classes, so it does not care "
        "where a layer class was defined; ModelLoader (load_mode=state_dict) "
        "reads it back into the same architecture.\n"
        "  2. If this layer comes from a custom node or a plugin you control, "
        "move its nn.Module subclass out to module scope and construct it "
        "from __init__ arguments. That is what CodefyUI did to its own "
        "Reshape / SelectIndex / TransformerEncoder wrappers in core#283."
    )


def _refused_function_attributes(model: Any, allowed: set[type]) -> list[str]:
    """Functions and classes stored as attributes that the loader will refuse.

    The class walk in :func:`_reload_note` reads ``type(submodule)`` and so
    sees only the modules. It misses a plain attribute holding a FUNCTION,
    which the restricted unpickler refuses just as firmly -- and there is a
    shipped case: ``nn.TransformerEncoderLayer`` stores its activation as
    ``torch.nn.functional.relu``, whatever spelling it was constructed with, so
    both transformer wrappers fail to load even though every CLASS in them is
    admitted. #222 left ``torch.nn.functional`` out deliberately (a function is
    a callable the unpickler may be asked to INVOKE, and the namespace also
    holds ``handle_torch_function``, which dispatches to an arbitrary object's
    ``__torch_function__``), and #288 records that as a decision still open. So
    the note has to be able to say it, or it would promise a round trip on the
    one shipped shape that does not have one.

    Deliberately NOT a full audit of the pickle graph: that is a
    re-implementation of the unpickler, and the loader is the authority. It
    covers the category that actually occurs -- a function or a class as a
    direct attribute value -- and anything subtler still surfaces at load time
    with the message :func:`_full_model_refusal` writes.
    """
    import types

    modules = getattr(model, "modules", None)
    candidates = list(modules()) if callable(modules) else [model]

    refused: list[str] = []
    for module in candidates:
        # ``getattr(..., "__dict__")`` rather than ``vars()``: this walk does
        # NOT only see nn.Modules. ``ModelSaver.model`` is a MODEL port, and
        # ``type_system`` lets an ANY-typed output into one, so a dict, an int
        # or a None genuinely arrives here -- ``CheckpointLoader``'s
        # ``grad_scaler_state`` emits dict-or-None, and PythonScript / Switch /
        # Reduce / GraphInput can pass anything. ``vars(5)`` raises TypeError,
        # which would replace the note this function exists to write with a
        # traceback, BEFORE the torch.save that would have succeeded. Found in
        # review of #288; the pre-#288 walk read only ``type(module)`` and so
        # never touched an instance dict.
        for value in (getattr(module, "__dict__", None) or {}).values():
            if isinstance(value, type):
                if value in allowed:
                    continue
            elif not isinstance(
                value,
                types.FunctionType | types.BuiltinFunctionType | types.MethodType,
            ):
                continue
            name = (
                f"{getattr(value, '__module__', '?')}."
                f"{getattr(value, '__qualname__', value)}"
            )
            if name not in refused:
                refused.append(name)
    return refused


def _reload_note(model: Any) -> tuple[str, str]:
    """What to tell the user about reading this full-model file back.

    Returns ``(level, text)``; ``("", "")`` when there is nothing to say.

    Saving is only half a round trip, and until #288 the two halves did not
    accept the same set of models: ``ModelLoader(load_mode="full_model")`` read
    under torch's restricted unpickler widened to ``torch.nn``'s own classes
    and nothing else (#222), so a model containing CodefyUI's own
    ``GraphModelModule`` -- which is what every layer-editor model is -- saved
    fine and was refused on the way back in. #283 made that reachable and #287
    could only warn about it.

    #288 closed the loop: CodefyUI's own module classes are now on the
    allowlist, so there are three answers rather than two, and which one a
    model gets is derived from :func:`full_model_safe_globals` rather than
    written down here -- the note cannot drift from what the loader accepts.

    * stock torch layers only -- silence. The round trip works and always did;
      a note here would be noise, and noise is how a note stops being read.
    * plus CodefyUI's own classes -- an INFO note. The round trip works, but
      the file is no longer self-contained the way a pure-torch one is: reading
      it needs CodefyUI. Worth one line, because that is the difference between
      a file a colleague can open and one they cannot.
    * anything else (a custom node, a plugin, a hand-written class) -- a
      WARNING, and the same trap #287 named. Still advisory rather than a
      refusal: the file is valid and ``torch.load(..., weights_only=False)``
      reads it outside CodefyUI, which is a legitimate reason to write one.
    """
    from .model_loader_node import codefyui_module_globals, full_model_safe_globals

    allowed = set(full_model_safe_globals())
    ours = set(codefyui_module_globals())
    modules = getattr(model, "modules", None)
    candidates = list(modules()) if callable(modules) else [model]

    foreign: list[type] = []
    codefyui: list[type] = []
    for module in candidates:
        cls = type(module)
        if cls in ours:
            if cls not in codefyui:
                codefyui.append(cls)
        elif cls not in allowed and cls not in foreign:
            foreign.append(cls)

    refused = sorted(cls.__qualname__ for cls in foreign)
    refused += _refused_function_attributes(model, allowed)

    if refused:
        named = ", ".join(refused)
        plural = len(refused) > 1
        return "warning", (
            f"Saved, but note: this file cannot be read back by ModelLoader in "
            f"full_model mode. It contains {named}, which "
            f"{'are' if plural else 'is'} not among the things full_model "
            f"loading reconstructs -- torch.nn's own layer classes and "
            f"CodefyUI's own module classes, and no functions at all (#288). "
            f"The file is still valid outside CodefyUI (torch.load with "
            f"weights_only=False). For a round trip inside CodefyUI use "
            f"save_mode=state_dict and load it back with load_mode=state_dict "
            f"against the same architecture."
        )

    if codefyui:
        named = ", ".join(sorted(cls.__qualname__ for cls in codefyui))
        return "info", (
            f"Saved. ModelLoader reads this file back in full_model mode: it "
            f"contains {named}, which CodefyUI defines and this version admits "
            f"to torch's restricted unpickler (#288). That does mean the file "
            f"is not self-contained -- a CodefyUI older than this one refuses "
            f"it, and outside CodefyUI torch.load needs weights_only=False "
            f"and CodefyUI's backend package on sys.path, because pickle "
            f"imports a class by name. save_mode=state_dict produces a file "
            f"with none of those conditions on it."
        )

    return "", ""


class ModelSaverNode(BaseNode):
    NODE_NAME = "ModelSaver"
    CATEGORY = "IO"
    DESCRIPTION = "Save model weights (state_dict) to a .pt/.pth/.safetensors file"

    # The write to disk IS this node's output. A cache hit returns the
    # recorded {"path": ..., "model": ...} without calling execute() again,
    # which is correct for a pure node and wrong here: if the user deleted
    # the file (or it never ran with Rec off), a cache hit would hand back a
    # path that no longer points at anything (#143).
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Trained model to save"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="path", data_type=DataType.STRING, description="Path to the saved file"),
            PortDefinition(name="model", data_type=DataType.MODEL, description="Pass-through model (for chaining)"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="path",
                param_type=ParamType.STRING,
                default="model_weights.pt",
                description="Output file path (.pt, .pth, or .safetensors)",
            ),
            ParamDefinition(
                name="save_mode",
                param_type=ParamType.SELECT,
                default="state_dict",
                description="Save mode: state_dict (recommended) or full model",
                options=["state_dict", "full_model"],
            ),
            ParamDefinition(
                name="format",
                param_type=ParamType.SELECT,
                default="pytorch",
                description="File format: pytorch (.pt/.pth) or safetensors (.safetensors)",
                options=["pytorch", "safetensors"],
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        from ...config import settings
        from ...core.data_paths import resolve_data_path

        model = inputs["model"]
        path = params.get("path", "model_weights.pt")
        save_mode = params.get("save_mode", "state_dict")
        fmt = params.get("format", "pytorch")
        note = ""
        note_level = ""

        # Inside the data directory, and not over CodefyUI's own storage --
        # ``core.data_paths`` owns both halves of that rule and is shared
        # with ImageWriter and the checkpoint writers (#224).
        p = resolve_data_path(path, base=settings.MODELS_DIR)

        if fmt == "safetensors":
            if save_mode == "full_model":
                raise ValueError("safetensors format only supports state_dict mode, not full_model")
            if p.suffix not in (".safetensors",):
                # Re-validated: the path written must be the path checked.
                # DB_PATH is env-overridable, so a database named
                # `store.safetensors` is reachable by asking for `store.pt`
                # and letting this rewrite land on it (#224).
                p = resolve_data_path(p.with_suffix(".safetensors"),
                                      base=settings.MODELS_DIR)

        p.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "safetensors":
            from safetensors.torch import save_file
            save_file(model.state_dict(), str(p))
            param_count = sum(p_.numel() for p_ in model.parameters())
            logger.info("Saved safetensors to %s (%s parameters)", p, f"{param_count:,}")
        elif save_mode == "state_dict":
            torch.save(model.state_dict(), str(p))
            param_count = sum(p_.numel() for p_ in model.parameters())
            logger.info("Saved state_dict to %s (%s parameters)", p, f"{param_count:,}")
        else:
            # Asked BEFORE the write, so a refusal leaves no half-written
            # file behind and can name the class rather than relay whatever
            # the pickler happened to trip on first (#283).
            offenders = _unnameable_module_classes(model)
            if offenders:
                raise ValueError(_full_model_save_refusal(offenders))

            note_level, note = _reload_note(model)

            try:
                torch.save(model, str(p))
            except (AttributeError, TypeError, pickle.PicklingError) as exc:
                # The residual cases the class walk cannot see -- a lambda or
                # a local function stored as an attribute, say. Same three
                # things in the message, minus the name we do not have.
                raise ValueError(
                    f"full_model mode could not pickle this model: "
                    f"{type(exc).__name__}: {exc}.\n"
                    "\n"
                    "A full-model file is a pickle, so every object hanging "
                    "off the module has to be one pickle can name -- a "
                    "lambda, a local function or a class defined inside a "
                    "function is not.\n"
                    "\n"
                    "Use save_mode=state_dict, this node's default and the "
                    "supported path: it stores tensors rather than the "
                    "objects around them."
                ) from exc

            logger.info("Saved full model to %s", p)
            if note_level == "warning":
                logger.warning("%s", note)
            elif note_level == "info":
                # INFO, not WARNING: since #288 this case is the round trip
                # WORKING, and a warning on a success is how a log stops being
                # a signal.
                logger.info("%s", note)

        result: dict[str, Any] = {"path": str(p), "model": model}
        if note:
            # ``__log__`` is the one result key the canvas Log tab renders,
            # and dunder keys are filtered out of recorded outputs and port
            # summaries -- so this adds a line the user can read and nothing
            # else (same mechanism as SequentialModel's resume note).
            result["__log__"] = note
        return result
