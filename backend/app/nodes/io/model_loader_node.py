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

#: Cache for :func:`codefyui_module_globals`, for the same reason.
_CODEFYUI_GLOBALS: list[type] | None = None

#: Cache for :func:`torch_function_globals`, for the same reason.
_FUNCTION_GLOBALS: list[Any] | None = None

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
    one rebuilds a ``__dict__``; it does not call user code. The NAMESPACE
    ``torch.nn.functional`` is still not admitted and never will be by this
    walk -- ``handle_torch_function`` lives there and dispatches to an
    arbitrary object's ``__torch_function__``. The two activation functions
    torch's own transformer layers store as attributes are admitted by exact
    identity instead; see :data:`_TORCH_FUNCTION_NAMES`.

    This is one of the three parts of the allowlist. CodefyUI's own module
    classes are the second and cannot be derived the same way -- see
    :func:`codefyui_module_globals`; the two torch functions are the third.
    :func:`full_model_safe_globals` joins them, and is what the loader and
    ``ModelSaver``'s save-time note both read.
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


# ── The curated half of the allowlist (#288) ─────────────────────────────
#
# #288's decision: a ``full_model`` file CodefyUI wrote has to load back into
# CodefyUI. Until it, the two halves of the round trip did not meet. #222 made
# the loader accept ``torch.nn``'s own classes and nothing else; #283 made
# CodefyUI's own module classes saveable. The product therefore wrote a file it
# then refused, and #287 could only paper over that with a save-time advisory.
#
# What makes widening affordable rather than reckless is WHERE this runs.
# "CodefyUI is a desktop tool that happens to speak HTTP" (docs/usage/
# shared-instances.md), deployed on localhost or behind a proxy on an intranet
# (docs/usage/deployment.md) -- there is no public upload endpoint, and the
# files an instance is asked to open are overwhelmingly the ones this install,
# or a colleague's install, wrote. That is an argument for widening the gate by
# one carefully chosen set of NAMES. It is not an argument for opening it:
# ``weights_only=True`` stays on, and #222's detonating payload still detonates
# nothing, because ``os.system`` is not on this list either.
#
# THE ADMISSION RULE, which is the whole safety argument. It has TWO halves,
# and an earlier draft of this comment stated only the first -- corrected in
# review of #288, because a future auditor following the wrong rule would admit
# the wrong class.
#
#   1. Reconstruction must only restore attributes. No ``__reduce__``, no
#      ``__setstate__``, no ``__getnewargs__`` -- nothing that turns restoring
#      an attribute into running something.
#
#   2. The CONSTRUCTOR must be safe to run with arbitrary, file-chosen
#      arguments. Read torch's ``_weights_only_unpickler``: the REDUCE opcode
#      does ``result = func(*args)`` for anything in the user-allowed globals
#      (torch 2.11, ``_weights_only_unpickler.py`` around :415). So admitting a
#      class does NOT only admit ``cls.__new__(cls)`` plus a ``__dict__``
#      update -- a crafted file can call ``cls(...)`` with values it chose.
#      Admissible therefore means: no filesystem, network or process side
#      effects, no mutation of global state (``torch.manual_seed`` and friends
#      -- a local ``torch.Generator`` is fine, which is what the seeded
#      constructors here use), no dynamic code (``eval`` / ``exec`` /
#      ``__import__``). Bad arguments raising, or allocating a large tensor,
#      is acceptable: that is a failed load, not a compromised one.
#
# Note that half 2 was ALREADY true of the torch half of the allowlist (#222) --
# admitting ``nn.Linear`` admits ``nn.Linear(...)`` on file-chosen sizes. It is
# not something #288 introduced; it is something #288's comment has to state,
# because this is now a list a human maintains.
#
# Every class below was audited against both halves, with the result recorded
# next to it. ``test_every_admitted_class_is_safe_to_reconstruct`` re-derives
# what is mechanically checkable -- the absence of pickle hooks, and the
# absence of the dangerous-call list from each class's constructor and the
# module-level helpers it reaches -- so a future edit that adds a
# ``__setstate__``, or an ``open()`` to one of these constructors, fails rather
# than quietly turning a name on this list into a code path. The part no test
# can prove (that running the constructor on nonsense arguments is merely
# useless) was hand-audited on 2026-08-12.
#
# EXACT IDENTITIES, never a module prefix. ``app.custom_nodes.*`` is code the
# user uploaded and ``cdui_plugins.*`` is code they installed; neither has been
# through review, both are deliberately outside this list, and an ``app.``
# prefix filter -- the obvious shortcut -- would have admitted the first of
# them. What is listed is what is admitted.
#
# Recorded as NAMES rather than imported at module scope, for two reasons. The
# node registry imports this module to read its metadata, and importing the
# LLM / VLA / diffusion node modules from here would drag ``torch`` onto that
# path -- the lazy-import property #283 went out of its way to preserve.
# Second, resolving at call time makes the answer independent of import ORDER:
# the torch half is a walk over LOADED subclasses, and deriving this half the
# same way would admit or refuse the same file depending on which nodes the
# session happened to have touched.
#
# The list is enumerated from the SAVE side: every ``nn.Module`` subclass
# ``app.nodes`` defines, because any of them can reach ``ModelSaver`` -- as the
# model itself, as a submodule of one, or (for the two loss modules) through an
# ANY-typed hop, which ``type_system`` allows into a MODEL port.
# ``test_the_allowlist_covers_every_codefyui_module_class`` fails when a new
# one is added and not audited, so this cannot rot quietly into the trap #288
# was opened about.
_CODEFYUI_MODULE_CLASSES: tuple[tuple[str, str], ...] = (
    # What every layer-editor model IS, and so #288's headline case. Audited:
    # holds an ``nn.ModuleDict`` of layers plus the plain dicts / lists /
    # tuples / strings describing the DAG. No pickle hooks.
    ("app.nodes.utility.graph_model", "GraphModelModule"),

    # The seven wrappers #283 hoisted out of a function into module scope.
    # Audited, all seven: each holds one torch submodule and/or the ints and
    # strings it was constructed from. No pickle hooks.
    ("app.nodes.utility.sequential_modules", "Reshape"),
    ("app.nodes.utility.sequential_modules", "SelectIndex"),
    ("app.nodes.utility.sequential_modules", "TransformerEncoderBlock"),
    ("app.nodes.utility.sequential_modules", "TransformerDecoderBlock"),
    ("app.nodes.utility.sequential_modules", "LSTMBlock"),
    ("app.nodes.utility.sequential_modules", "GRUBlock"),
    ("app.nodes.utility.sequential_modules", "MultiHeadAttentionBlock"),

    # The LLM wave's model (#289) and the three classes it is built from --
    # admitting the outer class alone would refuse the file one level down.
    # Audited: ints, floats, strings, bools, torch submodules, parameters and
    # buffers. No pickle hooks.
    ("app.nodes.llm.causal_lm_model_node", "CausalLMModule"),
    ("app.nodes.llm.causal_lm_model_node", "_DecoderBlock"),
    ("app.nodes.llm.causal_lm_model_node", "_CausalSelfAttention"),
    ("app.nodes.llm.causal_lm_model_node", "_FeedForward"),

    # The VLA policy and its two block types. Audited: ints, floats, strings,
    # torch submodules and parameters. No pickle hooks.
    ("app.nodes.vla.vla_model_node", "VLAModule"),
    ("app.nodes.vla.vla_model_node", "_EncoderBlock"),
    ("app.nodes.vla.vla_model_node", "_ExpertBlock"),

    # The diffusion U-Net and the two modules it composes. Audited: ints and
    # torch submodules. Their ``__init__`` seeds weights, and the unpickler CAN
    # reach it (rule 2 above), so that was checked rather than assumed: each
    # seeds a LOCAL ``torch.Generator`` and never ``torch.manual_seed``, so a
    # file-chosen seed changes only the tensor it then overwrites. No pickle
    # hooks.
    ("app.nodes.diffusion.diffusion_unet_node", "_DiffusionUNetModule"),
    ("app.nodes.diffusion._resblock_module", "_ResBlockModule"),
    ("app.nodes.diffusion.timestep_embedding_node", "_TimestepMLP"),

    # Transformer MoE, the seeded RNN cell, the RLHF reward head. Audited: ints
    # and torch submodules only; ``_SeededRNNCell``'s constructor is reachable
    # with a file-chosen seed and uses a local ``torch.Generator`` (the
    # ``torch.manual_seed`` calls in these two modules are in the NODES, which
    # the unpickler cannot reach). No pickle hooks.
    ("app.nodes.transformer.moe_layer_node", "_MoELayer"),
    ("app.nodes.transformer.moe_layer_node", "_ExpertFFN"),
    ("app.nodes.rnn.rnn_cell_node", "_SeededRNNCell"),
    ("app.nodes.rl.reward_model_node", "_RewardHead"),

    # The two loss modules. They travel on LOSS_FN ports rather than MODEL
    # ones, so they reach a save only through an ANY-typed hop -- listed
    # because that path exists, not because it is the usual one. Audited:
    # ints, floats and strings. No pickle hooks.
    ("app.nodes.llm.lm_cross_entropy_loss_node", "LMCrossEntropyLoss"),
    ("app.nodes.vla.vla_model_node", "VLABehaviorLoss"),
)


def codefyui_module_globals() -> list[type]:
    """The classes in :data:`_CODEFYUI_MODULE_CLASSES`, resolved and cached.

    The curated half of the ``full_model`` allowlist (#288). See the comment
    above the data for the decision, the threat model, the admission rule and
    the per-class audit.
    """
    global _CODEFYUI_GLOBALS
    if _CODEFYUI_GLOBALS is not None:
        return _CODEFYUI_GLOBALS

    from importlib import import_module

    resolved: list[type] = []
    for module_path, qualname in _CODEFYUI_MODULE_CLASSES:
        obj: Any = import_module(module_path)
        for part in qualname.split("."):
            obj = getattr(obj, part, None)
        if not isinstance(obj, type):
            # A rename that did not update the list. Raised rather than
            # skipped: skipping would un-admit a class silently, and the
            # symptom -- a file that used to load and now does not -- would
            # appear a release later with nothing pointing here.
            raise RuntimeError(
                f"The full_model allowlist names {module_path}.{qualname}, "
                f"which is no longer a class there. It was renamed, moved or "
                f"removed; update _CODEFYUI_MODULE_CLASSES in "
                f"app/nodes/io/model_loader_node.py to match (#288)."
            )
        resolved.append(obj)

    _CODEFYUI_GLOBALS = resolved
    return _CODEFYUI_GLOBALS


# ── The function part of the allowlist (#288 follow-up, 2026-08-12) ──────
#
# #334 shipped the two class parts above and left the function question open,
# with the cost recorded as two strict xfails: ``nn.TransformerEncoderLayer``
# and ``nn.TransformerDecoderLayer`` store their activation CALLABLE as a plain
# attribute, so every transformer checkpoint saved fine and was refused on the
# way back in -- the exact trap #288 was opened about, one category over from
# the one #334 closed.
#
# THE MAINTAINER'S RULING, 2026-08-12: torch-owned activation functions are
# admitted BY EXACT IDENTITY, because a pure tensor function invoked with
# arbitrary file-chosen arguments has strictly LESS surface than the
# already-admitted classes' constructors -- which the REDUCE path can also
# invoke, as the rule above ``_CODEFYUI_MODULE_CLASSES`` spells out. Admitting
# ``nn.Linear`` already admits ``nn.Linear(*file_chosen_args)``; admitting
# ``F.relu`` admits ``F.relu(*file_chosen_args)``, which returns a tensor or
# raises. The function case is the smaller of the two, and #334 admitted the
# larger one first. That was inconsistent, not cautious.
#
# THE AUDIT CRITERION FOR A FUNCTION -- four parts, all four required:
#
#   1. torch-owned. ``__module__`` starts with ``torch.``. Self-reported and
#      writable, so this is a boundary against the FILE (which chooses a NAME
#      the unpickler resolves), not against code already in the process.
#   2. A pure tensor operation: tensors in, a tensor out.
#   3. No filesystem, network or process side effects.
#   4. No global-state mutation, and safe under ARBITRARY arguments -- raising,
#      or allocating something large, is a failed load rather than a
#      compromised one.
#
# EXACT IDENTITIES, NEVER A NAMESPACE, and that is the whole of the difference
# between this and the widening #222 refused. ``torch.nn.functional`` as a
# PREFIX would admit ``handle_torch_function``, which dispatches to an
# arbitrary object's ``__torch_function__`` and so is a general-purpose call
# gadget; it would also admit whatever torch adds to that namespace next,
# unaudited. Two names are admitted. Everything else there is still refused,
# and ``test_the_function_allowlist_is_exact_not_a_namespace`` proves it with a
# sibling function from the same module.
#
# ENUMERATED FROM THE SAVE SIDE, like the class list above: a sweep over every
# layer ``_build_layer`` builds, every CodefyUI module family, and the
# layer-editor graph model, collecting function-valued instance attributes. The
# sweep finds exactly one -- ``TransformerEncoderLayer.activation`` /
# ``TransformerDecoderLayer.activation`` holding ``F.relu``, because no CodefyUI
# node exposes an activation choice and torch's default is ``relu``.
# ``F.gelu`` is admitted alongside it because it is the ONLY other value that
# attribute can hold: torch's ``_get_activation_fn`` maps the string to
# ``F.relu`` or ``F.gelu`` and raises on anything else, so enumerating the
# attribute's possible VALUES is complete in a way that enumerating today's
# node params is not -- a hand-built or future gelu-activated layer would
# otherwise reopen the same gap for the same reason.
# ``test_the_admitted_functions_are_what_the_save_side_stores`` re-runs that
# sweep, so a layer that starts storing a third function fails here rather than
# rotting into a refusal a release later.
#
# One correction to the ruling's wording, found by running the sweep rather
# than assuming: ``F.gelu`` is NOT a plain Python function. It is
# ``torch._C._nn.gelu``, a C binding of a single ATen op, and that is the name
# pickle writes down for it. That is LESS surface than ``F.relu`` (which has a
# Python body that can dispatch), not more -- so it is admitted, and the audit
# test asserts "a function or a builtin function, and never a type" rather
# than the stricter "plain function" the ruling described.
#
# Recorded as NAMES for the same two reasons the class list is: keeping
# ``torch`` off the node registry's import path, and making the answer
# independent of import order.
_TORCH_FUNCTION_NAMES: tuple[tuple[str, str], ...] = (
    # What ``nn.TransformerEncoderLayer(...)`` and ``...DecoderLayer(...)``
    # store as ``self.activation`` for the default ``activation="relu"``,
    # whatever spelling they were constructed with -- and so what every
    # CodefyUI transformer block, and every layer-editor graph containing one,
    # has hanging off it. Audited: elementwise ``max(x, 0)``. Pure, no side
    # effects, no global state; a hostile argument raises.
    ("torch.nn.functional", "relu"),

    # The other value ``_get_activation_fn`` can return. Note the module: this
    # is the C binding ``torch._C._nn.gelu``, which is the name pickle writes
    # and therefore the name that has to be admitted. Audited: elementwise
    # arithmetic, no side effects, no global state.
    ("torch._C._nn", "gelu"),
)


def torch_function_globals() -> list[Any]:
    """The functions in :data:`_TORCH_FUNCTION_NAMES`, resolved and cached.

    The third part of the ``full_model`` allowlist. See the comment above the
    data for the 2026-08-12 maintainer decision, its rationale, the four-part
    function audit criterion, and the save-side enumeration it came from.
    """
    global _FUNCTION_GLOBALS
    if _FUNCTION_GLOBALS is not None:
        return _FUNCTION_GLOBALS

    from importlib import import_module

    resolved: list[Any] = []
    for module_path, qualname in _TORCH_FUNCTION_NAMES:
        obj: Any = import_module(module_path)
        for part in qualname.split("."):
            obj = getattr(obj, part, None)
        if not callable(obj) or isinstance(obj, type):
            # Raised rather than skipped, for the same reason the class
            # resolver raises: skipping would silently un-admit a name, and the
            # symptom -- a transformer checkpoint that used to load and now
            # does not -- would surface a release later with nothing pointing
            # here.
            raise RuntimeError(
                f"The full_model allowlist names {module_path}.{qualname}, "
                f"which is no longer a function there. Torch moved, renamed or "
                f"removed it; update _TORCH_FUNCTION_NAMES in "
                f"app/nodes/io/model_loader_node.py to match (#288)."
            )
        resolved.append(obj)

    _FUNCTION_GLOBALS = resolved
    return _FUNCTION_GLOBALS


def full_model_safe_globals() -> list[Any]:
    """Every class and function ``load_mode="full_model"`` will reconstruct.

    The derived torch class part, the curated CodefyUI class part and the two
    torch activation functions, in ONE place, read by both ends of the round
    trip: this node's loader and ``ModelSaver``'s save-time note. The note
    being derived from the same call is what stops it drifting from what the
    loader actually accepts -- which was the whole reason #287 could describe
    the gap correctly while it existed, and what made the note's wording change
    a consequence of this widening rather than a second edit that could
    disagree with it.
    """
    return [
        *torch_nn_layer_globals(),
        *codefyui_module_globals(),
        *torch_function_globals(),
    ]


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
            f"it contains {refused.group(1)}, which is not one of the things "
            f"CodefyUI will reconstruct -- torch.nn's own layer classes, "
            f"CodefyUI's own module classes, and the two torch activation "
            f"functions those layers store (#288). A class from a custom node "
            f"or a plugin is not on that list, and neither is any other "
            f"function"
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
        "layers and of CodefyUI's own layers -- plus, by exact name, the two "
        "activation functions torch's transformer layers store -- and "
        "everything else is refused rather than executed.\n"
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
                    "stock torch.nn layers and of CodefyUI's own layers, and "
                    "refuses anything else -- including classes from custom nodes "
                    "or plugins, and any function other than the two torch "
                    "activations its transformer layers store"
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
        which is what the context manager is for -- to the classes in
        :func:`full_model_safe_globals`, and to nothing else.
        """
        import torch

        # Built BEFORE the try: a stale entry in the curated list raises a
        # RuntimeError that says exactly which name to fix, and swallowing it
        # into the "could not load this file" message would blame the file for
        # a bug in this module.
        allowed = full_model_safe_globals()

        try:
            with torch.serialization.safe_globals(allowed):
                return torch.load(str(p), map_location=load_device, weights_only=True)
        except Exception as exc:
            raise ValueError(_full_model_refusal(p, exc)) from exc
