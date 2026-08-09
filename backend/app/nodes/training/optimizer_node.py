from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
    is_param_visible,
)
from ...core.param_values import parse_float_sequence

#: The select vocabulary, and the source of every ``visible_when`` list
#: below. Kept as a module constant so the per-algorithm sets can be read
#: against it at a glance — an option that appears in none of them is a
#: hyperparameter nobody can reach.
OPTIMIZER_TYPES = [
    "Adam", "SGD", "AdamW", "RMSprop", "Adagrad", "RAdam", "NAdam", "Rprop",
    "ASGD",
]

# ── which hyperparameter belongs to which algorithm ───────────────────────
#
# Applicability is declared here rather than inferred from
# ``inspect.signature``, and the difference matters exactly once: Adagrad
# ACCEPTS ``eps`` but defaults it to 1e-10, not the 1e-8 the Adam family
# uses. Feeding it this node's default would silently change the behaviour
# of every existing Adagrad graph on upgrade. So each set below lists the
# algorithms whose torch default equals this node's declared default, which
# makes every new parameter a no-op until someone edits it.
_BETAS_TYPES = frozenset({"Adam", "AdamW", "RAdam", "NAdam"})
_EPS_TYPES = frozenset({"Adam", "AdamW", "RAdam", "NAdam", "RMSprop"})
_MOMENTUM_TYPES = frozenset({"SGD", "RMSprop"})
_SGD_ONLY = frozenset({"SGD"})
_AMSGRAD_TYPES = frozenset({"Adam", "AdamW"})

#: Node defaults, mirrored from torch so "left alone" means "unchanged".
DEFAULT_BETAS = "0.9, 0.999"
DEFAULT_EPS = 1e-8


_PARAM_DEFS: dict[str, ParamDefinition] = {}


def _reject_inapplicable(
    opt_type: str, param: str, value: Any, params: dict[str, Any],
) -> None:
    """Complain about a param this algorithm has no knob for -- IF it is one
    the user can actually see.

    Silence is the wrong answer for a VISIBLE param: someone who typed
    ``weight_decay=0.1`` on Rprop has a mental model to correct, and a run
    that quietly ignores it teaches them the wrong thing. ``weight_decay``
    is the one that reaches the raise today, because it is the one
    inapplicable param with no ``visible_when`` to hide it -- every other
    rule below is built from the SAME frozenset as its applicability check,
    so an inapplicable value is a hidden value by construction. That is a
    property of the current definitions, not a coincidence, and
    ``test_every_conditional_param_hides_exactly_where_it_does_not_apply``
    fails the day someone adds a param where it stops holding, which is the
    day this branch starts firing for them too.

    Silence is the RIGHT answer for a param the current ``type`` hides. The
    canvas materialises every default onto a node and never clears one when
    the sibling that hides it changes, so tuning ``eps`` under Adam and then
    switching to Adagrad used to fail the run naming a field that is not on
    the form -- unfixable without editing the JSON by hand. A hidden
    leftover means "not set"; it is preserved for when it applies again,
    which is exactly what the editor does with it.
    """
    if not _PARAM_DEFS:
        _PARAM_DEFS.update(
            {p.name: p for p in OptimizerNode.define_params()})
    definition = _PARAM_DEFS.get(param)
    if definition is not None and not is_param_visible(definition, params):
        return
    raise ValueError(
        f"Optimizer '{opt_type}' does not accept {param}; got {value!r}. "
        f"Leave {param} at its default or pick a different optimizer.")


class OptimizerNode(BaseNode):
    NODE_NAME = "Optimizer"
    CATEGORY = "Training"
    DESCRIPTION = "Create an optimizer for model parameters"

    # #254. An optimizer is a live handle: it holds references to the
    # model's parameters and accumulates state (momentum buffers, Adam's
    # running averages, step counts) that ``TrainingLoop`` mutates in place
    # every batch. The cache key describes how it was BUILT -- the
    # algorithm and the hyperparameters -- and stops describing it the
    # moment training starts, so a hit hands the next run an optimizer
    # carrying the previous run's momentum.
    #
    # No graph in the registry can reach that hit today, because every node
    # that hands out a model is now non-cacheable and the engine propagates
    # that downstream. This flag is what keeps it unreachable when a plugin
    # adds a cacheable model source -- and constructing an optimizer costs
    # microseconds, so there is nothing on the other side of the scale.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Model whose parameters to optimize"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="optimizer", data_type=DataType.OPTIMIZER, description="Configured optimizer"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="type",
                param_type=ParamType.SELECT,
                default="Adam",
                description="Optimizer algorithm",
                options=list(OPTIMIZER_TYPES),
            ),
            ParamDefinition(name="lr", param_type=ParamType.FLOAT, default=0.001, description="Learning rate", min_value=0.0),
            ParamDefinition(name="weight_decay", param_type=ParamType.FLOAT, default=0.0, description="Weight decay (L2 penalty)", min_value=0.0),
            # Basic, not advanced: momentum is the first thing anyone
            # reaches for after the learning rate, and an SGD node without
            # it visible is an SGD node people think cannot do momentum.
            ParamDefinition(
                name="momentum",
                param_type=ParamType.FLOAT,
                default=0.0,
                description="Momentum factor (0 = plain gradient descent)",
                min_value=0.0,
                visible_when={"type": sorted(_MOMENTUM_TYPES)},
            ),
            ParamDefinition(
                name="betas",
                param_type=ParamType.STRING,
                default=DEFAULT_BETAS,
                description=(
                    "Adam-family coefficients for the gradient and squared-"
                    "gradient running averages, as 'beta1, beta2'"
                ),
                visible_when={"type": sorted(_BETAS_TYPES)},
                advanced=True,
            ),
            ParamDefinition(
                name="eps",
                param_type=ParamType.FLOAT,
                default=DEFAULT_EPS,
                description=(
                    "Term added to the denominator for numerical stability. "
                    "Adagrad keeps its own 1e-10 default and ignores this."
                ),
                min_value=0.0,
                visible_when={"type": sorted(_EPS_TYPES)},
                advanced=True,
            ),
            ParamDefinition(
                name="amsgrad",
                param_type=ParamType.BOOL,
                default=False,
                description="Use the AMSGrad variant of Adam",
                visible_when={"type": sorted(_AMSGRAD_TYPES)},
                advanced=True,
            ),
            ParamDefinition(
                name="nesterov",
                param_type=ParamType.BOOL,
                default=False,
                description=(
                    "Nesterov accelerated gradient. Requires momentum > 0 "
                    "and dampening = 0."
                ),
                visible_when={"type": sorted(_SGD_ONLY)},
                advanced=True,
            ),
            ParamDefinition(
                name="dampening",
                param_type=ParamType.FLOAT,
                default=0.0,
                description="Damping applied to the momentum term",
                min_value=0.0,
                visible_when={"type": sorted(_SGD_ONLY)},
                advanced=True,
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import inspect
        import torch.optim as optim

        model = inputs["model"]
        opt_type = params.get("type", "Adam")
        lr = params.get("lr", 0.001)
        weight_decay = params.get("weight_decay", 0.0)

        optimizer_map = {
            "Adam": optim.Adam,
            "SGD": optim.SGD,
            "AdamW": optim.AdamW,
            "RMSprop": optim.RMSprop,
            "Adagrad": optim.Adagrad,
            "RAdam": optim.RAdam,
            "NAdam": optim.NAdam,
            "Rprop": optim.Rprop,
            "ASGD": optim.ASGD,
        }

        optimizer_cls = optimizer_map.get(opt_type)
        if optimizer_cls is None:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")

        # Not every optimizer accepts ``weight_decay`` (e.g. Rprop). Silently
        # drop the kwarg when the user left it at the default 0.0 — that's
        # equivalent to "not supplied" — but raise a clear error if they
        # intentionally set a non-zero value the optimizer can't honour.
        # Through the shared helper, like every other param: it is always
        # visible (no ``visible_when``), so it always raises — and routing
        # it here is what keeps that branch a live guard rather than a
        # decorative one.
        accepted = set(inspect.signature(optimizer_cls.__init__).parameters)
        kwargs: dict[str, Any] = {"lr": lr}
        if "weight_decay" in accepted:
            kwargs["weight_decay"] = weight_decay
        elif weight_decay:
            _reject_inapplicable(opt_type, "weight_decay", weight_decay, params)

        # Each block: apply where the algorithm has the knob, complain where
        # it does not and the user changed it, stay silent where it does not
        # and they did not.
        momentum = float(params.get("momentum", 0.0) or 0.0)
        if opt_type in _MOMENTUM_TYPES:
            kwargs["momentum"] = momentum
        elif momentum:
            _reject_inapplicable(opt_type, "momentum", momentum, params)

        betas = parse_float_sequence(
            params.get("betas", DEFAULT_BETAS), name="betas", length=2)
        if opt_type in _BETAS_TYPES:
            if betas is not None:
                kwargs["betas"] = betas
        elif betas is not None and betas != (0.9, 0.999):
            _reject_inapplicable(opt_type, "betas", betas, params)

        eps = params.get("eps", DEFAULT_EPS)
        eps = DEFAULT_EPS if eps is None else float(eps)
        if opt_type in _EPS_TYPES:
            kwargs["eps"] = eps
        elif eps != DEFAULT_EPS:
            _reject_inapplicable(opt_type, "eps", eps, params)

        amsgrad = bool(params.get("amsgrad", False))
        if opt_type in _AMSGRAD_TYPES:
            kwargs["amsgrad"] = amsgrad
        elif amsgrad:
            _reject_inapplicable(opt_type, "amsgrad", amsgrad, params)

        nesterov = bool(params.get("nesterov", False))
        dampening = float(params.get("dampening", 0.0) or 0.0)
        if opt_type in _SGD_ONLY:
            kwargs["nesterov"] = nesterov
            kwargs["dampening"] = dampening
        else:
            if nesterov:
                _reject_inapplicable(opt_type, "nesterov", nesterov, params)
            if dampening:
                _reject_inapplicable(opt_type, "dampening", dampening, params)

        optimizer = optimizer_cls(model.parameters(), **kwargs)

        return {"optimizer": optimizer}
