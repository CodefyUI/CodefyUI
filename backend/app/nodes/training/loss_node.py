from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from ...core.param_values import parse_float_sequence

#: The select vocabulary. Every entry accepts ``reduction``; the sets below
#: say which accept anything else.
LOSS_TYPES = [
    "CrossEntropyLoss", "MSELoss", "BCEWithLogitsLoss", "L1Loss",
    "SmoothL1Loss", "NLLLoss", "KLDivLoss", "HuberLoss", "BCELoss",
    "MarginRankingLoss", "CosineEmbeddingLoss",
]

#: Losses taking a per-class (or per-element) ``weight`` tensor.
_WEIGHT_TYPES = frozenset({
    "CrossEntropyLoss", "NLLLoss", "BCELoss", "BCEWithLogitsLoss",
})
#: Losses that can skip a target label entirely.
_IGNORE_INDEX_TYPES = frozenset({"CrossEntropyLoss", "NLLLoss"})
#: Label smoothing is classification-only, and torch puts it on exactly one
#: of the classes we expose.
_LABEL_SMOOTHING_TYPES = frozenset({"CrossEntropyLoss"})
#: Positive-class rebalancing for binary logits.
_POS_WEIGHT_TYPES = frozenset({"BCEWithLogitsLoss"})

REDUCTIONS = ["mean", "sum", "none"]

#: torch's own default, repeated here so "unchanged" is checkable.
DEFAULT_IGNORE_INDEX = -100


# Nothing here rejects an inapplicable param, and that is deliberate.
#
# Every optional param below declares ``visible_when`` from the SAME
# frozenset as its applicability check in ``execute``, so a value this loss
# cannot take is a value the editor is not showing -- and a leftover on a
# hidden field means "not set", not "fail the run". (The canvas writes every
# default onto a node and never clears one when the sibling that hides it
# changes, so an error would name a field that is not on the form and cannot
# be reset without hand-editing the graph JSON.)
#
# The #188 review's I5 fix added a guard that raised for a VISIBLE
# inapplicable param; the re-review then measured it at 0 raising
# combinations out of 55 -- dead code reading as a live guard. It is removed
# rather than left decorative, and the invariant it rested on is asserted
# directly, by
# ``test_every_conditional_param_hides_exactly_where_it_does_not_apply``
# in ``tests/test_loss_node.py``: add a param that is visible where it does
# not apply and that test fails, telling you to handle it explicitly -- the
# way ``Optimizer.weight_decay``, the one param of that shape in either
# node, does by raising.


class LossNode(BaseNode):
    NODE_NAME = "Loss"
    CATEGORY = "Training"
    DESCRIPTION = "Create a loss function"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="loss_fn", data_type=DataType.LOSS_FN, description="Loss function instance"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="type",
                param_type=ParamType.SELECT,
                default="CrossEntropyLoss",
                description="Loss function type",
                options=list(LOSS_TYPES),
            ),
            # Basic, not advanced: label smoothing is a lesson in its own
            # right (why an over-confident correct answer should still cost
            # something), not a knob to bury.
            ParamDefinition(
                name="label_smoothing",
                param_type=ParamType.FLOAT,
                default=0.0,
                description=(
                    "Soften the one-hot target: 0 = hard targets, 0.1 is a "
                    "common regulariser"
                ),
                min_value=0.0,
                max_value=1.0,
                visible_when={"type": sorted(_LABEL_SMOOTHING_TYPES)},
            ),
            ParamDefinition(
                name="reduction",
                param_type=ParamType.SELECT,
                default="mean",
                description=(
                    "How per-sample losses are combined: mean, sum, or none "
                    "(keep them per-sample)"
                ),
                options=list(REDUCTIONS),
                advanced=True,
            ),
            ParamDefinition(
                name="weight",
                param_type=ParamType.STRING,
                default="",
                description=(
                    "Per-class weights as a comma-separated list, e.g. "
                    "'1, 5' for an imbalanced two-class problem. Empty = "
                    "every class weighted equally."
                ),
                visible_when={"type": sorted(_WEIGHT_TYPES)},
                advanced=True,
            ),
            ParamDefinition(
                name="ignore_index",
                param_type=ParamType.INT,
                default=DEFAULT_IGNORE_INDEX,
                description=(
                    "Target value that contributes no loss and no gradient, "
                    "e.g. a padding label"
                ),
                visible_when={"type": sorted(_IGNORE_INDEX_TYPES)},
                advanced=True,
            ),
            ParamDefinition(
                name="pos_weight",
                param_type=ParamType.STRING,
                default="",
                description=(
                    "Weight of the positive class, as one number or one per "
                    "output. Empty = unweighted."
                ),
                visible_when={"type": sorted(_POS_WEIGHT_TYPES)},
                advanced=True,
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch
        import torch.nn as nn

        loss_type = params.get("type", "CrossEntropyLoss")

        loss_map = {
            "CrossEntropyLoss": nn.CrossEntropyLoss,
            "MSELoss": nn.MSELoss,
            "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
            "L1Loss": nn.L1Loss,
            "SmoothL1Loss": nn.SmoothL1Loss,
            "NLLLoss": nn.NLLLoss,
            "KLDivLoss": nn.KLDivLoss,
            "HuberLoss": nn.HuberLoss,
            "BCELoss": nn.BCELoss,
            "MarginRankingLoss": nn.MarginRankingLoss,
            "CosineEmbeddingLoss": nn.CosineEmbeddingLoss,
        }

        loss_cls = loss_map.get(loss_type)
        if loss_cls is None:
            raise ValueError(f"Unsupported loss function type: {loss_type}")

        # ``reduction`` is the one argument all eleven share.
        reduction = params.get("reduction", "mean") or "mean"
        if reduction not in REDUCTIONS:
            raise ValueError(
                f"Unsupported reduction: {reduction!r}; expected one of "
                f"{REDUCTIONS}")
        kwargs: dict[str, Any] = {"reduction": reduction}

        # Each block: apply where this loss takes the argument, ignore where
        # it does not — see the note at the top of the file for why "ignore"
        # is the right answer and not a shrug.
        label_smoothing = float(params.get("label_smoothing", 0.0) or 0.0)
        if loss_type in _LABEL_SMOOTHING_TYPES:
            kwargs["label_smoothing"] = label_smoothing

        # Weight tensors are registered as BUFFERS by the loss module, so
        # ``to_device(loss_fn, device)`` in the training loop moves them
        # along with it — building on CPU here is correct for a CUDA run too.
        weight = parse_float_sequence(params.get("weight"), name="weight")
        if loss_type in _WEIGHT_TYPES and weight is not None:
            kwargs["weight"] = torch.tensor(weight, dtype=torch.float32)

        ignore_index = int(params.get("ignore_index", DEFAULT_IGNORE_INDEX))
        if loss_type in _IGNORE_INDEX_TYPES:
            kwargs["ignore_index"] = ignore_index

        pos_weight = parse_float_sequence(
            params.get("pos_weight"), name="pos_weight")
        if loss_type in _POS_WEIGHT_TYPES and pos_weight is not None:
            kwargs["pos_weight"] = torch.tensor(
                pos_weight, dtype=torch.float32)

        loss_fn = loss_cls(**kwargs)

        return {"loss_fn": loss_fn}
