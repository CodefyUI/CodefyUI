"""MaskedFillNode — apply a boolean mask to a tensor.

``AttentionMask`` produces the mask the whole LLM category documents
(``True`` = blocked) and every attention node that consumes one does the
same single line internally::

    scores = scores.masked_fill(mask, float("-inf"))

Until this node existed that line was reachable only from *inside* a packaged
attention node. A graph that builds attention out of the primitive tensor ops
— ``MatMul`` → ``ScalarMultiply`` → ``Softmax`` → ``MatMul``, which is the
formula written out one node per step — could generate a mask and had no way
to apply it. So the causal-mask half of attention was the one step of the
mechanism that could not be shown on the canvas.

**Fill before softmax, not after.** ``-inf`` is what softmax needs to give a
blocked position exactly zero probability *and* renormalise what is left, so
each row still sums to 1. Zeroing entries after the softmax breaks that: the
row no longer sums to 1 and the "weighted average" downstream stops being an
average. The default fill value is therefore ``-inf``, and the node is meant
to sit between the scaling step and the softmax.

The mask broadcasts against the tensor by torch's normal rules, so a
``[seq, seq]`` mask applies to ``[batch, seq, seq]`` or ``[batch, heads, seq,
seq]`` scores unchanged. A non-boolean mask is accepted and read as
"non-zero means blocked", which is what a user who built one out of
comparisons rather than ``AttentionMask`` will have.
"""

from __future__ import annotations

import math
from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class MaskedFillNode(BaseNode):
    NODE_NAME = "MaskedFill"
    CATEGORY = "Tensor Operations"
    DESCRIPTION = (
        "Replace every masked position with a constant (default $-\\infty$). "
        "Pair it with `AttentionMask` and place it BEFORE `Softmax`: $-\\infty$ "
        "is what makes a blocked position get exactly zero probability while "
        "the surviving positions renormalise to sum to 1."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Tensor to fill — typically attention scores before softmax.",
            ),
            PortDefinition(
                name="mask",
                data_type=DataType.TENSOR,
                description=(
                    "Boolean mask, True = blocked (the convention `AttentionMask` emits). "
                    "Broadcasts against `tensor`; a non-boolean mask is read as non-zero = blocked."
                ),
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Same shape as the input, with masked positions replaced by `value`.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="value",
                param_type=ParamType.SELECT,
                default="-inf",
                options=["-inf", "zero", "custom"],
                description=(
                    "What to write into masked positions. '-inf' is the attention default "
                    "(softmax turns it into exactly 0). 'zero' and 'custom' are for "
                    "non-attention masking; they do NOT give correct attention weights."
                ),
            ),
            ParamDefinition(
                name="custom_value",
                param_type=ParamType.FLOAT,
                default=0.0,
                description="Fill value used when `value` is set to 'custom'.",
            ),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        import torch

        tensor = inputs["tensor"]
        mask = inputs["mask"]

        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask)

        # `masked_fill` requires a bool mask; a user-built mask of 0/1 floats
        # is the same intent, so convert rather than refuse.
        if mask.dtype != torch.bool:
            mask = mask != 0

        # -inf into an integer tensor is not representable and torch's own
        # error names a dtype the user never chose. Say what to do instead.
        if not tensor.is_floating_point():
            raise ValueError(
                "MaskedFill: `tensor` must be a floating-point tensor "
                f"(got {tensor.dtype}). Attention scores are float — if this is an "
                "integer tensor you are probably masking the wrong wire."
            )

        mode = str(params.get("value", "-inf"))
        if mode == "-inf":
            fill = -math.inf
        elif mode == "zero":
            fill = 0.0
        elif mode == "custom":
            fill = float(params.get("custom_value", 0.0))
        else:
            raise ValueError(f"MaskedFill: unknown value mode {mode!r}.")

        mask = mask.to(tensor.device)

        try:
            out = tensor.masked_fill(mask, fill)
        except RuntimeError as exc:
            raise ValueError(
                f"MaskedFill: mask shape {tuple(mask.shape)} does not broadcast against "
                f"tensor shape {tuple(tensor.shape)}. A causal mask is [seq, seq] and the "
                "scores it masks must end in [.., seq, seq]."
            ) from exc

        result: dict[str, Any] = {"tensor": out}

        if context is not None and getattr(context, "verbose", False):
            from ...core.step_trace import StepRecorder

            recorder = StepRecorder()
            recorder.record(
                "mask",
                "Boolean mask — True marks a position that is blocked.",
                mask=mask.to(tensor.dtype),
            )
            recorder.record(
                "filled",
                f"Every masked position replaced by {mode}.",
                scalars={"blocked_positions": float(mask.sum().item())},
                output=out,
            )
            result["__steps__"] = recorder.steps

        return result
