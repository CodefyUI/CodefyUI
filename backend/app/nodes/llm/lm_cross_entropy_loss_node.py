"""LMCrossEntropyLoss node (#289) -- cross-entropy for next-token prediction.

The generic ``Loss`` node's ``CrossEntropyLoss`` expects ``[B, C]`` logits
against ``[B]`` class indices. A language model produces ``[B, T, V]`` -- a
whole sequence of independent classification problems -- so every LM
implementation flattens the batch and time axes together before calling
cross-entropy. This node is that flatten, packaged so nobody has to remember
to do it.

**Why it is a plain nn.Module and not a subclass of nn.CrossEntropyLoss.**
``TrainingLoop`` decides whether to compute ``val_accuracy`` with
``isinstance(loss_fn, (nn.CrossEntropyLoss, nn.NLLLoss))``
(``training_loop_node.py``), and that branch then runs
``outputs.argmax(dim=1)`` against the targets. On ``[B, T, V]`` logits
``dim=1`` is the TIME axis, so a subclass would have produced a silent,
meaningless accuracy number on every LM run -- and early stopping can be
asked to monitor it. Composing ``F.cross_entropy`` inside an ordinary module
keeps that gate closed, which is the honest answer: token-level accuracy is
not what this loss measures, and perplexity (Task 3's Perplexity node) is.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)

#: torch's own ignore_index default, repeated so "unchanged" is checkable --
#: and the value ``loss_node.py`` already uses, so a padding label written for
#: one works in the other.
DEFAULT_IGNORE_INDEX = -100


class LMCrossEntropyLoss(nn.Module):
    """Mean cross-entropy over every position of every sequence.

    ``forward(logits (B, T, V), targets (B, T)) -> scalar``. The reshape to
    ``(B*T, V)`` / ``(B*T,)`` is the whole trick: cross-entropy does not care
    that the rows came from the same sentence, so a sequence of predictions is
    just a bigger batch of one-of-V decisions.

    Module scope, not a closure inside the node, so ``torch.save`` and the
    Python export can name the class (#283).
    """

    def __init__(
        self,
        ignore_index: int = DEFAULT_IGNORE_INDEX,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)

    def extra_repr(self) -> str:
        return (f"ignore_index={self.ignore_index}, "
                f"label_smoothing={self.label_smoothing}")

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor,
    ) -> torch.Tensor:
        if logits.dim() != 3:
            raise ValueError(
                f"LMCrossEntropyLoss expects logits shaped (batch, seq_len, "
                f"vocab_size), got {tuple(logits.shape)}. This loss is for "
                f"language models; use the Loss node for a plain classifier.")
        if targets.dim() != 2:
            raise ValueError(
                f"LMCrossEntropyLoss expects targets shaped (batch, "
                f"seq_len), got {tuple(targets.shape)} -- one token id per "
                f"position, not a one-hot vector.")
        if targets.shape != logits.shape[:2]:
            raise ValueError(
                f"LMCrossEntropyLoss: logits are "
                f"{tuple(logits.shape[:2])} (batch, seq_len) but targets are "
                f"{tuple(targets.shape)}. The dataset must yield labels the "
                f"same length as the input ids.")
        if targets.is_floating_point():
            raise ValueError(
                f"LMCrossEntropyLoss expects integer target token ids, got "
                f"dtype {targets.dtype}. Cast the labels to int64.")

        vocab_size = logits.shape[-1]
        # ``reshape`` rather than ``view``: the logits arriving from an
        # attention stack are frequently non-contiguous, and ``view`` would
        # fail on exactly the tensors this loss exists to consume.
        return F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1).long(),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )


class LMCrossEntropyLossNode(BaseNode):
    NODE_NAME = "LMCrossEntropyLoss"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Cross-entropy shaped for language models: it flattens (batch, "
        "seq_len, vocab_size) logits against (batch, seq_len) token ids and "
        "returns the mean loss over every position. Wire it into "
        "TrainingLoop's loss_fn alongside a CausalLMModel."
    )

    # Cacheable, and correctly so: the output is a small immutable function
    # object built from two numbers. Nothing downstream mutates it (the two
    # attributes are read, never written), so replaying the recorded handle
    # describes it exactly -- the same reasoning that leaves ``Loss``
    # cacheable. See ``BaseNode.cacheable`` for the four shapes that are not.

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="loss_fn",
                data_type=DataType.LOSS_FN,
                description=(
                    "Loss module: forward(logits (batch, seq_len, "
                    "vocab_size), targets (batch, seq_len)) -> scalar mean "
                    "cross-entropy."
                ),
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="ignore_index",
                param_type=ParamType.INT,
                default=DEFAULT_IGNORE_INDEX,
                description=(
                    "Target id that contributes no loss and no gradient. "
                    "Use it for padding, or for the prompt half of an "
                    "instruction example. -100 is the convention every "
                    "toolkit shares."
                ),
            ),
            ParamDefinition(
                name="label_smoothing",
                param_type=ParamType.FLOAT,
                default=0.0,
                min_value=0.0,
                max_value=0.3,
                description=(
                    "Spread a little probability mass over the other tokens "
                    "so an over-confident correct answer still costs "
                    "something (0 = disabled, 0.1 is a common value)."
                ),
            ),
        ]

    def execute(
        self, inputs: dict[str, Any], params: dict[str, Any],
    ) -> dict[str, Any]:
        ignore_index = params.get("ignore_index", DEFAULT_IGNORE_INDEX)
        try:
            ignore_index = int(
                DEFAULT_IGNORE_INDEX if ignore_index is None else ignore_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"LMCrossEntropyLoss: ignore_index must be a whole number, "
                f"got {params.get('ignore_index')!r}."
            ) from exc

        raw_smoothing = params.get("label_smoothing", 0.0)
        if raw_smoothing is None or raw_smoothing == "":
            raw_smoothing = 0.0
        try:
            smoothing = float(raw_smoothing)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"LMCrossEntropyLoss: label_smoothing must be a number, got "
                f"{params.get('label_smoothing')!r}."
            ) from exc
        # Clamped to F.cross_entropy's own domain rather than to the 0.3 the
        # editor offers: above 1.0 torch raises, and the reading of a
        # hand-written 0.5 is clear enough to honour.
        smoothing = min(1.0, max(0.0, smoothing))

        return {
            "loss_fn": LMCrossEntropyLoss(
                ignore_index=ignore_index, label_smoothing=smoothing),
        }
