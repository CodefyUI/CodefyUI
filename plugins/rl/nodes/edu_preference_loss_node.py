"""EduPreferenceLossNode — the RLHF reward-model objective, step by step.

Supports textbook lesson **C5-3 / I5 (RLHF 獎勵模型)**: instead of a black-box
trainer, expose the Bradley-Terry pairwise preference loss used to train a
reward model. Given the reward model's scalar scores for a CHOSEN response and
a REJECTED response, the loss pushes the chosen score above the rejected one:

    1. margin         = reward_chosen − reward_rejected
    2. scaled         = beta · margin
    3. prob_preferred = sigmoid(scaled)            (modeled P(chosen > rejected))
    4. loss           = − mean(logsigmoid(scaled))

``F.logsigmoid`` is used for the loss (rather than ``log(sigmoid(...))``) so the
objective stays numerically stable for large-magnitude margins. The node does
NOT call ``loss.backward()`` — pair it with a ``BackwardOnce`` node (production)
when you actually want the gradient.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from app.core.step_trace import StepRecorder


class EduPreferenceLossNode(BaseNode):
    NODE_NAME = "Edu-PreferenceLoss"
    CATEGORY = "RL"
    DESCRIPTION = (
        "Bradley-Terry pairwise preference loss for an RLHF reward model, exposed "
        "as: margin = r_chosen − r_rejected → scale by β → sigmoid (modeled "
        "P(chosen > rejected)) → loss = −mean(logsigmoid(β·margin)). Each "
        "intermediate is captured in verbose mode so students see how preference "
        "pairs turn into a scalar reward-model loss."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="reward_chosen",
                data_type=DataType.TENSOR,
                description="Reward-model score of the preferred response, shape [batch].",
            ),
            PortDefinition(
                name="reward_rejected",
                data_type=DataType.TENSOR,
                description="Reward-model score of the dispreferred response, shape [batch].",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="loss",
                data_type=DataType.TENSOR,
                description=(
                    "Scalar Bradley-Terry preference loss = "
                    "−mean(logsigmoid(β·(r_chosen − r_rejected))). Backward through it to get ∇θ."
                ),
            ),
            PortDefinition(
                name="prob_preferred",
                data_type=DataType.TENSOR,
                description=(
                    "Modeled P(chosen > rejected) = sigmoid(β·margin), shape [batch]. Display-only."
                ),
            ),
            PortDefinition(
                name="margin",
                data_type=DataType.TENSOR,
                description="r_chosen − r_rejected for each pair, shape [batch]. Display-only.",
            ),
            PortDefinition(
                name="accuracy",
                data_type=DataType.TENSOR,
                description=(
                    "Scalar fraction of pairs already ranked correctly "
                    "(r_chosen > r_rejected). Display-only."
                ),
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="beta",
                param_type=ParamType.FLOAT,
                default=1.0,
                min_value=0.0,
                description=(
                    "Logistic temperature / scale on the reward margin. Larger β "
                    "sharpens the modeled preference probability toward 0/1; β = 0 "
                    "makes every pair look like a coin flip."
                ),
            ),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        reward_chosen = inputs.get("reward_chosen")
        reward_rejected = inputs.get("reward_rejected")
        if reward_chosen is None or reward_rejected is None:
            raise ValueError(
                "EduPreferenceLoss requires `reward_chosen` and `reward_rejected` inputs."
            )

        if not isinstance(reward_chosen, torch.Tensor):
            reward_chosen = torch.as_tensor(reward_chosen, dtype=torch.float32)
        if not isinstance(reward_rejected, torch.Tensor):
            reward_rejected = torch.as_tensor(reward_rejected, dtype=torch.float32)
        reward_chosen = reward_chosen.float()
        reward_rejected = reward_rejected.float()

        if reward_chosen.ndim != 1:
            raise ValueError(
                f"reward_chosen must be 1-D [B]; got shape {tuple(reward_chosen.shape)}."
            )
        if reward_rejected.ndim != 1:
            raise ValueError(
                f"reward_rejected must be 1-D [B]; got shape {tuple(reward_rejected.shape)}."
            )
        if reward_chosen.shape != reward_rejected.shape:
            raise ValueError(
                "reward_chosen and reward_rejected must have the same length [B]; got "
                f"{tuple(reward_chosen.shape)} vs {tuple(reward_rejected.shape)}."
            )

        beta = float(params.get("beta", 1.0))
        if beta < 0:
            raise ValueError("beta must be >= 0.")

        verbose = context is not None and getattr(context, "verbose", False)
        recorder = StepRecorder() if verbose else None

        margin = reward_chosen - reward_rejected
        if recorder is not None:
            recorder.record(
                "margin",
                "margin = reward_chosen − reward_rejected. Positive means the model already scores the preferred response higher.",
                scalars={"beta": beta},
                reward_chosen=reward_chosen, reward_rejected=reward_rejected, margin=margin,
            )

        scaled = beta * margin
        if recorder is not None:
            recorder.record(
                "scaled",
                "scaled = β · margin. β controls how sharply the margin maps to a preference probability.",
                margin=margin, scaled=scaled,
            )

        prob_preferred = torch.sigmoid(scaled)
        if recorder is not None:
            recorder.record(
                "sigmoid",
                "prob_preferred = sigmoid(β·margin) — the modeled P(chosen > rejected).",
                scaled=scaled, prob_preferred=prob_preferred,
            )

        loss = -F.logsigmoid(scaled).mean()
        accuracy = (reward_chosen > reward_rejected).float().mean()
        if recorder is not None:
            recorder.record(
                "loss",
                "loss = −mean(logsigmoid(β·margin)). Descending it raises chosen scores above rejected ones; accuracy is the fraction of pairs already ordered correctly.",
                scalars={"loss": float(loss.item()), "accuracy": float(accuracy.item())},
                loss=loss, accuracy=accuracy,
            )

        result: dict[str, Any] = {
            "loss": loss,
            "prob_preferred": prob_preferred,
            "margin": margin,
            "accuracy": accuracy,
        }
        if recorder is not None and recorder.steps:
            result["__steps__"] = recorder.steps
        return result
