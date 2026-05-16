"""EduPolicyGradientNode — one REINFORCE update with intermediate quantities visible.

Supports textbook lesson **C5-2 (PPO)** and **C5-1 (RL 框架)**: instead of a
black-box trainer, expose the four pieces of one policy-gradient step:

    1. probs       = softmax(logits / T)
    2. action_probs= probs gathered at the actions taken
    3. log_probs   = log(action_probs)
    4. advantages  = rewards − baseline           (mean baseline by default)
    5. loss        = − mean(log_probs · advantages)

The node does NOT call ``loss.backward()`` — pair it with a ``BackwardOnce``
node (production) when you actually want the gradient.
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


class EduPolicyGradientNode(BaseNode):
    NODE_NAME = "EduPolicyGradient"
    CATEGORY = "RL"
    DESCRIPTION = (
        "One REINFORCE policy-gradient step exposed as: softmax(logits/T) → "
        "gather at actions → log → subtract baseline → loss = −mean(log_probs · "
        "advantages). Each intermediate is captured in verbose mode so students "
        "see the chain that turns logits and rewards into a scalar loss."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="logits",
                data_type=DataType.TENSOR,
                description="Policy logits, shape [batch, num_actions].",
            ),
            PortDefinition(
                name="actions",
                data_type=DataType.TENSOR,
                description="Integer indices of the actions actually taken, shape [batch].",
            ),
            PortDefinition(
                name="rewards",
                data_type=DataType.TENSOR,
                description="Per-trajectory return for each action, shape [batch].",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="log_probs",
                data_type=DataType.TENSOR,
                description="log π(a|s) for the action taken in each item, shape [batch].",
            ),
            PortDefinition(
                name="advantages",
                data_type=DataType.TENSOR,
                description="rewards − baseline, shape [batch].",
            ),
            PortDefinition(
                name="loss",
                data_type=DataType.TENSOR,
                description="Scalar policy-gradient loss. Backward through it to get ∇θ.",
            ),
            PortDefinition(
                name="probs",
                data_type=DataType.TENSOR,
                description="Full softmax distribution, shape [batch, num_actions].",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="baseline",
                param_type=ParamType.SELECT,
                default="mean",
                options=["none", "mean"],
                description=(
                    "How to compute the variance-reduction baseline. 'mean' subtracts "
                    "the batch-average reward; 'none' uses raw rewards as advantages."
                ),
            ),
            ParamDefinition(
                name="temperature",
                param_type=ParamType.FLOAT,
                default=1.0,
                min_value=0.01,
                description=(
                    "Softmax temperature. Higher = more exploratory distribution; "
                    "the lesson shows what happens at T → 0 (greedy) and T → ∞ (uniform)."
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
        logits = inputs.get("logits")
        actions = inputs.get("actions")
        rewards = inputs.get("rewards")
        if logits is None or actions is None or rewards is None:
            raise ValueError(
                "EduPolicyGradient requires `logits`, `actions`, and `rewards` inputs."
            )

        if not isinstance(logits, torch.Tensor):
            logits = torch.as_tensor(logits, dtype=torch.float32)
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.long)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
        logits = logits.float()
        actions = actions.long()
        rewards = rewards.float()

        if logits.ndim != 2:
            raise ValueError(
                f"EduPolicyGradient expects logits of shape [B, A]; got {tuple(logits.shape)}."
            )
        batch, num_actions = logits.shape
        if actions.shape != (batch,):
            raise ValueError(
                f"actions must have shape [{batch}]; got {tuple(actions.shape)}."
            )
        if rewards.shape != (batch,):
            raise ValueError(
                f"rewards must have shape [{batch}]; got {tuple(rewards.shape)}."
            )
        if (actions < 0).any() or (actions >= num_actions).any():
            raise ValueError(
                f"actions out of range [0, {num_actions}). Got min={int(actions.min())}, max={int(actions.max())}."
            )

        temperature = float(params.get("temperature", 1.0))
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        baseline_mode = str(params.get("baseline", "mean"))

        verbose = context is not None and getattr(context, "verbose", False)
        recorder = StepRecorder() if verbose else None

        scaled = logits / temperature
        probs = F.softmax(scaled, dim=-1)
        if recorder is not None:
            recorder.record(
                "softmax",
                "probs = softmax(logits / T)",
                scalars={"temperature": temperature, "num_actions": float(num_actions)},
                logits=logits, scaled=scaled, probs=probs,
            )

        action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        if recorder is not None:
            recorder.record(
                "gather",
                "Pick the probability of the action actually taken in each item.",
                actions=actions, action_probs=action_probs,
            )

        log_probs = action_probs.clamp_min(1e-12).log()
        if recorder is not None:
            recorder.record("log", "log of the action probability.", log_probs=log_probs)

        if baseline_mode == "mean":
            baseline = rewards.mean()
            advantages = rewards - baseline
        else:
            baseline = torch.zeros((), dtype=rewards.dtype)
            advantages = rewards
        if recorder is not None:
            recorder.record(
                "advantages",
                f"advantages = rewards − baseline ({baseline_mode}).",
                scalars={"baseline": float(baseline.item())},
                rewards=rewards, advantages=advantages,
            )

        loss = -(log_probs * advantages).mean()
        if recorder is not None:
            recorder.record(
                "loss",
                "loss = − mean(log_probs · advantages). Gradient descent on this increases probabilities of high-advantage actions.",
                scalars={"loss": float(loss.item())},
                loss=loss,
            )

        result: dict[str, Any] = {
            "log_probs": log_probs,
            "advantages": advantages,
            "loss": loss,
            "probs": probs,
        }
        if recorder is not None and recorder.steps:
            result["__steps__"] = recorder.steps
        return result
