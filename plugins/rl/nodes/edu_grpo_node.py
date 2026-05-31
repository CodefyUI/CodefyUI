"""EduGRPONode — one GRPO update with its group-relative advantage visible.

Supports textbook lesson **C5-4 / I5 (GRPO)**: Group Relative Policy
Optimization, the core training change introduced by DeepSeek R1. The *only*
difference from REINFORCE / Edu-PolicyGradient is how the advantage is built:

    REINFORCE / PG :  advantage = reward − baseline   (a learned / mean baseline)
    GRPO           :  advantage = (reward − group_mean) / (group_std + eps)

i.e. GRPO throws away the separate value-function baseline and instead samples
a *group* of G completions for the same prompt, then standardizes (z-scores)
their rewards **within the group**. The group mean plays the role of the
baseline and the group std rescales the advantage. Everything else — softmax,
gather the log-prob of the taken action, loss = −mean(log_probs · advantages) —
is identical to a plain policy-gradient step.

The five pieces exposed for students:

    1. probs       = softmax(logits / T)
    2. log_probs   = log(probs gathered at the actions taken)
    3. group_mean  = rewards.mean()
    4. group_std   = rewards.std(unbiased=False)
       advantages  = (rewards − group_mean) / (group_std + std_eps)
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


class EduGRPONode(BaseNode):
    NODE_NAME = "Edu-GRPO"
    CATEGORY = "RL"
    DESCRIPTION = (
        "One GRPO (Group Relative Policy Optimization) step, the core change in "
        "DeepSeek R1. Sample a GROUP of G completions and z-score their rewards "
        "within the group — advantages = (rewards − group_mean) / (group_std + "
        "eps) — instead of subtracting a learned value baseline. The ONLY "
        "difference from REINFORCE/Edu-PolicyGradient is this group-normalized "
        "advantage; the rest (softmax → gather log-prob → loss = −mean(log_probs "
        "· advantages)) is an ordinary policy-gradient step. Each intermediate is "
        "captured in verbose mode so students see logits + rewards become a loss."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="logits",
                data_type=DataType.TENSOR,
                description="Policy logits for the G grouped samples, shape [G, num_actions].",
            ),
            PortDefinition(
                name="actions",
                data_type=DataType.TENSOR,
                description="Integer index of the action/token taken in each sample, shape [G].",
            ),
            PortDefinition(
                name="rewards",
                data_type=DataType.TENSOR,
                description="Scalar reward for each sample in the group, shape [G].",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        # Order matters: the test asserts this exact list.
        return [
            PortDefinition(
                name="loss",
                data_type=DataType.TENSOR,
                description="Scalar GRPO loss = −mean(log_probs · advantages). Backward through it to get ∇θ.",
            ),
            PortDefinition(
                name="advantages",
                data_type=DataType.TENSOR,
                description="Group-normalized advantages (rewards − group_mean) / (group_std + eps), shape [G]. Display-only.",
            ),
            PortDefinition(
                name="log_probs",
                data_type=DataType.TENSOR,
                description="log π(a|s) for the action taken in each sample, shape [G]. Display-only.",
            ),
            PortDefinition(
                name="group_mean",
                data_type=DataType.TENSOR,
                description="Mean reward over the group — the GRPO baseline (scalar). Display-only.",
            ),
            PortDefinition(
                name="group_std",
                data_type=DataType.TENSOR,
                description="Population std of the group's rewards (scalar). Display-only.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
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
            ParamDefinition(
                name="std_eps",
                param_type=ParamType.FLOAT,
                default=1e-6,
                min_value=0.0,
                description=(
                    "Epsilon added to the group standard deviation before dividing, for "
                    "numerical stability. Keeps advantages finite when every sample in the "
                    "group earned the same reward (group_std = 0)."
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
                "EduGRPO requires `logits`, `actions`, and `rewards` inputs."
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
                f"EduGRPO expects logits of shape [G, A]; got {tuple(logits.shape)}."
            )
        group_size, num_actions = logits.shape
        if group_size < 2:
            raise ValueError(
                "EduGRPO needs a group of at least two samples to normalize "
                f"rewards within the group; got G={group_size}."
            )
        if actions.shape != (group_size,):
            raise ValueError(
                f"actions must have shape [{group_size}]; got {tuple(actions.shape)}."
            )
        if rewards.shape != (group_size,):
            raise ValueError(
                f"rewards must have shape [{group_size}]; got {tuple(rewards.shape)}."
            )
        if (actions < 0).any() or (actions >= num_actions).any():
            raise ValueError(
                f"actions out of range [0, {num_actions}). "
                f"Got min={int(actions.min())}, max={int(actions.max())}."
            )

        temperature = float(params.get("temperature", 1.0))
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        std_eps = float(params.get("std_eps", 1e-6))

        verbose = context is not None and getattr(context, "verbose", False)
        recorder = StepRecorder() if verbose else None

        # 1. Policy distribution over actions for every sample in the group.
        scaled = logits / temperature
        probs = F.softmax(scaled, dim=-1)
        if recorder is not None:
            recorder.record(
                "softmax",
                "probs = softmax(logits / T) for each of the G grouped samples.",
                scalars={"temperature": temperature, "group_size": float(group_size)},
                logits=logits, scaled=scaled, probs=probs,
            )

        # 2. Log-prob of the action actually taken in each sample.
        action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        log_probs = action_probs.clamp_min(1e-12).log()
        if recorder is not None:
            recorder.record(
                "log_probs",
                "log π(a|s): gather the probability of the action taken, then log.",
                actions=actions, action_probs=action_probs, log_probs=log_probs,
            )

        # 3. Group statistics — this is the GRPO substitute for a value baseline.
        #    group_mean acts as the baseline; group_std rescales the advantage.
        group_mean = rewards.mean()
        group_std = rewards.std(unbiased=False)
        if recorder is not None:
            recorder.record(
                "group_stats",
                "Group baseline statistics: mean (the baseline) and population std "
                "of the rewards within this group.",
                scalars={
                    "group_mean": float(group_mean.item()),
                    "group_std": float(group_std.item()),
                },
                rewards=rewards, group_mean=group_mean, group_std=group_std,
            )

        # 4. Group-normalized advantage. THIS is the only change vs REINFORCE /
        #    Edu-PolicyGradient: instead of (reward − baseline), GRPO z-scores the
        #    rewards within the group. std_eps keeps it finite when group_std == 0
        #    (every sample earned the same reward).
        advantages = (rewards - group_mean) / (group_std + std_eps)
        if recorder is not None:
            recorder.record(
                "advantages",
                "advantages = (rewards − group_mean) / (group_std + std_eps). "
                "The group-relative z-score replaces a learned value baseline — "
                "the single change that defines GRPO.",
                scalars={"std_eps": std_eps},
                rewards=rewards, advantages=advantages,
            )

        # 5. Ordinary policy-gradient loss on the group-normalized advantages.
        loss = -(log_probs * advantages).mean()
        if recorder is not None:
            recorder.record(
                "loss",
                "loss = − mean(log_probs · advantages). Gradient descent on this "
                "increases the probability of above-group-average completions.",
                scalars={"loss": float(loss.item())},
                loss=loss,
            )

        result: dict[str, Any] = {
            "loss": loss,
            "advantages": advantages,
            "log_probs": log_probs,
            "group_mean": group_mean,
            "group_std": group_std,
        }
        if recorder is not None and recorder.steps:
            result["__steps__"] = recorder.steps
        return result
