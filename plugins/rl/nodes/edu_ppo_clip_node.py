"""EduPPOClipNode — one PPO clipped-surrogate update with the clip made visible.

Supports textbook lesson **C5-2 / I5 (PPO)**: instead of a black-box trainer,
expose the three pieces that define the PPO clipped surrogate objective:

    1. log_probs      = log_softmax(logits / T) gathered at the actions, for
                        both the new (current) policy and the old (behaviour)
                        policy.
    2. ratio          = exp(logp_new − logp_old)        the probability ratio
    3. clipped_ratio  = clamp(ratio, 1 − ε, 1 + ε)      the clip
    4. objective      = min(ratio · A, clipped_ratio · A)  per-sample surrogate
    5. loss           = − mean(objective)

The whole point of PPO's clip is to stop a single update from moving the
policy too far: when the ratio leaves the [1−ε, 1+ε] trust region in the
"wrong" direction, ``min`` selects the clipped branch and the gradient through
that sample vanishes. This node lays out ``ratio``, ``clipped_ratio`` and the
per-sample ``objective`` side by side so students can see exactly when the clip
binds.

The old policy is treated as a constant (the behaviour policy that collected
the data); its log-probabilities are still computed so the ratio is explicit.
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


class EduPPOClipNode(BaseNode):
    NODE_NAME = "Edu-PPOClip"
    CATEGORY = "RL"
    DESCRIPTION = (
        "One PPO clipped-surrogate step exposed as: log_softmax(logits/T) "
        "gathered at the actions for the new and old policies → ratio = "
        "exp(logp_new − logp_old) → clip ratio to [1−ε, 1+ε] → "
        "objective = min(ratio·A, clip·A) → loss = −mean(objective). The ratio, "
        "the clipped ratio and the per-sample surrogate are all captured in "
        "verbose mode so students see exactly when the clip binds."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="logits_new",
                data_type=DataType.TENSOR,
                description="Current (new) policy logits, shape [batch, num_actions].",
            ),
            PortDefinition(
                name="logits_old",
                data_type=DataType.TENSOR,
                description="Behaviour (old) policy logits, shape [batch, num_actions].",
            ),
            PortDefinition(
                name="actions",
                data_type=DataType.TENSOR,
                description="Integer indices of the actions actually taken, shape [batch].",
            ),
            PortDefinition(
                name="advantages",
                data_type=DataType.TENSOR,
                description="Per-sample advantage estimates, shape [batch].",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="loss",
                data_type=DataType.TENSOR,
                description=(
                    "Scalar PPO clipped-surrogate loss "
                    "= −mean(min(ratio·A, clip(ratio,1±ε)·A)). Backward to get ∇θ."
                ),
            ),
            PortDefinition(
                name="ratio",
                data_type=DataType.TENSOR,
                description="Probability ratio exp(logp_new − logp_old), shape [batch] (display-only).",
            ),
            PortDefinition(
                name="clipped_ratio",
                data_type=DataType.TENSOR,
                description="ratio clamped to [1−ε, 1+ε], shape [batch] (display-only).",
            ),
            PortDefinition(
                name="objective",
                data_type=DataType.TENSOR,
                description=(
                    "Per-sample min surrogate min(ratio·A, clip·A), shape [batch] "
                    "(display-only)."
                ),
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="clip_epsilon",
                param_type=ParamType.FLOAT,
                default=0.2,
                min_value=0.0,
                description=(
                    "Clip range ε. The ratio is confined to [1−ε, 1+ε]; outside "
                    "the trust region the clipped branch removes the incentive to "
                    "move further. ε=0 freezes the ratio at 1."
                ),
            ),
            ParamDefinition(
                name="temperature",
                param_type=ParamType.FLOAT,
                default=1.0,
                min_value=0.01,
                description=(
                    "Softmax temperature applied to both policies before gathering "
                    "log-probabilities. Higher = softer distributions."
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
        logits_new = inputs.get("logits_new")
        logits_old = inputs.get("logits_old")
        actions = inputs.get("actions")
        advantages = inputs.get("advantages")
        if (
            logits_new is None
            or logits_old is None
            or actions is None
            or advantages is None
        ):
            raise ValueError(
                "EduPPOClip requires `logits_new`, `logits_old`, `actions`, and "
                "`advantages` inputs."
            )

        if not isinstance(logits_new, torch.Tensor):
            logits_new = torch.as_tensor(logits_new, dtype=torch.float32)
        if not isinstance(logits_old, torch.Tensor):
            logits_old = torch.as_tensor(logits_old, dtype=torch.float32)
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.long)
        if not isinstance(advantages, torch.Tensor):
            advantages = torch.as_tensor(advantages, dtype=torch.float32)
        logits_new = logits_new.float()
        logits_old = logits_old.float()
        actions = actions.long()
        advantages = advantages.float()

        if logits_new.ndim != 2:
            raise ValueError(
                f"EduPPOClip expects logits_new of shape [B, A]; got {tuple(logits_new.shape)}."
            )
        if logits_old.ndim != 2:
            raise ValueError(
                f"EduPPOClip expects logits_old of shape [B, A]; got {tuple(logits_old.shape)}."
            )
        if logits_new.shape != logits_old.shape:
            raise ValueError(
                "logits_new and logits_old must have the same shape [B, A]; got "
                f"{tuple(logits_new.shape)} and {tuple(logits_old.shape)}."
            )
        batch, num_actions = logits_new.shape
        if actions.shape != (batch,):
            raise ValueError(
                f"actions must have shape [{batch}]; got {tuple(actions.shape)}."
            )
        if advantages.shape != (batch,):
            raise ValueError(
                f"advantages must have shape [{batch}]; got {tuple(advantages.shape)}."
            )
        if (actions < 0).any() or (actions >= num_actions).any():
            raise ValueError(
                f"actions out of range [0, {num_actions}). Got min={int(actions.min())}, max={int(actions.max())}."
            )

        clip_epsilon = float(params.get("clip_epsilon", 0.2))
        if clip_epsilon < 0:
            raise ValueError("clip_epsilon must be >= 0.")
        temperature = float(params.get("temperature", 1.0))
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        verbose = context is not None and getattr(context, "verbose", False)
        recorder = StepRecorder() if verbose else None

        idx = actions.unsqueeze(1)
        logp_new = F.log_softmax(logits_new / temperature, dim=-1).gather(1, idx).squeeze(1)
        # The old policy is the constant behaviour policy that collected the
        # data — no gradient is needed through it, but we compute it explicitly
        # so the ratio is transparent.
        with torch.no_grad():
            logp_old = (
                F.log_softmax(logits_old / temperature, dim=-1).gather(1, idx).squeeze(1)
            )
        if recorder is not None:
            recorder.record(
                "log_probs",
                "log π(a|s) for the action taken, under the new and old policies.",
                scalars={"temperature": temperature, "num_actions": float(num_actions)},
                logp_new=logp_new, logp_old=logp_old,
            )

        ratio = torch.exp(logp_new - logp_old)
        if recorder is not None:
            recorder.record(
                "ratio",
                "ratio = exp(logp_new − logp_old) = π_new(a|s) / π_old(a|s).",
                scalars={"clip_epsilon": clip_epsilon},
                ratio=ratio,
            )

        clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        if recorder is not None:
            recorder.record(
                "clip",
                "clipped = clamp(ratio, 1 − ε, 1 + ε): confine the ratio to the trust region.",
                clipped=clipped,
            )

        surr1 = ratio * advantages
        surr2 = clipped * advantages
        if recorder is not None:
            recorder.record(
                "surrogates",
                "surr1 = ratio · A (unclipped); surr2 = clipped · A (clipped).",
                surr1=surr1, surr2=surr2,
            )

        objective = torch.min(surr1, surr2)
        if recorder is not None:
            recorder.record(
                "objective",
                "objective = min(surr1, surr2): the pessimistic (lower) surrogate per sample.",
                objective=objective,
            )

        loss = -objective.mean()
        if recorder is not None:
            recorder.record(
                "loss",
                "loss = − mean(objective). Descending it nudges the policy toward high-advantage actions, but only within the clip.",
                scalars={"loss": float(loss.item())},
                loss=loss,
            )

        result: dict[str, Any] = {
            "loss": loss,
            "ratio": ratio,
            "clipped_ratio": clipped,
            "objective": objective,
        }
        if recorder is not None and recorder.steps:
            result["__steps__"] = recorder.steps
        return result
