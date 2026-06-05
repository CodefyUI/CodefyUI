"""RewardModelNode — the scalar-reward head used in RLHF.

In Reinforcement Learning from Human Feedback, you train a small head on
top of a frozen language model that maps a hidden state to a single scalar
reward (how "good" the response is, per human raters). That scalar is what
PPO maximises during the policy-update phase.

This node is the head: a tiny MLP that takes the last-token hidden state
of a sequence and returns one number per sequence. For teaching, that's
enough — the textbook can wire it after a Transformer / LLM block and
demonstrate Bradley-Terry pairwise loss in the preset.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class _RewardHead(nn.Module):
    """Two-layer MLP → scalar reward."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, H] or [B, T, H] (we use the last token in the sequence case).
        if h.dim() == 3:
            h = h[:, -1, :]
        return self.net(h).squeeze(-1)  # [B]


class RewardModelNode(BaseNode):
    NODE_NAME = "RewardModel"
    CATEGORY = "RL"
    DESCRIPTION = (
        "RLHF reward head. Tiny MLP that scores a sequence with one scalar — "
        "the thing you train on human preferences and then have PPO maximise. "
        "Accepts [B, H] (one vector per item) or [B, T, H] (uses last token)."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="hidden_states",
                data_type=DataType.TENSOR,
                description="Hidden states from the policy model: [B, H] or [B, T, H].",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="The reward MLP head (nn.Module)."),
            PortDefinition(name="rewards", data_type=DataType.TENSOR, description="Scalar reward per item, shape [B]."),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="input_dim", param_type=ParamType.INT, default=128, min_value=1, description="Hidden-state dimension H."),
            ParamDefinition(name="hidden_dim", param_type=ParamType.INT, default=64, min_value=1, description="MLP bottleneck width."),
            ParamDefinition(name="seed", param_type=ParamType.INT, default=42, description="Init seed for reproducibility."),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        input_dim = int(params.get("input_dim", 128))
        hidden_dim = int(params.get("hidden_dim", 64))
        seed = int(params.get("seed", 42))

        gen_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(seed)
            model = _RewardHead(input_dim, hidden_dim)
        finally:
            torch.random.set_rng_state(gen_state)

        h = inputs.get("hidden_states")
        if h is None:
            rewards = torch.zeros(0)
        else:
            if not isinstance(h, torch.Tensor):
                h = torch.as_tensor(h, dtype=torch.float32)
            # Match the head to the input's (global) device before applying.
            from ...core.device_utils import to_device
            model = to_device(model, h.device)
            with torch.no_grad():
                rewards = model(h.float())

        return {"model": model, "rewards": rewards}
