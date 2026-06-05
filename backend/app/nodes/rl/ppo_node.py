from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class PPONode(BaseNode):
    NODE_NAME = "PPO"
    CATEGORY = "RL"
    DESCRIPTION = "Create a PPO actor-critic network for reinforcement learning"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="state", data_type=DataType.TENSOR, description="State tensor for inference", optional=True),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="PPO actor-critic model (nn.Module)"),
            PortDefinition(name="action", data_type=DataType.TENSOR, description="Action probabilities or selected action tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="state_dim", param_type=ParamType.INT, default=4, description="Dimension of the state space"),
            ParamDefinition(name="action_dim", param_type=ParamType.INT, default=2, description="Dimension of the action space"),
            ParamDefinition(name="hidden_dim", param_type=ParamType.INT, default=64, description="Hidden layer dimension"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch
        import torch.nn as nn

        state_dim = params.get("state_dim", 4)
        action_dim = params.get("action_dim", 2)
        hidden_dim = params.get("hidden_dim", 64)

        class ActorCritic(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.shared = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                )
                self.actor = nn.Linear(hidden_dim, action_dim)
                self.critic = nn.Linear(hidden_dim, 1)

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                features = self.shared(x)
                action_probs = torch.softmax(self.actor(features), dim=-1)
                value = self.critic(features)
                return action_probs, value

        model = ActorCritic()

        state = inputs.get("state")
        if state is not None:
            # Match the network to the input's (global) device before applying.
            from ...core.device_utils import to_device
            model = to_device(model, state.device)
            with torch.no_grad():
                action_probs, _value = model(state)
        else:
            action_probs = torch.zeros(action_dim)

        return {"model": model, "action": action_probs}
