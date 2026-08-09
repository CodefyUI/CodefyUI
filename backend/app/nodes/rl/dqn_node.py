from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class DQNNode(BaseNode):
    NODE_NAME = "DQN"
    CATEGORY = "RL"
    DESCRIPTION = "Create a Deep Q-Network (simple MLP) for reinforcement learning"

    # #253/#254. This is ``SequentialModel``'s shape exactly: it builds an
    # ``nn.Module`` that owns trainable weights, it has no required input,
    # and it was cacheable -- a root nothing could ever invalidate. Anything
    # downstream that trains the network mutates it in place, and a cache
    # hit then hands the next run the module the previous run already
    # trained. #253 settled the rule ("a node that owns trainable
    # parameters is not cacheable") and fixed ``SequentialModel``; the three
    # RL constructors were the same omission one package over, and they are
    # also the only nodes in the registry that made #254's cache hits
    # reachable at all. No ``StatefulModuleMixin`` here: unlike
    # ``SequentialModel`` these rebuild a fresh network from their params
    # every run (``RewardModel`` from a declared seed), and giving them a
    # weight-persistence story is a product decision this change does not
    # need to make.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="state", data_type=DataType.TENSOR, description="State tensor for inference", optional=True),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="DQN model (nn.Module)"),
            PortDefinition(name="action", data_type=DataType.TENSOR, description="Q-values or selected action tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="state_dim", param_type=ParamType.INT, default=4, description="Dimension of the state space"),
            ParamDefinition(name="action_dim", param_type=ParamType.INT, default=2, description="Dimension of the action space"),
            ParamDefinition(name="hidden_dim", param_type=ParamType.INT, default=128, description="Hidden layer dimension"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch
        import torch.nn as nn

        state_dim = params.get("state_dim", 4)
        action_dim = params.get("action_dim", 2)
        hidden_dim = params.get("hidden_dim", 128)

        model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        state = inputs.get("state")
        if state is not None:
            # Match the network to the input's (global) device before applying.
            from ...core.device_utils import to_device
            model = to_device(model, state.device)
            with torch.no_grad():
                q_values = model(state)
        else:
            q_values = torch.zeros(action_dim)

        return {"model": model, "action": q_values}
