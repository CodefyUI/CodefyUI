from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class EnvWrapperNode(BaseNode):
    NODE_NAME = "EnvWrapper"
    CATEGORY = "RL"
    DESCRIPTION = "Create an environment by ID and reset it"
    DETAILS = (
        "Wraps Gymnasium, so env_name is any ID gymnasium.make accepts, such as "
        "CartPole-v1, and the package must be installed. GridWorldEnv is the "
        "environment with no such dependency. The reset also returns the first "
        "observation."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="env", data_type=DataType.ANY, description="Gymnasium environment instance"),
            PortDefinition(name="observation", data_type=DataType.TENSOR, description="Initial observation as tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="env_name", param_type=ParamType.STRING, default="CartPole-v1", description="Gymnasium environment ID"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import gymnasium as gym
        import torch

        env_name = params.get("env_name", "CartPole-v1")

        env = gym.make(env_name)
        observation, _info = env.reset()
        observation_tensor = torch.tensor(observation, dtype=torch.float32)

        return {"env": env, "observation": observation_tensor}
