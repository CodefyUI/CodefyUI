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
        "observation. A seeded run resets the environment and seeds its action "
        "space from the run seed, so the first observation repeats; an "
        "unseeded run starts from fresh entropy."
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

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        import gymnasium as gym
        import torch

        env_name = params.get("env_name", "CartPole-v1")

        env = gym.make(env_name)
        # Gymnasium seeds its own generator from OS entropy on an unseeded
        # reset, which the run's seed_rngs (random, numpy, torch) does not
        # reach (#706). A seeded run hands it a seed derived from (run seed,
        # this node's id); None keeps an unseeded run unseeded.
        seed = None
        if context is not None:
            node_id = getattr(context, "current_node_id", "") or "env"
            seed = context.derive_seed(f"env:{node_id}")
        observation, _info = env.reset(seed=seed)
        if seed is not None:
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        observation_tensor = torch.tensor(observation, dtype=torch.float32)

        return {"env": env, "observation": observation_tensor}
