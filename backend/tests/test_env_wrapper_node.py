"""Tests for EnvWrapperNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.rl.env_wrapper_node import EnvWrapperNode


def test_node_metadata():
    assert EnvWrapperNode.NODE_NAME == "EnvWrapper"
    assert EnvWrapperNode.CATEGORY == "RL"
    out_names = [p.name for p in EnvWrapperNode.define_outputs()]
    assert "env" in out_names
    assert "observation" in out_names


def test_default_creates_cartpole_env():
    res = EnvWrapperNode().execute({}, {"env_name": "CartPole-v1"})
    assert res["env"] is not None
    # CartPole observation is 4D
    assert res["observation"].shape == (4,)
    assert res["observation"].dtype == torch.float32


def test_observation_is_tensor_type():
    res = EnvWrapperNode().execute({}, {"env_name": "CartPole-v1"})
    assert isinstance(res["observation"], torch.Tensor)


def test_unknown_env_raises():
    with pytest.raises(Exception):
        EnvWrapperNode().execute({}, {"env_name": "DefinitelyNotARealEnv-v99"})


def test_env_can_step():
    res = EnvWrapperNode().execute({}, {"env_name": "CartPole-v1"})
    env = res["env"]
    obs, reward, terminated, truncated, info = env.step(0)
    assert len(obs) == 4
