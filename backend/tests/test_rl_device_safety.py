"""The RL nodes follow their inputs' device; they never reach for a GPU.

**CPU is the default and the assumption.** ``ExecutionContext.device`` is
``"cpu"``, ``resolve_device`` falls back to CPU, and no RL node declares a
device parameter -- a saved graph therefore cannot pin a device that the next
reader's machine does not have. CUDA is opt-in, selected in the run options.

This suite has two halves. The first runs everywhere and pins the CPU
contract, because that is what most readers will execute. The second is a
regression guard for a real failure and needs a GPU, so it skips cleanly:
an environment is a plain Python object handing back CPU tensors, while
``SequentialModel`` places its weights on the run's device, and on a CUDA box
the I5-1 graph died with "Expected all tensors to be on the same device".
CPU-only testing could not catch that -- which is exactly how it shipped.

The contract:

  * ``PolicyRollout`` forwards on the model's own device and records back on
    the CPU, so no downstream node has to know where the policy sat (and on a
    CPU-only machine every transfer is a no-op);
  * the arithmetic nodes align their inputs, because a graph can feed them
    from two different branches;
  * and the numbers do not change with the device -- the textbook quotes them.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from app.nodes.rl.bradley_terry_node import BradleyTerryLossNode
from app.nodes.rl.discount_node import DiscountNode
from app.nodes.rl.gridworld_env_node import GridWorldEnvNode
from app.nodes.rl.group_relative_advantage_node import GroupRelativeAdvantageNode
from app.nodes.rl.policy_rollout_node import PolicyRolloutNode
from app.nodes.rl.ppo_clip_node import PPOClipObjectiveNode

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def _env(**params):
    p = {"size": 4, "traps": "1,1", "max_steps": 30}
    p.update(params)
    return GridWorldEnvNode().execute({}, p)["env"]


def _policy(device="cpu"):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 4)).to(device)


def _rollout(device, **params):
    p = {"episodes": 1, "temperature": 1.0, "seed": 8}
    p.update(params)
    return PolicyRolloutNode().execute(
        {"model": _policy(device), "env": _env()}, p
    )


# ── the CPU path: the default, and the only one most users have ─────────
#
# These never skip. CUDA is an opt-in the run options select; nothing in the
# RL nodes asks for a GPU, and the CPU path must stand on its own.

@pytest.mark.parametrize(
    "key", ["states", "actions", "rewards", "logits", "log_probs",
            "returns", "episode_lengths", "episode_ids"],
)
def test_rollout_on_cpu_keeps_everything_on_cpu(key):
    assert _rollout("cpu")[key].device.type == "cpu"


def test_rollout_follows_the_model_rather_than_reaching_for_a_gpu():
    """The device probe reads the policy's own placement. On a CPU-only box
    that is cpu, and the .to() calls are no-ops rather than transfers."""
    model = _policy("cpu")
    assert next(model.parameters()).device.type == "cpu"
    out = PolicyRolloutNode().execute(
        {"model": model, "env": _env()},
        {"episodes": 1, "temperature": 1.0, "seed": 8},
    )
    assert out["report"] == "episode 0:  14 steps, return +1.000, ended at goal"


def test_no_rl_node_declares_a_device_parameter():
    """A device knob would let a saved graph pin cuda and break every machine
    that does not have one. These nodes follow their inputs instead."""
    from app.nodes.rl.preference_dataset_node import PreferenceDatasetNode
    from app.nodes.rl.bradley_terry_node import BradleyTerryTrainNode

    for cls in (GridWorldEnvNode, PolicyRolloutNode, DiscountNode,
                PPOClipObjectiveNode, GroupRelativeAdvantageNode,
                BradleyTerryLossNode, PreferenceDatasetNode, BradleyTerryTrainNode):
        names = [p.name for p in cls.define_params()]
        assert "device" not in names, f"{cls.NODE_NAME} pins a device"


def test_gridworld_observations_are_cpu_tensors():
    """The env stays device-agnostic; PolicyRollout does the bridging."""
    env = _env()
    obs = env.reset()
    assert obs.device.type == "cpu"
    nxt, _, _, _ = env.step(1)
    assert nxt.device.type == "cpu"


# ── PolicyRollout ───────────────────────────────────────────────────────

@cuda_only
def test_rollout_runs_with_a_cuda_policy_and_a_cpu_env():
    """The exact failure: env observations are CPU, the policy is not."""
    out = _rollout("cuda")
    assert out["report"] == "episode 0:  14 steps, return +1.000, ended at goal"


@cuda_only
@pytest.mark.parametrize(
    "key", ["states", "actions", "rewards", "logits", "log_probs",
            "returns", "episode_lengths", "episode_ids"],
)
def test_rollout_records_on_the_cpu_whatever_the_policy_device(key):
    """Downstream nodes stay device-agnostic because everything comes back
    on the CPU. Without this, PPOClipObjective gets cuda log-probs and cpu
    advantages and fails the same way."""
    assert _rollout("cuda")[key].device.type == "cpu"


@cuda_only
def test_rollout_is_bit_identical_across_devices():
    """The textbook quotes this trajectory; it must not depend on hardware."""
    cpu, gpu = _rollout("cpu"), _rollout("cuda")
    assert cpu["actions"].tolist() == gpu["actions"].tolist()
    assert cpu["trajectory"] == gpu["trajectory"]
    assert cpu["log_probs"].tolist() == pytest.approx(gpu["log_probs"].tolist(), abs=1e-5)


def test_rollout_tolerates_a_model_without_parameters():
    """The device probe must not explode on an exotic policy."""

    class Constant(nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 4)

    out = PolicyRolloutNode().execute(
        {"model": Constant(), "env": _env()},
        {"episodes": 1, "temperature": 1.0, "seed": 8},
    )
    assert out["actions"].numel() > 0


# ── arithmetic nodes fed from two branches ──────────────────────────────

@cuda_only
def test_ppo_clip_accepts_advantages_and_ratio_on_different_devices():
    out = PPOClipObjectiveNode().execute(
        {"advantages": torch.tensor([0.5, 0.8, -0.3]),
         "ratio": torch.tensor([1.05, 1.40, 0.55], device="cuda")},
        {"epsilon": 0.2},
    )
    assert out["objective"].tolist() == pytest.approx([0.525, 0.96, -0.24], abs=1e-4)


@cuda_only
def test_ppo_clip_accepts_log_probs_on_different_devices():
    lp = [-0.5, -1.2]
    out = PPOClipObjectiveNode().execute(
        {"advantages": torch.tensor([1.0, -1.0]),
         "log_probs_new": torch.tensor(lp),
         "log_probs_old": torch.tensor(lp, device="cuda")},
        {"epsilon": 0.2},
    )
    assert out["ratio"].tolist() == pytest.approx([1.0, 1.0])


@cuda_only
def test_group_relative_advantage_accepts_a_cuda_expand_index():
    out = GroupRelativeAdvantageNode().execute(
        {"rewards": torch.tensor([0.0, 1.0]),
         "expand_index": torch.tensor([0, 0, 1], device="cuda")},
        {},
    )
    assert out["advantages_expanded"].tolist() == pytest.approx([-0.5, -0.5, 0.5])


@cuda_only
def test_group_relative_advantage_accepts_cuda_group_ids():
    out = GroupRelativeAdvantageNode().execute(
        {"rewards": torch.tensor([0.0, 1.0, 10.0, 20.0]),
         "group_ids": torch.tensor([0, 0, 1, 1], device="cuda")},
        {},
    )
    assert out["baseline"].tolist() == pytest.approx([0.5, 0.5, 15.0, 15.0])


@cuda_only
def test_discount_accepts_cuda_episode_ids():
    out = DiscountNode().execute(
        {"rewards": torch.tensor([0.0, 1.0, 0.0, 1.0]),
         "episode_ids": torch.tensor([0, 0, 1, 1], device="cuda")},
        {"gamma": 1.0},
    )
    assert out["returns"].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])


@cuda_only
def test_bradley_terry_loss_accepts_halves_on_different_devices():
    out = BradleyTerryLossNode().execute(
        {"reward_w": torch.tensor([1.82]),
         "reward_l": torch.tensor([-0.44], device="cuda")},
        {},
    )
    assert float(out["score_diff"]) == pytest.approx(2.26, abs=1e-4)
    assert float(out["preference_prob"]) == pytest.approx(0.905510, abs=1e-5)
