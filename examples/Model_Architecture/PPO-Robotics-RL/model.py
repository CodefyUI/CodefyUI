"""
Proximal Policy Optimization (PPO) -- Actor-Critic for Continuous Control

PPO, introduced by Schulman et al. (2017) at OpenAI, is the workhorse algorithm
behind many of the most impressive RL results of the past decade. It maximizes a
clipped surrogate objective that constrains policy updates to a trust region,
achieving the stability of TRPO with far simpler implementation.

Key components implemented here:
  1. Actor (Policy) Network: Outputs a Gaussian distribution over continuous
     actions parameterized by a learned mean and log standard deviation.
  2. Critic (Value) Network: Estimates the state-value function V(s) used as
     a baseline to reduce variance in policy gradient estimates.
  3. Generalized Advantage Estimation (GAE): Exponentially-weighted combination
     of multi-step TD errors that smoothly interpolates between bias and variance.
  4. Clipped Surrogate Objective: Prevents destructively large policy updates by
     clipping the importance-sampling ratio to [1-eps, 1+eps].

Significance:
  - Default algorithm for OpenAI's robotics research (dexterous manipulation,
    locomotion, sim-to-real transfer).
  - Core of RLHF (Reinforcement Learning from Human Feedback) used to align
    large language models such as ChatGPT and Claude.
  - Widely adopted in game AI (Dota 2, hide-and-seek), autonomous driving,
    and chip design optimization.

This script provides a complete, self-contained PPO training loop. It uses
gymnasium's CartPole-v1 when available (treating the discrete actions as
continuous indices) or falls back to a lightweight synthetic environment.

CodefyUI -- visual node-graph builder for ML pipelines.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ---------------------------------------------------------------------------
# Actor Network (Gaussian Policy)
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """Maps states to a diagonal Gaussian distribution over actions."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.net(state)
        mean = self.mean_head(features)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

# ---------------------------------------------------------------------------
# Critic Network (State-Value Function)
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    """Estimates the scalar value V(s) for a given state."""

    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPO:
    """Proximal Policy Optimization with clipped objective and GAE."""

    def __init__(self, state_dim: int, action_dim: int, device: torch.device,
                 lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_eps: float = 0.2, k_epochs: int = 10,
                 entropy_coef: float = 0.01, value_coef: float = 0.5) -> None:
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

    # -- action selection ---------------------------------------------------

    def act(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Sample an action from the policy, return (action, log_prob, value)."""
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, std = self.actor(s)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(s).squeeze(-1)
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    # -- evaluate stored transitions ----------------------------------------

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate log_probs, values, and entropy for a batch."""
        mean, std = self.actor(states)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(states).squeeze(-1)
        return log_probs, values, entropy

    # -- GAE computation ----------------------------------------------------

    @staticmethod
    def compute_gae(rewards: List[float], values: List[float],
                    dones: List[bool], gamma: float,
                    gae_lambda: float) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation and discounted returns."""
        advantages: List[float] = []
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + gamma * next_value * mask - values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages.insert(0, gae)
            next_value = values[t]

        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    # -- PPO update ---------------------------------------------------------

    def update(self, states: List[np.ndarray], actions: List[np.ndarray],
               log_probs: List[float], rewards: List[float],
               values: List[float], dones: List[bool]) -> Tuple[float, float]:
        """Run K epochs of PPO updates on the collected rollout."""
        advantages, returns = self.compute_gae(
            rewards, values, dones, self.gamma, self.gae_lambda
        )

        # Convert to tensors
        s_t = torch.FloatTensor(np.array(states)).to(self.device)
        a_t = torch.FloatTensor(np.array(actions)).to(self.device)
        old_lp = torch.FloatTensor(log_probs).to(self.device)
        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages for stable training
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0

        for _ in range(self.k_epochs):
            new_lp, new_values, entropy = self.evaluate(s_t, a_t)

            # Importance-sampling ratio
            ratio = (new_lp - old_lp).exp()

            # Clipped surrogate objective
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps,
                                1.0 + self.clip_eps) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss (simple MSE)
            value_loss = nn.functional.mse_loss(new_values, ret_t)

            # Combined loss
            loss = (policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy.mean())

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                max_norm=0.5,
            )
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        n = self.k_epochs
        return total_policy_loss / n, total_value_loss / n

# ---------------------------------------------------------------------------
# Simple Environment Wrapper
# ---------------------------------------------------------------------------

class SimpleEnv:
    """Wraps gymnasium CartPole or provides a synthetic continuous-control fallback."""

    def __init__(self) -> None:
        self.gym_env = None
        try:
            import gymnasium as gym
            self.gym_env = gym.make("CartPole-v1")
            self.state_dim = self.gym_env.observation_space.shape[0]
            self.action_dim = 1  # continuous proxy for discrete action
        except Exception:
            self.state_dim = 4
            self.action_dim = 1
            self._state: np.ndarray | None = None

    def reset(self) -> np.ndarray:
        if self.gym_env is not None:
            obs, _ = self.gym_env.reset()
            return obs.astype(np.float32)
        self._state = np.random.randn(self.state_dim).astype(np.float32) * 0.05
        return self._state.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        if self.gym_env is not None:
            discrete_action = 1 if float(action[0]) > 0.0 else 0
            obs, reward, terminated, truncated, _ = self.gym_env.step(discrete_action)
            return obs.astype(np.float32), float(reward), terminated or truncated
        # Synthetic dynamics: action pushes state, reward for staying near origin
        force = float(np.clip(action[0], -1.0, 1.0))
        self._state += np.array([force * 0.1, 0.02, -0.01, force * 0.05],
                                dtype=np.float32)
        self._state += np.random.randn(self.state_dim).astype(np.float32) * 0.02
        reward = 1.0 - float(np.abs(self._state).mean())
        done = bool(np.abs(self._state).max() > 2.5)
        return self._state.copy(), reward, done

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> None:
    device = get_device()
    print(f"[PPO] Using device: {device}")

    env = SimpleEnv()
    agent = PPO(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        device=device,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        k_epochs=10,
        entropy_coef=0.01,
        value_coef=0.5,
    )

    # Training hyperparameters
    num_episodes = 500
    max_steps = 500
    update_every = 2048  # collect this many steps before each PPO update

    # Rollout storage
    all_states: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    all_log_probs: List[float] = []
    all_rewards: List[float] = []
    all_values: List[float] = []
    all_dones: List[bool] = []
    total_steps = 0

    episode_rewards: List[float] = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        ep_reward = 0.0

        for step in range(max_steps):
            action, log_prob, value = agent.act(state)

            next_state, reward, done = env.step(action)

            all_states.append(state)
            all_actions.append(action)
            all_log_probs.append(log_prob)
            all_rewards.append(reward)
            all_values.append(value)
            all_dones.append(done)

            state = next_state
            ep_reward += reward
            total_steps += 1

            # PPO update when enough transitions have been collected
            if total_steps % update_every == 0 and len(all_states) > 0:
                p_loss, v_loss = agent.update(
                    all_states, all_actions, all_log_probs,
                    all_rewards, all_values, all_dones,
                )
                all_states.clear()
                all_actions.clear()
                all_log_probs.clear()
                all_rewards.clear()
                all_values.clear()
                all_dones.clear()

                if episode % 10 == 0:
                    print(f"    [update] policy_loss={p_loss:.4f}  "
                          f"value_loss={v_loss:.4f}")

            if done:
                break

        episode_rewards.append(ep_reward)
        avg_10 = np.mean(episode_rewards[-10:])

        if episode % 10 == 0 or episode == 1:
            print(f"  Episode {episode:>4d}  |  Reward: {ep_reward:7.2f}  |  "
                  f"Avg(10): {avg_10:7.2f}  |  Steps: {total_steps}")

    # Final update on any remaining rollout data
    if len(all_states) > 0:
        agent.update(all_states, all_actions, all_log_probs,
                     all_rewards, all_values, all_dones)

    print("[PPO] Training complete.")
    print(f"  Final 10-episode average reward: {np.mean(episode_rewards[-10:]):.2f}")


if __name__ == "__main__":
    train()
