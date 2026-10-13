# REINFORCE balances CartPole

Ronald J. Williams. *Simple statistical gradient-following algorithms for connectionist reinforcement learning.* Machine Learning 8, 229-256 (1992).

## Recipe

| | |
|---|---|
| Env | `EnvWrapper` CartPole-v1 |
| Model | `DQN` node used as a policy MLP: state_dim 4, action_dim 2, hidden_dim 128 |
| Optimizer | Adam, lr 0.002 |
| RLTrainingLoop | algorithm reinforce, episodes 1000, gamma 0.99, baseline mean, normalize_returns true, episodes_per_update 1, entropy_coef 0.01, target_return 475, device cpu |
| Seed | 0 (run seed and loop seed) |

## Measured

Measured on an M3 MacBook Air, CPU, torch 2.11, gymnasium 1.2.3, run seed 0.

Solved at episode 280 in 2.0 s. CartPole-v1 counts as solved when the mean return of the last 20 episodes reaches 475 (`target_return`), and the run stops there.

Seeds 1 to 3 (run seed in Settings and the loop's `seed` both set to the number): seed 1 solves it at episode 292; seed 2 solves it at episode 685; seed 3 solves it at episode 293.

## Compared with the paper

Williams states the estimator and proves it follows the gradient; the paper has no CartPole run. The batch-mean baseline, return normalisation and entropy bonus are common modern additions that make a one-episode update stable.
