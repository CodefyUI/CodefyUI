# PPO balances CartPole

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347 (2017). GAE from Schulman et al., ICLR (2016).

## Recipe

| | |
|---|---|
| Env | `EnvWrapper` CartPole-v1 |
| Model | `PPO` node (actor-critic, tanh trunk): state_dim 4, action_dim 2, hidden_dim 64 |
| Optimizer | Adam, lr 0.001 |
| RLTrainingLoop | algorithm ppo, episodes 1500, gamma 0.99, steps_per_update 512, ppo_epochs 10, minibatch_size 64, clip_eps 0.2, gae_lambda 0.95, value_coef 0.1, entropy_coef 0.01, grad_clip_norm 0.5, target_return 475, device cpu |
| Seed | 0 (run seed and loop seed) |

## Measured

Measured on an M3 MacBook Air, CPU, torch 2.11, gymnasium 1.2.3, run seed 0.

Solved at episode 532 in 7.7 s. CartPole-v1 counts as solved when the mean return of the last 20 episodes reaches 475 (`target_return`), and the run stops there.

Seeds 1 to 3 (run seed in Settings and the loop's `seed` both set to the number): seed 1 solves it at episode 514; seed 2 solves it at episode 427; seed 3 solves it at episode 397.

## Compared with the paper

The paper reports MuJoCo and Atari with parallel environments; this graph runs one CartPole environment and a smaller value-loss weight (0.1 against the paper's 0.5 to 1).
