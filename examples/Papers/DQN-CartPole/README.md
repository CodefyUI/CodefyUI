# DQN balances CartPole

Volodymyr Mnih et al. *Playing Atari with Deep Reinforcement Learning.* NIPS Deep Learning Workshop (2013). The target network is from Mnih et al., *Human-level control through deep reinforcement learning*, Nature 518 (2015); Double DQN from van Hasselt, Guez and Silver, AAAI (2016).

## Recipe

| | |
|---|---|
| Env | `EnvWrapper` CartPole-v1 |
| Model | `DQN` node: state_dim 4, action_dim 2, hidden_dim 128 |
| Optimizer | Adam, lr 0.0005 |
| RLTrainingLoop | algorithm dqn, episodes 1000, gamma 0.99, replay_size 50000, batch_size 64, epsilon 1.0 to 0.05 over 200 episodes, target_update_every 500, learning_starts 1000, double_dqn true, grad_clip_norm 10, target_return 475, device cpu |
| Seed | 0 (run seed and loop seed) |

## Measured

Measured on an M3 MacBook Air, CPU, torch 2.11, gymnasium 1.2.3, run seed 0.

Solved at episode 514 in 26.6 s. CartPole-v1 counts as solved when the mean return of the last 20 episodes reaches 475 (`target_return`), and the run stops there.

Seeds 1 to 3 (run seed in Settings and the loop's `seed` both set to the number): seed 1 solves it at episode 707; seed 2 solves it at episode 578; seed 3 solves it at episode 610. With grad_clip_norm 0 and a 100-episode epsilon decay, seeds 0 to 3 of this graph still solve CartPole, at episodes 572, 546, 487 and 514.

## Compared with the paper

The paper learns from Atari pixels with a convolutional network, RMSProp and frame skipping over millions of frames. Here the state is CartPole's 4 numbers, the network is a small MLP, and Double DQN and gradient clipping are added.
