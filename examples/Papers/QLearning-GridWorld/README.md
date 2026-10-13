# Q-learning crosses a trap grid

Christopher J. C. H. Watkins. *Learning from Delayed Rewards.* PhD thesis, University of Cambridge (1989). The gridworld follows Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018); the network and replay buffer follow Mnih et al. (2013).

## Recipe

| | |
|---|---|
| Env | `GridWorldEnv` 4x4, traps at (1,1) and (2,3), goal 1, trap -1, step 0, max_steps 30 |
| Model | `DQN` node: state_dim 16, action_dim 4, hidden_dim 32 |
| Optimizer | Adam, lr 0.001 |
| RLTrainingLoop | algorithm dqn, episodes 300, gamma 0.9, replay_size 50000, batch_size 64, epsilon 1.0 to 0.05 over 100 episodes, target_update_every 100, learning_starts 200, double_dqn true, device cpu |
| PolicyRollout | episodes 10, temperature 0.01, seed 0, on the trained model |
| Seed | 0 (run seed and loop seed) |

## Measured

Measured on an M3 MacBook Air, CPU, torch 2.11, gymnasium 1.2.3, run seed 0.

Training takes 0.8 s. The trained policy reaches the goal in 10 of 10 rollouts (`success_rate` 1.0), each in 6 steps, the shortest path.

Over seeds 1 to 3 the rollouts reach the goal every time, in 6 steps. With gamma 0.99 the policy still reaches the goal but takes 7 to 11 steps.

## Compared with the paper

Watkins' Q-learning updates a table of 16 x 4 values one transition at a time. This graph replaces the table with a small network trained from a replay buffer, which is the DQN form of the same update.
