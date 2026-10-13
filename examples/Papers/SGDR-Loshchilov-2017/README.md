# Cosine warm restarts (SGDR)

Ilya Loshchilov and Frank Hutter. "SGDR: Stochastic Gradient Descent with Warm Restarts." *ICLR* 2017.

**Claim reproduced:** cosine annealing with warm restarts ends at a lower loss than a constant learning rate in the same number of epochs, and the restarts show in the learning-rate curve.

## Recipe

| | |
|---|---|
| Data | MNIST, 5,000 training images (`DatasetSplit` subset, seed 0), batch 64; scored on the 10,000 test images |
| Model | MLP 784-256-256-10, ReLU, one copy per schedule |
| Optimizer | SGD, lr 0.05, momentum 0.9 |
| Schedule | Constant lr 0.05, or `CosineAnnealingWarmRestarts` with T_0 2 (the node's `T_max` field), T_mult 2, eta_min 0, stepped once per epoch |
| Loss | Cross-entropy |
| Epochs | 14 (SGDR cycles of 2, 4 and 8 epochs) |
| Run seed | `0` (`settings.seed` in `graph.json`) |

## Measured

CPU (Apple M3), torch 2.11, run seed 0. The graph executes in about 1.3 s.

| Schedule | Final training loss | Test accuracy |
|---|---|---|
| Constant lr 0.05 | 0.13 | 0.94 |
| SGDR | 0.015 | 0.95 |

Learning rate per epoch under SGDR: 0.05, 0.025, 0.05, 0.04268, 0.025, 0.00732, 0.05, 0.0481, 0.04268, 0.03457, 0.025, 0.01543, 0.00732, 0.0019.

`backend/tests/test_paper_examples.py` runs this graph and checks these numbers.

## Scale

The paper trains wide residual networks on CIFAR-10 and CIFAR-100 for up to 200 epochs. This graph uses an MLP on 5,000 MNIST images for 14 epochs.
