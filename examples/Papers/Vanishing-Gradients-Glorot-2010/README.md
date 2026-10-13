# Sigmoid gradients vanish in depth

Xavier Glorot and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." *AISTATS* 2010. ReLU: Vinod Nair and Geoffrey Hinton. "Rectified Linear Units Improve Restricted Boltzmann Machines." *ICML* 2010.

**Claim reproduced:** under the standard initialisation a deep sigmoid network stalls with a small gradient while the same depth with ReLU trains.

## Recipe

| | |
|---|---|
| Data | MNIST, 5,000 training images (`DatasetSplit` subset, seed 0), batch 64; scored on the 10,000 test images |
| Model | 4 hidden layers of 128 units, `Sigmoid` in one copy and `ReLU` in the other; PyTorch's default `Linear` init, U(-1/sqrt(n), 1/sqrt(n)) |
| Optimizer | SGD, lr 0.05, momentum 0.9 |
| Loss | Cross-entropy |
| Epochs | 10, with `log_grad_norm` on (`log_interval` 10) |
| Run seed | `0` (`settings.seed` in `graph.json`) |

## Measured

CPU (Apple M3), torch 2.11, run seed 0. The graph executes in about 1.1 s.

| Activation | Training loss, epoch 1 | Training loss, epoch 10 | Test accuracy | Mean grad_norm, second half |
|---|---|---|---|---|
| Sigmoid | 2.35 | 2.30 | 0.11 | 0.21 |
| ReLU | 1.66 | 0.20 | 0.91 | 1.30 |

The sigmoid net's test accuracy equals the share of the most common digit in the test set (1,135 of 10,000 are ones).

`backend/tests/test_paper_examples.py` runs this graph and checks these numbers.

## Scale

The paper trains five hidden layers of 1,000 units on Shapeset-3x2, MNIST and CIFAR-10 for millions of updates, and proposes normalised initialisation as the fix. This graph uses 4 layers of 128 units on 5,000 MNIST images for 10 epochs; `SequentialModel` has no init option, so it shows the ReLU fix.
