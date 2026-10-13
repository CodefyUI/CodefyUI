# BatchNorm tames a high LR

Sergey Ioffe and Christian Szegedy. "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." *ICML* 2015.

**Claim reproduced:** with batch normalisation a network trains at a learning rate the same network without it cannot use.

## Recipe

| | |
|---|---|
| Data | MNIST, 10,000 training images (`DatasetSplit` subset, seed 0), batch 64; scored on the 10,000 test images |
| Model | MLP 784-256-256-256-10, ReLU; one copy with `BatchNorm1d` after each hidden `Linear` |
| Optimizer | SGD, lr 0.5, no momentum |
| Loss | Cross-entropy |
| Epochs | 3 |
| Run seed | `0` (`settings.seed` in `graph.json`) |

## Measured

CPU (Apple M3), torch 2.11, run seed 0. The graph executes in about 0.7 s.

| Model | Training loss, epoch 1 | Training loss, epoch 3 | Test accuracy |
|---|---|---|---|
| No BatchNorm | 2.31 | 2.11 | 0.18 |
| BatchNorm | 0.43 | 0.11 | 0.90 |

`backend/tests/test_paper_examples.py` runs this graph and checks these numbers.

## Scale

The paper's headline result is an Inception network on ImageNet, with an MNIST experiment on a three-layer sigmoid MLP. This graph uses a ReLU MLP on 10,000 MNIST images for 3 epochs and one high learning rate.
