# Adam vs SGD vs momentum

Diederik P. Kingma and Jimmy Ba. "Adam: A Method for Stochastic Optimization." *ICLR* 2015.

**Claim reproduced:** Adam reaches a lower training loss than SGD, with or without momentum, in the same number of steps.

## Recipe

| | |
|---|---|
| Data | MNIST, 10,000 training images (`DatasetSplit` subset, seed 0), batch 128 |
| Model | MLP 784-256-256-10, ReLU, one copy per optimizer |
| Optimizers | SGD lr 0.01; SGD lr 0.01 with momentum 0.9; Adam lr 0.001, betas (0.9, 0.999), eps 1e-8 |
| Loss | Cross-entropy |
| Epochs | 5 |
| Run seed | `0` (`settings.seed` in `graph.json`) |

## Measured

CPU (Apple M3), torch 2.11, run seed 0. The graph executes in about 0.9 s.

| Optimizer | Training loss, epoch 1 | Training loss, epoch 5 |
|---|---|---|
| SGD | 2.20 | 0.79 |
| SGD + momentum 0.9 | 1.39 | 0.27 |
| Adam | 0.80 | 0.21 |

`backend/tests/test_paper_examples.py` runs this graph and checks these numbers.

## Scale

The paper's MLP experiment uses two hidden layers of 1,000 units on all of MNIST, with dropout, over many more epochs, and also compares AdaGrad and RMSProp. This graph uses 256-unit layers, 10,000 images and 5 epochs.
