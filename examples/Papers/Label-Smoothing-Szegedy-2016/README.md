# Label smoothing lowers confidence

Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens and Zbigniew Wojna. "Rethinking the Inception Architecture for Computer Vision." *CVPR* 2016. Rafael Müller, Simon Kornblith and Geoffrey Hinton. "When Does Label Smoothing Help?" *NeurIPS* 2019.

**Claim reproduced:** label smoothing keeps test accuracy at least as high and makes the model less confident.

## Recipe

| | |
|---|---|
| Data | MNIST, 2,000 training images (`DatasetSplit` subset, seed 0), batch 64; accuracy on the 10,000 test images, confidence on the first 1,000 |
| Model | MLP 784-512-512-10, ReLU, one copy per loss |
| Optimizer | Adam, lr 0.001 |
| Loss | Cross-entropy with `label_smoothing` 0 and 0.1 |
| Epochs | 20 |
| Run seed | `0` (`settings.seed` in `graph.json`) |

## Measured

CPU (Apple M3), torch 2.11, run seed 0. The graph executes in about 1.9 s.

| Loss | Test accuracy | Mean top-class probability | Final training loss |
|---|---|---|---|
| Hard labels | 0.92 | 0.96 | 0.01 |
| Label smoothing 0.1 | 0.94 | 0.80 | 0.52 |

The smoothed run's training loss cannot fall below 0.50, the entropy of its target distribution (0.91 and nine 0.01s).

`backend/tests/test_paper_examples.py` runs this graph and checks these numbers.

## Scale

Szegedy et al. apply label smoothing to Inception-v3 on ImageNet, and Müller et al. measure calibration on CIFAR and ImageNet models. This graph uses an MLP on 2,000 MNIST images; at this scale smoothing 0.1 moves the model from over-confident (confidence above accuracy) to under-confident.
