# Skip connections train deeper

Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. "Deep Residual Learning for Image Recognition." *CVPR* 2016.

**Claim reproduced:** a deep plain network fits its training set worse than the same depth with identity shortcuts (the degradation problem).

## Recipe

| | |
|---|---|
| Data | MNIST, 5,000 training images (`DatasetSplit` subset, seed 0), batch 64; scored on the subset and on the 10,000 test images |
| Model | 20 hidden blocks of `Linear` (width 32) -> `BatchNorm1d` -> `ReLU`, then `Linear` 32->10. The residual copy adds each block's input back before the ReLU with an `Add` merge in the layer editor's graph mode (19 shortcuts; the first block changes width and has none) |
| Optimizer | SGD, lr 0.05, momentum 0.9 |
| Loss | Cross-entropy |
| Epochs | 8 |
| Run seed | `0` (`settings.seed` in `graph.json`) |

## Measured

CPU (Apple M3), torch 2.11, run seed 0. The graph executes in about 3.8 s.

| Model | Training accuracy | Test accuracy | Final training loss |
|---|---|---|---|
| Plain, 20 layers | 0.77 | 0.76 | 0.79 |
| Residual, 20 layers | 0.95 | 0.91 | 0.27 |

`backend/tests/test_paper_examples.py` runs this graph and checks these numbers.

## Scale

The paper compares convolutional networks of 18 to 152 layers on ImageNet and 20 to 110 layers on CIFAR-10. This graph uses fully connected blocks of width 32 on 5,000 MNIST images for 8 epochs; the narrow width is what makes 20 plain layers hard to optimise at this scale.
