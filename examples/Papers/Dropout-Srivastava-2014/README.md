# Dropout narrows the overfit gap

Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever and Ruslan Salakhutdinov. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." *Journal of Machine Learning Research* 15 (2014): 1929-1958.

**Claim reproduced:** dropout on the hidden layers narrows the gap between training and test accuracy and raises test accuracy.

## Recipe

| | |
|---|---|
| Data | MNIST, 2,000 training images (`DatasetSplit` subset, seed 0), batch 64; scored on the 10,000 test images |
| Model | MLP 784-1024-1024-10, ReLU; one copy with `Dropout(0.5)` after each hidden layer |
| Optimizer | SGD, lr 0.01, momentum 0.9 |
| Loss | Cross-entropy |
| Epochs | 40 |
| Run seed | `2` (`settings.seed` in `graph.json`) |

## Measured

CPU (Apple M3), torch 2.11, run seed 2. The graph executes in about 5.5 s.

| Model | Training accuracy | Test accuracy | Final training loss |
|---|---|---|---|
| No dropout | 1.00 | 0.91 | 0.005 |
| Dropout 0.5 | 1.00 | 0.93 | 0.06 |

`backend/tests/test_paper_examples.py` runs this graph and checks these numbers.

## Scale

The paper trains on all 60,000 MNIST images for far longer, with max-norm regularisation, and reports test errors near 1%. This graph uses 2,000 images and 40 epochs so the plain network overfits within seconds; its accuracies are lower than the paper's.
