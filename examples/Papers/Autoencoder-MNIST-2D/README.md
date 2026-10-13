# Autoencoder squeezes digits to 2-D

Geoffrey E. Hinton and Ruslan R. Salakhutdinov. *Reducing the dimensionality of data with neural networks.* Science 313(5786), 2006.

Claim: a deep autoencoder compresses data into a few numbers and rebuilds it better than PCA with the same number of components, and its 2-D codes separate the classes far better than PCA's.

## Recipe

| | |
|---|---|
| Data | MNIST (ships in `backend/data/MNIST`): the first 10,000 training images to train on, the first 1000 test images to score; pixels in [0, 1], flattened to 784 |
| Model | Linear 784 -> 256, ReLU, Linear 256 -> 2, Linear 2 -> 256, ReLU, Linear 256 -> 784, Sigmoid |
| Loss / optimizer | MSE, Adam lr 0.002, batch size 128 |
| Epochs | 40 |
| Baseline | PCA with 2 components, fitted on the same 10,000 images (`PythonScript`) |
| Seed | 0 |

## Measured

Measured with `backend/run_graph.py` on an Apple M3 CPU, torch 2.11, run seed 0 (the graph's `settings.seed`).

| | Autoencoder | PCA |
|---|---|---|
| Reconstruction MSE on 1000 test digits | 0.0438 | 0.0547 |
| 5-NN accuracy in the 2-D code space (chance 0.10) | 0.587 | 0.449 |

Whole graph: 4.9 s.

## Scale

The paper used a 784-1000-500-250-2 network pretrained layer by layer as RBMs on all 60,000 images; this is a single hidden layer per side trained end to end with Adam on 10,000, so its margin over PCA is smaller than the paper's.
