# ResNet-18 / CIFAR-10 baseline

The standard CIFAR-10 reproduction, built entirely from GUI nodes. No Python is
written anywhere in this example — the architecture lives in the
`SequentialModel` layer editor, and everything else is node parameters.

For the full walkthrough, see the
[Reproducing Baselines](../../../docs/docs/usage/reproducing-baselines.md)
documentation page. This file is the provenance record: what was run, on what,
and what came out.

## Recipe

| | |
|---|---|
| Architecture | ResNet-18, CIFAR variant — 3x3 stride-1 stem, no maxpool, 4 stages of 2 BasicBlocks (64/128/256/512), 11,173,962 parameters |
| Optimizer | SGD, lr 0.1, momentum 0.9, Nesterov, weight decay 5e-4 |
| Schedule | `CosineAnnealingLR`, `T_max` = 200 (stepped once per epoch) |
| Loss | Cross-entropy, no label smoothing |
| Batch size | 128 train, 512 eval |
| Epochs | 200 |
| Augmentation | `RandomCrop(32, padding=4)`, `RandomHorizontalFlip(p=0.5)` |
| Normalization | CIFAR-10 statistics — mean (0.4914, 0.4822, 0.4465), std (0.2470, 0.2435, 0.2616) |
| Precision | bf16 autocast |
| Run options | `seed: 1337`, `deterministic: true`, `device: cuda` |

The parameter count is the check that the architecture is right: 11,173,962 is
exactly what `torchvision.models.resnet18(num_classes=10)` gives once its stem is
swapped for the CIFAR one and its maxpool removed.

### Why the CIFAR stem matters

`torchvision`'s ResNet-18 starts with a 7x7 stride-2 convolution and a 3x3
stride-2 max-pool, which is right for 224x224 ImageNet images and wrong for
32x32 CIFAR ones — it discards most of the spatial resolution before the first
residual block. The CIFAR variant used here replaces the stem with a single 3x3
stride-1 convolution and drops the max-pool. This is the usual reason a
"ResNet-18 on CIFAR-10" reproduction lands near 91% instead of 95%.

## Measured result

The graph in this directory, run unmodified, twice, with the same seed.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 (16 GB, compute capability 8.9) |
| Software | PyTorch 2.11.0+cu128, torchvision 0.26.0, Python 3.11, Windows 11 |
| Run options | `device: cuda`, `seed: 1337`, `deterministic: true` |
| **Top-1 test accuracy** | **95.48%** (9,548 / 10,000) |
| Wall clock | 22 min 29 s for 200 epochs, including dataset load and final evaluation |
| Throughput | 6.62 s/epoch mean (min 6.20, max 8.14), train pass plus test pass |
| Final train loss | 0.001669 |
| Final test loss | 0.172160 |

### Reproducibility

The second run was submitted with identical options after a server restart, so
it shared no process state with the first.

| | |
|---|---|
| Run A accuracy | 95.48% |
| Run B accuracy | 95.48% |
| **Difference** | **0.00 percentage points** |
| Metric points compared | 601 (200 train loss, 200 test loss, 200 lr, 1 accuracy) |
| Points differing | 0 — every value bitwise identical |

Bitwise equality on CUDA is a consequence of `deterministic: true`, which turns
off cuDNN's algorithm autotuner. Without it the same seed still gives a similar
curve, but the two runs drift apart numerically and the final accuracies differ
by a few hundredths of a point.

Different *seeds* vary by roughly plus or minus 0.3 points. That is the recipe's
inherent noise, not a defect.

### Evidence

`evidence/metrics-seed1337.csv` is the full metric export for run A, exactly as
`GET /api/runs/<id>/metrics?format=csv` produced it. `evidence/curves-seed1337.png`
plots the loss curves and the cosine schedule from that CSV.

There is no accuracy-against-epoch curve because no node emits one: `TrainingLoop`
records loss and learning rate only, and `EvaluateModel` records a single point at
the end. Tracked as issue #202.

### Stop and resume

Exercised on this recipe at 20 epochs (seed 2024): the run was stopped at epoch 8,
which wrote an interrupt checkpoint automatically, and a resumed graph with a
`CheckpointLoader` continued at epoch 9 and ran through to 20. Every epoch's
learning rate matched the theoretical cosine from `base_lrs` 0.1 to within 1e-12,
so the schedule was continuous across the stop rather than restarted.

The wiring order matters — see the note in the docs page. `Optimizer` feeds
`LRScheduler` feeds `CheckpointLoader`, never `Optimizer` into `CheckpointLoader`
into `LRScheduler`, because a scheduler built after the loader captures the
already-decayed learning rate as its `base_lrs` (issue #149).

## Running it

Open the example, then submit it from the Runs panel with `seed` 1337 and
`deterministic` enabled. The first run downloads CIFAR-10 (about 170 MB) into
`backend/data/`.

The run belongs to the server, not the browser tab — close the tab and come
back, and the Runs panel re-attaches and replays what it missed.

## Notes for anyone editing this graph

- **Every root node needs its own trigger edge from `Start`.** Execution walks
  forward from the entry points along data edges, so `Loss`, both `Dataset`
  nodes and the head of each transform chain would be pruned without one. This
  graph wires four. Leaving one out produces a missing-input error naming a node
  much further downstream (issue #201).
- **`EvaluateModel.device` is set to `cuda` explicitly.** It defaults to `cpu`
  and has no `auto` option, so it does not follow the run's device.
- **`num_workers` is 4 with `persistent_workers`.** The GPU does an epoch of
  compute in under 5 seconds on an RTX 4080; single-threaded PIL augmentation of
  50,000 images is several times that, so the data pipeline, not the GPU, sets
  the wall clock.
- The `TrainingLoop.epochs` value and `LRScheduler.T_max` must stay equal —
  cosine annealing is stepped once per epoch and reaches zero exactly at
  `T_max`.

## Continuous integration

`backend/tests/test_builtin_examples.py::test_resnet18_cifar10_baseline_short_epoch`
executes this exact graph for two optimizer steps on every CI run, against a
generated image folder rather than the real download. It asserts the layer
editor's spec still builds a 11,173,962-parameter model and that the whole
pipeline still runs, which the structural validation alone cannot check — the
layer graph is an opaque JSON string as far as `validate_graph` is concerned.
