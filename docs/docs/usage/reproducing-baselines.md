---
sidebar_position: 3.9
title: Reproducing Baselines
description: A research-grade walkthrough — reproduce the standard ResNet-18 / CIFAR-10 result end to end from GUI nodes, with a fixed seed, on the run queue.
---

# Reproducing Baselines

Most of this documentation is about learning. This page is about the other claim: that a graph you build on the canvas can produce a number you would be willing to put in a paper, and that someone else can re-derive it.

The worked example is the standard CIFAR-10 baseline — ResNet-18, SGD with momentum, cosine annealing, crop-and-flip augmentation. It is the "hello world" of image-classification reproducibility, its expected accuracy is common knowledge, and that is exactly what makes it a good test: there is a published number to miss.

Nothing below involves writing Python. The architecture, the optimizer, the schedule and the augmentation chain are all nodes.

## The example

Open **ResNet-18 / CIFAR-10 baseline** from the empty-canvas gallery, or load `examples/Usage_Example/ResNet18-CIFAR10-Baseline/graph.json`.

Nineteen nodes, in four strands that meet at `TrainingLoop`:

- **Augmentation** — `RandomCrop` to `RandomHorizontalFlip` to `ToTensorTransform` to `NormalizeTransform`, feeding the training `Dataset`'s `train_transform` port. Transform nodes chain directly into one another; there is no separate Compose step.
- **Evaluation preprocessing** — `ToTensorTransform` to `NormalizeTransform` only, feeding the test `Dataset`'s `eval_transform` port. No augmentation on the test set, ever.
- **Model** — one `SequentialModel` whose layer editor holds the 70-layer ResNet-18 graph.
- **Optimization** — `Optimizer`, `Loss` and `LRScheduler`.

`EvaluateModel` reads the trained model and the test split and reports top-1 accuracy. `CheckpointSaver` writes the weights. `Visualize` draws the loss curve.

## The recipe

| | |
|---|---|
| Architecture | ResNet-18, CIFAR variant — 3x3 stride-1 stem, no maxpool, 4 stages of 2 BasicBlocks (64/128/256/512), 11,173,962 parameters |
| Optimizer | SGD, lr 0.1, momentum 0.9, Nesterov, weight decay 5e-4 |
| Schedule | `CosineAnnealingLR`, `T_max` equal to the epoch count |
| Loss | Cross-entropy, no label smoothing |
| Batch size | 128 train, 512 eval |
| Epochs | 200 |
| Augmentation | `RandomCrop(32, padding=4)`, `RandomHorizontalFlip(p=0.5)` |
| Normalization | CIFAR-10 channel statistics — mean (0.4914, 0.4822, 0.4465), std (0.2470, 0.2435, 0.2616) |
| Precision | bf16 autocast |

### Why the stem is different

The ResNet-18 in `torchvision` is built for 224x224 ImageNet images: a 7x7 stride-2 convolution followed by a 3x3 stride-2 max-pool, which reduces the input fourfold before the first residual block. Applied to a 32x32 CIFAR image that throws away almost all the spatial information up front, and the same recipe lands several points lower.

The CIFAR variant replaces the whole stem with a single 3x3 stride-1 convolution and deletes the max-pool, so the first stage still sees 32x32. Every published CIFAR ResNet number assumes this. It is the single most common reason a "ResNet-18 on CIFAR-10" reproduction comes out at 91% instead of 95%.

You can see the stem in the layer editor: double-click `SequentialModel`, and the first three layers are `Conv2d(3, 64, kernel_size=3, stride=1, padding=1)`, `BatchNorm2d(64)`, `ReLU`.

## Running it reproducibly

Two run options do the work, both set when you submit the run rather than on any node:

- **`seed`** — one integer seeds everything. Weight initialization, the shuffling order of the training loader, and the augmentation draws all derive from it, per node, so the result does not depend on which node happened to execute first.
- **`deterministic`** — asks for deterministic kernels and turns off cuDNN's algorithm autotuner. Without it, cuDNN re-picks convolution algorithms per run and two runs with the same seed drift apart. On this workload it costs about 4% of throughput, which is a good trade for an exactly repeatable number.

Submit from the Runs panel, or over the API:

```bash
curl -X POST http://127.0.0.1:8000/api/runs \
  -H "Content-Type: application/json" \
  -H "X-CodefyUI-Token: $CODEFYUI_TOKEN" \
  -d '{
        "name": "resnet18-cifar10-seed1337",
        "graph": {"nodes": [], "edges": []},
        "options": {"device": "cuda", "seed": 1337, "deterministic": true}
      }'
```

(`nodes` and `edges` are the contents of the example's `graph.json`.)

## The measured result

Measured on the hardware below by running the shipped graph unmodified, twice, with the same seed.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 (16 GB, compute capability 8.9) |
| Software | PyTorch 2.11.0+cu128, torchvision 0.26.0, Python 3.11 |
| Seed | 1337, `deterministic: true` |
| Epochs | 200 |
| Wall clock | 22 min 29 s (6.62 s/epoch mean, including the test pass) |
| **Top-1 test accuracy** | **95.48%** |
| Same-seed re-run | 95.48% — a difference of **0.00 percentage points** |

The re-run was submitted after a server restart, so it shared no process state with the first. All 601 recorded metric points — 200 training losses, 200 test losses, 200 learning rates and the final accuracy — came out **bitwise identical**.

That exactness is what `deterministic: true` buys. Without it the same seed still produces a similar curve, but cuDNN re-picks its convolution algorithms and the two runs drift apart numerically.

Expect a spread of roughly plus or minus 0.3 percentage points across *different* seeds; that is the run-to-run noise of this recipe, not a defect. The same seed is what should be tight.

If your number is several points low rather than a fraction low, check the stem before anything else.

## Walking away from a run

A run belongs to the server, not to the browser tab that submitted it. This is the part worth exercising once so you trust it:

1. Submit the run.
2. Close the tab.
3. Re-open the canvas later.

The Runs panel re-attaches to the still-running job and replays the events it missed. Nothing is lost and nothing was paused while you were gone. See [Run Queue](./run-queue.md) for lanes and concurrency.

## Stopping and resuming

Pressing **Stop** is cooperative: the training node finishes the batch it is on, writes a checkpoint, and returns its partial results. The checkpoint is registered as a run artifact, with the number of *completed* epochs in its metadata.

To continue, add a `CheckpointLoader` to the graph, point it at that checkpoint, and route the three restored values through it:

- `CheckpointLoader.model` to `TrainingLoop.model`
- `CheckpointLoader.optimizer` to `TrainingLoop.optimizer`
- `CheckpointLoader.lr_scheduler` to `TrainingLoop.lr_scheduler`
- `CheckpointLoader.epoch` to `TrainingLoop.start_epoch`

`TrainingLoop.epochs` is an absolute target, not a count of further epochs, so leave it at 200 and the resumed run continues from where it stopped.

:::warning Build the scheduler before the loader, not after
Wire `Optimizer` into `LRScheduler` and then `LRScheduler` into `CheckpointLoader` — not `Optimizer` into `CheckpointLoader` into `LRScheduler`.

A PyTorch scheduler captures `base_lrs` from the optimizer at the moment it is constructed. If the loader has already restored a decayed learning rate onto the optimizer, a scheduler built afterwards treats that decayed value as its starting point and anneals a second time from there. Building the scheduler from the fresh optimizer and then restoring its state keeps `base_lrs` at 0.1. This is tracked as issue #149.
:::

## Checking the numbers yourself

Every run's metrics are queryable, and `train_loss`, `val_loss`, `lr` and `eval_accuracy` are all recorded per epoch:

```bash
curl "http://127.0.0.1:8000/api/runs/<run_id>/metrics?format=csv" -o metrics.csv
```

Setting `tensorboard: true` on `TrainingLoop` additionally writes event files under the run's artifact directory, readable by any TensorBoard install.

## Gotchas worth knowing

- **Give every root node a trigger edge.** Execution starts from `Start` and follows data edges forward, so a node with no incoming data edge — `Loss`, each `Dataset`, the head of each transform chain — is skipped unless `Start` points at it. Miss one and the run fails with a missing-input error naming a node much further downstream. The shipped example wires all four. Tracked as issue #201.
- **`EvaluateModel` does not follow the run device.** Its `device` parameter defaults to `cpu` and offers no `auto`. Set it to `cuda` explicitly or evaluation will be slow.
- **The first run downloads CIFAR-10** (about 170 MB) into `backend/data/`. Later runs reuse it.
