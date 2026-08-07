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

Open **ResNet-18 / CIFAR-10 baseline** from the empty-canvas gallery, or load its graph directly. Everything referred to on this page lives in [`examples/Usage_Example/ResNet18-CIFAR10-Baseline/`](https://github.com/CodefyUI/CodefyUI/tree/main/examples/Usage_Example/ResNet18-CIFAR10-Baseline) — the `graph.json`, a `README.md` recording exactly what was run, and an `evidence/` directory holding the raw metric export and the plotted curves.

Nineteen nodes, in four strands that meet at `TrainingLoop`:

- **Augmentation** — `RandomCrop` to `RandomHorizontalFlip` to `ToTensorTransform` to `NormalizeTransform`, feeding the training `Dataset`'s `train_transform` port. Transform nodes chain directly into one another; there is no separate Compose step.
- **Evaluation preprocessing** — `ToTensorTransform` to `NormalizeTransform` only, feeding the test `Dataset`'s `eval_transform` port. No augmentation on the test set, ever.
- **Model** — one `SequentialModel` whose layer editor holds the ResNet-18 graph: 70 nodes, of which 68 are modules and 2 are the `Input`/`Output` markers.
- **Optimization** — `Optimizer`, `Loss` and `LRScheduler`.

`EvaluateModel` reads the trained model and the test split and reports top-1 accuracy. `CheckpointSaver` writes the weights. `Visualize` draws the loss curve.

:::tip Do not place the architecture by hand
Sixty-eight layer nodes and seventy-seven connections is not something anyone should drag out one node at a time — it is several hundred precise interactions, and a mis-drawn residual shortcut stays invisible until the tensor shapes disagree.

Use the layer editor's **Import** instead. It accepts this example's `graph.json` directly and loads the whole architecture in one step, which is also how to reuse this ResNet-18 in a graph of your own: open your `SequentialModel`, **Import**, pick the file. **Export** writes the same format back out. The layer editor is the right place to *inspect* and *adjust* the architecture; it is not the right place to type it in from scratch.
:::

## The recipe

| | |
|---|---|
| Architecture | ResNet-18, CIFAR variant — 3x3 stride-1 stem, no maxpool, 4 stages of 2 BasicBlocks (64/128/256/512), 11,173,962 parameters |
| Optimizer | SGD, lr 0.1, momentum 0.9, Nesterov, weight decay 5e-4 |
| Schedule | `CosineAnnealingLR`, `T_max` equal to the epoch count (see the gotchas below — a mismatch is warned about, not enforced) |
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
        "graph": {"nodes": [ ... ], "edges": [ ... ]},
        "options": {"device": "cuda", "seed": 1337, "deterministic": true}
      }'
```

Substitute the `nodes` and `edges` arrays from the example's `graph.json` — the request body carries the whole graph, so this snippet does nothing until you paste them in.

## The measured result

Measured on the hardware below by running the shipped graph unmodified, twice, with the same seed.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 (16 GB, compute capability 8.9) |
| Software | PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, Python 3.11 |
| Seed | 1337, `deterministic: true` |
| Epochs | 200 |
| Wall clock | 22 min 29 s end to end (6.62 s/epoch mean including the test pass, ranging 6.0 to 8.1 s; the remaining half-minute is dataset load and the final evaluation) |
| **Top-1 test accuracy** | **95.48%** |
| Same-seed re-run | 95.48% — a difference of **0.00 percentage points** |

The re-run was submitted after a server restart, so it shared no process state with the first. All 601 recorded metric points — 200 training losses, 200 test losses, 200 learning rates and the final accuracy — came out **bitwise identical**.

`deterministic: true` is what makes that achievable: it asks for deterministic kernels and stops cuDNN re-picking convolution algorithms between runs. It is a best-effort request rather than a guarantee — an operation with no deterministic kernel warns and proceeds — but for this recipe it delivers exactly, verified across two processes. Without it the same seed still produces a similar curve, but the two runs drift apart numerically.

There is one honest caveat about the split layout: this graph has no validation set separate from the test set. The test split feeds both `TrainingLoop.val_dataloader` and `EvaluateModel`, which is the usual arrangement for a CIFAR-10 baseline. Nothing *selects* on it — early stopping is off, and the checkpoint saved is the final-epoch model, not a best-on-validation one — so the reported number is a clean held-out measurement. The curves are labelled "test" rather than "val" for that reason. If you intend to tune hyperparameters, carve a validation split out of the training set first.

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

To continue, add a `CheckpointLoader` to the graph, point it at that checkpoint, and route its four restored outputs into `TrainingLoop`:

- `CheckpointLoader.model` to `TrainingLoop.model`
- `CheckpointLoader.optimizer` to `TrainingLoop.optimizer`
- `CheckpointLoader.lr_scheduler` to `TrainingLoop.lr_scheduler`
- `CheckpointLoader.epoch` to `TrainingLoop.start_epoch`

`TrainingLoop.epochs` is an absolute target, not a count of further epochs, so leave it at 200 and the resumed run continues from the epoch after the one recorded in the checkpoint.

One boundary to know about if you are chasing exact numbers: a Stop that lands **mid-epoch** keeps that epoch's weight updates — they are already in the model — but does not count the epoch, because its loss average is incomplete. The resumed run re-runs that epoch from batch 0. The learning-rate schedule stays continuous across the stop; the data pass does not. A stopped-and-resumed run is therefore very close to, but not bitwise identical with, an uninterrupted one. Stopping between epochs avoids this entirely.

:::warning Wire the scheduler *into* the loader, not around it
Wire `Optimizer` into `LRScheduler`, then `LRScheduler` into `CheckpointLoader.lr_scheduler`. That input is optional, so it is easy to leave unconnected — and leaving it unconnected is what actually costs you something.

**It is not about `base_lrs`.** A scheduler constructed on an already-restored optimizer still starts from 0.1: `Optimizer.state_dict()` carries the `initial_lr` that the first scheduler stamped onto the param group, `load_state_dict` restores it, and `LRScheduler.__init__` reads it with `setdefault` rather than overwriting it. Both wirings were measured against the theoretical cosine and agree with it to 1.4e-17.

**What the wiring decides is whether the checkpoint's saved schedule position is restored or reconstructed.** `CheckpointLoader` can only restore `scheduler_state_dict` into a scheduler that is wired into it. Leave that input unconnected and the saved state is discarded with a log line, and the schedule is instead rebuilt by replaying `start_epoch` steps from `base_lrs`. For `CosineAnnealingLR` the replay is exact — which is why the recipe on this page is safe either way — but a metric-driven `ReduceLROnPlateau` cannot be replayed: its `best` and `num_bad_epochs` reset silently, postponing a decay that may have been one epoch away. Issue [#149](https://github.com/CodefyUI/CodefyUI/issues/149) tracks the general problem.
:::

## Checking the numbers yourself

Every run's metrics are queryable:

```bash
curl "http://127.0.0.1:8000/api/runs/<run_id>/metrics?format=csv" -o metrics.csv
```

`train_loss`, `val_loss` and `lr` are recorded once per epoch. **`eval_accuracy` is not** — `EvaluateModel` writes a single point when the run finishes. A 200-epoch export is therefore 601 rows, not 800, and there is no accuracy-against-epoch curve to plot: no node in the product emits one. Tracked as issue [#202](https://github.com/CodefyUI/CodefyUI/issues/202).

`tensorboard` is already `true` on this example's `TrainingLoop`, so each run also writes event files under its artifact directory, readable by any TensorBoard install.

## Gotchas worth knowing

- **Give every root node a trigger edge.** Execution starts from `Start` and follows *data* edges forward, so a node with no incoming data edge is skipped unless a trigger edge from `Start` points at it. This example has exactly four such nodes, and wires all four:

  | Trigger target | Why it is a root |
  |---|---|
  | `RandomCrop` | head of the training augmentation chain |
  | `ToTensorTransform` (evaluation) | head of the evaluation preprocessing chain |
  | `SequentialModel` | the model has no data input |
  | `Loss` | the loss function has no data input |

  Note what is **not** on that list: neither `Dataset` node is a root. Each one is fed by its transform chain (`aug-norm` into `ds-train.train_transform`, `ev-norm` into `ds-test.eval_transform`), so the forward walk reaches them on its own — wiring a trigger to a `Dataset` is harmless but does nothing.

  Miss a trigger and the run fails with a missing-input error naming a node much further downstream. Drop the one into `SequentialModel`, for instance, and `Optimizer` and `LRScheduler` are pruned with it; the run then dies complaining about `TrainingLoop`. Tracked as issue [#201](https://github.com/CodefyUI/CodefyUI/issues/201).

- **Keep `LRScheduler.T_max` equal to `TrainingLoop.epochs`.** Cosine annealing is stepped once per epoch and reaches zero exactly at `T_max`. Set `T_max` too high and the run stops partway down the curve, never annealing fully, which costs roughly a point of accuracy; too low and the cosine turns back **up** past `T_max`, so the tail of the run trains at a rising learning rate. `TrainingLoop` warns when the two disagree — in the server log, in the run log the Runs panel shows, and in the canvas Log tab — but it does not enforce the pair, because a truncated schedule is a legitimate choice and `CosineAnnealingWarmRestarts` reuses the same value as `T_0`, where equality would mean no restart ever happens. The same check covers `OneCycleLR.total_steps`, whose default of 1000 is a batch count no epoch budget reaches.
- **`EvaluateModel` does not follow the run device.** Its `device` parameter defaults to `cpu` and offers no `auto`. Set it to `cuda` explicitly or evaluation will be slow. Tracked as issue [#204](https://github.com/CodefyUI/CodefyUI/issues/204).
- **The first run downloads CIFAR-10** (about 170 MB). It lands in `backend/data/` by default, or in `<project>/assets/data` when a project directory is open. Later runs reuse it.

The CIFAR-10 dataset is Krizhevsky, *Learning Multiple Layers of Features from Tiny Images* (2009); it is downloaded at run time and not redistributed with CodefyUI.
