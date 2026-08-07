# ResNet-18 / CIFAR-10 baseline

The standard CIFAR-10 reproduction, built entirely from GUI nodes. No Python is
written anywhere in this example and none executes as part of it — the
architecture lives in the `SequentialModel` layer editor, and everything else is
node parameters.

The architecture inside that editor is 68 layer nodes and 77 connections (70
nodes counting the `Input`/`Output` markers). It is meant to be **imported**, not
placed by hand: the layer editor's `Import` accepts this directory's `graph.json`
directly and loads the whole thing in one step, and `Export` writes the same
format back out. That is also how to reuse this ResNet-18 in a graph of your own.

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
| Software | PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, Python 3.11, Windows 11 |
| Run options | `device: cuda`, `seed: 1337`, `deterministic: true` |
| **Top-1 test accuracy** | **95.48%** (9,548 / 10,000) |
| Wall clock | 22 min 29 s (1,349.3 s) for 200 epochs, including dataset load and final evaluation |
| Throughput | 6.62 s/epoch mean, ranging roughly 6.0 to 8.1 s, train pass plus test pass |
| Final train loss | 0.001669 |
| Final test loss | 0.172160 |

The metric export below spans 1,319 s: it is timestamped from the first
epoch-end to the last, so it excludes the dataset load before epoch 1 and the
final evaluation after epoch 200. Both are inside the 1,349.3 s wall clock.
Per-epoch gaps depend slightly on which series you time them from (6.0 to 8.1 s
from `lr`/`val_loss`, 6.2 to 8.1 s from `train_loss`, because the three points
of an epoch are written a fraction of a second apart); the mean is 6.62 s either
way.

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

`deterministic: true` is what makes bitwise equality achievable on CUDA: it asks
for deterministic kernels and turns off cuDNN's algorithm autotuner. It is a
best-effort request rather than a guarantee — an op with no deterministic kernel
warns and proceeds — but for this recipe it delivers exactly, across two
processes. Without it the same seed still gives a similar curve, but the two
runs drift apart numerically and the final accuracies differ by a few
hundredths of a point.

Different *seeds* vary by roughly plus or minus 0.3 points. That is the recipe's
inherent noise, not a defect.

Note also that this graph has no validation split separate from the test set:
`ds-test` feeds both `TrainingLoop.val_dataloader` and `EvaluateModel`, which is
the usual arrangement for a CIFAR-10 baseline. Nothing selects on it — early
stopping is off (`early_stopping_patience: 0`) and the saved checkpoint is the
final-epoch model, not a best-on-validation one — so 95.48% is a clean held-out
number, and the curves are labelled "test" rather than "val" for that reason.

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

The checkpoint's metadata records `{epoch: 8, batch: 272}` — the Stop landed
about 70% of the way through epoch 9 (391 batches at 128). Those partial-epoch
weight updates are kept, but the epoch is not counted, so the resumed run
re-ran epoch 9 from batch 0. The schedule is continuous across a mid-epoch stop;
the data pass is not. That is why the resumed run is a faithful continuation but
not a bitwise match for an uninterrupted 20-epoch run.

The wiring order matters, though not for the reason it might look like — see the
note in the docs page. `Optimizer` feeds `LRScheduler` feeds
`CheckpointLoader.lr_scheduler`. That last input is optional, and leaving it
unconnected does **not** corrupt `base_lrs`: `initial_lr` travels inside the
optimizer's own state dict and `LRScheduler.__init__` reads it with `setdefault`,
so both wirings start from 0.1 and both match theoretical cosine to 1.4e-17
(measured). What leaving it unconnected does do is discard the checkpoint's saved
schedule position, which the loader then reconstructs by replaying steps — exact
for `CosineAnnealingLR`, lossy for a metric-driven `ReduceLROnPlateau`. Related
to issue #149.

## Running it

Open the example, then submit it from the Runs panel with `seed` 1337 and
`deterministic` enabled. The first run downloads CIFAR-10 (about 170 MB) into
`backend/data/`, or into `<project>/assets/data` when a project directory is
open. CIFAR-10 is Krizhevsky, *Learning Multiple Layers of Features from Tiny
Images* (2009); it is fetched at run time and not redistributed here.

The run belongs to the server, not the browser tab — close the tab and come
back, and the Runs panel re-attaches and replays what it missed.

## Notes for anyone editing this graph

- **Every root node needs its own trigger edge from `Start`.** Execution walks
  forward from the entry points along *data* edges, so a node with no incoming
  data edge is pruned without one. This graph has exactly four such nodes and
  wires all four: `aug-crop` (`RandomCrop`, head of the training augmentation
  chain), `ev-totensor` (`ToTensorTransform`, head of the evaluation chain),
  `model` (`SequentialModel`) and `loss` (`Loss`). The two `Dataset` nodes are
  **not** roots — `aug-norm` feeds `ds-train.train_transform` and `ev-norm`
  feeds `ds-test.eval_transform`, so the forward walk reaches them by itself.
  Leaving a trigger out produces a missing-input error naming a node much
  further downstream: drop `e-trigger-3` and `model`, `opt` and `sched` all
  vanish, and the failure is reported against `train` (issue #201).
- **`EvaluateModel.device` is set to `cuda` explicitly.** It defaults to `cpu`
  and has no `auto` option, so it does not follow the run's device (issue #204).
- **`num_workers` is 4 with `persistent_workers`.** The GPU does an epoch of
  compute in under 5 seconds on an RTX 4080; single-threaded PIL augmentation of
  50,000 images is several times that, so the data pipeline, not the GPU, sets
  the wall clock.
- The `TrainingLoop.epochs` value and `LRScheduler.T_max` must stay equal —
  cosine annealing is stepped once per epoch and reaches zero exactly at
  `T_max`. `TrainingLoop` warns when they disagree but still runs, because a
  truncated schedule is a legitimate choice; the CI test below is what holds
  this graph to the pair.
- **The `layers` string is in its lean form.** The layer editor's serializer
  writes a `position` on every node, a `params: {}` on every param-less node and
  explicit null handles on every edge; none of those are in the committed text.
  Opening the layer editor and saving without changing anything is lossless but
  purely additive, and will inflate `layers` from about 9,200 to about 16,600
  characters. Expect that diff once, and do not mistake it for a real change.

## Continuous integration

Two tests in `backend/tests/test_builtin_examples.py` guard this example.

`test_resnet18_cifar10_baseline_recipe_intact` reads `graph.json` **unmodified**
and asserts that the file still is the file the 95.48% came from: every
hyperparameter the tables above quote, `T_max == epochs`, and — the property the
whole result rests on — that `EvaluateModel` still reads the `test` split, from a
different `Dataset` node than the training dataloader, with no augmentation
routed onto it.

`test_resnet18_cifar10_baseline_short_epoch` then executes the graph for two
optimizer steps against a generated image folder rather than the real download.
It asserts the layer spec still builds an 11,173,962-parameter model, that the
training chain randomizes while the evaluation chain does not, and that the
whole pipeline still runs — none of which structural validation can check, since
the layer graph is an opaque JSON string as far as `validate_graph` is
concerned. Note that this second test *shrinks* the graph before running it
(both `Dataset` nodes are swapped for `ImageFolderDataset`, the loop is capped at
two steps on CPU), so the recipe assertions deliberately happen first, on the
pristine file.
