---
sidebar_position: 3.8
title: Data & Augmentation
description: Build a preprocessing pipeline out of nodes, augment your training data, load your own images, and send the metrics to TensorBoard.
---

# Data and Augmentation

Before CodefyUI 1.5, a dataset got exactly three preprocessing choices: a resize, a `ToTensor`, and a normalisation hardcoded to `mean=0.5, std=0.5`. That is enough to make a graph run and not enough to reproduce a single published vision result. This page covers what replaced it.

Nothing here changes an existing graph. The old **Transform** node still works exactly as it did, with the same three parameters.

## A transform chain

A transform is now a node, and transforms connect to each other. Each one takes the steps before it and hands on the steps so far, so the chain you draw on the canvas is the pipeline in the order it runs:

```
RandomCrop -> RandomHorizontalFlip -> ToTensorTransform -> NormalizeTransform
```

The wire between them carries a `TRANSFORM` port, drawn in amber. It only connects to other `TRANSFORM` ports, or to an `ANY` port, so a pipeline cannot be wired into a place that expects a dataset.

That chain produces exactly `transforms.Compose([RandomCrop(...), RandomHorizontalFlip(...), ToTensor(), Normalize(...)])` — the same object you would write by hand, which is also what an exported Python script builds.

### The order matters, and torchvision's rules still apply

- **Geometric and colour steps first**, while the sample is still a PIL image: `RandomCrop`, `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`, `RandAugment`, `ResizeTransform`.
- **`ToTensorTransform` in the middle.** It converts to a `C x H x W` float tensor in `[0, 1]`.
- **`NormalizeTransform` last.** It needs a tensor.

A chain in the wrong order fails inside the DataLoader with torchvision's own error, which names the step that could not accept what it was given.

### The nodes

| Node | What it does | Key parameters |
| --- | --- | --- |
| `ResizeTransform` | Resize to a square | `size`; `interpolation` (advanced) |
| `ToTensorTransform` | PIL image to `[0, 1]` tensor | — |
| `NormalizeTransform` | `(x - mean) / std` per channel | `preset`, `mean`, `std` |
| `RandomCrop` | Pad, then take a random window | `size`, `padding`; `padding_mode` (advanced) |
| `RandomHorizontalFlip` | Mirror left-to-right | `p` |
| `RandomRotation` | Rotate by a random angle | `degrees`; `expand`, `fill` (advanced) |
| `ColorJitter` | Shift brightness / contrast / saturation / hue | `brightness`, `contrast`, `saturation`, `hue` |
| `RandAugment` | A whole policy behind two numbers | `num_ops`, `magnitude`; `num_magnitude_bins` (advanced) |
| `ComposeTransform` | Join several chains, in port order | `steps` |

`ComposeTransform` is only needed when two chains were built separately and one pipeline has to run both — node-to-node chaining already composes.

### Normalisation presets

`NormalizeTransform` ships the statistics that matter, so you never have to look them up:

| Preset | mean | std | Use it for |
| --- | --- | --- | --- |
| `Half` | `(0.5,)` | `(0.5,)` | The default, and what CodefyUI used before presets existed. Maps `[0, 1]` to `[-1, 1]`. |
| `ImageNet` | `(0.485, 0.456, 0.406)` | `(0.229, 0.224, 0.225)` | **Any torchvision pretrained model.** Those weights were trained against these numbers; using anything else shifts the input distribution away from what they expect. |
| `CIFAR-10` | `(0.4914, 0.4822, 0.4465)` | `(0.2470, 0.2435, 0.2616)` | Reproducing a CIFAR-10 baseline. |
| `CIFAR-100` | `(0.5071, 0.4865, 0.4409)` | `(0.2673, 0.2564, 0.2762)` | Reproducing a CIFAR-100 baseline. |
| `Custom` | your own | your own | Anything else. `mean` and `std` appear only when this is selected. |

A single value broadcasts across every channel, so `Half` is correct for one-channel MNIST and three-channel CIFAR alike.

## Wiring a chain to a dataset

**Dataset** and **ImageFolderDataset** each take two transform inputs:

- `train_transform` — used when `split` is `train`. This is where augmentation belongs.
- `eval_transform` — used for every other split, and as the fallback for the training split when `train_transform` is unwired.

The fallback only runs in that one direction, deliberately. A test split never picks up the augmenting chain, because a randomly distorted test set measures something different every time you look at it. **ImageFolderDataset**'s `(none)` split is the one exception, because there is no split there to fall back from; see below.

For the datasets that have no transform inputs — **HuggingFaceDataset**, **KaggleDataset**, or one of your own — wire the chain into the **Transform** node's `transform` input instead. When that input is wired, the node's three parameters are ignored.

### The CIFAR-10 recipe

The standard starting point for CIFAR-10, in four nodes:

```
RandomCrop(size=32, padding=4)
  -> RandomHorizontalFlip(p=0.5)
  -> ToTensorTransform
  -> NormalizeTransform(preset="CIFAR-10")
  -> Dataset(name="CIFAR10", split="train").train_transform
```

and, for the evaluation split, the same chain without the two random steps.

## More datasets

**Dataset** now offers `MNIST`, `FashionMNIST`, `CIFAR10`, `CIFAR100`, `SVHN` and `STL10`. All six download on first use into the same `data_dir`; in a project directory that is `assets/data/`.

### Your own images

**ImageFolderDataset** reads the layout torchvision's `ImageFolder` expects:

```
my-dataset/
  train/
    cat/  img001.png  img002.png ...
    dog/  img101.png ...
  val/
    cat/  ...
    dog/  ...
```

- `path` — the directory holding the splits. A relative path resolves against the data folder that also holds `models/` and `images/`; an absolute path is used as given.
- `split` — which sub-directory to load. Choose `(none)` when the class folders sit directly under `path` with no split level. At `(none)` there is no split to tell the two transform inputs apart, so whichever one is wired is the one used, and `train_transform` wins if both are.

Labels come from the folder names sorted alphabetically, so `cat` is 0 and `dog` is 1 on every machine. The node also outputs a `classes` list in label order.

### Anything else

For a dataset that is not one folder per class — a CSV of paths, a custom archive format, a generated distribution — write a custom node (see [Custom Nodes](../advanced/custom-nodes.md)) or use the [PythonScript node](../advanced/python-script-node.md). Give whatever you return a public, writable `transform` attribute that `__getitem__` applies, and every transform chain on this page works with it.

## Reproducibility

Augmentation is random, so a reproducible run has to reproduce the augmentations too. Give the run a **seed** (in the run options) and it does: the same seed produces the same crops, flips and colour shifts, every time.

Four details are worth knowing:

- **The stream is isolated.** A chain's randomness depends on the run seed and the identity of the node that attached it, and on nothing else. Changing the model or the dropout rate does not change which crops you get. What the batch size and the worker count still decide, once `num_workers` is above zero, is which sample draws from where in that stream — each worker has its own. So those two settings change what lands on a given image, but re-running the same configuration reproduces it exactly.
- **It still varies.** Reproducible does not mean frozen: samples differ from each other, and epoch 2 differs from epoch 1. That is the whole point of augmentation, and it holds with `num_workers` set as well as without.
- **An unseeded run is unchanged.** Without a seed the pipeline keeps torch's own entropy and pays no overhead at all. The extra bookkeeping is only installed when a run asks for a seed *and* the chain actually contains a random step.
- **An exported script keeps the seed.** "Export as Python" bakes the canvas's seed into the generated file as `GRAPH_SEED`, and it is the default for the script's own `--seed`. So the exported graph reproduces what the canvas produced — same crops, same flips — without anyone remembering to pass a flag. Override it with `--seed 123`, or pass `--no-seed` for fresh entropy on every invocation. The determinism toggle travels the same way, as `--deterministic` / `--no-deterministic`.

## TensorBoard

**TrainingLoop** has a **tensorboard** parameter under **Advanced**. Turn it on and the run writes its metrics as TensorBoard event files as well as into CodefyUI's own charts — the same series, from the same call site, so the two cannot disagree.

The files land under the run's own folder next to `models/` and `images/`:

```
<data root>/runs/<run id>/tb/<node id>/
```

Each training node in a graph gets its own leaf directory, so a pretrain loop and a finetune loop draw as two separate runs in TensorBoard rather than one zig-zag line.

The path is registered as an artifact of the run, so you can copy it out of the **Runs** panel and open it:

```bash
tensorboard --logdir <the path from the Runs panel>
```

CodefyUI does **not** depend on TensorBoard to write these files — it encodes the event format itself, so the feature costs a CodefyUI install nothing. You only need TensorBoard installed to *look* at them:

```bash
pip install tensorboard
```

Two things it deliberately does not do. It writes nothing when the run has nowhere to record the artifact row (an exported script, the CLI contract runner), because a directory nothing references is litter no cleanup can find. And a non-finite value is dropped rather than written, because a NaN poisons the y-axis auto-scaling of every other series in the same chart — your run's own metric store still keeps it.

## Exporting metrics as CSV

Every run's metrics are downloadable as CSV, from two places in the **Runs** panel: the **CSV** button on each run row, and **Download CSV** beside the chart when a run is open. Both produce the same file, one row per point, with the series name, step and value.
