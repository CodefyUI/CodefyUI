---
name: train-and-evaluate
description: Train, evaluate, checkpoint and compare models in CodefyUI - the standard TrainingLoop graph, seeds and reproducibility, reading the Training tab and run metrics, saving and resuming checkpoints, plus language-model and PushWorld VLA training. Use when the user wants to train a model, measure it, resume a run, or explain a loss curve.
license: AGPL-3.0-only
metadata:
  codefyui-skill-version: "1"
---

# Train and evaluate a model

Check every port and param below against the node schemas before wiring; this skill names
them as they are today, and the catalog is the authority.

## The standard training graph

```
Start --trigger--> SequentialModel, Dataset(train), Dataset(test), Loss

SequentialModel.model --> Optimizer.model
SequentialModel.model --> TrainingLoop.model          (the SAME model output feeds both)
Dataset(train).dataset --> DataLoader.dataset --> TrainingLoop.dataloader
Optimizer.optimizer --> TrainingLoop.optimizer
Loss.loss_fn --> TrainingLoop.loss_fn

TrainingLoop.model  --> EvaluateModel.model
Dataset(test).dataset --> EvaluateModel.dataset
EvaluateModel.accuracy --> Print.value
TrainingLoop.losses --> Visualize.data                 (loss curve)
TrainingLoop.model  --> ModelSaver.model               (optional)
```

- **One model, two consumers.** `Optimizer` must be built on the same `MODEL` output that
  `TrainingLoop.model` receives, or it updates weights nobody trains.
- **Validation.** For a held-out curve, split the training set with `DatasetSplit`
  (`train`, `val`; `val_fraction`, `seed`), give `val` its own `DataLoader`
  (`shuffle` off), and wire it into `TrainingLoop.val_dataloader`. That adds `val_loss`
  (and `val_accuracy` with `CrossEntropyLoss` or `NLLLoss`) per epoch. Keep the test split
  for the final `EvaluateModel` only; do not tune against it.
- **Models.** `SequentialModel` holds its layers in its `layers` param (edited in the
  Layers editor). `CausalLMModel` and `VLAModel` are whole models with their own params.
- **Presets.** The `Training Pipeline` preset packages Dataset -> DataLoader -> Loss ->
  Optimizer -> TrainingLoop in one card; the shipped `CNN-MNIST` example uses it.

## TrainingLoop settings that matter

| Param | Meaning |
| --- | --- |
| `epochs` | An absolute target, also when resuming |
| `device` | `auto` picks the best device; the graph's device select can override |
| `early_stopping_patience`, `monitor` | Stop after N epochs without improvement; `model` then holds the best epoch's weights |
| `grad_clip_norm` | Clip the gradient norm (0 = off) |
| `precision`, `accumulate_steps` | Memory: `bf16`/`fp16` and gradient accumulation |
| `max_steps` | End after N optimizer steps whatever `epochs` says |
| `scheduler_step` | When a wired `LRScheduler` steps: `epoch` or `optimizer_step` |
| `batch_metrics`, `log_grad_norm`, `log_update_ratio` | Extra per-batch / per-step series |
| `checkpoint_every`, `checkpoint_every_steps` | Periodic checkpoints into `models/periodic/` |
| `deterministic` | Deterministic kernels for the whole run |

**Schedules.** Wire `Optimizer.optimizer -> LRScheduler.optimizer -> TrainingLoop.lr_scheduler`.
With the default `scheduler_step = epoch`, keep `LRScheduler.T_max` equal to `epochs`
(cosine reaches zero exactly at `T_max`; past it the rate climbs again). The warmup types
(`warmup_cosine`, `warmup_linear`, `constant_with_warmup`) count `warmup_steps` and
`total_steps` in the scheduler's unit, so use them with `scheduler_step = optimizer_step`.

## Seeds and reproducibility

- The run-level seed is **Settings -> Training Behavior -> Random seed** (and
  **Deterministic algorithms**), or `options.seed` / `options.deterministic` on
  `POST /api/runs`. One seed derives every node's randomness (initialisation, shuffling,
  augmentation), per node.
- Some nodes also have their own `seed` param (`DatasetSplit`, `CausalLMModel`,
  `VLAModel`, `PushWorldDemos`, `TextGenerate`, `SyntheticDataset`, ...). Changing one
  of those is how a graph varies one source of randomness on purpose.
- Same seed with `deterministic` on should reproduce a number exactly on the same
  machine; different seeds show the run-to-run spread, which is the noise any claimed
  improvement has to beat. Report which seeds you used.

## Reading a training run

- The **Training** tab follows `TrainingLoop` live: epoch, latest and best loss, the loss
  curve (`train_loss`, plus `val_loss` with a validation loader) and a per-epoch table.
- Every recorded value is stored with the run as a named series (`train_loss`, `lr`,
  `val_loss`, `val_accuracy`, `train_loss_batch`, `grad_norm`, `update_ratio`,
  `val_loss_step`, `eval_accuracy`, ...). Read them from the Runs panel or
  `GET /api/runs/{id}/metrics`; quote numbers exactly with their series name.
- What to look for: training loss falling while validation loss rises is overfitting;
  a flat loss from epoch 1 often means a learning rate far too low or a disconnected
  optimizer; a loss that jumps to `nan` usually means too high a rate (lower `lr`, add
  `grad_clip_norm`). See [reading curves](references/reading-curves.md).
- The **Execution Log** carries warnings the loop prints, such as a `T_max` that disagrees
  with `epochs`.

## Checkpoints

- **Save:** `TrainingLoop.model -> CheckpointSaver.model`, `TrainingLoop.optimizer ->
  CheckpointSaver.optimizer` (the optimizer that actually trained), `TrainingLoop.epoch ->
  CheckpointSaver.epoch`, and `TrainingLoop.losses -> CheckpointSaver.losses`. For weights
  only, use `ModelSaver` (`state_dict` or `full_model`; `pytorch` or `safetensors`).
- **Resume:** add `CheckpointLoader` (its `path`), feed it the model and optimizer to
  restore into, and route `CheckpointLoader.model/.optimizer/.lr_scheduler/.epoch` into
  `TrainingLoop.model/.optimizer/.lr_scheduler/.start_epoch`. Keep `epochs` at the final
  target. Wire `Optimizer -> LRScheduler -> CheckpointLoader.lr_scheduler` so the schedule
  resumes too.
- **Stop** is cooperative: the loop finishes its batch, writes a checkpoint registered as
  a run artifact, and returns what it has.

## Specialised training

- Language models (`CausalLMModel`, `LMTokenizedDataset`, `PerplexityEvaluate`):
  [references/lm-training.md](references/lm-training.md).
- Vision-language-action policies on PushWorld (`VLAModel`, `VLARollout`):
  [references/vla-pushworld.md](references/vla-pushworld.md).

## Before you run

Training is real work: it can download datasets (CIFAR-10 is about 170 MB), take
minutes to hours, and use the GPU. Say how long and how large before starting, run only
when the user asked, and never stop or restart a long run on your own.
