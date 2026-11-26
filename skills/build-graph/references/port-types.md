# Port types and what connects to what

The rule is `backend/app/core/type_system.py::is_compatible`. An output of the type in the
left column may feed an input of any type in the right column.

| Output type | May feed |
| --- | --- |
| `TENSOR` | `TENSOR`, `ANY` |
| `MODEL` | `MODEL`, `ANY` |
| `DATASET` | `DATASET`, `ANY` |
| `DATALOADER` | `DATALOADER`, `ANY` |
| `OPTIMIZER` | `OPTIMIZER`, `ANY` |
| `LOSS_FN` | `LOSS_FN`, `ANY` |
| `SCALAR` | `SCALAR`, `ANY` |
| `STRING` | `STRING`, `ANY` |
| `IMAGE` | `IMAGE`, `TENSOR`, `ANY` |
| `LIST` | `LIST`, `ANY` |
| `TRANSFORM` | `TRANSFORM`, `ANY` |
| `ANY` | every type except `TRIGGER` |
| `TRIGGER` | `TRIGGER` only (the `Start` node's trigger edge) |

What the common types carry:

- `TENSOR` - a `torch.Tensor`. Most layer nodes (`Conv2d`, `Linear`, `Flatten`,
  `Reshape`, `Activation`) take `tensor` in and give `tensor` out.
- `MODEL` - an `nn.Module`. Produced by `SequentialModel`, `CausalLMModel`, `VLAModel`,
  `ModelLoader`, and passed through `TrainingLoop.model`.
- `DATASET` - an indexable dataset (`Dataset`, `DatasetSplit`, `TensorDataset`,
  `LMTokenizedDataset`, `PushWorldDemos`).
- `DATALOADER` - batches of a dataset, from `DataLoader`.
- `OPTIMIZER`, `LOSS_FN` - from `Optimizer` and `Loss` (or `LMCrossEntropyLoss`,
  or the `loss_fn` output of `VLAModel`).
- `SCALAR` - a number: an accuracy, a loss, an epoch count.
- `LIST` - a Python list; `Map` iterates one, `Reduce` folds one.
- `TRANSFORM` - a torchvision `Compose`, chained by the transform nodes and accepted by
  `Dataset.train_transform` / `Dataset.eval_transform`.

Layer nodes are not models. A stack of `Conv2d`/`Linear` nodes on the canvas computes on
tensors once; to train layers, build them inside `SequentialModel` (its `layers` param is
a small graph of its own, edited in the Layers editor) and train that `MODEL`.
