---
sidebar_position: 8
title: Training Memory
description: Fit a bigger run on one card — mixed precision, gradient accumulation, picking a specific GPU, what happens on an out-of-memory error, and the server's own memory budgets.
---

# Training Memory

Most training runs do not stop because the maths was wrong. They stop because the card ran out of room. This page covers the four things CodefyUI gives you to push that ceiling up, and the one thing it deliberately will not do for you.

Everything here is **off by default**. A graph saved before any of it existed trains exactly as it did before.

## Mixed precision

`TrainingLoop` has a **precision** parameter under **Advanced**:

| Value | What it does | When to use it |
| --- | --- | --- |
| `fp32` | Nothing. Full 32-bit throughout. | The default. Always correct, always the most memory. |
| `bf16` | Runs the forward pass and the loss under `autocast(torch.bfloat16)`. No gradient scaler. | **The one to reach for.** Ampere and newer — RTX 30xx, 40xx, 50xx, A100, H100. |
| `fp16` | Same, in `float16`, plus a `GradScaler`. | Cards without bfloat16 — Volta and Turing, i.e. GTX 16xx and RTX 20xx. |

Activations, not weights, are what fills a card on a deep model, and autocast halves them. Parameters stay in float32 either way, so a model whose memory is mostly weights sees a smaller win than one whose memory is mostly feature maps.

**Why bf16 and not fp16 on a modern card.** bfloat16 keeps float32's exponent range and spends its bits on mantissa instead. Small gradients therefore do not underflow, there is nothing for a loss scaler to do, and there are no skipped steps. float16 has five exponent bits, so it needs a scaler that multiplies the loss before `backward()` and divides the factor back out before the step — and backs off, skipping that step entirely, whenever a gradient overflows.

**What each device can honour.** The choice is resolved against the device before the run starts, and a device that cannot honour it falls back to `fp32` with a warning rather than failing:

- CUDA without bfloat16 support asks for `bf16` and gets `fp32`.
- MPS gets `fp32` for anything but `fp32`. Apple's autocast coverage varies by torch build and is not something CodefyUI can verify on your machine.
- CPU honours all three. Neither 16-bit mode is *fast* on a CPU — this is about being able to run the lesson on the machine in front of you.

The node's config frame and its `metrics` output both report `precision` (what ran) and, when they differ, `precision_requested` (what you asked for). Validation runs under the same autocast as training, so the two loss curves stay comparable.

### Resuming an fp16 run

The loss scale is part of the training state. `CheckpointSaver` and `CheckpointLoader` carry it in a `grad_scaler_state` port, and `TrainingLoop` has one on each side:

```
TrainingLoop.grad_scaler_state  →  CheckpointSaver.grad_scaler_state
CheckpointLoader.grad_scaler_state  →  TrainingLoop.grad_scaler_state
```

Leave both unconnected for `fp32` and `bf16` — there is no scale to carry. Losing the state is survivable rather than fatal: a fresh scaler re-finds its level within a few hundred steps, and the steps in between are taken at the wrong scale.

## Gradient accumulation

**accumulate_steps** (also under **Advanced**) runs N batches, divides each one's loss by N, and steps the optimizer once.

The gradient that reaches the optimizer is then the gradient of one batch N times larger — exactly, not approximately, when the loss is a mean and every micro-batch is full. So:

> batch_size 8 with accumulate_steps 4 **is** batch_size 32, at a quarter of the activation memory.

That is a tested claim, not a description: the test suite computes the full-batch gradient by hand and requires the accumulated run to land on the same weights.

Three interactions worth knowing:

- **Gradient clipping** happens at step time, over the whole accumulated gradient. Clipping each micro-batch instead would let N clipped gradients sum to N times the threshold.
- **`max_steps` counts optimizer steps**, not batches, so it means the same amount of learning at any `accumulate_steps`. The `metrics` output reports both: `total_steps` (optimizer steps) and `total_batches` (forward passes).
- **The reported loss is undivided.** The `/N` is a detail of how the gradient is assembled and never reaches the chart, so the same run at accumulate_steps 1 and 4 draws the same curve.

An epoch whose batch count is not a multiple of `accumulate_steps` ends with a short window, and that window is still stepped — those batches already paid for their forward and backward passes. It is divided by N like every other, so a short window takes a proportionally smaller step, which is the honest treatment of a smaller sample.

An accumulation window never spans an epoch boundary, and a **Stop** discards the pending window rather than taking one more step on the way out.

## Picking a specific GPU

On a machine with more than one CUDA device, every device dropdown — the global selector in **Settings** and each node's own **device** parameter — lists the cards individually:

```
CPU
NVIDIA CUDA          (whichever card torch is currently pointed at)
NVIDIA CUDA #0       cuda:0
NVIDIA CUDA #1       cuda:1
```

A single-GPU machine shows only `NVIDIA CUDA`, because there `cuda` and `cuda:0` are the same piece of hardware and offering both would be a choice with nothing behind it.

Each card gets **its own run queue**, so a six-hour job on `cuda:0` never delays a run submitted to `cuda:1`. See [Run Queue](/usage/run-queue).

**An index that does not exist** — `cuda:3` on a two-card box, or a graph saved on a workstation and opened on a laptop — falls back to the *current CUDA device*, with a warning naming the count. Not to the CPU: you asked to train on a GPU, and answering that with a silent forty-minute CPU run is the worse surprise.

:::note Distributed training is out of scope
A run is single-process and single-device. There is no `DistributedDataParallel`, no multi-card data parallelism, and no multi-node anything — one run uses one card. Several runs can occupy several cards at once, one per card, through the per-device queues.

This is a deliberate boundary rather than a gap waiting to be filled. DDP needs process launching, rendezvous, per-rank logging and per-rank checkpointing, and every one of those touches the run service, the event stream and the artifact store. Half of it would be worse than none.
:::

## When the card runs out anyway

A CUDA out-of-memory error is reported as a **NodeOOMError**: which node, which device, what the allocator was holding at the time, and what to change. It reads roughly like this:

```
Node TrainingLoop (n7) ran out of memory on cuda:0.

What to try, cheapest first:
  - reduce batch_size on the DataLoader node
  - set TrainingLoop's precision to bf16 (roughly halves activation memory on Ampere and newer)
  - raise TrainingLoop's accumulate_steps and lower batch_size by the same factor, which keeps the effective batch identical
  - make the model smaller, or shorten the sequence / shrink the image

CUDA memory on cuda:0: 14.82 GiB held by live tensors, 15.44 GiB reserved by
the caching allocator, 15.61 GiB peak this process, 15.99 GiB on the card.

Original error: CUDA out of memory. Tried to allocate 2.00 GiB ...
```

Alongside the message, two things happen so the *next* run starts from a clean card: whatever that node had cached is dropped, and the caching allocator's free blocks are handed back.

**The run is not retried, and the batch size is not reduced for you.** Re-running the same allocation gets the same answer. Halving the batch behind your back would change the numbers your run produces without telling you, so the same graph would mean two different things depending on how much VRAM happened to be free. You get the message and you make the change.

## The server's own memory

Three in-memory stores hold tensors between runs, and all three are bounded by bytes as well as by count:

| Store | What it holds | Budget | Setting |
| --- | --- | --- | --- |
| Execution cache | Node outputs, for "edit one node, only that subtree re-runs" | 1 GB **per open editor connection** | `CODEFYUI_EXECUTION_CACHE_MAX_MB` |
| Run output store | Captured port values for the Teaching Inspector | 2 GB | `CODEFYUI_RUN_OUTPUT_STORE_MAX_MB` |
| Node state store | Persistent `nn.Module` weights per node | 1 GB | `CODEFYUI_NODE_STATE_STORE_MAX_MB` |

Set any of them to `0` to disable the byte budget and leave the older count limit in charge.

Counts alone were never a memory limit: 256 cached node outputs is either 40 MB of scalars or 200 GB of feature maps, and a count cannot tell the two apart. Eviction is least-recently-used, and sizes are measured over tensor storage — recursively through lists, tuples and dicts, counting a tensor and a view of it once rather than twice.

`GET /api/health` reports what each store currently holds against its budget:

```json
{
  "status": "ok",
  "caches": {
    "execution_cache": {"instances": 2, "entries": 41, "bytes": 918212608, "max_bytes_each": 1073741824},
    "run_output_store": {"runs": 6, "max_runs": 20, "bytes": 244318208, "max_bytes": 2147483648},
    "node_state_store": {"modules": 12, "max_modules": 200, "bytes": 51380224, "max_bytes": 1073741824}
  }
}
```

Two caveats on the numbers, since a memory report that quietly rounds is worse than none. Module sizes are measured when the module is built, so the gradients a module grows while training are not re-counted. And CPU and CUDA bytes are added together — the budget compares one number, and this is that number.
