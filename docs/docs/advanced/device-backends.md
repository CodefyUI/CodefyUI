---
sidebar_position: 4
title: Device Backends
description: How CodefyUI selects and falls back across CPU, CUDA, MPS, and ROCm — plus the experimental native-MLX inference spike.
---

# Device Backends

CodefyUI runs on PyTorch, so it inherits PyTorch's device backends: **CPU**, **NVIDIA CUDA**, **Apple Silicon (MPS)**, and **AMD ROCm** (Linux). For installing the right wheel, see **[GPU & Device Setup](/getting-started/gpu-device)**; this page explains how device selection behaves at runtime.

## Global device selection

**CPU is the default, and nothing switches away from it on your behalf.** A run uses the CPU unless you pick an accelerator in Settings, where the dropdown lists every device PyTorch can actually see (via `device_utils.get_available_devices()`). A requested device is checked against what's available and **falls back to CPU with a warning** if it isn't present. Set it once rather than per node.

### Device alignment is guaranteed by the engine

You do not have to reason about where a tensor happens to live. Before a node runs, `graph_engine.invoke_node` moves every tensor in its inputs to the device that node runs on — its own `device` parameter when it declares one, otherwise the run's. Because every path into a node goes through that one function, the guarantee covers builtin nodes, plugin nodes and your own [custom nodes](./custom-nodes) alike.

This matters because a device mismatch cannot happen on a CPU-only machine, so it is invisible during most development and shows up only on someone else's GPU box. Two shipped graphs died that way — `Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same` — before alignment moved into the engine.

What alignment deliberately does **not** touch:

- **Modules.** `nn.Module.to()` is in-place, so relocating a model handed from one node to another would flip weights out from under the node that owns it. A node that wants a model on its own device says so with an explicit `to_device`.
- **Datasets, DataLoaders, environments and other non-tensor values.** They pass through untouched, so a dataset stays lazy and `TrainingLoop` keeps streaming batches to the GPU one at a time instead of resident VRAM.

## Addressing one card out of several

On a machine with more than one CUDA device, every dropdown also lists the cards individually — `cuda:0`, `cuda:1`, and so on — alongside the bare `cuda`, which means "whichever card torch is currently pointed at". A single-GPU machine shows only `cuda`, because there the two name the same hardware.

An index that does not exist on this machine falls back to the **current CUDA device**, not to the CPU: a graph pinned to `cuda:2` on a workstation should still train on the GPU when it is opened on a laptop. Each card also gets its own run queue. See **[Training Memory](./training-memory)** for the full picture, including what is deliberately out of scope (distributed training).

## The float64 + MPS constraint

MPS is **float32-native** and rejects float64 tensors. CodefyUI normalizes this in `device_utils.to_device`, but if you write a [custom node](./custom-nodes) that creates tensors directly, keep them in float32 on Apple GPUs to avoid runtime errors.

## ROCm presents as CUDA

On AMD + Linux with a ROCm build of PyTorch, `torch.cuda.is_available()` returns `True` because ROCm exposes a CUDA-compatible interface. The device shows up as `cuda` in the dropdown; that's expected.

## Experimental: native MLX (spike)

There is a **proof-of-concept** that ports a small MLP's *forward inference* from PyTorch to Apple's [MLX](https://github.com/ml-explore/mlx) framework, producing numerically identical results (max abs difference ~1.9e-7). Key points:

- **Apple acceleration in the real graph engine is PyTorch MPS**, which is wired up and verified end-to-end. MLX is **not** a shipped execution backend.
- MLX is a *distinct array framework*, not a PyTorch backend — there is no `torch.device("mlx")` — so it can't be a value in the global device selector (which drives `torch`).
- The spike is **inference-only** and **float32**, runnable ad-hoc:

  ```bash
  uv pip install mlx        # Apple Silicon only
  python scripts/mlx_spike.py
  ```

- `mlx` is **not** a committed dependency; the main app never imports it. Surface it only via `device_utils.mlx_available()` (detection) and the spike script.

**Recommendation:** keep **MPS** as the Apple default for all execution (training + inference); treat MLX as an optional inference accelerator to revisit only if there's a measured win for inference-heavy teaching demos.
