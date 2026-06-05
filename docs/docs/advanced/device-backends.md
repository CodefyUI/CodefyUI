---
sidebar_position: 4
title: Device Backends
description: How CodefyUI selects and falls back across CPU, CUDA, MPS, and ROCm — plus the experimental native-MLX inference spike.
---

# Device Backends

CodefyUI runs on PyTorch, so it inherits PyTorch's device backends: **CPU**, **NVIDIA CUDA**, **Apple Silicon (MPS)**, and **AMD ROCm** (Linux). For installing the right wheel, see **[GPU & Device Setup](/getting-started/gpu-device)**; this page explains how device selection behaves at runtime.

## Global device selection

A single global **device** setting drives all tensor-source nodes, so you set it once rather than per node. The backend exposes the devices PyTorch can actually see (via `device_utils.get_available_devices()`), and the UI populates each device dropdown from that list. A requested device is checked against what's available and **falls back to CPU with a warning** if it isn't present.

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
