---
sidebar_position: 3
title: GPU & Device Setup
description: Choose the right PyTorch build for NVIDIA CUDA, Apple Silicon (MPS), or AMD ROCm, and verify GPU detection.
---

# GPU & Device Setup

The default PyTorch install works on every platform (CPU, and Apple Silicon via MPS). Read on only if you need a specific CUDA version, AMD ROCm/DirectML, or want to verify GPU detection.

CodefyUI reads the available devices from the backend at runtime, so whatever PyTorch can see shows up in each node's **device** dropdown. The global device can be set once and applies to all tensor-source nodes.

## NVIDIA CUDA (specific version)

First check your installed CUDA version:

```bash
nvidia-smi
```

Read the `CUDA Version:` field in the top-right, and then install the matching wheel with `cdui install --gpu`. This command replaces the PyTorch build in the venv. Later `cdui update` commands read the variant from the installed wheel and preserve it:

```bash
# CUDA 12.8 — required for RTX 50 series (Blackwell, sm_120). Also works on RTX 30/40.
cdui install --gpu cu128

# CUDA 12.6 — RTX 30 / 40 series, a widely compatible default for modern drivers
cdui install --gpu cu126

# CUDA 11.8 — GTX 10 / RTX 20 series, or older drivers
cdui install --gpu cu118
```

The one-line installer uses `--gpu auto`. It maps the driver version reported by `nvidia-smi` to a wheel: 560 or newer selects `cu128`, 555 or newer selects `cu126`, 545 or newer selects `cu124`, 530 or newer selects `cu121`, and 520 or newer selects `cu118`. Older drivers select the CPU build. If the server was started with `cdui start`, the **GPU PyTorch** card in the Package Center performs the same installation and restarts the server. See [Installs that restart the server](/usage/optional-packs#installs-that-restart-the-server).

To install manually, activate `backend/.venv` and run the same `uv pip` command as the installer:

```bash
cd backend
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS / Linux
uv pip install --reinstall-package torch --reinstall-package torchvision torch torchvision --index-url https://download.pytorch.org/whl/cu128   # or cu126 / cu118
```

:::warning RTX 50 series (Blackwell)
RTX 5090 / 5080 / 5070 **require** `cu128` — older wheels lack the `sm_120` kernels and fail at runtime with `no kernel image is available for execution`.
:::

Verify CUDA is working:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## Apple Silicon (MPS)

The default install already ships the Metal Performance Shaders backend on M1/M2/M3/M4 Macs. Note that *shipping* it is not the same as *using* it: a run stays on the CPU until you pick `mps` in Settings — see [Device Backends](/advanced/device-backends). Verify the backend is present:

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

:::note float64 on MPS
MPS is float32-native and rejects float64 tensors. CodefyUI handles this in `device_utils.to_device`, but custom nodes should create float32 tensors on Apple GPUs. When MPS has no kernel for an operation, PyTorch runs it on the CPU instead of failing the run. [Device Backends](/advanced/device-backends) describes how to change this fallback and also documents the experimental native-MLX inference spike.
:::

## AMD GPU

AMD support depends heavily on your OS.

### Linux + AMD (ROCm, officially supported)

```bash
cdui install --gpu rocm6.2      # or rocm6.1
```

When `rocm-smi` is on `PATH`, `--gpu auto` selects `rocm6.2`. For a manual installation, activate `backend/.venv` and run `uv pip install --reinstall-package torch --reinstall-package torchvision torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2`.

Verify:

```bash
python -c "import torch; print('CUDA (ROCm):', torch.cuda.is_available())"
```

On ROCm, `torch.cuda.is_available()` returns `True` because ROCm presents itself as a CUDA-compatible backend.

### Windows + AMD (limited)

PyTorch does **not** ship an official Windows ROCm build. Your options:

- **(a) DirectML** — uses the AMD GPU but with reduced performance and requires code changes (the built-in nodes default to `cuda`/`cpu`):

  ```bash
  uv pip install torch-directml
  ```

- **(b) CPU mode** — the default install already works. Recommended for learning/prototyping on Windows with AMD.

## Troubleshooting

### Switching from CPU to CUDA (or vice versa)

```bash
cdui install --gpu cu128     # back again: cdui install --gpu cpu
```

### `uv pip install -e ".[ml]"` installs the wrong PyTorch version

The `[ml]` optional group in `pyproject.toml` does **not** specify an index URL, so uv installs whatever PyPI has as default — usually the CPU build on Windows, or a version that may not match your CUDA runtime. Always use the explicit `--index-url` command from this page.

### `torch.cuda.is_available()` returns False with an NVIDIA GPU

1. Run `nvidia-smi` to confirm the driver version.
2. Make sure you installed the matching CUDA PyTorch wheel (e.g. don't install `cu128` on a driver that only supports up to CUDA 11.8).
3. RTX 50 series + `no kernel image is available for execution` → you're on an older wheel; reinstall with `cu128`.
4. Update your NVIDIA driver if needed.

### The device dropdown in the UI doesn't show CUDA

The frontend reads available devices from the backend. If your GPU isn't listed:

1. Confirm PyTorch sees it: `python -c "import torch; print(torch.cuda.is_available())"`
2. Click the toolbar **Reload Nodes** button.
3. Reload the page.

### Verify device detection from the API

```bash
curl -s http://127.0.0.1:8000/api/nodes/TrainingLoop | python -c "import sys,json; d=json.load(sys.stdin); print([p['options'] for p in d['params'] if p['name']=='device'][0])"
```

This prints the available devices. An NVIDIA system reports `['auto', 'cpu', 'cuda']`, plus `cuda:0`, `cuda:1`, and additional indexed devices when multiple cards are present. When PyTorch detects no accelerator, it reports `['auto', 'cpu']`. `auto` is always available and follows the global device setting.
