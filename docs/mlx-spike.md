# Native MLX Spike (Phase 3 — inference-subset, non-blocking)

> Status: **spike / proof-of-concept**. Not a shipped execution backend.
> Apple acceleration in the graph engine is provided by **PyTorch MPS** (Phase 1 + 2).

## TL;DR

A small MLP's **forward inference** was ported from PyTorch to Apple's
[MLX](https://github.com/ml-explore/mlx) framework and produced numerically
identical results:

```
$ python scripts/mlx_spike.py
MLX default device: Device(gpu, 0)
torch MPS available: True

output shape       : torch (256, 10)  |  mlx (256, 10)
max abs difference : 1.937e-07
torch CPU forward  : 0.21 ms
mlx   GPU forward  : 0.68 ms

PASS — inference-subset parity within 1e-4 tolerance.
```

So: **feasible and correct** for the inference subset. MLX stays an *optional
future accelerator*, not a replacement for the MPS path.

## Why MLX is separate from the device selector

MLX is a **distinct array framework**, not a PyTorch backend. There is no
`tensor.to("mlx")` and no `torch.device("mlx")`. The graph engine is built on
`torch` tensors / `nn.Module`s, so MLX cannot be a value in the global device
selector (`cpu` / `cuda` / `mps`) — that selector drives `torch`. Apple GPU
execution for the real graph is therefore **PyTorch MPS** (already wired and
verified end-to-end). MLX is surfaced only via:

- `device_utils.mlx_available()` — detection (does not import torch).
- `scripts/mlx_spike.py` — this runnable PoC.

## What the spike does

1. Build a `Linear→ReLU→Linear→ReLU→Linear` MLP in torch; run inference on CPU.
2. Export the `state_dict` to numpy, then to `mlx.core` arrays.
3. Reimplement the forward with MLX ops (`mx.matmul`, `mx.maximum`), mirroring
   torch's `y = x @ Wᵀ + b` Linear convention.
4. `mx.eval()` (MLX is lazy), compare outputs, time each side.

## Scope & constraints (deliberate)

- **Inference only.** Training/autograd parity in MLX is explicitly out of
  scope — far larger, and unnecessary for the spike's question ("can we run a
  trained model's forward on MLX?").
- **float32.** MLX is float32-native on Apple silicon (same constraint that
  makes MPS reject float64 — see `device_utils.to_device`).
- Manual op-by-op port. A general `nn.Module → MLX` translator (covering Conv,
  attention, norms, …) is the next increment if this is ever productized.

## Conversion approach (for a future increment)

```
torch model (trained, on CPU/MPS)
        │  state_dict → numpy → mx.array
        ▼
MLX arrays + hand-written / generated forward
        │  mx.eval()
        ▼
outputs  ──(numpy)──▶ compare to torch reference
```

A productized version would generate the MLX forward from the same layer-graph
the `SequentialModel` / `GraphModelModule` already builds, reusing the node
metadata rather than hand-porting.

## Recommendation

- Keep **MPS** as the Apple default for all execution (training + inference).
- Treat MLX as an **optional inference accelerator** to revisit only if there's
  a measured win on Apple hardware for inference-heavy teaching demos.
- `mlx` is **not** a committed dependency. Install ad-hoc to run the spike:
  `uv pip install mlx` (Apple Silicon only). The main app never imports it.
