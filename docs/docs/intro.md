---
sidebar_position: 1
slug: /
title: Introduction
description: A visual, node-based deep learning pipeline builder. Design CNN, RNN, Transformer, and RL architectures in the browser and run them in real time.
---

# CodefyUI

**A visual, node-based deep learning pipeline builder.** Design CNN, RNN, Transformer, and RL architectures by dragging nodes onto a canvas, connecting them into a DAG, and executing the pipeline — all from the browser.

![CodefyUI screenshot](/img/ui-screenshot.png)

## What you can do

- **Build models visually** — drag-and-drop nodes, connect ports with type-safe edges, get real-time validation. **94 built-in nodes** across 15 categories (CNN, RNN, Transformer, RL, Data, Training, LLM, Diffusion, Classical ML, and more).
- **Watch the tensors flow** — the **Teaching Inspector** records every node's output so you can inspect input→output diffs cell-by-cell, capture gradients, and wrap a subgraph to compare just its head input and tail output.
- **Run in real time** — a WebSocket stream reports per-node progress, live training-loss charts, and `Print` output as the graph executes.
- **Extend it** — save subgraphs as reusable **presets**, drop in **custom nodes** (`.py` files), or install **plugin packs** of educational nodes.
- **Use any backend** — CPU, NVIDIA CUDA, Apple Silicon (MPS), or AMD ROCm, selected at install time and per run.

## Quick start

Install only what's needed to run the app (`git`, `uv`, and Python) — **no Node.js required for end users**:

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/treeleaves30760/CodefyUI/main/install.sh | bash
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/treeleaves30760/CodefyUI/main/install.ps1 | iex"
```

Then open a new terminal and run:

```bash
cdui start
```

Open [http://localhost:8000](http://localhost:8000) — a single uvicorn process serves both the API and the prebuilt React app.

→ Full instructions in **[Installation](/getting-started/installation)**.

## Where to go next

| If you want to… | Start here |
|-----------------|------------|
| Install and launch the app | [Getting Started → Installation](/getting-started/installation) |
| Pick the right GPU / CUDA / MPS build | [GPU & Device Setup](/getting-started/gpu-device) |
| Build and run your first graph | [Usage → Your First Graph](/usage/first-graph) |
| Inspect tensors and gradients while learning | [Teaching Inspector](/usage/teaching-inspector) |
| Browse every built-in node | [Node Reference](/usage/node-reference) |
| Write a custom node or plugin | [Advanced → Custom Nodes](/advanced/custom-nodes) · [Plugins](/advanced/plugins) |
| Understand how execution works | [Architecture](/advanced/architecture) |

## Architecture at a glance

```
frontend/   React 19 · TypeScript · React Flow 12 · Zustand 5 · Vite 6
backend/    Python 3.10+ · FastAPI · PyTorch
```

CodefyUI is **backend-authoritative**: `GET /api/nodes` returns every node definition, and a single React component renders all node types from those definitions. Add a node on the backend and it appears in the UI automatically — see [Architecture](/advanced/architecture) for the full picture.

## License

CodefyUI uses a dual-path licensing model:

- **Open source** — [AGPL-3.0-only](https://github.com/treeleaves30760/CodefyUI/blob/main/LICENSE) for individuals, small teams, education, research, and community use.
- **Commercial** — for proprietary, closed-source, SaaS, OEM, or enterprise use that needs terms outside AGPL-3.0, [contact the maintainers](https://github.com/treeleaves30760/CodefyUI/issues).
