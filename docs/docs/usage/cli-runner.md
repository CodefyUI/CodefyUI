---
sidebar_position: 7
title: CLI Graph Runner
description: Execute a saved graph.json directly from the command line with run_graph.py — no server required.
---

# CLI Graph Runner

You can execute any graph directly from the command line without starting the server. This is handy for batch runs, CI, or reproducing a pipeline headlessly.

If you want to call a *saved* graph on a *running* server instead — declared inputs in, declared outputs out, over HTTP — see **[Graph as a Function](./graph-as-a-function)**.

```bash
cd backend
python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
```

The runner discovers all nodes via the registry, validates the DAG, executes it topologically, and prints per-node output summaries.

## Options

| Flag | Effect |
|------|--------|
| `--validate-only` | Validate the graph (DAG, types, ports, Start node) without executing it. |
| `--verbose`, `-v` | `DEBUG`-level logging, plus the full traceback when a node fails at runtime. There is no CLI switch for the Inspector's step traces. |
| `--device` | Global compute device: `cpu` / `cuda` / `mps`. |
| `--seed N` | Seed every node from `N` so the run is reproducible. A seeded run executes one node at a time — see **[Reproducible runs](./running-graphs#reproducible-runs-seed)**. |
| `--deterministic` | Ask PyTorch for deterministic kernels (`warn_only`, so an op with no deterministic implementation warns rather than failing the run). |

```bash
# Validate an architecture without running it
python run_graph.py ../examples/Model_Architecture/ResNet-SkipConnection-CNN/graph.json --validate-only
```

## Where graphs come from

Any graph exported from the UI (**[Tabs & Persistence → Import / export](./tabs-persistence)**) is a plain JSON file in the same format, so you can build a pipeline visually and then run it from the CLI. The bundled examples under `examples/` are ready to run — see the **[Examples Gallery](./examples-gallery)**.
