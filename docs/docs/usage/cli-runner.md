---
sidebar_position: 7
title: CLI Graph Runner
description: Execute a saved graph.json directly from the command line with run_graph.py — no server required.
---

# CLI Graph Runner

You can execute any graph directly from the command line without starting the server. This is handy for batch runs, CI, or reproducing a pipeline headlessly.

```bash
cd backend
python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
```

The runner discovers all nodes via the registry, validates the DAG, executes it topologically, and prints per-node output summaries.

## Options

| Flag | Effect |
|------|--------|
| `--validate-only` | Validate the graph (DAG, types, ports, Start node) without executing it. |
| `--verbose` | Emit intermediate step traces, the same data the Inspector's **Steps** tab shows. |

```bash
# Validate an architecture without running it
python run_graph.py ../examples/Model_Architecture/ResNet-SkipConnection-CNN/graph.json --validate-only
```

## Where graphs come from

Any graph exported from the UI (**[Tabs & Persistence → Import / export](./tabs-persistence)**) is a plain JSON file in the same format, so you can build a pipeline visually and then run it from the CLI. The bundled examples under `examples/` are ready to run — see the **[Examples Gallery](./examples-gallery)**.
