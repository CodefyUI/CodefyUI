---
sidebar_position: 7
title: CLI Graph Runner
description: Execute a saved graph.json directly from the command line with run_graph.py — no server required.
---

# CLI Graph Runner

You can execute any graph directly from the command line without starting the server. This is handy for batch runs, CI, or reproducing a pipeline headlessly.

If you want to call a *saved* graph on a *running* server instead — declared inputs in, declared outputs out, over HTTP — see **[Graph as a Function](./graph-as-a-function)**. To queue a graph file on a running server, so the run outlives the terminal and appears in the **Runs** panel, use `cdui run`; see **[Run Queue](./run-queue#cdui-run)**.

`run_graph.py` imports the backend, so it runs in the backend's virtual environment (`backend/.venv`, which `cdui install` creates):

```bash
cd backend
source .venv/bin/activate    # Windows: .venv\Scripts\activate
python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
```

The runner discovers all nodes via the registry, validates the DAG, executes it topologically, and prints per-node output summaries.

## Options

| Flag | Effect |
|------|--------|
| `--validate-only` | Validate the graph (DAG, types, ports, Start node) without executing it. |
| `--verbose`, `-v` | `DEBUG`-level logging, plus the full traceback when a node fails at runtime. There is no CLI switch for the Inspector's step traces. |
| `--device` | `cpu` / `cuda` / `cuda:N` / `mps` / `auto`. Omitted: the graph file's [`settings.device`](/advanced/device-backends#the-graph-settings-object), else `cpu`. `auto` takes the best accelerator present. |
| `--seed N` | Seed every node from `N` so the run is reproducible. A seeded run executes one node at a time — see **[Reproducible runs](./running-graphs#reproducible-runs-seed)**. |
| `--deterministic` | Ask PyTorch for deterministic kernels (`warn_only`, so an op with no deterministic implementation warns rather than failing the run). |

```bash
# Validate an architecture without running it
python run_graph.py ../examples/Model_Architecture/ResNet-SkipConnection-CNN/graph.json --validate-only
```

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | The graph ran, or it is valid with `--validate-only` |
| 1 | The file is missing, the graph fails validation, or a node fails at runtime (`-v` adds the full traceback) |
| 2 | Bad command line (argument parsing) |

## Plugin nodes

The runner loads the plugin packs listed in the plugin lockfile under `CODEFYUI_USER_DATA_DIR`, or, when that variable is unset, under the platform data directory (`%LOCALAPPDATA%\codefyui`, `~/.local/share/codefyui` or `~/Library/Application Support/codefyui`). The `cdui` commands, `cdui plugin install` and servers started with `cdui start` included, keep their lockfile in `<install dir>/.codefyui_dev/` instead. To run a graph that uses nodes from those packs, set `CODEFYUI_USER_DATA_DIR=<install dir>/.codefyui_dev` first; otherwise the runner does not know those node types and validation fails.

## Relative file paths

A relative `path` in a reader node is resolved by the process that runs the graph. For the runner, the working directory is the one you start it in (`backend/` in the commands above).

- **ImageReader** uses an absolute path as written. A relative path is looked up first in the image upload store (`backend/data/images` by default), then relative to the working directory.
- **CSVReader** looks a bare file name up first in the data-file upload store (`backend/data/files` by default). Otherwise a relative path resolves against the working directory; with `CODEFYUI_PROJECT_DIR` set, it resolves inside that project directory instead and may not leave it (the bundled `data/samples/iris.csv` is the exception).

## Where graphs come from

Any graph exported from the UI (**[Tabs & Persistence → Import / export](./tabs-persistence)**) is a plain JSON file in the same format, so you can build a pipeline visually and then run it from the CLI. The bundled examples under `examples/` are ready to run — see the **[Examples Gallery](./examples-gallery)**.
