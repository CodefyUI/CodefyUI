---
sidebar_position: 4
title: Teaching Inspector
description: Record per-node outputs, inspect input→output tensor diffs, compare a subgraph segment, capture gradients, and view step traces.
---

# Teaching Inspector

CodefyUI can be used as an **interactive lesson** — students see the exact tensor that flows through every node. The Teaching Inspector captures node outputs during a run and renders them in the right-hand panel.

## Walkthrough

1. Drag a **`TensorInput`** node onto the canvas (Data category). Set `value_mode: explicit` and fill the inline grid with the numbers you want the pipeline to see.
2. Wire it through any chain of tensor-op nodes (e.g. `Reshape → Softmax → Print`).
3. **Add a `Start` node** and connect its trigger output to the first node you want executed — typically the `TensorInput`. Without this, the graph is a draft and **Run** is rejected (see [Your First Graph](./first-graph)).
4. **Record node outputs** is on by default; check it is still on under **Settings → Recording & Inspection**, then click **Run**. Every completed node's full output is captured in server memory, keyed by the run.
5. Click any node — the **Inspector** panel fetches that node's input and output, showing **shape, dtype, min/max/mean** and the actual values stacked top-to-bottom. Cells that changed are **heat-coloured**.
6. **Shift-select two nodes** and click **Create segment** (**Settings → Recording & Inspection → Compare segment**) to focus on just the head-input and tail-output; the canvas wraps them in a light-orange bubble with **HEAD** / **TAIL** badges.
7. Switch **Record node outputs** off before a heavy training run if you don't want each epoch captured — runs already captured stay fetchable until they are evicted or the server restarts.

:::note
Captured outputs live in one server-wide store shared by every tab: the newest 20 runs and 2 GiB (`CODEFYUI_RUN_OUTPUT_STORE_MAX_MB`, default `2048`). Whole runs are evicted oldest-first, and deleting a run from the Runs panel drops its captures — see [Training Memory](/advanced/training-memory#the-servers-own-memory). Segment markers are saved with the graph JSON.

Creating or clearing a marker is an undoable step: **Ctrl+Z** brings back a marker you removed by mistake — or one that a **Collapse to block** / **Delete node** swallowed — together with the focus it had.
:::

## The Inspector panel

The right-hand column appears when a node is selected, a segment is active, or a plugin panel is docked there. For a selected node it has three tabs:

| Tab | Shows |
| --- | --- |
| **Forward** | The node's inputs stacked above its outputs, each port with a type-coloured dot; tensors render as value grids with changed cells heat-coloured. |
| **Steps** | The `__steps__` trace an instrumented node emitted with **Verbose internals** on — see below. |
| **Backward** | The weight and output gradients captured with **Capture gradients** on — see below. |

A **segment** replaces the node view with a SEGMENT header, **Segment inputs (N)** — every edge entering the head-to-tail set — and **Segment outputs (N)**. **Create segment** needs exactly two selected nodes. Several segments can coexist on the canvas: the **x** on a bubble removes only that one, and **Clear active** clears the highlighted one the Inspector is showing. The panel collapses to a thin strip with the **›** button.

## Node details

Double-click a node, press **Enter** with it selected, or right-click → **Open details** to open the node-details modal: the parameter form on the left and, on the right, everything the Inspector knows about the node.

| Tab | Shows |
| --- | --- |
| **Code** | Script nodes only, and the tab they open on: the script editor and its input/output port counts. |
| **Subgraph** | Subgraph instances only: the block's boundary ports and an **Enter subgraph** button. |
| **Inputs** / **Outputs** | The captured values on each port, as the Inspector's Forward tab shows them. |
| **Steps** / **Backward** | The step trace and the captured gradients; available once the tab has a run to read from. |
| **Stats** | Summary statistics for every port, computed on the server — see below. |
| **Docs** | The node's description, its parameters with defaults, ranges and options, and its ports. |

**Left** / **Right** step to the previous / next node on the canvas without closing, **Esc** closes, and clicking the node name turns it into a rename field (Enter applies, Esc cancels). Click an edge after a run for a summary of what flowed through it — type, shape, dtype, min/max/mean — and its **View stats** link opens Node details with that port focused in **Stats**.

### Stats tab

`GET /api/execution/outputs/{run}/{node}/{port}/stats` answers with a fixed-size summary of a captured port rather than its values. Count, min, max, NaN and Inf counts, the zero fraction and — for integer label tensors — the class balance are always exact. Mean, std, the quantiles and the 64-bin histogram are exact up to 4 million elements and computed from a seeded 1-million-element sample above that, marked `"sampled": true` in the response (`CODEFYUI_STATS_SAMPLE_THRESHOLD`, `CODEFYUI_STATS_SAMPLE_SIZE`). Computed summaries are cached up to `CODEFYUI_STATS_CACHE_MAX_BYTES` (8 MB).

## Settings popover toggles

The toolbar **Settings** popover groups every per-tab teaching/training switch in one place, by section:

| Section | Setting | What it does |
|---|---|---|
| Execution | **Compute device** | The device the run uses; nodes set to `auto` follow it. CPU by default. |
| LLM Providers | **ChatGPT Codex account** | **Sign in** / **Sign out** / **Refresh** for the Codex provider — see [Graph Copilot](/advanced/graph-copilot). |
| Optional packs | **Package Center** | Opens the Package Center; the row counts installed packs. |
| Plugins | **Plugin Center** | Opens the [Plugin Center](/advanced/plugins#plugin-center). |
| Recording & Inspection | **Record node outputs** | Capture each completed node's full output for the Inspector. On by default; turn it off before a heavy training run. |
| | **Verbose internals** | Instrumented nodes record their intermediate steps (attention scores, softmax temperatures, ...) — feeds the **Steps** tab. With this on, nothing is served from cache; every node re-executes. |
| | **Compare segment** | **Create segment** wraps two selected nodes in a HEAD/TAIL bubble; **Clear active** removes the highlighted one. |
| Training Behavior | **Persist weights between runs** | Keep `Conv2d`/`Linear`/`Attention` weights across Run clicks so the model actually learns. On by default; when off, every run reinitialises. |
| | **Reset all weights now** | Drop every cached weight for this tab; the next Run starts fresh. |
| | **Capture gradients** | Run forward + `.backward()` and store each layer's gradient for the **Backward** tab. With this on, nothing is served from cache; every node re-executes. |
| | **Auto-synthesize loss** | When the graph has no `Loss`/`BackwardOnce` node, synthesize one so `.backward()` can run. |
| | **Random seed** | Seed every node from one number; blank means unseeded. A seeded run executes one node at a time — see [Reproducible runs](./running-graphs#reproducible-runs-seed). |
| | **Deterministic algorithms** | Ask PyTorch for deterministic kernels (`warn_only`). Sent with the run alongside the seed and the device. |
| Editor | **Grid snap** | Snap dragged nodes to the canvas grid. |
| | **Show node tooltips** | Reveal the description card when hovering nodes on the canvas. |
| | **Node category mode** | `Basic` shows only essential categories in the sidebar; `All` shows every category. |
| | **Connection style** | **Circuit** (default) draws value connections as circuit-board traces, **Curve** as smooth curves. |
| This Server | — | The version, the node and preset counts, and each in-memory store's usage against its budget, with a **Refresh** button. |

## Step traces (Verbose internals)

With **Verbose internals** on, instrumented nodes emit a `__steps__` trace that the Inspector renders one row at a time. Educational plugin nodes lean on this heavily — e.g. `Edu-ColumnStats` shows the population-std formula as `sum → divide → deviations² → variance → sqrt`. See **[Plugins](/advanced/plugins)**.

## Gradient capture (Backward tab)

With **Capture gradients** on, the engine runs a forward pass, calls `.backward()`, and stores each layer's gradient. Open a node's **Backward** tab in the Inspector to see gradient magnitudes per layer — useful for diagnosing vanishing/exploding gradients.

## Full-size viewers

The five attention cards — the attention heatmap and mask, and the three Edu attention nodes — have a **View full** button, and the embedding scatter card has **Open detailed view**; each opens the plot in a full-size heatmap or scatter viewer. It survives scrolling, zooming and layout changes on the canvas, and closes with **Esc**.
