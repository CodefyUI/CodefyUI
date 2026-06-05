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
4. Open the toolbar **⚙ Settings** popover and switch **Record outputs** ON, then click **Run**. Every completed node's full output is captured in server memory, keyed by the run.
5. Click any node — the **Inspector** panel fetches that node's input and output, showing **shape, dtype, min/max/mean** and the actual values stacked top-to-bottom. Cells that changed are **heat-coloured**.
6. **Shift-select two nodes** and use **Compare Segment** (under ⚙ Settings → Inspection) to focus on just the head-input and tail-output; the canvas wraps them in a light-orange bubble with **HEAD** / **TAIL** badges.
7. Switch **Record outputs** OFF before a heavy training run if you don't want each epoch captured — previously captured runs stay fetchable until the server restarts.

:::note
Captured data lives in per-session RAM (LRU, last 20 runs). Segment markers are saved with the graph JSON.
:::

## Settings popover toggles

The toolbar **⚙ Settings** popover groups every per-tab teaching/training switch in one place — same idea as VS Code's Settings UI:

| Toggle | What it does |
|--------|---|
| **Record outputs** | Capture each completed node's full output for the Inspector. Off by default; turn off for heavy training runs. |
| **Verbose mode** | Backend records intermediate algorithmic steps (attention scores, softmax temperatures, etc.) alongside outputs — feeds the Inspector **Steps** tab. |
| **Compare Segment** | Wraps two shift-selected nodes in a HEAD/TAIL bubble so the Inspector shows only that subgraph's boundaries. |
| **Persist weights between runs** | Keep `Conv2d`/`Linear`/`Attention` weights across Run clicks (so the model actually learns). When off, every run reinitialises. |
| **Reset all weights now** | Drop every cached weight for this tab; the next Run starts fresh. |
| **Capture gradients** | Run forward + `.backward()` and store each layer's gradient for the Inspector **Backward** tab. |
| **Auto-synthesize loss** | When the graph has no `Loss`/`BackwardOnce` node, synthesize one so `.backward()` can run. |
| **Grid snap** | Snap dragged nodes to the canvas grid. |
| **Show node tooltips** | Reveal the description card when hovering nodes on the canvas. |
| **Node category mode** | `Basic` shows only essential categories in the sidebar; `All` shows every category. |

## Step traces (Verbose mode)

With **Verbose mode** on, instrumented nodes emit a `__steps__` trace that the Inspector renders one row at a time. Educational plugin nodes lean on this heavily — e.g. `Edu-ColumnStats` shows the population-std formula as `sum → divide → deviations² → variance → sqrt`. See **[Plugins](/advanced/plugins)**.

## Gradient capture (Backward tab)

With **Capture gradients** on, the engine runs a forward pass, calls `.backward()`, and stores each layer's gradient. Open a node's **Backward** tab in the Inspector to see gradient magnitudes per layer — useful for diagnosing vanishing/exploding gradients.
