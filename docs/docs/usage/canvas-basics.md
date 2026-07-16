---
sidebar_position: 1
title: Canvas Basics
description: Orient yourself in the CodefyUI canvas — the node palette, type-safe ports and edges, config panel, and results panel.
---

# Canvas Basics

CodefyUI is a single-page app: a **canvas** in the middle, a **node palette** on the left, a **config panel** on the right, and a **results panel** at the bottom. This page orients you; the following pages go deep on building, running, and inspecting graphs.

## The canvas

- **Add a node** — drag it from the palette, or **double-click the canvas** to open the quick search panel and type a node or preset name.
- **Connect nodes** — drag from an output port to an input port. Edges are **type-safe**: ports carry an explicit data type (Tensor, Model, Dataset, DataLoader, Optimizer, Loss, Scalar, String, Image, List, Trigger, …) and incompatible connections are rejected with a tooltip.
- **Select** — click a node; **Shift**+click to multi-select; drag a box to marquee-select.
- **Auto layout** — press `Shift`+`L` to lay the graph out left-to-right in one flow direction; the viewport then re-fits to the result. The layout is **skip-aware**: parts of a pipeline bypassed by a skip connection sink below it, so a U-Net reads as a U, residual blocks as small dips under their bypass edges, and plain chains stay a single straight line.

See all shortcuts in **[Key Bindings](./keybindings)**.

## The node palette

The left sidebar lists every node, grouped by category and searchable. Categories are color-coded to match the nodes on the canvas. The **Node category mode** setting toggles between:

- **Basic** — only the essential categories a newcomer needs.
- **All** — every category (94 built-in nodes across 15 categories — see the [Node Reference](./node-reference)).

Installed [plugin packs](/advanced/plugins) add their own nodes here too, namespaced like `foundations:Edu-KNN`.

## The config panel

Select a node and the right-hand panel shows its parameters. Parameter widgets are driven by the backend definition — integers, floats, text, booleans, dropdowns (`select`), file pickers (model / image), and inline tensor-grid editors. Some parameters are **conditionally visible** (`visible_when`) and appear only when a related option is set.

## The results panel

The bottom panel is tabbed and resizable:

- **Execution Log** — per-node status as the graph runs, plus `Print` node output.
- **Training** — a live loss chart fed by the `TrainingLoop` node.

## Start nodes drive execution

Every runnable graph needs at least one **`Start`** node: connect its trigger output (the diamond handle) to the first node you want executed. Only nodes reachable from a `Start` run. Without one, **Run** rejects the graph with a *"No start node defined"* toast. This is covered in detail in **[Your First Graph](./first-graph)**.

## Settings popover

The toolbar **Settings** popover groups every per-tab teaching/training toggle in one place (record outputs, verbose mode, persist weights, capture gradients, grid snap, and more). These power the **[Teaching Inspector](./teaching-inspector)**.
