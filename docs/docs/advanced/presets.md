---
sidebar_position: 1
title: Presets
description: Save a subgraph as a reusable, parameterized preset — and reuse built-in model templates.
---

# Presets

A **preset** is a reusable subgraph packaged as a single black-box node. CodefyUI ships built-in model templates so you can start fast, and you can save your own selections as presets to reuse across graphs.

## Using a preset

Presets appear in the node palette and the quick search (double-click the canvas) alongside regular nodes. Drag one in and it behaves like any node — with the input ports, output ports, and parameters that the preset author chose to **expose**.

At execution time the graph engine **expands** each preset into its internal nodes before running, so a preset is a packaging convenience, not a separate runtime.

## Creating your own

1. Select the nodes you want to package (Shift-click or marquee).
2. Open the **Create Preset** modal.
3. **Expose** the inputs, outputs, and parameters you want the preset to surface — everything else stays internal.
4. Save. Your preset now appears in the palette for reuse.

A preset is stored as JSON containing its `nodes`, `edges`, `exposed_inputs`, `exposed_outputs`, and `exposed_params`. Built-in presets live in the backend's `presets/` directory and are loaded at startup; [plugin packs](./plugins) can ship their own presets too.

## REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/presets` | GET | List preset definitions. |
| `/api/presets/{name}` | GET | Get a single preset definition. |
| `/api/presets/create` | POST | Create a new preset from selected nodes. |

See the full **[API Reference](./api-reference)**.

:::tip Preset vs custom node
Use a **preset** when you want to package *a graph of existing nodes*. Write a **[custom node](./custom-nodes)** when you need *new behavior* in Python.
:::
