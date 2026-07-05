---
sidebar_position: 2
title: Custom Nodes
description: Add new node behavior by dropping a Python file into custom_nodes/ — hot-reloadable, no frontend changes.
---

# Custom Nodes

CodefyUI is **backend-authoritative**: a node's ports, parameters, and category all come from its Python definition, and the UI renders it automatically. To add new behavior, drop a `.py` file into `backend/app/custom_nodes/` that extends `BaseNode`.

## Minimal example

```python
from app.core.node_base import BaseNode, DataType, PortDefinition

class MyNode(BaseNode):
    NODE_NAME = "MyNode"
    CATEGORY = "Custom"
    DESCRIPTION = "Does something"

    @classmethod
    def define_inputs(cls):
        return [PortDefinition(name="input", data_type=DataType.TENSOR)]

    @classmethod
    def define_outputs(cls):
        return [PortDefinition(name="output", data_type=DataType.TENSOR)]

    def execute(self, inputs, params):
        return {"output": inputs["input"]}
```

## Hot reload

After adding or editing a custom node, reload without restarting the server:

- click the toolbar **Reload Nodes** button, or
- `POST /api/nodes/reload`.

The node appears in the palette immediately. You can also use the **Custom Node Manager** GUI to upload, enable/disable, and delete custom nodes.

## Anatomy of a node

| Member | Purpose |
|--------|---------|
| `NODE_NAME` | Unique identifier used in graph JSON (e.g. `"MyNode"`). |
| `CATEGORY` | Palette grouping and color. |
| `DESCRIPTION` | User-facing help text (LaTeX is supported). |
| `define_inputs()` / `define_outputs()` | Return `PortDefinition` lists — each has a `name`, a `data_type`, and optional `description` / `optional`. |
| `define_params()` | Return `ParamDefinition` lists — `int`, `float`, `string`, `bool`, `select`, file pickers, `tensor_grid`, or `secret`, with `default`, `options`, `min_value`/`max_value`, and `visible_when`. A `secret` param (e.g. an API key) is masked in the editor and its value is **never persisted** — it is blanked on save, export, and publish, so use an environment variable to supply it to published apps. |
| `define_outputs_dynamic(params)` | Optional — vary output ports by parameter values. |
| `execute(self, inputs, params, *, context=...)` | The work. Returns a dict keyed by output port name. |

## Data types

Ports use the shared `DataType` enum: `TENSOR`, `MODEL`, `DATASET`, `DATALOADER`, `OPTIMIZER`, `LOSS_FN`, `SCALAR`, `STRING`, `IMAGE`, `LIST`, `ANY`, `TRIGGER`. Matching types make an edge valid; the `TRIGGER` type drives execution order from [`Start`](/usage/first-graph) nodes.

:::tip
Need to package existing nodes rather than write new behavior? Use a **[preset](./presets)**. Want to share nodes with others as an installable bundle? Build a **[plugin pack](./plugins)**.
:::
