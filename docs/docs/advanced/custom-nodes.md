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
| `define_inputs()` / `define_outputs()` | Return `PortDefinition` lists — each has a `name`, a `data_type`, and optional `description` / `optional` / `media`. |
| `define_params()` | Return `ParamDefinition` lists — `int`, `float`, `string`, `bool`, `select`, file pickers, `tensor_grid`, or `secret`, with `default`, `options`, `min_value`/`max_value`, and `visible_when`. A `secret` param (e.g. an API key) is masked in the editor and its value is **never persisted** — it is blanked on save, export, and publish, so use an environment variable to supply it to published apps. |
| `define_outputs_dynamic(params)` | Optional — vary output ports by parameter values. |
| `execute(self, inputs, params, *, context=...)` | The work. Returns a dict keyed by output port name. |

## Data types

Ports use the shared `DataType` enum: `TENSOR`, `MODEL`, `DATASET`, `DATALOADER`, `OPTIMIZER`, `LOSS_FN`, `SCALAR`, `STRING`, `IMAGE`, `LIST`, `ANY`, `TRIGGER`. Matching types make an edge valid; the `TRIGGER` type drives execution order from [`Start`](/usage/first-graph) nodes.

## Showing an image in the results panel

A node that returns a picture must **declare** it with `media=MEDIA_IMAGE` on the output port. The value on that port is then a base64-encoded PNG string (no `data:` prefix), and the results panel renders it as an image:

```python
from app.core.node_base import MEDIA_IMAGE, BaseNode, DataType, PortDefinition

@classmethod
def define_outputs(cls):
    return [
        PortDefinition(
            name="image",
            data_type=DataType.STRING,
            media=MEDIA_IMAGE,
        ),
    ]
```

Declaring is mandatory — nothing inspects your values to work out what they are. An undeclared port stays plain data no matter how much it looks like an image, which is what keeps a long text output (an LLM answer, a token dump) from being rendered as a broken picture.

## Drawing a chart in the results panel

`media=MEDIA_CHART` is the same mechanism for plots. The port's value is a JSON **chart spec** — a plain dict — which the editor draws with its own SVG components, so the picture is themed, hoverable, and sharp at any zoom instead of a fixed-size PNG:

```python
from app.core.node_base import MEDIA_CHART, BaseNode, DataType, PortDefinition

@classmethod
def define_outputs(cls):
    return [
        PortDefinition(name="chart", data_type=DataType.ANY, media=MEDIA_CHART),
    ]

def execute(self, inputs, params, progress_callback=None, *, context=None):
    return {"chart": {
        "kind": "bar",                       # bar | line | scatter | heatmap
        "title": "Mean petal length by species",
        "bars": [{"label": "setosa", "value": 1.462}],
    }}
```

Every number in a spec must be a finite, plain Python `float` or `int`: run events are serialised with `allow_nan=False`, and a `numpy.float32` is not JSON-serialisable at all. Keep specs small — a `node_status` payload over 128 KB is replaced by an elision marker. The full per-kind payload reference lives in the [stats pack's README](https://github.com/CodefyUI/CodefyUI/blob/main/plugins/stats/README.md), which is the reference implementation.

### Your own media kind

Neither kind is special-cased. The resolver keys on whatever string a port declares, and any port value that is a non-empty dict is shipped through untouched, so a pack declaring `media="waveform"` already arrives in the browser as `{"output_kind": "waveform", ...}`. Only *rendering* it needs a frontend change; an editor that does not know a kind ignores it rather than breaking.

:::tip
Need to package existing nodes rather than write new behavior? Use a **[preset](./presets)**. Want to share nodes with others as an installable bundle? Build a **[plugin pack](./plugins)**.
:::
