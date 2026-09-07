---
sidebar_position: 2
title: Custom Nodes
description: Add new node behavior by dropping a Python file into custom_nodes/ — hot-reloadable, no frontend changes.
---

# Custom Nodes

CodefyUI is **backend-authoritative**: a node's ports, parameters, and category all come from its Python definition, and the UI renders it automatically. To add new behavior, drop a `.py` file into `backend/app/custom_nodes/` that extends `BaseNode`.

:::tip For a few lines of code, try the canvas first
If what you need is a short transform or a statistic over what a graph already produced, the [PythonScript node](./python-script-node.md) runs Python you type straight onto the canvas -- no file, no restart. Come back here when the code outgrows it, or when it needs files, the network, or a dependency outside the script allowlist.
:::

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

The node appears in the palette immediately. To upload a file instead of copying it into the directory, use the [Custom Node Manager](#uploading-through-the-custom-node-manager); it reloads node definitions after each action.

## Uploading through the Custom Node Manager

Open the manager with the **Manage...** button in the **Custom Nodes** section of the sidebar's **Custom & Plugins** tab. It lists each file in `custom_nodes/`, the node names defined by that file, and three actions:

- **Upload .py** sends one file to `POST /api/custom-nodes/upload`. The file must have a `.py` extension and cannot exceed `CODEFYUI_MAX_UPLOAD_SIZE` (500 MB). The server scans it with the plugin AST gate at [Tier 0](/advanced/plugins#security--three-tiers). Custom nodes cannot declare capabilities, so imports such as `requests` or `os` produce a `400` response with the gate's message. Put a node that needs imports outside Tier 0 in a [plugin pack](./plugins) with a `[security]` section. Files copied directly into `backend/app/custom_nodes/` are loaded at the next reload without this scan.
- **Enable / Disable** renames the file between `name.py` and `name.py.disabled`; a disabled file stays on disk and is skipped by discovery.
- **Delete** removes the file (names starting with `__` are protected).

After each action, the server rediscovers custom nodes, plugin packs, and presets. The palette reflects the result when the request completes; no separate reload is required.

## Anatomy of a node

| Member | Purpose |
|--------|---------|
| `NODE_NAME` | Unique identifier used in graph JSON (e.g. `"MyNode"`). |
| `CATEGORY` | Palette grouping and color. |
| `DESCRIPTION` | User-facing help text (LaTeX is supported). |
| `define_inputs()` / `define_outputs()` | Return `PortDefinition` lists — each has a `name`, a `data_type`, and optional `description` / `optional` / `media`. |
| `define_params()` | Return `ParamDefinition` lists — `int`, `float`, `string`, `bool`, `select`, file pickers (`model_file`, `image_file`, `data_file`), `tensor_grid`, `code` (a multi-line editor with syntax highlighting; still an ordinary string param), or `secret`, with `default`, `options`, `min_value`/`max_value`, and `visible_when`. A `secret` param (e.g. an API key) is masked in the editor and its value is **never persisted** — it is blanked on save, export, and publish, so use an environment variable to supply it to published apps. |
| `define_outputs_dynamic(params)` / `define_inputs_dynamic(params)` | Optional. Change output or input ports based on parameter values, such as `Split`'s `chunks` or `PythonScript`'s `input_ports`. The static methods must describe the default parameters because the palette uses them; validation, rendering, and preset export use the dynamic definitions. |
| `execute(self, inputs, params, progress_callback=None, *, context=None)` | Execute the node and return a dict keyed by output-port name. The engine passes each optional keyword argument only when the signature declares it. `progress_callback` receives a dict for each progress event; for example, the training loop sends `{"event": "epoch", ...}`. `context` provides the run's device, seed, and determinism flag. |
| `REQUIRES_PACK` | Optional class attribute identifying the [optional pack](/usage/optional-packs) required at execution time (`None` by default). `/api/nodes` exposes it as `requires_pack`, allowing the palette to display a pack badge and the editor to offer installation before execution fails. |
| `cacheable` / `align_inputs` / `cache_fingerprint(params)` | Optional cache and device controls. Set `cacheable = False` for a node with trainable state, a returned live object reference, or a side effect not represented by its return value. Set `align_inputs = False` when passing inputs directly to numpy, sklearn, or PIL; otherwise the engine moves input tensors to the run device, and `Tensor.numpy()` fails for tensors outside the CPU. Override `cache_fingerprint` to add external state referenced by a parameter, such as a file's modification time, to the cache key. |

## Data types

Ports use the shared `DataType` enum: `TENSOR`, `MODEL`, `DATASET`, `DATALOADER`, `OPTIMIZER`, `LOSS_FN`, `SCALAR`, `STRING`, `IMAGE`, `LIST`, `TRANSFORM`, `ANY`, `TRIGGER`. Matching types make an edge valid; the `TRIGGER` type drives execution order from [`Start`](/usage/first-graph) nodes.

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

Every number in a spec must be a finite, plain Python `float` or `int`: run events are serialised with `allow_nan=False`, and a `numpy.float32` is not JSON-serialisable at all. Keep specs small — a `node_status` payload over `CODEFYUI_RUN_EVENT_PAYLOAD_CAP_BYTES` (128 KB by default) is replaced by an elision marker. The full per-kind payload reference lives in the [stats pack's README](https://github.com/CodefyUI/CodefyUI/blob/main/plugins/stats/README.md), which is the reference implementation.

## Emitting a playable video

`media=MEDIA_VIDEO` is the third built-in kind, and it works by **reference**: a clip cannot ride the event stream (one `node_status` event is capped at 128 KB), so the file lives under `settings.MEDIA_DIR` and the port's value is a small dict pointing at it. `/api/media/<path>` serves the file inline with a real `Content-Type`, so the editor's `<video>`/`<img>` elements play it directly:

```python
{"path": "rollouts/run1.mp4",          # POSIX-style, RELATIVE to MEDIA_DIR
 "url": "/api/media/rollouts/run1.mp4",
 "format": "mp4",                       # mp4 | gif | webm
 "fps": 10.0, "frames": 240, "width": 96, "height": 96, "bytes": 81234}
```

Don't hand-build these — write frames through the `VideoWrite` node (or call `core.video_io` from your node), which owns the encoding (mp4 via an `ffmpeg` binary on PATH, gif via Pillow with no dependency at all), the path containment, and the reference shape. An absolute `path` is refused at the wire.

### Your own media kind

No kind is special-cased. The resolver keys on whatever string a port declares, and any port value that is a non-empty dict is shipped through untouched, so a pack declaring `media="waveform"` already arrives in the browser as `{"output_kind": "waveform", ...}`. Only *rendering* it needs a frontend change; an editor that does not know a kind ignores it rather than breaking.

:::tip
Need to package existing nodes rather than write new behavior? Use a **[preset](./presets)**. Want to share nodes with others as an installable bundle? Build a **[plugin pack](./plugins)**.
:::
