---
sidebar_position: 1
title: Presets
description: Save a subgraph as a reusable, parameterized preset and use built-in model templates.
---

# Presets

A **preset** packages a reusable subgraph as one node. CodefyUI includes the `lstm_sequence`, `simple_cnn_classifier`, and `training_pipeline` presets. You can also export a canvas as a preset and use it in other graphs.

## Using a preset

Presets appear in the sidebar's **Presets** tab and in quick search, which opens when you double-click the canvas. The tab contains a search box, the same categories as the Nodes tab, and the hint *Drag presets onto the canvas*. A placed preset behaves like any other node. It exposes every port that was unconnected inside the subgraph and every non-secret parameter of every internal node.

To configure a placed preset, double-click it or select **Configure Preset** in the Node Config panel or node detail view. The preset modal lists its internal nodes and groups exposed parameters by node. **Apply** writes the selected values to the internal nodes.

Before execution, the graph engine **expands** each preset into its internal nodes. A preset packages nodes but does not add a separate runtime.

## Creating your own

Exporting a preset includes the entire canvas. You cannot select a subset of nodes or choose individual items to expose. Use a canvas that contains only the nodes for the preset; a new tab is usually the simplest option.

1. Build the subgraph. Each unconnected port becomes a preset port, so leave the required inputs and outputs unconnected. The graph must have at least one unconnected port. Otherwise, the server rejects the export. Expand all collapsed blocks before exporting. Presets cannot include block definitions, so the server rejects a canvas that contains a collapsed block.
2. Open the toolbar **Export** menu, select **Export as Subgraph**, and enter a name. The server rejects a name already used by another preset with status `409`.
3. The server names each exposed port `<node>_<port>` and exposes every non-secret parameter, grouped by node type. It writes the preset and reloads the palette. The preset is then available in the Presets tab and quick search.

A preset is stored as JSON with `preset_name`, `category` (`Custom` for exported presets), `description`, `tags`, `nodes`, `edges`, `exposed_inputs`, `exposed_outputs`, and `exposed_params`. Exported and built-in presets are stored in `backend/app/presets/`. An exported preset uses `<name>.json`; its file name is lowercase, with spaces and slashes replaced by `_`. No route or button renames or deletes a preset. To remove one, delete its file and reload the nodes. [Plugin packs](./plugins) can also provide presets from their `presets/` directory.

## REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/presets` | GET | List preset definitions. |
| `/api/presets/{name}` | GET | Get a single preset definition (`404` if unknown). |
| `/api/presets/create` | POST | Create a preset from a graph. The body is `{name, nodes, edges, description?, category?, tags?}` and the request requires a session token. Returns `400` when the graph is empty, contains a collapsed block, or has no unconnected port. Returns `409` when the name is already used. |

See the full **[API Reference](./api-reference)**.

:::tip Preset vs custom node
Use a **preset** when you want to package *a graph of existing nodes*. Write a **[custom node](./custom-nodes)** when you need *new behavior* in Python.
:::
