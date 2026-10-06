---
sidebar_position: 1
title: Presets
description: Save a subgraph as a reusable, parameterized preset and use built-in model templates.
---

# Presets

A **preset** packages a reusable subgraph as one node. CodefyUI includes the `lstm_sequence`, `simple_cnn_classifier`, and `training_pipeline` presets. You can also export a canvas as a preset and use it in other graphs.

## Using a preset

Presets appear in the sidebar's **Nodes** tab, in a **Presets** group below the node categories, and in quick search, which opens when you double-click the canvas. The tab's search box matches preset names as well as node names. Drag a preset onto the canvas like any other item in the tab. A placed preset behaves like any other node. It exposes every port that was unconnected inside the subgraph and every non-secret parameter of every internal node. A trigger from `Start` into a placed preset starts every node inside it that no other node inside it feeds, and an edge on a port the preset does not expose is refused.

To configure a placed preset, double-click it or select **Configure** in the Node Config panel or node detail view. The preset modal lists its internal nodes and groups exposed parameters by node. **Apply** writes the selected values to the internal nodes.

Before execution, the graph engine **expands** each preset into its internal nodes. A preset packages nodes but does not add a separate runtime.

## Portable definitions

A saved or imported graph can embed the preset definitions it uses. These definitions belong to that graph, so a registry refresh or disabling the plugin that supplied a preset does not remove the open graph's copy. If an installed preset has the same name, the embedded definition wins inside that graph; the installed palette remains unchanged for other graphs. A **Map** node is the exception: it looks up the preset named by its `subgraph` parameter among the installed presets only, so it ignores an embedded definition and fails when no installed preset has that name.

Save and export include only definitions that the graph still references, including presets used only inside a collapsed block. Secret parameter values remain session-only: they are never made portable in a definition or written to a saved or exported graph.

## Creating your own

Exporting a preset includes the entire canvas. You cannot select a subset of nodes or choose individual items to expose. Use a canvas that contains only the nodes for the preset; a new tab is usually the simplest option.

1. Build the subgraph. Each unconnected port becomes a preset port, so leave the required inputs and outputs unconnected. The graph must have at least one unconnected port. Otherwise, the server rejects the export. Expand all collapsed blocks before exporting. Presets cannot include block definitions, so the server rejects a canvas that contains a collapsed block. `Start` nodes, their trigger wires, and notes are left out of the preset; a trigger into the placed preset starts it. A preset card on the canvas is copied in as its own nodes, with the settings made on the card, and a bypassed node is left out the way a run skips it.
2. Open the toolbar **Export** menu, select **Export as Subgraph**, and enter a name. The server rejects a name already used by another preset with status `409`, and the name box opens again with the reason under the input.
3. The server numbers the nodes in the order data flows through them, names each exposed port `<node>_<port>`, and exposes every non-secret parameter, grouped by node; a node type that occurs more than once is numbered (`Activation 1`, `Activation 2`). It writes the preset and reloads the palette. The preset is then available in the Nodes tab's **Presets** group and in quick search.

A preset is stored as JSON with `preset_name`, `category` (`Custom` for exported presets), `description`, `tags`, `nodes`, `edges`, `exposed_inputs`, `exposed_outputs`, and `exposed_params`. Built-in presets are stored in `backend/app/presets/`. Exported presets are stored in `backend/data/presets/`, next to saved graphs; set `CODEFYUI_USER_PRESETS_DIR` to use another directory. Every project the installation opens uses the same directory. Presets exported by older versions, which saved them in `backend/app/presets/`, still load from there. An exported preset uses `<name>.json`; its file name is lowercase, with spaces replaced by `_`. A name that Windows cannot store as a file name, such as one containing `/` or `?`, is refused (see [File names](./api-reference#limits-and-errors)). No route or button renames or deletes a preset. To remove one, delete its file and reload the nodes. [Plugin packs](./plugins) can also provide presets from their `presets/` directory.

## REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/presets` | GET | List preset definitions. |
| `/api/presets/{name}` | GET | Get a single preset definition (`404` if unknown). |
| `/api/presets/create` | POST | Create a preset from a graph. The body is `{name, nodes, edges, presets?, description?, category?, tags?}` and the request requires a session token; `presets` holds the graph's own preset definitions, which a card is copied in from ahead of an installed preset. Returns `400` when the graph has no node other than `Start` that is not bypassed (`preset_empty`), holds a card whose preset is neither installed nor in `presets` (`preset_card_unknown`) or a node whose type is not installed (`preset_node_unknown`), has no unconnected port (`preset_no_ports`), contains a collapsed block, or holds a card or a bypassed node that a run would refuse (the run's message). Returns `409` when the name is already used (`preset_exists`, or `preset_file_exists` when its file exists). A coded refusal's `detail` is `{code, ...}`. |

See the full **[API Reference](./api-reference)**.

:::tip Preset vs custom node
Use a **preset** when you want to package *a graph of existing nodes*. Write a **[custom node](./custom-nodes)** when you need *new behavior* in Python.
:::
