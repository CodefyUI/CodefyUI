---
sidebar_position: 1
title: Canvas Basics
description: Overview of the CodefyUI sidebar, canvas, type-safe connections, notes, toolbar, configuration, and results panels.
---

# Canvas Basics

CodefyUI is a single-page app with a **canvas** in the middle, a **sidebar** on the left, a **config panel** on the right when a node is selected, and a **results panel** at the bottom. This page describes these areas and their main controls.

## The sidebar

The left sidebar is an icon rail of tabs; the ones covered here are **Nodes**, **Presets**, **Templates**, and **Custom & Plugins**. To collapse the panel to the icon rail, click the active tab's icon or press `Ctrl/Cmd`+`Shift`+`B`. Drag the panel edge to set its width from 180 to 520 px. The browser preserves this width across reloads.

- **Nodes** lists all nodes by color-coded category and supports search, **Expand all**, **Collapse all**, and category shortcuts. The **Node category mode** setting switches between **Basic**, which shows the essential categories, and **All**, which shows all 152 built-in nodes in 16 categories. See [Node Reference](./node-reference). Nodes that require an [optional pack](./optional-packs) display a **Needs pack** chip. You can place these nodes before installing the pack, but execution fails and names the required pack.
- **Presets** lists reusable node groups that you can search and drag onto the canvas.
- **Templates** lists example graphs. Click a template to insert it into the current canvas, or drag it to select its location. See [Examples Gallery](./examples-gallery).
- **Custom & Plugins** provides **Custom Nodes → Manage...**, **Optional packs → Package Center...**, and **Plugins → Plugin Center...**. See [Plugin Center](/advanced/plugins#plugin-center). Installed [plugin packs](/advanced/plugins) add namespaced nodes such as `foundations:Edu-KNN` to the Nodes tab.

## The canvas

- **Add a node.** Drag it from the sidebar, or **double-click the canvas** to open quick search and enter a node or preset name.
- **Select nodes.** Click a node, or use `Shift`+click to select multiple nodes. Hold `Shift` and drag on empty canvas to select with a box; dragging without `Shift` pans the canvas.
- **Open Node details.** Double-click a node, or select it and press `Enter`. **Node details** shows its parameters, ports, and documentation. After a run, it also shows captured outputs and statistics.
- **Apply auto layout.** Press `Shift`+`L` to arrange the graph from left to right and fit the viewport to the result. The skip-aware layout places pipeline sections that a skip connection bypasses below that connection. This forms the U shape of a U-Net, places residual-block nodes below their bypass edges, and leaves simple chains in one line. By default, **Layout Experiments** arranges only the connected groups of nodes that contain a `Start` node. The toolbar's **Auto Layout** menu also provides **Layout All** and **Layout Selected**. `Shift`+`L` repeats the last selected mode. Unbound notes keep their positions.

See all shortcuts in **[Key Bindings](./keybindings)**.

## Edges

- **Connect ports.** Drag from an output port to an input port. Each port has an explicit data type, such as Tensor, Model, Dataset, DataLoader, Optimizer, Loss, Scalar, String, Image, List, Transform, or Trigger. An incompatible drop is rejected.
- **Detach or rewire an edge.** Drag a connected input port with the left mouse button. Drop the edge on another port to rewire it, or on empty space to delete it. A red ring appears before deletion. Hold `Shift`, `Ctrl`, or `Alt` while dragging to create another connection instead.
- **Inspect an edge value.** After a run, click an edge to see the type, shape, dtype, minimum, maximum, and mean of the value it carried. **View stats** opens the node's Stats tab.
- **Change connection style.** Under **Settings → Editor**, select circuit-board traces, the default, or smooth curves for value edges.

## Notes

Right-click empty canvas and select **Add Text Note** or **Add Image Note**. Double-click a text note to edit it. Click an image note to upload an image; CodefyUI resizes it to at most 800 px and stores it in the graph. A note's context menu provides **Bind to Nearest Node**, **Unbind Note**, **Change Color** (Yellow, Blue, Green, Red, Purple, or Gray), and **Delete**. A bound note has a line to its node and follows that node during auto layout. An unbound note remains in place, and a notification reports that it was not repositioned.

## Toolbar

The toolbar contains these controls from left to right: **Run** or **Stop**; **File** (**Save**, **Save As...**, and **Clear Canvas**); **Load** (searchable saved graphs and **Import JSON...**); **Export** (**Export as JSON**, **Export Diagram (SVG)**, **Export Diagram (PNG)**, **Export as Subgraph**, and **Export as Python**); **Templates**; **Reload Nodes**; **Custom Nodes**; **Auto Layout** and its mode menu; the status indicator; controls added by installed plugins; **Settings**; **?** for the shortcut list; the UI font-size control; and the language control. In project mode, a badge displays the open project's name. See **[Tabs & Persistence](./tabs-persistence)** for saving, loading, and exporting.

## The config panel

Select a node to display its parameters in the right panel. The backend definition determines the parameter widgets: integers, floats, text, booleans, dropdowns (`select`), model and image file pickers, and inline tensor-grid editors. Parameters with a `visible_when` rule appear only when a related option has one of the specified values.

Basic parameters appear immediately. Other parameters are in a collapsed **Advanced** section whose heading displays the hidden count. Advanced parameters are saved with the graph, included by the Python exporter, and processed in the same way as basic parameters.

Conditional visibility is applied before the Advanced count is calculated. For example, when an `Optimizer` uses `SGD`, the count excludes Adam's `betas` because that parameter does not apply. A rule can list several accepted values, so a parameter shared by four of nine optimizers appears for all four.

The config panel and **Node details** use the same form component, so they display the same parameter controls.

## The results panel

The bottom panel has tabs and can be resized or collapsed:

- **Execution Log** displays each node's status during a run and output from `Print` nodes.
- **Training** displays the live loss curve and epoch table from `TrainingLoop`, including loss, delta, and time. The tab is disabled until `TrainingLoop` starts and reports its configuration, then it is selected automatically unless **Runs** is open; epoch rows follow as they complete.
- **Runs** lists server-owned runs in queued, active, and finished states. See [Run Queue](./run-queue).

Installed plugins can add more tabs.

## Start nodes drive execution

Every runnable graph requires at least one **`Start`** node. Connect its trigger output, shown as a diamond handle, to the first node to execute. The executable set contains each directly triggered node, its downstream data flow, any upstream nodes that supply data to that set, and the internal roots of each reached preset or subgraph container. If the graph has no `Start` connection, **Run** rejects it and displays an error that asks you to add and connect a `Start` node. See **[Your First Graph](./first-graph)** for more detail.

## Settings popover

The toolbar's **Settings** popover groups controls into these sections: **Execution**, **LLM Providers**, **Optional packs** ([**Package Center**](./optional-packs)), **Plugins** ([**Plugin Center**](/advanced/plugins#plugin-center)), **Recording & Inspection**, **Training Behavior**, **Editor**, and **This Server**. The rows in each section are listed under [Settings popover toggles](./teaching-inspector#settings-popover-toggles); the recording and gradient controls there determine which data the **[Teaching Inspector](./teaching-inspector)** can display.
