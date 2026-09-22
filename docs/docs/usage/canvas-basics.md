---
sidebar_position: 1
title: Canvas Basics
description: Overview of the CodefyUI sidebar, canvas, type-safe connections, notes, node cards, toolbar, configuration panel, results panel, and notifications.
---

# Canvas Basics

CodefyUI is a single-page app with a **canvas** in the middle, a **sidebar** on the left, a **config panel** on the right when a node is selected, and a **results panel** at the bottom. This page describes these areas and their main controls. With no tab open, a [welcome screen](./tabs-persistence#the-welcome-screen) takes the place of all of them except the toolbar.

## The sidebar

The left sidebar is an icon rail of tabs; the ones covered here are **Nodes**, **Graphs**, **Templates**, and **Custom & Plugins**. To collapse the panel to the icon rail, click the active tab's icon or press `Ctrl/Cmd`+`Shift`+`B`. Drag the panel edge to set its width from 180 to 520 px. The browser preserves this width across reloads.

- **Nodes** lists all nodes by color-coded category and supports search, **Expand all**, **Collapse all**, and category shortcuts. The search box matches a node's name, its one-line description, its longer details text and, for a plugin node, the plugin's name; it matches the English text of each node, whatever the UI language. With **Show node tooltips** on, hovering a node shows its description, and a plugin node's card ends with a **From plugin:** line that names the plugin (by its id until the Plugin Center catalog has loaded). Nodes that require an [optional pack](./optional-packs) display a **Needs pack** chip. You can place these nodes before installing the pack, but execution fails and names the required pack. A **Presets** group below the node categories lists the reusable node groups you can drag onto the canvas; the search box matches their names, descriptions and tags. See [Node Reference](./node-reference) and [Presets](/advanced/presets).
- **Graphs** lists the graphs saved on the server, most recently modified first, and searchable. Clicking a row opens that graph in a tab of its own and binds the tab to the file, so Save writes straight back over it; the canvas in front of you is never touched, and a graph that is already open brings its tab forward instead of opening a second copy. The row's menu holds **Rename** and **Delete**. The header saves the current graph under a new name and refreshes the list. The footer's **Import...** takes a graph `.json`, which replaces the current tab's canvas, or a `.cduiworkspace` workspace file, which adds tabs. See [Tabs & Persistence](./tabs-persistence).
- **Templates** lists example graphs. Click a template to insert it into the current canvas, or drag it to select its location. See [Examples Gallery](./examples-gallery).
- **Custom & Plugins** provides **Custom Nodes → Manage...**, **Optional packs → Package Center...**, and **Plugins → Plugin Center...**. See [Plugin Center](/advanced/plugins#plugin-center). Installed [plugin packs](/advanced/plugins) add namespaced nodes such as `foundations:Edu-KNN` to the Nodes tab.

The **Node category mode** setting (**Settings → Editor**) sets how much of the library the **Nodes** tab lists. **All**, the default, lists all 152 built-in nodes in 16 categories. **Basic** lists only the **Data**, **CNN**, **Training** and **IO** categories, and only the presets filed under them. It hides every other category, including **Control**, which holds the `Start` node every runnable graph needs, **Utility** (`Print`, `Reshape`) and **Tensor Operations** (`Softmax`). Quick search is not filtered: double-click the canvas to add any node in either mode.

## The canvas

- **Add a node.** Drag it from the sidebar, or **double-click the canvas** to open quick search and enter a node or preset name. Quick search matches nodes on the same fields as the **Nodes** tab search, lists at most 20 results (`Start` first while the box is empty), marks presets with a **PRESET** badge, and places the node where you double-clicked.
- **Select nodes.** Click a node, or use `Shift`+click to select multiple nodes. Hold `Shift` and drag on empty canvas to select with a box; dragging without `Shift` pans the canvas.
- **Move around.** Scroll to zoom, down to 10%. The buttons at the bottom left zoom in, zoom out, fit the graph to the window, and lock the canvas so that nodes cannot be moved, connected or selected. Drag the minimap at the bottom right to pan the view; scroll over it to zoom. Each tab keeps its own pan and zoom while the page is open; after a reload the view is fitted to the graph again.
- **Copy and paste.** `Ctrl/Cmd`+`C` copies the selected nodes, the connections between them, and the definitions of any collapsed subgraphs among them; connections to nodes outside the selection are left behind. `Ctrl/Cmd`+`V` pastes into the active tab: the copies get new ids, land 50 px down and to the right, and arrive selected, and one undo removes them. The clipboard belongs to the editor, not to the system, so it carries nodes between tabs but not to another browser window, and a reload empties it. **Duplicate** in a node's right-click menu copies that one node with its parameters and no connections.
- **Open Node details.** Double-click a node, or select it and press `Enter`. **Node details** shows its parameters, ports, and documentation. After a run, it also shows captured outputs and statistics.
- **Apply auto layout.** Press `Shift`+`L` to arrange the graph from left to right and fit the viewport to the result. The skip-aware layout places pipeline sections that a skip connection bypasses below that connection. This forms the U shape of a U-Net, places residual-block nodes below their bypass edges, and leaves simple chains in one line. By default, **Layout Experiments** arranges only the connected groups of nodes that contain a `Start` node. The toolbar's **Auto Layout** menu also provides **Layout All** and **Layout Selected**. `Shift`+`L` repeats the last selected mode. Unbound notes keep their positions.

See all shortcuts in **[Key Bindings](./keybindings)**.

## Edges

- **Connect ports.** Drag from an output port to an input port. Each port has an explicit data type, such as Tensor, Model, Dataset, DataLoader, Optimizer, Loss, Scalar, String, Image, List, Transform, or Trigger. An incompatible drop is rejected.
- **Detach or rewire an edge.** Drag a connected input port with the left mouse button. Drop the edge on another port to rewire it, or on empty space to delete it. A red ring appears before deletion. Hold `Shift`, `Ctrl`, or `Alt` while dragging to create another connection instead.
- **Inspect an edge value.** After a run, click an edge to see the type, shape, dtype, minimum, maximum, and mean of the value it carried. **View stats** opens the node's Stats tab.
- **Change connection style.** Under **Settings → Editor**, select circuit-board traces, the default, or smooth curves. The style applies to every connection, including the dashed trigger edges from a `Start` node.

## Notes

Right-click empty canvas and select **Add Text Note** or **Add Image Note**. Double-click a text note to edit it; `Esc` or a click elsewhere ends the edit and keeps the text. Click an empty image note to upload an image, or double-click an image note to replace its image; CodefyUI resizes the image to at most 800 px and stores it in the graph. A note's context menu provides **Bind to Nearest Node**, **Unbind Note**, **Change Color** (Yellow, Blue, Green, Red, Purple, or Gray), and **Delete**. A bound note has a line to its node and follows that node during auto layout. An unbound note remains in place, and a notification reports that it was not repositioned.

## Node cards

A node card lists the node's ports and parameters. A parameter hidden by its `visible_when` rule is left out; otherwise basic parameters are always listed, and an **Advanced** parameter only once its value differs from the default. A secret value is shown as `••••••••`, and a code parameter shows its first four lines followed by **+N more lines**. While you drag a connection, the input ports that accept it are highlighted and the others are dimmed.

During and after a run, the card's border colour and footer show the node's status: **Running...**, **Completed**, **Cached**, **Skipped**, or **Error:** followed by the message, whose full text shows on hover. On a `TrainingLoop` card the epoch, the loss and a progress bar replace **Running...** only for a moment after each epoch ends; the **Training** tab of the [results panel](#the-results-panel) follows the run live. A completed `ModelSaver` or `CheckpointSaver` card also has a button with the saved file's name that downloads the file. A bypassed node shows a **BYPASS** badge. A node whose [optional pack](./optional-packs#what-changes-on-the-canvas) is missing shows a **PACK** badge, and a parameter whose chosen value needs a pack or model that is not installed shows **needs pack** beside the value.

## Toolbar

The toolbar contains these controls from left to right: **Run** and **Stop** (**Run** reads **Running...** and is disabled while a run is going; **Stop** is enabled only then); the device select for this graph, whose first option (**Follow Settings (CPU)** by default) runs the graph on the device chosen in Settings and whose other options assign one of this server's devices to the graph, saved in its file (see [GPU & Device Setup](/getting-started/gpu-device#when-a-device-is-unavailable)); a **Save** icon that writes the current tab back to the graph it was opened from, and asks for a name only when the tab is bound to no graph; **File** (**Save**, **Save As...**, and **Clear Canvas**); **Export** (**Export as JSON**, **Export Diagram (SVG)**, **Export Diagram (PNG)**, **Export as Subgraph**, **Export as Python**, and, below a divider, **Workspace (.cduiworkspace)**, which writes every open tab to one [workspace file](./tabs-persistence#workspace-files)); **Templates**; **Reload Nodes**; **Custom Nodes**; **Auto Layout** and its mode menu; the status indicator (the active tab's **Idle**, **Running**, **Completed** or **Error**); controls added by installed plugins; **Settings**; **?** for the shortcut list; the font-size control (**Small**, **Default**, **Large**); and the language control (**English**, **繁體中文**; on a first visit it follows the browser's language). In project mode, a badge next to the logo displays the open project's name. With no tab open, only that badge, **Settings**, **?**, the font size and the language remain. See **[Tabs & Persistence](./tabs-persistence)** for saving, loading, and exporting.

## The config panel

Select a node to display its parameters in the right panel. The backend definition determines the parameter widgets: integers, floats, text, booleans, dropdowns (`select`), model, image and data file pickers, inline tensor-grid editors, masked secret fields, and a code editor that checks a script's imports (see [PythonScript Node](/advanced/python-script-node)). Parameters with a `visible_when` rule appear only when a related option has one of the specified values.

Basic parameters appear immediately. Other parameters are in a collapsed **Advanced** section whose heading displays the hidden count. Advanced parameters are saved with the graph, included by the Python exporter, and processed in the same way as basic parameters.

Conditional visibility is applied before the Advanced count is calculated. For example, when an `Optimizer` uses `SGD`, the count excludes Adam's `betas` because that parameter does not apply. A rule can list several accepted values, so a parameter shared by four of nine optimizers appears for all four.

The config panel and **Node details** use the same form component, so they display the same parameter controls.

A file picker lists the files already on the server. Its three buttons upload a file from your computer and select it, download the selected file, and re-read the list. Model pickers take `.pt`, `.pth`, `.safetensors`, `.ckpt` and `.bin` files; image pickers take `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`, `.gif` and `.tiff`; data pickers take `.csv`, `.tsv`, `.txt` and `.json`. A secret field's value is never written to a saved graph, an export or the browser's autosave, so it lasts only for the session.

A tensor grid sizes itself from the node's `shape` (`TensorInput`) or `kernel_size` (`Conv2dExplicit`). On `TensorInput` it can be edited only while `value_mode` is `explicit`. **Fill 0**, **Fill 1** and **Random** (values between -1 and 1) fill every cell; a tensor with more than two dimensions shows **dim** selectors that pick the 2-D slice on screen; a tensor of more than 512 cells cannot be edited in the grid. When the shape changes, the stored values are kept in order and padded with zeros or cut to fit.

## The results panel

The bottom panel has tabs and can be resized or collapsed:

- **Execution Log** lists what the current tab's run reports, one time-stamped line per event: the start and end of the run, each node's outcome (a node that is running or served from the cache adds no line), the text nodes write (the output of `Print` and of a `PythonScript`'s `print()`, and notes from nodes such as `ModelSaver`, `CheckpointLoader` and `TrainingLoop`), and images, charts and video clips, which are drawn in the log. Each **Run** starts with an empty log. The number on the tab counts the lines.
- **Training** displays the live loss curve and epoch table from `TrainingLoop`, including loss, delta, and time; above them are the current epoch, the latest loss, the best loss and a progress bar. The tab is disabled until `TrainingLoop` starts and reports its configuration, then it is selected automatically unless **Runs** is open; epoch rows follow as they complete. The number on the tab counts the epochs.
- **Runs** lists server-owned runs in queued, active, and finished states. The number on the tab counts the runs that are running or queued on the server. See [Run Queue](./run-queue).

Installed plugins can add more tabs. **Clear** empties the current tab's log, and the Training chart with it; it is not shown on **Runs** or on a plugin's tab.

When a node fails, its line in the Execution Log gives the error. A few common errors are rewritten into a sentence that says what to change:

- a missing input: "Connect a 'tensor' input to this node." or "Connect the required input '…'."
- a `Linear` layer whose `in_features` does not match its input: "Size mismatch: this layer received … features but is configured for …"
- a reshape to a shape the tensor does not fit: "Cannot reshape to …"
- a convolution with the wrong number of input channels: "Channel mismatch: …"
- a node whose [optional pack](./optional-packs) is not installed: "This node needs the … pack. Install it from the Package Center."

A `ValueError` is shown without its class name, and any other error as the node raised it. Error text in the **Runs** panel goes through the same rewrites. Click an error line to open its details underneath, with the message's line breaks kept. The short id on a node's line (the first eight characters of the node id) selects that node on the canvas.

## Start nodes drive execution

Every runnable graph requires at least one **`Start`** node. Connect its trigger output, shown as a diamond handle, to the first node to execute. The executable set contains each directly triggered node, its downstream data flow, any upstream nodes that supply data to that set, and the internal roots of each reached preset or subgraph container. If the graph has no `Start` connection, **Run** rejects it and displays an error that asks you to add and connect a `Start` node. See **[Your First Graph](./first-graph)** for more detail.

## Settings popover

The toolbar's **Settings** popover groups controls into these sections: **Execution**, **LLM Providers**, **Optional Packs & Plugins** ([**Package Center**](./optional-packs) and [**Plugin Center**](/advanced/plugins#plugin-center)), **Recording & Inspection**, **Training Behavior**, **Editor**, and **This Server**. **Recording & Inspection** and **Training Behavior** belong to the active tab's graph and appear only while a tab is open. The rows in each section are listed under [Settings popover toggles](./teaching-inspector#settings-popover-toggles); the recording and gradient controls there determine which data the **[Teaching Inspector](./teaching-inspector)** can display.

## Notifications

Messages appear at the bottom right, each with a close button. Errors stay until you close them, and so does the warning that open graphs changed on disk, which has a **Reload** button. Every other message closes after four seconds, including one that has an **Open Package Center** or **Open Plugin Center** button.

These messages report the connection to the execution server and the state of a run:

- "Connection lost — reconnecting…": the connection dropped, usually because the server stopped or restarted. The editor retries after 1 s, then 2 s, 4 s and so on, up to 30 s between attempts.
- "Connection restored": the connection is back, and a tab that was following a run picks it up again.
- "Could not reconnect to the execution server": ten attempts failed. Start the server again, then reload the page.
- "Graph too large to send — …": the server refused a graph larger than its WebSocket message limit and closed the connection. Simplify the graph, or raise `CODEFYUI_WS_MAX_MESSAGE_BYTES` (see [Graph as a Function](./graph-as-a-function#8-limits-and-gotchas)).
- "Reconnected to a run that is still in progress": after a reload, a tab whose run is still going on the server follows it again.
- "This run was not started — …": the server refused the run. Either two interactive runs are already going, from any tab (`CODEFYUI_RUN_INTERACTIVE_MAX_CONCURRENT`; see [Run Queue](./run-queue#the-interactive-lanes-two-limits)), or this tab's previous run is still going; the Execution Log line ends with the server's reason. Wait for a run to finish, or stop one with **Stop** in the **Runs** panel. When the runs are in other tabs, this tab stays on **Running...** with **Run** disabled; reloading the page resets it.
