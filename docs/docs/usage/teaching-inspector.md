---
sidebar_position: 4
title: Teaching Inspector
description: Record per-node outputs, inspect input→output tensor diffs, compare a subgraph segment, capture gradients, and view step traces.
---

# Teaching Inspector

CodefyUI can be used as an **interactive lesson** — students see the exact tensor that flows through every node. The Teaching Inspector captures node outputs during a run and renders them in the right-hand panel.

## Walkthrough

1. Drag a **`TensorInput`** node onto the canvas (Data category). Set `value_mode: explicit` and type the numbers you want the pipeline to see into its 16-cell grid, or press **Random** to fill it.
2. Wire it through a chain of tensor-op nodes, e.g. `Reshape → Softmax → Print`, and set Reshape's `shape` to `1,16`: its default, `-1,784`, does not fit 16 values and stops the run (see [Your First Graph](./first-graph)).
3. **Add a `Start` node** and connect its trigger output to the first node you want executed — typically the `TensorInput`. Without this, the graph is a draft and **Run** is rejected (see [Your First Graph](./first-graph)).
4. **Record node outputs** is on by default; check it is still on under **Settings → Recording & Inspection**, then click **Run**. Every completed node's full output is captured in server memory, keyed by the run.
5. Click any node — the **Inspector** panel fetches that node's input and output, showing **shape, dtype, min/max/mean** and the actual values stacked top-to-bottom. On `Softmax` the cells that changed are **heat-coloured**; on `Reshape`, whose output has a different shape, the divider between input and output shows `[1, 4, 4] → [1, 16]` instead.
6. **Shift-select two nodes** and click **Create segment** (**Settings → Recording & Inspection → Compare segment**) to focus on just the head-input and tail-output; the canvas wraps them in a light-orange bubble with **HEAD** / **TAIL** badges.
7. Switch **Record node outputs** off before a heavy training run if you don't want each epoch captured — runs already captured stay fetchable until they are evicted or the server restarts.

:::note
Captured outputs live in one server-wide store shared by every tab: the newest 20 runs and 2 GiB (`CODEFYUI_RUN_OUTPUT_STORE_MAX_MB`, default `2048`). Whole runs are evicted oldest-first, and deleting a run from the Runs panel drops its captures — see [Training Memory](/advanced/training-memory#the-servers-own-memory). Segment markers are saved with the graph JSON.

Creating or clearing a marker is an undoable step: **Ctrl+Z** brings back a marker you removed by mistake — or one that **Collapse to subgraph** or **Delete** swallowed — together with the focus it had.
:::

## The Inspector panel

The right-hand column appears when a node is selected, a segment is active, or a plugin panel is docked there. For a selected node it has three tabs:

| Tab | Shows |
| --- | --- |
| **Forward** | The node's inputs stacked above its outputs, each port with a type-coloured dot; tensors render as value grids. When the node has exactly one tensor input and one tensor output of the same shape, changed cells are heat-coloured; when the two shapes differ, the divider between them shows the change instead. |
| **Steps** | The `__steps__` trace an instrumented node emitted with **Verbose internals** on — see below. |
| **Backward** | The weight and output gradients captured with **Capture gradients** on — see below. |

Until the tab has a run to read — before its first run, and after a page reload — the panel reads **Nothing captured yet**. During a run, each port waits for the node that produces its value (for an input, the upstream node): it reads **Waiting for this node to run…** until that node starts and **Node is running…** while it runs, and fills in by itself when the node finishes. The **Backward** tab reads **Graph is running…** until the whole run ends, because gradients are computed after the forward pass. **Run data expired — re-run to capture** means the server holds no capture for that port: the run was evicted or deleted, the server restarted (see the note above), or the run had **Record node outputs** off. The **Node details** tabs follow the same rules.

A **segment** replaces the node view with a SEGMENT header, **Segment inputs (N)** — every edge entering the head-to-tail set — and **Segment outputs (N)**. **Create segment** needs exactly two selected nodes: the one further left on the canvas is the head, the other the tail, and if no data path runs from head to tail it shows **Segment: no path from head to tail** and creates nothing. A new segment becomes the active one, the one the Inspector shows; click another bubble's border to make that one active. Several segments can coexist on the canvas: the **x** on a bubble removes only that one, and **Clear active** clears the highlighted one the Inspector is showing. The panel collapses to a thin strip with the **›** button.

## Node details

Double-click a node, press **Enter** with it selected, or right-click → **Open details** to open the node-details modal: the parameter form on the left and, on the right, everything the Inspector knows about the node.

| Tab | Shows |
| --- | --- |
| **Code** | Script nodes only, and the tab they open on: the script editor and its input/output port counts. |
| **Subgraph** | Subgraph instances only: the block's boundary ports and an **Enter subgraph** button. |
| **Inputs** / **Outputs** | The captured values on each port, as the Inspector's Forward tab shows them. |
| **Steps** / **Backward** | The step trace and the captured gradients; available once the tab has a run to read from. |
| **Stats** | Summary statistics for every port, computed on the server — see below. |
| **Docs** | The node's one-line description and its longer details text, its parameters with type, default, range and options, and its ports. |

**Left** / **Right** step to the previous / next node on the canvas without closing, **Esc** closes, and clicking the node name turns it into a rename field (Enter applies, Esc cancels). Click an edge after a run for a summary of what flowed through it — type, shape, dtype, min/max/mean — and its **View stats** link opens Node details with that port focused in **Stats**.

### Stats tab

`GET /api/execution/outputs/{run}/{node}/{port}/stats` answers with a fixed-size summary of a captured port rather than its values. Count, min, max, NaN and Inf counts, the zero fraction and — for integer label tensors — the class balance are always exact. Mean, std, the quantiles and the 64-bin histogram are exact up to 4 million elements and computed from a seeded 1-million-element sample above that, marked `"sampled": true` in the response (`CODEFYUI_STATS_SAMPLE_THRESHOLD`, `CODEFYUI_STATS_SAMPLE_SIZE`). Computed summaries are cached up to `CODEFYUI_STATS_CACHE_MAX_BYTES` (8 MB).

## Settings popover toggles

The toolbar **Settings** popover groups its rows by section. **Recording & Inspection** and **Training Behavior** belong to the current tab: they are saved with it, carried in workspace files, and left out while no tab is open (see [The welcome screen](./tabs-persistence#the-welcome-screen)). **Compute device** and the **Editor** rows are kept by this browser and apply to every tab. The **LLM Providers**, **Optional Packs & Plugins** and **This Server** rows are about the server. Every switch is off by default unless its row says otherwise.

| Section | Setting | What it does |
|---|---|---|
| Execution | **Compute device** | The device for graphs whose device select next to **Run** is on **Follow Settings (…)**; CPU until you change it. For what the row shows when this server lacks that device, see [When a device is unavailable](/getting-started/gpu-device#when-a-device-is-unavailable). |
| LLM Providers | **ChatGPT Codex account** | **Sign in** / **Sign out** / **Refresh** for the Codex provider — see [Graph Copilot](/advanced/graph-copilot). |
| Optional Packs & Plugins | **Package Center** | Opens the Package Center; the row counts installed packs. |
| | **Plugin Center** | Opens the [Plugin Center](/advanced/plugins#plugin-center); the row counts installed and available plugins. |
| Recording & Inspection | **Record node outputs** | Capture each completed node's full output for the Inspector. On by default; turn it off before a heavy training run. |
| | **Verbose internals** | Instrumented nodes record their intermediate steps (attention scores, softmax temperatures, ...) — feeds the **Steps** tab. With this on, nothing is served from cache; every node re-executes. |
| | **Compare segment** | **Create segment** wraps two selected nodes in a HEAD/TAIL bubble; **Clear active** removes the highlighted one. |
| Training Behavior | **Persist weights between runs** | Keep `Conv2d`/`Linear`/`Attention` weights across Run clicks so the model actually learns. On by default; when off, every run reinitialises. |
| | **Reset all weights now** | **Reset** asks first, then drops every cached weight for this tab; the next Run starts fresh. |
| | **Capture gradients** | Run forward + `.backward()` and store each layer's gradient for the **Backward** tab. With this on, nothing is served from cache; every node re-executes. |
| | **Auto-synthesize loss** | When the graph has no `Loss`/`BackwardOnce` node, synthesize one so `.backward()` can run. Available only while **Capture gradients** is on. |
| | **Random seed** | Seed every node from one number; blank means unseeded. A seeded run executes one node at a time — see [Reproducible runs](./running-graphs#reproducible-runs-seed). |
| | **Deterministic algorithms** | Ask PyTorch for deterministic kernels (`warn_only`). Sent with the run alongside the seed and the device. |
| Editor | **Grid snap** | Snap dragged nodes to the 24 px canvas grid; turning it on also snaps the nodes already on the canvas. |
| | **Show node tooltips** | Show the description card when hovering a node on the canvas or in the sidebar. On by default. |
| | **Node category mode** | Which categories the sidebar lists: **Basic** only Data, CNN, Training and IO; **All** (default) every category. Quick search (double-click the canvas) is not filtered — see [The sidebar](./canvas-basics#the-sidebar). |
| | **Connection style** | **Circuit** (default) draws connections as circuit-board traces, **Curve** as smooth curves; the trigger edges from a `Start` node follow the same setting. |
| This Server | — | The version, the node and preset counts, and each in-memory store's usage against its budget, with a **Refresh** button. |

## Step traces (Verbose internals)

With **Verbose internals** on, instrumented nodes emit a `__steps__` trace that the Inspector renders one row at a time. Educational plugin nodes lean on this heavily — e.g. `Edu-ColumnStats` shows the population-std formula as `sum → divide → deviations² → variance → sqrt`. See **[Plugins](/advanced/plugins)**.

## Gradient capture (Backward tab)

With **Capture gradients** on, the engine runs a forward pass, calls `.backward()`, and stores each layer's gradient. Open a node's **Backward** tab in the Inspector to see gradient magnitudes per layer — useful for diagnosing vanishing/exploding gradients.

The tab lists **Output gradients** (one per output port) and **Weight gradients** (one per parameter). Each is a grid coloured by |g| relative to its largest entry, under a badge that also prints the L2 norm ‖g‖: **vanishing** when ‖g‖ is below 1e-7 or the mean |g| is below 1e-8, **exploding** when ‖g‖ is above 1e3 or the largest |g| is above 1e2, and **healthy** otherwise.

## Full-size viewers

The five attention cards — the attention heatmap and mask, and the three Edu attention nodes — have a **View full** button, and the embedding scatter card has **Open detailed view**; each opens the plot in a full-size heatmap or scatter viewer. It survives scrolling, zooming and layout changes on the canvas, and closes with **Esc**.

The scatter viewer lists the 60 labels nearest the centre of the view under **Nearest to view centre**, with a **Filter labels…** box. Clicking a label centres the view on it, and its eye button hides or shows that point (**Show all** brings every point back). Drag to pan and scroll to zoom, or use the zoom-out, zoom-in and reset buttons above the plot. The heatmap viewer's footer shows `seq_len`, and **row-normalised colours** when each row is coloured on its own scale, as attention weights are.
