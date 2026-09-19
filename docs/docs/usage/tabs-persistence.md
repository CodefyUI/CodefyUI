---
sidebar_position: 5
title: Tabs & Persistence
description: Multi-tab workspaces, automatic in-browser saving, importing/exporting graphs as JSON, and moving every open tab to another browser.
---

# Tabs & Persistence

## Multi-tab workspace

CodefyUI supports multiple independent canvases as tabs. Each tab has its own:

- nodes, edges, and layout
- execution context and logs
- recorded outputs and persisted weights (see [Teaching Inspector](./teaching-inspector))
- undo/redo history (up to 50 steps)

This lets you keep several experiments side by side — for example a training graph in one tab and an inference graph in another — without their state interfering.

### The tab strip

Double-click a tab to rename it (`Enter` applies, `Esc` cancels). A running tab shows a dot next to its name. A **Read-only** badge means Save (and plugin writes) are refused — typically because the graph was written by a newer CodefyUI than this one — while renaming and closing still work; a tab a plugin opened says so when you hover it. Closing a tab that holds a graph asks first and names its node count, because there is no undo for a closed tab. When the server runs on a [project directory](./project-directories), each project keeps its own set of tabs.

## Automatic saving

All tabs are auto-saved in your browser, so your work is restored when you reload the page. This is local to the browser; it is not synced to the server. Another browser, or another computer, therefore starts with none of them — a [workspace file](#workspace-files) is how you take them along.

Saving uses **IndexedDB**, with one record per tab. That matters for large graphs: the older `localStorage` backend capped an origin at roughly 5MB, and a graph past that limit simply stopped being saved. IndexedDB has no comparable practical limit, and only the tab you edited is rewritten.

The first time you open a version with IndexedDB saving, whatever `localStorage` last held is copied across automatically — you do not need to do anything. The old `localStorage` copy is left behind (so downgrading still opens the graph it last saw) but stops being updated from then on.

If your browser has no usable IndexedDB — some private-browsing modes, or a sandboxed frame — saving falls back to `localStorage`, with its old size limit and its "storage is full" warning.

## Saving and loading

Saved graphs are stored by the server, not by the browser:

- **File → Save**, and the toolbar's **Save** icon, write the current tab's graph straight back to the graph the tab was opened from, with no prompt. That holds wherever the server runs; a project directory is not required for it. A tab bound to no graph — a new canvas, an import, an inserted example — and **Save As...** ask for a name instead, and warn before a name already in the list replaces the graph holding it. **File → Clear Canvas** empties the tab after a confirmation.
- The sidebar's **Graphs** tab lists them (searchable, most recently modified first), and the row bound to the current tab is marked. Clicking a row opens that graph in a new tab and binds the tab to the file. The canvas you were working on is never replaced, so there is nothing to confirm and nothing to lose. Clicking a graph that is already open switches to the tab holding it rather than opening it twice.
- The row's menu holds two items, **Rename** and **Delete**. A delete confirms first, then removes the file from disk, and both halves of it in a [project directory](./project-directories). When the renamed or deleted file is one a tab saves back to, that tab follows: it binds to the new name, or to nothing, so the next Save asks rather than quietly recreating what you just deleted. The panel's header has a **Save as...** of its own, so a save lands visibly in the list you are looking at.

## Import / export

You can export any graph to a JSON file and import it back later (or share it):

- **Export → Export as JSON** writes the current tab's graph (nodes, edges, parameters, segment markers and subgraph definitions) to a `.json` file.
- **Import...**, at the foot of the sidebar's **Graphs** tab, takes a graph `.json` or a [workspace file](#workspace-files) and tells them apart by what is inside, not by the file name. The two do opposite things: a graph **replaces** the current tab's canvas — open a new tab first to keep your graph — while a workspace file only **adds** tabs and leaves every open tab alone. A `.json` file that is not a graph is refused with a message instead of emptying the canvas, and a graph written by a newer CodefyUI opens read-only, with a notice. An imported graph is bound to no file, so the first Save asks for a name.
- **Export → Export Diagram (SVG / PNG)** draws the architecture only — nodes, ports and connections, no parameter values — on a light, document-friendly background.
- **Export → Export as Python** writes a readable, single-file Python program: one function per node (with its parameters inlined as editable literals), flow functions that wire the nodes together in execution order, and a `main()` entry point with a small CLI. Each node function delegates to the same node implementation the canvas uses, so results match what you saw on the canvas. Run it with the Python environment from a compatible CodefyUI installation; it does not need the web server. Use `--help` for device, GraphInput JSON, timeout, and project-asset options.

The same JSON format is what the backend's example graphs use, so an exported graph can also be run headless with the **[CLI Graph Runner](./cli-runner)**.

:::tip
Because graphs are plain JSON, they diff and version-control cleanly. Commit a graph alongside your code to capture an exact, reproducible pipeline.
:::

### Workspace files

**Export → Workspace (.cduiworkspace)** writes every open tab into one file, `workspace-YYYY-MM-DD.cduiworkspace`, so a whole session can move to another browser or another computer. **Import...** reads it back: the tabs are added beside the ones already open, no open graph is replaced, and a browser holding only one empty tab ends up with exactly the exported set.

For each tab the file carries its title, its graph and its run settings: Random seed, Deterministic algorithms, Record node outputs, Verbose internals, Persist weights between runs, Capture gradients and Auto-synthesize loss. It also carries which tab was active, and six preferences: Language, Font size, Connection style, Grid snap, Show node tooltips and Node category mode. Each `tabs[i].graph` in the file is an ordinary Export-as-JSON graph.

What never travels:

- secret parameter values
- anything a plugin stored in the browser, such as the provider API keys an assistant plugin keeps there
- run results and trained weights, which live in the server's memory
- the binding between a tab and a saved graph
- the Settings compute device and the panel layout, which belong to one machine

An imported tab is therefore bound to no saved graph: its first Save asks for a name, and a name already in the list is confirmed before it replaces the graph holding it. In a [project directory](./project-directories) nothing is stamped until that first Save.

Secret values are recognised from the node's definition, so a node whose type the exporting browser has not loaded — its plugin disabled or missing — has no secret the export can recognise; Save and **Export as JSON** have the same limit, so check such tabs before you share a file.

Empty tabs, read-only tabs and tabs a plugin opened temporarily are not exported. On import, an entry that is not a graph, or that would be the 33rd open tab, is skipped and the rest still open — an import stops at 32 tabs, and the lone empty tab it replaces is not one of them. A file over 64 MiB, or a workspace file written by a newer CodefyUI, is refused whole. A node type that is not installed here opens as a placeholder and a warning names the missing types; nothing re-links them later, so install the plugin or custom node they come from and import the file again.
