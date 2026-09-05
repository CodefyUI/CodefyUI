---
sidebar_position: 5
title: Tabs & Persistence
description: Multi-tab workspaces, automatic in-browser saving, and importing/exporting graphs as JSON.
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

All tabs are auto-saved in your browser, so your work is restored when you reload the page. This is local to the browser; it is not synced to the server.

Saving uses **IndexedDB**, with one record per tab. That matters for large graphs: the older `localStorage` backend capped an origin at roughly 5MB, and a graph past that limit simply stopped being saved. IndexedDB has no comparable practical limit, and only the tab you edited is rewritten.

The first time you open a version with IndexedDB saving, whatever `localStorage` last held is copied across automatically — you do not need to do anything. The old `localStorage` copy is left behind (so downgrading still opens the graph it last saw) but stops being updated from then on.

If your browser has no usable IndexedDB — some private-browsing modes, or a sandboxed frame — saving falls back to `localStorage`, with its old size limit and its "storage is full" warning.

## Saving and loading

Saved graphs are stored by the server, not by the browser:

- **File → Save** / **Save As...** write the current tab's graph under a name; Save asks before overwriting an existing one. **File → Clear Canvas** empties the tab after a confirmation.
- **Load** lists the saved graphs (searchable). **Load into this canvas tab** replaces what is on the canvas without binding the tab to the file, so the next Save asks where to put it; **Load and save** binds the tab to the file, so Save writes straight back over it.

## Import / export

You can export any graph to a JSON file and import it back later (or share it):

- **Export → Export as JSON** writes the current tab's graph (nodes, edges, parameters, segment markers and subgraph definitions) to a `.json` file.
- **Load → Import JSON...** replaces the current tab's canvas with the file — open a new tab first to keep your graph. A file written by a newer CodefyUI opens read-only, with a notice.
- **Export → Export Diagram (SVG / PNG)** draws the architecture only — nodes, ports and connections, no parameter values — on a light, document-friendly background.
- **Export → Export as Python** writes a readable, single-file Python program: one function per node (with its parameters inlined as editable literals), flow functions that wire the nodes together in execution order, and a `main()` entry point with a small CLI. Each node function delegates to the same node implementation the canvas uses, so results match what you saw on the canvas. Run it with the Python environment from a compatible CodefyUI installation; it does not need the web server. Use `--help` for device, GraphInput JSON, timeout, and project-asset options.

The same JSON format is what the backend's example graphs use, so an exported graph can also be run headless with the **[CLI Graph Runner](./cli-runner)**.

:::tip
Because graphs are plain JSON, they diff and version-control cleanly. Commit a graph alongside your code to capture an exact, reproducible pipeline.
:::
