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

## Automatic saving

All tabs are auto-saved in your browser, so your work is restored when you reload the page. This is local to the browser; it is not synced to the server.

Saving uses **IndexedDB**, with one record per tab. That matters for large graphs: the older `localStorage` backend capped an origin at roughly 5MB, and a graph past that limit simply stopped being saved. IndexedDB has no comparable practical limit, and only the tab you edited is rewritten.

The first time you open a version with IndexedDB saving, whatever `localStorage` last held is copied across automatically — you do not need to do anything. The old `localStorage` copy is left behind (so downgrading still opens the graph it last saw) but stops being updated from then on.

If your browser has no usable IndexedDB — some private-browsing modes, or a sandboxed frame — saving falls back to `localStorage`, with its old size limit and its "storage is full" warning.

## Import / export

You can export any graph to a JSON file and import it back later (or share it):

- **Export** writes the current tab's graph (nodes, edges, parameters, and segment markers) to a `.json` file.
- **Import** loads a `.json` graph into a new tab.
- **Export as Python** writes a readable, single-file Python program: one function per node (with its parameters inlined as editable literals), flow functions that wire the nodes together in execution order, and a `main()` entry point with a small CLI. Each node function delegates to the same node implementation the canvas uses, so results match what you saw on the canvas. Run it with the Python environment from a compatible CodefyUI installation; it does not need the web server. Use `--help` for device, GraphInput JSON, timeout, and project-asset options.

The same JSON format is what the backend's example graphs use, so an exported graph can also be run headless with the **[CLI Graph Runner](./cli-runner)**.

:::tip
Because graphs are plain JSON, they diff and version-control cleanly. Commit a graph alongside your code to capture an exact, reproducible pipeline.
:::
