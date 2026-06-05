---
sidebar_position: 5
title: Tabs & Persistence
description: Multi-tab workspaces, automatic localStorage saving, and importing/exporting graphs as JSON.
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

All tabs are auto-saved to your browser's `localStorage`, so your work is restored when you reload the page. This is local to the browser; it is not synced to the server.

## Import / export

You can export any graph to a JSON file and import it back later (or share it):

- **Export** writes the current tab's graph (nodes, edges, parameters, and segment markers) to a `.json` file.
- **Import** loads a `.json` graph into a new tab.

The same JSON format is what the backend's example graphs use, so an exported graph can also be run headless with the **[CLI Graph Runner](./cli-runner)**.

:::tip
Because graphs are plain JSON, they diff and version-control cleanly. Commit a graph alongside your code to capture an exact, reproducible pipeline.
:::
