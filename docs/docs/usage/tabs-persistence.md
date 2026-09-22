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

**+** at the end of the strip opens an empty tab. Double-click a tab to rename it (`Enter` applies, `Esc` cancels). A running tab shows a dot next to its name. A **Read-only** badge means Save (and plugin writes) are refused — typically because the graph was written by a newer CodefyUI than this one — while renaming and closing still work; a tab a plugin opened says so when you hover it. Every tab has a close button, the last one included. An empty tab closes at once; closing a tab that holds a graph asks first and names its node count, because there is no undo for a closed tab. Closing a running tab asks "This tab is still running. Close it anyway?"; the run itself keeps going on the server and stays in the **Runs** panel. When the server runs on a [project directory](./project-directories), each project keeps its own set of tabs.

### The welcome screen {/* #the-welcome-screen */}

With no tab open, CodefyUI shows a welcome screen in place of the editor. **New blank graph** opens an empty tab, **Browse all templates** opens the [Template Gallery](./examples-gallery#when-the-canvas-is-not-empty), and the cards under **Or start from an example** list the [examples](./examples-gallery); picking one opens it in a new tab.

While the welcome screen is shown:

- the toolbar keeps only the project badge, **Settings**, **?**, the font size and the language; buttons added by plugins come back when a tab opens
- **Settings** leaves out **Recording & Inspection** and **Training Behavior**, which belong to a graph
- the Template Gallery offers only **Open in new tab**
- there is no sidebar and no results panel, so **Graphs** (with its **Import...**), **Source Control** and **Runs** are reached by opening a tab first, with **+** or **New blank graph**

The empty state is saved like any other: a reload returns to the welcome screen. A browser that has never saved any tabs starts on an empty **Tab 1**.

## Automatic saving

All tabs are auto-saved in your browser, so your work is restored when you reload the page. This is local to the browser; it is not synced to the server. Another browser, or another computer, therefore starts with none of them — a [workspace file](#workspace-files) is how you take them along.

A reload restores each tab's graph (nodes, edges, layout, segment markers and subgraph definitions), its name and description, its device, the saved graph it is bound to, its read-only flag and its run settings. It does not restore the undo/redo history, the Execution Log, typed secret values, or the subgraph you were editing inside (you come back to the top level), and tabs a plugin opened temporarily are not saved. A tab whose run was still going re-attaches to that run if the server still has it.

Saving uses **IndexedDB**, with one record per tab. That matters for large graphs: the older `localStorage` backend capped an origin at roughly 5MB, and a graph past that limit simply stopped being saved. IndexedDB has no comparable practical limit, and only the tab you edited is rewritten.

The first time you open a version with IndexedDB saving, whatever `localStorage` last held is copied across automatically — you do not need to do anything. The old `localStorage` copy is left behind (so downgrading still opens the graph it last saw) but stops being updated from then on.

If your browser has no usable IndexedDB — some private-browsing modes, or a sandboxed frame — saving falls back to `localStorage`, with its old size limit.

Three warnings report a storage problem. Each stays on screen until you close it and repeats at most once a minute:

- "Could not save tabs — browser storage is full." The `localStorage` fallback could not store the tabs.
- "Browser storage dropped to a smaller fallback — large graphs may stop saving." An IndexedDB write failed and saving moved to `localStorage`. This one is shown once per session.
- "Browser storage is not working — the tabs shown may be out of date, and new changes may not be saved. …" The saved tabs could not be read when the page opened.

When one of them appears, export a [workspace file](#workspace-files) before you close the page.

## Saving and loading

Saved graphs are stored by the server, not by the browser:

- **File → Save**, and the toolbar's **Save** icon, write the current tab's graph straight back to the graph the tab was opened from, with no prompt. That holds wherever the server runs; a project directory is not required for it. A tab bound to no graph — a new canvas, an import, an opened example — and **Save As...** ask for a name instead, and warn before a name already in the list replaces the graph holding it. Inserting an example into a tab that is bound to a saved graph keeps the binding, so the next Save writes the merged graph back without asking. **File → Clear Canvas** empties the tab after a confirmation.
- The sidebar's **Graphs** tab lists them (searchable, most recently modified first), and the row bound to the current tab is marked. Clicking a row opens that graph in a new tab and binds the tab to the file. The canvas you were working on is never replaced, so there is nothing to confirm and nothing to lose. Clicking a graph that is already open switches to the tab holding it rather than opening it twice.
- The row's menu holds two items, **Rename** and **Delete**. A delete confirms first, then removes the file from disk, and both halves of it in a [project directory](./project-directories). When the renamed or deleted file is one a tab saves back to, that tab follows: it binds to the new name, or to nothing, so the next Save asks rather than quietly recreating what you just deleted. The panel's header has a **Save as...** of its own, so a save lands visibly in the list you are looking at.

## Import / export

You can export any graph to a JSON file and import it back later (or share it):

- **Export → Export as JSON** writes the current tab's graph (nodes, edges, parameters, segment markers and subgraph definitions) to a `.json` file.
- **Import...**, at the foot of the sidebar's **Graphs** tab, takes a graph `.json` or a [workspace file](#workspace-files) and tells them apart by what is inside, not by the file name. The two do opposite things: a graph **replaces** the current tab's canvas — open a new tab first to keep your graph — while a workspace file only **adds** tabs and leaves every open tab alone. A `.json` file that is not a graph is refused with a message instead of emptying the canvas, and a graph written by a newer CodefyUI opens read-only, with a notice. An imported graph is bound to no file, so the first Save asks for a name.
- **Export → Export Diagram (SVG / PNG)** draws the architecture only — nodes, ports and connections, no parameter values — on a light, document-friendly background.
- **Export → Export as Python** writes a readable, single-file Python program: one function per node (with its parameters inlined as editable literals), flow functions that wire the nodes together in execution order, and a `main()` entry point with a small CLI. Each node function delegates to the same node implementation the canvas uses, so results match what you saw on the canvas. Run it with the Python environment from a compatible CodefyUI installation; it does not need the web server. Use `--help` for device, GraphInput JSON, timeout, and project-asset options. A relative file path in a reader node resolves as described in [Relative file paths](./cli-runner#relative-file-paths).

The same JSON format is what the backend's example graphs use, so an exported graph can also be run headless with the **[CLI Graph Runner](./cli-runner)**.

:::tip
Because graphs are plain JSON, they diff and version-control cleanly. Commit a graph alongside your code to capture an exact, reproducible pipeline.
:::

### Workspace files

**Export → Workspace (.cduiworkspace)** writes every open tab into one file, `workspace-YYYY-MM-DD.cduiworkspace`, so a whole session can move to another browser or another computer. **Import...** reads it back: the tabs are added beside the ones already open, no open graph is replaced, and a browser holding only one empty tab ends up with exactly the exported set.

For each tab the file carries its title, its graph and its run settings: Random seed, Deterministic algorithms, Record node outputs, Verbose internals, Persist weights between runs, Capture gradients and Auto-synthesize loss. It also carries which tab was active, and six preferences: Language, Font size, Connection style, Grid snap, Show node tooltips and Node category mode. Each `tabs[i].graph` in the file is an ordinary Export-as-JSON graph. Import applies both: the tab that was active in the file becomes the active tab (the first imported tab if that one was skipped), and the six preferences replace this browser's, so the editor can switch language. A message then says how many tabs were imported and why any were skipped.

What never travels:

- secret parameter values
- anything a plugin stored in the browser, such as the provider API keys an assistant plugin keeps there
- run results and trained weights, which live in the server's memory
- the binding between a tab and a saved graph
- the Settings compute device and the panel layout, which belong to one machine

An imported tab is therefore bound to no saved graph: its first Save asks for a name, and a name already in the list is confirmed before it replaces the graph holding it. In a [project directory](./project-directories) nothing is stamped until that first Save.

Secret values are recognised from the node's definition, so a node whose type the exporting browser has not loaded — its plugin disabled or missing — has no secret the export can recognise; Save and **Export as JSON** have the same limit, so check such tabs before you share a file.

Empty tabs, read-only tabs and tabs a plugin opened temporarily are not exported. When that leaves nothing, Export says "Nothing to export: every tab is empty or read-only."; when it leaves out read-only tabs, it says how many. On import, an entry that is not a graph, or that would be the 33rd open tab, is skipped and the rest still open — an import stops at 32 tabs, and the lone empty tab it replaces is not one of them. A file over 64 MiB, or a workspace file written by a newer CodefyUI, is refused whole. A node type that is not installed here opens as a placeholder and a warning names the missing types; nothing re-links them later, so install the plugin or custom node they come from and import the file again.
