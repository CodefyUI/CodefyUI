---
sidebar_position: 6
title: Key Bindings
description: Keyboard and mouse shortcuts for the CodefyUI editor.
---

# Key Bindings

Keyboard chords are ignored while you are typing in an input, a textarea or a note.

| Action | Key / gesture |
|--------|---------------|
| Undo | `Ctrl/Cmd` + `Z` |
| Redo | `Ctrl/Cmd` + `Shift` + `Z` / `Ctrl/Cmd` + `Y` |
| Copy selected nodes | `Ctrl/Cmd` + `C` (yields to a text selection on the page) |
| Paste nodes | `Ctrl/Cmd` + `V` (same) |
| Delete selected nodes / edges | `Delete` (`Backspace` does nothing) |
| Multi-select | `Shift` + click |
| Box-select | `Shift` + drag on empty canvas (a plain drag pans) |
| Quick add node / preset | Double-click empty canvas; `Up` / `Down` + `Enter` picks, `Esc` closes |
| Open node details | `Enter` with one node selected; double-click a node; right-click → Open details |
| Double-click exceptions | `SequentialModel` → Model Architecture Editor; subgraph instance → enter the block; preset → Configure Preset; note → edit |
| Bypass / un-bypass selected node(s) | `Ctrl/Cmd` + `B` (when a bypassable node is selected); right-click → Bypass |
| Collapse / expand sidebar | `Ctrl/Cmd` + `B` when nothing bypassable is selected; `Ctrl/Cmd` + `Shift` + `B` always |
| Auto layout (last-used mode) | `Shift` + `L` |
| Save graph (project mode only) | `Ctrl/Cmd` + `S` |
| Show shortcuts overlay | `?` |
| Commit (Source Control message box) | `Ctrl/Cmd` + `Enter` |
| Rename node | Right-click → Rename, or the name field in Node details (`Enter` applies, `Esc` cancels) |
| Duplicate node | Right-click → Duplicate |
| Node details navigation | `Left` / `Right` for the previous / next node; `Esc` closes |
| Confirm / prompt dialogs | `Enter` confirms, `Esc` cancels |
| Rename tab | Double-click the tab; `Enter` applies, `Esc` cancels |
| Sidebar rail | `Up` / `Down` / `Home` / `End` move between tabs; click the open tab's icon to collapse |
| Detach / rewire an edge | Left-drag a connected input port (drop on empty space deletes it; `Shift` / `Ctrl` / `Alt` + drag starts a new connection instead) |
| Edge summary | Click an edge after a run; **View stats** opens Node details → Stats |
| Add note | Right-click empty canvas → Add Text Note / Add Image Note |

:::tip
Press `?` at any time to open the in-app overlay, which lists the most common shortcuts.
:::
