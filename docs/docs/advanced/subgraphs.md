---
sidebar_position: 2
title: Subgraphs
description: Collapse a selection into a reusable block you can open, edit, and expand again — with every instance sharing one definition.
---

# Subgraphs

A **subgraph** is a block of your graph collapsed into a single node. Unlike a
preset — which is flattened once, when you drop it — a subgraph keeps a live
definition: every instance of it points at the same block, so editing the
block changes every place you used it.

Subgraphs are **local to the graph file** they live in. The definition travels
in the graph's own `subgraphs` list, so a saved or exported graph is still one
portable artifact with nothing to install alongside it.

## Collapsing a selection

1. Select two or more nodes (Shift+click, or Shift+drag a box around them).
2. Right-click one of them and choose **Collapse to subgraph**.
3. Give the block a name. The default is `Subgraph`; you can rename it later
   from the breadcrumb.

The selection is replaced by one instance node. Every edge that crossed the
selection becomes a **boundary port** on that node, named after the inner port
it stands for. Trigger edges from **Start** are remembered too: one edge into
the block fans back out to exactly the nodes that were triggered before.

Collapse and expand are each a **single undo step** — one Ctrl+Z puts the graph
back exactly as it was, definition included.

### When collapse refuses

Collapse says no rather than producing a graph that reads wrong:

| Refusal | Why |
| --- | --- |
| Fewer than two nodes selected | A block of one node is just the node. |
| A **Start** node is selected | Start marks where the whole graph begins. Inside a reusable block it would mean every instance adds its own entry point. |
| A note is selected | Notes are annotations, not computation. |
| A node between two selected nodes is left out | The block would feed that node *and* be fed by it — a loop on the canvas that the flattened graph does not actually have. The message names the nodes to add to your selection. |

## Editing a block

**Double-click** an instance, or right-click it and select **Enter subgraph**, to open its definition on the canvas. The breadcrumb at the top shows a path such as `Main ▸ MyBlock`. You can drag, connect, delete, edit parameters, and undo changes. To open Node details, select the instance and press `Enter`. Its **Subgraph** tab maps each boundary port to an internal node and port, and also provides an **Enter subgraph** button.

Undo inside a block stays inside it: the block gets its own history while you
are in there, and your outer history is put back when you leave — so you can
never accidentally undo your way out through the boundary.

Leaving a block is itself **one undo step**. Everything you changed inside
lands as a single entry on the outer history, so one Ctrl+Z after you come out
reverts the whole visit and leaves whatever you did *outside* the block
untouched. Entering and leaving without changing anything adds no undo entry at
all.

Click the block's name in the breadcrumb to rename it (not offered on a graph
opened read-only), **Back** to leave one level, or **Main** to jump all the way
out.

Deleting a node inside a block also removes the boundary port it provided, and
the outer edge that named that port goes with it.

## Two instances, one definition

Copy an instance node, or collapse the same block twice, and both nodes point
at the same definition. Edit it through either one and both change — this is
the reuse a flattened preset cannot give you.

Delete the last instance of a block and its definition stops being saved with
the graph, so a file never carries a block nothing on the canvas can reach.
Undo brings both back.

:::note Per-instance parameters are out of scope in v1
Two instances of a subgraph are **identical blocks**. There is no way to give
one instance a different learning rate from another; if you need that, keep
the parameter outside the block and wire it in. Per-instance overrides are a
follow-up.
:::

## Running a graph with subgraphs

Nothing special is required. Before a run the server **inlines** every instance,
exactly as it already inlines presets, so a collapsed graph executes precisely
like the graph it was collapsed from. Collapse, run, expand, run again: same
numbers.

While the run is in flight the instance node shows one aggregate status —
running when the first node inside starts, completed when the last one
finishes, error or interrupted the moment anything inside fails or stops early.

Inner nodes get **namespaced ids** while they run, `<instance>/<node>`, which is
what you see in the Teaching Inspector and in validation messages.

### Validation across the boundary

Because validation inlines instances too, it checks the block's insides:
port types across the boundary, required inputs left unconnected, and — the
one you cannot see on the canvas — a **cycle inside a definition**. Those are
reported with a path naming both sides:

```
Graph contains a cycle: blk1/relu -> blk1/conv -> blk1/relu
  (crosses subgraph instance(s): blk1)
```

Every enclosing block is named, so a loop two boundaries down points at both
the block you can see and the one inside it
(`crosses subgraph instance(s): blk1, blk1/inner`).

A subgraph that contains itself, directly or through another subgraph, is
refused by name before anything runs.

Nesting is allowed up to 10 levels deep, the same budget preset nesting gets.

## What is stored where

In a [project directory](../usage/project-directories.md) the split follows the
same rule as the top level:

- `graphs/<name>.graph.json` — the definitions: nodes, edges, and the interface.
- `layout/<name>.layout.json` — `subgraphPositions`, the positions of the nodes
  inside each block.

So rewiring a block shows up in a review, and dragging inside one does not.

## Exporting to Python

`Export → Python` emits **one function per subgraph instance**:

```python
# ========================== Subgraph functions ==========================


def subgraph_myblock(ctx, results, provided):
    "subgraph 'myblock' - instance 'blk1'."
    # ScalarMultiply -> Print
    results['blk1/mul'] = n03_scalarmultiply(
        ctx,
        tensor=_port(results['src'], 'tensor'),
    )
    results['blk1/p'] = n04_print(
        ctx,
        value=_port(results['blk1/mul'], 'tensor'),
    )


# ============================ Flow functions ============================


def flow_1(ctx, results, provided):
    'Start -> TensorCreate -> ScalarMultiply -> Print'
    start = results['start'] = n01_start(ctx)
    src = results['src'] = n02_tensorcreate(ctx)
    subgraph_myblock(ctx, results, provided)
```

The block structure survives into the exported file instead of dissolving into
a flat run of node calls. Each generated node function also carries a comment
saying which subgraph it came from.

One exception, and the export says so when it applies: a block can only become
a single function if all of its nodes can run back to back. If something
outside the block has to run in the middle of it — because you wired the block
into a node that then feeds it again, or because the block contains a node with
no inputs of its own — the export emits its nodes in their real running order
instead, with a comment naming what got in the way. The script still runs, and
still computes what the canvas computes; it just does not get the extra
function.

## Limitations in v1

- No per-instance parameter overrides (see the note above).
- Subgraphs are local to one graph; there is no shared library of them yet.
- A `GraphInput` / `GraphOutput` node inside a subgraph is not part of the
  graph's published [API contract](../usage/graph-as-a-function.md) — keep
  those at the top level.
- The canvas provides **Bypass** for an instance node through its context menu or `Ctrl/Cmd+B`. Execution then fails validation and names the instance: `Bypass is not supported on subgraph instance(s): blk1`. A bypass can forward a value only between ports declared by a node class, and boundary ports have no such declaration. Remove the bypass from the instance, or enter the block and bypass its internal nodes.
- **Export → Export as Subgraph** creates a **preset**, not a block; both features use the term "subgraph." The export rejects a canvas that contains collapsed blocks and lists their names. A server-side preset stores nodes and edges but cannot store a block definition. Without this validation, the preset would contain a `subgraph:<id>` node without its definition and would not run independently of the source graph. Expand the blocks first by right-clicking each one and selecting **Expand subgraph here**, and then export.
