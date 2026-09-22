---
sidebar_position: 2
title: Your First Graph
description: Build and run a minimal pipeline, and learn why every graph needs a Start node to drive execution.
---

# Your First Graph

This walkthrough builds a tiny pipeline that feeds an explicit tensor through a couple of operations — enough to learn the **Start node** execution model that every CodefyUI graph relies on.

In **Basic** node category mode the sidebar hides `Start`, `Reshape`, `Softmax` and `Print`; add them by double-clicking the canvas and searching (see [The sidebar](./canvas-basics#the-sidebar)).

## 1. Add an input

Drag a **`TensorInput`** node (Data category) onto the canvas. Its `shape` defaults to `1,4,4`, so it holds 16 values. Set its `value_mode` to `explicit`, then type the numbers you want the pipeline to see into the inline grid, or fill it with **Fill 0**, **Fill 1** or **Random** (see [The config panel](./canvas-basics#the-config-panel)).

## 2. Wire up some operations

Connect it through a chain of tensor-op nodes. This one flattens the 16 values into one row and turns them into probabilities:

```
TensorInput → Reshape → Softmax → Print
```

Drag from each output port to the next input port. The edges validate types as you connect.

Set Reshape's `shape` to `1,16`. A new shape must hold the same number of elements as the tensor it receives, and Reshape's default, `-1,784`, needs a multiple of 784, so the run would stop at Reshape. Leave Softmax's `dim` at `-1`: the 16 probabilities then add up to 1.

## 3. Add a Start node

:::warning Every graph needs a Start node
Drag a **`Start`** node onto the canvas and connect its **trigger output** (the diamond handle on the right side) to the first node you want executed — typically the `TensorInput`.

Without a `Start → first-node` trigger edge, the graph is treated as a draft and **Run** rejects it with an error toast: *"No entry points defined. Drag a Start node from the palette and connect it to the node you want to start execution from."* The executable set includes each triggered node, its downstream data flow, any upstream nodes that feed data into that set, and internal roots in any reached preset or subgraph container.
:::

This trigger-based routing is what lets you keep scratch nodes on the canvas without running them, and it enables conditional branches (e.g. a `Switch` node) where only one path executes.

## 4. Run it

Click **Run**. Watch per-node progress stream into the **Execution Log**, and the `Print` node's output appear there too. See **[Running Graphs](./running-graphs)** for what happens during execution.

## 5. Inspect what flowed

**Record node outputs** is on by default (**Settings → Recording & Inspection**), so the run you just made already captured every node's output. Click any node to open the **[Teaching Inspector](./teaching-inspector)** and see the exact tensor — shape, dtype, min/max/mean, and values — at every step.

## Next steps

- Load a real example instead of building from scratch — see the **[Examples Gallery](./examples-gallery)** (e.g. *Train CNN on MNIST*).
- Browse every node you can drop on the canvas in the **[Node Reference](./node-reference)**.
