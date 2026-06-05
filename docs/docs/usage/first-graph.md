---
sidebar_position: 2
title: Your First Graph
description: Build and run a minimal pipeline, and learn why every graph needs a Start node to drive execution.
---

# Your First Graph

This walkthrough builds a tiny pipeline that feeds an explicit tensor through a couple of operations — enough to learn the **Start node** execution model that every CodefyUI graph relies on.

## 1. Add an input

Drag a **`TensorInput`** node (Data category) onto the canvas. Set its `value_mode` to `explicit` and fill the inline grid editor with the numbers you want the pipeline to see.

## 2. Wire up some operations

Connect it through any chain of tensor-op nodes, for example:

```
TensorInput → Reshape → Softmax → Print
```

Drag from each output port to the next input port. The edges validate types as you connect.

## 3. Add a Start node

:::warning Every graph needs a Start node
Drag a **`Start`** node onto the canvas and connect its **trigger output** (the diamond handle on the right side) to the first node you want executed — typically the `TensorInput`.

Without a `Start → first-node` trigger edge, the graph is treated as a draft and **Run** rejects it with a *"No start node defined"* toast. **Only nodes reachable from a Start are executed.**
:::

This trigger-based routing is what lets you keep scratch nodes on the canvas without running them, and it enables conditional branches (e.g. a `Switch` node) where only one path executes.

## 4. Run it

Click **Run**. Watch per-node progress stream into the **Execution Log**, and the `Print` node's output appear there too. See **[Running Graphs](./running-graphs)** for what happens during execution.

## 5. Inspect what flowed

Open the **Settings** popover, switch **Record outputs** ON, and run again. Now click any node to open the **[Teaching Inspector](./teaching-inspector)** and see the exact tensor — shape, dtype, min/max/mean, and values — at every step.

## Next steps

- Load a real example instead of building from scratch — see the **[Examples Gallery](./examples-gallery)** (e.g. *Train CNN on MNIST*).
- Browse every node you can drop on the canvas in the **[Node Reference](./node-reference)**.
