---
name: build-graph
description: Build or edit a CodefyUI node graph on the canvas - add nodes, wire typed ports, connect the Start trigger, set parameters, lay out, and fix validation errors. Use when the user asks to create, extend, rewire or repair a graph, or when validation reports missing inputs, type mismatches or no entry points.
license: AGPL-3.0-only
metadata:
  codefyui-skill-version: "1"
---

# Build a CodefyUI graph

A graph is a set of **nodes** joined by **edges**. Every node has a type (`Conv2d`,
`TrainingLoop`, `foundations:Edu-KNN` for a plugin node), typed **input ports**, typed
**output ports** and **params**. The server's node catalog (`GET /api/nodes`) is the only
source of truth for those names: never guess a port or a param, look it up first.

## The rules the editor enforces

1. **An edge goes from an output port to an input port of a compatible type.** Types are
   `TENSOR`, `MODEL`, `DATASET`, `DATALOADER`, `OPTIMIZER`, `LOSS_FN`, `SCALAR`, `STRING`,
   `IMAGE`, `LIST`, `TRANSFORM`, `ANY` and `TRIGGER`. A type connects to itself and to an
   `ANY` input; an `IMAGE` output also feeds a `TENSOR` input; an `ANY` output feeds any
   input except a trigger. See [port types](references/port-types.md).
2. **An input takes one source.** Connecting a second wire to an input replaces the first.
   One output may feed any number of inputs. To choose between sources, wire them into a
   `Switch` (`input_0`..`input_3`, picked by `selector`).
3. **Every input not marked optional must be connected** before the graph can run.
4. **A runnable graph needs a `Start` node.** `Start` has one output, `trigger`. Connect it
   with `source_handle: "trigger"` to the node where execution should begin (the editor
   stores that as a dashed trigger edge into the hidden `__trigger` handle every card has).
   One `Start` may trigger several nodes.
5. **A trigger only marks where execution starts.** Anything upstream that feeds data into
   a node that runs is pulled in and runs too, triggered or not. Disconnecting the data
   edge, not removing a trigger, is what takes a node out of a run.
6. **No cycles.** Data flows one way. Loops over items are `Map` (with a subgraph) and
   `Reduce`; training loops are nodes (`TrainingLoop`, `RLTrainingLoop`,
   `DiffusionTrainingLoop`).
7. **Params respect their schema:** `select` params take one of their `options`; `int` and
   `float` params stay within `min_value`..`max_value`. `secret` params are never written
   into saved graphs; do not invent or echo their values.

## Workflow

1. **Plan** the nodes and wires in a sentence or two before touching the canvas.
2. **Read the schemas** of every node type you will add or rewire (exact port names,
   which inputs are optional, param names, defaults, options and ranges).
3. **Build in small batches** of graph operations: `add_node` (give each a `ref` so later
   operations in the same batch can name it), `connect`, `set_params`. Connect the
   `Start` trigger as you go.
4. **Lay out** with one `auto_layout` at the end of the structural edits (the same as
   `Shift+L`): it arranges left to right, puts the section a skip connection bypasses
   below it, and fits the view. Do not lay out after every small batch.
5. **Validate** (`POST /api/graph/validate`, or your agent's validate tool) and fix what
   it reports. Validation is structural; it does not prove the graph will run or train.
6. **Run only when asked.** Running executes real code: it can download datasets, write
   files and use the GPU.

When an operation in a batch fails, the others in that batch still applied. Re-send only
the corrected failing operation; re-sending the whole batch duplicates nodes.

## Fixing what validation reports

| Code | What it means | Fix |
| --- | --- | --- |
| `no_entry_points` | No `Start` trigger anywhere | Add `Start`, connect `trigger` to the first node(s) |
| `missing_input` | A required input has no wire | Connect an output of the right type to it |
| `type_mismatch` | The wire joins incompatible types | Rewire from an output of the target's type |
| `multiple_sources` | One input has several wires (old file) | Keep one, or put a `Switch` in front |
| `unknown_node_type` | The type is not in the catalog | Check the spelling, the plugin namespace, or the pack |
| `param_not_an_option` | A `select` value outside its options | Use one of the listed options |
| `param_below_min` / `param_above_max` | A number outside its range | Move it into range |
| `cycle` | Wires form a loop | Remove the edge that closes it |

More codes, and the runtime errors a run can still hit (shape and channel mismatches),
are in [validation errors](references/validation-errors.md).

## Good habits

- Build the data path first (data source -> model -> loss -> optimizer -> loop), then the
  evaluation and reporting branch (`EvaluateModel`, `Print`, `Visualize`).
- Size layers from the data: `Linear.in_features` must equal the flattened width it
  receives, `Conv2d.in_channels` the channels it receives. When unsure, add a `Print` or
  run once and read the error; the editor rewrites these into plain sentences.
- Prefer an existing preset or a shipped example when one fits; the Templates gallery
  holds dozens of runnable graphs.
- Do not remove or rewire nodes the user built unless the request needs it, and say so
  when you do.
- `clear_graph` empties the canvas the user has open; use it only when asked to start over.
- Inside an open block (subgraph) the operations apply to the block, not the top level.
