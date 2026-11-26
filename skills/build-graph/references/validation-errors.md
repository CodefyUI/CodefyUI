# Validation codes and runtime errors

`POST /api/graph/validate` answers `{valid, errors, issues, warnings}`: `errors` are the
messages, `issues` the same findings in order with a `code` and the fields they name
(`node_id`, `port`, ...), and `warnings` findings that do not stop a run. The codes below are the ones
`backend/app/core/graph_engine.py` emits.

## Structure

| Code | Message starts with | Fix |
| --- | --- | --- |
| `no_entry_points` | "Graph has no entry points" | Add a `Start` node and connect its `trigger` |
| `trigger_source_missing` | "A trigger edge comes from node ..." | A trigger edge from a deleted node; remove it |
| `trigger_target_missing` | "A trigger edge goes to node ..." | A trigger edge to a deleted node; remove it |
| `missing_input` | "Missing required input '...'" | Connect the input. "(input dropped because '...' is bypassed)" means a bypassed node used to feed it |
| `type_mismatch` | "Type mismatch: A.out (T1) -> B.in (T2)" | Wire from an output of a compatible type (see port-types.md) |
| `multiple_sources` | "Input '...' on node ... has N wires" | Keep one wire, or insert a `Switch` |
| `invalid_input_port` / `invalid_output_port` | "Invalid input port '...'" | The handle name is wrong; read the node's schema |
| `unknown_node_type` | "Unknown node type: ..." | Wrong name, a plugin that is not enabled, or a missing namespace (`deep:...`) |
| `unknown_preset` | "Unknown preset: ..." | The preset is not installed on this server |
| `cycle` | "Graph contains a cycle" | Remove the edge that closes the loop |

## Params

| Code | Fix |
| --- | --- |
| `param_not_an_option` | A `select` param holds a value not in its `options` |
| `param_not_number` | An `int`/`float` param holds text |
| `param_below_min` / `param_above_max` | Outside `min_value`..`max_value` |

## Presets and blocks

| Code | Meaning |
| --- | --- |
| `preset_triggered_empty`, `subgraph_triggered_empty` | A trigger points at a preset or block with nothing inside to start |
| `preset_triggered_all_fed` | A trigger points at a preset whose every inner node is fed by another inner node, so nothing inside can start |
| `preset_input_not_exposed`, `preset_output_not_exposed` | A wire uses a preset port that the preset does not expose |

## Warnings (the graph still runs)

- `shared_model_training` - one `MODEL` output is trained by several training nodes. They
  share one set of weights and each continues from where the others left off. To compare
  training setups, give each arm its own model node.

## Errors only a run finds

Validation checks structure, not shapes. A run can still fail; the editor rewrites the
common ones into a sentence:

- "Size mismatch: this layer received N features but is configured for M" -
  `Linear.in_features` does not match the flattened input. Set it to N.
- "Channel mismatch: ..." - `Conv2d.in_channels` does not match the incoming channels.
- "Cannot reshape to ..." - `Reshape.shape` does not fit the tensor's element count.
- "Connect a 'tensor' input to this node." - an optional-looking input the node needs.
- "This node needs the ... pack. Install it from the Package Center." - an optional pack.
