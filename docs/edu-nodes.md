# Edu nodes — educational teaching nodes

CodefyUI ships **Edu nodes**: educational counterparts to the production nodes
that expand a single AI concept into a chain of *visible* intermediate steps,
so a learner can watch data change one operation at a time. They back the
hands-on (`I1`–`I5`) track of the companion AI-Algo textbook.

This document is the authoring reference: how the packs are organized, the type
naming, the visualization and testing standards every Edu node must meet, and a
step-by-step recipe for adding a new one.

## Packs (by direction)

Edu nodes are packaged into three direction packs that map onto the textbook's
hands-on modules and install **cumulatively** as a learner progresses:

| Pack | Hands-on modules | Install |
|------|------------------|---------|
| `foundations` | I1 data representation · I2 classical ML | `cdui plugin install foundations` |
| `deep` | I3 vision · I4 sequence | `cdui plugin install deep` |
| `rl` | I5 reinforcement learning | `cdui plugin install rl` |

The only cross-pack reuse is `Edu-FFN` (introduced by I2's MLP in `foundations`,
reused by I4's Transformer block in `deep`) — harmless because a learner
reaching I4 has already installed `foundations`.

### Node inventory

**`foundations`** — `Edu-ColumnStats`, `Edu-SlidingWindow`, `Edu-TokenEmbedding`,
`Edu-VectorSimilarity`, `Edu-KNN`, `Edu-LinearRegression`, `Edu-LogisticRegression`,
`Edu-SVM`, `Edu-DecisionTree`, `Edu-FFN`.

**`deep`** — `Edu-Conv2d`, `Edu-MaxPool2d`, `Edu-ResBlock`, `Edu-Resample`,
`Edu-DenoiseStep`, `Edu-CrossAttention`, `Edu-Patchify`, `Edu-RNNCell`,
`Edu-LSTMCell`, `Edu-SelfAttention`, `Edu-MultiHeadAttention`, `Edu-LayerNorm`.

**`rl`** — `Edu-PolicyGradient`, `Edu-PPOClip`, `Edu-GRPO`, `Edu-PreferenceLoss`.

## Naming

- The `NODE_NAME` (the type id stored in graphs and shown in the palette) is
  **`Edu-<Concept>`** with a dash, e.g. `Edu-KNN`, `Edu-SelfAttention`.
- Plugin nodes register under a `<plugin_id>:` namespace, so the qualified type
  in a saved graph is e.g. `foundations:Edu-KNN`, `deep:Edu-SelfAttention`,
  `rl:Edu-PolicyGradient` (see `backend/app/core/node_registry.py`).
- The Python **class** name and **file** name stay plain
  (`EduKNNNode` in `edu_knn_node.py`) — only the `NODE_NAME` string carries the
  dash.
- `CATEGORY` places the node in a sidebar group alongside related production
  nodes (`Classical`, `CNN`, `RNN`, `Diffusion`, `Transformer`, `RL`, `Data`,
  `Vision`).

## Visualization standard (the whole point)

Every Edu node MUST make its internals visible. In order of importance:

1. **Verbose `StepRecorder` trace.** When the execution context has
   `verbose=True`, record *every* intermediate quantity — not just inputs and
   outputs. Iterative algorithms (gradient descent, tree building, RNN
   unrolling, diffusion denoising) record a snapshot **per iteration/step** so
   the Inspector's *Steps* tab can be scrubbed. Cap very long per-step traces by
   sampling and say so in the step description.
2. **Display-only output ports** for anything that visualizes well as a tensor:
   attention/similarity matrices (heatmaps), support-vector masks, per-step
   stacks, argmax indices, tree structures. These feed the Inspector's
   heat-coloured `TensorGridView` and the existing viz nodes for free.
3. **Before/after pairing** where the node transforms a tensor, so the
   Inspector's diff colouring highlights what changed.
4. **Custom React viz** only where geometry beats a table — scatter/margin
   plots, the decision-tree diagram, the PPO clip curve, attention heatmaps.
   The frontend maps a bare `NODE_NAME` → renderer in
   `frontend/src/utils/index.ts` (`VIZ_NODE_TYPES`); the generic Inspector is
   always the fallback, so a node is fully usable on its backend + steps alone.

The canonical example to copy is
[`plugins/rl/nodes/edu_policy_gradient_node.py`](../plugins/rl/nodes/edu_policy_gradient_node.py)
— it exposes `softmax → gather → log → baseline → loss` step by step.

## Testing standard

Each node has `backend/tests/test_edu_<name>_node.py` covering:

- **metadata** — `NODE_NAME` (dashed), `CATEGORY`, output port names;
- **numerical correctness** — hand-checked small tensors, and equality against
  the PyTorch reference where one exists (`F.conv2d`, `F.max_pool2d`,
  `F.layer_norm`, `nn.RNNCell`, `nn.LSTMCell`, …);
- **input validation** — shape/dtype/range errors raise with clear messages;
- **verbose steps** — `__steps__` is present and well-formed when
  `context.verbose` is set, and absent otherwise.

Run: `uv run --directory backend pytest tests/ -q`.

## Adding a new Edu node

1. **Create** `plugins/<pack>/nodes/edu_<name>_node.py`. Subclass `BaseNode`
   (`backend/app/core/node_base.py`); set `NODE_NAME = "Edu-<Concept>"`,
   `CATEGORY`, `DESCRIPTION`. Implement `define_inputs`, `define_outputs`,
   `define_params`, and `execute(self, inputs, params, progress_callback=None,
   *, context=None)`.
2. **Record steps.** Gate on
   `verbose = context is not None and getattr(context, "verbose", False)`; build
   a `StepRecorder` (`backend/app/core/step_trace.py`) and, at the end, attach
   `result["__steps__"] = recorder.steps` when any were recorded. Add
   display-only output ports for anything visual.
3. **Stay inside the plugin sandbox.** Import only `torch` /
   `torch.nn.functional` / `math` — the AST validator
   (`backend/app/core/plugin_validator.py`) blocks `os`, `subprocess`, `pickle`,
   bare `torch.load`, dunder escapes, etc.
4. **Write** `backend/tests/test_edu_<name>_node.py` to the standard above;
   import via `from cdui_plugins.<pack>.nodes.edu_<name>_node import Edu<Name>Node`.
5. **Ship an example** under `plugins/<pack>/examples/<lesson>/<Name>/graph.json`
   using `TensorInput` sources and a `Print` sink; it is executed end-to-end by
   `backend/tests/test_chapter_examples.py`.
6. **Register & document.** Add the node to `plugins/<pack>/cdui.plugin.toml`'s
   pack (auto-discovered from `nodes/`), the README *Plugin Packs* table, and
   the inventory above. Run `uv run --directory backend pytest tests/ -q` and,
   if you touched the frontend, `pnpm -C frontend build`.
