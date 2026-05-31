# Edu plugins, reorganized by direction

**Status:** approved design — implementation in progress
**Date:** 2026-05-30
**Owner:** treeleaves30760
**Companion textbook:** `AI-Algo-Textbook` (the `I0`–`I5` hands-on track)

> This document is both the **spec** (what we're building and why) and the
> **plan** (the PR-by-PR steps). It will be slimmed into lasting plugin
> documentation once the implementation PRs are open.

---

## 1. Motivation

CodefyUI ships educational ("Edu") nodes that expand one AI concept into a
chain of *visible* small steps, so students can watch data change. Today these
12 nodes are packaged **by textbook chapter** (`plugins/c1` … `plugins/c6`).

The textbook's hands-on track (`I1`–`I5`) is organised **by direction**, not by
the chapter a node first appears in. The chapter packaging therefore produces
awkward cross-chapter installs. From the textbook's own tool page (`I0-0`):

> "I1 install `c1`, but the image sliding-window needs `c3` and embedding needs
> `c4`" · "I2 install `c2`, but the MLP part `EduFFN` lives in `c4`".

A node's *home chapter* ≠ *where the implementation track first uses it*.

We also discovered the custom visualisation components (KNN scatter, attention
heatmaps) silently stopped firing for plugin nodes after the `#26` namespacing
change — see §4.3.

## 2. Goals / non-goals

**Goals**

1. Repackage Edu nodes into **3 direction packs** that map cleanly onto the
   `I`-track and install cumulatively.
2. **Fill the `I`-track coverage gaps** with new Edu nodes so every major
   algorithm in `I1`–`I5` has a step-by-step teaching node.
3. Make **intermediate-state visualisation the headline feature** of every Edu
   node (richer `StepRecorder` traces, more display-only outputs, and bespoke
   visual components for the genuinely geometric cases).
4. Full test coverage; ship behaviour-preserving and additive work as a
   sequence of reviewable PRs, one per direction.

**Non-goals**

- No backwards-compatibility aliases for the old `cN:EduFoo` type ids. There
  are no end users yet (confirmed by owner), so we take a clean break.
- No change to the core engine, execution model, or Inspector framework.
- No `I6` / frontier hands-on pack (the textbook has no `I6`).

## 3. Target architecture

Three builtin packs replace `c1`–`c6`:

| Pack | Covers | Install |
|------|--------|---------|
| **`foundations`** — data representation + classical ML | `I1` + `I2` | `cdui plugin install foundations` |
| **`deep`** — vision + sequence deep models | `I3` + `I4` | `cdui plugin install deep` |
| **`rl`** — reinforcement learning | `I5` | `cdui plugin install rl` |

Install is **cumulative as the student progresses**: `foundations` (for I1–I2)
→ add `deep` (I3–I4) → add `rl` (I5). The only cross-pack reuse is `Edu-FFN`
(introduced by I2's MLP, in `foundations`; reused by I4's Transformer block in
`deep`) — harmless because a student reaching I4 has already installed
`foundations`.

## 4. Naming & migration mechanics

### 4.1 Dash in the node type id

Every `NODE_NAME` gains a dash after `Edu`: `EduKNN` → `Edu-KNN`. Python class
names (`EduKNNNode`) and file names (`edu_knn_node.py`) are **unchanged** — only
the `NODE_NAME` string and anything that stores/looks-up the type id changes.

Verified safe: `node_registry.qualify()` is a bare f-string
`f"{plugin_id}:{node_name}"` with **no character validation**; the plugin
loader's `_py_id()` maps the *plugin id* (not the node name) to a Python module
id, and our plugin ids (`foundations`/`deep`/`rl`) are already valid
identifiers.

### 4.2 Qualified registry keys

Plugin nodes register as `<plugin_id>:<NODE_NAME>` and graphs store that
qualified string. Net effect of this refactor on a saved type id:

`c2:EduKNN` → `foundations:Edu-KNN`.

The `/api/nodes` route emits `node_name = <qualified>` (`routes_nodes.py`), and
the frontend `defMap` is keyed by it, so palette/definition lookups stay
consistent after the rename.

### 4.3 Fix the dormant viz wiring (regression from #26)

`frontend/src/utils/index.ts` looks up the custom React renderer with
`VIZ_NODE_TYPES[nodeType]` where `nodeType` is the **full qualified** type
(`c2:EduKNN`). But `VIZ_NODE_TYPES` is keyed by **bare** names (`EduKNN`), so
the lookup misses and the node falls back to `baseNode`. The KNN scatter and
attention heatmaps have therefore been dark for namespaced plugin nodes.

**Fix (part of PR1):** strip the `<plugin>:` prefix before the lookup and key
`VIZ_NODE_TYPES` by the **bare dashed** name:

```ts
const bare = nodeType.includes(':') ? nodeType.slice(nodeType.lastIndexOf(':') + 1) : nodeType;
// ...
type: VIZ_NODE_TYPES[bare] ?? 'baseNode',
```

```ts
export const VIZ_NODE_TYPES: Record<string, string> = {
  // ...builtin viz keys unchanged...
  'Edu-SelfAttention': 'eduSelfAttentionNode',
  'Edu-MultiHeadAttention': 'eduMultiHeadAttentionNode',
  'Edu-CrossAttention': 'eduCrossAttentionNode',
  'Edu-KNN': 'eduKNNNode',
};
```

This makes every Edu viz fire regardless of which pack ships the node, and is
forward-compatible with future repackaging.

## 5. Node inventory (12 moved + 14 new = 26)

Status legend: **M** = moved/renamed from an old pack · **N** = new node.
"Custom viz" = a bespoke React component beyond the generic Inspector Steps tab.

### 5.1 `foundations` (I1 + I2)

| type id | St | Lesson | Key intermediate states (StepRecorder) | Custom viz |
|---|---|---|---|---|
| `Edu-ColumnStats` | M(c1) | I1-1 | col_sum → means → deviations → squared → stds → range | – |
| `Edu-SlidingWindow` | **N** | I1-2 | per window: receptive field → elementwise × → sum → feature-map cell; switchable kernel preset | sliding-window grid |
| `Edu-TokenEmbedding` | M(c4) | I1-3 | token → id → embedding vector | – |
| `Edu-VectorSimilarity` | **N** | I1-3 | q·kᵢ dot products → norms → (cosine) → similarity matrix | heatmap (reuse) |
| `Edu-KNN` | M(c2) | I2-1 | distances → k nearest → vote | scatter (existing) |
| `Edu-LinearRegression` | M(c2) | I2-1 | normal-eq / GD per step → weights | – |
| `Edu-LogisticRegression` | M(c2) | I2-1 | logits → softmax → cross-entropy; GD step | – |
| `Edu-SVM` | **N** | I2-2 | per epoch: margins, support-vector mask, hinge loss, w/b update | margin scatter |
| `Edu-DecisionTree` | **N** | I2-2 | per node: candidate splits → chosen feature+threshold → impurity/gain → child partitions | tree diagram |
| `Edu-FFN` | M(c4) | I2-3/4, I4-3 | Linear → activation → Linear; post-activation hidden | – |

### 5.2 `deep` (I3 + I4)

| type id | St | Lesson | Key intermediate states | Custom viz |
|---|---|---|---|---|
| `Edu-Conv2d` | **N** | I3-1 | im2col unfold (patch matrix) → weight reshape → matmul → feature map; shape at each step | sliding-window (reuse) |
| `Edu-MaxPool2d` | **N** | I3-1 | per window: values → max → argmax; downsample shape | sliding-window (reuse) |
| `Edu-ResBlock` | M(c3) | I3-2 | GN → SiLU → Conv → (+time emb) → GN → SiLU → Conv → + skip | – |
| `Edu-Resample` | **N** | I3-2 | down/up sample: H×W before → after; skip-concat shapes | – |
| `Edu-DenoiseStep` | **N** | I3-3 | ᾱₜ coeff → predicted x₀ → x₍ₜ₋₁₎; before/after tensors | before/after image |
| `Edu-CrossAttention` | M(c3) | I3-3 | Q(query)/K,V(context) → weights | heatmap (existing) |
| `Edu-Patchify` | M(c6) | bonus/ViT | unfold → permute → flatten into tokens | – |
| `Edu-RNNCell` | **N** | I4-1 | per timestep: Wₓxₜ, W_h h₍ₜ₋₁₎, +b, tanh → hₜ | unrolled timeline |
| `Edu-LSTMCell` | **N** | I4-1 | per timestep: forget/input/output gates, candidate, cell update | gate timeline |
| `Edu-SelfAttention` | M(c4) | I4-2 | Q·Kᵀ/√d → softmax → ×V → weights | heatmap (existing) |
| `Edu-MultiHeadAttention` | M(c4) | I4-2/3 | per-head weights, W_o mix | heatmap (existing) |
| `Edu-LayerNorm` | **N** *(optional)* | I4-3 | mean → var → normalize → scale·shift | – |

### 5.3 `rl` (I5)

| type id | St | Lesson | Key intermediate states | Custom viz |
|---|---|---|---|---|
| `Edu-PolicyGradient` | M(c5) | I5-1 | softmax → gather → log → baseline → loss | – |
| `Edu-PPOClip` | **N** | I5 (C5-2) | log-ratio → ratio → clip(1±ε) → min(ratio·A, clip·A) → loss | clip curve |
| `Edu-GRPO` | **N** | I5 (C5-4) | group mean/std → normalized advantage → loss | group bars |
| `Edu-PreferenceLoss` | **N** *(optional)* | I5 (C5-3) | r_chosen − r_rejected → σ(β·Δ) → −log p | – |

> The textbook currently only has `I5-0`/`I5-1` (policy gradient). The PPO /
> GRPO / RLHF nodes are forward-looking tools for `C5-2/3/4`; `Edu-PreferenceLoss`
> and `Edu-LayerNorm` are explicitly optional and may be deferred.

## 6. Visualisation standard (every Edu node)

This is the headline requirement. For each node:

1. **Verbose `StepRecorder` trace** that records *every* intermediate quantity
   in the "Key intermediate states" column — not just inputs/outputs. Iterative
   algorithms (GD, tree building, RNN unrolling, denoising) record a snapshot
   **per iteration/step** so the Inspector's Steps tab can be scrubbed.
2. **Display-only output ports** for anything that visualises well as a tensor
   (matrices for heatmaps, per-step stacks, masks, shapes). These feed the
   Inspector's heat-coloured `TensorGridView` and existing viz nodes for free.
3. **Before/after pairing** where a node transforms a tensor, so the Inspector's
   diff colouring highlights what changed.
4. **Custom React viz** only where geometry beats a table (see "Custom viz"
   column): scatter/margin plots, the decision-tree diagram, the PPO clip curve,
   sliding-window and gate/timeline heatmaps. Generic Inspector is the fallback
   and is always available, so a node is shippable on its backend + steps alone.

## 7. Testing standard

Each node gets `backend/tests/test_edu_<name>_node.py` mirroring the existing
style:

- **metadata** test: `NODE_NAME` (dashed), `CATEGORY`, output port names.
- **numerical correctness**: hand-checked small tensors; compare against a
  reference (`torch.nn` builtin where one exists, e.g. Conv2d/LayerNorm/LSTM).
- **input validation**: shape/dtype/range errors raise with clear messages.
- **verbose steps**: `__steps__` present & well-formed when `context.verbose`.

Run: `uv run --directory backend pytest tests/ -q`. Green baseline before this
work: 223 edu/plugin/namespacing tests, full suite ~1133.

## 8. PR1 migration touch-list (behaviour-preserving)

Derived from a full reference sweep. Every item below must change in lock-step:

**Plugins**
- Create `plugins/{foundations,deep,rl}/cdui.plugin.toml` (+ `nodes/`,
  `examples/`); redistribute `[lessons]` chapters/lessons.
- `git mv` the 12 node `.py` files into the right pack; change each `NODE_NAME`
  to the dashed form.
- `plugins/registry.json`: replace the 6 `cN` catalog entries with 3.
- Delete `plugins/c1`…`c6`.
- Rewrite every `plugins/*/examples/**/graph.json` `"type"` from `cN:EduFoo`
  → `<pack>:Edu-Foo` (inventory: KNN-from-Scratch, Linear-Logistic-Compare,
  Column-Stats-101, Cross-Attention-101, Mini-UNet-Expanded, Self-Attention-101,
  Multi-Head-Causal, Co-Reference-Attention, Transformer-Block-Assembled,
  LLM-Inference-Pipeline, Policy-Gradient-101, Patchify-101, ViT-Full-Forward,
  VLM-Cross-Modal-Attention, …).

**Backend**
- `backend/tests/conftest.py`: the lockfile dict + the two
  `("c1".."c6")` discovery loops → `("foundations","deep","rl")`.
- `backend/app/main.py`: startup plugin discovery loop / default list.
- `backend/tests/test_plugin_api.py`, `test_plugin_cli.py`,
  `test_plugin_enable_disable.py`, `test_plugin_uninstall_builtin.py`,
  `test_plugin_loader.py`: plugin-id and node-name assertions.
- `backend/tests/test_edu_*.py` (12): import paths `cdui_plugins.cN…` →
  `cdui_plugins.<pack>…`; `NODE_NAME` assertions to dashed form.
- `backend/tests/test_node_namespacing.py`: the illustrative `c2:EduKNN`
  literals → `foundations:Edu-KNN` (keep coverage of the qualify contract).

**Frontend**
- `frontend/src/utils/index.ts`: namespace-strip fix + dashed `VIZ_NODE_TYPES`
  keys (§4.3).
- `frontend/src/components/Canvas/FlowCanvas.tsx`: confirm `nodeTypes`
  registration of `eduKNNNode` etc. unchanged.
- `frontend/src/i18n/nodeLocales/zh-TW.ts`: add dashed-name entries (additive;
  currently no Edu entries — non-blocking).

**Scripts / docs**
- `scripts/dev.py`, `scripts/plugins.py`: default builtin-pack list / help text.
- `README.md`: Plugin Packs table, BREAKING note, `cdui plugin install` example.
- Node-count references (a prior commit "sync node counts" — re-check totals).

**Exit criteria:** `pytest` fully green; `pnpm -C frontend build` (tsc) clean.

## 9. PR sequence (stacked branches)

1. **PR1 `refactor/edu-plugins-reorg`** (base `main`): §8 in full. No new
   algorithms; behaviour-preserving; tests green. Carries this spec doc.
2. **PR2 `feat/edu-foundations-nodes`** (base PR1): `Edu-SlidingWindow`,
   `Edu-VectorSimilarity`, `Edu-SVM`, `Edu-DecisionTree` + steps + tests +
   example graphs + custom viz where listed.
3. **PR3 `feat/edu-deep-nodes`** (base PR2): `Edu-Conv2d`, `Edu-MaxPool2d`,
   `Edu-Resample`, `Edu-DenoiseStep`, `Edu-RNNCell`, `Edu-LSTMCell`,
   `Edu-LayerNorm`(opt).
4. **PR4 `feat/edu-rl-nodes`** (base PR3): `Edu-PPOClip`, `Edu-GRPO`,
   `Edu-PreferenceLoss`(opt).
5. **PR5 (textbook repo)**: `content/I0/I0-0.mdx` install table + each `Ix-0`
   overview → 3-pack cumulative scheme; visual-verify per textbook CLAUDE.md.

Each CodefyUI PR: branch → tests green → `gh pr create`. Stacked PRs note their
base; GitHub retargets to `main` as earlier PRs merge.

## 10. Risks & open items

- **Stacked-PR review load.** Mitigated by keeping PR1 purely mechanical so the
  creative PRs diff cleanly.
- **Custom viz scope.** React components for SVM/DecisionTree/PPO/timelines are
  the largest unknown; backend + StepRecorder always ships first so each node is
  functional and inspectable even if a bespoke component slips to a follow-up.
- **`torch` reference parity.** New nodes that re-implement a `torch.nn` op
  (Conv2d, LayerNorm, LSTM) assert equality against the builtin in tests.
- **AST plugin validator.** New nodes must use only `torch`/`torch.nn.functional`
  and avoid the blocked modules/dunders — existing Edu nodes already comply.

## 11. Final step (per owner request)

After the implementation PRs are open, **delete this working plan** and fold its
durable parts into proper documentation: per-pack `README`/manifest descriptions,
the main `README.md` Plugin Packs table, and a short "authoring an Edu node"
guide capturing the §6 visualisation standard and §7 testing standard.
