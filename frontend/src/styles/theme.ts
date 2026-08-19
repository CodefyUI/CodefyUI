/**
 * TypeScript mirror of the semantic palettes in `styles/tokens.css`.
 *
 * `tokens.css` is the source of truth for every colour in the app. These
 * values exist here only because canvas/SVG code has to *compute* with them
 * (mixing a header tint, stroking an edge) and cannot read a CSS variable.
 * `theme.test.ts` parses `tokens.css` and asserts every entry below matches
 * its variable, so the two cannot drift.
 *
 * For plain styling, use the CSS variable — do not import from here.
 */

export const CATEGORY_COLORS: Record<string, string> = {
  CNN: '#4caf50',
  RNN: '#2397f3',
  Transformer: '#c279ce',
  LLM: '#a78bfa',
  Diffusion: '#ee60a6',
  Classical: '#f59e0b',
  RL: '#ff9800',
  Data: '#00bcd4',
  Training: '#f66358',
  IO: '#a78f86',
  'Data Flow': '#ff6f00',
  Utility: '#8097a2',
  Normalization: '#26a69a',
  'Tensor Operations': '#838fcf',
};

/** CSS variable name for each category, so components can prefer `var()`. */
export const CATEGORY_VARS: Record<string, string> = {
  CNN: '--cat-cnn',
  RNN: '--cat-rnn',
  Transformer: '--cat-transformer',
  LLM: '--cat-llm',
  Diffusion: '--cat-diffusion',
  Classical: '--cat-classical',
  RL: '--cat-rl',
  Data: '--cat-data',
  Training: '--cat-training',
  IO: '--cat-io',
  'Data Flow': '--cat-dataflow',
  Utility: '--cat-utility',
  Normalization: '--cat-normalization',
  'Tensor Operations': '--cat-tensorops',
};

/**
 * The same fourteen categories, darkened for a white page.
 *
 * `CATEGORY_COLORS` is tuned for the app's near-black canvas; on the white
 * card the SVG exporter draws by default, those hues measured 2.15:1 to
 * 3.09:1 as a border and every one of them failed the 4.5:1 the exporter also
 * needs from them as the card's title text (core#227). Each value below is its
 * counterpart above converted to OKLCH, hue held, chroma pushed to the gamut
 * edge and lightness lowered until it clears the bar — so the hue a reader
 * knows from the app survives while the measurement changes. Drift is at most
 * 1.8 degrees of CIE Lab hue angle, which is what the contrast gate measures.
 *
 * Mirrors the `--diagram-light-*` tokens; `theme.test.ts` asserts the match,
 * and `scripts/check-contrast.mjs` re-derives every measurement above.
 */
export const CATEGORY_COLORS_ON_LIGHT: Record<string, string> = {
  CNN: '#1a8626',
  RNN: '#0077c8',
  Transformer: '#9f58ab',
  LLM: '#7f61cc',
  Diffusion: '#c93d86',
  Classical: '#7e4e00',
  RL: '#a96300',
  Data: '#018091',
  Training: '#d14039',
  IO: '#866f66',
  'Data Flow': '#ad4900',
  Utility: '#617782',
  Normalization: '#008278',
  'Tensor Operations': '#6570ad',
};

/** `--diagram-light-*` variable name for each category. */
export const CATEGORY_LIGHT_VARS: Record<string, string> = {
  CNN: '--diagram-light-cnn',
  RNN: '--diagram-light-rnn',
  Transformer: '--diagram-light-transformer',
  LLM: '--diagram-light-llm',
  Diffusion: '--diagram-light-diffusion',
  Classical: '--diagram-light-classical',
  RL: '--diagram-light-rl',
  Data: '--diagram-light-data',
  Training: '--diagram-light-training',
  IO: '--diagram-light-io',
  'Data Flow': '--diagram-light-dataflow',
  Utility: '--diagram-light-utility',
  Normalization: '--diagram-light-normalization',
  'Tensor Operations': '--diagram-light-tensorops',
};

/**
 * Port/wire hues darkened for a white page, same derivation as
 * `CATEGORY_COLORS_ON_LIGHT`. A wire has to be visible as a line before its
 * colour can mean anything, and LIST measured 1.51:1 on white. Held to the
 * 3:1 graphic bar rather than 4.5:1 — the hue is never drawn as text, and the
 * type it stands for is spelled out in bold beside both of its ports. Five of
 * the twelve already cleared 3:1 and keep their canvas value.
 *
 * Keyed by the same uppercase data-type names as `DATA_TYPE_COLORS`.
 */
export const DATA_TYPE_COLORS_ON_LIGHT: Record<string, string> = {
  TENSOR: '#40a445',
  MODEL: '#2095f2',
  DATASET: '#d27c00',
  DATALOADER: '#9c27b0',
  OPTIMIZER: '#f44336',
  LOSS_FN: '#e91e63',
  SCALAR: '#029fb3',
  STRING: '#6aa01f',
  IMAGE: '#ff5722',
  LIST: '#8d9800',
  ANY: '#919191',
  // Deliberately NOT lightened alongside the canvas TRANSFORM (#197 item 5).
  // This palette is drawn on a white page and held to 3:1 there; a lighter
  // amber measures 2.6:1 or worse. Same hue as the canvas value (83 vs 91
  // degrees in Lab), darkened — which is what a light-export colour is.
  TRANSFORM: '#b78901',
};

/** `--diagram-light-type-*` variable name for each data type. */
export const DATA_TYPE_LIGHT_VARS: Record<string, string> = {
  TENSOR: '--diagram-light-type-tensor',
  MODEL: '--diagram-light-type-model',
  DATASET: '--diagram-light-type-dataset',
  DATALOADER: '--diagram-light-type-dataloader',
  OPTIMIZER: '--diagram-light-type-optimizer',
  LOSS_FN: '--diagram-light-type-loss-fn',
  SCALAR: '--diagram-light-type-scalar',
  STRING: '--diagram-light-type-string',
  IMAGE: '--diagram-light-type-image',
  LIST: '--diagram-light-type-list',
  ANY: '--diagram-light-type-any',
  TRANSFORM: '--diagram-light-type-transform',
};

/**
 * Layer-type palette for the layers editor (core#228).
 *
 * Its own scheme, deliberately: the layers editor paints kinds of torch layer,
 * not node categories, and the two disagree about what a hue means (purple is
 * normalisation here, Transformer on the canvas). Where the *hue* is the same
 * idea — "the app's blue", "the app's red" — the value is the same value, so a
 * reader is not shown two different reds on two surfaces.
 *
 * This lives in TypeScript for the same reason `CATEGORY_COLORS` does: the
 * editor's nodes are React Flow cards whose header fill is computed with
 * `mixColor`, and its serializer has to stamp a colour into node data. Mirrors
 * the `--layer-*` tokens; `theme.test.ts` asserts the match and
 * `scripts/check-contrast.mjs` re-derives every measurement.
 */
export const LAYER_TYPE_COLORS: Record<string, string> = {
  Input: '#4caf50',
  Convolution: '#4caf50',
  Normalization: '#c279ce',
  Pooling: '#2397f3',
  Regularization: '#ff9800',
  Linear: '#00bcd4',
  Utility: '#8097a2',
  Activation: '#f66358',
  Merge: '#ff9800',
  Recurrent: '#2397f3',
  Attention: '#c279ce',
  Output: '#f66358',
  Unknown: '#f66358',
};

/** `--layer-*` variable name for each layer category. */
export const LAYER_TYPE_VARS: Record<string, string> = {
  Input: '--layer-input',
  Convolution: '--layer-convolution',
  Normalization: '--layer-normalization',
  Pooling: '--layer-pooling',
  Regularization: '--layer-regularization',
  Linear: '--layer-linear',
  Utility: '--layer-utility',
  Activation: '--layer-activation',
  Merge: '--layer-merge',
  Recurrent: '--layer-recurrent',
  Attention: '--layer-attention',
  Output: '--layer-output',
  Unknown: '--layer-unknown',
};

/** `--type-*` variable name for each data type (the canvas / dark values). */
export const DATA_TYPE_VARS: Record<string, string> = {
  TENSOR: '--type-tensor',
  MODEL: '--type-model',
  DATASET: '--type-dataset',
  DATALOADER: '--type-dataloader',
  OPTIMIZER: '--type-optimizer',
  LOSS_FN: '--type-loss-fn',
  SCALAR: '--type-scalar',
  STRING: '--type-string',
  IMAGE: '--type-image',
  LIST: '--type-list',
  ANY: '--type-any',
  TRANSFORM: '--type-transform',
};

/**
 * Non-category chrome for each SVG export theme — the page, the card it draws
 * nodes on, the ink for port labels, the default wire, and the two nodes that
 * carry a brand rather than a category (preset gold, Start green).
 *
 * Mirrors the `--diagram-light-*` / `--diagram-dark-*` tokens. `exportDiagram`
 * assembles these into `DIAGRAM_THEMES`; they live here so `theme.test.ts` can
 * hold every one of them against `tokens.css`.
 */
export const DIAGRAM_CHROME = {
  light: {
    page: '#ffffff',
    card: '#ffffff',
    ink: '#0f172a',
    wire: '#64748b',
    preset: '#956e00',
    start: '#00712f',
    startFill: '#dcfce7',
  },
  dark: {
    page: '#0a0a0a',
    card: '#1e1e1e',
    ink: '#e5e7eb',
    wire: '#888888',
    preset: '#d4a017',
    start: '#22c55e',
    startFill: '#1f3c2a',
  },
} as const;

/** `--diagram-<theme>-<role>` variable name for each entry of {@link DIAGRAM_CHROME}. */
export const DIAGRAM_CHROME_VARS: Record<keyof typeof DIAGRAM_CHROME.light, string> = {
  page: 'page',
  card: 'card',
  ink: 'ink',
  wire: 'wire',
  preset: 'preset',
  start: 'start',
  startFill: 'start-fill',
};

/**
 * How much of a category hue is mixed into `--surface-raised` to produce a
 * node's header fill.
 *
 * Node headers used to be painted with the raw category hue and titled in
 * `#eeeeee`, which measured 1.85:1 to 2.69:1 on twelve of the fourteen
 * categories — the node title, on every node, in every graph, was below half
 * the AA minimum. Tinting the hue into the surface instead keeps the colour
 * coding while putting every title between 9.2:1 and 11.7:1.
 */
export const NODE_HEADER_TINT = 0.18;

/** Mix `color` into `base` by `amount` (0..1) and return a hex string. */
export function mixColor(base: string, color: string, amount: number): string {
  const parse = (h: string): [number, number, number] => {
    let s = h.replace('#', '');
    if (s.length === 3)
      s = s
        .split('')
        .map((c) => c + c)
        .join('');
    return [0, 2, 4].map((i) => parseInt(s.slice(i, i + 2), 16)) as [
      number,
      number,
      number,
    ];
  };
  const [br, bg, bb] = parse(base);
  const [cr, cg, cb] = parse(color);
  const ch = (a: number, b: number) =>
    Math.round(a + (b - a) * amount)
      .toString(16)
      .padStart(2, '0');
  return `#${ch(br, cr)}${ch(bg, cg)}${ch(bb, cb)}`;
}

/**
 * Token chip palette — soft pastels at ~60% saturation, tuned for the dark IDE
 * surface. Cycled by token index in the tokenizer visualization. Adjacent chips
 * remain distinguishable without jarring against #121212.
 */
export const TOKEN_COLORS: readonly string[] = [
  '#7DD3FC', // sky
  '#FCA5A5', // rose
  '#86EFAC', // mint
  '#FCD34D', // amber
  '#C4B5FD', // lavender
  '#FDA4AF', // pink
  '#6EE7B7', // emerald
  '#FDBA74', // peach
  '#A5B4FC', // indigo
  '#67E8F9', // cyan
  '#F0ABFC', // fuchsia
  '#BEF264', // lime
] as const;

export function getTokenColor(index: number): string {
  return TOKEN_COLORS[index % TOKEN_COLORS.length];
}

/**
 * Restrained material-style hue per builtin EXAMPLE category. Unknown
 * categories (plugin-defined folders) fall back to `EXAMPLE_CATEGORY_FALLBACK`.
 * Shared by the empty-canvas gallery and the sidebar's Templates tab so the
 * same example wears the same colour in both.
 */
export const EXAMPLE_CATEGORY_COLORS: Record<string, string> = {
  Usage_Example: '#4caf50',
  Model_Architecture: '#2397f3',
  Classical: '#26a69a',
  LLM: '#c279ce',
  Diffusion: '#ef6292',
  Transformer: '#26c6da',
  RNN: '#838fcf',
  RL: '#f16663',
  Stats: '#a286d3',
};

export const EXAMPLE_CATEGORY_FALLBACK = '#ff9800';

export const DIFFICULTY_COLORS: Record<string, string> = {
  beginner: '#4caf50',
  intermediate: '#ff9800',
  advanced: '#f66358',
};

export const STATUS_COLORS: Record<string, string> = {
  running: '#ffc107',
  completed: '#4caf50',
  error: '#ff6b63',
  cached: '#2397f3',
  skipped: '#9e9e9e',
  // Was #444, which measured 1.86:1 against the panel it sits on — an
  // indicator you could not actually see. Neutral enough to still read as
  // "nothing is happening", light enough to be visible.
  idle: '#8593a3',
  // core#122: stopped part-way with partial results. Deeper amber than
  // `running` so the two are not confused at a glance, and deliberately not
  // red — an interruption is what the user asked for, not a failure.
  interrupted: '#ff8f00',
};

/**
 * Control-flow green — trigger edges and the Start node, as opposed to the
 * blue data wires.
 *
 * In the app itself use `var(--flow-trigger)` / `var(--flow-trigger-deep)`.
 * This mirror exists for SVG *export*, where the produced file is standalone
 * and a CSS variable would have nothing to resolve against. `theme.test.ts`
 * asserts it matches tokens.css.
 */
export const FLOW_COLORS = {
  trigger: '#22c55e',
  triggerDeep: '#16a34a',
} as const;

/**
 * The gold that marks a preset / reusable-subgraph node, as opposed to a
 * category hue. Same reason as `FLOW_COLORS`: prefer `var(--preset-gold)` in
 * styling, and use this only where the value has to be a string in JS.
 */
export const PRESET_GOLD = '#d4a017';

/**
 * Mirrors `--surface-raised` in `styles/tokens.css` — the surface a node card,
 * popover or gallery card sits on, and the base that category hues are tinted
 * into to make a header or badge fill.
 *
 * Style with `var(--surface-raised)`; this exists only for `mixColor()`, which
 * has to do arithmetic on the value. Three components had each declared their
 * own copy of this literal before it was exported. `theme.test.ts` asserts it
 * matches the CSS.
 */
export const SURFACE_RAISED = '#242b37';

/** Shared zoom-out floor for both React Flow canvases. Wide skip-aware valley
 * layouts (long UNet-style graphs) need well below React Flow's 0.5 default
 * to fit on screen. */
export const CANVAS_MIN_ZOOM = 0.1;

/*
 * `SURFACE`, `TEXT`, `BRAND` and `TOOLBAR` used to be declared here as the
 * "single source of truth" for chrome colours. They were imported by nothing —
 * verified by grep across all 290 TS/TSX files — and were exercised only by
 * this module's own test, while all 54 CSS modules carried their own hex
 * literals. Two grey systems had drifted apart behind that fiction, and the
 * bottom three `TEXT` tiers (#666/#555/#444) failed WCAG AA on every surface
 * they were nominally for.
 *
 * The real tokens now live in `styles/tokens.css` as CSS custom properties,
 * which CSS modules can actually use. Style with `var(--text-secondary)`,
 * `var(--surface-panel)` and friends. Only the semantic data palettes above
 * are mirrored in TypeScript, and only because canvas code computes with them.
 */
