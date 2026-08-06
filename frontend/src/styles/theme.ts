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
