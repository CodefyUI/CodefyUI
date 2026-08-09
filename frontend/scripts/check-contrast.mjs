#!/usr/bin/env node
/**
 * Contrast gate for the design tokens.
 *
 * Parses `src/styles/tokens.css` and re-derives every colour relationship the
 * token file claims, so the numbers in its comments cannot quietly rot. Run via
 * `pnpm contrast`; CI runs it too.
 *
 * Deliberately reads the CSS rather than importing a duplicated JS copy of the
 * palette — a checker with its own copy of the values proves only that the copy
 * is self-consistent.
 *
 * Implements WCAG 2.1 exactly: sRGB linearisation (c <= 0.03928 ? c/12.92 :
 * ((c+0.055)/1.055)^2.4), relative luminance L = 0.2126R + 0.7152G + 0.0722B,
 * ratio = (Lighter + 0.05) / (Darker + 0.05). Sanity-checked below against the
 * canonical #767676-on-white = 4.54:1 pair and black-on-white = 21:1.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const TOKENS = join(here, '..', 'src', 'styles', 'tokens.css');

/* ---------- colour maths ---------- */

const toRgb = (value) => {
  const hex = value.trim().match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
  if (hex) {
    let h = hex[1];
    if (h.length === 3) h = [...h].map((c) => c + c).join('');
    return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  }
  const rgba = value.trim().match(/^rgba?\(([^)]+)\)$/i);
  if (rgba) {
    const parts = rgba[1].split(',').map((p) => parseFloat(p));
    return parts.slice(0, 3);
  }
  throw new Error(`cannot parse colour: ${value}`);
};

const luminance = (rgb) => {
  const [r, g, b] = rgb.map((v) => {
    const c = v / 255;
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};

const contrast = (a, b) => {
  const la = luminance(toRgb(a));
  const lb = luminance(toRgb(b));
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
};

/** CIE L* — the right axis for judging whether two dark surfaces read as
 *  separate planes. Contrast ratio compresses near black and would call a
 *  1.03:1 pair "different". */
const lstar = (colour) => {
  const y = luminance(toRgb(colour));
  return y <= 216 / 24389 ? (y * 24389) / 27 : Math.cbrt(y) * 116 - 16;
};

/** CIE Lab (D65). Needed for CIEDE2000 below; L* alone cannot tell two hues
 *  of equal lightness apart, which is exactly the failure mode a categorical
 *  palette has — fourteen colours that all clear 3:1 on white but look alike
 *  is a different bug, not a fix. */
const lab = (colour) => {
  const [r, g, b] = toRgb(colour).map((v) => {
    const c = v / 255;
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  });
  const xyz = [
    0.4124564 * r + 0.3575761 * g + 0.1804375 * b,
    0.2126729 * r + 0.7151522 * g + 0.072175 * b,
    0.0193339 * r + 0.119192 * g + 0.9503041 * b,
  ];
  const wp = [0.95047, 1.0, 1.08883];
  const f = (t) => (t > 216 / 24389 ? Math.cbrt(t) : ((t * 24389) / 27 + 16) / 116);
  const [fx, fy, fz] = xyz.map((v, i) => f(v / wp[i]));
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
};

/** CIEDE2000 perceptual colour difference. ~1.0 is the just-noticeable
 *  difference; a categorical palette wants its closest pair well above that.
 *  Sanity-checked below against two pairs from the Sharma et al. reference
 *  data set. */
const deltaE2000Lab = ([L1, a1, b1], [L2, a2, b2]) => {
  const deg = (x) => (x * 180) / Math.PI;
  const rad = (x) => (x * Math.PI) / 180;
  const C1 = Math.hypot(a1, b1);
  const C2 = Math.hypot(a2, b2);
  const Cb = (C1 + C2) / 2;
  const G = 0.5 * (1 - Math.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7)));
  const a1p = (1 + G) * a1;
  const a2p = (1 + G) * a2;
  const C1p = Math.hypot(a1p, b1);
  const C2p = Math.hypot(a2p, b2);
  const h1p = (deg(Math.atan2(b1, a1p)) + 360) % 360;
  const h2p = (deg(Math.atan2(b2, a2p)) + 360) % 360;
  const dLp = L2 - L1;
  const dCp = C2p - C1p;
  let dhp = 0;
  if (C1p * C2p !== 0) {
    dhp = h2p - h1p;
    if (dhp > 180) dhp -= 360;
    else if (dhp < -180) dhp += 360;
  }
  const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin(rad(dhp) / 2);
  const Lbp = (L1 + L2) / 2;
  const Cbp = (C1p + C2p) / 2;
  let hbp;
  if (C1p * C2p === 0) hbp = h1p + h2p;
  else if (Math.abs(h1p - h2p) <= 180) hbp = (h1p + h2p) / 2;
  else hbp = h1p + h2p < 360 ? (h1p + h2p + 360) / 2 : (h1p + h2p - 360) / 2;
  const T =
    1 -
    0.17 * Math.cos(rad(hbp - 30)) +
    0.24 * Math.cos(rad(2 * hbp)) +
    0.32 * Math.cos(rad(3 * hbp + 6)) -
    0.2 * Math.cos(rad(4 * hbp - 63));
  const Rc = 2 * Math.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7));
  const Sl = 1 + (0.015 * (Lbp - 50) ** 2) / Math.sqrt(20 + (Lbp - 50) ** 2);
  const Sc = 1 + 0.045 * Cbp;
  const Sh = 1 + 0.015 * Cbp * T;
  const Rt = -Math.sin(rad(60 * Math.exp(-(((hbp - 275) / 25) ** 2)))) * Rc;
  return Math.sqrt(
    (dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh)
  );
};
const deltaE2000 = (a, b) => deltaE2000Lab(lab(a), lab(b));

/** Smallest CIEDE2000 distance between any two members of a palette, with the
 *  pair that produced it — that pair is the one a reader has to tell apart. */
const closestPair = (entries) => {
  let min = Infinity;
  let pair = ['', ''];
  for (let i = 0; i < entries.length; i++) {
    for (let j = i + 1; j < entries.length; j++) {
      const d = deltaE2000(entries[i][1], entries[j][1]);
      if (d < min) {
        min = d;
        pair = [entries[i][0], entries[j][0]];
      }
    }
  }
  return { min, pair };
};

/** Alpha-composite `fg` over `bg`, both as rgb triples. */
const overlay = (fg, alpha, bg) =>
  fg.map((v, i) => v * alpha + bg[i] * (1 - alpha));

const mixHex = (a, b, t) => {
  const [ra, rb] = [toRgb(a), toRgb(b)];
  const out = ra.map((v, i) => Math.round(v + (rb[i] - v) * t));
  return `#${out.map((v) => v.toString(16).padStart(2, '0')).join('')}`;
};

/* ---------- self-test: prove the maths before trusting a verdict ---------- */

const selfTests = [
  ['#767676 on #ffffff', contrast('#767676', '#ffffff'), 4.54, 0.01],
  ['#000000 on #ffffff', contrast('#000000', '#ffffff'), 21.0, 0.001],
  ['#ffffff on #ffffff', contrast('#ffffff', '#ffffff'), 1.0, 0.001],
  // CIEDE2000 against the Sharma/Wu/Dalal reference pairs. Pair 1 is the one
  // that exercises the hue-rotation term, which is the part of the formula
  // everyone gets wrong; pair 4 is a constructed unit distance.
  [
    'dE00 Sharma pair 1',
    deltaE2000Lab([50, 2.6772, -79.7751], [50, 0, -82.7485]),
    2.0425,
    0.0001,
  ],
  [
    'dE00 Sharma pair 2',
    deltaE2000Lab([50, 3.1571, -77.2803], [50, 0, -82.7485]),
    2.8615,
    0.0001,
  ],
  [
    'dE00 Sharma pair 4',
    deltaE2000Lab([50, -1.3802, -84.2814], [50, 0, -82.7485]),
    1.0,
    0.0001,
  ],
  ['dE00 of a colour with itself', deltaE2000('#4caf50', '#4caf50'), 0, 1e-9],
  // Lab of pure white is (100, 0, 0) — proves the sRGB -> Lab leg, which the
  // reference pairs above skip because they start in Lab.
  ['L* of #ffffff via lab()', lab('#ffffff')[0], 100, 0.001],
  ['L* of #000000 via lab()', lab('#000000')[0], 0, 0.001],
];
for (const [name, got, want, tol] of selfTests) {
  if (Math.abs(got - want) > tol) {
    console.error(`colour maths is wrong: ${name} = ${got.toFixed(4)}, expected ${want}`);
    process.exit(2);
  }
}

/* ---------- read the tokens ---------- */

const css = readFileSync(TOKENS, 'utf8');
const tokens = new Map();
for (const m of css.matchAll(/^\s*(--[\w-]+)\s*:\s*([^;]+);/gm)) {
  tokens.set(m[1], m[2].trim());
}
const t = (name) => {
  if (!tokens.has(name)) throw new Error(`token ${name} is not defined in tokens.css`);
  return tokens.get(name);
};

const SURFACES = [
  '--surface-canvas',
  '--surface-app',
  '--surface-panel',
  '--surface-raised',
  '--surface-input',
  '--surface-hover',
  '--surface-active',
];
const CATEGORIES = [...tokens.keys()].filter((k) => k.startsWith('--cat-'));

const failures = [];
let checked = 0;
const check = (label, actual, min) => {
  checked += 1;
  if (actual < min) failures.push(`${label}: ${actual.toFixed(2)} (needs ${min})`);
};

/* 1. Every usable text tier must clear AA on every surface.
      --text-disabled is exempt by design and is asserted separately. */
for (const tier of ['--text-primary', '--text-secondary', '--text-muted']) {
  for (const surface of SURFACES) {
    check(`${tier} on ${surface}`, contrast(t(tier), t(surface)), 4.5);
  }
}

/* 2. The disabled tier must actually look disabled — below AA on purpose —
      while still being distinguishable from the background. Asserting the
      upper bound stops someone "fixing" it into a fourth readable tier. */
const disabledOnPanel = contrast(t('--text-disabled'), t('--surface-panel'));
check('--text-disabled on --surface-panel (lower bound)', disabledOnPanel, 2.5);
if (disabledOnPanel >= 4.5) {
  failures.push(
    `--text-disabled on --surface-panel: ${disabledOnPanel.toFixed(2)} — ` +
      `that now passes AA, so it is no longer a disabled tier. Use --text-muted instead.`
  );
}

/* 3. Surfaces must be far enough apart in L* to read as separate planes.
      ~3 L* is the floor at which a step is perceptible on a dark screen. */
for (let i = 1; i < SURFACES.length; i++) {
  const step = lstar(t(SURFACES[i])) - lstar(t(SURFACES[i - 1]));
  if (step < 3) {
    failures.push(
      `${SURFACES[i - 1]} -> ${SURFACES[i]}: only ${step.toFixed(1)} L* apart (needs 3)`
    );
  }
}

/* 4. WCAG 1.4.11 — control outlines against the surface behind the control. */
for (const surface of ['--surface-app', '--surface-panel', '--surface-raised']) {
  check(`--border-strong on ${surface}`, contrast(t('--border-strong'), t(surface)), 3);
}

/* 5. Canvas wires are meaningful graphics, so 1.4.11 applies to them too.
      The trigger-flow green is the same kind of thing: it is what tells you an
      edge carries execution order rather than data. */
for (const wire of ['--wire', '--wire-data', '--wire-active', '--flow-trigger', '--flow-trigger-deep']) {
  check(`${wire} on --surface-canvas`, contrast(t(wire), t('--surface-canvas')), 3);
}

/* 5b. Preset nodes keep a filled gold header, so the ink on it is ordinary
       text and needs the full AA bar against every stop of the gradient. */
for (const stop of t('--preset-fill').match(/#[0-9a-f]{6}/gi) ?? []) {
  check(`--preset-ink on preset gradient stop ${stop}`, contrast(t('--preset-ink'), stop), 4.5);
}

/* 6. Accent and status colours are used as text, so they need the text bar. */
for (const colour of [
  '--accent',
  '--accent-deep',
  '--status-success',
  '--status-error',
  '--status-warning',
  '--status-info',
  '--status-idle',
  '--status-preset',
]) {
  for (const surface of ['--surface-panel', '--surface-raised', '--surface-canvas']) {
    check(`${colour} on ${surface}`, contrast(t(colour), t(surface)), 4.5);
  }
}

/* 7. Text inside a filled accent button. */
check(
  '--text-on-accent on --accent',
  contrast(t('--text-on-accent'), t('--accent')),
  4.5
);
check(
  '--text-on-accent on --accent-deep',
  contrast(t('--text-on-accent'), t('--accent-deep')),
  4.5
);

/* 8. Node headers. The header fill is the category hue tinted into
      --surface-raised at --node-header-tint; the title sits on that fill and
      the hue itself is drawn on it as the category label. Both must hold for
      all fourteen categories — the old design painted the raw hue as the fill
      and #eee on top, which failed on twelve of them. The tint is read from
      the CSS rather than restated here so it cannot drift. */
const NODE_HEADER_TINT = parseFloat(t('--node-header-tint'));
if (!(NODE_HEADER_TINT > 0 && NODE_HEADER_TINT < 1)) {
  failures.push(`--node-header-tint must be between 0 and 1, got ${t('--node-header-tint')}`);
}
for (const cat of CATEGORIES) {
  const fill = mixHex(t('--surface-raised'), t(cat), NODE_HEADER_TINT);
  check(`node title on ${cat} header`, contrast(t('--text-primary'), fill), 4.5);
  // The hue on its own tinted header is the accent bar / chip border — a
  // graphic, so 3:1. It is deliberately NOT used as the label text: a hue on a
  // tint of itself measured 3.50-4.16:1 and stays under 4.5 at every tint
  // strength and every lift, so that pairing is unreadable by construction.
  check(`${cat} accent bar on its own header`, contrast(t(cat), fill), 3);
  // The label that sits on the tinted header is ordinary text.
  check(`--text-muted label on ${cat} header`, contrast(t('--text-muted'), fill), 4.5);
  check(`${cat} as text on --surface-raised`, contrast(t(cat), t('--surface-raised')), 4.5);
}

/* 8b. The gallery-category, difficulty and run-status palettes are drawn the
       same way — a hue on a tint of itself — so they get the same two checks.
       These are what produced the 1:1-looking badges in the example gallery. */
const BADGE_PALETTES = [...tokens.keys()].filter(
  (k) =>
    k.startsWith('--ex-') ||
    k.startsWith('--difficulty-') ||
    (k.startsWith('--status-') && k !== '--status-idle')
);
for (const token of BADGE_PALETTES) {
  const fill = mixHex(t('--surface-raised'), t(token), NODE_HEADER_TINT);
  check(`${token} as a badge border on its own tint`, contrast(t(token), fill), 3);
  check(`--text-muted label on ${token} badge`, contrast(t('--text-muted'), fill), 4.5);
  check(`${token} as text on --surface-raised`, contrast(t(token), t('--surface-raised')), 4.5);
  check(`${token} as text on --surface-panel`, contrast(t(token), t('--surface-panel')), 4.5);
}

/* 8c. The layers editor's own layer-type palette (core#228). It was a pair of
       hand-synced hex tables inside `components/LayersEditor` that this gate
       could not see, still on the pre-lift Material tones the rest of the app
       had already moved off. Its nodes are built the same way canvas nodes
       are — a card on --surface-raised, a header tinted from the hue, the hue
       itself as the card's outline — so it gets the same checks as section 8.
       If a future edit puts a raw hue back behind white header text, the first
       of these is what fails. */
const LAYER_TYPES = [...tokens.keys()].filter((k) => k.startsWith('--layer-'));
if (LAYER_TYPES.length === 0) {
  failures.push('no --layer-* tokens found — the layers editor palette is missing');
}
for (const layer of LAYER_TYPES) {
  const fill = mixHex(t('--surface-raised'), t(layer), NODE_HEADER_TINT);
  check(`layer title on ${layer} header`, contrast(t('--text-primary'), fill), 4.5);
  // The hue at full strength as the header's accent rule, on the tint of
  // itself just above.
  check(`${layer} accent on its own header`, contrast(t(layer), fill), 3);
  // It also outlines the whole card, so it has to hold both sides of that
  // border: the card fill inside and the editor's canvas outside.
  check(`${layer} outline against the card`, contrast(t(layer), t('--surface-raised')), 3);
  check(`${layer} outline against the canvas`, contrast(t(layer), t('--surface-canvas')), 3);
  // And it is the dot beside each entry in the palette list.
  check(`${layer} as a palette dot on --surface-panel`, contrast(t(layer), t('--surface-panel')), 3);
}

/* Seven distinct hues across eleven roles (Input shares the green with
   Convolution, Output and the unknown-type fallback share the red with
   Activation). Measure the distinct ones against each other: a palette whose
   members all clear 3:1 but look alike is a different failure. The floor is
   well under the 16.1 dE00 the current set achieves — it records that these
   are not near-neighbours, it is not a target to design down to. */
const distinctLayerHues = [...new Set(LAYER_TYPES.map((k) => t(k)))].map((v) => [v, v]);
const layerPair = closestPair(distinctLayerHues);
checked += 1;
if (layerPair.min < 10) {
  failures.push(
    `layer palette: ${layerPair.pair.join(' and ')} are only ${layerPair.min.toFixed(2)} dE00 apart (needs 10)`
  );
}

/* 9. Washes are translucent fills. Check what they actually composite to over
      the panel they land on, not what their own rgba() claims in isolation —
      that is the mistake that made the old gallery badges look like 1:1 pairs.
      Each wash is paired with the text colour that is drawn on it. */
const WASHES = [
  ['--accent-wash', '--accent'],
  ['--success-wash', '--status-success'],
  ['--danger-wash', '--status-error'],
  ['--warning-wash', '--status-warning'],
  ['--info-wash', '--status-info'],
];
for (const [wash, ink] of WASHES) {
  const alpha = parseFloat(t(wash).match(/,\s*([\d.]+)\s*\)/)[1]);
  const composited = overlay(toRgb(t(ink)), alpha, toRgb(t('--surface-panel')));
  const hex = `#${composited.map((v) => Math.round(v).toString(16).padStart(2, '0')).join('')}`;
  check(`--text-primary on ${wash} over panel`, contrast(t('--text-primary'), hex), 4.5);
  check(`--text-secondary on ${wash} over panel`, contrast(t('--text-secondary'), hex), 4.5);
  check(`${ink} on ${wash} over panel`, contrast(t(ink), hex), 4.5);
}

/* 9b. Code editor. Comments and line numbers carry meaning in source, so they
       are held to the text bar, not to a "decoration" bar. */
for (const role of ['--code-text', '--code-comment', '--code-gutter']) {
  check(`${role} on --surface-code`, contrast(t(role), t('--surface-code')), 4.5);
}

/* 10. Glows are box-shadows and must stay translucent. An opaque one reads as
       a hard ring, not a glow — easy to introduce when swapping an rgba()
       literal for a solid colour token. */
for (const glow of ['--glow-running', '--glow-accent']) {
  if (!/rgba\([^)]*,\s*0?\.\d+\s*\)/.test(t(glow))) {
    failures.push(`${glow} must use a translucent rgba() colour, got: ${t(glow)}`);
  }
}

/* ---------- 11. The SVG diagram export ----------

   Everything above gates the app. `utils/exportDiagram.ts` renders the same
   graph as a standalone SVG in a second theme, and the default is the *light*
   one — the version that ends up in a student's report or a slide deck. That
   path had no gate at all, which is how seven of the fourteen category hues
   came to be drawn as a card border on white at 2.15:1 to 3.00:1, and all
   fourteen as the card's title text below 4.5:1 (core#227).

   Adding a second palette without extending this script would just move the
   unchecked colour decision one layer down, so the export's palettes are
   declared in tokens.css and re-derived here like everything else. The theme
   names and roles below must stay in step with DIAGRAM_THEMES; theme.test.ts
   is what holds the TypeScript mirror to these same tokens. */

const DIAGRAM_CATEGORIES = [
  'cnn', 'rnn', 'transformer', 'llm', 'diffusion', 'classical', 'rl',
  'data', 'training', 'io', 'dataflow', 'utility', 'normalization', 'tensorops',
];
const DIAGRAM_TYPES = [
  'tensor', 'model', 'dataset', 'dataloader', 'optimizer', 'loss-fn',
  'scalar', 'string', 'image', 'list', 'transform', 'any',
];

/* Each theme names where its palettes come from. The dark export reuses the
   app's hues (already tuned for a dark surface); the light export has its own,
   because a hue legible on a near-black canvas is not legible on paper. */
const DIAGRAM_THEMES = [
  {
    name: 'light',
    page: '--diagram-light-page',
    card: '--diagram-light-card',
    ink: '--diagram-light-ink',
    wire: '--diagram-light-wire',
    preset: '--diagram-light-preset',
    start: '--diagram-light-start',
    startFill: '--diagram-light-start-fill',
    category: (slug) => `--diagram-light-${slug}`,
    type: (slug) => `--diagram-light-type-${slug}`,
    // The light palette is a fresh design, so it is held to a real floor.
    minSeparation: 8,
  },
  {
    name: 'dark',
    page: '--diagram-dark-page',
    card: '--diagram-dark-card',
    ink: '--diagram-dark-ink',
    wire: '--diagram-dark-wire',
    preset: '--diagram-dark-preset',
    start: '--diagram-dark-start',
    startFill: '--diagram-dark-start-fill',
    category: (slug) => `--cat-${slug}`,
    type: (slug) => `--type-${slug}`,
    // The canvas palette's closest pair is --cat-classical / --cat-rl at 3.71
    // dE00 — two ambers that were already hard to tell apart before any of
    // this, and repainting the whole app to part them is a separate change.
    // The floor records that state so it cannot get worse; it is NOT a licence
    // to relax the light one.
    minSeparation: 3.5,
  },
];

for (const theme of DIAGRAM_THEMES) {
  const page = t(theme.page);
  const card = t(theme.card);
  const label = `diagram/${theme.name}`;

  // Port labels are ordinary text on the card.
  check(`${label}: ink on the card`, contrast(t(theme.ink), card), 4.5);

  const accents = [
    ...DIAGRAM_CATEGORIES.map((slug) => [slug, theme.category(slug), card]),
    ['preset', theme.preset, card],
    // The Start card has its own fill, so its accent is measured against that
    // rather than the ordinary card.
    ['start', theme.start, t(theme.startFill)],
  ];
  for (const [role, token, behind] of accents) {
    // Two roles, two bars: the hue outlines the card (1.4.11, a graphic that
    // identifies an object) *and* is painted as the card's title (1.4.3, text).
    check(`${label}: ${role} as a card border`, contrast(t(token), behind), 3);
    check(`${label}: ${role} as the card title`, contrast(t(token), behind), 4.5);
  }
  // A Start card is a filled box, so its fill has to hold the border's other
  // side and be told apart from an ordinary card.
  check(`${label}: start accent against the page`, contrast(t(theme.start), page), 3);
  if (deltaE2000(t(theme.startFill), card) < 5) {
    failures.push(
      `${label}: the Start card fill is only ${deltaE2000(t(theme.startFill), card).toFixed(1)} ` +
        `dE00 from an ordinary card (needs 5) — the entry point stops standing out`
    );
  }

  // Wires are drawn on the page, under the cards. An edge is a graphic that
  // says two nodes are connected, so it needs 3:1 whatever hue it wears.
  check(`${label}: neutral wire on the page`, contrast(t(theme.wire), page), 3);
  for (const slug of DIAGRAM_TYPES) {
    check(`${label}: ${slug} wire on the page`, contrast(t(theme.type(slug)), page), 3);
  }

  // Fourteen hues that all clear 3:1 on white but look alike is a different
  // failure, so measure them against each other too.
  const cats = closestPair(
    DIAGRAM_CATEGORIES.map((slug) => [slug, t(theme.category(slug))])
  );
  if (cats.min < theme.minSeparation) {
    failures.push(
      `${label}: categories ${cats.pair.join(' and ')} are only ${cats.min.toFixed(2)} dE00 ` +
        `apart (needs ${theme.minSeparation}) — they are the pair a reader has to tell apart`
    );
  }
  checked += 1;
  const types = closestPair(DIAGRAM_TYPES.map((slug) => [slug, t(theme.type(slug))]));
  if (types.min < 5) {
    failures.push(
      `${label}: wire types ${types.pair.join(' and ')} are only ${types.min.toFixed(2)} dE00 apart (needs 5)`
    );
  }
  checked += 1;
}

/* 11b. A light category hue is a restatement of its dark counterpart for a
        white page, not a new colour: same hue, lower lightness. If someone
        "fixes" a contrast failure by rotating a hue instead of darkening it,
        the export stops agreeing with the app a reader is looking at, and the
        colour coding silently means two different things in two places. */
for (const slug of DIAGRAM_CATEGORIES) {
  const [, da, db] = lab(t(`--cat-${slug}`));
  const [, la, lb] = lab(t(`--diagram-light-${slug}`));
  const drift = Math.abs(
    ((Math.atan2(lb, la) - Math.atan2(db, da)) * 180) / Math.PI
  );
  const wrapped = drift > 180 ? 360 - drift : drift;
  checked += 1;
  if (wrapped > 12) {
    failures.push(
      `--diagram-light-${slug} sits ${wrapped.toFixed(1)} degrees of hue from --cat-${slug} ` +
        `(needs <= 12) — the light export must be the same colour, darkened, not a different one`
    );
  }
}

/* ---------- report ---------- */

if (failures.length) {
  console.error(`\nContrast gate FAILED — ${failures.length} problem(s):\n`);
  for (const f of failures) console.error(`  - ${f}`);
  console.error(
    '\nEdit src/styles/tokens.css. Do not relax a threshold to make this pass;\n' +
      'the thresholds are WCAG 2.1 AA (1.4.3 text, 1.4.11 non-text).\n'
  );
  process.exit(1);
}

console.log(`Contrast gate passed — ${checked} colour relationships verified across ${tokens.size} tokens.`);
