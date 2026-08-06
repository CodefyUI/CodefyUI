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
];
for (const [name, got, want, tol] of selfTests) {
  if (Math.abs(got - want) > tol) {
    console.error(`contrast maths is wrong: ${name} = ${got.toFixed(3)}, expected ${want}`);
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

/* 5. Canvas wires are meaningful graphics, so 1.4.11 applies to them too. */
for (const wire of ['--wire', '--wire-data', '--wire-active']) {
  check(`${wire} on --surface-canvas`, contrast(t(wire), t('--surface-canvas')), 3);
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
  check(`${cat} label on its own header`, contrast(t(cat), fill), 3);
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
  check(`${token} on its own badge tint`, contrast(t(token), fill), 3);
  check(`${token} as text on --surface-raised`, contrast(t(token), t('--surface-raised')), 4.5);
  check(`${token} as text on --surface-panel`, contrast(t(token), t('--surface-panel')), 4.5);
}

/* 9. The accent wash is a translucent fill; check what it actually composites
      to, not what its own rgba() claims. */
const washAlpha = parseFloat(t('--accent-wash').match(/,\s*([\d.]+)\s*\)/)[1]);
const washOnPanel = overlay(toRgb(t('--accent')), washAlpha, toRgb(t('--surface-panel')));
const washHex = `#${washOnPanel.map((v) => Math.round(v).toString(16).padStart(2, '0')).join('')}`;
check('--text-primary on --accent-wash over panel', contrast(t('--text-primary'), washHex), 4.5);
check('--accent on --accent-wash over panel', contrast(t('--accent'), washHex), 4.5);

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
