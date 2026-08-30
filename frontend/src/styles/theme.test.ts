import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  CATEGORY_COLORS,
  CATEGORY_VARS,
  CATEGORY_COLORS_ON_LIGHT,
  CATEGORY_LIGHT_VARS,
  DATA_TYPE_COLORS_ON_LIGHT,
  DATA_TYPE_LIGHT_VARS,
  DATA_TYPE_VARS,
  LAYER_TYPE_COLORS,
  LAYER_TYPE_VARS,
  DIAGRAM_CHROME,
  DIAGRAM_CHROME_VARS,
  TOKEN_COLORS,
  getTokenColor,
  DIFFICULTY_COLORS,
  STATUS_COLORS,
  EXAMPLE_CATEGORY_COLORS,
  EXAMPLE_CATEGORY_FALLBACK,
  NODE_HEADER_TINT,
  FLOW_COLORS,
  PRESET_GOLD,
  SURFACE_RAISED,
  mixColor,
} from './theme';
import { DATA_TYPE_COLORS } from '../utils';

/**
 * `tokens.css` is the source of truth for colour; the maps in `theme.ts` are a
 * mirror kept only for code that has to compute with the values. These tests
 * parse the CSS and assert the mirror matches, so an edit to one without the
 * other fails here rather than shipping two different greens.
 */
const tokensCss = readFileSync(
  join(dirname(fileURLToPath(import.meta.url)), 'tokens.css'),
  'utf8',
);

const cssTokens = new Map<string, string>();
for (const m of tokensCss.matchAll(/^\s*(--[\w-]+)\s*:\s*([^;]+);/gm)) {
  cssTokens.set(m[1], m[2].trim());
}

const cssVar = (name: string): string => {
  const value = cssTokens.get(name);
  if (value === undefined) throw new Error(`${name} is not defined in tokens.css`);
  return value;
};

describe('tokens.css / theme.ts agreement', () => {
  it('defines a CSS variable for every node category', () => {
    expect(Object.keys(CATEGORY_VARS).sort()).toEqual(Object.keys(CATEGORY_COLORS).sort());
  });

  it.each(Object.keys(CATEGORY_COLORS))('category %s matches its CSS variable', (name) => {
    expect(CATEGORY_COLORS[name]).toBe(cssVar(CATEGORY_VARS[name]));
  });

  it.each([
    ['Usage_Example', '--ex-usage-example'],
    ['Model_Architecture', '--ex-model-architecture'],
    ['Classical', '--ex-classical'],
    ['LLM', '--ex-llm'],
    ['Diffusion', '--ex-diffusion'],
    ['Transformer', '--ex-transformer'],
    ['RNN', '--ex-rnn'],
    ['RL', '--ex-rl'],
    ['Stats', '--ex-stats'],
  ])('example category %s matches %s', (name, token) => {
    expect(EXAMPLE_CATEGORY_COLORS[name]).toBe(cssVar(token));
  });

  it('example fallback matches its CSS variable', () => {
    expect(EXAMPLE_CATEGORY_FALLBACK).toBe(cssVar('--ex-fallback'));
  });

  it.each([
    ['beginner', '--difficulty-beginner'],
    ['intermediate', '--difficulty-intermediate'],
    ['advanced', '--difficulty-advanced'],
  ])('difficulty %s matches %s', (name, token) => {
    expect(DIFFICULTY_COLORS[name]).toBe(cssVar(token));
  });

  it.each([
    ['running', '--status-running'],
    ['completed', '--status-completed'],
    ['error', '--status-error'],
    ['cached', '--status-cached'],
    ['skipped', '--status-skipped'],
    ['idle', '--status-idle'],
    ['interrupted', '--status-interrupted'],
  ])('status %s matches %s', (name, token) => {
    expect(STATUS_COLORS[name]).toBe(cssVar(token));
  });

  it('node header tint matches the CSS variable', () => {
    expect(String(NODE_HEADER_TINT)).toBe(cssVar('--node-header-tint'));
  });

  it('defines a CSS variable for every layer category', () => {
    expect(Object.keys(LAYER_TYPE_VARS).sort()).toEqual(
      Object.keys(LAYER_TYPE_COLORS).sort(),
    );
  });

  it.each(Object.keys(LAYER_TYPE_COLORS))(
    'layer category %s matches its CSS variable',
    (name) => {
      expect(LAYER_TYPE_COLORS[name]).toBe(cssVar(LAYER_TYPE_VARS[name]));
    },
  );

  it('has no orphaned --layer-* variable on either side', () => {
    const declared = [...cssTokens.keys()].filter((k) => k.startsWith('--layer-')).sort();
    expect(declared).toEqual(Object.values(LAYER_TYPE_VARS).sort());
  });

  it.each([
    ['trigger', '--flow-trigger'],
    ['triggerDeep', '--flow-trigger-deep'],
  ])('flow colour %s matches %s', (name, token) => {
    expect(FLOW_COLORS[name as keyof typeof FLOW_COLORS]).toBe(cssVar(token));
  });

  it('preset gold matches its CSS variable', () => {
    expect(PRESET_GOLD).toBe(cssVar('--preset-gold'));
  });

  it('raised surface matches its CSS variable', () => {
    // Three components each kept their own copy of this literal before it was
    // exported; the mix maths behind every node header and badge fill depends
    // on it agreeing with the CSS.
    expect(SURFACE_RAISED).toBe(cssVar('--surface-raised'));
  });

  it('has no orphaned category variable on either side', () => {
    const declared = [...cssTokens.keys()].filter((k) => k.startsWith('--cat-')).sort();
    expect(declared).toEqual(Object.values(CATEGORY_VARS).sort());
  });

  // Hex case differs by history: tokens.css is lowercase throughout, while
  // DATA_TYPE_COLORS predates it and is uppercase. Same colour either way.
  const sameColour = (a: string, b: string) => expect(a.toLowerCase()).toBe(b.toLowerCase());

  it.each(Object.keys(DATA_TYPE_COLORS))('data type %s matches its CSS variable', (name) => {
    sameColour(DATA_TYPE_COLORS[name], cssVar(DATA_TYPE_VARS[name]));
  });

  it('has no orphaned data-type variable on either side', () => {
    const declared = [...cssTokens.keys()].filter((k) => k.startsWith('--type-')).sort();
    expect(declared).toEqual(Object.values(DATA_TYPE_VARS).sort());
    expect(Object.keys(DATA_TYPE_VARS).sort()).toEqual(Object.keys(DATA_TYPE_COLORS).sort());
  });
});

/**
 * The SVG export's light theme (core#227). It is a second palette, so it is
 * exactly the kind of thing that drifts from the token layer — these hold it
 * in place, and `scripts/check-contrast.mjs` proves the tokens themselves are
 * legible. Between the two, neither a bad value nor a stale copy can ship.
 */
describe('tokens.css / diagram export palette agreement', () => {
  it('covers every node category', () => {
    expect(Object.keys(CATEGORY_COLORS_ON_LIGHT).sort()).toEqual(
      Object.keys(CATEGORY_COLORS).sort(),
    );
    expect(Object.keys(CATEGORY_LIGHT_VARS).sort()).toEqual(
      Object.keys(CATEGORY_COLORS).sort(),
    );
  });

  it.each(Object.keys(CATEGORY_COLORS_ON_LIGHT))(
    'light category %s matches its CSS variable',
    (name) => {
      expect(CATEGORY_COLORS_ON_LIGHT[name]).toBe(cssVar(CATEGORY_LIGHT_VARS[name]));
    },
  );

  it('covers every data type', () => {
    expect(Object.keys(DATA_TYPE_COLORS_ON_LIGHT).sort()).toEqual(
      Object.keys(DATA_TYPE_COLORS).sort(),
    );
    expect(Object.keys(DATA_TYPE_LIGHT_VARS).sort()).toEqual(
      Object.keys(DATA_TYPE_COLORS).sort(),
    );
  });

  it.each(Object.keys(DATA_TYPE_COLORS_ON_LIGHT))(
    'light data type %s matches its CSS variable',
    (name) => {
      expect(DATA_TYPE_COLORS_ON_LIGHT[name]).toBe(cssVar(DATA_TYPE_LIGHT_VARS[name]));
    },
  );

  it('keeps the exported DATASET and TRANSFORM ambers apart in lightness', () => {
    // The canvas pair is asserted the same way in utils/index.test.ts (#197
    // item 5); this is its light-export twin (core#323), where the two ambers
    // measured 11.37 dE00 apart with an L* gap of 0.1 — every bit of the
    // separation on the red-green axis, so a dichromat read one colour and a
    // greyscale print showed one grey. TRANSFORM is the half that cannot move
    // (a lighter amber misses 3:1 on the white page), so DATASET is the half
    // that must stay below it. 4 L* is the floor `check-contrast.mjs` holds a
    // same-hue-family wire pair to; the shipped gap is ~7.9.
    const lstar = (hex: string) => {
      const [r, g, b] = [0, 2, 4].map((i) => {
        const c = parseInt(hex.slice(1 + i, 3 + i), 16) / 255;
        return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
      });
      const y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      return y <= 216 / 24389 ? (y * 24389) / 27 : Math.cbrt(y) * 116 - 16;
    };
    const gap =
      lstar(DATA_TYPE_COLORS_ON_LIGHT.TRANSFORM) - lstar(DATA_TYPE_COLORS_ON_LIGHT.DATASET);
    expect(gap).toBeGreaterThan(4);
  });

  it.each(
    (['light', 'dark'] as const).flatMap((theme) =>
      (Object.keys(DIAGRAM_CHROME_VARS) as (keyof typeof DIAGRAM_CHROME.light)[]).map(
        (role) => [theme, role] as const,
      ),
    ),
  )('%s diagram chrome %s matches its CSS variable', (theme, role) => {
    expect(DIAGRAM_CHROME[theme][role]).toBe(
      cssVar(`--diagram-${theme}-${DIAGRAM_CHROME_VARS[role]}`),
    );
  });

  it('has no orphaned diagram variable on either side', () => {
    const declared = [...cssTokens.keys()].filter((k) => k.startsWith('--diagram-')).sort();
    const mirrored = [
      ...Object.values(CATEGORY_LIGHT_VARS),
      ...Object.values(DATA_TYPE_LIGHT_VARS),
      ...(['light', 'dark'] as const).flatMap((theme) =>
        Object.values(DIAGRAM_CHROME_VARS).map((role) => `--diagram-${theme}-${role}`),
      ),
    ].sort();
    expect(declared).toEqual(mirrored);
  });
});

/**
 * core#228. The layers editor kept a private copy of a layer-type palette, in
 * two files that were hand-synced, and never got the dark-surface lift the rest
 * of the app did. So the same conceptual colour was one value on the canvas and
 * a different, dimmer one inside the editor.
 *
 * These are hue identities, not semantic ones: purple means Normalization to
 * the layers editor and Transformer to the canvas, and that difference is
 * deliberate (the two palettes classify different things). What has to hold is
 * that the app has ONE purple.
 */
describe('the layers editor and the canvas agree on a hue', () => {
  it.each([
    ['the green', 'Convolution', 'CNN'],
    ['the purple', 'Normalization', 'Transformer'],
    ['the purple', 'Attention', 'Transformer'],
    ['the blue', 'Pooling', 'RNN'],
    ['the blue', 'Recurrent', 'RNN'],
    ['the orange', 'Regularization', 'RL'],
    ['the cyan', 'Linear', 'Data'],
    ['the blue-grey', 'Utility', 'Utility'],
    ['the red', 'Activation', 'Training'],
  ])('%s is the same value in both palettes', (_hue, layer, category) => {
    expect(LAYER_TYPE_COLORS[layer]).toBe(CATEGORY_COLORS[category]);
  });

  it('carries none of the pre-lift Material tones', () => {
    // Measured too dark to read on a dark surface and lifted app-wide; the
    // layers editor was the last place still shipping the originals.
    const preLift = ['#9c27b0', '#2196f3', '#f44336', '#607d8b'];
    for (const [name, value] of Object.entries(LAYER_TYPE_COLORS)) {
      expect(preLift, `${name} is still a pre-lift tone`).not.toContain(
        value.toLowerCase(),
      );
    }
  });

  it('gives the boundary and unknown slots their own names, not a shared one', () => {
    // Seven hues across thirteen roles. The duplication is intentional and
    // recorded here so a future edit to one of them is a decision rather than
    // an accident. Recurrent and Attention joined the layers palette without
    // adding a hue: they reuse the blue and the purple, which is what keeps
    // a layer the same colour as the canvas node it corresponds to.
    expect(LAYER_TYPE_COLORS.Input).toBe(LAYER_TYPE_COLORS.Convolution);
    expect(LAYER_TYPE_COLORS.Output).toBe(LAYER_TYPE_COLORS.Activation);
    expect(LAYER_TYPE_COLORS.Unknown).toBe(LAYER_TYPE_COLORS.Activation);
    expect(LAYER_TYPE_COLORS.Merge).toBe(LAYER_TYPE_COLORS.Regularization);
    expect(new Set(Object.values(LAYER_TYPE_COLORS)).size).toBe(7);
  });
});

describe('theme tokens', () => {
  it('exposes category colors', () => {
    expect(CATEGORY_COLORS.CNN).toBe('#4caf50');
    expect(CATEGORY_COLORS.Transformer).toBe('#c279ce');
    expect(CATEGORY_COLORS['Tensor Operations']).toBe('#838fcf');
  });

  it('exposes a 12-color token palette', () => {
    expect(TOKEN_COLORS).toHaveLength(12);
    expect(TOKEN_COLORS[0]).toBe('#7DD3FC');
    expect(TOKEN_COLORS[TOKEN_COLORS.length - 1]).toBe('#BEF264');
  });

  it('exposes difficulty colors', () => {
    expect(DIFFICULTY_COLORS.beginner).toBe('#4caf50');
    expect(DIFFICULTY_COLORS.intermediate).toBe('#ff9800');
    expect(DIFFICULTY_COLORS.advanced).toBe('#f66358');
  });

  it('exposes status colors', () => {
    expect(STATUS_COLORS.running).toBe('#ffc107');
  });

  it('gives idle a visible colour rather than the old near-invisible grey', () => {
    // #444 on the panel it sits on measured 1.86:1 — the dot was not there.
    expect(STATUS_COLORS.idle).not.toBe('#444');
    expect(STATUS_COLORS.idle).toBe('#8593a3');
  });
});

describe('mixColor', () => {
  it('returns the base at amount 0 and the colour at amount 1', () => {
    expect(mixColor('#000000', '#ffffff', 0)).toBe('#000000');
    expect(mixColor('#000000', '#ffffff', 1)).toBe('#ffffff');
  });

  it('mixes channel-wise at the midpoint', () => {
    expect(mixColor('#000000', '#ffffff', 0.5)).toBe('#808080');
    expect(mixColor('#204060', '#60a0e0', 0.5)).toBe('#4070a0');
  });

  it('accepts 3-digit hex on either side', () => {
    expect(mixColor('#000', '#fff', 0.5)).toBe('#808080');
  });

  it('produces the header fill the contrast gate verifies', () => {
    // Kept explicit: the gate computes this same mix independently, so if the
    // two ever disagree one of them is wrong about what ships.
    expect(mixColor('#242b37', CATEGORY_COLORS.Training, NODE_HEADER_TINT)).toBe('#4a353d');
  });
});

describe('getTokenColor', () => {
  it('returns the palette color at the given index', () => {
    expect(getTokenColor(0)).toBe(TOKEN_COLORS[0]);
    expect(getTokenColor(3)).toBe(TOKEN_COLORS[3]);
  });

  it('wraps around the palette length via modulo', () => {
    expect(getTokenColor(TOKEN_COLORS.length)).toBe(TOKEN_COLORS[0]);
    expect(getTokenColor(TOKEN_COLORS.length + 5)).toBe(TOKEN_COLORS[5]);
  });
});
