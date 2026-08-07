import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  CATEGORY_COLORS,
  CATEGORY_VARS,
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
