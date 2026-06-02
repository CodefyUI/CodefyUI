import { describe, expect, it } from 'vitest';
import {
  CATEGORY_COLORS,
  TOKEN_COLORS,
  getTokenColor,
  DIFFICULTY_COLORS,
  STATUS_COLORS,
  SURFACE,
  TEXT,
  BRAND,
  TOOLBAR,
} from './theme';

describe('theme tokens', () => {
  it('exposes category colors', () => {
    expect(CATEGORY_COLORS.CNN).toBe('#4CAF50');
    expect(CATEGORY_COLORS.Transformer).toBe('#9C27B0');
    expect(CATEGORY_COLORS['Tensor Operations']).toBe('#5C6BC0');
  });

  it('exposes a 12-color token palette', () => {
    expect(TOKEN_COLORS).toHaveLength(12);
    expect(TOKEN_COLORS[0]).toBe('#7DD3FC');
    expect(TOKEN_COLORS[TOKEN_COLORS.length - 1]).toBe('#BEF264');
  });

  it('exposes difficulty colors', () => {
    expect(DIFFICULTY_COLORS.beginner).toBe('#4CAF50');
    expect(DIFFICULTY_COLORS.intermediate).toBe('#FF9800');
    expect(DIFFICULTY_COLORS.advanced).toBe('#F44336');
  });

  it('exposes status colors', () => {
    expect(STATUS_COLORS.running).toBe('#FFC107');
    expect(STATUS_COLORS.idle).toBe('#444');
  });

  it('exposes surface / text / brand / toolbar token groups', () => {
    expect(SURFACE.bg).toBe('#121212');
    expect(TEXT.primary).toBe('#eee');
    expect(BRAND.primary).toBe('#06b6d4');
    expect(TOOLBAR.border).toBe('#1a2230');
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
