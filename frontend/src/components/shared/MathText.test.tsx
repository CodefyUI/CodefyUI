import { describe, it, expect } from 'vitest';
import { parseMathSegments } from './MathText';

describe('parseMathSegments', () => {
  it('returns plain text as a single segment', () => {
    expect(parseMathSegments('hello world')).toEqual([
      { kind: 'text', value: 'hello world' },
    ]);
  });

  it('extracts a single inline math segment', () => {
    expect(parseMathSegments('value $x$ here')).toEqual([
      { kind: 'text', value: 'value ' },
      { kind: 'inline', value: 'x' },
      { kind: 'text', value: ' here' },
    ]);
  });

  it('extracts multiple inline math segments', () => {
    const out = parseMathSegments('$a$ then $b+c$');
    expect(out).toEqual([
      { kind: 'inline', value: 'a' },
      { kind: 'text', value: ' then ' },
      { kind: 'inline', value: 'b+c' },
    ]);
  });

  it('extracts a block math segment', () => {
    expect(parseMathSegments('before $$\\frac{1}{2}$$ after')).toEqual([
      { kind: 'text', value: 'before ' },
      { kind: 'block', value: '\\frac{1}{2}' },
      { kind: 'text', value: ' after' },
    ]);
  });

  it('treats unmatched $ as literal text', () => {
    // No closing $ → single trailing $ stays as text.
    expect(parseMathSegments('cost $5')).toEqual([
      { kind: 'text', value: 'cost $5' },
    ]);
  });

  it('respects \\$ escape', () => {
    expect(parseMathSegments('price is \\$5 and $x$')).toEqual([
      { kind: 'text', value: 'price is $5 and ' },
      { kind: 'inline', value: 'x' },
    ]);
  });

  it('handles empty input', () => {
    expect(parseMathSegments('')).toEqual([]);
  });

  it('keeps text + block + text + inline mixed in order', () => {
    const out = parseMathSegments('A $$B$$ C $D$ E');
    expect(out).toEqual([
      { kind: 'text', value: 'A ' },
      { kind: 'block', value: 'B' },
      { kind: 'text', value: ' C ' },
      { kind: 'inline', value: 'D' },
      { kind: 'text', value: ' E' },
    ]);
  });
});
