import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { parseMathSegments, MathText } from './MathText';

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

  it('skips an escaped \\$ while scanning for the inline close delimiter', () => {
    // The first $ opens inline math; the \$ in the body is skipped (not the
    // close), and the final $ closes it. Body keeps the literal "a \$ b".
    expect(parseMathSegments('$a \\$ b$ tail')).toEqual([
      { kind: 'inline', value: 'a \\$ b' },
      { kind: 'text', value: ' tail' },
    ]);
  });

  it('treats $$ with no closing $$ as the start of inline scanning', () => {
    // No closing "$$": the block branch is skipped, then the inline branch
    // scans from the first $; the immediately-following $ closes an empty
    // inline segment, leaving the rest as text.
    expect(parseMathSegments('$$x')).toEqual([
      { kind: 'inline', value: '' },
      { kind: 'text', value: 'x' },
    ]);
  });
});

describe('MathText component', () => {
  it('renders an empty tag (span by default) when text is null/undefined/empty', () => {
    const { container, rerender } = render(<MathText text={null} className="c1" />);
    const span = container.querySelector('span.c1');
    expect(span).toBeTruthy();
    expect(span?.textContent).toBe('');

    rerender(<MathText text={undefined} className="c1" />);
    expect(container.querySelector('span.c1')).toBeTruthy();

    rerender(<MathText text="" className="c1" />);
    expect(container.querySelector('span.c1')).toBeTruthy();
  });

  it('renders an empty div tag when as="div" and text is empty', () => {
    const { container } = render(<MathText text="" as="div" className="d1" />);
    expect(container.querySelector('div.d1')).toBeTruthy();
  });

  it('renders plain text segments inside the chosen tag', () => {
    const { container } = render(<MathText text="hello world" as="div" className="wrap" />);
    const div = container.querySelector('div.wrap');
    expect(div?.textContent).toContain('hello world');
  });

  it('renders inline math via KaTeX (katex markup present)', () => {
    const { container } = render(<MathText text="value $x+1$ end" />);
    // react-katex injects elements carrying the "katex" class on success.
    expect(container.querySelector('.katex')).toBeTruthy();
    expect(container.textContent).toContain('value ');
    expect(container.textContent).toContain(' end');
  });

  it('renders block math via KaTeX', () => {
    const { container } = render(<MathText text="$$\\frac{1}{2}$$" />);
    expect(container.querySelector('.katex')).toBeTruthy();
  });

  it('renders a mixed string with text, inline, and block segments', () => {
    const { container } = render(<MathText text={'A $$y$$ B $z$ C'} />);
    expect(container.textContent).toContain('A ');
    expect(container.textContent).toContain(' B ');
    expect(container.textContent).toContain(' C');
    expect(container.querySelectorAll('.katex').length).toBeGreaterThanOrEqual(1);
  });

  it('falls back to a monospace literal when an inline formula is malformed', () => {
    // Unclosed group "\frac{" is a KaTeX parse error → renderError fires and
    // emits the styled fallback span containing the raw source and error name.
    const { container } = render(<MathText text={'bad $\\frac{$ ok'} />);
    const fallback = container.querySelector('[class*="fallback"]');
    expect(fallback).toBeTruthy();
    expect(fallback?.textContent).toContain('\\frac{');
  });

  it('falls back to a monospace literal when a block formula is malformed', () => {
    const { container } = render(<MathText text={'$$\\frac{$$'} />);
    const fallback = container.querySelector('[class*="fallback"]');
    expect(fallback).toBeTruthy();
    expect(fallback?.textContent).toContain('\\frac{');
  });
});
