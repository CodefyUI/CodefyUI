import { Fragment, type ReactNode } from 'react';
import { InlineMath, BlockMath } from 'react-katex';
import styles from './MathText.module.css';

interface Props {
  text: string | undefined | null;
  className?: string;
  /** Default 'span'. Use 'div' for block-level descriptions. */
  as?: 'span' | 'div';
}

type Segment =
  | { kind: 'text'; value: string }
  | { kind: 'inline'; value: string }
  | { kind: 'block'; value: string };

/**
 * Split a string with `$$...$$` block math, `$...$` inline math, and `\$`
 * literal-dollar escapes. Anything else is returned as-is text.
 */
export function parseMathSegments(input: string): Segment[] {
  const segments: Segment[] = [];
  let buffer = '';
  let i = 0;

  const flushText = () => {
    if (buffer.length > 0) {
      segments.push({ kind: 'text', value: buffer });
      buffer = '';
    }
  };

  while (i < input.length) {
    const ch = input[i];

    // Backslash-escaped dollar — emit literal '$' to text buffer.
    if (ch === '\\' && input[i + 1] === '$') {
      buffer += '$';
      i += 2;
      continue;
    }

    // Block math $$...$$ (greedy match shortest body).
    if (ch === '$' && input[i + 1] === '$') {
      const close = input.indexOf('$$', i + 2);
      if (close !== -1) {
        flushText();
        segments.push({ kind: 'block', value: input.slice(i + 2, close) });
        i = close + 2;
        continue;
      }
    }

    // Inline math $...$
    if (ch === '$') {
      // Find next unescaped $.
      let j = i + 1;
      while (j < input.length) {
        if (input[j] === '\\' && input[j + 1] === '$') {
          j += 2;
          continue;
        }
        if (input[j] === '$') break;
        j++;
      }
      if (j < input.length && input[j] === '$') {
        flushText();
        segments.push({ kind: 'inline', value: input.slice(i + 1, j) });
        i = j + 1;
        continue;
      }
    }

    buffer += ch;
    i++;
  }
  flushText();
  return segments;
}

function renderSegment(seg: Segment, key: number): ReactNode {
  if (seg.kind === 'text') {
    return <Fragment key={key}>{seg.value}</Fragment>;
  }
  // Wrap KaTeX in an error-tolerant boundary by relying on react-katex's
  // built-in errorColor / renderError. We render plain monospace fallback
  // when the formula is malformed, instead of crashing the parent.
  if (seg.kind === 'block') {
    return (
      <BlockMath
        key={key}
        math={seg.value}
        renderError={(err) => (
          <span className={styles.fallback}>${`$${seg.value}$$`} ({err.name})</span>
        )}
      />
    );
  }
  return (
    <InlineMath
      key={key}
      math={seg.value}
      renderError={(err) => (
        <span className={styles.fallback}>${seg.value}$ ({err.name})</span>
      )}
    />
  );
}

/**
 * Render text containing inline `$x$` and block `$$x$$` LaTeX via KaTeX.
 * Falls back to monospace text on parse errors so a single bad formula
 * never crashes the surrounding panel.
 */
export function MathText({ text, className, as = 'span' }: Props) {
  const Tag = as as 'span' | 'div';
  if (!text) return <Tag className={className} />;
  const segments = parseMathSegments(text);
  return (
    <Tag className={className}>
      {segments.map(renderSegment)}
    </Tag>
  );
}
