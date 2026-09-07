import { Fragment, useEffect, useMemo, useState, type ReactNode } from 'react';
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

type KatexModule = typeof import('./katexRenderer');

/**
 * Module-level cache of the dynamic import (the shape CodeEditor.tsx uses
 * for CodeMirror).
 *
 * KaTeX is about 260 kB and only a handful of node descriptions contain a
 * formula, so it is fetched the first time one is rendered rather than at
 * startup. Keeping the promise means the chunk is requested once per
 * session, and `katexModule` seeds later mounts so every description after
 * the first one typesets synchronously, with no source-to-math swap.
 */
let katexPromise: Promise<KatexModule> | null = null;
let katexModule: KatexModule | null = null;

function loadKatex(): Promise<KatexModule> {
  if (!katexPromise) {
    katexPromise = import('./katexRenderer')
      .then((mod) => {
        katexModule = mod;
        return mod;
      })
      .catch((error) => {
        // Drop the rejected promise so the next mount asks again instead of
        // inheriting this failure; this mount keeps the raw source. A chunk
        // can fail on a flaky network, or 404 once `cdui update` has replaced
        // dist/ under an open page. Browsers remember a failed module fetch
        // for the life of the page, so in practice the formula typesets
        // again after a reload; nothing here is lost by asking sooner.
        katexPromise = null;
        throw error;
      });
  }
  return katexPromise;
}

function renderSegment(seg: Segment, key: number, katex: KatexModule | null): ReactNode {
  if (seg.kind === 'text') {
    return <Fragment key={key}>{seg.value}</Fragment>;
  }
  const source = seg.kind === 'block' ? `$$${seg.value}$$` : `$${seg.value}$`;
  // Until the chunk resolves the raw source stands in, so a description is
  // never blank and stays readable if the chunk never arrives.
  if (!katex) {
    return <Fragment key={key}>{source}</Fragment>;
  }
  // Wrap KaTeX in an error-tolerant boundary by relying on react-katex's
  // built-in errorColor / renderError. We render plain monospace fallback
  // when the formula is malformed, instead of crashing the parent.
  const MathComponent = seg.kind === 'block' ? katex.BlockMath : katex.InlineMath;
  return (
    <MathComponent
      key={key}
      math={seg.value}
      renderError={(err) => (
        <span className={styles.fallback}>
          {source} ({err.name})
        </span>
      )}
    />
  );
}

/**
 * Render text containing inline `$x$` and block `$$x$$` LaTeX via KaTeX.
 * Falls back to monospace text on parse errors so a single bad formula
 * never crashes the surrounding panel.
 *
 * KaTeX itself is loaded lazily, and only when the text contains a formula:
 * a text-only description never requests the chunk.
 */
export function MathText({ text, className, as = 'span' }: Props) {
  const Tag = as as 'span' | 'div';
  const segments = useMemo(() => (text ? parseMathSegments(text) : []), [text]);
  const needsKatex = segments.some((seg) => seg.kind !== 'text');
  const [katex, setKatex] = useState<KatexModule | null>(katexModule);

  useEffect(() => {
    if (!needsKatex || katex) return;
    let cancelled = false;
    loadKatex()
      .then((mod) => {
        if (!cancelled) setKatex(mod);
      })
      .catch(() => {
        // The raw source is already on screen; nothing else to do here.
      });
    return () => {
      cancelled = true;
    };
  }, [needsKatex, katex]);

  if (!text) return <Tag className={className} />;
  return <Tag className={className}>{segments.map((seg, i) => renderSegment(seg, i, katex))}</Tag>;
}
