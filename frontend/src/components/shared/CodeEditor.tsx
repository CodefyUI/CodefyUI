import { useEffect, useRef, useState } from 'react';
import styles from './CodeEditor.module.css';

/**
 * A Python source editor (core#131).
 *
 * CodeMirror 6 is loaded LAZILY, through a dynamic `import()` inside the
 * mount effect. That is the whole reason the repo could take its first
 * editor dependency: nothing about it reaches the entry chunk, so a user who
 * never opens a PythonScript node never downloads a byte of it. Until the
 * chunk resolves — and forever, if it fails to — this renders a plain
 * `<textarea>` with identical behaviour, so the component works with
 * JavaScript chunks blocked, in jsdom, and offline.
 *
 * The `nodrag`/`nowheel` classes are required whenever this is rendered
 * inside a React Flow node: without them the canvas steals the drag and the
 * scroll wheel and the editor cannot be used.
 */
export interface CodeEditorProps {
  value: string;
  onChange: (next: string) => void;
  /** Rendered height of the scrollable editing area. */
  minHeight?: number;
  readOnly?: boolean;
  ariaLabel: string;
  /** 1-based line to mark as the error line, if any. */
  errorLine?: number | null;
  className?: string;
}

type CodeMirrorModule = typeof import('./codemirrorSetup');

/**
 * Module-level cache of the dynamic import.
 *
 * Two editors mount together (the param column and the Code tab), and the
 * modal remounts on every open. Keeping the promise means the chunk is
 * fetched once per session rather than once per mount, and the second editor
 * mounts synchronously enough that no fallback flash is visible.
 */
let setupPromise: Promise<CodeMirrorModule> | null = null;

function loadCodeMirror(): Promise<CodeMirrorModule> {
  if (!setupPromise) {
    setupPromise = import('./codemirrorSetup').catch((error) => {
      // Let a later mount retry (a chunk can fail on a flaky network) and
      // fall back to the textarea for this one.
      setupPromise = null;
      throw error;
    });
  }
  return setupPromise;
}

export function CodeEditor({
  value,
  onChange,
  minHeight = 220,
  readOnly = false,
  ariaLabel,
  errorLine = null,
  className,
}: CodeEditorProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const [ready, setReady] = useState(false);

  // Live props, read through refs so the CodeMirror instance is built once
  // and never has to be torn down when the parent re-renders. The value and
  // the error line are also what the editor is SEEDED with: both can change
  // between mount and the chunk resolving (a fast typist, or a verdict that
  // was already cached), and the per-prop effects below fire before the
  // controller exists, so seeding from the mount-time closure would drop
  // whatever happened in between.
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  const valueRef = useRef(value);
  valueRef.current = value;
  const errorLineRef = useRef(errorLine);
  errorLineRef.current = errorLine;

  const controllerRef = useRef<{
    setValue: (next: string) => void;
    setErrorLine: (line: number | null) => void;
    destroy: () => void;
  } | null>(null);

  useEffect(() => {
    let cancelled = false;
    const host = hostRef.current;
    if (!host) return;

    loadCodeMirror()
      .then((mod) => {
        if (cancelled || !hostRef.current) return;
        const controller = mod.createPythonEditor({
          parent: hostRef.current,
          doc: valueRef.current,
          readOnly,
          onChange: (next) => onChangeRef.current(next),
          ariaLabel,
        });
        controller.setErrorLine(errorLineRef.current);
        controllerRef.current = controller;
        setReady(true);
      })
      .catch(() => {
        // Stay on the textarea. Nothing is logged: an offline user opening a
        // script node does not need a console error, they need an editor.
      });

    return () => {
      cancelled = true;
      controllerRef.current?.destroy();
      controllerRef.current = null;
      setReady(false);
    };
    // Mount-only: `value` is pushed in through the effect below so typing
    // never rebuilds the editor.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [readOnly, ariaLabel]);

  useEffect(() => {
    controllerRef.current?.setValue(value);
  }, [value]);

  useEffect(() => {
    controllerRef.current?.setErrorLine(errorLine);
  }, [errorLine]);

  return (
    <div className={`${styles.wrapper} ${className ?? ''}`.trim()}>
      <div
        ref={hostRef}
        className={`${styles.host} nodrag nowheel`}
        style={{ height: minHeight, display: ready ? undefined : 'none' }}
        data-testid="code-editor-host"
      />
      {!ready && (
        <textarea
          className={`${styles.fallback} nodrag nowheel`}
          style={{ height: minHeight }}
          value={value}
          readOnly={readOnly}
          spellCheck={false}
          aria-label={ariaLabel}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
    </div>
  );
}
