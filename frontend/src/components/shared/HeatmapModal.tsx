import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { HeatmapPlot, type HeatmapColormap } from './HeatmapPlot';
import { fetchOutput } from '../../api/executionOutputs';
import styles from './HeatmapModal.module.css';

interface HeatmapModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  /**
   * Inline values when the backend embedded them in the WebSocket summary.
   * When missing (e.g. tensors with numel > 256), the modal falls back to
   * fetching the full tensor via REST using the runId/nodeId/port triple.
   */
  inlineData?: number[][] | number[][][] | null;
  rowLabels?: string[];
  colLabels?: string[];
  colormap?: HeatmapColormap;
  detectCausalMask?: boolean;
  /** When inlineData is missing, used to REST-fetch the values. */
  runId?: string | null;
  nodeId?: string;
  port?: string;
  /** "boolean" → treat 0/1 values as a mask (no causal-pattern detection). */
  variant?: 'attention' | 'mask';
  /** Forwarded to HeatmapPlot — see its docs. Defaults to true for the
   *  attention variant since later positions in causal attention spread
   *  softmax mass over many cells, making absolute weights tiny. */
  normalizePerRow?: boolean;
}

function coerceTensorValues(
  values: unknown,
): number[][] | number[][][] | null {
  if (!Array.isArray(values) || values.length === 0) return null;
  // Detect 4D ([B, H, seq, seq]) — collapse to first batch.
  if (
    Array.isArray(values[0]) &&
    Array.isArray((values[0] as unknown[])[0]) &&
    Array.isArray(((values[0] as unknown[])[0] as unknown[])[0])
  ) {
    return (values as unknown as number[][][][])[0];
  }
  // 3D
  if (Array.isArray(values[0]) && Array.isArray((values[0] as unknown[])[0])) {
    return values as unknown as number[][][];
  }
  // 2D
  return values as unknown as number[][];
}

/**
 * Full-screen modal for inspecting attention heatmaps. Opens larger panels
 * with axis labels readable for sequences up to ~32 tokens. When the
 * backend didn't embed values inline (numel > 256), this modal REST-fetches
 * the full tensor with a higher max_elements cap.
 */
export function HeatmapModal({
  isOpen,
  onClose,
  title,
  inlineData,
  rowLabels,
  colLabels,
  colormap = 'viridis',
  detectCausalMask = true,
  runId,
  nodeId,
  port,
  variant = 'attention',
  normalizePerRow,
}: HeatmapModalProps) {
  const [fetchedData, setFetchedData] = useState<
    number[][] | number[][][] | null
  >(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [viewport, setViewport] = useState(() => ({
    w: typeof window !== 'undefined' ? window.innerWidth : 1280,
    h: typeof window !== 'undefined' ? window.innerHeight : 800,
  }));

  // Track viewport dims so the panel grows / shrinks live as the user
  // resizes the window with the modal open.
  useEffect(() => {
    if (!isOpen) return;
    const onResize = () =>
      setViewport({ w: window.innerWidth, h: window.innerHeight });
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [isOpen]);

  // Fetch full tensor via REST when not embedded inline.
  useEffect(() => {
    if (!isOpen) return;
    if (inlineData) return;
    if (!runId || !nodeId || !port) {
      setError('Cannot fetch: run is no longer available.');
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchOutput(runId, nodeId, port, { maxElements: 4096 })
      .then((data) => {
        if (cancelled) return;
        // Narrow to tensor — GenericOutput has `type: string`, so a literal
        // check alone doesn't discriminate it out. The `'values' in data`
        // guard rules out outputs that don't carry tensor values at all.
        if (data.type !== 'tensor' || !('values' in data)) {
          setError(`Expected tensor, got ${data.type}`);
          return;
        }
        const coerced = coerceTensorValues((data as { values: unknown }).values);
        if (variant === 'mask' && coerced) {
          // Coerce booleans / 0-1 floats to 0/1 numbers for the mask viz.
          const flatten = (m: number[][]) =>
            m.map((row) => row.map((x) => (x ? 1 : 0)));
          if (Array.isArray(coerced[0]) && Array.isArray((coerced[0] as unknown[])[0])) {
            setFetchedData((coerced as number[][][]).map(flatten));
          } else {
            setFetchedData(flatten(coerced as number[][]));
          }
        } else {
          setFetchedData(coerced);
        }
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e?.message ?? String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [isOpen, inlineData, runId, nodeId, port, variant]);

  // ESC closes
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const data = inlineData ?? fetchedData;

  const isMultiPanel =
    data !== null &&
    Array.isArray(data) &&
    data.length > 0 &&
    Array.isArray(data[0]) &&
    Array.isArray((data[0] as unknown[])[0]);
  const numPanels = isMultiPanel ? (data as number[][][]).length : 1;

  const seqLen = (() => {
    if (!data || data.length === 0) return 0;
    if (isMultiPanel) {
      const first = (data as number[][][])[0];
      return first.length;
    }
    return (data as number[][]).length;
  })();

  // Pick panel size: grow with the viewport up to 80% of the window's
  // smaller dimension, but cap so short sequences don't render absurdly
  // huge cells. The viewportLimit accounts for modal padding (32px each
  // side), header/footer (~110px combined), and per-panel gap (8px).
  const panelSize = (() => {
    if (seqLen === 0) return 480;
    const maxW = viewport.w * 0.8 - 64 - Math.max(0, numPanels - 1) * 8;
    const maxH = viewport.h * 0.8 - 110;
    const viewportLimit = Math.min(
      Math.floor(maxW / Math.max(1, numPanels)),
      Math.floor(maxH),
    );
    // Per-seq cap so a 5-token chat preset doesn't fill the whole monitor.
    const cap = seqLen <= 6
      ? 480
      : seqLen <= 12
        ? 600
        : seqLen <= 20
          ? 760
          : 1200;
    return Math.max(280, Math.min(viewportLimit, cap));
  })();

  const effectiveNormalize = normalizePerRow ?? variant === 'attention';

  return createPortal(
    <div className={styles.backdrop} onClick={onClose} role="dialog">
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <span className={styles.title}>{title}</span>
          <button
            type="button"
            onClick={onClose}
            className={styles.closeBtn}
            aria-label="Close"
          >
            ×
          </button>
        </div>
        <div className={styles.content}>
          {loading && <div className={styles.status}>Loading full tensor…</div>}
          {error && !loading && (
            <div className={`${styles.status} ${styles.error}`}>
              <div>Couldn't load: {error}</div>
              <div className={styles.errorHint}>
                Re-run the graph if the previous run has expired, or shorten
                the input sequence so values fit in the inline preview.
              </div>
            </div>
          )}
          {data && !loading && (
            <HeatmapPlot
              data={data}
              rowLabels={rowLabels}
              colLabels={colLabels}
              colormap={colormap}
              detectCausalMask={detectCausalMask}
              panelWidth={panelSize}
              panelHeight={panelSize}
              normalizePerRow={effectiveNormalize}
            />
          )}
        </div>
        <div className={styles.footer}>
          <span>
            seq_len = {seqLen}
            {effectiveNormalize && data && (
              <span className={styles.dim}> · row-normalised colours</span>
            )}
          </span>
          <span className={styles.dim}>click outside or press Esc to close</span>
        </div>
      </div>
    </div>,
    document.body,
  );
}
