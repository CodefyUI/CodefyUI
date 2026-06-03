import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { getTokenColor } from '../../styles/theme';
import { useI18n } from '../../i18n';
import { fetchOutput } from '../../api/executionOutputs';
import type { ScatterPoint } from './ScatterPlot';
import { CloseIcon, EyeIcon, EyeOffIcon, FitIcon, ZoomInIcon, ZoomOutIcon } from './Icons';
import styles from './ScatterModal.module.css';

export interface ScatterModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  /**
   * Inline points when the WebSocket summary embedded them (small N). When
   * absent/empty and the run coordinates are available, the modal REST-fetches
   * the full [N, 2] projection (and labels) — this is the common case for the
   * "many embeddings" scenario, where the inline preview is suppressed because
   * the tensor exceeds the WS inline cap.
   */
  inlinePoints?: ScatterPoint[] | null;
  runId?: string | null;
  nodeId?: string;
  /** Output port carrying the [N, 2] coordinates. */
  pointsPort?: string;
  /** Output port carrying the per-point label list. */
  labelsPort?: string;
}

interface View {
  /** Pixels per world unit. */
  scale: number;
  /** World coordinate displayed at the centre of the plot. */
  cx: number;
  cy: number;
}

/** Nearest-label rows rendered in the sidebar (and labelled on the plot). */
const LIST_MAX = 60;
const DOT_R = 3.5;
const DOT_R_HOVER = 6;
const CULL_PAD = 28;

/** Build ScatterPoints from a raw [N, 2] tensor value + optional label list. */
function buildPoints(rows: unknown, labels: unknown[] | null): ScatterPoint[] {
  if (!Array.isArray(rows)) return [];
  return rows.map((row, i) => {
    const r = Array.isArray(row) ? row : [];
    const x = typeof r[0] === 'number' ? r[0] : 0;
    const y = typeof r[1] === 'number' ? r[1] : 0;
    const label = labels && typeof labels[i] === 'string' ? (labels[i] as string) : undefined;
    return { x, y, label, cluster: i };
  });
}

/**
 * Full-screen, zoomable inspector for embedding scatter plots. Solves the
 * "hundreds of labels overlap into mush on the node" problem: pan/zoom into a
 * region, and only the points nearest the view centre (mirrored in the left
 * sidebar) get labelled. The sidebar also lets you temporarily hide noisy
 * points so a cluster reads cleanly.
 */
export function ScatterModal({ isOpen, ...rest }: ScatterModalProps) {
  // Mount the body only while open so the fetch / key / resize listeners are
  // plain mount effects and stale data is dropped on close — same shape as
  // HeatmapModal.
  if (!isOpen) return null;
  return <ScatterModalBody {...rest} />;
}

function ScatterModalBody({
  onClose,
  title,
  inlinePoints,
  runId,
  nodeId,
  pointsPort = 'points_2d',
  labelsPort = 'labels',
}: Omit<ScatterModalProps, 'isOpen'>) {
  const { t } = useI18n();

  const hasInline = !!inlinePoints && inlinePoints.length > 0;
  const canFetch = !hasInline && !!runId && !!nodeId;
  const [fetched, setFetched] = useState<ScatterPoint[] | null>(null);
  const [loading, setLoading] = useState(canFetch);
  const [error, setError] = useState<string | null>(
    !hasInline && !canFetch ? t('scatter.unavailable') : null,
  );

  // REST-fetch the full projection when it wasn't embedded inline.
  useEffect(() => {
    if (!canFetch) return;
    let cancelled = false;
    Promise.all([
      fetchOutput(runId as string, nodeId as string, pointsPort, { maxElements: 200_000 }),
      fetchOutput(runId as string, nodeId as string, labelsPort, { maxElements: 200_000 }).catch(
        () => null,
      ),
    ])
      .then(([coords, labels]) => {
        if (cancelled) return;
        if (coords.type !== 'tensor' || !('values' in coords)) {
          setError(t('scatter.loadError', { error: `expected tensor, got ${coords.type}` }));
          return;
        }
        const labelVals =
          labels && 'values' in labels && Array.isArray((labels as { values?: unknown }).values)
            ? (labels as { values: unknown[] }).values
            : null;
        setFetched(buildPoints((coords as { values: unknown }).values, labelVals));
      })
      .catch((e) => {
        if (cancelled) return;
        setError(t('scatter.loadError', { error: e?.message ?? String(e) }));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [canFetch, runId, nodeId, pointsPort, labelsPort, t]);

  const points = hasInline ? (inlinePoints as ScatterPoint[]) : fetched;

  // Track viewport so the plot grows/shrinks live with the window.
  const [viewport, setViewport] = useState(() => ({
    // window always exists under jsdom / the browser, so the SSR fallbacks are
    // unreachable in any environment this runs in.
    /* v8 ignore start */
    w: typeof window !== 'undefined' ? window.innerWidth : 1280,
    h: typeof window !== 'undefined' ? window.innerHeight : 800,
    /* v8 ignore stop */
  }));
  useEffect(() => {
    const onResize = () => setViewport({ w: window.innerWidth, h: window.innerHeight });
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const plotW = Math.max(320, Math.round(viewport.w * 0.9) - 264 - 48);
  const plotH = Math.max(280, Math.round(viewport.h * 0.88) - 132);

  const bounds = useMemo(() => {
    if (!points || points.length === 0) return null;
    let xMin = Infinity;
    let xMax = -Infinity;
    let yMin = Infinity;
    let yMax = -Infinity;
    for (const p of points) {
      if (p.x < xMin) xMin = p.x;
      if (p.x > xMax) xMax = p.x;
      if (p.y < yMin) yMin = p.y;
      if (p.y > yMax) yMax = p.y;
    }
    return { xMin, xMax, yMin, yMax };
  }, [points]);

  const fitView = useCallback((): View => {
    if (!bounds) return { scale: 1, cx: 0, cy: 0 };
    const { xMin, xMax, yMin, yMax } = bounds;
    const xr = xMax - xMin;
    const yr = yMax - yMin;
    // Fall back to the other axis (or 1) when an extent collapses to a line or
    // a single point, so the scale stays finite.
    const effXr = xr || yr || 1;
    const effYr = yr || xr || 1;
    const scale = 0.88 * Math.min(plotW / effXr, plotH / effYr);
    return { scale, cx: (xMin + xMax) / 2, cy: (yMin + yMax) / 2 };
  }, [bounds, plotW, plotH]);

  const fitScale = useMemo(() => fitView().scale, [fitView]);
  const minScale = fitScale * 0.25;
  const maxScale = fitScale * 500;

  const [view, setView] = useState<View | null>(null);
  // Fit once when the data first arrives. Later resizes keep the user's pan/zoom.
  useEffect(() => {
    if (points && points.length > 0 && view === null) setView(fitView());
  }, [points, view, fitView]);

  const v = view ?? fitView();

  const [hidden, setHidden] = useState<Set<number>>(() => new Set());
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  // Tooltip carries its own display data (label + world coords) so it never has
  // to index back into `points` — keeps the render guard a single condition.
  const [tip, setTip] = useState<{
    i: number;
    label?: string;
    px: number;
    py: number;
    x: number;
    y: number;
  } | null>(null);
  const [query, setQuery] = useState('');

  // Latest geometry for the native wheel listener and the drag handlers, which
  // outlive a render. Kept current via a post-render effect (no deps) rather
  // than an assignment during render.
  const stateRef = useRef({ v, plotW, plotH, minScale, maxScale });
  useEffect(() => {
    stateRef.current = { v, plotW, plotH, minScale, maxScale };
  });
  const svgRef = useRef<SVGSVGElement | null>(null);

  // Wheel zoom-to-cursor. Attached natively (non-passive) so preventDefault
  // actually suppresses page scroll.
  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const s = stateRef.current;
      const rect = el.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const worldX = s.v.cx + (sx - s.plotW / 2) / s.v.scale;
      const worldY = s.v.cy - (sy - s.plotH / 2) / s.v.scale;
      const factor = Math.exp(-e.deltaY * 0.0015);
      const newScale = Math.max(s.minScale, Math.min(s.maxScale, s.v.scale * factor));
      setView({
        scale: newScale,
        cx: worldX - (sx - s.plotW / 2) / newScale,
        cy: worldY + (sy - s.plotH / 2) / newScale,
      });
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [points]);

  // Drag to pan via window-level listeners (robust when the cursor leaves the
  // svg mid-drag, and avoids jsdom's missing PointerEvent / pointer capture).
  const dragRef = useRef<{ x: number; y: number; cx: number; cy: number; scale: number } | null>(
    null,
  );
  const [isDragging, setIsDragging] = useState(false);

  const onPlotMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    const s = stateRef.current;
    dragRef.current = { x: e.clientX, y: e.clientY, cx: s.v.cx, cy: s.v.cy, scale: s.v.scale };
    setIsDragging(true);
  };

  useEffect(() => {
    if (!isDragging) return;
    const onMove = (e: MouseEvent) => {
      const d = dragRef.current;
      // dragRef is always set at mousedown before this listener is attached, so
      // the null guard is unreachable — it only satisfies the type narrowing.
      /* v8 ignore start */
      if (!d) return;
      /* v8 ignore stop */
      setView({
        scale: d.scale,
        cx: d.cx - (e.clientX - d.x) / d.scale,
        cy: d.cy + (e.clientY - d.y) / d.scale,
      });
    };
    const onUp = () => {
      dragRef.current = null;
      setIsDragging(false);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [isDragging]);

  // ESC closes.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  // These read the effective view `v` (always defined) from the current render,
  // and the handlers are recreated each render, so no stale-closure fallback is
  // needed. Wheel/drag use stateRef instead since they outlive a render.
  const zoomBy = (factor: number) =>
    setView({ ...v, scale: Math.max(minScale, Math.min(maxScale, v.scale * factor)) });
  const resetView = () => setView(fitView());
  const centerOn = (p: ScatterPoint) => setView({ ...v, cx: p.x, cy: p.y });

  const toggleHidden = (i: number) =>
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });

  // Points nearest the view centre, filtered by the label query. This is the
  // sidebar list and also the set that gets labelled on the plot.
  const nearest = useMemo(() => {
    if (!points) return [];
    const q = query.trim().toLowerCase();
    const scored = points.map((p, i) => ({
      i,
      p,
      d2: (p.x - v.cx) ** 2 + (p.y - v.cy) ** 2,
    }));
    const filtered = q
      ? scored.filter((s) => (s.p.label ?? `#${s.i}`).toLowerCase().includes(q))
      : scored;
    filtered.sort((a, b) => a.d2 - b.d2);
    return filtered.slice(0, LIST_MAX);
  }, [points, v.cx, v.cy, query]);

  const labelledSet = useMemo(() => new Set(nearest.map((n) => n.i)), [nearest]);

  // Visible (non-hidden, in-viewport) dots with their screen positions.
  const rendered = useMemo(() => {
    if (!points) return [];
    const out: { i: number; p: ScatterPoint; sx: number; sy: number }[] = [];
    for (let i = 0; i < points.length; i += 1) {
      if (hidden.has(i)) continue;
      const p = points[i];
      const sx = plotW / 2 + (p.x - v.cx) * v.scale;
      const sy = plotH / 2 - (p.y - v.cy) * v.scale;
      if (sx < -CULL_PAD || sx > plotW + CULL_PAD || sy < -CULL_PAD || sy > plotH + CULL_PAD) {
        continue;
      }
      out.push({ i, p, sx, sy });
    }
    return out;
  }, [points, hidden, plotW, plotH, v.cx, v.cy, v.scale]);

  const zoomPct = Math.round((v.scale / fitScale) * 100);
  const ready = !loading && !error && points !== null && points.length > 0;

  return createPortal(
    <>
      <div className={styles.backdrop} onClick={onClose} role="dialog">
        <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
          <div className={styles.header}>
            <span className={styles.title}>{title}</span>
            <button
              type="button"
              onClick={onClose}
              className={styles.closeBtn}
              aria-label={t('scatter.close')}
            >
              <CloseIcon size={17} />
            </button>
          </div>

          <div className={styles.body}>
            <aside className={styles.sidebar}>
              <div className={styles.sidebarHeader}>
                <span className={styles.sidebarTitle}>{t('scatter.nearestToCenter')}</span>
                {hidden.size > 0 && (
                  <button
                    type="button"
                    className={styles.showAllBtn}
                    onClick={() => setHidden(new Set())}
                  >
                    {t('scatter.showAll')}
                  </button>
                )}
              </div>
              <input
                className={styles.search}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={t('scatter.searchPlaceholder')}
                aria-label={t('scatter.searchPlaceholder')}
              />
              <div className={styles.list}>
                {nearest.length === 0 ? (
                  <div className={styles.emptyList}>{t('scatter.noMatches')}</div>
                ) : (
                  nearest.map(({ i, p }) => {
                    const isHidden = hidden.has(i);
                    return (
                      <div
                        key={i}
                        className={`${styles.row} ${hoverIdx === i ? styles.rowActive : ''} ${
                          isHidden ? styles.rowHidden : ''
                        }`}
                        onClick={() => centerOn(p)}
                        onMouseEnter={() => setHoverIdx(i)}
                        onMouseLeave={() => setHoverIdx(null)}
                      >
                        <span
                          className={styles.swatch}
                          style={{ background: getTokenColor(p.cluster ?? i) }}
                        />
                        <span className={styles.label}>{p.label ?? `#${i}`}</span>
                        <button
                          type="button"
                          className={styles.eyeBtn}
                          aria-label={isHidden ? t('scatter.showPoint') : t('scatter.hidePoint')}
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleHidden(i);
                          }}
                        >
                          {isHidden ? <EyeOffIcon size={15} /> : <EyeIcon size={15} />}
                        </button>
                      </div>
                    );
                  })
                )}
              </div>
              <div className={styles.recenterHint}>{t('scatter.recenterHint')}</div>
            </aside>

            <section className={styles.plot}>
              {loading && <div className={styles.status}>{t('scatter.loading')}</div>}
              {error && !loading && (
                <div className={`${styles.status} ${styles.error}`}>{error}</div>
              )}
              {!loading && !error && (!points || points.length === 0) && (
                <div className={styles.status}>{t('scatter.noData')}</div>
              )}
              {ready && (
                <>
                  <div className={styles.toolbar}>
                    <button
                      type="button"
                      className={styles.toolBtn}
                      onClick={() => zoomBy(1 / 1.4)}
                      aria-label={t('scatter.zoomOut')}
                      title={t('scatter.zoomOut')}
                    >
                      <ZoomOutIcon size={16} />
                    </button>
                    <span className={styles.zoomLabel}>{zoomPct}%</span>
                    <button
                      type="button"
                      className={styles.toolBtn}
                      onClick={() => zoomBy(1.4)}
                      aria-label={t('scatter.zoomIn')}
                      title={t('scatter.zoomIn')}
                    >
                      <ZoomInIcon size={16} />
                    </button>
                    <button
                      type="button"
                      className={styles.toolBtn}
                      onClick={resetView}
                      aria-label={t('scatter.resetView')}
                      title={t('scatter.resetView')}
                    >
                      <FitIcon size={16} />
                    </button>
                  </div>
                  <svg
                    ref={svgRef}
                    width={plotW}
                    height={plotH}
                    className={`${styles.canvas} ${isDragging ? styles.canvasDragging : ''}`}
                    onMouseDown={onPlotMouseDown}
                    onMouseLeave={() => setTip(null)}
                  >
                    {/* View-centre crosshair — the reference for the sidebar list. */}
                    <line
                      className={styles.crosshair}
                      x1={plotW / 2 - 7}
                      y1={plotH / 2}
                      x2={plotW / 2 + 7}
                      y2={plotH / 2}
                    />
                    <line
                      className={styles.crosshair}
                      x1={plotW / 2}
                      y1={plotH / 2 - 7}
                      x2={plotW / 2}
                      y2={plotH / 2 + 7}
                    />
                    {rendered.map(({ i, p, sx, sy }) => {
                      const color = getTokenColor(p.cluster ?? i);
                      const active = hoverIdx === i;
                      const showLabel = (labelledSet.has(i) || active) && !!p.label;
                      return (
                        <g key={i}>
                          <circle
                            cx={sx}
                            cy={sy}
                            r={active ? DOT_R_HOVER : DOT_R}
                            fill={color}
                            fillOpacity={0.88}
                            stroke={active ? '#fff' : 'none'}
                            strokeWidth={1}
                            className={styles.dot}
                            data-idx={i}
                            onMouseEnter={(e) => {
                              setHoverIdx(i);
                              setTip({ i, label: p.label, px: p.x, py: p.y, x: e.clientX, y: e.clientY });
                            }}
                            onMouseMove={(e) =>
                              setTip({ i, label: p.label, px: p.x, py: p.y, x: e.clientX, y: e.clientY })
                            }
                            onMouseLeave={() => {
                              setHoverIdx(null);
                              setTip(null);
                            }}
                          />
                          {showLabel && (
                            <text
                              x={sx + 7}
                              y={sy + 3}
                              className={styles.dotLabel}
                              fill={color}
                              fillOpacity={active ? 1 : 0.78}
                            >
                              {p.label}
                            </text>
                          )}
                        </g>
                      );
                    })}
                  </svg>
                </>
              )}
            </section>
          </div>

          <div className={styles.footer}>
            <span>
              {t('scatter.points', { count: points?.length ?? 0 })}
              {ready && <span className={styles.dim}> · {zoomPct}%</span>}
              {hidden.size > 0 && (
                <span className={styles.dim}>
                  {' '}
                  · {t('scatter.hiddenCount', { count: hidden.size })}
                </span>
              )}
            </span>
            <span className={styles.dim}>{t('scatter.closeHint')}</span>
          </div>
        </div>
      </div>
      {tip && (
        <div className={styles.tooltip} style={{ left: tip.x + 14, top: tip.y + 14 }}>
          <div className={styles.tooltipLabel}>{tip.label ?? `#${tip.i}`}</div>
          <div className={styles.tooltipCoords}>
            ({tip.px.toFixed(3)}, {tip.py.toFixed(3)})
          </div>
        </div>
      )}
    </>,
    document.body,
  );
}
