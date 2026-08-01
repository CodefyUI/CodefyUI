import { useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import styles from './HistogramPlot.module.css';

/** One bar. `x0`/`x1` are the bin edges for a continuous histogram. */
export interface HistogramBar {
  label: string;
  count: number;
  x0?: number;
  x1?: number;
}

interface HistogramPlotProps {
  bars: HistogramBar[];
  width?: number;
  height?: number;
  /**
   * Categorical bars get a gap between them and their own tick label;
   * continuous bins sit flush so the shape of the distribution reads as one
   * curve rather than 64 separate things.
   */
  variant?: 'continuous' | 'categorical';
  /** Axis captions under the left and right ends of the plot. */
  axisMin?: string;
  axisMax?: string;
  className?: string;
  /** Accessible name; also the SVG's `aria-label`. */
  ariaLabel?: string;
}

interface HoverState {
  index: number;
  screenX: number;
  screenY: number;
}

/**
 * Pure-SVG bar chart for a distribution: either a fixed-bin histogram or a
 * class-balance chart.
 *
 * Same shape as `ScatterPlot` on purpose — stateless with respect to data, no
 * chart library, hover tooltip through a portal so an `overflow: hidden`
 * ancestor cannot clip it. The two components are the repo's plotting
 * convention and a third one would be a third thing to keep in sync.
 */
export function HistogramPlot({
  bars,
  width = 420,
  height = 140,
  variant = 'continuous',
  axisMin,
  axisMax,
  className,
  ariaLabel,
}: HistogramPlotProps) {
  const [hover, setHover] = useState<HoverState | null>(null);

  const { rects, maxCount } = useMemo(() => {
    if (bars.length === 0) return { rects: [], maxCount: 0 };
    const maxCount = Math.max(...bars.map((b) => b.count));
    const gap = variant === 'categorical' ? Math.min(4, 240 / bars.length) : 0;
    const slot = width / bars.length;
    const barWidth = Math.max(1, slot - gap);
    const rects = bars.map((bar, i) => {
      // A non-empty bin always gets a visible sliver: a bar rounded to zero
      // pixels reads as "no data here", which is the opposite of the truth.
      const scaled = maxCount > 0 ? (bar.count / maxCount) * height : 0;
      const h = bar.count > 0 ? Math.max(1.5, scaled) : 0;
      return { bar, i, x: i * slot + gap / 2, w: barWidth, h, y: height - h };
    });
    return { rects, maxCount };
  }, [bars, width, height, variant]);

  if (bars.length === 0) {
    return (
      <div className={`${styles.empty} ${className ?? ''}`} style={{ width, height }}>
        <span>no distribution</span>
      </div>
    );
  }

  return (
    <div className={`${styles.wrapper} ${className ?? ''}`}>
      <svg
        width={width}
        height={height}
        className={styles.svg}
        role="img"
        aria-label={ariaLabel}
        onMouseLeave={() => setHover(null)}
      >
        {rects.map(({ bar, i, x, w, h, y }) => (
          <g key={i}>
            {/* A full-height transparent target: hovering a one-pixel bar in
                the tail is otherwise a game of skill. */}
            <rect
              x={x}
              y={0}
              width={w}
              height={height}
              fill="transparent"
              onMouseEnter={(e) => setHover({ index: i, screenX: e.clientX, screenY: e.clientY })}
              onMouseMove={(e) => setHover({ index: i, screenX: e.clientX, screenY: e.clientY })}
            />
            {bar.count > 0 && (
              <rect
                className={`${styles.bar} ${hover?.index === i ? styles.barActive : ''}`}
                x={x}
                y={y}
                width={w}
                height={h}
                data-count={bar.count}
                data-label={bar.label}
                pointerEvents="none"
              />
            )}
          </g>
        ))}
        <line x1={0} y1={height - 0.5} x2={width} y2={height - 0.5} className={styles.baseline} />
      </svg>

      {(axisMin !== undefined || axisMax !== undefined) && (
        <div className={styles.axisRow} style={{ width }}>
          <span>{axisMin}</span>
          <span className={styles.axisPeak}>peak {maxCount.toLocaleString()}</span>
          <span>{axisMax}</span>
        </div>
      )}

      {/* Indexed defensively: a refetch can swap a 64-bin histogram for a
          2-bar class balance while the pointer is still down on bin 40. */}
      {hover && bars[hover.index] &&
        createPortal(
          <div
            className={styles.tooltip}
            style={{ left: hover.screenX + 12, top: hover.screenY + 12 }}
          >
            <div className={styles.tooltipLabel}>{bars[hover.index].label}</div>
            <div className={styles.tooltipCount}>
              {bars[hover.index].count.toLocaleString()}
            </div>
          </div>,
          document.body,
        )}
    </div>
  );
}
