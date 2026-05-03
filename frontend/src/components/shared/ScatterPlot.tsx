import { useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { getTokenColor } from '../../styles/theme';
import styles from './ScatterPlot.module.css';

export interface ScatterPoint {
  x: number;
  y: number;
  label?: string;
  /** Optional cluster index, used for color-cycling. Defaults to point index. */
  cluster?: number;
}

interface ScatterPlotProps {
  points: ScatterPoint[];
  width?: number;
  height?: number;
  /** Render labels next to each dot. Useful for word-embedding scatters. */
  showLabels?: boolean;
  /** Padding (in pixels) around the data extent. */
  padding?: number;
  /** Extra CSS class for the wrapper, e.g. for component-scoped sizing. */
  className?: string;
}

interface HoverState {
  index: number;
  screenX: number;
  screenY: number;
}

/**
 * Pure-SVG scatter plot. Stateless w.r.t. data — the parent component owns
 * the points array. Hover surfaces a fixed-position tooltip via React Portal
 * so it isn't clipped by overflow:hidden node bodies.
 */
export function ScatterPlot({
  points,
  width = 320,
  height = 240,
  showLabels = true,
  padding = 16,
  className,
}: ScatterPlotProps) {
  const [hover, setHover] = useState<HoverState | null>(null);

  const { transformed, xMin, xMax, yMin, yMax } = useMemo(() => {
    if (points.length === 0) {
      return { transformed: [], xMin: -1, xMax: 1, yMin: -1, yMax: 1 };
    }
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;

    const innerW = width - padding * 2;
    const innerH = height - padding * 2;

    const transformed = points.map((p, i) => {
      const sx = padding + ((p.x - xMin) / xRange) * innerW;
      // Flip Y — SVG y grows downward.
      const sy = padding + (1 - (p.y - yMin) / yRange) * innerH;
      return { ...p, sx, sy, idx: i };
    });

    return { transformed, xMin, xMax, yMin, yMax };
  }, [points, width, height, padding]);

  if (points.length === 0) {
    return (
      <div className={`${styles.empty} ${className ?? ''}`} style={{ width, height }}>
        <span>no data</span>
      </div>
    );
  }

  return (
    <div className={`${styles.wrapper} ${className ?? ''}`} style={{ width, height }}>
      <svg
        width={width}
        height={height}
        className={styles.svg}
        onMouseLeave={() => setHover(null)}
      >
        {/* Origin axes if range crosses zero */}
        {xMin < 0 && xMax > 0 && (
          <line
            x1={padding + ((0 - xMin) / (xMax - xMin || 1)) * (width - padding * 2)}
            y1={padding}
            x2={padding + ((0 - xMin) / (xMax - xMin || 1)) * (width - padding * 2)}
            y2={height - padding}
            className={styles.axis}
          />
        )}
        {yMin < 0 && yMax > 0 && (
          <line
            x1={padding}
            y1={padding + (1 - (0 - yMin) / (yMax - yMin || 1)) * (height - padding * 2)}
            x2={width - padding}
            y2={padding + (1 - (0 - yMin) / (yMax - yMin || 1)) * (height - padding * 2)}
            className={styles.axis}
          />
        )}

        {/* Dots */}
        {transformed.map((p) => {
          const color = getTokenColor(p.cluster ?? p.idx);
          const isHover = hover?.index === p.idx;
          return (
            <g key={p.idx}>
              <circle
                cx={p.sx}
                cy={p.sy}
                r={isHover ? 5 : 3.5}
                fill={color}
                fillOpacity={0.85}
                stroke={isHover ? '#fff' : 'none'}
                strokeWidth={1}
                className={styles.dot}
                onMouseEnter={(e) => {
                  setHover({ index: p.idx, screenX: e.clientX, screenY: e.clientY });
                }}
                onMouseMove={(e) => {
                  setHover({ index: p.idx, screenX: e.clientX, screenY: e.clientY });
                }}
              />
              {showLabels && p.label && (
                <text
                  x={p.sx + 6}
                  y={p.sy + 3}
                  className={`${styles.label} ${isHover ? styles.labelActive : ''}`}
                  fill={color}
                  fillOpacity={isHover ? 1 : 0.65}
                >
                  {p.label}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      {hover &&
        createPortal(
          <div
            className={styles.tooltip}
            style={{
              left: hover.screenX + 12,
              top: hover.screenY + 12,
            }}
          >
            <div className={styles.tooltipLabel}>{points[hover.index].label ?? `pt ${hover.index}`}</div>
            <div className={styles.tooltipCoords}>
              ({points[hover.index].x.toFixed(3)}, {points[hover.index].y.toFixed(3)})
            </div>
          </div>,
          document.body,
        )}
    </div>
  );
}
