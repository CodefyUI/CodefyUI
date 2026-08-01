import { useEffect, useMemo, useRef, useState } from 'react';
import styles from './ResultsPanel.module.css';

/** One named line. `points` must be sorted by `x` ascending. */
export interface ChartSeries {
  name: string;
  points: { x: number; y: number }[];
  /** Override the palette pick — otherwise assigned by position. */
  color?: string;
}

/**
 * Palette for named series. Amber first on purpose: a single-series chart
 * built from `series` then looks exactly like the Training tab's loss curve
 * has always looked, so the Runs panel does not introduce a second visual
 * identity for the same quantity.
 */
export const SERIES_COLORS = [
  '#FFC107', // amber   — train_loss / the one-series case
  '#06b6d4', // teal    — the app accent, usually val_loss
  '#66BB6A', // green   — lr
  '#EF5350', // red
  '#AB47BC', // purple
  '#42A5F5', // blue
] as const;

interface LossChartProps {
  /**
   * Legacy single-series form: one y per epoch, x implied as 1..N. Kept
   * because the Training tab charts a series it derives from progress
   * entries and has no step numbers of its own.
   */
  losses?: number[];
  /** Multi-series form (#124) — takes precedence over `losses` when given. */
  series?: ChartSeries[];
  height?: number;
  /** X-axis caption. `epoch` for the Training tab, `step` for run metrics. */
  xLabel?: string;
}

export function LossChart({
  losses,
  series,
  height = 80,
  xLabel = 'epoch',
}: LossChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [svgWidth, setSvgWidth] = useState(240);

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setSvgWidth(entry.contentRect.width);
      }
    });
    observer.observe(containerRef.current);
    setSvgWidth(containerRef.current.clientWidth);
    return () => observer.disconnect();
  }, []);

  const padding = { top: 6, right: 8, bottom: 16, left: 36 };
  const chartW = svgWidth - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  // One normalised shape for both prop forms, so everything below this line
  // only ever deals with named series over explicit x values.
  const lines: ChartSeries[] = useMemo(() => {
    if (series) return series.filter((s) => s.points.length > 0);
    if (!losses || losses.length === 0) return [];
    return [{ name: 'loss', points: losses.map((y, i) => ({ x: i + 1, y })) }];
  }, [series, losses]);

  const { paths, yMin, yMax, yTicks, xMin, xMax } = useMemo(() => {
    const empty = {
      paths: [] as { points: string; last: { x: number; y: number } | null }[],
      yMin: 0,
      yMax: 1,
      yTicks: [] as number[],
      xMin: 0,
      xMax: 0,
    };
    if (lines.length === 0) return empty;

    // Extents span EVERY series so two curves stay comparable — a per-series
    // y-scale would make a val_loss of 3.0 sit level with a train_loss of
    // 0.3 and quietly invert what the picture says.
    const ys = lines.flatMap((s) => s.points.map((p) => p.y));
    const xs = lines.flatMap((s) => s.points.map((p) => p.x));

    if (chartW <= 0 || chartH <= 0) {
      // Still hand back one empty path per line: the legacy contract renders
      // a <polyline points=""> at a non-positive chart size rather than
      // dropping the element. The axis extents are honest even here — the
      // labels are drawn from them regardless of whether there is room to
      // plot anything between them.
      return {
        ...empty,
        paths: lines.map(() => ({ points: '', last: null })),
        xMin: Math.min(...xs),
        xMax: Math.max(...xs),
      };
    }

    const min = Math.min(...ys);
    const max = Math.max(...ys);
    const range = max - min || 1;
    const yLo = min - range * 0.05;
    const yHi = max + range * 0.05;
    const xLo = Math.min(...xs);
    const xHi = Math.max(...xs);
    const xSpan = xHi - xLo;

    const projectX = (x: number) =>
      padding.left + (xSpan === 0 ? chartW / 2 : ((x - xLo) / xSpan) * chartW);
    const projectY = (y: number) =>
      padding.top + chartH - ((y - yLo) / (yHi - yLo)) * chartH;

    const paths = lines.map((line) => {
      const pts = line.points.map((p) => `${projectX(p.x)},${projectY(p.y)}`);
      const tail = line.points[line.points.length - 1];
      return {
        points: pts.join(' '),
        last: { x: projectX(tail.x), y: projectY(tail.y) },
      };
    });

    return {
      paths,
      yMin: yLo,
      yMax: yHi,
      yTicks: [yHi, (yHi + yLo) / 2, yLo],
      xMin: xLo,
      xMax: xHi,
    };
  }, [lines, chartW, chartH, padding.left, padding.top]);

  if (lines.length === 0) return null;

  const formatTick = (v: number) =>
    v < 0.001 ? v.toExponential(1) : v < 1 ? v.toFixed(3) : v.toFixed(2);

  const colorOf = (line: ChartSeries, i: number) =>
    line.color ?? SERIES_COLORS[i % SERIES_COLORS.length];

  return (
    <div className={styles.chartContainer} ref={containerRef}>
      {/* Legend only in the named-series form: the legacy single loss curve
          has nothing to disambiguate, and adding a caption to it would
          change every existing Training-tab screenshot for no information. */}
      {series && (
        <div className={styles.chartLegend}>
          {lines.map((line, i) => (
            <span key={line.name} className={styles.chartLegendItem}>
              <span
                className={styles.chartLegendSwatch}
                style={{ background: colorOf(line, i) }}
              />
              {line.name}
            </span>
          ))}
        </div>
      )}
      <svg width={svgWidth} height={height} className={styles.chartSvg}>
        {/* Y axis ticks */}
        {yTicks.map((tick, i) => {
          // yMax-yMin is 0 only with a single equal point; the ||1 guard is never the chosen arm in practice
          /* v8 ignore start */
          const y = padding.top + chartH - ((tick - yMin) / (yMax - yMin || 1)) * chartH;
          /* v8 ignore stop */
          return (
            <g key={i}>
              <line
                x1={padding.left} y1={y}
                x2={svgWidth - padding.right} y2={y}
                stroke="#333" strokeWidth={0.5}
              />
              <text x={padding.left - 3} y={y + 3} textAnchor="end" fill="#777" fontSize={9}>
                {formatTick(tick)}
              </text>
            </g>
          );
        })}
        {/* X axis labels */}
        <text x={padding.left} y={height - 2} fill="#777" fontSize={9}>{xMin}</text>
        <text x={padding.left + chartW} y={height - 2} fill="#777" fontSize={9} textAnchor="end">
          {xMax}
        </text>
        <text x={padding.left + chartW / 2} y={height - 2} fill="#666" fontSize={8} textAnchor="middle">
          {xLabel}
        </text>
        {/* One polyline + current-value dot per series */}
        {paths.map((path, i) => (
          <g key={lines[i].name}>
            <polyline
              points={path.points}
              fill="none"
              stroke={colorOf(lines[i], i)}
              strokeWidth={1.5}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            {path.last && (
              <circle cx={path.last.x} cy={path.last.y} r={3} fill={colorOf(lines[i], i)} />
            )}
          </g>
        ))}
      </svg>
    </div>
  );
}
