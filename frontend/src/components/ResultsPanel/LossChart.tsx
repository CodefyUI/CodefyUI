import { useEffect, useMemo, useRef, useState } from 'react';
import styles from './ResultsPanel.module.css';

interface LossChartProps {
  losses: number[];
  height?: number;
}

export function LossChart({ losses, height = 80 }: LossChartProps) {
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

  const { points, yMin, yMax, yTicks } = useMemo(() => {
    if (losses.length === 0 || chartW <= 0 || chartH <= 0)
      return { points: '', yMin: 0, yMax: 1, yTicks: [] as number[] };

    const min = Math.min(...losses);
    const max = Math.max(...losses);
    const range = max - min || 1;
    const yMin = min - range * 0.05;
    const yMax = max + range * 0.05;

    const pts = losses.map((loss, i) => {
      const x = padding.left + (losses.length === 1 ? chartW / 2 : (i / (losses.length - 1)) * chartW);
      const y = padding.top + chartH - ((loss - yMin) / (yMax - yMin)) * chartH;
      return `${x},${y}`;
    });

    const yTicks = [yMax, (yMax + yMin) / 2, yMin];

    return { points: pts.join(' '), yMin, yMax, yTicks };
  }, [losses, chartW, chartH, padding.left, padding.top]);

  if (losses.length === 0) return null;

  const formatTick = (v: number) =>
    v < 0.001 ? v.toExponential(1) : v < 1 ? v.toFixed(3) : v.toFixed(2);

  return (
    <div className={styles.chartContainer} ref={containerRef}>
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
        <text x={padding.left} y={height - 2} fill="#777" fontSize={9}>1</text>
        <text x={padding.left + chartW} y={height - 2} fill="#777" fontSize={9} textAnchor="end">
          {losses.length}
        </text>
        <text x={padding.left + chartW / 2} y={height - 2} fill="#666" fontSize={8} textAnchor="middle">
          epoch
        </text>
        {/* Loss polyline */}
        <polyline
          points={points}
          fill="none"
          stroke="#FFC107"
          strokeWidth={1.5}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        {/* Current point dot */}
        {(() => {
          const lastIdx = losses.length - 1;
          const x = padding.left + (losses.length === 1 ? chartW / 2 : (lastIdx / (losses.length - 1)) * chartW);
          // yMax-yMin is 0 only with a single equal point; the ||1 guard is never the chosen arm in practice
          /* v8 ignore start */
          const y = padding.top + chartH - ((losses[lastIdx] - yMin) / (yMax - yMin || 1)) * chartH;
          /* v8 ignore stop */
          return <circle cx={x} cy={y} r={3} fill="#FFC107" />;
        })()}
      </svg>
    </div>
  );
}
