import { useMemo } from 'react';
import { useI18n } from '../../i18n';
import type { LogChartPayload } from '../../store/tabStore';
import { HistogramPlot, type HistogramBar } from './HistogramPlot';
import { ScatterPlot, type ScatterPoint } from './ScatterPlot';
import { HeatmapPlot, type HeatmapColormap } from './HeatmapPlot';
import { LossChart, type ChartSeries } from '../ResultsPanel/LossChart';
import styles from './ChartView.module.css';

const COLORMAPS: HeatmapColormap[] = ['viridis', 'blues', 'RdBu'];

function isColormap(value: string | undefined): value is HeatmapColormap {
  return COLORMAPS.includes(value as HeatmapColormap);
}

interface ChartViewProps {
  chart: LogChartPayload;
  width?: number;
  height?: number;
  className?: string;
}

/**
 * Renders a backend `chart` output entry (#130).
 *
 * Deliberately NOT a fifth plotting component: it dispatches a chart spec to
 * the four the app already has, so a bar chart from a node looks exactly like
 * the port-statistics histogram and a line chart exactly like the loss curve.
 * A spec whose `kind` is unknown renders as a caption rather than as an error
 * or an empty box — the backend's kinds are open strings, so an editor will
 * meet one it predates, and saying so is the honest outcome.
 */
export function ChartView({ chart, width = 420, height = 180, className }: ChartViewProps) {
  const { t } = useI18n();

  const bars: HistogramBar[] = useMemo(
    () => (chart.bars ?? []).map((bar) => ({ label: bar.label, count: bar.value })),
    [chart.bars],
  );

  const series: ChartSeries[] = useMemo(
    () =>
      (chart.series ?? []).map((line, i) => ({
        // An unnamed series still needs a stable key and a stable colour.
        name: line.name || `${t('chart.series')} ${i + 1}`,
        points: line.points.map(([x, y]) => ({ x, y })),
      })),
    [chart.series, t],
  );

  const points: ScatterPoint[] = useMemo(
    () =>
      (chart.points ?? []).map((point) => ({
        x: point.x,
        y: point.y,
        label: point.label,
        cluster: point.cluster,
      })),
    [chart.points],
  );

  let body: React.ReactNode;
  if (chart.kind === 'bar') {
    body = (
      <HistogramPlot
        bars={bars}
        variant="categorical"
        width={width}
        height={height}
        ariaLabel={chart.title || t('chart.bar')}
      />
    );
  } else if (chart.kind === 'line') {
    body = <LossChart series={series} height={height} xLabel={chart.x_label ?? 'x'} />;
  } else if (chart.kind === 'scatter') {
    body = (
      <ScatterPlot
        points={points}
        width={width}
        height={height}
        // Labels on every dot turn a few hundred points into a smear; the
        // hover tooltip already names the one under the pointer.
        showLabels={points.length <= 12}
      />
    );
  } else if (chart.kind === 'heatmap' && chart.matrix?.length) {
    const side = Math.min(width, height + 80);
    body = (
      <HeatmapPlot
        data={chart.matrix}
        rowLabels={chart.row_labels}
        colLabels={chart.col_labels}
        panelWidth={side}
        panelHeight={side}
        colormap={isColormap(chart.colormap) ? chart.colormap : 'viridis'}
        // A confusion matrix is often lower-triangular by accident; the
        // causal-mask stripes would then claim a masking that never happened.
        detectCausalMask={false}
        valueRange={[chart.vmin ?? 0, chart.vmax ?? 1]}
      />
    );
  } else {
    body = (
      <div className={styles.unknown}>{t('chart.unknownKind', { kind: chart.kind })}</div>
    );
  }

  return (
    <div className={`${styles.wrapper} ${className ?? ''}`} data-chart-kind={chart.kind}>
      {chart.title && <div className={styles.title}>{chart.title}</div>}
      {chart.note && <div className={styles.note}>{chart.note}</div>}
      <div className={styles.body}>{body}</div>
      {(chart.x_label || chart.y_label) && chart.kind !== 'line' && (
        <div className={styles.axes}>
          {chart.y_label && <span className={styles.axisY}>{chart.y_label}</span>}
          {chart.x_label && <span className={styles.axisX}>{chart.x_label}</span>}
        </div>
      )}
    </div>
  );
}
