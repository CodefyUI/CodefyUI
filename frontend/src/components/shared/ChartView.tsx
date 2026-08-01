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

  // A kind this build knows, whose spec is missing the payload that kind needs
  // (a heatmap with no `matrix`), is a different failure from a kind that
  // postdates the editor — and saying "needs a newer editor" about it sends
  // the reader to upgrade instead of to the node that produced it.
  const KNOWN: string[] = ['bar', 'line', 'scatter', 'heatmap'];
  const malformed =
    KNOWN.includes(chart.kind) &&
    ((chart.kind === 'bar' && !chart.bars) ||
      (chart.kind === 'line' && !chart.series) ||
      (chart.kind === 'scatter' && !chart.points) ||
      (chart.kind === 'heatmap' && !chart.matrix?.length));

  let body: React.ReactNode;
  if (malformed) {
    body = (
      <div className={styles.unknown}>
        {t('chart.malformed', { kind: chart.kind })}
      </div>
    );
  } else if (chart.kind === 'bar') {
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
    // LossChart draws the x caption itself but has no y caption, so the
    // footer below carries y_label for this kind too — dropping it silently
    // lost the unit on every line chart.
    body = (
      <LossChart series={series} height={height} xLabel={chart.x_label ?? 'x'} />
    );
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
  } else if (chart.kind === 'heatmap') {
    const side = Math.min(width, height + 80);
    body = (
      <HeatmapPlot
        data={chart.matrix as number[][]}
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

  const showY = Boolean(chart.y_label);
  const showX = Boolean(chart.x_label) && chart.kind !== 'line';

  return (
    <div className={`${styles.wrapper} ${className ?? ''}`} data-chart-kind={chart.kind}>
      {chart.title && <div className={styles.title}>{chart.title}</div>}
      {chart.note && <div className={styles.note}>{chart.note}</div>}
      <div className={styles.body}>{body}</div>
      {/* LossChart already prints x under its own axis, so repeating it here
          would caption the same axis twice — which means a line chart with
          only an x_label has nothing to show, and the row must not render at
          all rather than as an empty strip of padding. */}
      {(showY || showX) && (
        <div className={styles.axes}>
          {showY && <span className={styles.axisY}>{chart.y_label}</span>}
          {showX && <span className={styles.axisX}>{chart.x_label}</span>}
        </div>
      )}
    </div>
  );
}
