import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ChartView } from './ChartView';
import { useI18n } from '../../i18n';
import type { LogChartPayload } from '../../store/tabStore';

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
});

const BAR: LogChartPayload = {
  kind: 'bar',
  title: 'Mean petal length by species',
  x_label: 'species',
  y_label: 'cm',
  bars: [
    { label: 'setosa', value: 1.46 },
    { label: 'versicolor', value: 4.26 },
    { label: 'virginica', value: 5.55 },
  ],
};

function kindOf(container: HTMLElement): string | null {
  return container.querySelector('[data-chart-kind]')?.getAttribute('data-chart-kind') ?? null;
}

describe('ChartView', () => {
  it('renders a bar spec through the histogram plot, one bar per entry', () => {
    const { container } = render(<ChartView chart={BAR} />);
    expect(kindOf(container)).toBe('bar');
    const bars = [...container.querySelectorAll('rect[data-count]')];
    expect(bars.map((r) => r.getAttribute('data-label'))).toEqual([
      'setosa',
      'versicolor',
      'virginica',
    ]);
    expect(bars.map((r) => r.getAttribute('data-count'))).toEqual(['1.46', '4.26', '5.55']);
  });

  it('shows the title, the axis captions and a note', () => {
    render(<ChartView chart={{ ...BAR, note: '2 values excluded' }} />);
    expect(screen.getByText('Mean petal length by species')).toBeInTheDocument();
    expect(screen.getByText('2 values excluded')).toBeInTheDocument();
    expect(screen.getByText('species')).toBeInTheDocument();
    expect(screen.getByText('cm')).toBeInTheDocument();
  });

  it('renders a line spec as a polyline per series', () => {
    const { container } = render(
      <ChartView
        chart={{
          kind: 'line',
          series: [
            { name: 'a', points: [[0, 1], [1, 2]] },
            { name: 'b', points: [[0, 3], [1, 4]] },
          ],
        }}
      />,
    );
    expect(kindOf(container)).toBe('line');
    expect(container.querySelectorAll('polyline')).toHaveLength(2);
    expect(screen.getByText('a')).toBeInTheDocument();
  });

  it('names an unnamed line series so its colour and key stay stable', () => {
    render(<ChartView chart={{ kind: 'line', series: [{ name: '', points: [[0, 1]] }] }} />);
    expect(screen.getByText('series 1')).toBeInTheDocument();
  });

  it('renders a scatter spec as one dot per point', () => {
    const { container } = render(
      <ChartView
        chart={{
          kind: 'scatter',
          points: [
            { x: 1, y: 2, label: 'p' },
            { x: 3, y: 4, label: 'q' },
          ],
        }}
      />,
    );
    expect(kindOf(container)).toBe('scatter');
    expect(container.querySelectorAll('circle').length).toBeGreaterThanOrEqual(2);
  });

  it('renders a heatmap spec as a cell grid with both axes labelled', () => {
    const { container } = render(
      <ChartView
        chart={{
          kind: 'heatmap',
          matrix: [[13, 0], [1, 12]],
          row_labels: ['setosa', 'versicolor'],
          col_labels: ['setosa', 'versicolor'],
          vmin: 0,
          vmax: 13,
        }}
      />,
    );
    expect(kindOf(container)).toBe('heatmap');
    expect(container.querySelectorAll('rect[data-i]')).toHaveLength(4);
    expect(screen.getAllByText('setosa').length).toBeGreaterThanOrEqual(2);
  });

  it('colours a heatmap against the spec range, not a hard-coded 0..1', () => {
    // Without `vmax`, every count above 1 would clamp to full brightness and
    // a 13-vs-1 confusion would look like a 13-vs-13 one.
    const { container } = render(
      <ChartView
        chart={{ kind: 'heatmap', matrix: [[13, 1]], vmin: 0, vmax: 13 }}
      />,
    );
    const cells = [...container.querySelectorAll('rect[data-i]')];
    expect(cells[0].getAttribute('data-color-t')).toBe('1.000');
    expect(Number(cells[1].getAttribute('data-color-t'))).toBeCloseTo(1 / 13, 3);
  });

  it('does not stripe a lower-triangular heatmap as causally masked', () => {
    // A confusion matrix is often zero above the diagonal by accident, which
    // the attention-oriented default would report as a mask that never was.
    const { container } = render(
      <ChartView chart={{ kind: 'heatmap', matrix: [[1, 0], [1, 1]], vmin: 0, vmax: 1 }} />,
    );
    const masked = [...container.querySelectorAll('rect[data-masked="true"]')];
    expect(masked).toHaveLength(0);
  });

  // ── boundaries ────────────────────────────────────────────────────────────

  it('says so, rather than throwing, when the kind postdates the editor', () => {
    render(<ChartView chart={{ kind: 'sankey' }} />);
    expect(
      screen.getByText('This chart kind (sankey) needs a newer editor'),
    ).toBeInTheDocument();
  });

  // A known kind with a missing payload is a producer bug, not an old editor.
  // Telling the reader to upgrade would send them to the wrong place.
  it.each([
    ['heatmap', { kind: 'heatmap' }],
    ['bar', { kind: 'bar' }],
    ['line', { kind: 'line' }],
    ['scatter', { kind: 'scatter' }],
  ])('reports a %s missing its payload as malformed, not as too new', (kind, chart) => {
    render(<ChartView chart={chart as LogChartPayload} />);
    expect(
      screen.getByText(`This ${kind} chart arrived without its data`),
    ).toBeInTheDocument();
    expect(screen.queryByText(/needs a newer editor/)).toBeNull();
  });

  it('draws an empty-but-present payload rather than calling it malformed', () => {
    // `bars: []` is a real answer ("nothing in range"), unlike a missing key.
    render(<ChartView chart={{ kind: 'bar', bars: [] }} />);
    expect(screen.queryByText(/arrived without its data/)).toBeNull();
  });

  it('captions the y axis of a line chart', () => {
    // LossChart draws x itself but has no y caption, so the footer must carry
    // it — otherwise every line chart silently loses its unit.
    render(
      <ChartView
        chart={{
          kind: 'line',
          series: [{ name: 's', points: [[0, 1]] }],
          x_label: 'step',
          y_label: 'loss',
        }}
      />,
    );
    expect(screen.getByText('loss')).toBeInTheDocument();
    // x is not repeated: LossChart already prints it under its own axis.
    expect(screen.getAllByText('step')).toHaveLength(1);
  });

  it('renders an empty bar chart without crashing', () => {
    const { container } = render(<ChartView chart={{ kind: 'bar', bars: [] }} />);
    expect(kindOf(container)).toBe('bar');
    expect(screen.getByText('no distribution')).toBeInTheDocument();
  });

  it('localizes its own strings', () => {
    useI18n.setState({ locale: 'zh-TW' });
    render(<ChartView chart={{ kind: 'sankey' }} />);
    expect(screen.getByText(/需要更新版的編輯器/)).toBeInTheDocument();
  });
});
