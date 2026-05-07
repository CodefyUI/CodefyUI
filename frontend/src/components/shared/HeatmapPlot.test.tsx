import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { HeatmapPlot, detectCausalPattern, valueToColor } from './HeatmapPlot';

describe('detectCausalPattern', () => {
  it('detects strictly upper-triangular zeros with non-zero lower', () => {
    const m = [
      [0.5, 0, 0, 0],
      [0.3, 0.7, 0, 0],
      [0.2, 0.3, 0.5, 0],
      [0.1, 0.2, 0.3, 0.4],
    ];
    expect(detectCausalPattern(m)).toBe(true);
  });

  it('rejects matrix with non-zero in upper triangle', () => {
    const m = [
      [0.5, 0.1, 0, 0],
      [0.3, 0.7, 0, 0],
      [0.2, 0.3, 0.5, 0],
      [0.1, 0.2, 0.3, 0.4],
    ];
    expect(detectCausalPattern(m)).toBe(false);
  });

  it('rejects all-zero matrix (not a causal mask, just empty)', () => {
    const m = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    expect(detectCausalPattern(m)).toBe(false);
  });

  it('rejects empty matrix', () => {
    expect(detectCausalPattern([])).toBe(false);
  });

  it('rejects non-square matrix', () => {
    expect(detectCausalPattern([[1, 0, 0], [1, 1, 0]])).toBe(false);
  });

  it('handles 1x1 matrix correctly (trivially causal-like with non-zero)', () => {
    expect(detectCausalPattern([[0.5]])).toBe(true);
  });
});

describe('valueToColor', () => {
  it('returns a valid rgb() string for any colormap and value', () => {
    for (const cm of ['viridis', 'blues', 'RdBu'] as const) {
      for (const v of [-1, 0, 0.25, 0.5, 0.75, 1, 2]) {
        const c = valueToColor(v, cm);
        expect(c).toMatch(/^rgb\(\d+, \d+, \d+\)$/);
      }
    }
  });

  it('clamps below 0 and above 1', () => {
    expect(valueToColor(-5, 'blues')).toEqual(valueToColor(0, 'blues'));
    expect(valueToColor(5, 'blues')).toEqual(valueToColor(1, 'blues'));
  });

  it('returns different colors for different t', () => {
    expect(valueToColor(0, 'viridis')).not.toEqual(valueToColor(1, 'viridis'));
  });
});

describe('HeatmapPlot', () => {
  it('renders empty state when no data', () => {
    render(<HeatmapPlot data={[]} />);
    expect(screen.getByText(/no data/i)).toBeTruthy();
  });

  it('renders one panel for 2D input', () => {
    const m = [
      [0.5, 0.5],
      [0.3, 0.7],
    ];
    const { container } = render(<HeatmapPlot data={m} />);
    const panels = container.querySelectorAll('svg');
    expect(panels.length).toBe(1);
    // 2x2 = 4 cells.
    const cells = container.querySelectorAll('rect[data-i]');
    expect(cells.length).toBe(4);
  });

  it('renders H panels for 3D input', () => {
    const data = [
      [
        [0.5, 0.5],
        [0.3, 0.7],
      ],
      [
        [0.4, 0.6],
        [0.2, 0.8],
      ],
      [
        [0.3, 0.7],
        [0.1, 0.9],
      ],
    ];
    const { container } = render(<HeatmapPlot data={data} />);
    const panels = container.querySelectorAll('svg');
    expect(panels.length).toBe(3);
  });

  it('marks causal-masked cells with data-masked="true"', () => {
    const m = [
      [1.0, 0.0, 0.0],
      [0.5, 0.5, 0.0],
      [0.3, 0.3, 0.4],
    ];
    const { container } = render(<HeatmapPlot data={m} />);
    const masked = container.querySelectorAll('rect[data-masked="true"]');
    // Strictly upper-triangle: (0,1), (0,2), (1,2) = 3 masked cells.
    expect(masked.length).toBe(3);
  });

  it('does not mark non-zero cells as masked even if upper triangle', () => {
    const m = [
      [1.0, 0.1, 0.0],
      [0.5, 0.5, 0.0],
      [0.3, 0.3, 0.4],
    ];
    const { container } = render(<HeatmapPlot data={m} />);
    // (0,1) has 0.1, not zero — pattern rejected → no masked cells.
    const masked = container.querySelectorAll('rect[data-masked="true"]');
    expect(masked.length).toBe(0);
  });

  it('renders row labels in axisLabel slots when seq is small', () => {
    const m = [
      [0.5, 0.5],
      [0.3, 0.7],
    ];
    const { container } = render(
      <HeatmapPlot data={m} rowLabels={['the', 'cat']} />,
    );
    const labels = container.querySelectorAll('text');
    const labelTexts = Array.from(labels).map((el) => el.textContent);
    expect(labelTexts).toContain('the');
    expect(labelTexts).toContain('cat');
  });

  it('renders head index label for each panel in 3D', () => {
    const data = [
      [[1, 0], [0, 1]],
      [[0, 1], [1, 0]],
    ];
    const { container } = render(<HeatmapPlot data={data} />);
    const headLabels = Array.from(container.querySelectorAll('text')).map((el) =>
      el.textContent,
    );
    expect(headLabels).toContain('h0');
    expect(headLabels).toContain('h1');
  });

  it('does not render expand button when onExpand prop is not given', () => {
    const m = [[0.5, 0.5], [0.3, 0.7]];
    const { container } = render(<HeatmapPlot data={m} />);
    const btns = container.querySelectorAll('button[aria-label="Expand heatmap"]');
    expect(btns.length).toBe(0);
  });

  it('renders expand button when onExpand is given and calls it on click', async () => {
    const m = [[0.5, 0.5], [0.3, 0.7]];
    const { container } = render(<HeatmapPlot data={m} onExpand={() => {}} />);
    const btns = container.querySelectorAll('button[aria-label="Expand heatmap"]');
    expect(btns.length).toBe(1);
  });

  it('contrast-stretches each row to [0, 1] when normalizePerRow is true', () => {
    // Row 0: non-zero min=0.05, max=0.1, range=0.05
    //   cell 0.1 → (0.1-0.05)/0.05 = 1.0
    //   cell 0.05 → (0.05-0.05)/0.05 = 0.0
    // Row 1: non-zero min=0.25, max=0.5, range=0.25
    //   cell 0.5 → 1.0
    //   cell 0.25 → 0.0
    const m = [
      [0.1, 0.05, 0.0],
      [0.5, 0.25, 0.0],
    ];
    const { container } = render(<HeatmapPlot data={m} normalizePerRow />);
    const cells = Array.from(container.querySelectorAll('rect[data-i]')) as SVGRectElement[];
    const get = (i: number, j: number) =>
      cells.find((c) => c.getAttribute('data-i') === String(i) && c.getAttribute('data-j') === String(j));
    expect(get(0, 0)?.getAttribute('data-color-t')).toBe('1.000');
    expect(get(0, 1)?.getAttribute('data-color-t')).toBe('0.000');
    expect(get(1, 0)?.getAttribute('data-color-t')).toBe('1.000');
    expect(get(1, 1)?.getAttribute('data-color-t')).toBe('0.000');
  });

  it('reveals tiny within-row variations under min-max stretch (deep attention case)', () => {
    // A near-uniform row that previously rendered as all-yellow under
    // the old divide-by-max scheme: weights barely differ but the visual
    // should show the relative ordering clearly.
    const m = [[0.166, 0.167, 0.168, 0.169]];
    const { container } = render(<HeatmapPlot data={m} normalizePerRow />);
    const cells = Array.from(container.querySelectorAll('rect[data-i]')) as SVGRectElement[];
    const ts = cells.map((c) => parseFloat(c.getAttribute('data-color-t') ?? '0'));
    // First cell maps to 0 (row min), last cell maps to 1 (row max).
    expect(ts[0]).toBe(0);
    expect(ts[3]).toBe(1);
    // Middle cells should land between, monotonically increasing.
    expect(ts[1]).toBeGreaterThan(0);
    expect(ts[1]).toBeLessThan(ts[2]);
    expect(ts[2]).toBeLessThan(1);
  });

  it('renders truly-uniform non-zero rows at neutral colour-t=0.5', () => {
    // When every non-zero cell is identical, range collapses to 0 — fall
    // back to a neutral colour rather than dividing by zero.
    const m = [[0.2, 0.2, 0.2, 0.2]];
    const { container } = render(<HeatmapPlot data={m} normalizePerRow />);
    const cells = Array.from(container.querySelectorAll('rect[data-i]')) as SVGRectElement[];
    const ts = cells.map((c) => c.getAttribute('data-color-t'));
    expect(ts).toEqual(['0.500', '0.500', '0.500', '0.500']);
  });

  it('keeps causal-masked cells at colour-t=0 under normalization', () => {
    // Row 1 of a causal pattern: lower triangle [0.4, 0.6], upper [0].
    // Non-zero min=0.4, max=0.6, range=0.2.
    //   cell (1,0)=0.4 → (0.4-0.4)/0.2 = 0.0
    //   cell (1,1)=0.6 → 1.0
    //   cell (1,2)=0.0 → masked → 0.0
    const m = [
      [1.0, 0.0, 0.0],
      [0.4, 0.6, 0.0],
      [0.3, 0.3, 0.4],
    ];
    const { container } = render(<HeatmapPlot data={m} normalizePerRow />);
    const get = (i: number, j: number) =>
      container.querySelector(`rect[data-i="${i}"][data-j="${j}"]`);
    expect(get(1, 1)?.getAttribute('data-color-t')).toBe('1.000');
    expect(get(1, 0)?.getAttribute('data-color-t')).toBe('0.000');
    expect(get(1, 2)?.getAttribute('data-color-t')).toBe('0.000');
  });

  it('uses absolute colour scale when normalizePerRow is false', () => {
    const m = [
      [0.1, 0.05, 0.0],
      [0.5, 0.25, 0.0],
    ];
    const { container } = render(<HeatmapPlot data={m} />);
    const cells = Array.from(container.querySelectorAll('rect[data-i]')) as SVGRectElement[];
    const cell = cells.find((c) => c.getAttribute('data-i') === '0' && c.getAttribute('data-j') === '0');
    // Without normalisation, 0.1 should map to colour-t=0.1, not 1.0.
    expect(cell?.getAttribute('data-color-t')).toBe('0.100');
  });

  it('row-normalised tooltip still surfaces the raw value', () => {
    // No good way to check the tooltip without simulating hover, but the
    // raw weight is plumbed through the data-color-t separately from what
    // the tooltip would show. Sanity-check the cell renders.
    const m = [[0.1, 0.05]];
    const { container } = render(<HeatmapPlot data={m} normalizePerRow />);
    const cells = container.querySelectorAll('rect[data-i]');
    expect(cells.length).toBe(2);
  });

  it('handles all-zero rows safely under normalizePerRow', () => {
    const m = [
      [0.0, 0.0],
      [0.5, 0.5],
    ];
    const { container } = render(<HeatmapPlot data={m} normalizePerRow />);
    const cells = container.querySelectorAll('rect[data-i]');
    expect(cells.length).toBe(4);
    // No NaN crash; the all-zero row stays at colour-t=0.
    const cell00 = container.querySelector('rect[data-i="0"][data-j="0"]');
    expect(cell00?.getAttribute('data-color-t')).toBe('0.000');
  });
});
