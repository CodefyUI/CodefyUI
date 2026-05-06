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
});
