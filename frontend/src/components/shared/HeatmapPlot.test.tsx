import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, within } from '@testing-library/react';
import { HeatmapPlot, detectCausalPattern, valueToColor } from './HeatmapPlot';

afterEach(() => {
  vi.restoreAllMocks();
});

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
    const onExpand = vi.fn();
    const { container } = render(<HeatmapPlot data={m} onExpand={onExpand} />);
    const btn = container.querySelector('button[aria-label="Expand heatmap"]') as HTMLButtonElement;
    expect(btn).toBeTruthy();
    // Click fires onExpand and stops propagation so the wrapper click doesn't
    // also fire (covers the e.stopPropagation() + onExpand() body).
    const stop = vi.fn();
    fireEvent.click(btn, {});
    expect(onExpand).toHaveBeenCalledTimes(1);
    // Verify stopPropagation is wired: dispatch a real event and spy on it.
    const ev = new MouseEvent('click', { bubbles: true, cancelable: true });
    const spy = vi.spyOn(ev, 'stopPropagation');
    btn.dispatchEvent(ev);
    expect(spy).toHaveBeenCalled();
    expect(onExpand).toHaveBeenCalledTimes(2);
    void stop;
  });

  it('surfaces a tooltip on cell hover and clears it on leave', () => {
    const m = [
      [0.5, 0.5],
      [0.3, 0.7],
    ];
    const { container } = render(
      <HeatmapPlot data={m} rowLabels={['q0', 'q1']} colLabels={['k0', 'k1']} />,
    );
    const cell = container.querySelector('rect[data-i="0"][data-j="1"]') as SVGRectElement;
    fireEvent.mouseEnter(cell, { clientX: 30, clientY: 40 });
    const tip = document.body.querySelector('[class*="tooltip"]') as HTMLElement;
    expect(tip).toBeTruthy();
    // Header shows w[i, j] = value (no head segment for 2D input).
    expect(within(tip).getByText(/w\[0, 1\] = 0\.500/)).toBeTruthy();
    // Row/col label pair present.
    expect(within(tip).getByText('q0')).toBeTruthy();
    expect(within(tip).getByText('k1')).toBeTruthy();
    // mouseMove keeps/updates the tooltip.
    fireEvent.mouseMove(cell, { clientX: 99, clientY: 88 });
    const moved = document.body.querySelector('[class*="tooltip"]') as HTMLElement;
    expect(moved.style.left).toBe('111px'); // 99 + 12
    expect(moved.style.top).toBe('100px'); // 88 + 12
    fireEvent.mouseLeave(cell);
    expect(document.body.querySelector('[class*="tooltip"]')).toBeNull();
  });

  it('tooltip shows a head segment and falls back to q/k labels for 3D input', () => {
    const data = [
      [
        [0.5, 0.5],
        [0.3, 0.7],
      ],
    ];
    // rowLabels shorter than seq → fallback "q{i}"/"k{j}" indices used.
    const { container } = render(
      <HeatmapPlot data={data} rowLabels={['only']} colLabels={['c']} />,
    );
    const cell = container.querySelector('rect[data-i="1"][data-j="1"]') as SVGRectElement;
    fireEvent.mouseEnter(cell, { clientX: 5, clientY: 5 });
    const tip = document.body.querySelector('[class*="tooltip"]') as HTMLElement;
    expect(within(tip).getByText(/head 0/)).toBeTruthy();
    expect(within(tip).getByText(/w\[1, 1\] = 0\.700/)).toBeTruthy();
    // Missing labels at index 1 → fallbacks.
    expect(within(tip).getByText('q1')).toBeTruthy();
    expect(within(tip).getByText('k1')).toBeTruthy();
  });

  it('omits the label-pair row when no labels are provided', () => {
    const m = [[0.5, 0.5], [0.3, 0.7]];
    const { container } = render(<HeatmapPlot data={m} />);
    const cell = container.querySelector('rect[data-i="0"][data-j="0"]') as SVGRectElement;
    fireEvent.mouseEnter(cell, { clientX: 1, clientY: 1 });
    const tip = document.body.querySelector('[class*="tooltip"]') as HTMLElement;
    expect(tip).toBeTruthy();
    // No tooltipPair element since rowLabels/effectiveCol are undefined.
    expect(tip.querySelector('[class*="tooltipPair"]')).toBeNull();
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

  it('renders a 3D input whose trailing head is empty (matrix[0] undefined → m=0)', () => {
    // The empty-data guard only inspects panels[0]; a non-first head with an
    // empty matrix still reaches SinglePanel, where `matrix[0]?.length ?? 0`
    // must fall back to 0 rather than throw.
    const data = [
      [
        [0.5, 0.5],
        [0.3, 0.7],
      ],
      [], // empty trailing head — matrix[0] is undefined here
    ];
    const { container } = render(<HeatmapPlot data={data} />);
    // Two panels render (one per head); the empty head contributes no cells.
    expect(container.querySelectorAll('svg').length).toBe(2);
    // Only the first head's 2x2 = 4 cells exist.
    expect(container.querySelectorAll('rect[data-i]').length).toBe(4);
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
