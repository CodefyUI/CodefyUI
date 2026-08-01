import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { HistogramPlot, type HistogramBar } from './HistogramPlot';

const bars: HistogramBar[] = [
  { label: '[0, 1)', count: 4 },
  { label: '[1, 2)', count: 0 },
  { label: '[2, 3)', count: 8 },
];

function barRects(container: HTMLElement): SVGRectElement[] {
  return [...container.querySelectorAll<SVGRectElement>('rect[data-count]')];
}

describe('HistogramPlot', () => {
  it('renders one bar per non-empty bin', () => {
    const { container } = render(<HistogramPlot bars={bars} ariaLabel="Distribution" />);
    const rects = barRects(container);
    expect(rects).toHaveLength(2);
    expect(rects.map((r) => r.getAttribute('data-count'))).toEqual(['4', '8']);
  });

  it('scales bar height against the tallest bin', () => {
    const { container } = render(<HistogramPlot bars={bars} height={100} />);
    const [small, tall] = barRects(container);
    expect(Number(tall.getAttribute('height'))).toBe(100);
    expect(Number(small.getAttribute('height'))).toBeCloseTo(50);
  });

  it('keeps a tiny bin visible instead of rounding it away', () => {
    const { container } = render(
      <HistogramPlot bars={[{ label: 'a', count: 1 }, { label: 'b', count: 100000 }]} height={100} />,
    );
    const [tiny] = barRects(container);
    expect(Number(tiny.getAttribute('height'))).toBeGreaterThanOrEqual(1.5);
  });

  it('renders an empty-state instead of an empty chart', () => {
    render(<HistogramPlot bars={[]} />);
    expect(screen.getByText('no distribution')).toBeInTheDocument();
    expect(document.querySelector('rect[data-count]')).toBeNull();
  });

  it('shows the axis captions and the peak count when given', () => {
    render(<HistogramPlot bars={bars} axisMin="-2" axisMax="3" />);
    expect(screen.getByText('-2')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
    expect(screen.getByText('peak 8')).toBeInTheDocument();
  });

  it('omits the axis row entirely when no captions are supplied', () => {
    render(<HistogramPlot bars={bars} />);
    expect(screen.queryByText(/^peak/)).toBeNull();
  });

  it('opens a tooltip on hover and drops it on leave', () => {
    const { container } = render(<HistogramPlot bars={bars} />);
    // The hover target is the full-height transparent rect, not the bar.
    const targets = container.querySelectorAll('rect[fill="transparent"]');
    fireEvent.mouseEnter(targets[2]);
    expect(screen.getByText('[2, 3)')).toBeInTheDocument();
    expect(screen.getByText('8')).toBeInTheDocument();

    fireEvent.mouseLeave(container.querySelector('svg') as SVGSVGElement);
    expect(screen.queryByText('[2, 3)')).toBeNull();
  });

  it('follows the pointer while it moves across a bin', () => {
    const { container } = render(<HistogramPlot bars={bars} />);
    const targets = container.querySelectorAll('rect[fill="transparent"]');
    fireEvent.mouseEnter(targets[0], { clientX: 10, clientY: 10 });
    fireEvent.mouseMove(targets[0], { clientX: 80, clientY: 60 });
    const tooltip = screen.getByText('[0, 1)').parentElement as HTMLElement;
    expect(tooltip.style.left).toBe('92px');
    expect(tooltip.style.top).toBe('72px');
  });

  it('gaps categorical bars and butts continuous ones together', () => {
    const { container: continuous } = render(
      <HistogramPlot bars={bars} width={300} variant="continuous" />,
    );
    const { container: categorical } = render(
      <HistogramPlot bars={bars} width={300} variant="categorical" />,
    );
    const continuousWidth = Number(barRects(continuous)[0].getAttribute('width'));
    const categoricalWidth = Number(barRects(categorical)[0].getAttribute('width'));
    expect(continuousWidth).toBe(100);
    expect(categoricalWidth).toBeLessThan(continuousWidth);
  });

  it('exposes the accessible name on the plot', () => {
    render(<HistogramPlot bars={bars} ariaLabel="Class balance" />);
    expect(screen.getByLabelText('Class balance')).toBeInTheDocument();
  });

  // ── boundaries ────────────────────────────────────────────────────────────

  it('draws no bars when every bin is empty, and does not divide by zero', () => {
    const empty = [
      { label: 'a', count: 0 },
      { label: 'b', count: 0 },
    ];
    const { container } = render(<HistogramPlot bars={empty} axisMin="0" axisMax="1" />);
    expect(barRects(container)).toHaveLength(0);
    // An all-zero histogram is a real answer (an all-NaN slice), not an error.
    expect(screen.queryByText('no distribution')).toBeNull();
    expect(screen.getByText('peak 0')).toBeInTheDocument();
  });

  it('renders the degenerate single-bin histogram of a constant tensor', () => {
    const { container } = render(
      <HistogramPlot bars={[{ label: '[2.5, 2.5]', count: 100 }]} height={80} />,
    );
    const rects = barRects(container);
    expect(rects).toHaveLength(1);
    expect(Number(rects[0].getAttribute('height'))).toBe(80);
  });

  it('drops the tooltip when the bars change out from under it', () => {
    const { container, rerender } = render(<HistogramPlot bars={bars} />);
    fireEvent.mouseEnter(container.querySelectorAll('rect[fill="transparent"]')[2]);
    expect(screen.getByText('[2, 3)')).toBeInTheDocument();

    // A refetch can swap a 64-bin histogram for a 2-bar class balance.
    rerender(<HistogramPlot bars={[{ label: 'only', count: 1 }]} />);
    expect(screen.queryByText('[2, 3)')).toBeNull();
  });

  it('localizes its own strings', () => {
    // The repo's plots were previously the one place hard-coded English lived.
    render(<HistogramPlot bars={[]} />);
    expect(screen.getByText('no distribution')).toBeInTheDocument();
  });
});
