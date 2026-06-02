import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, within } from '@testing-library/react';
import { ScatterPlot, type ScatterPoint } from './ScatterPlot';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('ScatterPlot', () => {
  it('renders the empty state when there are no points', () => {
    const { container } = render(<ScatterPlot points={[]} className="extra" />);
    expect(screen.getByText('no data')).toBeTruthy();
    // No SVG in the empty branch; wrapper carries the extra class.
    expect(container.querySelector('svg')).toBeNull();
    expect(container.firstElementChild?.className).toContain('extra');
  });

  it('renders the empty state with no className (covers the ?? fallback)', () => {
    const { container } = render(<ScatterPlot points={[]} />);
    expect(screen.getByText('no data')).toBeTruthy();
    // className omitted → trailing space only, no extra token.
    expect(container.firstElementChild?.className.trim()).not.toContain(' ');
  });

  it('renders one circle per point', () => {
    const points: ScatterPoint[] = [
      { x: 0, y: 0, label: 'a' },
      { x: 1, y: 1, label: 'b' },
      { x: 2, y: 0.5 },
    ];
    const { container } = render(<ScatterPlot points={points} />);
    expect(container.querySelectorAll('circle').length).toBe(3);
  });

  it('renders labels when showLabels is true and the point has a label', () => {
    const { container } = render(
      <ScatterPlot points={[{ x: 0, y: 0, label: 'hello' }]} showLabels />,
    );
    const texts = Array.from(container.querySelectorAll('text')).map((t) => t.textContent);
    expect(texts).toContain('hello');
  });

  it('hides labels when showLabels is false', () => {
    const { container } = render(
      <ScatterPlot points={[{ x: 0, y: 0, label: 'hidden' }]} showLabels={false} />,
    );
    expect(container.querySelector('text')).toBeNull();
  });

  it('does not render a <text> for a point without a label even when showLabels', () => {
    const { container } = render(
      <ScatterPlot points={[{ x: 0, y: 0 }]} showLabels />,
    );
    expect(container.querySelector('text')).toBeNull();
  });

  it('draws the vertical zero axis only when x range crosses zero', () => {
    const crosses = render(<ScatterPlot points={[{ x: -1, y: 1 }, { x: 1, y: 2 }]} />);
    // axis line is the <line> element(s); x-crossing produces the vertical axis.
    expect(crosses.container.querySelectorAll('line').length).toBeGreaterThanOrEqual(1);
    crosses.unmount();

    const noCross = render(<ScatterPlot points={[{ x: 1, y: 1 }, { x: 2, y: 2 }]} />);
    // x all positive, y all positive → neither axis drawn.
    expect(noCross.container.querySelectorAll('line').length).toBe(0);
  });

  it('draws the horizontal zero axis only when y range crosses zero', () => {
    const { container } = render(
      <ScatterPlot points={[{ x: 1, y: -1 }, { x: 2, y: 1 }]} />,
    );
    // y crosses zero → horizontal axis present; x all positive → no vertical.
    expect(container.querySelectorAll('line').length).toBe(1);
  });

  it('draws both axes when both ranges cross zero', () => {
    const { container } = render(
      <ScatterPlot points={[{ x: -1, y: -1 }, { x: 1, y: 1 }]} />,
    );
    expect(container.querySelectorAll('line').length).toBe(2);
  });

  it('handles a degenerate single point (zero range) without crashing', () => {
    const { container } = render(<ScatterPlot points={[{ x: 5, y: 5 }]} />);
    // Range falls back to 1; one circle, no zero axes (range doesn\'t cross 0).
    expect(container.querySelectorAll('circle').length).toBe(1);
    expect(container.querySelectorAll('line').length).toBe(0);
  });

  it('handles equal-coordinate points (xRange/yRange === 0 fallback)', () => {
    const { container } = render(
      <ScatterPlot points={[{ x: 2, y: 3 }, { x: 2, y: 3 }]} />,
    );
    expect(container.querySelectorAll('circle').length).toBe(2);
  });

  it('hovering a dot enlarges it, highlights label, and shows a tooltip via portal', () => {
    const { container } = render(
      <ScatterPlot points={[{ x: 1, y: 1, label: 'tok' }]} />,
    );
    const circle = container.querySelector('circle') as SVGCircleElement;
    fireEvent.mouseEnter(circle, { clientX: 100, clientY: 200 });
    // Tooltip is portaled to document.body.
    const tip = document.body.querySelector('[class*="tooltip"]');
    expect(tip).toBeTruthy();
    expect(within(tip as HTMLElement).getByText('tok')).toBeTruthy();
    // Coordinates formatted to 3 decimals.
    expect(within(tip as HTMLElement).getByText(/\(1\.000, 1\.000\)/)).toBeTruthy();
    // Hover radius grows to 5.
    expect(circle.getAttribute('r')).toBe('5');
  });

  it('mouseMove updates the hover position', () => {
    const { container } = render(<ScatterPlot points={[{ x: 1, y: 1, label: 'm' }]} />);
    const circle = container.querySelector('circle') as SVGCircleElement;
    fireEvent.mouseEnter(circle, { clientX: 10, clientY: 10 });
    fireEvent.mouseMove(circle, { clientX: 50, clientY: 60 });
    const tip = document.body.querySelector('[class*="tooltip"]') as HTMLElement;
    expect(tip).toBeTruthy();
    // left = screenX + 12, top = screenY + 12.
    expect(tip.style.left).toBe('62px');
    expect(tip.style.top).toBe('72px');
  });

  it('falls back to "pt N" in the tooltip when the hovered point has no label', () => {
    const { container } = render(<ScatterPlot points={[{ x: 0, y: 0 }]} />);
    const circle = container.querySelector('circle') as SVGCircleElement;
    fireEvent.mouseEnter(circle, { clientX: 0, clientY: 0 });
    const tip = document.body.querySelector('[class*="tooltip"]') as HTMLElement;
    expect(within(tip).getByText('pt 0')).toBeTruthy();
  });

  it('mouseLeave on the svg clears the tooltip', () => {
    const { container } = render(<ScatterPlot points={[{ x: 1, y: 1, label: 'x' }]} />);
    const circle = container.querySelector('circle') as SVGCircleElement;
    fireEvent.mouseEnter(circle, { clientX: 5, clientY: 5 });
    expect(document.body.querySelector('[class*="tooltip"]')).toBeTruthy();
    fireEvent.mouseLeave(container.querySelector('svg') as SVGSVGElement);
    expect(document.body.querySelector('[class*="tooltip"]')).toBeNull();
  });

  it('uses point.cluster for colour cycling when provided (else index)', () => {
    // Just exercise both code paths; assert the circles render with a fill.
    const { container } = render(
      <ScatterPlot
        points={[
          { x: 0, y: 0, cluster: 2 },
          { x: 1, y: 1 },
        ]}
      />,
    );
    const circles = container.querySelectorAll('circle');
    expect(circles[0].getAttribute('fill')).toBeTruthy();
    expect(circles[1].getAttribute('fill')).toBeTruthy();
  });
});
