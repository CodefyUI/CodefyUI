import { describe, it, expect, afterEach, vi, beforeEach } from 'vitest';
import { render, act } from '@testing-library/react';
import { LossChart, SERIES_COLORS } from './LossChart';

/**
 * Control the ResizeObserver + clientWidth so the chart's `chartW`/`chartH`
 * branches are deterministic. The global setup installs a no-op
 * ResizeObserver; here we install a controllable one that captures the
 * callback so a test can drive a contentRect-width resize event.
 */
let observerCb: ((entries: any[]) => void) | null = null;

beforeEach(() => {
  observerCb = null;
  (globalThis as any).ResizeObserver = class {
    constructor(cb: (entries: any[]) => void) {
      observerCb = cb;
    }
    observe() {}
    unobserve() {}
    disconnect() {}
  };
  // jsdom returns 0 for clientWidth; give the container a non-zero width so the
  // initial measure produces a positive chart area.
  Object.defineProperty(HTMLElement.prototype, 'clientWidth', {
    configurable: true,
    get() {
      return 240;
    },
  });
});

afterEach(() => {
  vi.restoreAllMocks();
  // remove the clientWidth override
  delete (HTMLElement.prototype as any).clientWidth;
});

describe('LossChart', () => {
  it('renders nothing when there are no losses', () => {
    const { container } = render(<LossChart losses={[]} />);
    expect(container.querySelector('svg')).toBeNull();
    expect(container.firstChild).toBeNull();
  });

  it('renders a single-point chart (centered x, range fallback)', () => {
    // single value: max === min so `range = max - min || 1` falls to 1
    const { container } = render(<LossChart losses={[0.5]} height={120} />);
    const svg = container.querySelector('svg');
    expect(svg).not.toBeNull();
    // polyline has exactly one point
    const polyline = container.querySelector('polyline');
    expect(polyline).not.toBeNull();
    const pts = polyline!.getAttribute('points') ?? '';
    expect(pts.trim().split(' ').filter(Boolean).length).toBe(1);
    // current-point dot present
    expect(container.querySelector('circle')).not.toBeNull();
    // x-axis end label shows count of 1
    expect(container.textContent).toContain('1');
    expect(container.textContent).toContain('epoch');
  });

  it('renders a multi-point chart with three y ticks and end label = count', () => {
    const { container } = render(<LossChart losses={[2, 1.5, 1, 0.5]} height={120} />);
    const polyline = container.querySelector('polyline');
    const pts = (polyline!.getAttribute('points') ?? '').trim().split(' ').filter(Boolean);
    expect(pts.length).toBe(4);
    // 3 y-tick groups (yMax, midpoint, yMin)
    const texts = Array.from(container.querySelectorAll('text')).map((t) => t.textContent);
    // last epoch label equals the number of losses
    expect(texts).toContain('4');
    // axis tick labels formatted with toFixed(2) for values >= 1
    expect(container.querySelectorAll('line').length).toBe(3);
  });

  it('formats ticks: exponential for tiny values (< 0.001)', () => {
    const { container } = render(<LossChart losses={[0.0001, 0.0002, 0.0003]} height={120} />);
    const text = container.textContent ?? '';
    // toExponential(1) produces an "e" in the rendered tick label
    expect(text).toMatch(/e[+-]?\d/i);
  });

  it('formats ticks: toFixed(3) for values in [0.001, 1)', () => {
    const { container } = render(<LossChart losses={[0.2, 0.4, 0.6]} height={120} />);
    const texts = Array.from(container.querySelectorAll('text')).map((t) => t.textContent ?? '');
    // some tick should have 3 decimal places like 0.xxx
    expect(texts.some((t) => /^\d\.\d{3}$/.test(t))).toBe(true);
  });

  it('handles an equal-valued series (degenerate min === max range)', () => {
    const { container } = render(<LossChart losses={[1, 1, 1]} height={120} />);
    // still renders a polyline with 3 points and does not throw on /0
    const polyline = container.querySelector('polyline');
    const pts = (polyline!.getAttribute('points') ?? '').trim().split(' ').filter(Boolean);
    expect(pts.length).toBe(3);
    expect(container.querySelector('circle')).not.toBeNull();
  });

  it('returns empty geometry when chart area is non-positive (tiny height)', () => {
    // height < padding (top+bottom = 22) makes chartH <= 0 -> early-return path
    const { container } = render(<LossChart losses={[1, 2, 3]} height={10} />);
    const polyline = container.querySelector('polyline');
    // points string is empty because the memo early-returns
    expect(polyline!.getAttribute('points')).toBe('');
    // yTicks empty => no tick lines
    expect(container.querySelectorAll('line').length).toBe(0);
  });

  it('reacts to a ResizeObserver width change', () => {
    const { container } = render(<LossChart losses={[1, 2, 3]} height={120} />);
    expect(observerCb).not.toBeNull();
    // fire a resize with a new contentRect width — drives setSvgWidth branch
    act(() => {
      observerCb!([{ contentRect: { width: 480 } }]);
    });
    const svg = container.querySelector('svg');
    expect(svg!.getAttribute('width')).toBe('480');
  });

  it('uses default height when height prop omitted', () => {
    const { container } = render(<LossChart losses={[1, 2]} />);
    const svg = container.querySelector('svg');
    expect(svg!.getAttribute('height')).toBe('80');
  });

  it('disconnects the observer on unmount', () => {
    const disconnect = vi.fn();
    (globalThis as any).ResizeObserver = class {
      constructor(cb: (entries: any[]) => void) {
        observerCb = cb;
      }
      observe() {}
      unobserve() {}
      disconnect = disconnect;
    };
    const { unmount } = render(<LossChart losses={[1, 2, 3]} />);
    unmount();
    expect(disconnect).toHaveBeenCalled();
  });
});

// ── #124: named multi-series form ────────────────────────────────────────
// The Runs panel charts whatever series a run recorded (train_loss,
// val_loss, lr, …) against real step numbers, so the chart had to grow past
// "one array of losses indexed by position".

describe('LossChart — named series', () => {
  const twoSeries = [
    { name: 'train_loss', points: [{ x: 1, y: 2 }, { x: 2, y: 1 }, { x: 3, y: 0.5 }] },
    { name: 'val_loss', points: [{ x: 1, y: 2.4 }, { x: 3, y: 1.2 }] },
  ];

  it('draws one polyline and one current-value dot per series', () => {
    const { container } = render(<LossChart series={twoSeries} height={120} />);
    const polylines = Array.from(container.querySelectorAll('polyline'));
    expect(polylines).toHaveLength(2);
    expect(
      (polylines[0].getAttribute('points') ?? '').trim().split(' '),
    ).toHaveLength(3);
    expect(
      (polylines[1].getAttribute('points') ?? '').trim().split(' '),
    ).toHaveLength(2);
    expect(container.querySelectorAll('circle')).toHaveLength(2);
  });

  it('colours series from the palette in order and shows a legend', () => {
    const { container } = render(<LossChart series={twoSeries} height={120} />);
    const polylines = Array.from(container.querySelectorAll('polyline'));
    expect(polylines[0].getAttribute('stroke')).toBe(SERIES_COLORS[0]);
    expect(polylines[1].getAttribute('stroke')).toBe(SERIES_COLORS[1]);
    expect(container.textContent).toContain('train_loss');
    expect(container.textContent).toContain('val_loss');
  });

  it('honours a per-series colour override', () => {
    const { container } = render(
      <LossChart series={[{ name: 'lr', points: [{ x: 1, y: 0.1 }], color: '#ff00ff' }]} />,
    );
    expect(container.querySelector('polyline')!.getAttribute('stroke')).toBe('#ff00ff');
  });

  it('shares one y-scale across series so the curves stay comparable', () => {
    // val_loss tops out at 2.4; if each series were scaled on its own, both
    // first points would land at the same height.
    const { container } = render(<LossChart series={twoSeries} height={120} />);
    const [train, val] = Array.from(container.querySelectorAll('polyline'));
    const firstY = (el: Element) =>
      Number((el.getAttribute('points') ?? '').split(' ')[0].split(',')[1]);
    // Bigger loss = higher on the chart = SMALLER svg y.
    expect(firstY(val)).toBeLessThan(firstY(train));
  });

  it('labels the x axis from real step numbers, not positions', () => {
    const { container } = render(
      <LossChart
        series={[{ name: 'loss', points: [{ x: 10, y: 1 }, { x: 40, y: 0.5 }] }]}
        height={120}
        xLabel="step"
      />,
    );
    const texts = Array.from(container.querySelectorAll('text')).map((t) => t.textContent);
    expect(texts).toContain('10');
    expect(texts).toContain('40');
    expect(texts).toContain('step');
    expect(texts).not.toContain('epoch');
  });

  it('centres a single-step series instead of dividing by a zero span', () => {
    const { container } = render(
      <LossChart series={[{ name: 'eval_accuracy', points: [{ x: 1, y: 0.9 }] }]} height={120} />,
    );
    const pts = container.querySelector('polyline')!.getAttribute('points') ?? '';
    expect(pts.split(' ')).toHaveLength(1);
    expect(Number(pts.split(',')[0])).toBeGreaterThan(0);
  });

  it('drops empty series and renders nothing when they are all empty', () => {
    const { container } = render(
      <LossChart series={[{ name: 'loss', points: [] }]} height={120} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it('keeps axis extents honest when there is no room to plot', () => {
    const { container } = render(
      <LossChart
        series={[{ name: 'loss', points: [{ x: 5, y: 1 }, { x: 9, y: 2 }] }]}
        height={10}
      />,
    );
    expect(container.querySelector('polyline')!.getAttribute('points')).toBe('');
    const texts = Array.from(container.querySelectorAll('text')).map((t) => t.textContent);
    expect(texts).toContain('5');
    expect(texts).toContain('9');
  });

  it('takes precedence over a legacy `losses` prop passed alongside it', () => {
    const { container } = render(
      <LossChart losses={[1, 2, 3, 4, 5]} series={[{ name: 'a', points: [{ x: 1, y: 1 }] }]} />,
    );
    expect(container.querySelectorAll('polyline')).toHaveLength(1);
    expect(container.textContent).toContain('a');
  });
});
