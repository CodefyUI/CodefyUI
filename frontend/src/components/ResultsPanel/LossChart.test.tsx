import { describe, it, expect, afterEach, vi, beforeEach } from 'vitest';
import { render, act } from '@testing-library/react';
import { LossChart } from './LossChart';

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
