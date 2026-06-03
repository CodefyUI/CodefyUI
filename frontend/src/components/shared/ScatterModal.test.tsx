import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, act, within } from '@testing-library/react';
import { useI18n } from '../../i18n';
import { ScatterModal } from './ScatterModal';
import type { ScatterPoint } from './ScatterPlot';

const g = globalThis as unknown as { fetch: typeof fetch };
let originalFetch: typeof fetch;
let originalInnerWidth: number;
let originalInnerHeight: number;

/** Mock fetch, dispatching the response by URL substring (points vs labels). */
function mockFetchByUrl(handler: (url: string) => { status: number; body: unknown }) {
  g.fetch = vi.fn().mockImplementation(async (url: string) => {
    const { status, body } = handler(String(url));
    return {
      ok: status >= 200 && status < 300,
      status,
      statusText: 'mock',
      json: async () => body,
    } as unknown as Response;
  }) as unknown as typeof fetch;
  return g.fetch as unknown as ReturnType<typeof vi.fn>;
}

function inline(n: number): ScatterPoint[] {
  return Array.from({ length: n }, (_, i) => ({
    x: Math.cos((i / n) * Math.PI * 2) * 0.8,
    y: Math.sin((i / n) * Math.PI * 2) * 0.8,
    label: `w${i}`,
    cluster: i,
  }));
}

function zoomLabel(): string {
  return document.querySelector('[class*="zoomLabel"]')?.textContent ?? '';
}

/** The sidebar list, so label queries don't collide with on-plot SVG labels. */
function sidebar(): HTMLElement {
  return document.querySelector('[class*="sidebar"]') as HTMLElement;
}

/** The plot <svg> (class `canvas`) — distinct from the toolbar/button icon SVGs. */
function plotSvg(): SVGSVGElement {
  return document.querySelector('svg[class*="canvas"]') as SVGSVGElement;
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  originalFetch = g.fetch;
  originalInnerWidth = window.innerWidth;
  originalInnerHeight = window.innerHeight;
});

afterEach(() => {
  g.fetch = originalFetch;
  (window as unknown as { innerWidth: number }).innerWidth = originalInnerWidth;
  (window as unknown as { innerHeight: number }).innerHeight = originalInnerHeight;
  vi.restoreAllMocks();
});

describe('ScatterModal — open/close', () => {
  it('renders nothing when isOpen=false', () => {
    render(<ScatterModal isOpen={false} onClose={() => {}} title="t" inlinePoints={inline(3)} />);
    expect(document.querySelector('[role="dialog"]')).toBeNull();
  });

  it('renders inline points without a fetch', () => {
    const fetchMock = mockFetchByUrl(() => ({ status: 200, body: {} }));
    render(<ScatterModal isOpen onClose={() => {}} title="emb" inlinePoints={inline(4)} />);
    expect(screen.getByText('emb')).toBeTruthy();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(document.querySelectorAll('circle[data-idx]').length).toBe(4);
    // Footer reports the count.
    expect(screen.getByText(/4 points/)).toBeTruthy();
  });

  it('ESC closes; other keys are ignored', () => {
    const onClose = vi.fn();
    render(<ScatterModal isOpen onClose={onClose} title="t" inlinePoints={inline(3)} />);
    fireEvent.keyDown(window, { key: 'Enter' });
    expect(onClose).not.toHaveBeenCalled();
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('clicking the backdrop closes; clicking inside does not', () => {
    const onClose = vi.fn();
    render(<ScatterModal isOpen onClose={onClose} title="t" inlinePoints={inline(3)} />);
    const dialog = document.querySelector('[role="dialog"]') as HTMLElement;
    fireEvent.click(dialog.firstElementChild as HTMLElement); // the modal box → stopPropagation
    expect(onClose).not.toHaveBeenCalled();
    fireEvent.click(dialog); // the backdrop
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('the × button closes', () => {
    const onClose = vi.fn();
    render(<ScatterModal isOpen onClose={onClose} title="t" inlinePoints={inline(3)} />);
    fireEvent.click(screen.getByLabelText('Close'));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe('ScatterModal — REST fetch path', () => {
  it('fetches points + labels and renders them with labels', async () => {
    const fetchMock = mockFetchByUrl((url) =>
      url.includes('/labels')
        ? { status: 200, body: { type: 'list', values: ['cat', 'dog'] } }
        : { status: 200, body: { type: 'tensor', values: [[0, 0], [0.5, 0.5]] } },
    );
    render(
      <ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={null} runId="r1" nodeId="n1" />,
    );
    await waitFor(() => expect(document.querySelectorAll('circle[data-idx]').length).toBe(2));
    // Points port requested with the high element cap.
    const urls = fetchMock.mock.calls.map((c) => String(c[0]));
    expect(urls.some((u) => u.includes('/n1/points_2d') && u.includes('max_elements=200000'))).toBe(
      true,
    );
    expect(urls.some((u) => u.includes('/n1/labels'))).toBe(true);
    expect(within(sidebar()).getByText('cat')).toBeTruthy();
  });

  it('renders points without labels when the labels fetch fails', async () => {
    mockFetchByUrl((url) =>
      url.includes('/labels')
        ? { status: 500, body: { detail: 'no labels' } }
        : { status: 200, body: { type: 'tensor', values: [[0, 0], [1, 1]] } },
    );
    render(
      <ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={null} runId="r1" nodeId="n1" />,
    );
    await waitFor(() => expect(document.querySelectorAll('circle[data-idx]').length).toBe(2));
    // No <text> labels because labels couldn't be loaded.
    expect(document.querySelectorAll('text').length).toBe(0);
  });

  it('treats a non-array labels payload as no labels', async () => {
    mockFetchByUrl((url) =>
      url.includes('/labels')
        ? { status: 200, body: { type: 'list' } } // values undefined
        : { status: 200, body: { type: 'tensor', values: [[0, 0]] } },
    );
    render(
      <ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={null} runId="r1" nodeId="n1" />,
    );
    await waitFor(() => expect(document.querySelectorAll('circle[data-idx]').length).toBe(1));
    expect(document.querySelectorAll('text').length).toBe(0);
  });

  it('coerces malformed tensor rows (non-array row / non-number coords) to origin', async () => {
    mockFetchByUrl((url) =>
      url.includes('/labels')
        ? { status: 200, body: { type: 'list', values: [] } }
        : { status: 200, body: { type: 'tensor', values: [[0, 0], ['x', 'y'], 'bad'] } },
    );
    render(
      <ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={null} runId="r1" nodeId="n1" />,
    );
    await waitFor(() => expect(document.querySelectorAll('circle[data-idx]').length).toBe(3));
  });

  it('shows an error when the fetched points output is not a tensor', async () => {
    mockFetchByUrl(() => ({ status: 200, body: { type: 'scalar', value: 3 } }));
    render(
      <ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={null} runId="r1" nodeId="n1" />,
    );
    await waitFor(() => expect(screen.getByText(/expected tensor, got scalar/i)).toBeTruthy());
  });

  it('shows an error when the points fetch fails', async () => {
    mockFetchByUrl((url) =>
      url.includes('/labels')
        ? { status: 200, body: { type: 'list', values: [] } }
        : { status: 500, body: { detail: 'kaboom' } },
    );
    render(
      <ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={null} runId="r1" nodeId="n1" />,
    );
    await waitFor(() => expect(screen.getByText(/kaboom/i)).toBeTruthy());
  });

  it('stringifies a non-Error rejection from the points fetch', async () => {
    g.fetch = vi.fn().mockImplementation(async (url: string) => {
      if (String(url).includes('/labels')) {
        return { ok: true, status: 200, statusText: 'ok', json: async () => ({ type: 'list', values: [] }) } as unknown as Response;
      }
      return {
        ok: true,
        status: 200,
        statusText: 'ok',
        json: async () => {
          throw 'weird-failure';
        },
      } as unknown as Response;
    }) as unknown as typeof fetch;
    render(
      <ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={null} runId="r1" nodeId="n1" />,
    );
    await waitFor(() => expect(screen.getByText(/weird-failure/i)).toBeTruthy());
  });

  it('shows the loading status while the fetch is in flight', () => {
    g.fetch = vi.fn().mockReturnValue(new Promise(() => {})) as unknown as typeof fetch;
    render(
      <ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={null} runId="r1" nodeId="n1" />,
    );
    expect(screen.getByText(/Loading points/i)).toBeTruthy();
  });

  it('shows "unavailable" when there is no inline data and no run to fetch from', () => {
    render(<ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={null} runId={null} />);
    expect(screen.getByText(/run is no longer available/i)).toBeTruthy();
  });

  it('shows the no-data status when the fetch returns an empty tensor', async () => {
    mockFetchByUrl((url) =>
      url.includes('/labels')
        ? { status: 200, body: { type: 'list', values: [] } }
        : { status: 200, body: { type: 'tensor', values: [] } },
    );
    render(
      <ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={null} runId="r1" nodeId="n1" />,
    );
    await waitFor(() => expect(screen.getByText(/No points to display/i)).toBeTruthy());
    expect(document.querySelectorAll('circle[data-idx]').length).toBe(0);
  });

  it('treats a non-array tensor "values" payload as no points', async () => {
    mockFetchByUrl((url) =>
      url.includes('/labels')
        ? { status: 200, body: { type: 'list', values: [] } }
        : { status: 200, body: { type: 'tensor', values: 5 } }, // scalar — not an array
    );
    render(
      <ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={null} runId="r1" nodeId="n1" />,
    );
    await waitFor(() => expect(screen.getByText(/No points to display/i)).toBeTruthy());
    expect(document.querySelectorAll('circle[data-idx]').length).toBe(0);
  });

  it('drops fetched data that resolves after unmount (cancelled guard)', async () => {
    let resolveFetch: (r: Response) => void = () => {};
    const pending = new Promise<Response>((res) => {
      resolveFetch = res;
    });
    g.fetch = vi.fn().mockReturnValue(pending) as unknown as typeof fetch;
    const { unmount } = render(
      <ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={null} runId="r1" nodeId="n1" />,
    );
    unmount();
    await act(async () => {
      resolveFetch({
        ok: true,
        status: 200,
        statusText: 'ok',
        json: async () => ({ type: 'tensor', values: [[0, 0]] }),
      } as unknown as Response);
      await Promise.resolve();
    });
    expect(document.querySelectorAll('circle[data-idx]').length).toBe(0);
  });

  it('swallows a rejection that lands after unmount (cancelled catch)', async () => {
    let rejectFetch: (e: unknown) => void = () => {};
    const pending = new Promise<Response>((_res, rej) => {
      rejectFetch = rej;
    });
    g.fetch = vi.fn().mockReturnValue(pending) as unknown as typeof fetch;
    const { unmount } = render(
      <ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={null} runId="r1" nodeId="n1" />,
    );
    unmount();
    await act(async () => {
      rejectFetch(new Error('late'));
      await Promise.resolve().catch(() => {});
    });
    expect(screen.queryByText(/late/i)).toBeNull();
  });
});

describe('ScatterModal — zoom & pan', () => {
  it('zoom in/out buttons and reset adjust the zoom level', () => {
    render(<ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={inline(5)} />);
    expect(zoomLabel()).toBe('100%');
    fireEvent.click(screen.getByLabelText('Zoom in'));
    expect(zoomLabel()).toBe('140%');
    fireEvent.click(screen.getByLabelText('Zoom out'));
    expect(zoomLabel()).toBe('100%');
    // Zoom in then reset returns to the fitted 100%.
    fireEvent.click(screen.getByLabelText('Zoom in'));
    fireEvent.click(screen.getByLabelText('Reset view'));
    expect(zoomLabel()).toBe('100%');
  });

  it('wheel zooms toward the cursor', () => {
    render(<ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={inline(5)} />);
    const svg = plotSvg();
    fireEvent.wheel(svg, { deltaY: -200, clientX: 50, clientY: 50 });
    // Scrolling up zooms in → > 100%.
    expect(parseInt(zoomLabel(), 10)).toBeGreaterThan(100);
  });

  it('clamps zoom to the max on a huge wheel-in and the min on a huge wheel-out', () => {
    render(<ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={inline(5)} />);
    const svg = plotSvg();
    fireEvent.wheel(svg, { deltaY: -100000, clientX: 0, clientY: 0 });
    expect(zoomLabel()).toBe('50000%'); // fitScale * 500
    fireEvent.wheel(svg, { deltaY: 100000, clientX: 0, clientY: 0 });
    expect(zoomLabel()).toBe('25%'); // fitScale * 0.25
  });

  it('dragging pans the plot (left button); other buttons are ignored', () => {
    render(<ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={inline(4)} />);
    const svg = plotSvg();
    const dot = document.querySelector('circle[data-idx="0"]') as SVGCircleElement;
    const x0 = parseFloat(dot.getAttribute('cx') as string);

    // Right button → no pan.
    fireEvent.mouseDown(svg, { button: 2, clientX: 100, clientY: 100 });
    fireEvent.mouseMove(window, { clientX: 160, clientY: 100 });
    expect(parseFloat(dot.getAttribute('cx') as string)).toBeCloseTo(x0, 3);
    fireEvent.mouseUp(window);

    // Left button drag → content follows the cursor (+60px right).
    fireEvent.mouseDown(svg, { button: 0, clientX: 100, clientY: 100 });
    fireEvent.mouseMove(window, { clientX: 160, clientY: 100 });
    const x1 = parseFloat(dot.getAttribute('cx') as string);
    expect(x1 - x0).toBeCloseTo(60, 0);
    fireEvent.mouseUp(window);

    // After release, further moves do nothing.
    fireEvent.mouseMove(window, { clientX: 400, clientY: 400 });
    expect(parseFloat(dot.getAttribute('cx') as string)).toBeCloseTo(x1, 3);
  });
});

describe('ScatterModal — sidebar list', () => {
  it('lists nearest labels and hides/restores a point', () => {
    render(<ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={inline(4)} />);
    expect(document.querySelectorAll('circle[data-idx]').length).toBe(4);
    // Hide the first listed point via its eye toggle.
    const firstHideBtn = screen.getAllByLabelText('Hide')[0];
    fireEvent.click(firstHideBtn);
    expect(document.querySelectorAll('circle[data-idx]').length).toBe(3);
    // The same row now offers "Show" — toggling it back restores the point.
    fireEvent.click(screen.getAllByLabelText('Show')[0]);
    expect(document.querySelectorAll('circle[data-idx]').length).toBe(4);
    // Hide two, then "Show all" clears them in one click.
    fireEvent.click(screen.getAllByLabelText('Hide')[0]);
    fireEvent.click(screen.getAllByLabelText('Hide')[0]);
    expect(document.querySelectorAll('circle[data-idx]').length).toBe(2);
    fireEvent.click(screen.getByText('Show all'));
    expect(document.querySelectorAll('circle[data-idx]').length).toBe(4);
  });

  it('filters the list by label query and shows the empty state for no matches', () => {
    render(<ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={inline(6)} />);
    const search = screen.getByPlaceholderText(/Filter labels/i);
    fireEvent.change(search, { target: { value: 'w3' } });
    // Only the w3 row remains in the sidebar.
    expect(within(sidebar()).getByText('w3')).toBeTruthy();
    expect(within(sidebar()).queryByText('w0')).toBeNull();
    fireEvent.change(search, { target: { value: 'zzz' } });
    expect(screen.getByText(/No labels match/i)).toBeTruthy();
  });

  it('clicking a label row recentres the view on that point', () => {
    const pts: ScatterPoint[] = [
      { x: 0, y: 0, label: 'origin' },
      { x: 0.9, y: 0.9, label: 'corner' },
      { x: -0.9, y: -0.9, label: 'anti' },
    ];
    render(<ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={pts} />);
    const svg = plotSvg();
    const plotW = Number(svg.getAttribute('width'));
    const plotH = Number(svg.getAttribute('height'));
    fireEvent.click(within(sidebar()).getByText('corner'));
    const dot = document.querySelector('circle[data-idx="1"]') as SVGCircleElement;
    // 'corner' should now sit at the view centre (the crosshair).
    expect(parseFloat(dot.getAttribute('cx') as string)).toBeCloseTo(plotW / 2, 1);
    expect(parseFloat(dot.getAttribute('cy') as string)).toBeCloseTo(plotH / 2, 1);
  });

  it('hovering a row marks it active', () => {
    render(<ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={inline(3)} />);
    const row = within(sidebar()).getByText('w0').closest('div') as HTMLElement;
    fireEvent.mouseEnter(row);
    expect(row.className).toMatch(/rowActive/);
    fireEvent.mouseLeave(row);
    expect(row.className).not.toMatch(/rowActive/);
  });

  it('falls back to "#index" labels and default colour when a point has no label/cluster', () => {
    render(
      <ScatterModal
        isOpen
        onClose={() => {}}
        title="t"
        inlinePoints={[{ x: 0, y: 0 }, { x: 0.2, y: 0.1 }]}
      />,
    );
    // Sidebar rows fall back to #0 / #1.
    expect(screen.getByText('#0')).toBeTruthy();
    // A label query still evaluates the "#index" fallback for unlabeled points.
    const search = screen.getByPlaceholderText(/Filter labels/i);
    fireEvent.change(search, { target: { value: '#1' } });
    expect(screen.getByText('#1')).toBeTruthy();
    expect(screen.queryByText('#0')).toBeNull();
  });
});

describe('ScatterModal — plot interactions', () => {
  it('hovering a dot enlarges it and shows a tooltip; mouseleave clears it', () => {
    render(<ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={inline(3)} />);
    const dot = document.querySelector('circle[data-idx="0"]') as SVGCircleElement;
    fireEvent.mouseEnter(dot, { clientX: 80, clientY: 90 });
    expect(dot.getAttribute('r')).toBe('6');
    const tip = document.querySelector('[class*="tooltip"]') as HTMLElement;
    expect(tip).toBeTruthy();
    expect(tip.textContent).toContain('w0');
    fireEvent.mouseMove(dot, { clientX: 120, clientY: 130 });
    expect((document.querySelector('[class*="tooltip"]') as HTMLElement).style.left).toBe('134px');
    fireEvent.mouseLeave(dot);
    expect(document.querySelector('[class*="tooltip"]')).toBeNull();
  });

  it('mouseLeave on the svg clears the tooltip', () => {
    render(<ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={inline(3)} />);
    const dot = document.querySelector('circle[data-idx="0"]') as SVGCircleElement;
    fireEvent.mouseEnter(dot, { clientX: 10, clientY: 10 });
    expect(document.querySelector('[class*="tooltip"]')).toBeTruthy();
    fireEvent.mouseLeave(plotSvg());
    expect(document.querySelector('[class*="tooltip"]')).toBeNull();
  });

  it('shows the "#index" tooltip fallback for an unlabeled dot', () => {
    render(<ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={[{ x: 0, y: 0 }]} />);
    const dot = document.querySelector('circle[data-idx="0"]') as SVGCircleElement;
    fireEvent.mouseEnter(dot, { clientX: 0, clientY: 0 });
    const tip = document.querySelector('[class*="tooltip"]') as HTMLElement;
    expect(tip.textContent).toContain('#0');
  });

  it('labels only the nearest LIST_MAX points, leaving the rest unlabelled', () => {
    // 65 points all in view → 60 nearest get labels, 5 do not.
    render(<ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={inline(65)} />);
    expect(document.querySelectorAll('circle[data-idx]').length).toBe(65);
    expect(document.querySelectorAll('text').length).toBe(60);
  });

  it('does not render a label for a point that has none even when nearest', () => {
    render(
      <ScatterModal
        isOpen
        onClose={() => {}}
        title="t"
        inlinePoints={[{ x: 0, y: 0 }, { x: 0.1, y: 0.1 }]}
      />,
    );
    // Both are "nearest" but neither has a label → no <text>.
    expect(document.querySelectorAll('text').length).toBe(0);
  });

  it('culls points that leave the viewport when zoomed in', () => {
    // Grid including the origin so at least one dot stays centred.
    const grid: ScatterPoint[] = [];
    for (let r = -2; r <= 2; r += 1) {
      for (let c = -2; c <= 2; c += 1) {
        grid.push({ x: c / 2, y: r / 2, label: `${r},${c}` });
      }
    }
    render(<ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={grid} />);
    expect(document.querySelectorAll('circle[data-idx]').length).toBe(grid.length);
    const zoomIn = screen.getByLabelText('Zoom in');
    for (let i = 0; i < 14; i += 1) fireEvent.click(zoomIn);
    const remaining = document.querySelectorAll('circle[data-idx]').length;
    expect(remaining).toBeGreaterThanOrEqual(1);
    expect(remaining).toBeLessThan(grid.length);
  });

  it('handles a single point (degenerate extent) without crashing', () => {
    render(<ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={[{ x: 5, y: 5, label: 'lonely' }]} />);
    expect(document.querySelectorAll('circle[data-idx]').length).toBe(1);
    expect(zoomLabel()).toBe('100%');
    fireEvent.click(screen.getByLabelText('Reset view'));
    expect(zoomLabel()).toBe('100%');
  });

  it('reports the hidden count in the footer and keeps working after a window resize', () => {
    render(<ScatterModal isOpen onClose={() => {}} title="t" inlinePoints={inline(4)} />);
    fireEvent.click(screen.getAllByLabelText('Hide')[0]);
    expect(screen.getByText(/1 hidden/)).toBeTruthy();
    act(() => {
      (window as unknown as { innerWidth: number }).innerWidth = 1600;
      (window as unknown as { innerHeight: number }).innerHeight = 1000;
      window.dispatchEvent(new Event('resize'));
    });
    expect(document.querySelector('[role="dialog"]')).toBeTruthy();
  });
});
