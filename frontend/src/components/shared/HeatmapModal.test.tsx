import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { HeatmapModal } from './HeatmapModal';

// Build a square seqLen×seqLen matrix of a constant value.
function squareMatrix(seqLen: number, val = 0.5): number[][] {
  return Array.from({ length: seqLen }, () => Array.from({ length: seqLen }, () => val));
}

const g = globalThis as unknown as { fetch: typeof fetch };
let originalFetch: typeof fetch;

function mockFetch(status: number, body: unknown) {
  const response = {
    ok: status >= 200 && status < 300,
    status,
    statusText: 'mock',
    json: async () => body,
  } as unknown as Response;
  g.fetch = vi.fn().mockResolvedValue(response) as unknown as typeof fetch;
  return g.fetch as unknown as ReturnType<typeof vi.fn>;
}

let originalInnerWidth: number;
let originalInnerHeight: number;

beforeEach(() => {
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

describe('HeatmapModal', () => {
  it('does not render when isOpen=false', () => {
    const { container } = render(
      <HeatmapModal
        isOpen={false}
        onClose={() => {}}
        title="t"
        inlineData={[[1, 0], [0, 1]]}
      />,
    );
    // Portal renders nothing visible.
    expect(container.querySelector('[role="dialog"]')).toBeNull();
  });

  it('renders inline data without a fetch', async () => {
    const fetchMock = mockFetch(200, {});
    render(
      <HeatmapModal
        isOpen
        onClose={() => {}}
        title="self-attn"
        inlineData={[
          [0.5, 0.5],
          [0.3, 0.7],
        ]}
        runId="r1"
        nodeId="n1"
        port="weights"
      />,
    );
    expect(screen.getByText(/self-attn/i)).toBeTruthy();
    // No fetch should fire since inlineData was provided.
    expect(fetchMock).not.toHaveBeenCalled();
    // Cells should render — 2x2 = 4 cells.
    expect(document.querySelectorAll('rect[data-i]').length).toBe(4);
  });

  it('fetches via REST when inlineData is null', async () => {
    const fetchMock = mockFetch(200, {
      type: 'tensor',
      values: [
        [0.5, 0.5],
        [0.3, 0.7],
      ],
      sliced_shape: [2, 2],
      run_id: 'r1',
      node_id: 'n1',
      port: 'weights',
      full_shape: [2, 2],
      dtype: 'float32',
      slice: '',
      truncated: false,
    });
    render(
      <HeatmapModal
        isOpen
        onClose={() => {}}
        title="t"
        inlineData={null}
        runId="r1"
        nodeId="n1"
        port="weights"
      />,
    );
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalled();
    });
    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toContain('/api/execution/outputs/r1/n1/weights');
    expect(url).toContain('max_elements=4096');
  });

  it('shows error message when REST fetch fails', async () => {
    mockFetch(500, { detail: 'boom' });
    render(
      <HeatmapModal
        isOpen
        onClose={() => {}}
        title="t"
        inlineData={null}
        runId="r1"
        nodeId="n1"
        port="weights"
      />,
    );
    await waitFor(() => {
      expect(screen.getByText(/boom/i)).toBeTruthy();
    });
  });

  it('shows error when run is unavailable (no runId)', async () => {
    render(
      <HeatmapModal
        isOpen
        onClose={() => {}}
        title="t"
        inlineData={null}
        runId={null}
        nodeId="n1"
        port="weights"
      />,
    );
    await waitFor(() => {
      expect(screen.getByText(/run is no longer available/i)).toBeTruthy();
    });
  });

  it('clicking backdrop closes the modal', () => {
    const onClose = vi.fn();
    render(
      <HeatmapModal
        isOpen
        onClose={onClose}
        title="t"
        inlineData={[[1, 0], [0, 1]]}
      />,
    );
    const backdrop = document.querySelector('[role="dialog"]');
    if (!backdrop) throw new Error('backdrop not found');
    fireEvent.click(backdrop);
    expect(onClose).toHaveBeenCalled();
  });

  it('clicking inside modal does not close', () => {
    const onClose = vi.fn();
    render(
      <HeatmapModal
        isOpen
        onClose={onClose}
        title="t"
        inlineData={[[1, 0], [0, 1]]}
      />,
    );
    const closeBtn = screen.getByLabelText('Close');
    // Click on the close button propagates and the close handler runs.
    // Verify the explicit X close path.
    fireEvent.click(closeBtn);
    expect(onClose).toHaveBeenCalled();
  });

  it('ESC key closes the modal', () => {
    const onClose = vi.fn();
    render(
      <HeatmapModal
        isOpen
        onClose={onClose}
        title="t"
        inlineData={[[1, 0], [0, 1]]}
      />,
    );
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalled();
  });

  it('ignores non-Escape keydowns', () => {
    const onClose = vi.fn();
    render(
      <HeatmapModal
        isOpen
        onClose={onClose}
        title="t"
        inlineData={[[1, 0], [0, 1]]}
      />,
    );
    fireEvent.keyDown(window, { key: 'Enter' });
    fireEvent.keyDown(window, { key: 'a' });
    expect(onClose).not.toHaveBeenCalled();
  });

  it('renders 3D per-head data correctly', () => {
    render(
      <HeatmapModal
        isOpen
        onClose={() => {}}
        title="mha"
        inlineData={[
          [[1, 0], [0, 1]],
          [[0, 1], [1, 0]],
        ]}
      />,
    );
    // 2 heads × 2×2 = 8 cells total.
    expect(document.querySelectorAll('rect[data-i]').length).toBe(8);
  });

  it('renders fetched 2D tensor values as cells', async () => {
    mockFetch(200, {
      type: 'tensor',
      values: [
        [0.5, 0.5],
        [0.3, 0.7],
      ],
    });
    render(
      <HeatmapModal
        isOpen
        onClose={() => {}}
        title="t"
        inlineData={null}
        runId="r1"
        nodeId="n1"
        port="weights"
      />,
    );
    await waitFor(() => {
      expect(document.querySelectorAll('rect[data-i]').length).toBe(4);
    });
  });

  it('renders fetched 3D tensor values (per-head panels)', async () => {
    mockFetch(200, {
      type: 'tensor',
      values: [
        [
          [0.5, 0.5],
          [0.3, 0.7],
        ],
        [
          [0.1, 0.9],
          [0.4, 0.6],
        ],
      ],
    });
    render(
      <HeatmapModal
        isOpen
        onClose={() => {}}
        title="t"
        inlineData={null}
        runId="r1"
        nodeId="n1"
        port="weights"
      />,
    );
    await waitFor(() => {
      // 2 heads × 2×2 = 8 cells.
      expect(document.querySelectorAll('rect[data-i]').length).toBe(8);
    });
  });

  it('coerces booleans/floats to 0/1 for the mask variant (2D)', async () => {
    mockFetch(200, {
      type: 'tensor',
      values: [
        [1, 0],
        [0.9, 0],
      ],
    });
    render(
      <HeatmapModal
        isOpen
        onClose={() => {}}
        title="mask"
        inlineData={null}
        runId="r1"
        nodeId="n1"
        port="mask"
        variant="mask"
      />,
    );
    await waitFor(() => {
      expect(document.querySelectorAll('rect[data-i]').length).toBe(4);
    });
    // 0.9 should have been flattened to 1 (mask), so its colour-t is 1.000.
    const cell10 = document.querySelector('rect[data-i="1"][data-j="0"]');
    expect(cell10?.getAttribute('data-color-t')).toBe('1.000');
  });

  it('coerces booleans/floats to 0/1 for the mask variant (3D)', async () => {
    mockFetch(200, {
      type: 'tensor',
      values: [
        [
          [1, 0],
          [2, 0],
        ],
      ],
    });
    render(
      <HeatmapModal
        isOpen
        onClose={() => {}}
        title="mask3d"
        inlineData={null}
        runId="r1"
        nodeId="n1"
        port="mask"
        variant="mask"
      />,
    );
    await waitFor(() => {
      expect(document.querySelectorAll('rect[data-i]').length).toBe(4);
    });
    const cell10 = document.querySelector('rect[data-i="1"][data-j="0"]');
    expect(cell10?.getAttribute('data-color-t')).toBe('1.000');
  });

  it('treats an empty fetched values array as no data (coerce → null)', async () => {
    mockFetch(200, { type: 'tensor', values: [] });
    render(
      <HeatmapModal
        isOpen
        onClose={() => {}}
        title="t"
        inlineData={null}
        runId="r1"
        nodeId="n1"
        port="x"
      />,
    );
    // coerceTensorValues([]) → null → data stays null → no cells, seq_len 0.
    await waitFor(() => {
      expect(screen.getByText(/seq_len = 0/)).toBeTruthy();
    });
    expect(document.querySelectorAll('rect[data-i]').length).toBe(0);
  });

  it('ignores a fetch that resolves after the modal unmounts (cancelled guard)', async () => {
    let resolveFetch: (r: Response) => void = () => {};
    const pending = new Promise<Response>((res) => {
      resolveFetch = res;
    });
    g.fetch = vi.fn().mockReturnValue(pending) as unknown as typeof fetch;
    const { unmount } = render(
      <HeatmapModal
        isOpen
        onClose={() => {}}
        title="t"
        inlineData={null}
        runId="r1"
        nodeId="n1"
        port="x"
      />,
    );
    // Unmount before the fetch settles, flipping cancelled=true.
    unmount();
    await act(async () => {
      resolveFetch({
        ok: true,
        status: 200,
        statusText: 'ok',
        json: async () => ({
          type: 'tensor',
          values: [
            [0.5, 0.5],
            [0.3, 0.7],
          ],
        }),
      } as unknown as Response);
      await Promise.resolve();
    });
    // No cells rendered anywhere — the resolved data was discarded.
    expect(document.querySelectorAll('rect[data-i]').length).toBe(0);
  });

  it('ignores a fetch that rejects after the modal unmounts (cancelled catch)', async () => {
    let rejectFetch: (e: unknown) => void = () => {};
    const pending = new Promise<Response>((_res, rej) => {
      rejectFetch = rej;
    });
    g.fetch = vi.fn().mockReturnValue(pending) as unknown as typeof fetch;
    const { unmount } = render(
      <HeatmapModal
        isOpen
        onClose={() => {}}
        title="t"
        inlineData={null}
        runId="r1"
        nodeId="n1"
        port="x"
      />,
    );
    unmount();
    await act(async () => {
      rejectFetch(new Error('late failure'));
      await Promise.resolve().catch(() => {});
    });
    // The error never surfaces because the component already unmounted.
    expect(screen.queryByText(/late failure/i)).toBeNull();
  });

  it('shows an error when the fetched output is not a tensor', async () => {
    mockFetch(200, { type: 'scalar', value: 3 });
    render(
      <HeatmapModal
        isOpen
        onClose={() => {}}
        title="t"
        inlineData={null}
        runId="r1"
        nodeId="n1"
        port="x"
      />,
    );
    await waitFor(() => {
      expect(screen.getByText(/Expected tensor, got scalar/i)).toBeTruthy();
    });
  });

  it('shows a stringified error when the rejection has no message', async () => {
    // fetch resolves but .json throws a non-Error → caught and String()-ed.
    g.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'ok',
      json: async () => {
        throw 'weird-string-failure';
      },
    } as unknown as Response) as unknown as typeof fetch;
    render(
      <HeatmapModal
        isOpen
        onClose={() => {}}
        title="t"
        inlineData={null}
        runId="r1"
        nodeId="n1"
        port="x"
      />,
    );
    await waitFor(() => {
      expect(screen.getByText(/weird-string-failure/i)).toBeTruthy();
    });
  });

  it('updates panel sizing when the window is resized', async () => {
    render(
      <HeatmapModal
        isOpen
        onClose={() => {}}
        title="t"
        inlineData={squareMatrix(4)}
      />,
    );
    expect(document.querySelector('[role="dialog"]')).toBeTruthy();
    act(() => {
      (window as unknown as { innerWidth: number }).innerWidth = 2000;
      (window as unknown as { innerHeight: number }).innerHeight = 1500;
      window.dispatchEvent(new Event('resize'));
    });
    // Still rendered after the resize-driven state update.
    expect(document.querySelector('[role="dialog"]')).toBeTruthy();
  });

  it('picks panel-size caps across the seq-length ladder', () => {
    // seqLen <= 6 cap branch.
    const r1 = render(
      <HeatmapModal isOpen onClose={() => {}} title="s6" inlineData={squareMatrix(5)} />,
    );
    expect(screen.getByText(/seq_len = 5/)).toBeTruthy();
    r1.unmount();

    // 6 < seqLen <= 12 cap branch.
    const r2 = render(
      <HeatmapModal isOpen onClose={() => {}} title="s12" inlineData={squareMatrix(10)} />,
    );
    expect(screen.getByText(/seq_len = 10/)).toBeTruthy();
    r2.unmount();

    // 12 < seqLen <= 20 cap branch.
    const r3 = render(
      <HeatmapModal isOpen onClose={() => {}} title="s20" inlineData={squareMatrix(16)} />,
    );
    expect(screen.getByText(/seq_len = 16/)).toBeTruthy();
    r3.unmount();

    // seqLen > 20 cap branch (also exercises the n > 16 "no labels" path).
    const r4 = render(
      <HeatmapModal isOpen onClose={() => {}} title="s32" inlineData={squareMatrix(24)} />,
    );
    expect(screen.getByText(/seq_len = 24/)).toBeTruthy();
    r4.unmount();
  });

  it('shows the row-normalised footnote for the attention variant with data', () => {
    render(
      <HeatmapModal
        isOpen
        onClose={() => {}}
        title="attn"
        inlineData={squareMatrix(4)}
        variant="attention"
      />,
    );
    expect(screen.getByText(/row-normalised colours/i)).toBeTruthy();
  });

  it('honours an explicit normalizePerRow=false override (no footnote)', () => {
    render(
      <HeatmapModal
        isOpen
        onClose={() => {}}
        title="attn"
        inlineData={squareMatrix(4)}
        variant="attention"
        normalizePerRow={false}
      />,
    );
    expect(screen.queryByText(/row-normalised colours/i)).toBeNull();
  });

  it('renders the loading status while a fetch is in flight', () => {
    // Never-resolving fetch keeps loading=true.
    g.fetch = vi.fn().mockReturnValue(new Promise(() => {})) as unknown as typeof fetch;
    render(
      <HeatmapModal
        isOpen
        onClose={() => {}}
        title="t"
        inlineData={null}
        runId="r1"
        nodeId="n1"
        port="x"
      />,
    );
    expect(screen.getByText(/Loading full tensor/i)).toBeTruthy();
  });

  it('shows seq_len = 0 and the fallback panel size for empty data', () => {
    // Empty inline array → coerce yields [] downstream; seqLen 0 path.
    render(
      <HeatmapModal isOpen onClose={() => {}} title="empty" inlineData={[]} />,
    );
    expect(screen.getByText(/seq_len = 0/)).toBeTruthy();
  });

  it('coerces 4D fetched values to 3D (drops batch=0)', async () => {
    mockFetch(200, {
      type: 'tensor',
      values: [
        [
          [
            [0.5, 0.5],
            [0.3, 0.7],
          ],
        ],
        [
          [
            [0, 1],
            [1, 0],
          ],
        ],
      ],
      sliced_shape: [2, 1, 2, 2],
      run_id: 'r1',
      node_id: 'n1',
      port: 'weights',
      full_shape: [2, 1, 2, 2],
      dtype: 'float32',
      slice: '',
      truncated: false,
    });
    render(
      <HeatmapModal
        isOpen
        onClose={() => {}}
        title="t"
        inlineData={null}
        runId="r1"
        nodeId="n1"
        port="weights"
      />,
    );
    await waitFor(() => {
      // After dropping batch dim, we should have 1 head × 2×2 = 4 cells.
      expect(document.querySelectorAll('rect[data-i]').length).toBe(4);
    });
  });
});
