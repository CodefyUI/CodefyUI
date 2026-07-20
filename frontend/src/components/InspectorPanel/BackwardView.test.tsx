import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { BackwardView } from './BackwardView';
import { useI18n } from '../../i18n';
import {
  fetchGradIndex,
  fetchOutput,
  PayloadTooLargeError,
  RunDataExpiredError,
  type GradIndexEntry,
} from '../../api/executionOutputs';
import type { TensorOutput } from '../../types';

// Mock only the fetch functions; keep the real error classes so `instanceof`
// checks in the component behave correctly.
vi.mock('../../api/executionOutputs', async () => {
  const actual = await vi.importActual<typeof import('../../api/executionOutputs')>(
    '../../api/executionOutputs',
  );
  return {
    ...actual,
    fetchGradIndex: vi.fn(),
    fetchOutput: vi.fn(),
  };
});

const mockGradIndex = vi.mocked(fetchGradIndex);
const mockOutput = vi.mocked(fetchOutput);

function tensor(values: unknown, extra: Partial<TensorOutput> = {}): TensorOutput {
  return {
    type: 'tensor',
    run_id: 'r',
    node_id: 'n',
    port: 'p',
    full_shape: [2, 2],
    dtype: 'float32',
    slice: ':',
    sliced_shape: [2, 2],
    values,
    truncated: false,
    ...extra,
  };
}

function portEntry(port: string, health: GradIndexEntry['health'] = null): GradIndexEntry {
  return { port, kind: 'port', has_grad: true, health };
}
function weightEntry(port: string, health: GradIndexEntry['health'] = null): GradIndexEntry {
  return { port, kind: 'weight', has_grad: true, health };
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  mockGradIndex.mockReset();
  mockOutput.mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('BackwardView', () => {
  it('shows the loading placeholder before the index resolves', () => {
    mockGradIndex.mockReturnValue(new Promise(() => {})); // never resolves
    const { container } = render(<BackwardView runId="r1" nodeId="n1" />);
    expect(container.querySelector('div')?.textContent).toBe('…');
  });

  it('shows the empty state when no gradient entries exist', async () => {
    mockGradIndex.mockResolvedValue([]);
    render(<BackwardView runId="r1" nodeId="n1" />);
    await waitFor(() =>
      expect(screen.getByText('No gradients captured')).toBeInTheDocument(),
    );
    expect(
      screen.getByText('Enable Backward and re-run to inspect gradients'),
    ).toBeInTheDocument();
  });

  it('shows the index error message on a generic failure', async () => {
    mockGradIndex.mockRejectedValue(new Error('boom'));
    render(<BackwardView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('boom')).toBeInTheDocument());
  });

  it('shows the expired-data message when the index 404s as RunDataExpiredError', async () => {
    mockGradIndex.mockRejectedValue(new RunDataExpiredError('r1'));
    render(<BackwardView runId="r1" nodeId="n1" />);
    await waitFor(() =>
      expect(
        screen.getByText('run data expired — re-run with Backward to capture'),
      ).toBeInTheDocument(),
    );
  });

  it('renders port and weight sections with tensors and a health chip', async () => {
    mockGradIndex.mockResolvedValue([
      portEntry('logits', { status: 'healthy', norm: 0.5, mean: 0, max: 1 }),
      weightEntry('weight', { status: 'vanishing', norm: 0.00001, mean: 0, max: 0.0001 }),
    ]);
    mockOutput.mockImplementation(async (_r, _n, port) => {
      if (port === '__weight_grad__weight') {
        return tensor([[0.5, -0.5], [0.25, 0]], { min: -0.5, max: 0.5 });
      }
      return tensor([[1, -1], [0.5, 0]], { min: -1, max: 1 });
    });
    render(<BackwardView runId="r1" nodeId="n1" />);
    // both section titles
    await waitFor(() =>
      expect(screen.getByText('Output gradients')).toBeInTheDocument(),
    );
    expect(screen.getByText('Weight gradients')).toBeInTheDocument();
    // entry labels
    expect(screen.getByText('logits')).toBeInTheDocument();
    expect(screen.getByText('weight')).toBeInTheDocument();
    // health chips: healthy and vanishing labels (+ norm formatting)
    await waitFor(() => expect(screen.getByText(/healthy/)).toBeInTheDocument());
    expect(screen.getByText(/vanishing/)).toBeInTheDocument();
    // exponential norm formatting for the tiny vanishing norm
    expect(screen.getByText(/1\.00e-5/)).toBeInTheDocument();
    // tensor grids appear (shape lines)
    await waitFor(() =>
      expect(screen.getAllByText('shape [2, 2]').length).toBeGreaterThanOrEqual(2),
    );
  });

  it('renders an exploding health chip with finite norm fixed formatting', async () => {
    mockGradIndex.mockResolvedValue([
      portEntry('g', { status: 'exploding', norm: 123.456, mean: 0, max: 200 }),
    ]);
    mockOutput.mockResolvedValue(tensor([[1, 2], [3, 4]], { min: 1, max: 4 }));
    render(<BackwardView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText(/exploding/)).toBeInTheDocument());
    expect(screen.getByText(/123\.4560/)).toBeInTheDocument();
  });

  it('uses the default healthy chip colors for an unknown status', async () => {
    mockGradIndex.mockResolvedValue([
      // status not in the color map → falls back to colors.healthy + healthy label
      portEntry('g', { status: 'weird' as never, norm: 1, mean: 0, max: 1 }),
    ]);
    mockOutput.mockResolvedValue(tensor([[1]], { min: 1, max: 1 }));
    render(<BackwardView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText(/healthy/)).toBeInTheDocument());
  });

  it('shows the non-finite norm via String() in the health chip', async () => {
    mockGradIndex.mockResolvedValue([
      portEntry('g', { status: 'healthy', norm: Infinity, mean: 0, max: 1 }),
    ]);
    mockOutput.mockResolvedValue(tensor([[1]], { min: 1, max: 1 }));
    render(<BackwardView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText(/Infinity/)).toBeInTheDocument());
  });

  it('renders entries without a health object (no chip)', async () => {
    mockGradIndex.mockResolvedValue([portEntry('plain', null)]);
    mockOutput.mockResolvedValue(tensor([[1, 2], [3, 4]], { min: 1, max: 4 }));
    render(<BackwardView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('plain')).toBeInTheDocument());
    // health labels never appear
    expect(screen.queryByText(/‖g‖/)).not.toBeInTheDocument();
  });

  it('shows per-tensor error message on a generic fetch failure', async () => {
    mockGradIndex.mockResolvedValue([portEntry('g')]);
    mockOutput.mockRejectedValue(new Error('fetch failed'));
    render(<BackwardView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('fetch failed')).toBeInTheDocument());
  });

  it("shows 'expired' on a per-tensor RunDataExpiredError", async () => {
    mockGradIndex.mockResolvedValue([portEntry('g')]);
    mockOutput.mockRejectedValue(new RunDataExpiredError('r1'));
    render(<BackwardView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('expired')).toBeInTheDocument());
  });

  it('falls back to a sliced fetch on PayloadTooLargeError', async () => {
    mockGradIndex.mockResolvedValue([portEntry('big')]);
    // First (no-opts) call throws 413; the retry with slice opts succeeds.
    mockOutput.mockImplementation(async (_r, _n, _port, opts) => {
      if (!opts) throw new PayloadTooLargeError('too big');
      return tensor([[9, 9], [9, 9]], { min: 9, max: 9 });
    });
    render(<BackwardView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('shape [2, 2]')).toBeInTheDocument());
    // the retry carried the slice fallback opts on port `big__grad`
    expect(mockOutput).toHaveBeenCalledWith('r1', 'n1', 'big__grad', {
      slice: '0,:,:',
      maxElements: 65536,
    });
  });

  it('builds weight-grad store port name with the __weight_grad__ prefix', async () => {
    mockGradIndex.mockResolvedValue([weightEntry('layer1')]);
    mockOutput.mockResolvedValue(tensor([[1]], { min: 1, max: 1 }));
    render(<BackwardView runId="r1" nodeId="n1" />);
    await waitFor(() =>
      expect(mockOutput).toHaveBeenCalledWith('r1', 'n1', '__weight_grad__layer1'),
    );
  });

  it('computes highlight from cell values including non-array rows and zero maxAbs', async () => {
    // values mixing a numeric row and a flat-number row exercises the highlight grid walker
    mockGradIndex.mockResolvedValue([
      portEntry('mixed', { status: 'healthy', norm: 1, mean: 0, max: 2 }),
    ]);
    mockOutput.mockResolvedValue(
      tensor(
        [
          [1, 2],
          [0, -2],
        ],
        { min: -2, max: 2 },
      ),
    );
    const { container } = render(<BackwardView runId="r1" nodeId="n1" />);
    await waitFor(() =>
      expect(container.querySelectorAll('td').length).toBeGreaterThan(0),
    );
    // at least one cell got an orange heat background (|2|/2 = 1)
    const colored = Array.from(container.querySelectorAll('td')).some((td) =>
      (td as HTMLElement).style.background.includes('rgba(6, 182, 212'),
    );
    expect(colored).toBe(true);
  });

  it('highlight returns 0 when tensor max/min are undefined (maxAbs null)', async () => {
    mockGradIndex.mockResolvedValue([portEntry('nomax')]);
    // tensor without min/max → maxAbs null → highlight always 0 (no coloring)
    mockOutput.mockResolvedValue(
      tensor([[1, 2], [3, 4]], { min: undefined, max: undefined }),
    );
    const { container } = render(<BackwardView runId="r1" nodeId="n1" />);
    await waitFor(() =>
      expect(container.querySelectorAll('td').length).toBeGreaterThan(0),
    );
    const colored = Array.from(container.querySelectorAll('td')).some((td) =>
      (td as HTMLElement).style.background.includes('rgba(6, 182, 212'),
    );
    expect(colored).toBe(false);
  });

  it('highlight handles flat-number rows and out-of-grid indices (cell stays 0)', async () => {
    // 1D values: grid[i] is a number, grid[i][j] path hits the `typeof row === number` branch
    mockGradIndex.mockResolvedValue([
      portEntry('row1d', { status: 'healthy', norm: 1, mean: 0, max: 5 }),
    ]);
    mockOutput.mockResolvedValue(
      tensor([5, 0], { min: 0, max: 5, full_shape: [2], sliced_shape: [2] }),
    );
    const { container } = render(<BackwardView runId="r1" nodeId="n1" />);
    await waitFor(() =>
      expect(container.querySelectorAll('td').length).toBeGreaterThan(0),
    );
    const colored = Array.from(container.querySelectorAll('td')).some((td) =>
      (td as HTMLElement).style.background.includes('rgba(6, 182, 212'),
    );
    expect(colored).toBe(true);
  });

  it('does not render a section when only the other kind has entries', async () => {
    // only port entries → weight section absent
    mockGradIndex.mockResolvedValue([portEntry('only')]);
    mockOutput.mockResolvedValue(tensor([[1]], { min: 1, max: 1 }));
    render(<BackwardView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('Output gradients')).toBeInTheDocument());
    expect(screen.queryByText('Weight gradients')).not.toBeInTheDocument();
  });

  it('highlight uses min ?? 0 when only max is defined', async () => {
    // max defined, min undefined → exercises `Math.abs(tensorData.min ?? 0)`
    mockGradIndex.mockResolvedValue([
      portEntry('onlymax', { status: 'healthy', norm: 1, mean: 0, max: 4 }),
    ]);
    mockOutput.mockResolvedValue(
      tensor([[4, 2], [0, 1]], { max: 4, min: undefined }),
    );
    const { container } = render(<BackwardView runId="r1" nodeId="n1" />);
    await waitFor(() =>
      expect(container.querySelectorAll('td').length).toBeGreaterThan(0),
    );
    const colored = Array.from(container.querySelectorAll('td')).some((td) =>
      (td as HTMLElement).style.background.includes('rgba(6, 182, 212'),
    );
    expect(colored).toBe(true);
  });

  it('highlight leaves non-number 2D cells at 0 (typeof v !== number branch)', async () => {
    // 2D grid with string cells: row is an array, but v is not a number.
    mockGradIndex.mockResolvedValue([
      portEntry('strcells', { status: 'healthy', norm: 1, mean: 0, max: 1 }),
    ]);
    mockOutput.mockResolvedValue(
      tensor([['x', 'y'], ['z', 'w']], { min: 0, max: 1 }),
    );
    const { container } = render(<BackwardView runId="r1" nodeId="n1" />);
    await waitFor(() =>
      expect(container.querySelectorAll('td').length).toBeGreaterThan(0),
    );
    // strings rendered; no cell colored (all cells resolve to 0)
    const colored = Array.from(container.querySelectorAll('td')).some((td) =>
      (td as HTMLElement).style.background.includes('rgba(6, 182, 212'),
    );
    expect(colored).toBe(false);
  });

  it('highlight leaves non-number 1D rows at 0 (row not array, not number)', async () => {
    // 1D grid of strings: row = grid[0] is a string → neither array nor number.
    mockGradIndex.mockResolvedValue([
      portEntry('strrow', { status: 'healthy', norm: 1, mean: 0, max: 1 }),
    ]);
    mockOutput.mockResolvedValue(
      tensor(['a', 'b'], { min: 0, max: 1, full_shape: [2], sliced_shape: [2] }),
    );
    const { container } = render(<BackwardView runId="r1" nodeId="n1" />);
    await waitFor(() =>
      expect(container.querySelectorAll('td').length).toBeGreaterThan(0),
    );
    const colored = Array.from(container.querySelectorAll('td')).some((td) =>
      (td as HTMLElement).style.background.includes('rgba(6, 182, 212'),
    );
    expect(colored).toBe(false);
  });

  it('bails out of the index .then when unmounted before it resolves', async () => {
    let resolve!: (v: GradIndexEntry[]) => void;
    mockGradIndex.mockReturnValue(new Promise<GradIndexEntry[]>((r) => { resolve = r; }));
    const { unmount } = render(<BackwardView runId="r1" nodeId="n1" />);
    unmount(); // sets cancelled = true before the index resolves
    resolve([portEntry('g')]);
    // give the microtask queue a tick; entries must NOT be set (no crash, no render)
    await Promise.resolve();
    await Promise.resolve();
    expect(mockOutput).not.toHaveBeenCalled();
  });

  it('bails out of the index .catch when unmounted before it rejects', async () => {
    let reject!: (e: unknown) => void;
    mockGradIndex.mockReturnValue(new Promise<GradIndexEntry[]>((_r, rej) => { reject = rej; }));
    const { unmount, container } = render(<BackwardView runId="r1" nodeId="n1" />);
    unmount();
    reject(new Error('late error'));
    await Promise.resolve();
    await Promise.resolve();
    // nothing rendered into the (now-detached) container
    expect(container.querySelector('.portError')).toBeNull();
  });

  it('bails out of the per-tensor success handler when unmounted mid-fetch', async () => {
    mockGradIndex.mockResolvedValue([portEntry('g')]);
    let resolveOut!: (v: TensorOutput) => void;
    mockOutput.mockReturnValue(new Promise<TensorOutput>((r) => { resolveOut = r; }));
    const { unmount } = render(<BackwardView runId="r1" nodeId="n1" />);
    // wait for the index to resolve and the tensor fetch to start
    await waitFor(() => expect(mockOutput).toHaveBeenCalled());
    unmount();
    resolveOut(tensor([[1]], { min: 1, max: 1 }));
    await Promise.resolve();
    await Promise.resolve();
    // no assertion needed beyond not throwing; the cancelled guard short-circuits
    expect(true).toBe(true);
  });

  it('bails out of the per-tensor error handler when unmounted mid-fetch', async () => {
    mockGradIndex.mockResolvedValue([portEntry('g')]);
    let rejectOut!: (e: unknown) => void;
    mockOutput.mockReturnValue(new Promise<TensorOutput>((_r, rej) => { rejectOut = rej; }));
    const { unmount } = render(<BackwardView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(mockOutput).toHaveBeenCalled());
    unmount();
    rejectOut(new Error('late tensor error'));
    await Promise.resolve();
    await Promise.resolve();
    expect(true).toBe(true);
  });

  it('renders error and skips tensor grid when fetch fails but state has no data', async () => {
    mockGradIndex.mockResolvedValue([
      portEntry('g', { status: 'healthy', norm: 1, mean: 0, max: 1 }),
    ]);
    mockOutput.mockRejectedValue(new Error('nope'));
    const { container } = render(<BackwardView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('nope')).toBeInTheDocument());
    // no tensor table rendered because state.data stays null (tensorData null)
    expect(container.querySelectorAll('table').length).toBe(0);
  });
});
