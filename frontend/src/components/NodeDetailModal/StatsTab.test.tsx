import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import type { Edge, Node } from '@xyflow/react';
import type { NodeData, NodeDefinition } from '../../types';

vi.mock('../../api/executionOutputs', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('../../api/executionOutputs')>();
  return { ...actual, fetchPortStats: vi.fn() };
});

import {
  fetchPortStats,
  StatsNotCapturedError,
  type PortStats,
} from '../../api/executionOutputs';
import { StatsTab, formatStat } from './StatsTab';
import type { NodeDetailTabContext } from './tabs';

const mockStats = vi.mocked(fetchPortStats);

// ── fixtures ─────────────────────────────────────────────────────────────────

const definition = (outputs: string[]): NodeDefinition =>
  ({
    node_name: 'Demo',
    category: 'Utility',
    description: '',
    params: [],
    inputs: [{ name: 'x', data_type: 'TENSOR' }],
    outputs: outputs.map((name) => ({ name, data_type: 'TENSOR' })),
  }) as unknown as NodeDefinition;

const node = (id: string, outputs: string[] = ['out']): Node<NodeData> =>
  ({
    id,
    type: 'baseNode',
    position: { x: 0, y: 0 },
    data: { type: 'Demo', label: id, params: {}, definition: definition(outputs) },
  }) as Node<NodeData>;

function ctx(overrides: Partial<NodeDetailTabContext> = {}): NodeDetailTabContext {
  const nodes = [node('src'), node('n1')];
  const edges: Edge[] = [
    { id: 'e1', source: 'src', target: 'n1', sourceHandle: 'out', targetHandle: 'x' },
  ];
  return {
    nodeId: 'n1',
    node: nodes[1],
    runId: 'run1',
    nodes,
    edges,
    recordOutputs: true,
    outputSummaries: {},
    focusPort: null,
    ...overrides,
  };
}

const tensorStats = (overrides: Partial<PortStats> = {}): PortStats => ({
  run_id: 'run1',
  node_id: 'n1',
  port: 'out',
  kind: 'tensor',
  sampled: false,
  sample_size: null,
  shape: [2, 3],
  dtype: 'torch.float32',
  device: 'cpu',
  count: 6,
  mean: 0.5,
  std: 1.25,
  min: -2,
  max: 3,
  quantiles: { p1: -1.9, p5: -1.5, p25: -0.5, p50: 0.4, p75: 1.5, p95: 2.5, p99: 2.9 },
  nan_count: 0,
  inf_count: 0,
  zero_frac: 0.125,
  histogram: {
    bins: 4,
    edges: [-2, -0.75, 0.5, 1.75, 3],
    counts: [1, 2, 2, 1],
  },
  value_counts: null,
  ...overrides,
});

beforeEach(() => {
  mockStats.mockReset();
  mockStats.mockResolvedValue(tensorStats());
});

// ── rendering ────────────────────────────────────────────────────────────────

describe('StatsTab', () => {
  it('fetches stats for every input and output port of the node', async () => {
    render(<StatsTab ctx={ctx()} />);
    await waitFor(() => expect(mockStats).toHaveBeenCalledTimes(2));
    // Input side is keyed by the UPSTREAM node — that is where the value was
    // captured, and it is what makes the edge tooltip's link resolve.
    expect(mockStats).toHaveBeenCalledWith('run1', 'src', 'out', expect.anything());
    expect(mockStats).toHaveBeenCalledWith('run1', 'n1', 'out', expect.anything());
  });

  it('renders the tensor stat table from the response', async () => {
    render(<StatsTab ctx={ctx()} />);
    await waitFor(() => expect(screen.getAllByText('Shape').length).toBe(2));
    const block = screen.getByTestId('stats-port-n1-out');
    expect(within(block).getByText('[2, 3]')).toBeInTheDocument();
    expect(within(block).getByText('torch.float32')).toBeInTheDocument();
    expect(within(block).getByText('cpu')).toBeInTheDocument();
    expect(within(block).getByText('0.5')).toBeInTheDocument();
    expect(within(block).getByText('1.25')).toBeInTheDocument();
  });

  it('renders the quantile row', async () => {
    render(<StatsTab ctx={ctx()} />);
    await waitFor(() => expect(screen.getAllByText('p50').length).toBe(2));
    const block = screen.getByTestId('stats-port-n1-out');
    expect(within(block).getByText('p1')).toBeInTheDocument();
    expect(within(block).getByText('p99')).toBeInTheDocument();
    expect(within(block).getByText('0.4')).toBeInTheDocument();
  });

  it('draws a histogram bar per bin', async () => {
    render(<StatsTab ctx={ctx()} />);
    const block = await screen.findByTestId('stats-port-n1-out');
    await waitFor(() =>
      expect(within(block).getByLabelText('Distribution')).toBeInTheDocument(),
    );
    const chart = within(block).getByLabelText('Distribution');
    expect(chart.querySelectorAll('rect[data-count]')).toHaveLength(4);
  });

  // ── NaN / Inf prominence ───────────────────────────────────────────────────

  it('shows NaN and Inf counts for a clean tensor without alarm styling', async () => {
    render(<StatsTab ctx={ctx()} />);
    const block = await screen.findByTestId('stats-port-n1-out');
    const nan = within(block).getByTestId('stat-nan');
    expect(nan).toHaveTextContent('NaN 0');
    expect(nan.className).not.toMatch(/healthAlert/);
  });

  it('flags a tensor that contains NaNs', async () => {
    mockStats.mockResolvedValue(tensorStats({ nan_count: 12, inf_count: 3 }));
    render(<StatsTab ctx={ctx()} />);
    const block = await screen.findByTestId('stats-port-n1-out');
    await waitFor(() =>
      expect(within(block).getByTestId('stat-nan')).toHaveTextContent('NaN 12'),
    );
    expect(within(block).getByTestId('stat-nan').className).toMatch(/healthAlert/);
    expect(within(block).getByTestId('stat-inf').className).toMatch(/healthAlert/);
  });

  it('renders null statistics as an em dash rather than NaN', async () => {
    mockStats.mockResolvedValue(
      tensorStats({
        mean: null,
        std: null,
        min: null,
        max: null,
        quantiles: {},
        histogram: null,
        nan_count: 6,
      }),
    );
    render(<StatsTab ctx={ctx()} />);
    const block = await screen.findByTestId('stats-port-n1-out');
    await waitFor(() =>
      expect(within(block).getAllByText('—').length).toBeGreaterThanOrEqual(4),
    );
    // "NaN" appears exactly once, as the health badge's LABEL — never as the
    // rendered value of a statistic.
    expect(within(block).getByTestId('stat-nan')).toHaveTextContent('NaN 6');
    expect(block.textContent?.match(/NaN/g)).toHaveLength(1);
  });

  // ── sampling ───────────────────────────────────────────────────────────────

  it('marks a sampled port and explains what stayed exact', async () => {
    mockStats.mockResolvedValue(
      tensorStats({ sampled: true, sample_size: 1_000_000, count: 500_000_000 }),
    );
    render(<StatsTab ctx={ctx()} />);
    const block = await screen.findByTestId('stats-port-n1-out');
    const chip = await within(block).findByText(/sampled/);
    expect(chip).toHaveTextContent((1_000_000).toLocaleString());
    expect(chip.title).toMatch(/exact/);
    // The true element count survives sampling.
    expect(within(block).getByText((500_000_000).toLocaleString())).toBeInTheDocument();
  });

  // ── class balance ──────────────────────────────────────────────────────────

  it('renders a class-balance chart instead of a histogram for label tensors', async () => {
    mockStats.mockResolvedValue(
      tensorStats({
        dtype: 'torch.int64',
        histogram: null,
        value_counts: [
          { value: 0, count: 900 },
          { value: 1, count: 100 },
        ],
      }),
    );
    render(<StatsTab ctx={ctx()} />);
    const block = await screen.findByTestId('stats-port-n1-out');
    await waitFor(() =>
      expect(within(block).getByLabelText('Class balance')).toBeInTheDocument(),
    );
    expect(within(block).queryByLabelText('Distribution')).toBeNull();
    const bars = within(block)
      .getByLabelText('Class balance')
      .querySelectorAll('rect[data-count]');
    expect(bars).toHaveLength(2);
    expect(bars[0].getAttribute('data-label')).toContain('900');
  });

  it('renders boolean class balance labels', async () => {
    mockStats.mockResolvedValue(
      tensorStats({
        dtype: 'torch.bool',
        histogram: null,
        value_counts: [
          { value: false, count: 3 },
          { value: true, count: 1 },
        ],
      }),
    );
    render(<StatsTab ctx={ctx()} />);
    const block = await screen.findByTestId('stats-port-n1-out');
    await waitFor(() => expect(within(block).getByLabelText('Class balance')).toBeTruthy());
    const bars = within(block)
      .getByLabelText('Class balance')
      .querySelectorAll('rect[data-count]');
    expect(bars[0].getAttribute('data-label')).toContain('false');
  });

  // ── tabular ────────────────────────────────────────────────────────────────

  it('renders a per-column table for a tabular value', async () => {
    mockStats.mockResolvedValue({
      run_id: 'run1',
      node_id: 'n1',
      port: 'out',
      kind: 'tabular',
      rows: 150,
      column_count: 2,
      columns_truncated: false,
      sampled: false,
      sample_size: null,
      columns: [
        {
          name: 'petal_width',
          dtype: 'float',
          count: 148,
          missing: 2,
          unique: 22,
          top: [{ value: 0.2, count: 29 }],
          mean: 1.2,
          min: 0.1,
          max: 2.5,
        },
        {
          name: 'species',
          dtype: 'str',
          count: 150,
          missing: 0,
          unique: 3,
          top: [{ value: 'setosa', count: 50 }],
          mean: null,
          min: null,
          max: null,
        },
      ],
    });
    render(<StatsTab ctx={ctx()} />);
    const block = await screen.findByTestId('stats-port-n1-out');
    await waitFor(() => expect(within(block).getByText('petal_width')).toBeInTheDocument());
    expect(within(block).getByText('species')).toBeInTheDocument();
    expect(within(block).getByText('setosa (50)')).toBeInTheDocument();
    // Row and column totals sit in the badge strip above the table.
    expect(within(block).getByText('Rows').parentElement).toHaveTextContent('Rows 150');
    expect(within(block).getByText('Columns').parentElement).toHaveTextContent(
      'Columns 2',
    );
    // A column with missing cells calls them out.
    expect(within(block).getByText('2', { selector: 'td' })).toBeInTheDocument();
  });

  it('notes when a wide table had its columns truncated', async () => {
    mockStats.mockResolvedValue({
      run_id: 'run1',
      node_id: 'n1',
      port: 'out',
      kind: 'tabular',
      rows: 2,
      column_count: 400,
      columns_truncated: true,
      columns: [
        {
          name: 'c0',
          dtype: 'int',
          count: 2,
          missing: 0,
          unique: 2,
          top: [],
          mean: 1.5,
          min: 1,
          max: 2,
        },
      ],
    });
    render(<StatsTab ctx={ctx()} />);
    const block = await screen.findByTestId('stats-port-n1-out');
    await waitFor(() =>
      expect(within(block).getByText('Showing 1 of 400 columns')).toBeInTheDocument(),
    );
  });

  it('says so plainly for a value type with no statistics', async () => {
    mockStats.mockResolvedValue({
      run_id: 'run1',
      node_id: 'n1',
      port: 'out',
      kind: 'unsupported',
      type: 'Linear',
    });
    render(<StatsTab ctx={ctx()} />);
    const block = await screen.findByTestId('stats-port-n1-out');
    await waitFor(() =>
      expect(
        within(block).getByText('No statistics for this value type (Linear)'),
      ).toBeInTheDocument(),
    );
  });

  // ── states ─────────────────────────────────────────────────────────────────

  it('shows the pre-run empty state and fetches nothing without a run', () => {
    render(<StatsTab ctx={ctx({ runId: null })} />);
    expect(screen.getByText('Node statistics')).toBeInTheDocument();
    expect(mockStats).not.toHaveBeenCalled();
  });

  it("warns when Record outputs is off", async () => {
    render(<StatsTab ctx={ctx({ recordOutputs: false })} />);
    expect(
      screen.getByText('Record outputs is off — re-run with Rec on to capture values'),
    ).toBeInTheDocument();
  });

  it('localizes the not-captured hint instead of echoing the server', async () => {
    // The server detail is English by construction; the one error whose cause
    // is known exactly gets the translated wording.
    mockStats.mockRejectedValue(
      new StatsNotCapturedError(
        "nothing captured for 'n1.out' in run 'run1'. Turn on Record outputs (the Rec toggle in the toolbar) and re-run the graph to capture port data.",
      ),
    );
    render(<StatsTab ctx={ctx()} />);
    await waitFor(() =>
      expect(
        screen.getAllByText('Nothing captured for this port — re-run with Rec on'),
      ).toHaveLength(2),
    );
    expect(screen.queryByText(/Rec toggle in the toolbar/)).toBeNull();
  });

  it('reports an unexpected failure without blanking the tab', async () => {
    mockStats.mockRejectedValue(new Error('boom'));
    render(<StatsTab ctx={ctx()} />);
    await waitFor(() => expect(screen.getAllByText('boom').length).toBe(2));
  });

  it('aborts the previous round when the port set changes', async () => {
    const signals: AbortSignal[] = [];
    mockStats.mockImplementation(async (_r, _n, _p, opts) => {
      if (opts?.signal) signals.push(opts.signal);
      return tensorStats();
    });
    const { rerender } = render(<StatsTab ctx={ctx()} />);
    await waitFor(() => expect(signals.length).toBe(2));

    // Dropping the edge drops the input port, so the set changes.
    rerender(<StatsTab ctx={ctx({ edges: [] })} />);
    await waitFor(() => expect(signals.length).toBeGreaterThan(2));
    expect(signals[0].aborted).toBe(true);
    expect(signals[signals.length - 1].aborted).toBe(false);
  });

  it('aborts in-flight requests when the tab unmounts', async () => {
    let captured: AbortSignal | undefined;
    mockStats.mockImplementation(async (_r, _n, _p, opts) => {
      captured = opts?.signal;
      return tensorStats();
    });
    const { unmount } = render(<StatsTab ctx={ctx()} />);
    await waitFor(() => expect(captured).toBeDefined());
    expect(captured!.aborted).toBe(false);
    unmount();
    expect(captured!.aborted).toBe(true);
  });

  // ── focus port (the edge tooltip's destination) ────────────────────────────

  it('rings the port the edge tooltip pointed at', async () => {
    render(<StatsTab ctx={ctx({ focusPort: 'src::out' })} />);
    const focused = await screen.findByTestId('stats-port-src-out');
    const other = screen.getByTestId('stats-port-n1-out');
    expect(focused.className).toMatch(/portFocused/);
    expect(other.className).not.toMatch(/portFocused/);
  });

  it('rings nothing when the modal was opened without a port', async () => {
    render(<StatsTab ctx={ctx()} />);
    const block = await screen.findByTestId('stats-port-src-out');
    expect(block.className).not.toMatch(/portFocused/);
  });

  it('tells the user when a node has no connected inputs', async () => {
    render(<StatsTab ctx={ctx({ edges: [] })} />);
    expect(screen.getByText('No inputs connected')).toBeInTheDocument();
  });
});

// ── the number formatter ─────────────────────────────────────────────────────

describe('formatStat', () => {
  it('renders an em dash for a statistic that does not exist', () => {
    expect(formatStat(null)).toBe('—');
    expect(formatStat(undefined)).toBe('—');
    expect(formatStat(NaN)).toBe('—');
  });

  it('keeps integers plain', () => {
    expect(formatStat(0)).toBe('0');
    expect(formatStat(-42)).toBe('-42');
  });

  it('trims trailing zeros off a decimal', () => {
    expect(formatStat(0.5)).toBe('0.5');
    expect(formatStat(1.25)).toBe('1.25');
  });

  it('falls back to exponent notation at the extremes', () => {
    // A vanishing gradient must not round to a flat zero.
    expect(formatStat(3.7e-9)).toBe('3.700e-9');
    expect(formatStat(1.5e12)).toBe('1.500e+12');
  });
});
