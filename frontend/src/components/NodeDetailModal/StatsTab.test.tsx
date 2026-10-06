import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { act, cleanup, render, screen, waitFor, within } from '@testing-library/react';
import type { Edge, Node } from '@xyflow/react';
import type { ExecutionStatus, NodeData, NodeDefinition } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { flushTabNodeUpdates, queueTabNodeStatus } from '../../store/nodeUpdateQueue';
import { enterBlocks as enter } from '../../test/openBlocks';

vi.mock('../../api/executionOutputs', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('../../api/executionOutputs')>();
  return { ...actual, fetchPortStats: vi.fn(), listRunOutputs: vi.fn() };
});

// The run record says whether the last run is over; these tests are after it.
vi.mock('../../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/rest')>();
  return { ...actual, getRun: vi.fn() };
});

import {
  fetchPortStats,
  listRunOutputs,
  NoValueError,
  StatsNotCapturedError,
  type PortStats,
} from '../../api/executionOutputs';
import { useI18n } from '../../i18n';
import { keyOf } from '../InspectorPanel/PortGroup';
import {
  _resetRunIndexesForTests,
  _setRunEndPollForTests,
  _setRunRecordRetryForTests,
} from '../InspectorPanel/portCaptures';
import { getRun, type RunInfo } from '../../api/rest';
import { StatsTab, formatStat } from './StatsTab';
import type { NodeDetailTabContext } from './tabs';

const mockStats = vi.mocked(fetchPortStats);
const mockList = vi.mocked(listRunOutputs);
const mockGetRun = vi.mocked(getRun);

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
  // The locale is module-global, so a test that switches it would otherwise
  // hand the next one a Chinese panel to assert English against.
  useI18n.setState({ locale: 'en' });
  // No run index unless a test gives one: a list that cannot be read leaves
  // every port asked for, as before.
  _resetRunIndexesForTests();
  // A record still saying "running" is asked again three times, at once, and
  // a run the tab is not running is polled for its end every 5 ms.
  _setRunRecordRetryForTests(3, 0);
  _setRunEndPollForTests(5);
  mockList.mockReset();
  mockList.mockRejectedValue(new Error('offline'));
  mockGetRun.mockReset();
  mockGetRun.mockResolvedValue({ id: 'run1', status: 'succeeded' } as RunInfo);
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
    expect(screen.getByText('No statistics yet')).toBeInTheDocument();
    expect(mockStats).not.toHaveBeenCalled();
  });

  it('asks for Record node outputs to be turned on, before a run, only when it is off', () => {
    const on = render(<StatsTab ctx={ctx({ runId: null, recordOutputs: true })} />);
    expect(screen.getByText('Run the graph to capture its values')).toBeInTheDocument();
    expect(screen.queryByText(/Turn on Record node outputs/)).toBeNull();
    on.unmount();

    render(<StatsTab ctx={ctx({ runId: null, recordOutputs: false })} />);
    expect(
      screen.getByText('Turn on Record node outputs in Settings, then run the graph'),
    ).toBeInTheDocument();
  });

  it('warns when recording is off', async () => {
    render(<StatsTab ctx={ctx({ recordOutputs: false })} />);
    expect(
      screen.getByText(
        'Record node outputs is off — turn it on in Settings and re-run to capture values',
      ),
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
        screen.getAllByText(
          'Nothing captured for this port — turn on Record node outputs in Settings and re-run',
        ),
      ).toHaveLength(2),
    );
    expect(screen.queryByText(/Rec toggle in the toolbar/)).toBeNull();
  });

  it('shows the not-captured hint in the locale chosen after the fetch failed', async () => {
    // The state holds the translation KEY. Holding the sentence would freeze
    // this line in whichever language was current when the request failed.
    mockStats.mockRejectedValue(new StatsNotCapturedError('nothing captured'));
    render(<StatsTab ctx={ctx()} />);
    await waitFor(() =>
      expect(screen.getAllByText(
        'Nothing captured for this port — turn on Record node outputs in Settings and re-run',
      )).toHaveLength(2),
    );
    const callsBefore = mockStats.mock.calls.length;

    act(() => useI18n.setState({ locale: 'zh-TW' }));

    expect(screen.getAllByText(
      '這個連接埠沒有擷取到資料 — 請在設定中開啟「錄製節點輸出」後重新執行',
    )).toHaveLength(2);
    expect(screen.queryByText(/Nothing captured for this port/)).toBeNull();
    // A locale switch refetches nothing, so the line has to translate itself.
    expect(mockStats).toHaveBeenCalledTimes(callsBefore);
  });

  it('reports an unexpected failure without blanking the tab', async () => {
    mockStats.mockRejectedValue(new Error('boom'));
    render(<StatsTab ctx={ctx()} />);
    await waitFor(() => expect(screen.getAllByText('boom').length).toBe(2));
  });

  it('says a port produced no value instead of pointing at Record outputs', async () => {
    // The server's 204: the node ran and returned None for this port.
    // Recording was on, so the Record hint would send the user the wrong way.
    mockStats.mockImplementation(async (_r, nodeId, port) => {
      if (nodeId === 'n1') throw new NoValueError('run1', nodeId, port);
      return tensorStats();
    });
    render(<StatsTab ctx={ctx()} />);
    const block = screen.getByTestId('stats-port-n1-out');
    const note = await within(block).findByText('No value this run');
    // A quiet note, not an error line.
    expect(note.className).toMatch(/muted/);
    expect(screen.queryByText(/Nothing captured for this port/)).toBeNull();
    // The sibling port is unaffected.
    await waitFor(() =>
      expect(within(screen.getByTestId('stats-port-src-out')).getByText('[2, 3]')).toBeInTheDocument(),
    );
  });

  it('shows the no-value note in the locale chosen after the fetch', async () => {
    mockStats.mockRejectedValue(new NoValueError('run1', 'n1', 'out'));
    render(<StatsTab ctx={ctx()} />);
    await waitFor(() => expect(screen.getAllByText('No value this run')).toHaveLength(2));

    act(() => useI18n.setState({ locale: 'zh-TW' }));

    expect(screen.getAllByText('這次執行沒有值')).toHaveLength(2);
    expect(screen.queryByText('No value this run')).toBeNull();
  });

  it('aborts the port that left the view, and leaves the one that stayed alone', async () => {
    const signals = new Map<string, AbortSignal>();
    // Still computing — an abort is only observable on a request in flight.
    mockStats.mockImplementation((_r, nodeId, port, opts) => {
      if (opts?.signal) signals.set(keyOf(nodeId, port), opts.signal);
      return new Promise<PortStats>(() => {});
    });
    const { rerender } = render(<StatsTab ctx={ctx()} />);
    await waitFor(() => expect(signals.size).toBe(2));

    // Dropping the edge drops the input port, so the set changes.
    rerender(<StatsTab ctx={ctx({ edges: [] })} />);

    expect(signals.get(keyOf('src', 'out'))!.aborted).toBe(true);
    // The port that stayed is neither cancelled nor computed a second time:
    // its statistics did not change because a sibling row went away.
    expect(signals.get(keyOf('n1', 'out'))!.aborted).toBe(false);
    expect(mockStats).toHaveBeenCalledTimes(2);
  });

  it('aborts in-flight requests when the tab unmounts', async () => {
    let captured: AbortSignal | undefined;
    mockStats.mockImplementation((_r, _n, _p, opts) => {
      captured = opts?.signal;
      return new Promise<PortStats>(() => {});
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

// ── Statistics of a node that has not returned yet ───────────────────────────
// `/stats` summarises the same captures the Inputs and Outputs tabs read, so
// it answers 404 until the node owning the port has returned. Read too early
// the row says nothing was captured and points at a setting — about a node
// that has simply not run yet.

describe('StatsTab — while the graph is running', () => {
  /** The ctx node as the engine last reported it, on the tab the hook reads. */
  function atStatus(id: string, executionStatus: ExecutionStatus): Node<NodeData> {
    const n = node(id);
    return { ...n, data: { ...n.data, executionStatus } };
  }

  /** Put the active tab in a run, holding the same nodes the ctx names. */
  function seedRun(statuses: Record<string, ExecutionStatus>) {
    const tab = useTabStore.getState().tabs[0];
    useTabStore.setState({
      tabs: [
        {
          ...tab,
          status: 'running',
          lastRunId: 'run1',
          nodes: Object.entries(statuses).map(([id, s]) => atStatus(id, s)),
        },
      ],
      activeTabId: tab.id,
    });
  }

  /** One `node_status` frame, through the queue the socket handler writes to. */
  function report(nodeId: string, status: ExecutionStatus) {
    act(() => {
      queueTabNodeStatus(useTabStore.getState().activeTabId, nodeId, status);
      flushTabNodeUpdates();
    });
  }

  afterEach(() => {
    const tab = useTabStore.getState().tabs[0];
    useTabStore.setState({ tabs: [{ ...tab, status: 'idle', lastRunId: null, nodes: [] }] });
  });

  it('says the node is running instead of reporting its port uncaptured', async () => {
    seedRun({ src: 'completed', n1: 'running' });
    render(<StatsTab ctx={ctx()} />);
    await waitFor(() =>
      expect(mockStats).toHaveBeenCalledWith('run1', 'src', 'out', expect.anything()),
    );

    const block = screen.getByTestId('stats-port-n1-out');
    expect(within(block).getByText('Node is running…')).toBeInTheDocument();
    expect(screen.queryByText(/Nothing captured for this port/)).toBeNull();
    // Asking for this node's statistics now could only be answered 404.
    expect(mockStats).toHaveBeenCalledTimes(1);
  });

  it('computes the statistics by itself when the node finishes', async () => {
    seedRun({ src: 'completed', n1: 'running' });
    render(<StatsTab ctx={ctx()} />);
    await waitFor(() => expect(mockStats).toHaveBeenCalledTimes(1));

    report('n1', 'completed');

    // No reselect: the row was on screen the whole time.
    await waitFor(() =>
      expect(mockStats).toHaveBeenCalledWith('run1', 'n1', 'out', expect.anything()),
    );
    const block = screen.getByTestId('stats-port-n1-out');
    await waitFor(() => expect(within(block).getByText('[2, 3]')).toBeInTheDocument());
    expect(within(block).queryByText('Node is running…')).toBeNull();
    // The upstream port was summarised once, and not again.
    expect(mockStats).toHaveBeenCalledTimes(2);
  });

  it('says a queued node is waiting, not running', () => {
    seedRun({ src: 'idle', n1: 'idle' });
    render(<StatsTab ctx={ctx()} />);
    expect(screen.getAllByText('Waiting for this node to run…')).toHaveLength(2);
    expect(screen.queryByText('Node is running…')).toBeNull();
    expect(mockStats).not.toHaveBeenCalled();
  });
});

// ── One port in two rows ─────────────────────────────────────────────────────
// One output wired into two inputs of the node (`x` into both inputs of Add)
// gives two input rows for the same port. Rows that shared a React key stayed
// on screen after the list changed, as in the Inspector (#562).

describe('StatsTab — the same port in two rows', () => {
  /** `src.out` wired into both inputs of `n1`. */
  const twice: Edge[] = [
    { id: 'e1', source: 'src', target: 'n1', sourceHandle: 'out', targetHandle: 'x' },
    { id: 'e2', source: 'src', target: 'n1', sourceHandle: 'out', targetHandle: 'y' },
  ];

  it('shows both rows, asks for the port once, and repeats no React key', async () => {
    const error = vi.spyOn(console, 'error');
    try {
      render(<StatsTab ctx={ctx({ edges: twice })} />);
      await waitFor(() => expect(screen.getAllByText('[2, 3]')).toHaveLength(3));
      expect(screen.getAllByTestId('stats-port-src-out')).toHaveLength(2);
      // `src.out` once, for both rows, and `n1.out`.
      expect(mockStats).toHaveBeenCalledTimes(2);
      expect(error.mock.calls.flat().join('\n')).not.toMatch(/same key/);
    } finally {
      error.mockRestore();
    }
  });

  it('leaves only the new rows when the inputs change', async () => {
    const { rerender } = render(<StatsTab ctx={ctx({ edges: twice })} />);
    await waitFor(() => expect(screen.getAllByText('[2, 3]')).toHaveLength(3));

    const other: Edge = {
      id: 'e3', source: 'n0', target: 'n1', sourceHandle: 'out', targetHandle: 'x',
    };
    rerender(<StatsTab ctx={ctx({ edges: [other] })} />);
    await waitFor(() => expect(screen.getAllByText('[2, 3]')).toHaveLength(2));
    expect(screen.queryAllByTestId('stats-port-src-out')).toHaveLength(0);
    expect(screen.getAllByTestId('stats-port-n0-out')).toHaveLength(1);
  });
});

// ── An input fed through the open block's own inputs ─────────────────────────

describe('StatsTab — inputs fed through the open block', () => {
  afterEach(() => {
    cleanup();
    enter();
    const { tabs, activeTabId } = useTabStore.getState();
    useTabStore.setState({
      tabs: tabs.map((t) => (t.id === activeTabId ? { ...t, subgraphs: [] } : t)),
    });
  });

  it('says the input comes in through the block instead of that nothing is connected', async () => {
    enter('blk');
    const { tabs, activeTabId } = useTabStore.getState();
    useTabStore.setState({
      tabs: tabs.map((t) =>
        t.id === activeTabId
          ? {
              ...t,
              subgraphs: [{
                id: 'outer', name: 'Outer', description: '', nodes: [], edges: [],
                interface: {
                  inputs: [{ port: 'in', innerNode: 'n1', innerPort: 'x', data_type: 'TENSOR' }],
                  outputs: [],
                  triggerTargets: [],
                },
              }],
            }
          : t,
      ),
    });
    render(<StatsTab ctx={ctx({ edges: [] })} />);
    expect(screen.getByText("From the block's input: in")).toBeInTheDocument();
    expect(screen.queryByText('No inputs connected')).toBeNull();
    await waitFor(() => expect(screen.getAllByText('[2, 3]')).toHaveLength(1));
  });
});

// ── A node the last run has nothing for ─────────────────────────────────────
// Added after the run: `/stats` could only answer 404, which reads as "turn
// on Record outputs" -- a setting that is on.

describe('StatsTab — a node the last run has nothing for', () => {
  it('says the node was not in the last run, and asks only for the port the run had', async () => {
    mockList.mockResolvedValue([{ node_id: 'src', port: 'out', type: 'tensor', full_shape: [2, 3] }]);
    render(<StatsTab ctx={ctx()} />);
    const block = screen.getByTestId('stats-port-n1-out');
    await waitFor(() => expect(within(block).getByText('Not in the last run')).toBeInTheDocument());
    await waitFor(() => expect(screen.getAllByText('[2, 3]')).toHaveLength(1));
    expect(mockStats).toHaveBeenCalledTimes(1);
    expect(mockStats).toHaveBeenCalledWith('run1', 'src', 'out', expect.anything());
    expect(screen.queryByText(/Nothing captured for this port/)).toBeNull();
  });

  it('waits for a node a watched run has not reached, and computes its statistics once the run ends', async () => {
    // Watch in the Runs panel, or a reload mid-run: the tab is idle while the
    // run goes on, and `src` has finished in it.
    mockGetRun.mockResolvedValue({ id: 'run1', status: 'running' } as RunInfo);
    mockList.mockResolvedValue([{ node_id: 'src', port: 'out', type: 'tensor', full_shape: [2, 3] }]);
    render(<StatsTab ctx={ctx()} />);
    await waitFor(() =>
      expect(
        within(screen.getByTestId('stats-port-n1-out')).getByText('Waiting for this node to run…'),
      ).toBeInTheDocument(),
    );
    await waitFor(() => expect(screen.getAllByText('[2, 3]')).toHaveLength(1));
    expect(mockStats).toHaveBeenCalledTimes(1);

    // The run ends, and `n1` with it. Nothing in the tab changes.
    mockGetRun.mockResolvedValue({ id: 'run1', status: 'succeeded' } as RunInfo);
    mockList.mockResolvedValue([
      { node_id: 'src', port: 'out', type: 'tensor', full_shape: [2, 3] },
      { node_id: 'n1', port: 'out', type: 'tensor', full_shape: [2, 3] },
    ]);
    await waitFor(() =>
      expect(within(screen.getByTestId('stats-port-n1-out')).getByText('[2, 3]')).toBeInTheDocument(),
    );
    expect(mockStats).toHaveBeenCalledWith('run1', 'n1', 'out', expect.anything());
    // The upstream port was summarised once, and not again.
    expect(mockStats).toHaveBeenCalledTimes(2);
  });

  it('follows the Record node outputs switch, in a run made with it off', async () => {
    const setRecord = (on: boolean) => {
      const { tabs, activeTabId } = useTabStore.getState();
      useTabStore.setState({
        tabs: tabs.map((t) => (t.id === activeTabId ? { ...t, recordOutputs: on } : t)),
      });
    };
    setRecord(true);
    mockGetRun.mockResolvedValue({
      id: 'run1', status: 'succeeded', options: { record_outputs: false },
    } as unknown as RunInfo);
    const view = render(<StatsTab ctx={ctx()} />);
    try {
      await waitFor(() =>
        expect(screen.getAllByText('Run the graph to capture its values')).toHaveLength(2),
      );
      act(() => setRecord(false));
      expect(
        screen.getAllByText('Turn on Record node outputs in Settings, then run the graph'),
      ).toHaveLength(2);
      expect(mockStats).not.toHaveBeenCalled();
    } finally {
      // Unmounted first: the tab would redraw for the reset.
      view.unmount();
      setRecord(true);
    }
  });

  it('stops asking about the run once the tab runs it itself', async () => {
    // Watch, before the server acknowledges the attach: the tab is still idle.
    mockGetRun.mockResolvedValue({ id: 'run1', status: 'running' } as RunInfo);
    mockList.mockResolvedValue([]);
    const view = render(<StatsTab ctx={ctx()} />);
    await waitFor(() =>
      expect(screen.getAllByText('Waiting for this node to run…')).toHaveLength(2),
    );

    // The attach is acknowledged: the tab's own frames follow the run now.
    const setStatus = (status: 'running' | 'idle') => {
      const { tabs, activeTabId } = useTabStore.getState();
      useTabStore.setState({
        tabs: tabs.map((t) => (t.id === activeTabId ? { ...t, status } : t)),
      });
    };
    try {
      act(() => setStatus('running'));
      const asked = mockGetRun.mock.calls.length;
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 40));
      });
      expect(mockGetRun.mock.calls.length).toBe(asked);
      expect(mockStats).not.toHaveBeenCalled();
    } finally {
      view.unmount();
      setStatus('idle');
    }
  });
});

// ── Inside an open block (#621) ──────────────────────────────────────────────
// The run captured the block's nodes as `<instance>/<inner>`; the open block
// shows them under their own ids, and two copies of one block share those.

describe('StatsTab — inside an open block', () => {
  // Unmount first: a tab still mounted would read the reset as leaving the
  // block, and ask again for every port.
  afterEach(() => {
    cleanup();
    enter();
  });

  it('asks for the bare ids at the top level, and for the ids the run gave them inside', async () => {
    const top = render(<StatsTab ctx={ctx()} />);
    await waitFor(() => expect(screen.getAllByText('[2, 3]')).toHaveLength(2));
    expect(mockStats).toHaveBeenCalledWith('run1', 'src', 'out', expect.anything());
    expect(mockStats).toHaveBeenCalledWith('run1', 'n1', 'out', expect.anything());
    top.unmount();

    mockStats.mockClear();
    enter('blk', 'nest');
    render(<StatsTab ctx={ctx()} />);
    // Shown under the canvas ids, as at the top level.
    await waitFor(() => expect(screen.getAllByText('[2, 3]')).toHaveLength(2));
    expect(mockStats).toHaveBeenCalledTimes(2);
    expect(mockStats).toHaveBeenCalledWith('run1', 'blk/nest/src', 'out', expect.anything());
    expect(mockStats).toHaveBeenCalledWith('run1', 'blk/nest/n1', 'out', expect.anything());
  });

  it('asks again in the other copy of the block, and stops what the first copy was computing', async () => {
    const signals = new Map<string, AbortSignal>();
    mockStats.mockImplementation((_r, nodeId, port, opts) => {
      if (opts?.signal) signals.set(keyOf(nodeId, port), opts.signal);
      return new Promise<PortStats>(() => {});
    });
    enter('blk');
    render(<StatsTab ctx={ctx()} />);
    await waitFor(() => expect(signals.size).toBe(2));

    act(() => enter('blk2'));
    await waitFor(() => expect(signals.size).toBe(4));
    expect(signals.get(keyOf('blk/n1', 'out'))!.aborted).toBe(true);
    expect(signals.get(keyOf('blk2/n1', 'out'))!.aborted).toBe(false);
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
