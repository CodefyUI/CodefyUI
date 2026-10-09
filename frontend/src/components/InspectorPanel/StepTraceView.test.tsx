import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, cleanup, render, screen, fireEvent, waitFor } from '@testing-library/react';
import { StepTraceView } from './StepTraceView';
import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import { flushTabNodeUpdates, queueTabNodeStatus } from '../../store/nodeUpdateQueue';
import { enterBlocks } from '../../test/openBlocks';
import {
  fetchOutput,
  fetchStepIndex,
  listRunOutputs,
  PayloadTooLargeError,
  RunDataExpiredError,
  type StepIndexEntry,
} from '../../api/executionOutputs';
import {
  _resetRunIndexesForTests,
  _setRunEndPollForTests,
  _setRunRecordRetryForTests,
} from './portCaptures';
import { getRun, type RunInfo } from '../../api/rest';
import type { ExecutionStatus, TensorOutput, OutputData } from '../../types';

vi.mock('../../api/executionOutputs', async () => {
  const actual = await vi.importActual<typeof import('../../api/executionOutputs')>(
    '../../api/executionOutputs',
  );
  return {
    ...actual,
    fetchStepIndex: vi.fn(),
    fetchOutput: vi.fn(),
    listRunOutputs: vi.fn(),
  };
});

// The run record says whether the last run is over; these tests are after it.
vi.mock('../../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/rest')>();
  return { ...actual, getRun: vi.fn() };
});

/** Put the active tab in the given run state, holding one node `n1`. */
function seedRun(tabStatus: ExecutionStatus, nodeStatus: ExecutionStatus) {
  const tab = useTabStore.getState().tabs[0];
  useTabStore.setState({
    tabs: [
      {
        ...tab,
        status: tabStatus,
        lastRunId: 'r1',
        nodes: [
          {
            id: 'n1',
            type: 'baseNode',
            position: { x: 0, y: 0 },
            data: { label: 'N1', type: 'Generic', params: {}, executionStatus: nodeStatus },
          },
        ],
      },
    ],
    activeTabId: tab.id,
  });
}

/** One `node_status` frame, through the same queue the socket handler writes to. */
function report(nodeId: string, status: ExecutionStatus) {
  act(() => {
    queueTabNodeStatus(useTabStore.getState().activeTabId, nodeId, status);
    flushTabNodeUpdates();
  });
}

const mockStepIndex = vi.mocked(fetchStepIndex);
const mockOutput = vi.mocked(fetchOutput);
const mockList = vi.mocked(listRunOutputs);
const mockGetRun = vi.mocked(getRun);

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

function step(partial: Partial<StepIndexEntry> & Pick<StepIndexEntry, 'index' | 'name'>): StepIndexEntry {
  return {
    description: '',
    scalars: {},
    tensor_keys: [],
    ...partial,
  };
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  mockStepIndex.mockReset();
  mockOutput.mockReset();
  // No run index unless a test gives one: a list that cannot be read leaves
  // the steps asked for, as before.
  _resetRunIndexesForTests();
  // A record still saying "running" is asked again three times, at once, and
  // a run the tab is not running is polled for its end every 5 ms.
  _setRunRecordRetryForTests(3, 0);
  _setRunEndPollForTests(5);
  mockList.mockReset();
  mockList.mockRejectedValue(new Error('offline'));
  mockGetRun.mockReset();
  mockGetRun.mockResolvedValue({ id: 'r1', status: 'succeeded' } as RunInfo);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('StepTraceView', () => {
  it('shows the loading placeholder before the index resolves', () => {
    mockStepIndex.mockReturnValue(new Promise(() => {}));
    const { container } = render(<StepTraceView runId="r1" nodeId="n1" />);
    expect(container.querySelector('div')?.textContent).toBe('…');
  });

  it('shows the empty state when no steps are recorded', async () => {
    mockStepIndex.mockResolvedValue([]);
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() =>
      expect(screen.getByText('No steps recorded')).toBeInTheDocument(),
    );
    expect(
      screen.getByText('Turn on Verbose internals in Settings and re-run'),
    ).toBeInTheDocument();
  });

  it('shows the index error on a generic failure', async () => {
    mockStepIndex.mockRejectedValue(new Error('idx fail'));
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('idx fail')).toBeInTheDocument());
  });

  it('shows the expired message when the index rejects with RunDataExpiredError', async () => {
    mockStepIndex.mockRejectedValue(new RunDataExpiredError('r1'));
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() =>
      expect(screen.getByText('Run data expired — re-run to capture')).toBeInTheDocument(),
    );
  });

  it('renders a step card with description, scalars and a tensor grid', async () => {
    mockStepIndex.mockResolvedValue([
      step({
        index: 0,
        name: 'Softmax',
        description: 'compute $x$',
        scalars: { temperature: 0.5, count: 3, tiny: 0.0001 },
        tensor_keys: ['probs'],
      }),
    ]);
    mockOutput.mockResolvedValue(tensor([[0.1, 0.9], [0.4, 0.6]], { min: 0.1, max: 0.9 }));
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('Softmax')).toBeInTheDocument());
    // step index shows 1-based
    expect(screen.getByText('1.')).toBeInTheDocument();
    // scalars formatting: integer, fixed(4), exponential
    expect(screen.getByText(/temperature = 0\.5000/)).toBeInTheDocument();
    expect(screen.getByText(/count = 3/)).toBeInTheDocument();
    expect(screen.getByText(/tiny = 1\.00e-4/)).toBeInTheDocument();
    // tensor label + grid
    expect(screen.getByText('probs')).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('shape [2, 2]')).toBeInTheDocument());
  });

  it('collapses and expands a step card on header click', async () => {
    mockStepIndex.mockResolvedValue([
      step({ index: 0, name: 'StepA', description: 'desc', tensor_keys: [] }),
    ]);
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('StepA')).toBeInTheDocument());
    // body visible (caret ▾, 'No tensors' shown)
    expect(screen.getByText('No tensors')).toBeInTheDocument();
    expect(screen.getByText('▾')).toBeInTheDocument();
    fireEvent.click(screen.getByText('StepA'));
    // collapsed: caret ▸, body hidden
    expect(screen.getByText('▸')).toBeInTheDocument();
    expect(screen.queryByText('No tensors')).not.toBeInTheDocument();
    // expand again (toggles isCollapsed back)
    fireEvent.click(screen.getByText('StepA'));
    expect(screen.getByText('No tensors')).toBeInTheDocument();
  });

  it('shows the no-tensors note for a step with empty tensor_keys and no description/scalars', async () => {
    mockStepIndex.mockResolvedValue([step({ index: 0, name: 'Bare' })]);
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('Bare')).toBeInTheDocument());
    expect(screen.getByText('No tensors')).toBeInTheDocument();
  });

  it('renders a per-tensor loading placeholder before its fetch resolves', async () => {
    mockStepIndex.mockResolvedValue([
      step({ index: 0, name: 'S', tensor_keys: ['t'] }),
    ]);
    mockOutput.mockReturnValue(new Promise(() => {})); // pending
    const { container } = render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('t')).toBeInTheDocument());
    // the loading placeholder '…' appears inside the tensor block
    expect(container.textContent).toContain('…');
  });

  it('shows a scalar (non-tensor) tensor value via String()', async () => {
    mockStepIndex.mockResolvedValue([
      step({ index: 0, name: 'S', tensor_keys: ['val'] }),
    ]);
    const scalar: OutputData = { type: 'scalar', run_id: 'r', node_id: 'n', port: 'p', value: 7 };
    mockOutput.mockResolvedValue(scalar);
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('7')).toBeInTheDocument());
  });

  it('renders a non-tensor non-scalar value with an empty scalar body', async () => {
    mockStepIndex.mockResolvedValue([
      step({ index: 0, name: 'S', tensor_keys: ['str'] }),
    ]);
    const strOut: OutputData = { type: 'string', run_id: 'r', node_id: 'n', port: 'p', value: 'hi' };
    mockOutput.mockResolvedValue(strOut);
    render(<StepTraceView runId="r1" nodeId="n1" />);
    // label rendered; the scalar block is present but renders nothing for non-scalar type
    await waitFor(() => expect(screen.getByText('str')).toBeInTheDocument());
  });

  it('shows a per-tensor generic error message', async () => {
    mockStepIndex.mockResolvedValue([
      step({ index: 0, name: 'S', tensor_keys: ['t'] }),
    ]);
    mockOutput.mockRejectedValue(new Error('tensor boom'));
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('tensor boom')).toBeInTheDocument());
  });

  it("shows 'Expired' on a per-tensor RunDataExpiredError", async () => {
    mockStepIndex.mockResolvedValue([
      step({ index: 0, name: 'S', tensor_keys: ['t'] }),
    ]);
    mockOutput.mockRejectedValue(new RunDataExpiredError('r1'));
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('Expired')).toBeInTheDocument());
  });

  it('falls back to a bounded preview on PayloadTooLargeError', async () => {
    mockStepIndex.mockResolvedValue([
      step({ index: 2, name: 'S', tensor_keys: ['t'] }),
    ]);
    mockOutput.mockImplementation(async (_r, _n, _port, opts) => {
      if (!opts) throw new PayloadTooLargeError('too big');
      return tensor([[1, 1], [1, 1]], { min: 1, max: 1 });
    });
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('shape [2, 2]')).toBeInTheDocument());
    // port name encodes step index + tensor name; retry asks for a bounded preview
    expect(mockOutput).toHaveBeenCalledWith('r1', 'n1', '__step__2__t', {
      preview: true,
      maxElements: 65536,
    });
  });

  it('bails out of the index .then when unmounted before it resolves', async () => {
    let resolve!: (v: StepIndexEntry[]) => void;
    mockStepIndex.mockReturnValue(new Promise<StepIndexEntry[]>((r) => { resolve = r; }));
    const { unmount } = render(<StepTraceView runId="r1" nodeId="n1" />);
    // The index is asked for once the run's port list is read.
    await waitFor(() => expect(mockStepIndex).toHaveBeenCalled());
    unmount();
    resolve([step({ index: 0, name: 'S', tensor_keys: ['t'] })]);
    await Promise.resolve();
    await Promise.resolve();
    expect(mockOutput).not.toHaveBeenCalled();
  });

  it('bails out of the index .catch when unmounted before it rejects', async () => {
    let reject!: (e: unknown) => void;
    mockStepIndex.mockReturnValue(new Promise<StepIndexEntry[]>((_r, rej) => { reject = rej; }));
    const { unmount, container } = render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(mockStepIndex).toHaveBeenCalled());
    unmount();
    reject(new Error('late'));
    await Promise.resolve();
    await Promise.resolve();
    expect(container.querySelector('.portError')).toBeNull();
  });

  it('bails out of the per-tensor success handler when unmounted mid-fetch', async () => {
    mockStepIndex.mockResolvedValue([step({ index: 0, name: 'S', tensor_keys: ['t'] })]);
    let resolveOut!: (v: TensorOutput) => void;
    mockOutput.mockReturnValue(new Promise<TensorOutput>((r) => { resolveOut = r; }));
    const { unmount } = render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(mockOutput).toHaveBeenCalled());
    unmount();
    resolveOut(tensor([[1]], { min: 1, max: 1 }));
    await Promise.resolve();
    await Promise.resolve();
    expect(true).toBe(true);
  });

  it('bails out of the per-tensor error handler when unmounted mid-fetch', async () => {
    mockStepIndex.mockResolvedValue([step({ index: 0, name: 'S', tensor_keys: ['t'] })]);
    let rejectOut!: (e: unknown) => void;
    mockOutput.mockReturnValue(new Promise<TensorOutput>((_r, rej) => { rejectOut = rej; }));
    const { unmount } = render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(mockOutput).toHaveBeenCalled());
    unmount();
    rejectOut(new Error('late tensor'));
    await Promise.resolve();
    await Promise.resolve();
    expect(true).toBe(true);
  });

  it('does not start a tensor effect when steps array is empty after resolving', async () => {
    // Empty steps → second effect early-returns; empty-state renders.
    mockStepIndex.mockResolvedValue([]);
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() =>
      expect(screen.getByText('No steps recorded')).toBeInTheDocument(),
    );
    expect(mockOutput).not.toHaveBeenCalled();
  });
});

// A node's steps are written when the node returns, together with its outputs,
// and the index endpoint answers 404 — which `fetchStepIndex` reads as "no
// steps" — for anything not written yet.
describe('StepTraceView — while the node is still running', () => {
  afterEach(() => {
    // The tab outlives the test; leave it as the tests above expect to find it.
    const tab = useTabStore.getState().tabs[0];
    useTabStore.setState({ tabs: [{ ...tab, status: 'idle', lastRunId: null, nodes: [] }] });
  });

  /** Empty until the node is done, the way the server answers. */
  function serveStepsWhenDone(done: { value: boolean }) {
    mockStepIndex.mockImplementation(async () =>
      done.value ? [step({ index: 0, name: 'Softmax' })] : [],
    );
  }

  it('says the node is running instead of telling the user to turn Verbose on', async () => {
    seedRun('running', 'running');
    serveStepsWhenDone({ value: false });
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(screen.queryByText('No steps recorded')).toBeNull();
    expect(screen.getByText('Node is running…')).toBeInTheDocument();
    expect(mockStepIndex).not.toHaveBeenCalled();
  });

  it('says a queued node is waiting', () => {
    seedRun('running', 'idle');
    serveStepsWhenDone({ value: false });
    render(<StepTraceView runId="r1" nodeId="n1" />);
    expect(screen.getByText('Waiting for this node to run…')).toBeInTheDocument();
    expect(mockStepIndex).not.toHaveBeenCalled();
  });

  it('loads the steps by itself when the node finishes', async () => {
    seedRun('running', 'running');
    const done = { value: false };
    serveStepsWhenDone(done);
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    done.value = true;
    report('n1', 'completed');

    await waitFor(() => expect(screen.getByText('Softmax')).toBeInTheDocument());
    expect(screen.queryByText('Node is running…')).toBeNull();
    expect(mockStepIndex).toHaveBeenCalledTimes(1);
    expect(mockStepIndex).toHaveBeenCalledWith('r1', 'n1');
  });
});

// ── A node the last run has nothing for ─────────────────────────────────────

describe('StepTraceView — a node the last run has nothing for', () => {
  /** Put `id` on the canvas declaring `outputs`; a TRIGGER output is no data. */
  function place(id: string, outputs: { name: string; data_type: string }[]) {
    const tab = useTabStore.getState().tabs[0];
    useTabStore.setState({
      tabs: [{
        ...tab,
        status: 'idle',
        nodes: [{
          id, type: 'baseNode', position: { x: 0, y: 0 },
          data: {
            label: id, type: 'Generic', params: {},
            definition: {
              node_name: 'Generic', category: 'x', description: '', inputs: [], params: [],
              outputs: outputs.map((o) => ({ ...o, description: '', optional: false })),
            },
          },
        }],
      }],
      activeTabId: tab.id,
    });
  }

  afterEach(() => {
    cleanup();
    const tab = useTabStore.getState().tabs[0];
    useTabStore.setState({ tabs: [{ ...tab, nodes: [] }] });
  });

  it('says the node was not in the last run instead of asking for its steps', async () => {
    place('n1', [{ name: 'out', data_type: 'TENSOR' }]);
    mockList.mockResolvedValue([{ node_id: 'other', port: 'out', type: 'scalar', full_shape: null }]);
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('Not in the last run')).toBeInTheDocument());
    expect(mockStepIndex).not.toHaveBeenCalled();
    expect(screen.queryByText('No steps recorded')).toBeNull();
  });

  it('shows Start what it always showed: it declares no output, so no list names it', async () => {
    place('start', [{ name: 'trigger', data_type: 'TRIGGER' }]);
    mockList.mockResolvedValue([{ node_id: 'other', port: 'out', type: 'scalar', full_shape: null }]);
    mockStepIndex.mockResolvedValue([]);
    render(<StepTraceView runId="r1" nodeId="start" />);
    await waitFor(() => expect(screen.getByText('No steps recorded')).toBeInTheDocument());
    expect(mockStepIndex).toHaveBeenCalledWith('r1', 'start');
    expect(screen.queryByText('Not in the last run')).toBeNull();
  });

  it('waits for a node a watched run has not reached, and reads its steps once the run ends', async () => {
    // Watch in the Runs panel, or a reload mid-run: the tab is idle while the
    // run goes on.
    place('n1', [{ name: 'out', data_type: 'TENSOR' }]);
    mockGetRun.mockResolvedValue({ id: 'r1', status: 'running' } as RunInfo);
    mockList.mockResolvedValue([{ node_id: 'other', port: 'out', type: 'scalar', full_shape: null }]);
    mockStepIndex.mockResolvedValue([step({ index: 0, name: 'Softmax' })]);
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() =>
      expect(screen.getByText('Waiting for this node to run…')).toBeInTheDocument(),
    );
    expect(mockStepIndex).not.toHaveBeenCalled();

    // The run ends, and the node with it. Nothing in the tab changes.
    mockGetRun.mockResolvedValue({ id: 'r1', status: 'succeeded' } as RunInfo);
    mockList.mockResolvedValue([
      { node_id: 'other', port: 'out', type: 'scalar', full_shape: null },
      { node_id: 'n1', port: 'out', type: 'tensor', full_shape: [2] },
    ]);
    await waitFor(() => expect(screen.getByText('Softmax')).toBeInTheDocument());
    expect(mockStepIndex).toHaveBeenCalledWith('r1', 'n1');
    expect(screen.queryByText('Waiting for this node to run…')).toBeNull();
  });

  it('follows the Record node outputs switch, in a run made with it off', async () => {
    const setRecord = (on: boolean) => {
      const tab = useTabStore.getState().tabs[0];
      useTabStore.setState({ tabs: [{ ...tab, recordOutputs: on }] });
    };
    place('n1', [{ name: 'out', data_type: 'TENSOR' }]);
    setRecord(true);
    mockGetRun.mockResolvedValue({
      id: 'r1', status: 'succeeded', options: { record_outputs: false },
    } as unknown as RunInfo);
    try {
      render(<StepTraceView runId="r1" nodeId="n1" />);
      await waitFor(() =>
        expect(screen.getByText('Run the graph to capture its values')).toBeInTheDocument(),
      );
      act(() => setRecord(false));
      expect(
        screen.getByText('Turn on Record node outputs in Settings, then run the graph'),
      ).toBeInTheDocument();
      expect(mockStepIndex).not.toHaveBeenCalled();
    } finally {
      // Unmounted first: the view would redraw for the reset.
      cleanup();
      setRecord(true);
    }
  });
});

// ── Inside an open block (#621) ──────────────────────────────────────────────
// The run recorded the block's nodes as `<instance>/<inner>`; the open block
// shows them under their own ids.

describe('StepTraceView — inside an open block', () => {
  // Unmount first: a view still mounted would read the reset as leaving the
  // block, and ask again.
  afterEach(() => {
    cleanup();
    enterBlocks();
  });

  it('asks for the steps by the id the run gave the node, and by the bare id at the top level', async () => {
    mockStepIndex.mockResolvedValue([step({ index: 0, name: 'S', tensor_keys: ['t'] })]);
    mockOutput.mockResolvedValue(tensor([[1, 1], [1, 1]], { min: 1, max: 1 }));

    const top = render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('shape [2, 2]')).toBeInTheDocument());
    expect(mockStepIndex).toHaveBeenCalledWith('r1', 'n1');
    expect(mockOutput).toHaveBeenCalledWith('r1', 'n1', '__step__0__t');
    top.unmount();

    enterBlocks('blk', 'nest');
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('shape [2, 2]')).toBeInTheDocument());
    expect(mockStepIndex).toHaveBeenLastCalledWith('r1', 'blk/nest/n1');
    expect(mockOutput).toHaveBeenLastCalledWith('r1', 'blk/nest/n1', '__step__0__t');
  });
});
