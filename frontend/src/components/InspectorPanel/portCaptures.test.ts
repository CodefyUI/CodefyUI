import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, cleanup, renderHook, waitFor } from '@testing-library/react';
import type { Edge, Node } from '@xyflow/react';
import type { ExecutionStatus, NodeData, NodeDefinition, OutputData } from '../../types';
import {
  fetchOutput,
  listRunOutputs,
  NoValueError,
  RunDataExpiredError,
} from '../../api/executionOutputs';
import { getRun, type RunInfo, type RunStatus } from '../../api/rest';
import { useTabStore, type TabState } from '../../store/tabStore';
import { enterBlocks as enter } from '../../test/openBlocks';
import { keyOf } from './PortGroup';
import {
  _resetRunIndexesForTests,
  _setRunEndPollForTests,
  _setRunRecordRetryForTests,
  capturePhase,
  capturePhaseNoteKey,
  portDataType,
  resolveInputSources,
  resolveSingleNodePorts,
  takeDuePorts,
  usePortFetches,
  useRunNodeId,
  useRunNodePrefix,
} from './portCaptures';

vi.mock('../../api/executionOutputs', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/executionOutputs')>();
  return { ...actual, fetchOutput: vi.fn(), listRunOutputs: vi.fn() };
});

vi.mock('../../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/rest')>();
  return { ...actual, getRun: vi.fn() };
});

const mockOutput = vi.mocked(fetchOutput);
const mockList = vi.mocked(listRunOutputs);
const mockGetRun = vi.mocked(getRun);

/** The run record as the server reports it: its status, and the options it ran with. */
function runRecord(status: RunStatus, options: Record<string, unknown> = {}): RunInfo {
  return { id: 'run1', status, options } as RunInfo;
}

beforeEach(() => {
  // No index unless a test gives one: a list that cannot be read leaves
  // every port asked for, as before.
  _resetRunIndexesForTests();
  // A record still saying "running" is asked again three times, at once, and
  // a run the tab is not running is polled for its end every 5 ms.
  _setRunRecordRetryForTests(3, 0);
  _setRunEndPollForTests(5);
  mockList.mockReset();
  mockList.mockRejectedValue(new Error('offline'));
  mockGetRun.mockReset();
  mockGetRun.mockResolvedValue(runRecord('succeeded'));
});

function def(outputs: { name: string; data_type: string }[]): NodeDefinition {
  return {
    node_name: 'N',
    category: 'c',
    description: '',
    inputs: [],
    outputs: outputs.map((o) => ({ ...o, description: '', optional: false })),
    params: [],
  };
}

function node(
  id: string,
  over: { label?: string; definition?: NodeDefinition } = {},
): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x: 0, y: 0 },
    data: {
      label: over.label ?? id,
      type: 'N',
      params: {},
      ...(over.definition !== undefined ? { definition: over.definition } : {}),
    },
  };
}

function edge(
  id: string,
  source: string,
  target: string,
  over: Partial<Edge> = {},
): Edge {
  return { id, source, target, sourceHandle: 'out', ...over } as Edge;
}

describe('portDataType', () => {
  it('reads the declared type off the source node definition', () => {
    const nodes = [node('a', { definition: def([{ name: 'out', data_type: 'TENSOR' }]) })];
    expect(portDataType(nodes, 'a', 'out')).toBe('TENSOR');
  });

  it('returns undefined for an unknown node, port, or a node with no definition', () => {
    const nodes = [
      node('a', { definition: def([{ name: 'out', data_type: 'TENSOR' }]) }),
      node('b'),
    ];
    expect(portDataType(nodes, 'ghost', 'out')).toBeUndefined();
    expect(portDataType(nodes, 'a', 'nope')).toBeUndefined();
    expect(portDataType(nodes, 'b', 'out')).toBeUndefined();
  });
});

describe('resolveInputSources', () => {
  it('collects the connected upstream (node, port) pairs', () => {
    const edges = [edge('e1', 's1', 'target', { sourceHandle: 'x' }), edge('e2', 's2', 'other')];
    expect(resolveInputSources('target', edges)).toEqual([{ nodeId: 's1', port: 'x' }]);
  });

  it('skips trigger edges in both shapes, and edges naming no source port', () => {
    const edges = [
      edge('e1', 's1', 'target', { type: 'triggerEdge' }),
      edge('e2', 's2', 'target', { data: { type: 'trigger' } }),
      edge('e3', 's3', 'target', { sourceHandle: null }),
    ];
    expect(resolveInputSources('target', edges)).toEqual([]);
  });
});

describe('resolveSingleNodePorts', () => {
  it('labels inputs with their provenance and lists the node’s own outputs', () => {
    const nodes = [
      node('n1', { definition: def([{ name: 'logits', data_type: 'TENSOR' }]) }),
      node('src', { label: 'Source', definition: def([{ name: 'y', data_type: 'SCALAR' }]) }),
    ];
    const edges = [edge('e1', 'src', 'n1', { sourceHandle: 'y' })];
    expect(resolveSingleNodePorts('n1', nodes, edges)).toEqual({
      inputs: [{ nodeId: 'src', port: 'y', displayName: 'Source.y', dataType: 'SCALAR' }],
      outputs: [{ nodeId: 'n1', port: 'logits', dataType: 'TENSOR' }],
    });
  });

  it('returns nothing for a node that is not on the canvas', () => {
    expect(resolveSingleNodePorts('ghost', [], [])).toEqual({ inputs: [], outputs: [] });
  });

  it('falls back to a truncated id when the upstream node has no usable label', () => {
    // Source is absent from `nodes` entirely, and its label would be blank
    // anyway — either way the row still names where the value came from.
    const nodes = [node('n1', { definition: def([]) })];
    const edges = [edge('e1', 'sourcenode123', 'n1', { sourceHandle: 'y' })];
    expect(resolveSingleNodePorts('n1', nodes, edges).inputs).toEqual([
      { nodeId: 'sourcenode123', port: 'y', displayName: 'source.y', dataType: undefined },
    ]);
  });

  it('treats a node with no definition as having no outputs', () => {
    const nodes = [node('n1')];
    expect(resolveSingleNodePorts('n1', nodes, [])).toEqual({ inputs: [], outputs: [] });
  });

  it('leaves out a trigger output, which has no value behind it', () => {
    // Start returns nothing for `trigger`, so a request for it could only 404.
    const nodes = [
      node('start', { definition: def([{ name: 'trigger', data_type: 'TRIGGER' }]) }),
    ];
    expect(resolveSingleNodePorts('start', nodes, [])).toEqual({ inputs: [], outputs: [] });
  });

  it('keeps the data outputs listed next to a trigger output', () => {
    const nodes = [
      node('n1', {
        definition: def([
          { name: 'trigger', data_type: 'TRIGGER' },
          { name: 'out', data_type: 'TENSOR' },
        ]),
      }),
    ];
    expect(resolveSingleNodePorts('n1', nodes, []).outputs).toEqual([
      { nodeId: 'n1', port: 'out', dataType: 'TENSOR' },
    ]);
  });
});

// ── A port the node left empty ──────────────────────────────────────────────
// The server answers 204 when the node ran and returned None for the port
// (TrainingLoop's grad_scaler_state outside fp16), and `fetchOutput` turns
// that into NoValueError. The row says so quietly; it is neither an error nor
// an expired run.

describe('usePortFetches', () => {
  const PORTS = [{ nodeId: 'train', port: 'grad_scaler_state' }];
  const KEY = keyOf('train', 'grad_scaler_state');

  beforeEach(() => {
    mockOutput.mockReset();
  });

  it('notes a port that produced no value instead of failing it', async () => {
    mockOutput.mockRejectedValue(new NoValueError('run1', 'train', 'grad_scaler_state'));
    const { result } = renderHook(() => usePortFetches('run1', PORTS));
    await waitFor(() =>
      expect(result.current[KEY]).toEqual({
        loading: false,
        error: null,
        errorKey: null,
        noteKey: 'inspector.noValue',
        data: null,
      }),
    );
    expect(mockOutput).toHaveBeenCalledTimes(1);
  });

  it('still reads a 404 as expired run data', async () => {
    mockOutput.mockRejectedValue(new RunDataExpiredError('run1'));
    const { result } = renderHook(() => usePortFetches('run1', PORTS));
    await waitFor(() =>
      expect(result.current[KEY]).toMatchObject({
        loading: false,
        error: null,
        errorKey: 'inspector.dataExpired',
        data: null,
      }),
    );
    expect(result.current[KEY].noteKey).toBeFalsy();
  });
});

// ── Inside an open block (#621) ──────────────────────────────────────────────
// The run captured an inner node as `<instance>/<inner>`; the open block shows
// it as `<inner>`. Two copies of one block share their inner canvas ids, so a
// result kept under the canvas id alone would show one copy's values in the
// other.

function scalar(nodeId: string, value: number): OutputData {
  return { type: 'scalar', run_id: 'run1', node_id: nodeId, port: 'tensor', value };
}

// Unmount first: a hook still mounted would read the reset as leaving the
// block, and ask again for every port.
function leaveAll() {
  cleanup();
  enter();
}

describe('the run id of a canvas node', () => {
  afterEach(leaveAll);

  it('is the canvas id at the top level, and the entered instances in front of it inside', () => {
    const { result } = renderHook(() => [useRunNodePrefix(), useRunNodeId('mul')]);
    expect(result.current).toEqual(['', 'mul']);

    act(() => enter('blk'));
    expect(result.current).toEqual(['blk/', 'blk/mul']);

    act(() => enter('blk', 'nest'));
    expect(result.current).toEqual(['blk/nest/', 'blk/nest/mul']);
  });
});

describe('usePortFetches inside an open block', () => {
  const MUL = [{ nodeId: 'mul', port: 'tensor' }];
  const KEY = keyOf('mul', 'tensor');

  beforeEach(() => {
    mockOutput.mockReset();
    mockOutput.mockImplementation(async (_run, nodeId) =>
      scalar(nodeId, nodeId === 'blk2/mul' ? 6 : 4),
    );
  });

  afterEach(leaveAll);

  it('asks for the id the run gave the node, and keys the answer by the canvas id', async () => {
    enter('blk');
    const { result } = renderHook(() => usePortFetches('run1', MUL));
    await waitFor(() => expect(result.current[KEY]?.data).toEqual(scalar('blk/mul', 4)));
    expect(mockOutput).toHaveBeenCalledWith('run1', 'blk/mul', 'tensor');
    expect(mockOutput).toHaveBeenCalledTimes(1);
  });

  it('asks again for the same canvas id in the other copy of the block, and never shows the first copy there', async () => {
    enter('blk');
    const { result } = renderHook(() => usePortFetches('run1', MUL));
    await waitFor(() => expect(result.current[KEY]?.data).toEqual(scalar('blk/mul', 4)));

    act(() => enter('blk2'));
    // Not one frame of the first copy's value under the second copy.
    expect(result.current[KEY]?.data ?? null).toBeNull();
    await waitFor(() => expect(result.current[KEY]?.data).toEqual(scalar('blk2/mul', 6)));
    expect(mockOutput).toHaveBeenLastCalledWith('run1', 'blk2/mul', 'tensor');
  });

  it('asks for a block inside the block by both instances', async () => {
    enter('blk', 'nest');
    renderHook(() => usePortFetches('run1', MUL));
    await waitFor(() => expect(mockOutput).toHaveBeenCalledWith('run1', 'blk/nest/mul', 'tensor'));
  });

  it('asks for the bare id again once back at the top level', async () => {
    enter('blk');
    const { result } = renderHook(() => usePortFetches('run1', MUL));
    await waitFor(() => expect(result.current[KEY]?.data).toBeTruthy());

    act(() => enter());
    await waitFor(() => expect(mockOutput).toHaveBeenLastCalledWith('run1', 'mul', 'tensor'));
  });
});

// ── Captures exist only once a node has returned ────────────────────────────
// The engine writes a node's captures after the node returns and answers 404
// for anything not written yet, so a port read mid-run has three states, not
// two: nothing to read YET is different from nothing to read.

describe('capturePhase', () => {
  const TERMINAL: ExecutionStatus[] = ['completed', 'cached', 'error', 'skipped', 'interrupted'];

  it('is running while the node runs in a run that is in progress', () => {
    expect(capturePhase('running', true)).toBe('running');
  });

  it('is pending for a node the run has not reached, and never claims it is running', () => {
    expect(capturePhase('idle', true)).toBe('pending');
    expect(capturePhase(undefined, true)).toBe('pending');
  });

  it('is settled once the node reports any terminal status', () => {
    for (const status of TERMINAL) expect(capturePhase(status, true)).toBe('settled');
  });

  it('is settled for every status when no run is in progress', () => {
    const all: (ExecutionStatus | undefined)[] = [undefined, 'idle', 'running', ...TERMINAL];
    for (const status of all) expect(capturePhase(status, false)).toBe('settled');
  });
});

describe('capturePhaseNoteKey', () => {
  it('names a line for the two phases that have nothing to read yet, and none once settled', () => {
    expect(capturePhaseNoteKey('running')).toBe('inspector.nodeRunning');
    expect(capturePhaseNoteKey('pending')).toBe('inspector.nodePending');
    expect(capturePhaseNoteKey('settled')).toBeNull();
  });
});

describe('takeDuePorts', () => {
  const A = { nodeId: 'a', port: 'out' };
  const B = { nodeId: 'b', port: 'out' };

  it('hands out a settled port once, however often it is asked', () => {
    const asked = new Set<string>();
    expect(takeDuePorts([A, B], ['settled', 'settled'], asked)).toEqual([A, B]);
    expect(takeDuePorts([A, B], ['settled', 'settled'], asked)).toEqual([]);
  });

  it('holds back a port whose owner has not returned, and releases it when it has', () => {
    const asked = new Set<string>();
    expect(takeDuePorts([A, B], ['settled', 'running'], asked)).toEqual([A]);
    expect(takeDuePorts([A, B], ['settled', 'pending'], asked)).toEqual([]);
    // B finishing must not hand A out a second time.
    expect(takeDuePorts([A, B], ['settled', 'settled'], asked)).toEqual([B]);
  });

  it('hands a port out again after its owner ran a second time', () => {
    const asked = new Set<string>();
    expect(takeDuePorts([A], ['settled'], asked)).toEqual([A]);
    expect(takeDuePorts([A], ['running'], asked)).toEqual([]);
    expect(takeDuePorts([A], ['settled'], asked)).toEqual([A]);
  });

  it('forgets a port that left the view, so coming back to it reads it afresh', () => {
    const asked = new Set<string>();
    expect(takeDuePorts([A], ['settled'], asked)).toEqual([A]);
    expect(takeDuePorts([B], ['settled'], asked)).toEqual([B]);
    expect(takeDuePorts([A], ['settled'], asked)).toEqual([A]);
  });

  it('asks once for a port listed twice', () => {
    expect(takeDuePorts([A, { ...A }], ['settled', 'settled'], new Set())).toEqual([A]);
  });
});

// ── A node the last run has nothing for ─────────────────────────────────────
// Added after the run, or moved into a block made after it: the run never
// captured it under the id the canvas now has, so a request could only be
// answered 404, which reads as "Run data expired". The finished run's port
// list says so up front, and the row says what is true instead.

describe('usePortFetches — a node the last run has nothing for', () => {
  const RAN = { nodeId: 'a', port: 'out' };
  const ADDED = { nodeId: 'added', port: 'out' };

  function patchActiveTab(patch: Partial<TabState>) {
    const { tabs, activeTabId } = useTabStore.getState();
    useTabStore.setState({
      tabs: tabs.map((t) => (t.id === activeTabId ? { ...t, ...patch } : t)),
    });
  }

  function nodeAt(id: string, executionStatus: ExecutionStatus) {
    return { ...node(id), data: { ...node(id).data, executionStatus } };
  }

  beforeEach(() => {
    mockOutput.mockReset();
    mockOutput.mockImplementation(async (_run, nodeId) => scalar(nodeId, 1));
    mockList.mockResolvedValue([
      { node_id: 'a', port: 'out', type: 'scalar', full_shape: null },
      { node_id: 'blk/a', port: 'out', type: 'scalar', full_shape: null },
    ]);
  });

  afterEach(() => {
    leaveAll();
    patchActiveTab({ nodes: [], status: 'idle' });
  });

  it('says so instead of asking for it, and asks for the node the run had', async () => {
    const { result } = renderHook(() => usePortFetches('run1', [RAN, ADDED]));
    await waitFor(() =>
      expect(result.current[keyOf('added', 'out')]).toEqual({
        loading: false,
        error: null,
        errorKey: null,
        noteKey: 'inspector.capture.notInRun',
        data: null,
      }),
    );
    await waitFor(() => expect(result.current[keyOf('a', 'out')]?.data).toBeTruthy());
    expect(mockOutput).toHaveBeenCalledTimes(1);
    expect(mockOutput).toHaveBeenCalledWith('run1', 'a', 'out');
  });

  it('reads the run by its own ids inside an open block', async () => {
    enter('blk');
    const { result } = renderHook(() => usePortFetches('run1', [RAN, ADDED]));
    await waitFor(() =>
      expect(result.current[keyOf('added', 'out')]?.noteKey).toBe('inspector.capture.notInRun'),
    );
    await waitFor(() => expect(result.current[keyOf('a', 'out')]?.data).toBeTruthy());
    expect(mockOutput).toHaveBeenCalledTimes(1);
    expect(mockOutput).toHaveBeenCalledWith('run1', 'blk/a', 'out');
  });

  it('says a node that failed in the run failed, not that it was not in it', async () => {
    patchActiveTab({ nodes: [nodeAt('added', 'error')] });
    const { result } = renderHook(() => usePortFetches('run1', [ADDED]));
    await waitFor(() =>
      expect(result.current[keyOf('added', 'out')]?.noteKey).toBe(
        'inspector.capture.failedInRun',
      ),
    );
    expect(mockOutput).not.toHaveBeenCalled();
  });

  it('asks as before, and still reads expiry as expiry, when the list cannot be read', async () => {
    mockList.mockRejectedValue(new Error('offline'));
    mockOutput.mockRejectedValue(new RunDataExpiredError('run1'));
    const { result } = renderHook(() => usePortFetches('run1', [ADDED]));
    await waitFor(() =>
      expect(result.current[keyOf('added', 'out')]?.errorKey).toBe('inspector.dataExpired'),
    );
    expect(mockOutput).toHaveBeenCalledTimes(1);
  });

  it('does not read the list while a run is in progress, when it is still growing', async () => {
    patchActiveTab({ status: 'running', nodes: [nodeAt('a', 'completed')] });
    const { result } = renderHook(() => usePortFetches('run1', [RAN]));
    await waitFor(() => expect(result.current[keyOf('a', 'out')]?.data).toBeTruthy());
    expect(mockList).not.toHaveBeenCalled();
  });

  it('reads the list once per run, however many views ask', async () => {
    const first = renderHook(() => usePortFetches('run1', [ADDED]));
    const second = renderHook(() => usePortFetches('run1', [RAN, ADDED]));
    await waitFor(() =>
      expect(first.result.current[keyOf('added', 'out')]?.noteKey).toBe('inspector.capture.notInRun'),
    );
    await waitFor(() => expect(second.result.current[keyOf('a', 'out')]?.data).toBeTruthy());
    expect(mockList).toHaveBeenCalledTimes(1);
  });

  it('reads the run record once per finished run, across views and later passes', async () => {
    const first = renderHook(() => usePortFetches('run1', [ADDED]));
    await waitFor(() =>
      expect(first.result.current[keyOf('added', 'out')]?.noteKey).toBe('inspector.capture.notInRun'),
    );
    first.unmount();
    const later = renderHook(() => usePortFetches('run1', [RAN]));
    await waitFor(() => expect(later.result.current[keyOf('a', 'out')]?.data).toBeTruthy());
    expect(mockGetRun).toHaveBeenCalledTimes(1);
  });

  it('asks the run record again while it has not caught up with the run that just ended', async () => {
    // The run service sends the run's last event, which ends the tab's run,
    // a moment before it marks the run finished.
    mockGetRun
      .mockResolvedValueOnce(runRecord('running'))
      .mockResolvedValue(runRecord('succeeded'));
    const { result } = renderHook(() => usePortFetches('run1', [ADDED]));
    await waitFor(() =>
      expect(result.current[keyOf('added', 'out')]?.noteKey).toBe('inspector.capture.notInRun'),
    );
    expect(mockGetRun).toHaveBeenCalledTimes(2);
    expect(mockOutput).not.toHaveBeenCalled();
  });

  // A tab can name a run that is still going while the tab itself is not
  // running: Watch in the Runs panel, or a reload mid-run, and the tab may
  // never hear that run end. A node the run has not reached waits, as in the
  // tab's own run, and is read once the run's record says it is over.

  it('waits, without asking, for a node a watched run has not reached, and reads one it has', async () => {
    mockGetRun.mockResolvedValue(runRecord('running'));
    mockList.mockResolvedValue([{ node_id: 'a', port: 'out', type: 'scalar', full_shape: null }]);
    const { result } = renderHook(() => usePortFetches('run1', [RAN, ADDED]));
    await waitFor(() =>
      expect(result.current[keyOf('added', 'out')]?.noteKey).toBe('inspector.nodePending'),
    );
    await waitFor(() => expect(result.current[keyOf('a', 'out')]?.data).toBeTruthy());
    expect(mockOutput).toHaveBeenCalledTimes(1);
    expect(mockOutput).toHaveBeenCalledWith('run1', 'a', 'out');
  });

  it('asks for nothing while a watched run is still queued: nothing in it has run', async () => {
    mockGetRun.mockResolvedValue(runRecord('queued'));
    const { result } = renderHook(() => usePortFetches('run1', [RAN, ADDED]));
    await waitFor(() =>
      expect(result.current[keyOf('a', 'out')]?.noteKey).toBe('inspector.nodePending'),
    );
    expect(result.current[keyOf('added', 'out')]?.noteKey).toBe('inspector.nodePending');
    expect(mockList).not.toHaveBeenCalled();
    expect(mockOutput).not.toHaveBeenCalled();
  });

  it('reads a waiting node by itself once the run ends, and stops asking about the run', async () => {
    // A reload mid-run: the tab is idle, the run goes on.
    mockGetRun.mockResolvedValue(runRecord('running'));
    mockList.mockResolvedValue([{ node_id: 'a', port: 'out', type: 'scalar', full_shape: null }]);
    const { result } = renderHook(() => usePortFetches('run1', [ADDED]));
    await waitFor(() =>
      expect(result.current[keyOf('added', 'out')]?.noteKey).toBe('inspector.nodePending'),
    );
    expect(mockOutput).not.toHaveBeenCalled();

    // The run finishes, and `added` with it. Nothing else changes in the tab.
    mockGetRun.mockResolvedValue(runRecord('succeeded'));
    mockList.mockResolvedValue([
      { node_id: 'a', port: 'out', type: 'scalar', full_shape: null },
      { node_id: 'added', port: 'out', type: 'scalar', full_shape: null },
    ]);
    await waitFor(() => expect(result.current[keyOf('added', 'out')]?.data).toBeTruthy());
    expect(mockOutput).toHaveBeenCalledWith('run1', 'added', 'out');

    const asked = mockGetRun.mock.calls.length;
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 40));
    });
    expect(mockGetRun.mock.calls.length).toBe(asked);
  });

  it('stops asking about the run when the view goes away', async () => {
    mockGetRun.mockResolvedValue(runRecord('running'));
    mockList.mockResolvedValue([]);
    const { result, unmount } = renderHook(() => usePortFetches('run1', [ADDED]));
    await waitFor(() =>
      expect(result.current[keyOf('added', 'out')]?.noteKey).toBe('inspector.nodePending'),
    );
    unmount();
    const asked = mockGetRun.mock.calls.length;
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 40));
    });
    expect(mockGetRun.mock.calls.length).toBe(asked);
  });

  it('stops asking about the run once the tab runs it itself', async () => {
    // Watch, before the server acknowledges the attach: the tab is still idle.
    mockGetRun.mockResolvedValue(runRecord('running'));
    mockList.mockResolvedValue([]);
    patchActiveTab({ nodes: [nodeAt('added', 'idle')] });
    const { result, unmount } = renderHook(() => usePortFetches('run1', [ADDED]));
    await waitFor(() =>
      expect(result.current[keyOf('added', 'out')]?.noteKey).toBe('inspector.nodePending'),
    );

    // The attach is acknowledged: the tab's own frames follow the run now.
    act(() => patchActiveTab({ status: 'running' }));
    const asked = mockGetRun.mock.calls.length;
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 40));
    });
    expect(mockGetRun.mock.calls.length).toBe(asked);
    expect(result.current[keyOf('added', 'out')]?.noteKey).toBe('inspector.nodePending');
    // Unmounted before the reset below, which would otherwise re-render it.
    unmount();
  });

  it('asks as before when a recorded run lists gradients only', async () => {
    // Every node that ran would be missing from such a list, not only the new ones.
    mockList.mockResolvedValue([
      { node_id: 'lin', port: 'out__grad', type: 'tensor', full_shape: [2] },
      { node_id: 'lin', port: '__weight_grad__weight', type: 'tensor', full_shape: [2, 2] },
    ]);
    const { result } = renderHook(() => usePortFetches('run1', [{ nodeId: 'relu', port: 'out' }]));
    await waitFor(() => expect(result.current[keyOf('relu', 'out')]?.data).toBeTruthy());
    expect(mockOutput).toHaveBeenCalledWith('run1', 'relu', 'out');
  });

  // ── A run the server holds nothing for ──
  // It failed before its first capture (Start -> TensorCreate with a bad
  // shape), so the port list answers 404, though the run record exists.

  describe('in a run that stored nothing before it failed', () => {
    beforeEach(() => {
      mockGetRun.mockResolvedValue(runRecord('failed', { record_outputs: true }));
      mockList.mockRejectedValue(new RunDataExpiredError('run1'));
      mockOutput.mockRejectedValue(new RunDataExpiredError('run1'));
    });

    it('says the node that failed failed, with no request', async () => {
      patchActiveTab({ nodes: [nodeAt('added', 'error')] });
      const { result } = renderHook(() => usePortFetches('run1', [ADDED]));
      await waitFor(() =>
        expect(result.current[keyOf('added', 'out')]?.noteKey).toBe('inspector.capture.failedInRun'),
      );
      expect(mockOutput).not.toHaveBeenCalled();
    });

    it('asks for a node the run never reached, and lets the 404 read as expired', async () => {
      // With no list there is nothing certain to say about it: its status
      // cannot tell "never reached" from "its values expired".
      patchActiveTab({ nodes: [nodeAt('added', 'idle')] });
      const { result } = renderHook(() => usePortFetches('run1', [ADDED]));
      await waitFor(() =>
        expect(result.current[keyOf('added', 'out')]?.errorKey).toBe('inspector.dataExpired'),
      );
      expect(mockOutput).toHaveBeenCalledTimes(1);
    });

    it('goes by no card status inside an open block: it asks, and the 404 reads as expired', async () => {
      // Inner cards never get a run status; one they carry is not this run's.
      enter('blk');
      patchActiveTab({ nodes: [nodeAt('added', 'error')] });
      const { result } = renderHook(() => usePortFetches('run1', [ADDED]));
      await waitFor(() =>
        expect(result.current[keyOf('added', 'out')]?.errorKey).toBe('inspector.dataExpired'),
      );
      expect(mockOutput).toHaveBeenCalledWith('run1', 'blk/added', 'out');
    });

    it('still asks for a node that completed, whose values really expired', async () => {
      patchActiveTab({ nodes: [nodeAt('a', 'completed')] });
      const { result } = renderHook(() => usePortFetches('run1', [RAN]));
      await waitFor(() =>
        expect(result.current[keyOf('a', 'out')]?.errorKey).toBe('inspector.dataExpired'),
      );
      expect(mockOutput).toHaveBeenCalledTimes(1);
    });
  });

  // ── A run made with Record node outputs off ──
  // Nothing forward was kept, so every request could only 404 and read as
  // expired; re-running with the setting still off changes nothing.

  describe('in a run made with Record node outputs off', () => {
    beforeEach(() => {
      mockGetRun.mockResolvedValue(runRecord('succeeded', { record_outputs: false }));
      // With Capture gradients on, the run kept gradients only.
      mockList.mockResolvedValue([
        { node_id: 'a', port: 'out__grad', type: 'tensor', full_shape: [2] },
      ]);
    });

    it('asks for the setting instead of asking the server, while it is still off', async () => {
      patchActiveTab({ recordOutputs: false, nodes: [nodeAt('a', 'completed')] });
      const { result } = renderHook(() => usePortFetches('run1', [RAN, ADDED]));
      await waitFor(() =>
        expect(result.current[keyOf('a', 'out')]?.noteKey).toBe('inspector.empty.notRunHint'),
      );
      expect(result.current[keyOf('added', 'out')]?.noteKey).toBe('inspector.empty.notRunHint');
      expect(mockOutput).not.toHaveBeenCalled();
      // Not even the run's list: no answer it gives would change a word.
      expect(mockList).not.toHaveBeenCalled();
    });

    it('asks for a run instead once the setting is on', async () => {
      patchActiveTab({ recordOutputs: true, nodes: [nodeAt('a', 'completed')] });
      const { result } = renderHook(() => usePortFetches('run1', [RAN]));
      await waitFor(() =>
        expect(result.current[keyOf('a', 'out')]?.noteKey).toBe('inspector.capture.runHint'),
      );
      expect(mockOutput).not.toHaveBeenCalled();
      expect(mockList).not.toHaveBeenCalled();
    });

    it('follows the setting as it is switched, with the node still selected', async () => {
      patchActiveTab({ recordOutputs: true, nodes: [nodeAt('a', 'completed')] });
      const { result, unmount } = renderHook(() => usePortFetches('run1', [RAN]));
      await waitFor(() =>
        expect(result.current[keyOf('a', 'out')]?.noteKey).toBe('inspector.capture.runHint'),
      );
      act(() => patchActiveTab({ recordOutputs: false }));
      expect(result.current[keyOf('a', 'out')]?.noteKey).toBe('inspector.empty.notRunHint');
      act(() => patchActiveTab({ recordOutputs: true }));
      expect(result.current[keyOf('a', 'out')]?.noteKey).toBe('inspector.capture.runHint');
      // Nothing is asked again for it: the note is chosen as the row is drawn.
      expect(mockOutput).not.toHaveBeenCalled();
      expect(mockGetRun).toHaveBeenCalledTimes(1);
      unmount();
    });

    it('still says a node that failed failed', async () => {
      patchActiveTab({ recordOutputs: false, nodes: [nodeAt('added', 'error')] });
      const { result } = renderHook(() => usePortFetches('run1', [ADDED]));
      await waitFor(() =>
        expect(result.current[keyOf('added', 'out')]?.noteKey).toBe('inspector.capture.failedInRun'),
      );
      expect(mockOutput).not.toHaveBeenCalled();
      expect(mockList).not.toHaveBeenCalled();
    });
  });
});
