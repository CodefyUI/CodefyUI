import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useGraphExecution } from './useGraphExecution';
import { useTabStore } from '../store/tabStore';
import { useToastStore } from '../store/toastStore';
import { useUIStore } from '../store/uiStore';

// Mock the REST validation endpoint — the hook calls validateGraph() before
// sending. Each test drives its resolved/rejected value.
vi.mock('../api/rest', () => ({
  validateGraph: vi.fn(),
}));
import { validateGraph } from '../api/rest';
const validateGraphMock = vi.mocked(validateGraph);

// ── Fake WebSocket ───────────────────────────────────────────────────────────
// The real ExecutionWebSocket opens a browser WebSocket. The hook only uses
// on/off/send/connect/connected, so a hand-rolled fake lets us both assert
// calls and *drive* the registered handlers to exercise every WS code path.
interface FakeWs {
  on: ReturnType<typeof vi.fn>;
  off: ReturnType<typeof vi.fn>;
  send: ReturnType<typeof vi.fn>;
  connect: ReturnType<typeof vi.fn>;
  connected: boolean;
  handlers: Map<string, Array<(data: unknown) => void>>;
  emit: (type: string, data?: unknown) => void;
}

function makeFakeWs(connected = true): FakeWs {
  const handlers = new Map<string, Array<(data: unknown) => void>>();
  const ws: FakeWs = {
    handlers,
    connected,
    on: vi.fn((type: string, h: (d: unknown) => void) => {
      if (!handlers.has(type)) handlers.set(type, []);
      handlers.get(type)!.push(h);
    }),
    off: vi.fn((type: string, h: (d: unknown) => void) => {
      const arr = handlers.get(type);
      if (arr) handlers.set(type, arr.filter((fn) => fn !== h));
    }),
    send: vi.fn(),
    connect: vi.fn(async () => {}),
    emit: (type: string, data: unknown = {}) => {
      for (const h of handlers.get(type) ?? []) h(data);
    },
  };
  return ws;
}

/** Construct a TabState-shaped object with a fake ws and overridable fields. */
function makeTab(id: string, overrides: Partial<any> = {}): any {
  return {
    id,
    name: id,
    nodes: [],
    edges: [],
    selectedNodeId: null,
    presetModalNodeId: null,
    subgraphModalNodeId: null,
    undoStack: [],
    redoStack: [],
    dirtyNodeIds: new Set<string>(),
    status: 'idle',
    logs: [],
    ws: makeFakeWs(),
    outputSummaries: {},
    recordOutputs: true,
    lastRunId: null,
    activeSegment: null,
    segmentGroups: [],
    verboseMode: false,
    graphId: `graph-${id}`,
    weightsPersistent: true,
    backwardMode: false,
    autoBackward: false,
    ...overrides,
  };
}

function setTabs(tabs: any[], activeTabId = tabs[0]?.id) {
  useTabStore.setState({ tabs, activeTabId });
}

beforeEach(() => {
  validateGraphMock.mockReset();
  validateGraphMock.mockResolvedValue({ valid: true, errors: [] });
  useToastStore.setState({ toasts: [] });
  // Default: one connected tab with a trigger edge so execute() proceeds.
  setTabs([
    makeTab('t1', {
      nodes: [{ id: 'n1', data: { label: 'Node One' } }],
      edges: [{ id: 'e1', source: 's', target: 'n1', data: { type: 'trigger' } }],
    }),
  ]);
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function tabById(id: string): any {
  return useTabStore.getState().tabs.find((t) => t.id === id);
}

// ── WS listener attachment (the useEffect) ────────────────────────────────────

describe('useGraphExecution - WS listener lifecycle', () => {
  it('attaches the five event handlers to existing tabs on mount', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());

    const types = ws.on.mock.calls.map((c) => c[0]);
    expect(types).toEqual([
      'node_status',
      'execution_complete',
      'execution_error',
      'execution_start',
      'execution_stopped',
    ]);
  });

  it('detaches all handlers on unmount', () => {
    const ws = tabById('t1').ws as FakeWs;
    const { unmount } = renderHook(() => useGraphExecution());
    unmount();
    const offTypes = ws.off.mock.calls.map((c) => c[0]);
    expect(offTypes).toContain('node_status');
    expect(ws.off).toHaveBeenCalledTimes(5);
  });

  it('does not re-attach to a tab that is already attached', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    const callsAfterMount = ws.on.mock.calls.length;

    // A no-op state change re-runs the subscribe callback, which calls
    // attachTab('t1') again — but it must early-return (already attached).
    act(() => {
      useTabStore.setState((s) => ({ tabs: [...s.tabs] }));
    });
    expect(ws.on.mock.calls.length).toBe(callsAfterMount);
  });

  it('attaches to a newly added tab and detaches a removed tab', () => {
    renderHook(() => useGraphExecution());

    const t2 = makeTab('t2');
    act(() => {
      useTabStore.setState((s) => ({ tabs: [...s.tabs, t2] }));
    });
    expect((t2.ws as FakeWs).on).toHaveBeenCalledTimes(5);

    // Remove t1 → its handlers must be released (detachTab path).
    const ws1 = tabById('t1') ? (tabById('t1').ws as FakeWs) : null;
    const removedWs = ws1!;
    act(() => {
      useTabStore.setState((s) => ({ tabs: s.tabs.filter((t) => t.id !== 't1'), activeTabId: 't2' }));
    });
    expect(removedWs.off).toHaveBeenCalledTimes(5);
  });
});

// ── onNodeStatus handler branches ─────────────────────────────────────────────

describe('useGraphExecution - node_status handler', () => {
  it('handles progress events with epoch/config logging', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());

    act(() => {
      ws.emit('node_status', {
        node_id: 'n1',
        status: 'progress',
        progress: { event: 'epoch', value: 1 },
      });
    });

    const tab = tabById('t1');
    expect(tab.nodes[0].data.progress).toEqual({ event: 'epoch', value: 1 });
    // epoch event → a __PROGRESS__ log is appended.
    expect(tab.logs.some((l: any) => l.message.startsWith('__PROGRESS__:'))).toBe(true);
  });

  it('handles progress events WITHOUT epoch/config (no progress log)', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());

    act(() => {
      ws.emit('node_status', {
        node_id: 'n1',
        status: 'progress',
        progress: { event: 'batch', value: 5 },
      });
    });

    const tab = tabById('t1');
    expect(tab.nodes[0].data.progress).toEqual({ event: 'batch', value: 5 });
    expect(tab.logs.some((l: any) => l.message.startsWith('__PROGRESS__:'))).toBe(false);
  });

  it('suppresses logs for running status but updates node status', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());

    act(() => {
      ws.emit('node_status', { node_id: 'n1', status: 'running' });
    });
    const tab = tabById('t1');
    expect(tab.nodes[0].data.executionStatus).toBe('running');
    // running is suppressed — no "Node ... running" log.
    expect(tab.logs.some((l: any) => l.message.includes('running'))).toBe(false);
  });

  it('suppresses logs for cached status', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('node_status', { node_id: 'n1', status: 'cached' });
    });
    const tab = tabById('t1');
    expect(tab.logs.some((l: any) => l.message.includes('cached'))).toBe(false);
  });

  it('logs completed status with the node label as success', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('node_status', { node_id: 'n1', status: 'completed' });
    });
    const log = tabById('t1').logs.find((l: any) => l.message.includes('completed'));
    expect(log.message).toBe('Node Node One completed');
    expect(log.type).toBe('success');
  });

  it('logs error status with the error appended as type error', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('node_status', { node_id: 'n1', status: 'error', error: 'boom' });
    });
    const log = tabById('t1').logs.find((l: any) => l.type === 'error');
    expect(log.message).toBe('Node Node One error: boom');
  });

  it('logs a non-terminal/non-error status (e.g. skipped) as info', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('node_status', { node_id: 'n1', status: 'skipped' });
    });
    const log = tabById('t1').logs.find((l: any) => l.message.includes('skipped'));
    expect(log.type).toBe('info');
  });

  it('falls back to a truncated node id when the node label is missing', () => {
    // Active tab has no node matching node_id → label fallback to id.slice(0,8).
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('node_status', { node_id: 'abcdefgh123456', status: 'completed' });
    });
    const log = tabById('t1').logs.find((l: any) => l.message.includes('completed'));
    expect(log.message).toBe('Node abcdefgh completed');
  });

  it('appends log, image and output_summary side-channels', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('node_status', {
        node_id: 'n1',
        status: 'completed',
        log: 'hello log',
        image: 'data:img',
        output_summary: { out: { shape: [1] } },
      });
    });
    const tab = tabById('t1');
    expect(tab.logs.some((l: any) => l.message === 'hello log')).toBe(true);
    expect(tab.logs.some((l: any) => l.message === '__IMAGE__:data:img')).toBe(true);
    expect(tab.outputSummaries.n1).toEqual({ out: { shape: [1] } });
  });
});

// ── Other execution lifecycle events ──────────────────────────────────────────

describe('useGraphExecution - lifecycle events', () => {
  it('execution_complete sets status completed and logs success', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => ws.emit('execution_complete'));
    const tab = tabById('t1');
    expect(tab.status).toBe('completed');
    expect(tab.logs.some((l: any) => l.message === 'Execution completed successfully')).toBe(true);
  });

  it('execution_error sets status error and logs the error', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => ws.emit('execution_error', { error: 'kaboom' }));
    const tab = tabById('t1');
    expect(tab.status).toBe('error');
    expect(tab.logs.some((l: any) => l.message === 'Execution error: kaboom')).toBe(true);
  });

  it('execution_start sets status running and records run_id when a string', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => ws.emit('execution_start', { run_id: 'run-xyz' }));
    const tab = tabById('t1');
    expect(tab.status).toBe('running');
    expect(tab.lastRunId).toBe('run-xyz');
    expect(tab.logs.some((l: any) => l.message === 'Execution started')).toBe(true);
  });

  it('execution_start does not set run_id when it is absent', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => ws.emit('execution_start', {}));
    expect(tabById('t1').lastRunId).toBeNull();
  });

  it('execution_stopped sets status idle and logs cancellation', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => ws.emit('execution_stopped'));
    const tab = tabById('t1');
    expect(tab.status).toBe('idle');
    expect(tab.logs.some((l: any) => l.message === 'Execution cancelled')).toBe(true);
  });
});

// ── execute() ─────────────────────────────────────────────────────────────────

describe('useGraphExecution - execute', () => {
  it('shows a toast and aborts when there are no entry points', async () => {
    setTabs([makeTab('t1', { nodes: [{ id: 'n1', data: {} }], edges: [] })]);
    const ws = tabById('t1').ws as FakeWs;
    const { result } = renderHook(() => useGraphExecution());

    await act(async () => {
      await result.current.execute();
    });

    expect(useToastStore.getState().toasts.length).toBe(1);
    expect(useToastStore.getState().toasts[0].type).toBe('error');
    expect(ws.send).not.toHaveBeenCalled();
  });

  it('connects when the ws is not connected, then sends execute', async () => {
    setTabs([
      makeTab('t1', {
        nodes: [{ id: 'n1', data: { label: 'N' } }],
        edges: [{ id: 'e1', source: 's', target: 'n1', data: { type: 'trigger' } }],
        ws: makeFakeWs(false),
      }),
    ]);
    const ws = tabById('t1').ws as FakeWs;
    const { result } = renderHook(() => useGraphExecution());

    await act(async () => {
      await result.current.execute();
    });

    expect(ws.connect).toHaveBeenCalledTimes(1);
    expect(ws.send).toHaveBeenCalledTimes(1);
    expect(ws.send.mock.calls[0][0].action).toBe('execute');
  });

  it('logs an error and aborts when the ws connection fails', async () => {
    const failingWs = makeFakeWs(false);
    failingWs.connect.mockRejectedValueOnce(new Error('no server'));
    setTabs([
      makeTab('t1', {
        nodes: [{ id: 'n1', data: {} }],
        edges: [{ id: 'e1', source: 's', target: 'n1', data: { type: 'trigger' } }],
        ws: failingWs,
      }),
    ]);
    const { result } = renderHook(() => useGraphExecution());

    await act(async () => {
      await result.current.execute();
    });

    expect(failingWs.send).not.toHaveBeenCalled();
    expect(
      tabById('t1').logs.some((l: any) => l.message === 'Failed to connect to execution server'),
    ).toBe(true);
  });

  it('shows per-error toasts and aborts when validation fails', async () => {
    validateGraphMock.mockResolvedValueOnce({ valid: false, errors: ['e1', 'e2'] });
    const ws = tabById('t1').ws as FakeWs;
    const { result } = renderHook(() => useGraphExecution());

    await act(async () => {
      await result.current.execute();
    });

    expect(useToastStore.getState().toasts.map((t) => t.message)).toEqual(['e1', 'e2']);
    expect(ws.send).not.toHaveBeenCalled();
  });

  it('proceeds to send even when the validation endpoint throws', async () => {
    validateGraphMock.mockRejectedValueOnce(new Error('unreachable'));
    const ws = tabById('t1').ws as FakeWs;
    const { result } = renderHook(() => useGraphExecution());

    await act(async () => {
      await result.current.execute();
    });

    expect(ws.send).toHaveBeenCalledTimes(1);
  });

  it('filters out note nodes and includes changed_nodes when dirty', async () => {
    // Make the active tab report a serialized graph with a note + a real node,
    // and a dirty node so changed_nodes is attached.
    setTabs([
      makeTab('t1', {
        nodes: [{ id: 'n1', data: { label: 'N' } }],
        edges: [{ id: 'e1', source: 's', target: 'n1', data: { type: 'trigger' } }],
        dirtyNodeIds: new Set(['n1']),
      }),
    ]);
    // Stub getSerializedGraph to return a note node we expect to be filtered.
    const realSerialize = useTabStore.getState().getSerializedGraph;
    useTabStore.setState({
      getSerializedGraph: () => ({
        nodes: [
          { id: 'n1', type: 'Dataset', position: { x: 0, y: 0 }, data: {} },
          { id: 'note1', type: 'note', position: { x: 0, y: 0 }, data: {} },
        ],
        edges: [{ id: 'e1', source: 'n1', target: 'n1' }],
        presets: [],
      }),
    } as any);

    const ws = tabById('t1').ws as FakeWs;
    const { result } = renderHook(() => useGraphExecution());

    await act(async () => {
      await result.current.execute();
    });

    const payload = ws.send.mock.calls[0][0];
    expect(payload.nodes.map((n: any) => n.id)).toEqual(['n1']); // note removed
    expect(payload.changed_nodes).toEqual(['n1']);
    expect(payload.record_outputs).toBe(true);
    expect(typeof payload.run_id).toBe('string');

    useTabStore.setState({ getSerializedGraph: realSerialize } as any);
  });

  it('passes graph-embedded presets to validation and the execute message (#84)', async () => {
    setTabs([
      makeTab('t1', {
        nodes: [{ id: 'n1', data: { label: 'N' } }],
        edges: [{ id: 'e1', source: 's', target: 'n1', data: { type: 'trigger' } }],
      }),
    ]);
    const presets = [{ preset_name: 'EmbeddedPr', nodes: [], edges: [] }];
    const serializedNodes = [
      { id: 'p', type: 'preset:EmbeddedPr', position: { x: 0, y: 0 }, data: {} },
    ];
    const realSerialize = useTabStore.getState().getSerializedGraph;
    useTabStore.setState({
      getSerializedGraph: () => ({ nodes: serializedNodes, edges: [], presets }),
    } as any);

    const ws = tabById('t1').ws as FakeWs;
    const { result } = renderHook(() => useGraphExecution());

    await act(async () => {
      await result.current.execute();
    });

    // Pre-run REST validation must see the embedded presets, otherwise a
    // portable graph fails "Unknown preset" before the run even starts.
    expect(validateGraphMock).toHaveBeenCalledWith(serializedNodes, [], presets);
    expect(ws.send.mock.calls[0][0].presets).toEqual(presets);

    useTabStore.setState({ getSerializedGraph: realSerialize } as any);
  });

  it('omits changed_nodes when nothing is dirty', async () => {
    const ws = tabById('t1').ws as FakeWs;
    const { result } = renderHook(() => useGraphExecution());

    await act(async () => {
      await result.current.execute();
    });

    expect('changed_nodes' in ws.send.mock.calls[0][0]).toBe(false);
  });

  it('sends the global device from the UI store in the execute payload', async () => {
    useUIStore.getState().setGlobalDevice('mps');
    const ws = tabById('t1').ws as FakeWs;
    const { result } = renderHook(() => useGraphExecution());

    await act(async () => {
      await result.current.execute();
    });

    expect(ws.send.mock.calls[0][0].device).toBe('mps');
    useUIStore.getState().setGlobalDevice('cpu'); // reset for other tests
  });

  it('uses the crypto.randomUUID run id when available', async () => {
    const ws = tabById('t1').ws as FakeWs;
    const { result } = renderHook(() => useGraphExecution());

    await act(async () => {
      await result.current.execute();
    });
    const runId = ws.send.mock.calls[0][0].run_id;
    // jsdom's crypto.randomUUID is present → not the fallback format.
    expect(runId.startsWith('run-')).toBe(false);
  });

  it('falls back to a timestamp run id when crypto.randomUUID is unavailable', async () => {
    // Replace crypto with an object lacking randomUUID to hit the fallback.
    vi.stubGlobal('crypto', {});
    const ws = tabById('t1').ws as FakeWs;
    const { result } = renderHook(() => useGraphExecution());

    await act(async () => {
      await result.current.execute();
    });
    expect(ws.send.mock.calls[0][0].run_id).toMatch(/^run-\d+-/);
  });
});

// ── stop() ────────────────────────────────────────────────────────────────────

describe('useGraphExecution - stop', () => {
  it('sends a stop action to the active tab ws', () => {
    const ws = tabById('t1').ws as FakeWs;
    const { result } = renderHook(() => useGraphExecution());
    act(() => {
      result.current.stop();
    });
    expect(ws.send).toHaveBeenCalledWith({ action: 'stop' });
  });
});

// Keep waitFor import used in case of async settle needs.
void waitFor;
