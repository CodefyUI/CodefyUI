import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useGraphExecution } from './useGraphExecution';
import { useTabStore } from '../store/tabStore';
import { discardTabNodeUpdates } from '../store/nodeUpdateQueue';
import { useToastStore } from '../store/toastStore';
import { useUIStore } from '../store/uiStore';
import { useI18n } from '../i18n';
import { dismissValidationToasts } from '../utils/validationToasts';

/**
 * Run's validation toasts, through the hook that raises them: they name the
 * node by its title, offer to show it, and are one set that the next Run and
 * a tab switch take down. Before, every Run added a full set of raw English
 * lines on top of the last, error toasts never time out, and they stayed
 * across tab switches and successful runs.
 */

// The hook calls validateGraph() before sending and getRun() on mount.
vi.mock('../api/rest', () => ({
  validateGraph: vi.fn(),
  getRun: vi.fn(),
}));
import { getRun, validateGraph } from '../api/rest';
const validateGraphMock = vi.mocked(validateGraph);
const getRunMock = vi.mocked(getRun);

interface FakeWs {
  on: ReturnType<typeof vi.fn>;
  off: ReturnType<typeof vi.fn>;
  send: ReturnType<typeof vi.fn>;
  connect: ReturnType<typeof vi.fn>;
  connected: boolean;
}

/** Just what the hook uses: on/off/send/connect/connected. */
function makeFakeWs(): FakeWs {
  const ws: FakeWs = {
    connected: true,
    on: vi.fn(),
    off: vi.fn(),
    send: vi.fn(),
    connect: vi.fn(async () => {
      ws.connected = true;
    }),
  };
  return ws;
}

const ENCODER = '3f2a9c1e-0b7d-4c55-9e21-6a1f0c2d8b41';

/** A tab whose graph has an entry point, so Run goes on to the server check. */
function makeTab(id: string, overrides: Partial<any> = {}): any {
  return {
    id,
    name: id,
    nodes: [
      { id: 'start', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'Start', type: 'Start', params: {} } },
      { id: ENCODER, type: 'baseNode', position: { x: 300, y: 120 }, data: { label: 'Encoder', type: 'Linear', params: {} } },
    ],
    edges: [{ id: 'e1', source: 'start', target: ENCODER, data: { type: 'trigger' } }],
    selectedNodeId: null,
    presetModalNodeId: null,
    layersModalNodeId: null,
    undoStack: [],
    redoStack: [],
    dirtyNodeIds: new Set<string>(),
    status: 'idle',
    logs: [],
    ws: makeFakeWs(),
    outputSummaries: {},
    recordOutputs: true,
    lastRunId: null,
    lastRunCursor: 0,
    activeSegment: null,
    segmentGroups: [],
    subgraphStack: [],
    verboseMode: false,
    graphId: `graph-${id}`,
    weightsPersistent: false,
    backwardMode: false,
    autoBackward: false,
    ...overrides,
  };
}

const MISSING_TENSOR = {
  message: `Missing required input 'tensor' on node ${ENCODER} (Linear) -- connect an output to this port`,
  code: 'missing_input',
  node_id: ENCODER,
  params: { port: 'tensor', type: 'Linear' },
};

const REFUSED = {
  valid: false,
  errors: [MISSING_TENSOR.message],
  issues: [MISSING_TENSOR],
};

const toasts = () => useToastStore.getState().toasts;
const tabById = (id: string): any => useTabStore.getState().tabs.find((t) => t.id === id);

async function run(result: { current: ReturnType<typeof useGraphExecution> }) {
  await act(async () => {
    await result.current.execute();
  });
}

beforeEach(() => {
  validateGraphMock.mockReset();
  validateGraphMock.mockResolvedValue({ valid: true, errors: [] });
  getRunMock.mockReset();
  getRunMock.mockResolvedValue(null);
  useI18n.setState({ locale: 'en' });
  useToastStore.setState({ toasts: [] });
  useUIStore.setState({ layoutFitRequest: null });
  useTabStore.setState({ tabs: [makeTab('t1'), makeTab('t2')], activeTabId: 't1' });
});

afterEach(() => {
  // Module state: a test that ended with toasts up must not hand their ids on.
  dismissValidationToasts();
  vi.restoreAllMocks();
  discardTabNodeUpdates();
});

describe('Run validation toasts', () => {
  it('name the node by its title and offer to show it', async () => {
    validateGraphMock.mockResolvedValueOnce(REFUSED);
    const { result } = renderHook(() => useGraphExecution());

    await run(result);

    expect(toasts().map((toast) => toast.message)).toEqual([
      'Encoder: input "tensor" is not connected',
    ]);
    expect(toasts()[0].type).toBe('error');
    act(() => toasts()[0].action!.onClick());
    expect(tabById('t1').selectedNodeId).toBe(ENCODER);
    expect(useUIStore.getState().layoutFitRequest).not.toBeNull();
    expect(tabById('t1').ws.send).not.toHaveBeenCalled();
  });

  it('are in Chinese when the UI is', async () => {
    useI18n.setState({ locale: 'zh-TW' });
    validateGraphMock.mockResolvedValueOnce(REFUSED);
    const { result } = renderHook(() => useGraphExecution());

    await run(result);

    expect(toasts().map((toast) => toast.message)).toEqual(['「Encoder」的輸入「tensor」尚未連線']);
    expect(toasts()[0].action?.label).toBe('顯示');
  });

  it('read an older server that sends sentences only', async () => {
    validateGraphMock.mockResolvedValueOnce({ valid: false, errors: [MISSING_TENSOR.message] });
    const { result } = renderHook(() => useGraphExecution());

    await run(result);

    expect(toasts().map((toast) => toast.message)).toEqual([
      "Missing required input 'tensor' on node Encoder (Linear) -- connect an output to this port",
    ]);
  });

  it('do not pile up when Run is pressed again on the same graph', async () => {
    validateGraphMock.mockResolvedValue(REFUSED);
    const { result } = renderHook(() => useGraphExecution());

    await run(result);
    await run(result);

    expect(toasts()).toHaveLength(1);
  });

  it('are gone before a Run whose graph passes is sent', async () => {
    validateGraphMock.mockResolvedValueOnce(REFUSED);
    const { result } = renderHook(() => useGraphExecution());
    await run(result);
    expect(toasts()).toHaveLength(1);

    const ws = tabById('t1').ws as FakeWs;
    let upAtSend = -1;
    ws.send.mockImplementation(() => {
      upAtSend = toasts().length;
    });
    await run(result);

    expect(ws.send).toHaveBeenCalledTimes(1);
    expect(upAtSend).toBe(0);
    expect(toasts()).toEqual([]);
  });

  it('are gone when the next Run cannot reach the check', async () => {
    validateGraphMock.mockResolvedValueOnce(REFUSED);
    const { result } = renderHook(() => useGraphExecution());
    await run(result);

    validateGraphMock.mockRejectedValueOnce(new Error('unreachable'));
    await run(result);

    expect(toasts()).toEqual([]);
  });

  it('go when another tab comes to the front', async () => {
    validateGraphMock.mockResolvedValueOnce(REFUSED);
    const { result } = renderHook(() => useGraphExecution());
    await run(result);
    expect(toasts()).toHaveLength(1);

    act(() => useTabStore.getState().setActiveTab('t2'));

    expect(toasts()).toEqual([]);
  });

  it('leave every other toast alone', async () => {
    useToastStore.getState().addToast('Graph saved', 'error');
    validateGraphMock.mockResolvedValueOnce(REFUSED);
    const { result } = renderHook(() => useGraphExecution());
    await run(result);

    act(() => useTabStore.getState().setActiveTab('t2'));

    expect(toasts().map((toast) => toast.message)).toEqual(['Graph saved']);
  });

  it('never land on the tab the user switched to while the check ran', async () => {
    let answer: (value: any) => void = () => {};
    validateGraphMock.mockImplementationOnce(
      () => new Promise((resolve) => { answer = resolve; }) as any);
    const { result } = renderHook(() => useGraphExecution());

    let pending: Promise<void> = Promise.resolve();
    await act(async () => {
      pending = result.current.execute();
      await Promise.resolve();
    });
    act(() => useTabStore.getState().setActiveTab('t2'));
    await act(async () => {
      answer(REFUSED);
      await pending;
    });

    expect(toasts()).toEqual([]);
  });

  it('include the missing-Start refusal, which the next Run also clears', async () => {
    useTabStore.setState({
      tabs: [makeTab('t1', { edges: [] }), makeTab('t2')],
      activeTabId: 't1',
    });
    const { result } = renderHook(() => useGraphExecution());

    await run(result);
    const noStart = useI18n.getState().t('execution.error.noEntryPoints');
    expect(toasts().map((toast) => toast.message)).toEqual([noStart]);
    expect(validateGraphMock).not.toHaveBeenCalled();

    // Start wired back in, and Run again.
    act(() => {
      useTabStore.setState({
        tabs: [makeTab('t1'), makeTab('t2')],
        activeTabId: 't1',
      });
    });
    await run(result);

    expect(toasts()).toEqual([]);
    expect(tabById('t1').ws.send).toHaveBeenCalledTimes(1);
  });

  it('never run a refused graph when drawing them fails', async () => {
    // Only the request may fail quietly: an unreachable check lets the run
    // go ahead, a fault in the toasts must not be mistaken for one.
    validateGraphMock.mockResolvedValueOnce(REFUSED);
    const realAddToast = useToastStore.getState().addToast;
    useToastStore.setState({
      addToast: vi.fn(() => {
        throw new Error('toast layer broke');
      }),
    });
    const { result } = renderHook(() => useGraphExecution());

    try {
      await act(async () => {
        await expect(result.current.execute()).rejects.toThrow('toast layer broke');
      });
    } finally {
      useToastStore.setState({ addToast: realAddToast });
    }

    expect(tabById('t1').ws.send).not.toHaveBeenCalled();
    expect(tabById('t1').status).toBe('idle');
  });

  it('go when the last tab closes and the toolbar unmounts the hook', async () => {
    validateGraphMock.mockResolvedValueOnce(REFUSED);
    const { result, unmount } = renderHook(() => useGraphExecution());
    await run(result);

    unmount();

    expect(toasts()).toEqual([]);
  });
});
