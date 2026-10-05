/**
 * The run-log note while "Persist weights between runs" is on (2.8.9).
 *
 * With the switch on, a run's numbers include the training earlier runs did,
 * and Export Python never does that. The log says so after every successful
 * run, so a student comparing the canvas with the exported script is told
 * why they differ. Off (the default), nothing is added.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useGraphExecution } from './useGraphExecution';
import { useTabStore } from '../store/tabStore';
import { discardTabNodeUpdates } from '../store/nodeUpdateQueue';
import { useToastStore } from '../store/toastStore';
import { useI18n } from '../i18n';

// The hook calls validateGraph() before sending and getRun() on mount; this
// file sends nothing and re-attaches to nothing, so both just answer.
vi.mock('../api/rest', () => ({
  validateGraph: vi.fn(),
  getRun: vi.fn(),
}));
import { getRun, validateGraph } from '../api/rest';

// The same hand-rolled socket `useGraphExecution.test.ts` uses: the hook only
// calls on/off/send/connect, and `emit` drives the handlers it registered.
interface FakeWs {
  on: ReturnType<typeof vi.fn>;
  off: ReturnType<typeof vi.fn>;
  send: ReturnType<typeof vi.fn>;
  connect: ReturnType<typeof vi.fn>;
  connected: boolean;
  emit: (type: string, data?: unknown) => void;
}

function makeFakeWs(): FakeWs {
  const handlers = new Map<string, Array<(data: unknown) => void>>();
  const ws: FakeWs = {
    connected: true,
    on: vi.fn((type: string, h: (d: unknown) => void) => {
      if (!handlers.has(type)) handlers.set(type, []);
      handlers.get(type)!.push(h);
    }),
    off: vi.fn((type: string, h: (d: unknown) => void) => {
      const arr = handlers.get(type);
      if (arr) handlers.set(type, arr.filter((fn) => fn !== h));
    }),
    send: vi.fn(),
    connect: vi.fn(async () => {
      ws.connected = true;
    }),
    emit: (type: string, data: unknown = {}) => {
      for (const h of handlers.get(type) ?? []) h(data);
      for (const h of handlers.get('*') ?? []) h({ type, ...(data as object) });
    },
  };
  return ws;
}

function makeTab(id: string, weightsPersistent: boolean, over: Record<string, unknown> = {}): any {
  return {
    id,
    name: id,
    nodes: [{ id: 'n1', data: { label: 'Node One' } }],
    edges: [],
    selectedNodeId: null,
    presetModalNodeId: null,
    layersModalNodeId: null,
    undoStack: [],
    redoStack: [],
    dirtyNodeIds: new Set<string>(),
    status: 'running',
    logs: [],
    ws: makeFakeWs(),
    outputSummaries: {},
    recordOutputs: true,
    lastRunId: null,
    lastRunCursor: 0,
    activeSegment: null,
    segmentGroups: [],
    verboseMode: false,
    graphId: `graph-${id}`,
    weightsPersistent,
    backwardMode: false,
    autoBackward: false,
    ...over,
  };
}

function setupTab(weightsPersistent: boolean, over: Record<string, unknown> = {}): FakeWs {
  const tab = makeTab('t1', weightsPersistent, over);
  useTabStore.setState({ tabs: [tab], activeTabId: 't1' });
  return tab.ws as FakeWs;
}

/** An idle tab with an entry point, so Run actually sends. */
function setupRunnableTab(weightsPersistent: boolean): FakeWs {
  return setupTab(weightsPersistent, {
    status: 'idle',
    edges: [{ id: 'e1', source: 's', target: 'n1', data: { type: 'trigger' } }],
  });
}

/** Flip the switch on the tab, the way the Settings toggle does. */
function setSwitch(weightsPersistent: boolean): void {
  useTabStore.setState((s) => ({
    tabs: s.tabs.map((t) => ({ ...t, weightsPersistent })),
  }));
}

function messages(): string[] {
  return useTabStore.getState().tabs[0].logs.map((l: { message: string }) => l.message);
}

const SUCCESS = 'Execution completed successfully';

beforeEach(() => {
  vi.mocked(validateGraph).mockReset();
  vi.mocked(validateGraph).mockResolvedValue({ valid: true, errors: [] });
  vi.mocked(getRun).mockReset();
  vi.mocked(getRun).mockResolvedValue(null);
  useToastStore.setState({ toasts: [] });
  useI18n.setState({ locale: 'en' });
});

afterEach(() => {
  vi.restoreAllMocks();
  discardTabNodeUpdates();
});

describe('the run note while weights are kept between runs', () => {
  it('follows the success line when the switch is on', () => {
    const ws = setupTab(true);
    renderHook(() => useGraphExecution());
    act(() => ws.emit('execution_complete'));

    const note = useI18n.getState().t('settings.persist.runNote');
    expect(messages()).toEqual([SUCCESS, note]);
    expect(useTabStore.getState().tabs[0].logs[1].type).toBe('info');
    // The words the user reads: a key that is missing renders as the key.
    expect(note).not.toBe('settings.persist.runNote');
    expect(note).toContain('Export Python');
  });

  it('is not added when the switch is off', () => {
    const ws = setupTab(false);
    renderHook(() => useGraphExecution());
    act(() => ws.emit('execution_complete'));
    expect(messages()).toEqual([SUCCESS]);
  });

  it('is not added to a failed run', () => {
    const ws = setupTab(true);
    renderHook(() => useGraphExecution());
    act(() => ws.emit('execution_error', { error: 'kaboom' }));
    expect(messages()).toEqual(['Execution error: kaboom']);
  });

  it('reads in the language the UI is in', () => {
    useI18n.setState({ locale: 'zh-TW' });
    const ws = setupTab(true);
    renderHook(() => useGraphExecution());
    act(() => ws.emit('execution_complete'));
    expect(messages()[1]).toContain('保留權重');
  });
});

describe('the note follows the switch as the run was sent', () => {
  it('still appears when the switch is turned off while the run is going', async () => {
    const ws = setupRunnableTab(true);
    const { result } = renderHook(() => useGraphExecution());
    await act(async () => {
      await result.current.execute();
    });
    expect(ws.send).toHaveBeenCalledWith(
      expect.objectContaining({ action: 'execute', weights_persistent: true }),
    );

    act(() => setSwitch(false));
    act(() => ws.emit('execution_complete'));

    expect(messages()).toEqual([SUCCESS, useI18n.getState().t('settings.persist.runNote')]);
  });

  it('does not appear when the switch is turned on while the run is going', async () => {
    const ws = setupRunnableTab(false);
    const { result } = renderHook(() => useGraphExecution());
    await act(async () => {
      await result.current.execute();
    });
    expect(ws.send).toHaveBeenCalledWith(
      expect.objectContaining({ action: 'execute', weights_persistent: false }),
    );

    act(() => setSwitch(true));
    act(() => ws.emit('execution_complete'));

    expect(messages()).toEqual([SUCCESS]);
  });
});
