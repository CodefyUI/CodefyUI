/**
 * The run log's own status lines, in the language the UI is in (#598).
 *
 * The editor writes these lines itself ("Execution started", "Node X
 * completed"), so they are UI text and follow the language setting. The text
 * a node prints is different: it is the graph's own output, the same text the
 * exported script prints, so it stays exactly as written in either language.
 *
 * English is pinned word for word as well: the lines kept their wording when
 * they moved into the locale tables.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useGraphExecution } from './useGraphExecution';
import { useTabStore } from '../store/tabStore';
import { discardTabNodeUpdates } from '../store/nodeUpdateQueue';
import { useToastStore } from '../store/toastStore';
import { useI18n } from '../i18n';
import en from '../i18n/locales/en';
import zhTW from '../i18n/locales/zh-TW';

// The hook calls validateGraph() before sending and getRun() on mount; each
// test here only needs them to answer.
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

/** One idle tab with a node and an entry point, so Run actually sends. */
function setupTab(): FakeWs {
  const ws = makeFakeWs();
  useTabStore.setState({
    tabs: [
      {
        id: 't1',
        name: 't1',
        nodes: [{ id: 'n1', data: { label: 'Node One' } }],
        edges: [{ id: 'e1', source: 's', target: 'n1', data: { type: 'trigger' } }],
        selectedNodeId: null,
        presetModalNodeId: null,
        layersModalNodeId: null,
        undoStack: [],
        redoStack: [],
        dirtyNodeIds: new Set<string>(),
        status: 'idle',
        logs: [],
        ws,
        outputSummaries: {},
        recordOutputs: true,
        lastRunId: null,
        lastRunCursor: 0,
        activeSegment: null,
        segmentGroups: [],
        verboseMode: false,
        graphId: 'graph-t1',
        weightsPersistent: false,
        backwardMode: false,
        autoBackward: false,
      } as any,
    ],
    activeTabId: 't1',
  });
  return ws;
}

function messages(): string[] {
  return useTabStore.getState().tabs[0].logs.map((l: { message: string }) => l.message);
}

function lastLog(): { message: string; type: string } {
  const logs = useTabStore.getState().tabs[0].logs;
  return logs[logs.length - 1] as { message: string; type: string };
}

/** Run, with the socket closing while the graph is checked. */
async function runWithSocketLostDuringValidation(ws: FakeWs): Promise<void> {
  vi.mocked(validateGraph).mockImplementationOnce(async () => {
    ws.connected = false;
    ws.connect.mockRejectedValueOnce(new Error('server down'));
    return { valid: true, errors: [] };
  });
  const { result } = renderHook(() => useGraphExecution());
  await act(async () => {
    await result.current.execute();
  });
}

/** Run with a socket that cannot connect at all. */
async function runWithNoServer(ws: FakeWs): Promise<void> {
  ws.connected = false;
  ws.connect.mockRejectedValueOnce(new Error('no server'));
  const { result } = renderHook(() => useGraphExecution());
  await act(async () => {
    await result.current.execute();
  });
}

beforeEach(() => {
  vi.mocked(validateGraph).mockReset();
  vi.mocked(validateGraph).mockResolvedValue({ valid: true, errors: [] });
  vi.mocked(getRun).mockReset();
  vi.mocked(getRun).mockResolvedValue(null);
  useToastStore.setState({ toasts: [] });
});

afterEach(() => {
  vi.restoreAllMocks();
  discardTabNodeUpdates();
  useI18n.setState({ locale: 'en' });
});

describe('run log status lines in the Chinese UI', () => {
  beforeEach(() => {
    useI18n.setState({ locale: 'zh-TW' });
  });

  it('reads a whole run in Chinese', () => {
    const ws = setupTab();
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('execution_start', { run_id: 'run-1' });
      ws.emit('node_status', { node_id: 'n1', status: 'running' });
      ws.emit('node_status', { node_id: 'n1', status: 'completed' });
      ws.emit('execution_complete', { run_id: 'run-1' });
    });
    expect(messages()).toEqual(['開始執行', '節點 Node One 已完成', '執行完成']);
  });

  it('leaves the text a node prints exactly as written', () => {
    const ws = setupTab();
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('node_status', {
        node_id: 'n1',
        status: 'completed',
        outputs: [{ output_kind: 'text', text: '[p] Execution started 42' }],
      });
    });
    expect(messages()).toEqual(['節點 Node One 已完成', '[p] Execution started 42']);
  });

  it.each([
    ['skipped', '節點 Node One 已跳過'],
    ['interrupted', '節點 Node One 已中斷'],
  ])('says a node was %s in Chinese', (status, line) => {
    const ws = setupTab();
    renderHook(() => useGraphExecution());
    act(() => ws.emit('node_status', { node_id: 'n1', status }));
    expect(messages()).toEqual([line]);
    expect(lastLog().type).toBe('info');
  });

  it('names a failed node and keeps the error after it', () => {
    const ws = setupTab();
    renderHook(() => useGraphExecution());
    act(() => ws.emit('node_status', { node_id: 'n1', status: 'error', error: 'boom' }));
    expect(lastLog()).toMatchObject({ message: '節點 Node One 錯誤：boom', type: 'error' });
  });

  it('says a node failed when the frame carries no error text', () => {
    const ws = setupTab();
    renderHook(() => useGraphExecution());
    act(() => ws.emit('node_status', { node_id: 'n1', status: 'error' }));
    expect(lastLog()).toMatchObject({ message: '節點 Node One 發生錯誤', type: 'error' });
  });

  it('shows a status it has no words for as the raw token', () => {
    const ws = setupTab();
    renderHook(() => useGraphExecution());
    act(() => ws.emit('node_status', { node_id: 'n1', status: 'paused' }));
    expect(messages()).toEqual(['節點 Node One：paused']);
  });

  it('says a stopped run was cancelled', () => {
    const ws = setupTab();
    renderHook(() => useGraphExecution());
    act(() => ws.emit('execution_stopped', { run_id: 'run-1' }));
    expect(messages()).toEqual(['已取消執行']);
  });

  it('says a run failed, with the error', () => {
    const ws = setupTab();
    renderHook(() => useGraphExecution());
    act(() => ws.emit('execution_error', { error: 'kaboom' }));
    expect(lastLog()).toMatchObject({ message: '執行錯誤：kaboom', type: 'error' });
  });

  it('reports a refusal from the execution server', () => {
    const ws = setupTab();
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('error', { error: "run 'r-1' not found" });
      ws.emit('error', {});
    });
    expect(messages()).toEqual(["執行伺服器：run 'r-1' not found", '執行伺服器：未知錯誤']);
  });

  it('says the execution server cannot be reached, before validation and after it', async () => {
    const first = setupTab();
    await runWithNoServer(first);
    expect(lastLog()).toMatchObject({ message: '無法連線到執行伺服器', type: 'error' });

    const second = setupTab();
    await runWithSocketLostDuringValidation(second);
    expect(second.send).not.toHaveBeenCalled();
    expect(lastLog()).toMatchObject({ message: '無法連線到執行伺服器', type: 'error' });
  });
});

describe('run log status lines in the English UI, word for word', () => {
  beforeEach(() => {
    useI18n.setState({ locale: 'en' });
  });

  it('keeps every node line as it was', () => {
    const ws = setupTab();
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('node_status', { node_id: 'n1', status: 'completed' });
      ws.emit('node_status', { node_id: 'n1', status: 'skipped' });
      ws.emit('node_status', { node_id: 'n1', status: 'interrupted' });
      ws.emit('node_status', { node_id: 'n1', status: 'error', error: 'boom' });
      ws.emit('node_status', { node_id: 'n1', status: 'error' });
      ws.emit('node_status', { node_id: 'n1', status: 'paused' });
    });
    expect(messages()).toEqual([
      'Node Node One completed',
      'Node Node One skipped',
      'Node Node One interrupted',
      'Node Node One error: boom',
      'Node Node One error',
      'Node Node One paused',
    ]);
  });

  it('keeps every run line as it was', () => {
    const ws = setupTab();
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('execution_start', { run_id: 'run-1' });
      ws.emit('execution_complete', { run_id: 'run-1' });
      ws.emit('execution_stopped', { run_id: 'run-2' });
      ws.emit('execution_error', { error: 'kaboom' });
      ws.emit('error', { error: 'bad cursor' });
      ws.emit('error', {});
    });
    expect(messages()).toEqual([
      'Execution started',
      'Execution completed successfully',
      'Execution cancelled',
      'Execution error: kaboom',
      'Execution server: bad cursor',
      'Execution server: unknown error',
    ]);
  });

  it('keeps braces in a node title and in an error as written', () => {
    // t() fills slots one at a time; a title or an error that happens to
    // contain a slot name must not be filled in as one.
    const ws = setupTab();
    useTabStore.setState((s) => ({
      tabs: s.tabs.map((t) => ({
        ...t,
        nodes: [{ id: 'n1', data: { label: 'Box {detail} {status}' } } as any],
      })),
    }));
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('node_status', { node_id: 'n1', status: 'error', error: 'no key {label} {status}' });
      ws.emit('node_status', { node_id: 'n1', status: 'paused' });
    });
    expect(messages()).toEqual([
      'Node Box {detail} {status} error: no key {label} {status}',
      'Node Box {detail} {status} paused',
    ]);
  });

  it('has {label} ahead of {detail} in every translation, which that fill order needs', () => {
    let checked = 0;
    for (const dict of [en, zhTW] as Record<string, string>[]) {
      for (const [key, value] of Object.entries(dict)) {
        if (!key.startsWith('runLog.node.') || !value.includes('{detail}')) continue;
        checked += 1;
        expect(value.indexOf('{label}'), `${key}: ${value}`).toBeGreaterThanOrEqual(0);
        expect(value.indexOf('{label}'), `${key}: ${value}`).toBeLessThan(value.indexOf('{detail}'));
      }
    }
    expect(checked).toBe(2); // runLog.node.error, in each locale
  });
});
