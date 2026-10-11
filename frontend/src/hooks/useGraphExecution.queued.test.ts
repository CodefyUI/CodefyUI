/**
 * #680: a run that is submitted but waiting to start says so.
 *
 * A canvas run waits on the process-wide seed lock while another run holds
 * it; a server-owned run waits in its device's FIFO. Either way its row says
 * `queued` until it begins, and the tab used to show Running with an empty
 * log. The tab now records why it waits (the toolbar reads that), the log
 * says it is queued and then that it started, and Stop still cancels it.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { QUEUE_CHECK_DELAY_MS, QUEUE_POLL_MS, useGraphExecution } from './useGraphExecution';
import { useTabStore } from '../store/tabStore';
import { discardTabNodeUpdates } from '../store/nodeUpdateQueue';
import { useToastStore } from '../store/toastStore';
import { useI18n } from '../i18n';

vi.mock('../api/rest', () => ({
  validateGraph: vi.fn(),
  getRun: vi.fn(),
}));
import { getRun, validateGraph, type RunInfo } from '../api/rest';
const getRunMock = vi.mocked(getRun);

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
  return {
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
    connect: vi.fn(async () => {}),
    emit: (type: string, data: unknown = {}) => {
      for (const h of handlers.get(type) ?? []) h(data);
      for (const h of handlers.get('*') ?? []) h({ type, ...(data as object) });
    },
  };
}

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

const tab = () => useTabStore.getState().tabs[0];
const messages = () => tab().logs.map((l) => l.message);

function row(over: Partial<RunInfo>): RunInfo {
  return {
    id: 'r1',
    status: 'queued',
    options: {},
    queue_key: 'cpu',
    queue_position: null,
    ...over,
  } as RunInfo;
}

/** Click Run, and have the server attach the socket to run `r1`. */
async function runAndAttach(ws: FakeWs, status: 'running' | 'queued') {
  const hook = renderHook(() => useGraphExecution());
  await act(async () => {
    await hook.result.current.execute();
  });
  act(() => ws.emit('attached', { run_id: 'r1', cursor: 0, status }));
  return hook;
}

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.mocked(validateGraph).mockReset();
  vi.mocked(validateGraph).mockResolvedValue({ valid: true, errors: [] });
  getRunMock.mockReset();
  getRunMock.mockResolvedValue(null);
  useToastStore.setState({ toasts: [] });
  useI18n.setState({ locale: 'en' });
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  discardTabNodeUpdates();
});

describe('a canvas run waiting on the seed lock', () => {
  it('shows queued with the reason, then logs that it started', async () => {
    const ws = setupTab();
    // The ack of a canvas run says `running` even while it waits.
    getRunMock.mockResolvedValue(row({ options: { lane: 'interactive', seed: 7 } }));
    await runAndAttach(ws, 'running');
    expect(tab().runQueue ?? null).toBeNull();

    await advance(QUEUE_CHECK_DELAY_MS);

    expect(getRunMock).toHaveBeenCalledWith('r1');
    expect(tab().status).toBe('running');
    expect(tab().runQueue).toEqual({ reason: 'seed' });
    expect(messages()).toEqual([
      'Queued: waiting for another run to finish. A run with a fixed seed never runs alongside another run.',
    ]);

    act(() => ws.emit('execution_start', { run_id: 'r1' }));

    expect(tab().runQueue).toBeNull();
    expect(tab().lastRunId).toBe('r1');
    expect(messages().slice(-1)[0]).toBe('Execution started');
    // Nothing asks again once the run has started.
    getRunMock.mockClear();
    await advance(QUEUE_POLL_MS * 2);
    expect(getRunMock).not.toHaveBeenCalled();
  });

  it('logs the reason in Traditional Chinese', async () => {
    useI18n.setState({ locale: 'zh-TW' });
    const ws = setupTab();
    getRunMock.mockResolvedValue(row({ options: { lane: 'interactive' } }));
    await runAndAttach(ws, 'running');
    await advance(QUEUE_CHECK_DELAY_MS);
    expect(messages()).toEqual([
      '排隊中：正在等待另一個執行完成。設定亂數種子的執行不會與其他執行同時進行。',
    ]);
  });

  it('says nothing about a queue when the run starts right away', async () => {
    const ws = setupTab();
    getRunMock.mockResolvedValue(row({ options: { lane: 'interactive' } }));
    await runAndAttach(ws, 'running');
    act(() => ws.emit('execution_start', { run_id: 'r1' }));

    await advance(QUEUE_CHECK_DELAY_MS * 2);

    expect(getRunMock).not.toHaveBeenCalled();
    expect(tab().runQueue).toBeNull();
    expect(messages()).toEqual(['Execution started']);
  });

  it('shows no queue when the row says the run is already running', async () => {
    const ws = setupTab();
    getRunMock.mockResolvedValue(row({ status: 'running', options: { lane: 'interactive' } }));
    await runAndAttach(ws, 'running');
    await advance(QUEUE_CHECK_DELAY_MS);
    expect(tab().runQueue ?? null).toBeNull();
    expect(messages()).toEqual([]);
  });
});

describe('a run waiting in its device queue', () => {
  it('shows its place in line, follows it down, and logs the wait once', async () => {
    const ws = setupTab();
    getRunMock.mockResolvedValue(row({ queue_key: 'cpu', queue_position: 2 }));
    await runAndAttach(ws, 'queued');
    await advance(0);

    expect(tab().runQueue).toEqual({ reason: 'device', device: 'cpu', position: 2 });
    expect(messages()).toEqual([
      'Queued: #2 in the cpu queue. The run starts when the runs ahead of it finish.',
    ]);

    getRunMock.mockResolvedValue(row({ queue_key: 'cpu', queue_position: 1 }));
    await advance(QUEUE_POLL_MS);

    expect(tab().runQueue).toEqual({ reason: 'device', device: 'cpu', position: 1 });
    expect(messages()).toHaveLength(1);

    act(() => ws.emit('execution_start', { run_id: 'r1' }));
    expect(tab().runQueue).toBeNull();
    expect(messages().slice(-1)[0]).toBe('Execution started');
  });

  it('names the device the run asked for when the server cannot place it', async () => {
    const ws = setupTab();
    getRunMock.mockResolvedValue(
      row({ queue_key: null, queue_position: null, options: { device: 'cuda:0' } }),
    );
    await runAndAttach(ws, 'queued');
    await advance(0);

    expect(tab().runQueue).toEqual({ reason: 'device', device: 'cuda:0', position: null });
    expect(messages()).toEqual([
      'Queued in the cuda:0 queue. The run starts when the runs ahead of it finish.',
    ]);
  });

  it('Stop still cancels it, and the queue goes with it', async () => {
    const ws = setupTab();
    getRunMock.mockResolvedValue(row({ queue_position: 1 }));
    const { result } = await runAndAttach(ws, 'queued');
    await advance(0);
    expect(tab().runQueue).not.toBeNull();

    act(() => result.current.stop());
    expect(ws.send).toHaveBeenLastCalledWith({ action: 'cancel', run_id: 'r1' });
    act(() => ws.emit('execution_stopped', { run_id: 'r1' }));

    expect(tab().status).toBe('idle');
    expect(tab().runQueue).toBeNull();
    getRunMock.mockClear();
    await advance(QUEUE_POLL_MS * 2);
    expect(getRunMock).not.toHaveBeenCalled();
  });

  it('stops asking when the server cannot be reached', async () => {
    const ws = setupTab();
    getRunMock.mockRejectedValue(new Error('offline'));
    await runAndAttach(ws, 'queued');
    await advance(0);
    expect(tab().runQueue ?? null).toBeNull();
    expect(getRunMock).toHaveBeenCalledTimes(1);
    await advance(QUEUE_POLL_MS * 2);
    expect(getRunMock).toHaveBeenCalledTimes(1);
  });

  it('ignores an answer for a run the tab is no longer attached to', async () => {
    const ws = setupTab();
    let answer: (run: RunInfo) => void = () => {};
    getRunMock.mockImplementationOnce(
      () => new Promise<RunInfo | null>((resolve) => { answer = resolve; }),
    );
    await runAndAttach(ws, 'queued');
    await advance(0);
    // The socket moves on to another run before the answer lands.
    getRunMock.mockResolvedValue(row({ id: 'r2', status: 'running' }));
    act(() => ws.emit('attached', { run_id: 'r2', cursor: 0, status: 'running' }));
    await act(async () => {
      answer(row({ queue_position: 3 }));
    });
    expect(tab().runQueue ?? null).toBeNull();
    expect(messages()).toEqual([]);
  });
});
