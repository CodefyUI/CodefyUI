/**
 * The edges of useGraphExecution's frame handling: a queue that does not
 * change between polls, a queue with no device to name, frames that carry
 * no run id, media entries with their optional fields left out, a frame a
 * tab hears after another listener of the same frame closed it, and the
 * REST check behind a refused attach or a Stop that found nothing to stop.
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

/** Dispatches over the handler list as it stood when the frame arrived, as the real socket does. */
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

/** Click Run: the tab goes to Running with no run id yet. */
async function clickRun() {
  const hook = renderHook(() => useGraphExecution());
  await act(async () => {
    await hook.result.current.execute();
  });
  return hook;
}

/** Click Run, and have the server attach the socket to run `r1`. */
async function runAndAttach(ws: FakeWs, status: 'running' | 'queued') {
  const hook = await clickRun();
  act(() => ws.emit('attached', { run_id: 'r1', cursor: 0, status }));
  return hook;
}

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

/** Let pending promise callbacks (a REST answer) run. */
async function settle() {
  await act(async () => {});
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

describe('queue polling', () => {
  it('leaves a seed-lock wait alone when the next poll says the same', async () => {
    const ws = setupTab();
    getRunMock.mockResolvedValue(row({ options: { lane: 'interactive' } }));
    await runAndAttach(ws, 'running');
    await advance(QUEUE_CHECK_DELAY_MS);
    const first = tab().runQueue;
    expect(first).toEqual({ reason: 'seed' });

    await advance(QUEUE_POLL_MS);

    expect(getRunMock).toHaveBeenCalledTimes(2);
    // Not rewritten: the same object the first poll stored.
    expect(tab().runQueue).toBe(first);
    expect(messages()).toHaveLength(1);
  });

  it('names no device when neither the server nor the run says one', async () => {
    const ws = setupTab();
    getRunMock.mockResolvedValue(
      row({ queue_key: null, queue_position: null, options: { device: 7 } as any }),
    );
    await runAndAttach(ws, 'queued');
    await advance(0);

    expect(tab().runQueue).toEqual({ reason: 'device', device: null, position: null });
    expect(messages()).toEqual([
      'Queued in the ? queue. The run starts when the runs ahead of it finish.',
    ]);
  });
});

describe('frames with no run id', () => {
  it('an interruption with no run id still ends the run and says why', async () => {
    const ws = setupTab();
    await clickRun();
    expect(tab().status).toBe('running');

    act(() => ws.emit('execution_stopped', { reason: 'interrupted' }));

    const why = 'The server stopped or restarted, so this run was interrupted.';
    expect(tab().status).toBe('idle');
    expect(useToastStore.getState().toasts.map((t) => t.message)).toEqual([why]);
    expect(messages()).toContain(why);
  });

  it('a finished ack with no run id ends the run on its status', async () => {
    const ws = setupTab();
    await clickRun();

    act(() => ws.emit('attached', { status: 'failed' }));

    expect(tab().status).toBe('error');
  });

  it('a re-attach to a run that has already started does not ask about a queue', async () => {
    const ws = setupTab();
    await runAndAttach(ws, 'running');
    act(() => ws.emit('execution_start', { run_id: 'r1' }));

    act(() => ws.emit('attached', { run_id: 'r1', cursor: 3, status: 'running' }));
    await advance(QUEUE_CHECK_DELAY_MS * 2);

    expect(tab().status).toBe('running');
    expect(getRunMock).not.toHaveBeenCalled();
    expect(tab().runQueue ?? null).toBeNull();
  });
});

describe('media entries', () => {
  it('fills in an image\'s format and encoding, and skips a clip with no url or format', () => {
    const ws = setupTab();
    renderHook(() => useGraphExecution());

    act(() => ws.emit('node_status', {
      node_id: 'n1',
      status: 'completed',
      outputs: [
        { output_kind: 'image', image: { data: 'AAAA' } },
        { output_kind: 'video', video: { format: 'mp4' } },
        { output_kind: 'video', video: { url: '/media/a.mp4' } },
        { output_kind: 'video', video: { url: '/media/b.mp4', format: 'mp4' } },
      ],
    }));

    const logs = tab().logs;
    expect(logs.map((l) => l.kind ?? null)).toEqual([null, 'image', 'video']);
    expect(logs[0].message).toBe('Node Node One completed');
    expect(logs[1].image).toEqual({ format: 'png', encoding: 'base64', data: 'AAAA' });
    expect(logs[2].video).toEqual({ url: '/media/b.mp4', format: 'mp4' });
  });
});

describe('a frame for a tab another listener just closed', () => {
  /** A listener ahead of the hook's that closes the tab on this frame. */
  function closeTabOn(ws: FakeWs, type: string) {
    const on = ws.on as unknown as (t: string, h: () => void) => void;
    on(type, () => useTabStore.setState({ tabs: [] }));
  }

  it('labels a node status by its id when the tab is gone', () => {
    const ws = setupTab();
    closeTabOn(ws, 'node_status');
    renderHook(() => useGraphExecution());
    const addTabLog = vi.spyOn(useTabStore.getState(), 'addTabLog');

    act(() => ws.emit('node_status', { node_id: 'abcdefghijkl', status: 'completed' }));

    expect(useTabStore.getState().tabs).toEqual([]);
    expect(addTabLog).toHaveBeenCalledWith('t1', expect.objectContaining({
      message: 'Node abcdefgh completed',
      type: 'success',
    }));
  });

  it('settles a Stop with no canvas left to settle', () => {
    const ws = setupTab();
    closeTabOn(ws, 'execution_stopped');
    renderHook(() => useGraphExecution());
    const addTabLog = vi.spyOn(useTabStore.getState(), 'addTabLog');

    expect(() => act(() => ws.emit('execution_stopped', { run_id: 'r1' }))).not.toThrow();

    expect(addTabLog).toHaveBeenCalledWith('t1', { message: 'Execution cancelled', type: 'info' });
  });
});

describe('asking REST whether the run is over', () => {
  it('a refused attach with no run id returns the tab to idle without asking', async () => {
    const ws = setupTab();
    await clickRun();
    expect(tab().lastRunId).toBeNull();

    act(() => ws.emit('error', { error: 'Unknown action' }));
    await settle();

    expect(getRunMock).not.toHaveBeenCalled();
    expect(tab().status).toBe('idle');
  });

  it('a Stop that found nothing asks about the tab\'s own run when the ack names none', async () => {
    const ws = setupTab();
    await runAndAttach(ws, 'running');
    getRunMock.mockResolvedValue(row({ status: 'succeeded' }));

    act(() => ws.emit('cancel_ack', { cancelled: false }));
    await settle();

    expect(getRunMock).toHaveBeenCalledWith('r1');
    expect(tab().status).toBe('completed');
  });

  it('a status this client does not know ends the run as idle', async () => {
    const ws = setupTab();
    await runAndAttach(ws, 'running');
    getRunMock.mockResolvedValue(row({ status: 'archived' as any }));

    act(() => ws.emit('cancel_ack', { run_id: 'r1', cancelled: false }));
    await settle();

    expect(tab().status).toBe('idle');
  });

  it('ignores the answer when the tab has moved on to another run meanwhile', async () => {
    const ws = setupTab();
    await runAndAttach(ws, 'running');
    let answer: (run: RunInfo) => void = () => {};
    getRunMock.mockImplementationOnce(
      () => new Promise<RunInfo | null>((resolve) => { answer = resolve; }),
    );

    act(() => ws.emit('cancel_ack', { run_id: 'r1', cancelled: false }));
    act(() => ws.emit('attached', { run_id: 'r2', cursor: 0, status: 'running' }));
    expect(tab().lastRunId).toBe('r2');
    await act(async () => {
      answer(row({ status: 'failed' }));
    });

    expect(tab().status).toBe('running');
    expect(tab().lastRunId).toBe('r2');
  });

  it('a nothing-to-stop ack for a tab that is not running asks nothing', async () => {
    const ws = setupTab();
    renderHook(() => useGraphExecution());

    act(() => ws.emit('cancel_ack', { run_id: 'r1', cancelled: false }));
    await settle();

    expect(getRunMock).not.toHaveBeenCalled();
    expect(tab().status).toBe('idle');
  });
});
