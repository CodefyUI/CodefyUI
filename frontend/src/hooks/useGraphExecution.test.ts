import { StrictMode } from 'react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useGraphExecution } from './useGraphExecution';
import { useTabStore } from '../store/tabStore';
import {
  discardTabNodeUpdates,
  flushTabNodeUpdates,
} from '../store/nodeUpdateQueue';
import { useToastStore } from '../store/toastStore';
import { useUIStore } from '../store/uiStore';
import { usePackStore } from '../store/packStore';
import type { PackSummary } from '../api/rest';

// Mock the REST layer — the hook calls validateGraph() before sending, and
// getRun() on mount to decide whether to re-attach (#121). Each test drives
// their resolved/rejected values.
vi.mock('../api/rest', () => ({
  validateGraph: vi.fn(),
  getRun: vi.fn(),
}));
import { getRun, validateGraph } from '../api/rest';
const validateGraphMock = vi.mocked(validateGraph);
const getRunMock = vi.mocked(getRun);

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
    // Mirrors ExecutionWebSocket.dispatch: typed handlers, then '*'.
    emit: (type: string, data: unknown = {}) => {
      for (const h of handlers.get(type) ?? []) h(data);
      for (const h of handlers.get('*') ?? []) h({ type, ...(data as object) });
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

/**
 * What `require_pack` raises, word for word. The trailing `(pack=<id>)` is
 * the contract PR 1 promises the frontend; the rest is the sentence the
 * backend composes for a pack the catalog knows.
 */
const PACK_MISSING =
  'Word vectors is not installed. Open Package Center (toolbar > Settings > ' +
  'Optional packs) to install it; graph runs never download (pack=word-vectors)';

beforeEach(() => {
  validateGraphMock.mockReset();
  validateGraphMock.mockResolvedValue({ valid: true, errors: [] });
  getRunMock.mockReset();
  getRunMock.mockResolvedValue(null);
  useToastStore.setState({ toasts: [] });
  usePackStore.setState({ byId: {} });
  useUIStore.setState({ packCenterOpen: false, packCenterFocusPackId: null });
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
  // Node status/progress are frame-buffered since #125; anything a test left
  // queued must not leak into the next one's store.
  discardTabNodeUpdates();
});

function tabById(id: string): any {
  return useTabStore.getState().tabs.find((t) => t.id === id);
}

/**
 * Commit the frame buffer (#125). `node_status` no longer writes straight to
 * the store — it accumulates and lands once per animation frame — so a test
 * that asserts on node DATA has to run the frame first. Log assertions need
 * nothing: logs are still appended synchronously.
 */
function flushFrame() {
  act(() => flushTabNodeUpdates());
}

// ── WS listener attachment (the useEffect) ────────────────────────────────────

describe('useGraphExecution - WS listener lifecycle', () => {
  it('attaches every event handler to existing tabs on mount', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());

    const types = ws.on.mock.calls.map((c) => c[0]);
    expect(types).toEqual([
      'node_status',
      'execution_complete',
      'execution_error',
      'execution_start',
      'execution_stopped',
      // #121: the attach acknowledgement (which carries the run's real
      // status), protocol-level refusals, resuming after a dropped socket,
      // and the run / cursor every frame carries.
      'attached',
      'error',
      'reconnected',
      '*',
    ]);
  });

  it('detaches all handlers on unmount', () => {
    const ws = tabById('t1').ws as FakeWs;
    const { unmount } = renderHook(() => useGraphExecution());
    unmount();
    const offTypes = ws.off.mock.calls.map((c) => c[0]);
    expect(offTypes).toContain('node_status');
    expect(ws.off).toHaveBeenCalledTimes(9);
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
    expect((t2.ws as FakeWs).on).toHaveBeenCalledTimes(9);

    // Remove t1 → its handlers must be released (detachTab path).
    const ws1 = tabById('t1') ? (tabById('t1').ws as FakeWs) : null;
    const removedWs = ws1!;
    act(() => {
      useTabStore.setState((s) => ({ tabs: s.tabs.filter((t) => t.id !== 't1'), activeTabId: 't2' }));
    });
    expect(removedWs.off).toHaveBeenCalledTimes(9);
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
        outputs: [{ output_kind: 'progress', progress: { event: 'epoch', value: 1 } }],
      });
    });
    flushFrame();

    const tab = tabById('t1');
    expect(tab.nodes[0].data.progress).toEqual({ event: 'epoch', value: 1 });
    // epoch event → a structured progress log entry is appended (#117).
    const entry = tab.logs.find((l: any) => l.kind === 'progress');
    expect(entry.progress).toEqual({ event: 'epoch', value: 1 });
    // ...and never a magic-prefixed string.
    expect(tab.logs.some((l: any) => l.message.startsWith('__PROGRESS__:'))).toBe(false);
  });

  it('handles progress events WITHOUT epoch/config (no progress log)', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());

    act(() => {
      ws.emit('node_status', {
        node_id: 'n1',
        status: 'progress',
        outputs: [{ output_kind: 'progress', progress: { event: 'batch', value: 5 } }],
      });
    });
    flushFrame();

    const tab = tabById('t1');
    expect(tab.nodes[0].data.progress).toEqual({ event: 'batch', value: 5 });
    expect(tab.logs.some((l: any) => l.kind === 'progress')).toBe(false);
  });

  it('suppresses logs for running status but updates node status', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());

    act(() => {
      ws.emit('node_status', { node_id: 'n1', status: 'running' });
    });
    flushFrame();
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

  // core#260: the engine now settles a PRESET as 'cached' (every node inside
  // it was a cache hit) or 'skipped' (every node inside it was passed over),
  // where it used to say 'completed' regardless. Nothing on this path is
  // preset-aware — a node id is a node id — and that is the claim worth
  // pinning, because it is what lets PresetNode render the new statuses at
  // all. Asserted end to end, from a wire frame to the node's own data.
  it.each(['cached', 'skipped'] as const)(
    'lands a %s status on a preset node exactly as on an ordinary one',
    (status) => {
      setTabs([
        makeTab('t1', {
          nodes: [
            { id: 'box', type: 'presetNode', data: { label: 'Pipeline', isPreset: true } },
          ],
          edges: [],
        }),
      ]);
      const ws = tabById('t1').ws as FakeWs;
      renderHook(() => useGraphExecution());

      act(() => {
        ws.emit('node_status', { node_id: 'box', status: 'running' });
        ws.emit('node_status', { node_id: 'box', status });
      });
      flushFrame();

      expect(tabById('t1').nodes[0].data.executionStatus).toBe(status);
    },
  );

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

  // PR 2. The log line already explains the failure, but the fix lives in a
  // panel two menus away — so the toast carries the way there.
  it('fires an actionable toast when a node fails with a missing pack', () => {
    usePackStore.setState({
      byId: {
        'word-vectors': { id: 'word-vectors', title: 'Word vectors' } as PackSummary,
      },
    });
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('node_status', {
        node_id: 'n1',
        status: 'error',
        error: PACK_MISSING,
        error_type: 'PackMissingError',
      });
    });

    const toasts = useToastStore.getState().toasts;
    expect(toasts).toHaveLength(1);
    expect(toasts[0].type).toBe('error');
    expect(toasts[0].message).toBe('This run needs the Word vectors pack.');
    expect(toasts[0].action?.label).toBe('Open Package Center');

    // ...and the log keeps its own, node-scoped sentence.
    const log = tabById('t1').logs.find((l: any) => l.type === 'error');
    expect(log.message).toContain('This node needs the Word vectors pack');

    // The action opens the Package Center ON the pack that is missing.
    act(() => toasts[0].action!.onClick());
    expect(useUIStore.getState().packCenterOpen).toBe(true);
    expect(useUIStore.getState().packCenterFocusPackId).toBe('word-vectors');
  });

  it('falls back to the pack id when the catalog has not loaded', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('node_status', {
        node_id: 'n1',
        status: 'error',
        error: PACK_MISSING,
        error_type: 'PackMissingError',
      });
    });
    expect(useToastStore.getState().toasts[0].message).toBe(
      'This run needs the word-vectors pack.',
    );
  });

  it('does not toast for an ordinary node failure', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => ws.emit('node_status', { node_id: 'n1', status: 'error', error: 'boom' }));
    expect(useToastStore.getState().toasts).toEqual([]);
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

  it('labels a background tab log from its own nodes, not the active tab (#163)', () => {
    // t1 (active, from beforeEach) and t2 (background) each have a node
    // sharing the id 'n1' but carrying a DIFFERENT label -- a wrong-tab
    // lookup would silently "succeed" by resolving t1's label instead of
    // t2's own, rather than failing loudly.
    const t2 = makeTab('t2', {
      nodes: [{ id: 'n1', data: { label: 'Background Node' } }],
      edges: [],
    });
    setTabs([tabById('t1'), t2], 't1');
    renderHook(() => useGraphExecution());

    act(() => {
      (t2.ws as FakeWs).emit('node_status', { node_id: 'n1', status: 'completed' });
    });

    const log = tabById('t2').logs.find((l: any) => l.message.includes('completed'));
    expect(log.message).toBe('Node Background Node completed');
  });

  it('routes structured text / image / tensor_summary outputs (#117)', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('node_status', {
        node_id: 'n1',
        status: 'completed',
        outputs: [
          { output_kind: 'text', text: 'hello log' },
          {
            output_kind: 'image',
            port: 'image',
            image: { format: 'png', encoding: 'base64', data: 'QUJD' },
          },
          { output_kind: 'tensor_summary', tensor_summary: { out: { shape: [1] } } },
        ],
      });
    });
    const tab = tabById('t1');
    const text = tab.logs.find((l: any) => l.message === 'hello log');
    expect(text.kind).toBe('text');
    const img = tab.logs.find((l: any) => l.kind === 'image');
    expect(img.image).toEqual({
      format: 'png',
      encoding: 'base64',
      data: 'QUJD',
      port: 'image',
    });
    // No magic prefix is ever produced any more.
    expect(tab.logs.some((l: any) => l.message.startsWith('__IMAGE__:'))).toBe(false);
    expect(tab.outputSummaries.n1).toEqual({ out: { shape: [1] } });
  });

  it('appends one log entry per text output, in backend order', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('node_status', {
        node_id: 'n1',
        status: 'completed',
        outputs: [
          { output_kind: 'text', text: 'first line' },
          { output_kind: 'text', text: 'second line' },
        ],
      });
    });
    const texts = tabById('t1').logs.filter((l: any) => l.kind === 'text');
    expect(texts.map((l: any) => l.message)).toEqual(['first line', 'second line']);
  });

  it('appends one log entry per image output when a node declares several', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('node_status', {
        node_id: 'n1',
        status: 'completed',
        outputs: [
          { output_kind: 'image', port: 'a', image: { format: 'png', encoding: 'base64', data: 'AA' } },
          { output_kind: 'image', port: 'b', image: { format: 'png', encoding: 'base64', data: 'BB' } },
        ],
      });
    });
    const imgs = tabById('t1').logs.filter((l: any) => l.kind === 'image');
    expect(imgs.map((l: any) => l.image.data)).toEqual(['AA', 'BB']);
  });

  it('never turns a long alphanumeric text output into an image (#117)', () => {
    // Headline regression: the pre-#117 backend sniffed this exact shape and
    // sent it as `image`, so the panel rendered a broken <img>.
    const longText = 'TokenIds0123456789abcdef'.repeat(21).slice(0, 500);
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('node_status', {
        node_id: 'n1',
        status: 'completed',
        outputs: [
          { output_kind: 'text', text: longText },
          { output_kind: 'tensor_summary', tensor_summary: { answer: { type: 'string' } } },
        ],
      });
    });
    const tab = tabById('t1');
    expect(tab.logs.some((l: any) => l.kind === 'image')).toBe(false);
    expect(tab.logs.find((l: any) => l.message === longText).kind).toBe('text');
  });

  it('skips malformed / unknown output entries without throwing', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('node_status', {
        node_id: 'n1',
        status: 'completed',
        outputs: [
          null,
          'nonsense',
          { output_kind: 'waveform', waveform: { hz: 440 } }, // a future kind
          { output_kind: 'chart', chart: { series: [] } }, // chart with no kind
          { output_kind: 'image' }, // no payload
          { output_kind: 'text', text: '' }, // empty text
        ],
      });
    });
    const tab = tabById('t1');
    expect(tab.logs.some((l: any) => l.kind === 'image')).toBe(false);
    expect(tab.logs.some((l: any) => l.kind === 'text')).toBe(false);
    expect(tab.logs.some((l: any) => l.kind === 'chart')).toBe(false);
  });

  // ── Chart output entries (#130) ────────────────────────────────────────
  it('turns a chart output into a chart log entry carrying its spec', () => {
    const ws = tabById('t1').ws as FakeWs;
    const spec = {
      kind: 'heatmap',
      title: 'Confusion matrix',
      matrix: [[1, 0], [0, 1]],
      row_labels: ['a', 'b'],
      col_labels: ['a', 'b'],
      vmin: 0,
      vmax: 1,
    };
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('node_status', {
        node_id: 'n1',
        status: 'completed',
        outputs: [{ output_kind: 'chart', port: 'chart', chart: spec }],
      });
    });
    const entry = tabById('t1').logs.find((l: any) => l.kind === 'chart');
    expect(entry.nodeId).toBe('n1');
    expect(entry.chart).toEqual({ ...spec, port: 'chart' });
  });

  it('appends one entry per chart when a node emits several', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('node_status', {
        node_id: 'n1',
        status: 'completed',
        outputs: [
          { output_kind: 'chart', port: 'a', chart: { kind: 'bar', bars: [] } },
          { output_kind: 'chart', port: 'b', chart: { kind: 'line', series: [] } },
        ],
      });
    });
    const charts = tabById('t1').logs.filter((l: any) => l.kind === 'chart');
    expect(charts.map((l: any) => l.chart.kind)).toEqual(['bar', 'line']);
  });

  it('keeps a chart alongside the text a node emits with it', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('node_status', {
        node_id: 'n1',
        status: 'completed',
        outputs: [
          { output_kind: 'text', text: 'rendered table' },
          { output_kind: 'chart', chart: { kind: 'bar', bars: [] } },
        ],
      });
    });
    const kinds = tabById('t1').logs.map((l: any) => l.kind).filter(Boolean);
    expect(kinds).toEqual(['text', 'chart']);
  });

  // ── Deprecated flat fields (remove one release after #117) ──────────────
  // A frontend built from source can talk to an older prebuilt backend that
  // still sends `log` / `image` / `progress` / `output_summary` at the top
  // level (and guessed the image). Keep reading them for one release.
  it('still reads the legacy flat log / image / output_summary fields', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('node_status', {
        node_id: 'n1',
        status: 'completed',
        log: 'hello log',
        image: 'QUJD',
        output_summary: { out: { shape: [1] } },
      });
    });
    const tab = tabById('t1');
    expect(tab.logs.find((l: any) => l.message === 'hello log').kind).toBe('text');
    expect(tab.logs.find((l: any) => l.kind === 'image').image).toEqual({
      format: 'png',
      encoding: 'base64',
      data: 'QUJD',
    });
    expect(tab.outputSummaries.n1).toEqual({ out: { shape: [1] } });
  });

  it('still reads the legacy flat progress field', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('node_status', {
        node_id: 'n1',
        status: 'progress',
        progress: { event: 'config', config: { lr: 0.1 } },
      });
    });
    flushFrame();
    const tab = tabById('t1');
    expect(tab.nodes[0].data.progress).toEqual({ event: 'config', config: { lr: 0.1 } });
    expect(tab.logs.find((l: any) => l.kind === 'progress').progress).toEqual({
      event: 'config',
      config: { lr: 0.1 },
    });
  });

  // #125 acceptance: a burst of frames must cost ONE nodes-array rebuild.
  // Before the frame buffer, each `node_status` rebuilt the array on its own,
  // and React Flow re-diffed every node in the graph once per event.
  it('rebuilds the nodes array once for a burst of node_status frames', () => {
    setTabs([
      makeTab('t1', {
        nodes: [
          { id: 'n1', data: { label: 'One' } },
          { id: 'n2', data: { label: 'Two' } },
        ],
        edges: [],
      }),
    ]);
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());

    let rebuilds = 0;
    let previous = tabById('t1').nodes;
    const unsubscribe = useTabStore.subscribe((state) => {
      const nodes = state.tabs.find((t) => t.id === 't1')?.nodes;
      if (nodes && nodes !== previous) {
        previous = nodes;
        rebuilds += 1;
      }
    });

    act(() => {
      for (let i = 0; i < 25; i += 1) {
        ws.emit('node_status', {
          node_id: i % 2 === 0 ? 'n1' : 'n2',
          status: 'progress',
          outputs: [{ output_kind: 'progress', progress: { event: 'batch', value: i } }],
        });
      }
    });
    expect(rebuilds).toBe(0);

    flushFrame();
    expect(rebuilds).toBe(1);
    unsubscribe();

    const tab = tabById('t1');
    expect(tab.nodes[0].data.progress).toEqual({ event: 'batch', value: 24 });
    expect(tab.nodes[1].data.progress).toEqual({ event: 'batch', value: 23 });
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

  // #123: a REFUSED submit is not a run outcome. The server turned a click
  // down without starting anything, and the run this tab is already
  // following is still executing — so falling through to `error` would
  // re-enable Run and disable Stop mid-run.
  it('a rejected execution_error keeps the tab running and only logs a notice', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => ws.emit('execution_start', { run_id: 'run-1' }));
    expect(tabById('t1').status).toBe('running');

    act(() =>
      ws.emit('execution_error', {
        error: 'this editor session already has a run in flight',
        rejected: true,
        run_id: 'run-1',
      }),
    );

    const tab = tabById('t1');
    // Still running: Run stays disabled and Stop stays enabled, both of
    // which are derived from this status.
    expect(tab.status).toBe('running');
    expect(tab.lastRunId).toBe('run-1');
    const notice = tab.logs[tab.logs.length - 1];
    expect(notice.type).toBe('info');
    expect(notice.message).toContain('was not started');
    expect(notice.message).toContain('already has a run in flight');
    expect(tab.logs.some((l: any) => l.type === 'error')).toBe(false);
  });

  it('a rejected execution_error warns the user with a toast', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => ws.emit('execution_error', { error: 'busy', rejected: true }));
    const toasts = useToastStore.getState().toasts;
    expect(toasts.length).toBe(1);
    expect(toasts[0].type).toBe('warning');
    expect(toasts[0].message).toContain('was not started');
  });

  // A fail-fast run re-raises the node's exception, so the whole-run frame
  // carries the same `str(exc)` — with no `error_type` beside it.
  it('mirrors the missing-pack toast on a whole-run execution_error', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => ws.emit('execution_error', { error: PACK_MISSING }));
    const toasts = useToastStore.getState().toasts;
    expect(toasts).toHaveLength(1);
    expect(toasts[0].message).toBe('This run needs the word-vectors pack.');
    expect(toasts[0].action?.label).toBe('Open Package Center');
    // Still an ordinary run failure in every other respect.
    expect(tabById('t1').status).toBe('error');
  });

  it('does not stack a second toast when the run echoes the node that failed', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => {
      ws.emit('node_status', {
        node_id: 'n1',
        status: 'error',
        error: PACK_MISSING,
        error_type: 'PackMissingError',
      });
      ws.emit('execution_error', { error: PACK_MISSING });
    });
    expect(useToastStore.getState().toasts).toHaveLength(1);
  });

  it('a rejected execution_error never toasts about a pack', () => {
    // A refused submit did not run anything; its message is the server's,
    // not a node's, so the pack rule must not fire on it.
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    act(() => ws.emit('execution_error', { error: PACK_MISSING, rejected: true }));
    const toasts = useToastStore.getState().toasts;
    expect(toasts).toHaveLength(1);
    expect(toasts[0].type).toBe('warning');
    expect(toasts[0].action).toBeUndefined();
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

  // ── reproducibility options (core#134) ────────────────────────────────

  it('omits seed and deterministic when the tab has not set them', async () => {
    setTabs([
      makeTab('t1', {
        nodes: [{ id: 'n1', data: { label: 'N' } }],
        edges: [{ id: 'e1', source: 's', target: 'n1', data: { type: 'trigger' } }],
      }),
    ]);
    const ws = tabById('t1').ws as FakeWs;
    const { result } = renderHook(() => useGraphExecution());

    await act(async () => {
      await result.current.execute();
    });

    const message = ws.send.mock.calls[0][0];
    // Absent, not `null`: the message stays byte-identical to the pre-#134
    // one for everyone who never touches the field.
    expect('seed' in message).toBe(false);
    expect('deterministic' in message).toBe(false);
  });

  it('sends the tab seed and deterministic flag when they are set', async () => {
    setTabs([
      makeTab('t1', {
        nodes: [{ id: 'n1', data: { label: 'N' } }],
        edges: [{ id: 'e1', source: 's', target: 'n1', data: { type: 'trigger' } }],
        seed: 1234,
        deterministic: true,
      }),
    ]);
    const ws = tabById('t1').ws as FakeWs;
    const { result } = renderHook(() => useGraphExecution());

    await act(async () => {
      await result.current.execute();
    });

    const message = ws.send.mock.calls[0][0];
    expect(message.seed).toBe(1234);
    expect(message.deterministic).toBe(true);
  });

  it('sends seed 0 rather than dropping it as falsy', async () => {
    setTabs([
      makeTab('t1', {
        nodes: [{ id: 'n1', data: { label: 'N' } }],
        edges: [{ id: 'e1', source: 's', target: 'n1', data: { type: 'trigger' } }],
        seed: 0,
      }),
    ]);
    const ws = tabById('t1').ws as FakeWs;
    const { result } = renderHook(() => useGraphExecution());

    await act(async () => {
      await result.current.execute();
    });

    expect(ws.send.mock.calls[0][0].seed).toBe(0);
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
    expect(validateGraphMock).toHaveBeenCalledWith(
      serializedNodes, [], presets, undefined,
    );
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

  it('mints no run id of its own — the server owns it (#121)', async () => {
    // Before #121 the client generated a UUID and the backend echoed it back.
    // Now RunService creates the row, so its id is THE id: sending a second
    // one would be a second identity for the same run.
    const ws = tabById('t1').ws as FakeWs;
    const { result } = renderHook(() => useGraphExecution());

    await act(async () => {
      await result.current.execute();
    });
    expect('run_id' in ws.send.mock.calls[0][0]).toBe(false);

    // ...and the server's id is what the tab ends up holding.
    act(() => ws.emit('execution_start', { run_id: 'server-run-1', cursor: 1 }));
    expect(tabById('t1').lastRunId).toBe('server-run-1');
  });
});

// ── stop() ────────────────────────────────────────────────────────────────────

describe('useGraphExecution - stop', () => {
  it('sends an explicit cancel naming the run (#121)', () => {
    setTabs([
      makeTab('t1', {
        nodes: [{ id: 'n1', data: { label: 'N' } }],
        edges: [{ id: 'e1', source: 's', target: 'n1', data: { type: 'trigger' } }],
        lastRunId: 'run-7',
      }),
    ]);
    const ws = tabById('t1').ws as FakeWs;
    const { result } = renderHook(() => useGraphExecution());
    act(() => {
      result.current.stop();
    });
    // Cancel is the ONLY thing that stops a run now — detaching and closing
    // the socket deliberately do not — so it has to name the run.
    expect(ws.send).toHaveBeenCalledWith({ action: 'cancel', run_id: 'run-7' });
  });

  it('omits run_id when the tab has never seen one', () => {
    const ws = tabById('t1').ws as FakeWs;
    const { result } = renderHook(() => useGraphExecution());
    act(() => {
      result.current.stop();
    });
    expect(ws.send).toHaveBeenCalledWith({ action: 'cancel' });
  });

  it('never names the PREVIOUS run when Stop beats the new run id', async () => {
    // Stop is enabled the instant execute() sets `running`, but the new
    // run's id only arrives with `attached`. Naming the old run here would
    // cancel nothing — or, if retention had already pruned it, get back an
    // `execution_stopped` that unsticks the UI while the new run trains on.
    setTabs([
      makeTab('t1', {
        nodes: [{ id: 'n1', data: { label: 'N' } }],
        edges: [{ id: 'e1', source: 's', target: 'n1', data: { type: 'trigger' } }],
        lastRunId: 'previous-run',
      }),
    ]);
    const ws = tabById('t1').ws as FakeWs;
    const { result } = renderHook(() => useGraphExecution());

    await act(async () => {
      await result.current.execute();
    });
    act(() => {
      result.current.stop(); // before any `attached` frame
    });

    expect(ws.send).toHaveBeenLastCalledWith({ action: 'cancel' });
  });
});

// ── Re-attach (#121) ──────────────────────────────────────────────────────────
// The browser half of "close the tab, the run survives". On mount the hook
// asks the server whether the run this tab was watching is still going and,
// if so, attaches to replay its history and follow it live.
//
// NOTE: the hook remembers which (tab, run) pairs it has already resumed for
// the lifetime of the page, so every test here uses a DISTINCT run id (the
// StrictMode test deliberately reuses one).

describe('useGraphExecution - re-attach on mount', () => {
  function tabWatching(runId: string, overrides: Partial<any> = {}) {
    setTabs([
      makeTab('t1', {
        nodes: [{ id: 'n1', data: { label: 'N' } }],
        edges: [{ id: 'e1', source: 's', target: 'n1', data: { type: 'trigger' } }],
        lastRunId: runId,
        ...overrides,
      }),
    ]);
    return tabById('t1').ws as FakeWs;
  }

  it('attaches from cursor 0 when the run is still running', async () => {
    getRunMock.mockResolvedValue({ id: 'r-run', status: 'running' } as any);
    const ws = tabWatching('r-run');

    renderHook(() => useGraphExecution());

    await waitFor(() =>
      expect(ws.send).toHaveBeenCalledWith({
        action: 'attach',
        run_id: 'r-run',
        cursor: 0,
      }),
    );
    expect(useToastStore.getState().toasts.length).toBe(1);
    // The tab is NOT optimistically `running` — an attach the server refuses
    // must not disable Run forever.
    expect(tabById('t1').status).toBe('idle');

    // The server's acknowledgement is what restores it, which is also what
    // re-disables the Run button so a reload mid-training cannot start a
    // second run against the same persistent weights.
    act(() => ws.emit('attached', { run_id: 'r-run', cursor: 0, status: 'running' }));
    expect(tabById('t1').status).toBe('running');
  });

  it('stays idle when the server refuses the attach', async () => {
    getRunMock.mockResolvedValue({ id: 'r-refused', status: 'running' } as any);
    const ws = tabWatching('r-refused');

    renderHook(() => useGraphExecution());
    await waitFor(() => expect(ws.send).toHaveBeenCalled());

    // The run was pruned between the status check and the attach.
    act(() => ws.emit('error', { error: "run 'r-refused' not found" }));

    expect(tabById('t1').status).toBe('idle');
    const log = tabById('t1').logs.find((l: any) => l.type === 'error');
    expect(log.message).toContain('not found');
  });

  it('attaches to a queued run too', async () => {
    getRunMock.mockResolvedValue({ id: 'r-queued', status: 'queued' } as any);
    const ws = tabWatching('r-queued');

    renderHook(() => useGraphExecution());

    await waitFor(() => expect(ws.send).toHaveBeenCalled());
    expect(ws.send.mock.calls[0][0].action).toBe('attach');
  });

  it('does not attach to a run that already finished', async () => {
    getRunMock.mockResolvedValue({ id: 'r-done', status: 'succeeded' } as any);
    const ws = tabWatching('r-done');

    renderHook(() => useGraphExecution());

    await waitFor(() => expect(getRunMock).toHaveBeenCalledWith('r-done'));
    expect(ws.send).not.toHaveBeenCalled();
    expect(tabById('t1').status).toBe('idle');
    // ...and the stale handle is dropped, so the Inspector shows "not run
    // yet" rather than an empty view of a run nothing on screen describes.
    await waitFor(() => expect(tabById('t1').lastRunId).toBeNull());
  });

  it('does not attach to a run the server has never heard of', async () => {
    // Retention pruned it, or it belongs to a previous install.
    getRunMock.mockResolvedValue(null);
    const ws = tabWatching('r-gone');

    renderHook(() => useGraphExecution());

    await waitFor(() => expect(getRunMock).toHaveBeenCalledWith('r-gone'));
    expect(ws.send).not.toHaveBeenCalled();
    await waitFor(() => expect(tabById('t1').lastRunId).toBeNull());
  });

  it('stays quiet when the server is unreachable', async () => {
    getRunMock.mockRejectedValue(new Error('offline'));
    const ws = tabWatching('r-offline');

    renderHook(() => useGraphExecution());

    await waitFor(() => expect(getRunMock).toHaveBeenCalledWith('r-offline'));
    expect(ws.send).not.toHaveBeenCalled();
    expect(tabById('t1').status).toBe('idle');
  });

  it('opens the socket first when it is closed', async () => {
    getRunMock.mockResolvedValue({ id: 'r-closed', status: 'running' } as any);
    const ws = tabWatching('r-closed', { ws: makeFakeWs(false) });

    renderHook(() => useGraphExecution());

    await waitFor(() => expect(ws.send).toHaveBeenCalled());
    expect(ws.connect).toHaveBeenCalledTimes(1);
  });

  it('gives up quietly when the socket will not open', async () => {
    getRunMock.mockResolvedValue({ id: 'r-nows', status: 'running' } as any);
    const failing = makeFakeWs(false);
    failing.connect.mockRejectedValueOnce(new Error('no server'));
    const ws = tabWatching('r-nows', { ws: failing });

    renderHook(() => useGraphExecution());

    await waitFor(() => expect(ws.connect).toHaveBeenCalled());
    expect(ws.send).not.toHaveBeenCalled();
  });

  it('still attaches, exactly once, under StrictMode double-mounting', async () => {
    // THE regression this guards: StrictMode is the app's dev configuration
    // (main.tsx wraps <StrictMode>), and it runs the effect, its cleanup,
    // then the effect again — synchronously, before any await resolves. A
    // `cancelled` flag set by that cleanup made this fire ZERO times: pass 1
    // claimed the key, pass 2 saw it claimed and returned, pass 1 then bailed
    // on the flag. Everyone on `cdui dev` reloaded into an idle-looking tab
    // with an invisible, unstoppable live run.
    getRunMock.mockResolvedValue({ id: 'r-strict', status: 'running' } as any);
    const ws = tabWatching('r-strict');

    renderHook(() => useGraphExecution(), { wrapper: StrictMode });

    await waitFor(() => expect(ws.send).toHaveBeenCalledTimes(1));
    expect(ws.send.mock.calls[0][0]).toEqual({
      action: 'attach',
      run_id: 'r-strict',
      cursor: 0,
    });
  });

  it('does not attach again on a genuinely later mount', async () => {
    getRunMock.mockResolvedValue({ id: 'r-remount', status: 'running' } as any);
    const ws = tabWatching('r-remount');

    const first = renderHook(() => useGraphExecution());
    await waitFor(() => expect(ws.send).toHaveBeenCalledTimes(1));
    first.unmount();
    renderHook(() => useGraphExecution());
    await waitFor(() => expect(getRunMock).toHaveBeenCalled());

    expect(ws.send).toHaveBeenCalledTimes(1);
  });

  it('abandons the re-attach when the user starts a new run first', async () => {
    // The race: between the status check and the attach, the user clicks Run.
    // Their run is already submitted and attached; a late attach to the OLD
    // run detaches it server-side, so the new one trains invisibly and Stop
    // cancels the wrong one.
    let releaseGetRun: (value: any) => void = () => {};
    getRunMock.mockReturnValue(
      new Promise((resolve) => {
        releaseGetRun = resolve;
      }) as any,
    );
    const ws = tabWatching('r-raced');

    const { result } = renderHook(() => useGraphExecution());
    await waitFor(() => expect(getRunMock).toHaveBeenCalledWith('r-raced'));

    await act(async () => {
      await result.current.execute();
    });
    const sendsAfterRun = ws.send.mock.calls.length;
    expect(ws.send.mock.calls[sendsAfterRun - 1][0].action).toBe('execute');

    await act(async () => {
      releaseGetRun({ id: 'r-raced', status: 'running' });
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(ws.send).toHaveBeenCalledTimes(sendsAfterRun);
    expect(ws.send.mock.calls.some((c) => c[0].action === 'attach')).toBe(false);
  });

  it('does nothing for a tab that was never watching a run', async () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());
    await waitFor(() => expect(ws.on).toHaveBeenCalled());
    expect(getRunMock).not.toHaveBeenCalled();
    expect(ws.send).not.toHaveBeenCalled();
  });
});

describe('useGraphExecution - re-attach after a dropped socket', () => {
  it('resumes from the last cursor it rendered', () => {
    setTabs([
      makeTab('t1', {
        status: 'running',
        lastRunId: 'run-live',
        lastRunCursor: 42,
      }),
    ]);
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());

    act(() => ws.emit('reconnected'));

    // From the cursor, not from 0: the log panel still holds everything up
    // to 42, and replaying it would double every line.
    expect(ws.send).toHaveBeenCalledWith({
      action: 'attach',
      run_id: 'run-live',
      cursor: 42,
    });
  });

  it('ignores a reconnect when the tab is not running', () => {
    setTabs([makeTab('t1', { status: 'idle', lastRunId: 'run-old' })]);
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());

    act(() => ws.emit('reconnected'));

    expect(ws.send).not.toHaveBeenCalled();
  });

  it('ignores a reconnect when no run was ever watched', () => {
    setTabs([makeTab('t1', { status: 'running', lastRunId: null })]);
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());

    act(() => ws.emit('reconnected'));

    expect(ws.send).not.toHaveBeenCalled();
  });

  it('unsticks the tab when the server refuses the re-attach', async () => {
    // The reconnect path attaches with no preceding status check, so a
    // refusal (an older backend answering "Unknown action", a store hiccup)
    // would otherwise leave the tab on Running with nothing forwarding to
    // it and Run disabled forever.
    getRunMock.mockResolvedValue({ id: 'run-live', status: 'succeeded' } as any);
    setTabs([makeTab('t1', { status: 'running', lastRunId: 'run-live', lastRunCursor: 4 })]);
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());

    act(() => ws.emit('error', { error: 'Unknown action: attach' }));

    // The run's REAL status decides — not the error string, which may not
    // even be about the attach.
    await waitFor(() => expect(tabById('t1').status).toBe('completed'));
    expect(tabById('t1').logs.some((l: any) => l.type === 'error')).toBe(true);
  });

  it('leaves a healthy run alone when an unrelated error arrives', async () => {
    getRunMock.mockResolvedValue({ id: 'run-live', status: 'running' } as any);
    setTabs([makeTab('t1', { status: 'running', lastRunId: 'run-live' })]);
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());

    act(() => ws.emit('error', { error: 'Unknown action: foobar' }));

    await waitFor(() => expect(getRunMock).toHaveBeenCalledWith('run-live'));
    expect(tabById('t1').status).toBe('running');
  });

  it('leaves the tab alone when the server cannot be reached at all', async () => {
    getRunMock.mockRejectedValue(new Error('offline'));
    setTabs([makeTab('t1', { status: 'running', lastRunId: 'run-live' })]);
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());

    act(() => ws.emit('error', { error: 'something went wrong' }));

    await waitFor(() => expect(getRunMock).toHaveBeenCalled());
    // Not ours to declare over: the run may well still be training.
    expect(tabById('t1').status).toBe('running');
  });
});

describe('useGraphExecution - frame bookkeeping', () => {
  it('tracks the run id and cursor carried by every frame', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());

    act(() => ws.emit('node_status', { run_id: 'run-9', cursor: 3, node_id: 'n1', status: 'running' }));
    expect(tabById('t1').lastRunId).toBe('run-9');
    expect(tabById('t1').lastRunCursor).toBe(3);

    act(() => ws.emit('node_status', { run_id: 'run-9', cursor: 7, node_id: 'n1', status: 'completed' }));
    expect(tabById('t1').lastRunCursor).toBe(7);
  });

  it('never rewinds the cursor on a stale frame', () => {
    // A frame from a previous attachment can still be in the receive buffer
    // when a new one starts; it must not move the resume point backwards.
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());

    act(() => ws.emit('node_status', { run_id: 'run-9', cursor: 7, node_id: 'n1', status: 'completed' }));
    act(() => ws.emit('node_status', { run_id: 'run-9', cursor: 2, node_id: 'n1', status: 'running' }));
    expect(tabById('t1').lastRunCursor).toBe(7);
  });

  it('restarts the cursor when a different run takes over the tab', () => {
    const ws = tabById('t1').ws as FakeWs;
    renderHook(() => useGraphExecution());

    act(() => ws.emit('node_status', { run_id: 'run-a', cursor: 9, node_id: 'n1', status: 'completed' }));
    act(() => ws.emit('attached', { run_id: 'run-b', cursor: 0 }));
    expect(tabById('t1').lastRunId).toBe('run-b');
    expect(tabById('t1').lastRunCursor).toBe(0);
  });
});
