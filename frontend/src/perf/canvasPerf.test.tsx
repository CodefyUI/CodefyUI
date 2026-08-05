import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach, vi } from 'vitest';
import { render, cleanup, act } from '@testing-library/react';
import { ReactFlowProvider } from '@xyflow/react';
import type { NodeChange } from '@xyflow/react';

import { FlowCanvas } from '../components/Canvas/FlowCanvas';
import { useTabStore } from '../store/tabStore';
import { syntheticGraph } from './syntheticGraph';

/**
 * Canvas perf harness (#125).
 *
 * A dev-only measurement rig that mounts a synthetic 300-node / 400-edge
 * graph and times the JS work an interaction frame costs: the store commit
 * plus React's re-render and effects. That is the part of a frame this issue
 * changes; paint and layout need a real browser and are measured separately
 * in the PR's e2e pass.
 *
 * Two profiles:
 *
 *   pnpm test                     smoke -- a small graph, few frames, no
 *                                 timing assertions. Runs with every suite so
 *                                 the rig cannot rot, at a couple of seconds.
 *   CDUI_PERF=full pnpm test ...  the real thing -- 300 nodes / 400 edges,
 *                                 120 interaction frames. This is where the
 *                                 documented numbers come from.
 *   CDUI_PERF_ASSERT=1 ...        full, and the frame budget becomes a hard
 *                                 assertion. For a machine you trust; shared
 *                                 CI runners' timings mean nothing.
 *
 * These are wall-clock numbers off one machine and they move with what else
 * that machine is doing — the same drag scenario measured p95 26ms idle and
 * p95 41ms with a browser open on a 300-node graph. Quote a number only from
 * an otherwise-idle run, and read the before/after as a comparison of two
 * runs taken under the same conditions, not as an absolute.
 *
 * Deliberately written to run unchanged on a pre-#125 checkout, so the
 * before/after numbers come from the same rig: the only branch-dependent part
 * is `updateSink()`, which routes execution events through the frame queue
 * when the build has one and straight at the store when it does not.
 */

const HARD_ASSERT = process.env.CDUI_PERF_ASSERT === '1';
const FULL = HARD_ASSERT || process.env.CDUI_PERF === 'full';

const NODE_COUNT = FULL ? 300 : 60;
const EDGE_COUNT = FULL ? 400 : 80;
const DRAG_FRAMES = FULL ? 120 : 20;
const EXEC_FRAMES = FULL ? 100 : 20;
const UNDO_PUSHES = FULL ? 60 : 20;
const SAVE_TICKS = FULL ? 20 : 5;
const LANE_BUILDS = FULL ? 200 : 20;
/** Tabs a pre-#125 App would have had mounted at once. */
const TAB_FANOUT = 5;
/** Generous, because the full profile mounts five 300-node canvases. */
const SCENARIO_TIMEOUT_MS = 180_000;

/** Frame budget the acceptance criterion names: two 60Hz frames. */
const FRAME_BUDGET_MS = 32;

// ── Measurement ─────────────────────────────────────────────────────────────

interface Timings {
  label: string;
  samples: number;
  p50: number;
  p95: number;
  max: number;
  total: number;
}

const results: Timings[] = [];

function summarize(label: string, durations: number[]): Timings {
  const sorted = [...durations].sort((a, b) => a - b);
  const at = (q: number) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  const summary: Timings = {
    label,
    samples: sorted.length,
    p50: at(0.5),
    p95: at(0.95),
    max: sorted[sorted.length - 1],
    total: durations.reduce((a, b) => a + b, 0),
  };
  results.push(summary);
  return summary;
}

/** Run `body` `iterations` times, timing each run separately. */
function measure(label: string, iterations: number, body: (i: number) => void): Timings {
  const durations: number[] = [];
  for (let i = 0; i < iterations; i += 1) {
    const started = performance.now();
    body(i);
    durations.push(performance.now() - started);
  }
  return summarize(label, durations);
}

function pad(text: string, width: number): string {
  return text.length >= width ? text : text + ' '.repeat(width - text.length);
}

function padLeft(text: string, width: number): string {
  return text.length >= width ? text : ' '.repeat(width - text.length) + text;
}

function report(): string {
  const head = ['scenario', 'n', 'p50 ms', 'p95 ms', 'max ms', 'total ms'];
  const widths = [46, 5, 8, 8, 8, 9];
  const lines = [
    head.map((h, i) => (i === 0 ? pad(h, widths[i]) : padLeft(h, widths[i]))).join(' '),
    widths.map((w) => '-'.repeat(w)).join(' '),
  ];
  for (const r of results) {
    lines.push(
      [
        pad(r.label, widths[0]),
        padLeft(String(r.samples), widths[1]),
        padLeft(r.p50.toFixed(2), widths[2]),
        padLeft(r.p95.toFixed(2), widths[3]),
        padLeft(r.max.toFixed(2), widths[4]),
        padLeft(r.total.toFixed(1), widths[5]),
      ].join(' '),
    );
  }
  return lines.join('\n');
}

// ── Build-dependent plumbing ────────────────────────────────────────────────

interface UpdateSink {
  /** Whether execution updates are coalesced into one commit per frame. */
  batched: boolean;
  /** One `node_status` frame off the socket. */
  emit(tabId: string, nodeId: string, value: number): void;
  /** End of an animation frame: commit whatever the frame accumulated. */
  commit(): void;
}

/**
 * Route execution events the way the build under test routes them.
 *
 * Post-#125 the WS handler buffers into `nodeUpdateQueue` and commits once
 * per animation frame; before it, every event wrote straight through. The
 * module specifier is held in a variable so a pre-#125 checkout — where the
 * module does not exist — fails at runtime into the fallback rather than at
 * transform time.
 */
async function updateSink(): Promise<UpdateSink> {
  const specifier = '../store/nodeUpdateQueue';
  try {
    const queue = await import(/* @vite-ignore */ specifier);
    return {
      batched: true,
      emit: (tabId, nodeId, value) =>
        queue.queueTabNodeProgress(tabId, nodeId, { event: 'batch', value }),
      commit: () => act(() => queue.flushTabNodeUpdates()),
    };
  } catch {
    return {
      batched: false,
      emit: (tabId, nodeId, value) =>
        act(() =>
          useTabStore
            .getState()
            .setTabNodeProgress(tabId, nodeId, { event: 'batch', value }),
        ),
      commit: () => {},
    };
  }
}

// ── Fixture ─────────────────────────────────────────────────────────────────

const TAB_ID = 'perf-tab';
const graph = syntheticGraph(NODE_COUNT, EDGE_COUNT);
let originalTabs: ReturnType<typeof useTabStore.getState>['tabs'];
let originalActive: string;

/** Install `tabCount` tabs, each holding the synthetic graph. */
function seedTabs(tabCount: number) {
  const template = originalTabs[0];
  const tabs = Array.from({ length: tabCount }, (_, i) => ({
    ...template,
    id: i === 0 ? TAB_ID : `${TAB_ID}-${i}`,
    name: `Perf ${i}`,
    nodes: graph.nodes,
    edges: graph.edges,
    undoStack: [],
    redoStack: [],
    logs: [],
  }));
  useTabStore.setState({ tabs, activeTabId: TAB_ID });
}

function activeTab() {
  const s = useTabStore.getState();
  return s.tabs.find((t) => t.id === s.activeTabId)!;
}

/** Mount `count` independent canvases, as App did before #125 (one per tab). */
function mountCanvases(count: number) {
  return render(
    <>
      {Array.from({ length: count }, (_, i) => (
        <ReactFlowProvider key={i}>
          <FlowCanvas tabId={TAB_ID} />
        </ReactFlowProvider>
      ))}
    </>,
  );
}

/** One frame of a node drag: the change React Flow emits per pointermove. */
function dragFrame(i: number) {
  const target = graph.nodes[Math.floor(NODE_COUNT / 2)];
  const change: NodeChange = {
    id: target.id,
    type: 'position',
    position: { x: target.position.x + i, y: target.position.y + (i % 7) },
    dragging: true,
  };
  act(() => useTabStore.getState().onNodesChange([change]));
}

/** One debounce window's worth of autosave: dirty one tab, let it fire. */
function autosaveTick(i: number) {
  act(() => useTabStore.getState().renameTab(TAB_ID, `Perf edit ${i}`));
  vi.advanceTimersByTime(300);
}

/** Count replacements of the active tab's nodes array. */
function watchRebuilds() {
  let rebuilds = 0;
  let previous = activeTab().nodes;
  const stop = useTabStore.subscribe((state) => {
    const nodes = state.tabs.find((t) => t.id === TAB_ID)?.nodes;
    if (nodes && nodes !== previous) {
      previous = nodes;
      rebuilds += 1;
    }
  });
  return { count: () => rebuilds, stop };
}

beforeAll(() => {
  originalTabs = useTabStore.getState().tabs;
  originalActive = useTabStore.getState().activeTabId;
  // Fake ONLY the timer functions, so the 250ms autosave debounce cannot fire
  // in the middle of an interaction measurement (it gets its own scenario)
  // while `performance.now()` stays real and actually measures something.
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] });
});

afterAll(() => {
  vi.useRealTimers();
  useTabStore.setState({ tabs: originalTabs, activeTabId: originalActive });
  // eslint-disable-next-line no-console
  console.log(
    [
      '',
      `canvas perf harness [${FULL ? 'full' : 'smoke'}] -- ` +
        `${NODE_COUNT} nodes / ${EDGE_COUNT} edges`,
      `frame budget ${FRAME_BUDGET_MS}ms (p95), hard assert: ${HARD_ASSERT ? 'on' : 'off'}`,
      report(),
      '',
      'note: the (IndexedDB) autosave row names the ENVIRONMENT, not the',
      'backend. It measures whichever tier this build reaches for when',
      'IndexedDB is available -- one record per tab since #125, and the same',
      'whole-tree localStorage blob as the row above it on any earlier build.',
      '',
    ].join('\n'),
  );
});

beforeEach(() => {
  seedTabs(1);
});

afterEach(() => {
  cleanup();
});

// ── Scenarios ───────────────────────────────────────────────────────────────

describe('canvas perf harness', () => {
  it(
    'mounts the synthetic graph',
    () => {
      const started = performance.now();
      const view = mountCanvases(1);
      summarize('mount 1 canvas', [performance.now() - started]);
      expect(view.container.querySelectorAll('.react-flow__node').length).toBe(NODE_COUNT);
    },
    SCENARIO_TIMEOUT_MS,
  );

  it(
    'mounts one canvas per tab, as the pre-#125 App did',
    () => {
      // The old App rendered a TabContent (provider + canvas + palette +
      // panels) for EVERY tab and hid the inactive ones with display:none.
      // This is the canvas half of that cost for a five-tab workspace.
      const started = performance.now();
      const view = mountCanvases(TAB_FANOUT);
      summarize(`mount ${TAB_FANOUT} canvases (pre-#125 shape)`, [
        performance.now() - started,
      ]);
      expect(view.container.querySelectorAll('.react-flow__node').length).toBe(
        NODE_COUNT * TAB_FANOUT,
      );
    },
    SCENARIO_TIMEOUT_MS,
  );

  it(
    'drags a node with one canvas mounted',
    () => {
      mountCanvases(1);
      const summary = measure('drag frame, 1 canvas mounted', DRAG_FRAMES, dragFrame);
      if (HARD_ASSERT) expect(summary.p95).toBeLessThan(FRAME_BUDGET_MS);
      expect(summary.samples).toBe(DRAG_FRAMES);
    },
    SCENARIO_TIMEOUT_MS,
  );

  it(
    'drags a node with every tab\'s canvas mounted',
    () => {
      // Same interaction, but paying what the old App paid: one store commit
      // re-rendered every mounted tab's canvas, not just the visible one.
      mountCanvases(TAB_FANOUT);
      const summary = measure(
        `drag frame, ${TAB_FANOUT} canvases mounted`,
        DRAG_FRAMES,
        dragFrame,
      );
      expect(summary.samples).toBe(DRAG_FRAMES);
    },
    SCENARIO_TIMEOUT_MS,
  );

  it(
    'streams execution updates at 20 events/sec',
    async () => {
      mountCanvases(1);
      const sink = await updateSink();
      const watcher = watchRebuilds();
      // 20 events/sec against a 60Hz repaint is one event every third frame,
      // but events arrive in bursts (a batch loop emits several before the
      // socket yields), so each measured frame carries a small burst.
      const EVENTS_PER_FRAME = 3;
      const summary = measure('exec frame, 20 events/sec', EXEC_FRAMES, (frame) => {
        for (let e = 0; e < EVENTS_PER_FRAME; e += 1) {
          const node = graph.nodes[(frame * EVENTS_PER_FRAME + e) % NODE_COUNT];
          sink.emit(TAB_ID, node.id, frame);
        }
        sink.commit();
      });
      watcher.stop();

      const rebuilds = watcher.count();
      // eslint-disable-next-line no-console
      console.log(
        `  exec stream: ${rebuilds} nodes-array rebuilds for ` +
          `${EXEC_FRAMES * EVENTS_PER_FRAME} events over ${EXEC_FRAMES} frames ` +
          `(batched: ${sink.batched})`,
      );
      if (sink.batched) {
        // The acceptance criterion: at most one rebuild per frame, no matter
        // how many events landed inside it.
        expect(rebuilds).toBeLessThanOrEqual(EXEC_FRAMES);
      }
      if (HARD_ASSERT) expect(summary.p95).toBeLessThan(FRAME_BUDGET_MS);
    },
    SCENARIO_TIMEOUT_MS,
  );

  it(
    'assigns edge lanes for the whole graph',
    async () => {
      // Lanes are what keep two wires from being drawn on top of each other. The
      // map is rebuilt only when the wiring changes - never on a drag frame, which
      // is why no drag scenario above pays for it - so what matters here is that
      // the rebuild itself is cheap and scales with the edge count rather than
      // with its square. The signature row is the guard that decides whether a
      // rebuild is needed at all, so it runs on every edges-array change and has
      // to be cheaper still.
      //
      // Imported by specifier so this harness still runs against a checkout from
      // before lanes existed, the way its other build-dependent parts do.
      const specifier = '../utils/edgeLanes';
      let lanes: typeof import('../utils/edgeLanes');
      try {
        lanes = await import(/* @vite-ignore */ specifier);
      } catch {
        return;
      }
      const edges = graph.edges;
      const build = measure(`edge lanes, rebuild for ${EDGE_COUNT} edges`, LANE_BUILDS, () => {
        lanes.computeEdgeLanes(edges);
      });
      const sign = measure(`edge lanes, signature for ${EDGE_COUNT} edges`, LANE_BUILDS, () => {
        lanes.edgeLaneSignature(edges);
      });
      expect(lanes.computeEdgeLanes(edges).size).toBe(edges.length);
      expect(build.samples).toBe(LANE_BUILDS);
      expect(sign.samples).toBe(LANE_BUILDS);
      // A drag never touches the edge array, so a rebuild costs nothing per frame;
      // even so, one whole rebuild must stay far inside a single frame.
      if (HARD_ASSERT) expect(build.p95).toBeLessThan(FRAME_BUDGET_MS / 4);
    },
    SCENARIO_TIMEOUT_MS,
  );

  it(
    'pushes undo snapshots',
    () => {
      // Snapshots are taken on every drag start and every delete, so they sit
      // directly on the interaction path.
      const store = useTabStore.getState();
      const summary = measure('undo snapshot', UNDO_PUSHES, () => {
        store.pushUndoSnapshot();
      });
      expect(summary.samples).toBe(UNDO_PUSHES);
      expect(activeTab().undoStack.length).toBeGreaterThan(0);
    },
    SCENARIO_TIMEOUT_MS,
  );

  it(
    'runs the autosave debounce with every tab loaded',
    () => {
      // One dirty tab used to force all five to be walked and re-serialized.
      // jsdom has no IndexedDB, so this measures the localStorage tier --
      // the one backend a pre- and post-#125 checkout both use, which keeps
      // the comparison honest.
      seedTabs(TAB_FANOUT);
      const summary = measure(
        `autosave tick, ${TAB_FANOUT} tabs x ${NODE_COUNT} (localStorage)`,
        SAVE_TICKS,
        autosaveTick,
      );
      expect(summary.samples).toBe(SAVE_TICKS);
    },
    SCENARIO_TIMEOUT_MS,
  );

  it(
    'runs the autosave debounce against IndexedDB',
    async () => {
      // The tier a browser actually gets. What is measured is the SYNCHRONOUS
      // main-thread cost of one debounce tick, because that is what shows up
      // as jank: post-#125 that is building one changed record and handing it
      // to IndexedDB, with no whole-tree JSON.stringify anywhere. (The
      // structured clone of that one record happens off this tick.) On a
      // pre-#125 checkout there is no IndexedDB path, so this measures the
      // same blob write as the scenario above -- which is precisely the
      // before/after.
      const { IDBFactory, IDBKeyRange } = await import('fake-indexeddb');
      vi.stubGlobal('indexedDB', new IDBFactory());
      vi.stubGlobal('IDBKeyRange', IDBKeyRange);
      try {
        seedTabs(TAB_FANOUT);
        const summary = measure(
          `autosave tick, ${TAB_FANOUT} tabs x ${NODE_COUNT} (IndexedDB)`,
          SAVE_TICKS,
          autosaveTick,
        );
        expect(summary.samples).toBe(SAVE_TICKS);
      } finally {
        vi.unstubAllGlobals();
      }
    },
    SCENARIO_TIMEOUT_MS,
  );
});
