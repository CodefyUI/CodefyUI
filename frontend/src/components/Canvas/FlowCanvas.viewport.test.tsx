import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { StrictMode } from 'react';
import { render, act, cleanup } from '@testing-library/react';
import {
  ReactFlowProvider,
  getViewportForBounds,
  useReactFlow,
  useStoreApi,
  type Viewport,
} from '@xyflow/react';
import type { Node } from '@xyflow/react';

import { FlowCanvas } from './FlowCanvas';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import type { NodeData } from '../../types';
import { nodesBoundingBox } from '../../utils/autoLayout';
import {
  recallViewport,
  rememberViewport,
  _resetViewportMemory,
} from '../../utils/viewportMemory';

/**
 * Per-tab viewport handover (#125).
 *
 * Only the active tab's canvas is mounted now, and TabContent renders without
 * a `key`, so one `<ReactFlowProvider>` — and therefore ONE pan/zoom — serves
 * every tab. FlowCanvas is what makes each tab still feel like it kept its
 * own view: it stashes the outgoing tab's viewport and restores the incoming
 * one's.
 *
 * Mounts the REAL `<ReactFlow>` (the sibling FlowCanvas suite stubs it) so
 * `setViewport` / `getViewport` go through React Flow's actual pan-zoom
 * instance rather than a no-op.
 */

// EmptyCanvasOverlay fires a REST call on mount; every tab here has nodes so
// it never renders, but stub it for safety.
vi.mock('./EmptyCanvasOverlay', () => ({
  EmptyCanvasOverlay: () => <div data-testid="empty-overlay" />,
}));

const ORIGINAL_TABS = useTabStore.getState().tabs;
const ORIGINAL_ACTIVE = useTabStore.getState().activeTabId;

function node(id: string): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x: 0, y: 0 },
    data: { label: id, type: 'Test', params: {} },
  };
}

/** Three tabs: two with a node, one empty. */
function seedTabs() {
  const template = ORIGINAL_TABS[0];
  useTabStore.setState({
    tabs: [
      { ...template, id: 'tab-a', name: 'A', nodes: [node('a1')], edges: [] },
      { ...template, id: 'tab-b', name: 'B', nodes: [node('b1')], edges: [] },
      { ...template, id: 'tab-empty', name: 'Empty', nodes: [], edges: [] },
    ],
    activeTabId: 'tab-a',
  });
}

// Grab the flow instance from inside the provider so the test can drive and
// read the viewport exactly as the app does. The store is read only to check
// the size React Flow last measured.
let flow: ReturnType<typeof useReactFlow> | null = null;
let flowStore: ReturnType<typeof useStoreApi> | null = null;
function ViewportProbe() {
  flow = useReactFlow();
  flowStore = useStoreApi();
  return null;
}

/** The canvas as `App` mounts it: its `tabId` follows the tab on screen. */
function FollowingCanvas() {
  const activeTabId = useTabStore((s) => s.activeTabId);
  return <FlowCanvas tabId={activeTabId} />;
}

function mount(strict = false, following = false) {
  const tree = (
    <ReactFlowProvider>
      <ViewportProbe />
      {following ? <FollowingCanvas /> : <FlowCanvas tabId="tab-a" />}
    </ReactFlowProvider>
  );
  return render(strict ? <StrictMode>{tree}</StrictMode> : tree);
}

function setViewport(viewport: Viewport) {
  act(() => {
    void flow!.setViewport(viewport);
  });
}

function switchTo(tabId: string) {
  act(() => {
    useTabStore.getState().setActiveTab(tabId);
  });
}

/**
 * Give every element a non-zero box for the duration of a test.
 *
 * The fit path bails when the canvas container has no width (it cannot know
 * how much to inflate small bounds by), and jsdom reports 0 for everything.
 */
function withCanvasSize(width = 900, height = 600): () => void {
  const proto = HTMLElement.prototype;
  const original = {
    width: Object.getOwnPropertyDescriptor(proto, 'offsetWidth'),
    height: Object.getOwnPropertyDescriptor(proto, 'offsetHeight'),
  };
  Object.defineProperty(proto, 'offsetWidth', { configurable: true, get: () => width });
  Object.defineProperty(proto, 'offsetHeight', { configurable: true, get: () => height });
  return () => {
    if (original.width) Object.defineProperty(proto, 'offsetWidth', original.width);
    if (original.height) Object.defineProperty(proto, 'offsetHeight', original.height);
  };
}

/** Mount on a 600 px wide canvas, then widen it to 900 px without React Flow seeing it. */
function mountNarrowThenWiden(): () => void {
  const narrow = withCanvasSize(600, 600);
  try {
    mount();
  } finally {
    narrow();
  }
  return withCanvasSize(900, 600);
}

function expectViewportCloseTo(expected: Viewport) {
  const actual = flow!.getViewport();
  expect(actual.x).toBeCloseTo(expected.x, 3);
  expect(actual.y).toBeCloseTo(expected.y, 3);
  expect(actual.zoom).toBeCloseTo(expected.zoom, 3);
}

/** How FlowCanvas frames a box on the 900x600 canvas: inflated to 85% of it, then fitted. */
function framed(box: { x: number; y: number; width: number; height: number }): Viewport {
  let { x, y, width, height } = box;
  if (width < 765) {
    x -= (765 - width) / 2;
    width = 765;
  }
  if (height < 510) {
    y -= (510 - height) / 2;
    height = 510;
  }
  return getViewportForBounds({ x, y, width, height }, 900, 600, 0.1, 2, 0.2);
}

beforeEach(() => {
  _resetViewportMemory();
  seedTabs();
  useUIStore.setState({ layoutFitRequests: {} });
  flow = null;
  flowStore = null;
});

afterEach(() => {
  cleanup();
  _resetViewportMemory();
  useUIStore.setState({ layoutFitRequests: {} });
  useTabStore.setState({ tabs: ORIGINAL_TABS, activeTabId: ORIGINAL_ACTIVE });
});

describe('FlowCanvas per-tab viewport', () => {
  it('remembers where the outgoing tab was looking', () => {
    mount();
    setViewport({ x: 120, y: -40, zoom: 1.75 });
    switchTo('tab-b');
    expect(recallViewport('tab-a')).toEqual({ x: 120, y: -40, zoom: 1.75 });
  });

  it('restores a tab to the viewport it had when you left it', () => {
    mount();
    setViewport({ x: 120, y: -40, zoom: 1.75 });
    switchTo('tab-b');
    setViewport({ x: -10, y: 300, zoom: 0.5 });

    switchTo('tab-a');
    expect(flow!.getViewport()).toEqual({ x: 120, y: -40, zoom: 1.75 });

    switchTo('tab-b');
    expect(flow!.getViewport()).toEqual({ x: -10, y: 300, zoom: 0.5 });
  });

  it('gives a never-seen tab an overview fit, not the outgoing tab view', () => {
    // The fit is computed from the STORE's node positions, so it lands in the
    // same tick as the switch. Waiting for React Flow to measure the incoming
    // nodes instead would leave the outgoing tab's viewport on screen over the
    // incoming tab's graph until measurement completed.
    const restore = withCanvasSize();
    try {
      mount();
      setViewport({ x: 55, y: 66, zoom: 1.25 });
      switchTo('tab-b');
      expect(flow!.getViewport()).not.toEqual({ x: 55, y: 66, zoom: 1.25 });
    } finally {
      restore();
    }
  });

  it('starts a never-seen EMPTY tab at the default view, not the outgoing zoom (#622)', () => {
    // Nothing to frame, so it opens where a canvas mounted for it would. Left
    // at a zoomed-in tab's zoom, a new "+" tab showed the first nodes dropped
    // into it huge.
    const restore = withCanvasSize();
    try {
      mount();
      setViewport({ x: 55, y: 66, zoom: 1.9 });
      switchTo('tab-empty');
      expect(flow!.getViewport()).toEqual({ x: 0, y: 0, zoom: 1 });
    } finally {
      restore();
    }
  });

  it('stores each tab separately rather than overwriting one slot', () => {
    mount();
    setViewport({ x: 1, y: 2, zoom: 1 });
    switchTo('tab-b');
    setViewport({ x: 3, y: 4, zoom: 2 });
    switchTo('tab-empty');
    expect(recallViewport('tab-a')).toEqual({ x: 1, y: 2, zoom: 1 });
    expect(recallViewport('tab-b')).toEqual({ x: 3, y: 4, zoom: 2 });
  });

  it('re-remembers on every visit, not just the first', () => {
    mount();
    setViewport({ x: 1, y: 1, zoom: 1 });
    switchTo('tab-b');
    switchTo('tab-a');
    setViewport({ x: 9, y: 9, zoom: 3 });
    switchTo('tab-b');
    expect(recallViewport('tab-a')).toEqual({ x: 9, y: 9, zoom: 3 });
  });

  it('survives StrictMode double mounting without stashing a bogus viewport', () => {
    // StrictMode runs every effect twice. A handover keyed only on "the
    // effect ran" would record the ACTIVE tab's viewport against itself on
    // the second pass and then restore it over the tab being switched to.
    mount(true);
    setViewport({ x: 42, y: 42, zoom: 2.5 });
    expect(recallViewport('tab-a')).toBeUndefined();
    switchTo('tab-b');
    expect(recallViewport('tab-a')).toEqual({ x: 42, y: 42, zoom: 2.5 });
    expect(recallViewport('tab-b')).toBeUndefined();
  });

  it('forgets a closed tab so a recycled id does not inherit its view', () => {
    mount();
    setViewport({ x: 7, y: 7, zoom: 1.5 });
    switchTo('tab-b');
    expect(recallViewport('tab-a')).toBeTruthy();
    act(() => {
      useTabStore.getState().removeTab('tab-a');
    });
    expect(recallViewport('tab-a')).toBeUndefined();
  });

  it('leaves a remembered viewport alone when nothing switches', () => {
    rememberViewport('tab-b', { x: 999, y: 999, zoom: 4 });
    mount();
    setViewport({ x: 1, y: 1, zoom: 1 });
    expect(recallViewport('tab-b')).toEqual({ x: 999, y: 999, zoom: 4 });
  });

  // #563. A node selected in the outgoing tab holds the config panel open, so
  // React Flow measured a narrow canvas. The incoming tab has no selection and
  // the panel goes in the same commit as the switch, but React Flow's resize
  // observer reports the wider canvas only after the fit has run: framed with
  // React Flow's size, the graph came out small and pushed to the left.
  describe('framing for the size the canvas has now (#563)', () => {
    // Tab B's one node is a 200x80 box at the origin (the layout fallbacks),
    // inflated around its centre to 85% of a 900x600 canvas.
    const framedOn900x600 = () =>
      getViewportForBounds({ x: -282.5, y: -215, width: 765, height: 510 }, 900, 600, 0.1, 2, 0.2);

    it('frames a never-seen tab for the canvas as it is, not as React Flow last measured it', () => {
      const restore = mountNarrowThenWiden();
      try {
        expect(flowStore!.getState().width).toBe(600);
        switchTo('tab-b');
        expectViewportCloseTo(framedOn900x600());
      } finally {
        restore();
      }
    });

    it('frames a one-shot fit request the same way', () => {
      const restore = mountNarrowThenWiden();
      try {
        expect(flowStore!.getState().width).toBe(600);
        act(() => {
          useUIStore.getState().requestLayoutFit('tab-a', { x: 0, y: 0, width: 200, height: 80 });
        });
        expectViewportCloseTo(framedOn900x600());
        expect(useUIStore.getState().layoutFitRequests).toEqual({});
      } finally {
        restore();
      }
    });
  });

  // #522. A fit request names its tab and waits until that tab is on screen.
  // A plugin handler that lays out one tab and switches tabs in the same turn
  // renders the canvas once with both changes, and the request used to be
  // spent on whichever tab was on screen by then.
  describe('a fit asked for in the same turn as a tab switch (#522)', () => {
    // Looking at tab A's graph before the layout moves it to the origin.
    const LOOKING = { x: -4900, y: -4900, zoom: 1 };
    const LAID_OUT = [
      { ...node('a1'), position: { x: 0, y: 0 } },
      { ...node('a2'), position: { x: 280, y: 0 } },
    ];
    const BOX = { x: 0, y: 0, width: 480, height: 80 };
    // Tab B's first visit frames its one node, a 200x80 fallback box.
    const TAB_B = framed({ x: 20000, y: 20000, width: 200, height: 80 });

    let restoreSize: () => void;
    beforeEach(() => {
      restoreSize = withCanvasSize();
      useTabStore.setState((state) => ({
        tabs: state.tabs.map((tab) =>
          tab.id === 'tab-a'
            ? {
                ...tab,
                nodes: [
                  { ...node('a1'), position: { x: 5000, y: 5000 } },
                  { ...node('a2'), position: { x: 9000, y: 9000 } },
                ],
              }
            : tab.id === 'tab-b'
              ? { ...tab, nodes: [{ ...node('b1'), position: { x: 20000, y: 20000 } }] }
              : tab,
        ),
      }));
      mount(false, true);
      setViewport(LOOKING);
    });
    afterEach(() => {
      restoreSize();
    });

    /** Lay out tab A and ask for its fit, as `auto_layout` does for that tab. */
    function layOutTabA() {
      useTabStore.setState((state) => ({
        tabs: state.tabs.map((tab) => (tab.id === 'tab-a' ? { ...tab, nodes: LAID_OUT } : tab)),
      }));
      useUIStore.getState().requestLayoutFit('tab-a', nodesBoundingBox(LAID_OUT as Node[])!);
    }

    it('frames the tab switched to by its own graph, and the laid-out tab on its next visit', () => {
      act(() => {
        layOutTabA();
        useTabStore.getState().setActiveTab('tab-b');
      });
      expectViewportCloseTo(TAB_B);
      expect(useUIStore.getState().layoutFitRequests).toEqual({ 'tab-a': BOX });

      switchTo('tab-a');
      expectViewportCloseTo(framed(BOX));
      expect(useUIStore.getState().layoutFitRequests).toEqual({});
    });

    it('frames a tab laid out after it went to the background on its next visit, not by the view it had', () => {
      act(() => {
        useTabStore.getState().setActiveTab('tab-b');
        layOutTabA();
      });
      expectViewportCloseTo(TAB_B);
      expect(recallViewport('tab-a')).toEqual(LOOKING);

      switchTo('tab-a');
      expectViewportCloseTo(framed(BOX));
      expect(useUIStore.getState().layoutFitRequests).toEqual({});
    });

    it('fits the latest request for a tab', () => {
      act(() => {
        useTabStore.getState().setActiveTab('tab-b');
        useUIStore.getState().requestLayoutFit('tab-a', { x: 9000, y: 9000, width: 200, height: 80 });
        layOutTabA();
      });
      switchTo('tab-a');
      expectViewportCloseTo(framed(BOX));
    });

    it('drops the request of a tab that closes before its visit', () => {
      act(() => {
        useTabStore.getState().setActiveTab('tab-b');
        layOutTabA();
      });
      act(() => {
        useTabStore.getState().removeTab('tab-a');
      });
      expect(useUIStore.getState().layoutFitRequests).toEqual({});
    });
  });

  // #622. A fit is worked out from the store's nodes, where a node React Flow
  // has not measured yet counts at the layout fallback size (200x80). A graph
  // that was just installed has no sizes until React Flow reports them a frame
  // later, so a tall note counted as a card and ran off the bottom of the
  // canvas. The canvas frames the same nodes again once their sizes are in,
  // unless the view was moved in between.
  describe('framing again once React Flow has measured (#622)', () => {
    const CARD = { width: 200, height: 80 };
    // The note "Call a graph over HTTP" opens with.
    const TALL_NOTE = { width: 300, height: 873 };

    function at(id: string, x: number, y: number): Node<NodeData> {
      return { ...node(id), position: { x, y } };
    }

    /**
     * Report sizes the way React Flow does: `dimensions` changes through the
     * store's `onNodesChange`, which writes them to each node's `measured`.
     * jsdom delivers no ResizeObserver callbacks, so nothing else measures a
     * node here.
     */
    function reportSizes(sizes: Record<string, { width: number; height: number }>) {
      useTabStore.getState().onNodesChange(
        Object.entries(sizes).map(([id, dimensions]) => ({
          id,
          type: 'dimensions' as const,
          dimensions,
        })),
      );
    }

    function measure(sizes: Record<string, { width: number; height: number }>) {
      act(() => reportSizes(sizes));
    }

    /**
     * Mount on a canvas whose graph React Flow has already measured, as it
     * has after one frame on screen. Until React Flow has measured a set of
     * nodes once, it still holds the fit its `fitView` prop queued on mount,
     * and would spend it on the first sizes a test reports. That fit finishes
     * a few microtasks after it starts, hence the async act.
     */
    async function mountOnScreen() {
      mount();
      await act(async () => reportSizes({ a1: CARD }));
    }

    /** Install nodes in the tab on screen and ask for a fit, as Open and Import do. */
    function install(nodes: Node<NodeData>[], fitOver: Node<NodeData>[] = nodes) {
      act(() => {
        useTabStore.getState().setNodes(nodes);
        useUIStore
          .getState()
          .requestLayoutFit(useTabStore.getState().activeTabId, nodesBoundingBox(fitOver as Node[])!);
      });
    }

    let restoreSize: () => void;
    beforeEach(() => {
      restoreSize = withCanvasSize();
    });
    afterEach(() => {
      restoreSize();
    });

    it('frames a freshly opened graph again by its real sizes', async () => {
      await mountOnScreen();
      // A card and a tall note, framed at once by the fallback sizes.
      install([at('s1', 0, 0), at('note', 0, 120)]);
      const estimated = framed({ x: 0, y: 0, width: 200, height: 200 });
      expectViewportCloseTo(estimated);

      measure({ s1: CARD, note: TALL_NOTE });
      expectViewportCloseTo(framed({ x: 0, y: 0, width: 300, height: 993 }));
      expect(flow!.getViewport().zoom).toBeLessThan(estimated.zoom);
    });

    it('leaves the view where the user moved it before the sizes arrived', async () => {
      await mountOnScreen();
      install([at('s1', 0, 0), at('note', 0, 120)]);
      const estimated = flow!.getViewport();
      setViewport({ x: 10, y: 20, zoom: 1.3 });

      measure({ s1: CARD, note: TALL_NOTE });
      expect(flow!.getViewport()).toEqual({ x: 10, y: 20, zoom: 1.3 });

      // Decided once and for all: back on the first fit, a later size change
      // still frames nothing.
      setViewport(estimated);
      measure({ note: { width: 300, height: 1400 } });
      expect(flow!.getViewport()).toEqual(estimated);
    });

    it('frames again once, not on every later size change', async () => {
      await mountOnScreen();
      install([at('s1', 0, 0), at('note', 0, 120)]);
      measure({ s1: CARD, note: TALL_NOTE });
      const refitted = framed({ x: 0, y: 0, width: 300, height: 993 });
      expectViewportCloseTo(refitted);

      // The note grows while it is edited; the view stays put.
      measure({ note: { width: 300, height: 1400 } });
      expectViewportCloseTo(refitted);
    });

    it('frames a never-seen tab again once its sizes arrive', async () => {
      await mountOnScreen();
      switchTo('tab-b');
      expectViewportCloseTo(framed({ x: 0, y: 0, width: 200, height: 80 }));

      measure({ b1: { width: 320, height: 900 } });
      expectViewportCloseTo(framed({ x: 0, y: 0, width: 320, height: 900 }));
    });

    it('drops a re-fit when the tab changes, so the tab comes back as it was left', async () => {
      rememberViewport('tab-b', { x: 5, y: 5, zoom: 1 });
      await mountOnScreen();
      install([at('s1', 0, 0), at('note', 0, 120)]);
      const left = flow!.getViewport();
      switchTo('tab-b');
      switchTo('tab-a');
      expect(flow!.getViewport()).toEqual(left);

      measure({ s1: CARD, note: TALL_NOTE });
      expect(flow!.getViewport()).toEqual(left);
    });

    it('fits nodes that are already measured once, as auto layout does', async () => {
      await mountOnScreen();
      act(() => {
        const { nodes } = useTabStore.getState().getActiveTab();
        useUIStore.getState().requestLayoutFit('tab-a', nodesBoundingBox(nodes as Node[])!);
      });
      const fitted = framed({ x: 0, y: 0, width: 200, height: 80 });
      expectViewportCloseTo(fitted);

      // Nothing was waiting for a size, so a later one moves nothing.
      measure({ a1: { width: 200, height: 600 } });
      expectViewportCloseTo(fitted);
    });

    it('frames an inserted block again without the graph above it', async () => {
      await mountOnScreen();
      // A template inserted below the graph: the fit asks for the inserted
      // nodes alone, and only they are still unmeasured.
      const [a1] = useTabStore.getState().getActiveTab().nodes;
      const inserted = [at('t1', 0, 176), at('t2', 260, 176)];
      install([a1, ...inserted], inserted);
      expectViewportCloseTo(framed({ x: 0, y: 176, width: 460, height: 80 }));

      measure({ t1: CARD, t2: TALL_NOTE });
      expectViewportCloseTo(framed({ x: 0, y: 176, width: 560, height: 873 }));
    });

    it('does not zoom in on the first node dropped on a canvas that opened empty', async () => {
      // React Flow's own first fit waited for the first measured node, and
      // framed that one card at full zoom.
      useTabStore.setState({ activeTabId: 'tab-empty' });
      mount();
      act(() => useTabStore.getState().setNodes([at('first', 300, 200)]));
      await act(async () => reportSizes({ first: CARD }));
      expect(flow!.getViewport()).toEqual({ x: 0, y: 0, zoom: 1 });
    });

    it('frames a starter opened from the welcome screen as the gallery does', async () => {
      // The welcome screen installs a starter in a new tab and asks for a fit
      // before the canvas exists. React Flow's own first fit then ran on the
      // first sizes and replaced the overview with a close-up at zoom 2.
      const starter = [at('s1', 0, 0), at('s2', 260, 0)];
      useTabStore.getState().setNodes(starter);
      useUIStore.getState().requestLayoutFit('tab-a', nodesBoundingBox(starter as Node[])!);
      mount();
      expectViewportCloseTo(framed({ x: 0, y: 0, width: 460, height: 80 }));

      await act(async () => reportSizes({ s1: CARD, s2: { width: 240, height: 120 } }));
      expectViewportCloseTo(framed({ x: 0, y: 0, width: 500, height: 120 }));
    });

    // A block opens on the same canvas. Collapse stores its nodes relative to
    // the block's top-left corner, so inside, they sit near the origin, far
    // from the view the graph around the block had: a blank canvas.
    describe('entering and leaving a block', () => {
      const OUTER = { x: -2500, y: -1700, zoom: 0.9 };

      /** Collapse the selection on the level on screen; returns the instance. */
      function collapseAll(name: string): string {
        let instanceId = '';
        act(() => {
          const store = useTabStore.getState();
          store.setNodes(store.getActiveTab().nodes.map((n) => ({ ...n, selected: true })));
          const result = store.collapseSelectionToSubgraph(name);
          if (!result.ok) throw new Error(`collapse refused: ${result.reason}`);
          instanceId = result.instanceId;
        });
        return instanceId;
      }

      /**
       * Two cards collapsed into a block at (3000, 2000), with the view on it.
       * Inside, the cards are at (0, 0) and (300, 0).
       */
      async function blockOnScreen(): Promise<string> {
        await mountOnScreen();
        act(() => useTabStore.getState().setNodes([at('b1', 3000, 2000), at('b2', 3300, 2000)]));
        const instanceId = collapseAll('Block');
        setViewport(OUTER);
        return instanceId;
      }

      function enter(instanceId: string) {
        act(() => {
          useTabStore.getState().enterSubgraph(instanceId);
        });
      }

      function exit() {
        act(() => useTabStore.getState().exitSubgraph());
      }

      it('frames the inside of a block when it is entered, then by its measured sizes', async () => {
        enter(await blockOnScreen());
        expectViewportCloseTo(framed({ x: 0, y: 0, width: 500, height: 80 }));

        measure({ b1: CARD, b2: TALL_NOTE });
        expectViewportCloseTo(framed({ x: 0, y: 0, width: 600, height: 873 }));
      });

      it('puts back the view the graph around the block had when it is left', async () => {
        enter(await blockOnScreen());
        setViewport({ x: 40, y: 50, zoom: 1.2 });

        exit();
        expect(flow!.getViewport()).toEqual(OUTER);
      });

      it('goes back out one level at a time, each to its own view', async () => {
        enter(await blockOnScreen());
        const nested = collapseAll('Inner');
        setViewport({ x: 7, y: 8, zoom: 1.1 });
        enter(nested);

        exit();
        expect(flow!.getViewport()).toEqual({ x: 7, y: 8, zoom: 1.1 });
        exit();
        expect(flow!.getViewport()).toEqual(OUTER);
      });

      it('lands on the top level view when every level is left at once', async () => {
        enter(await blockOnScreen());
        enter(collapseAll('Inner'));
        setViewport({ x: 7, y: 8, zoom: 1.1 });

        act(() => useTabStore.getState().exitAllSubgraphs());
        expect(flow!.getViewport()).toEqual(OUTER);
      });

      it('treats a tab switch inside a block as a tab switch, not a change of level', async () => {
        enter(await blockOnScreen());
        setViewport({ x: 40, y: 50, zoom: 1.2 });
        switchTo('tab-b');
        switchTo('tab-a');
        expect(flow!.getViewport()).toEqual({ x: 40, y: 50, zoom: 1.2 });

        exit();
        expect(flow!.getViewport()).toEqual(OUTER);
      });

      it('starts an empty block at the default view', async () => {
        const instanceId = await blockOnScreen();
        enter(instanceId);
        act(() => useTabStore.getState().setNodes([]));
        exit();

        enter(instanceId);
        expect(flow!.getViewport()).toEqual({ x: 0, y: 0, zoom: 1 });
      });

      it('frames the graph around a block it never saw entered', async () => {
        // Mounted again while the tab is inside the block: no view was kept.
        enter(await blockOnScreen());
        cleanup();
        mount();

        exit();
        expectViewportCloseTo(framed({ x: 3000, y: 2000, width: 200, height: 80 }));
      });
    });
  });
});
