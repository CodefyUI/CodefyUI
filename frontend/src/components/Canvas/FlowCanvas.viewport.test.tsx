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

function mount(strict = false) {
  const tree = (
    <ReactFlowProvider>
      <ViewportProbe />
      <FlowCanvas tabId="tab-a" />
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

beforeEach(() => {
  _resetViewportMemory();
  seedTabs();
  useUIStore.setState({ layoutFitRequest: null });
  flow = null;
  flowStore = null;
});

afterEach(() => {
  cleanup();
  _resetViewportMemory();
  useUIStore.setState({ layoutFitRequest: null });
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

  it('leaves the viewport alone when switching to an EMPTY tab', () => {
    // Nothing to frame; inventing a position for a blank canvas would just
    // be a jump the user did not ask for.
    const restore = withCanvasSize();
    try {
      mount();
      setViewport({ x: 55, y: 66, zoom: 1.25 });
      switchTo('tab-empty');
      expect(flow!.getViewport()).toEqual({ x: 55, y: 66, zoom: 1.25 });
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
          useUIStore.getState().requestLayoutFit({ x: 0, y: 0, width: 200, height: 80 });
        });
        expectViewportCloseTo(framedOn900x600());
        expect(useUIStore.getState().layoutFitRequest).toBeNull();
      } finally {
        restore();
      }
    });
  });
});
