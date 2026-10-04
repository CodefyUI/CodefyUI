import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { cleanup, fireEvent } from '@testing-library/react';
import { Position, type Edge, type Node } from '@xyflow/react';
import type { NodeData, NodeDefinition, PortDefinition } from '../../types';

// #551. Every card carries a hidden, 0x0 `__trigger` target handle at its
// top-left corner, where the Start node's trigger lands. React Flow still
// snaps a dragged wire to it from anywhere inside its 20 px connection
// radius, so a data wire released over a card's header near its left edge
// was saved onto it. Validation and Export Python then refused the graph,
// while the canvas showed the wire as connected.

type RFProps = Record<string, any>;
const captured: { rf: RFProps } = { rf: {} };
// false: <ReactFlow> is a stub that records its props, for asking the
// validator directly. true: the real <ReactFlow>, so that a drag runs React
// Flow's own search for the nearest handle and consults the validator the
// way it does in a browser.
const realFlow = { value: false };

vi.mock('@xyflow/react', async (importActual) => {
  const actual = await importActual<typeof import('@xyflow/react')>();
  return {
    ...actual,
    ReactFlow: (props: RFProps) => {
      captured.rf = props;
      if (!realFlow.value) return <div data-testid="reactflow">{props.children}</div>;
      // jsdom lays nothing out, so the canvas measures 0x0. Culling would
      // then hide every card that has a size, and auto-pan would read every
      // pointer as past the canvas edge and scroll the viewport mid-drag.
      return (
        <actual.ReactFlow {...props} onlyRenderVisibleElements={false} autoPanOnConnect={false} />
      );
    },
    MiniMap: () => null,
    Background: () => null,
    Controls: () => null,
  };
});

import { FlowCanvas } from './FlowCanvas';
import { renderWithFlow } from '../../test/utils';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useI18n } from '../../i18n';

// ── Cards ───────────────────────────────────────────────────────────────────

const port = (name: string, data_type: string): PortDefinition => ({
  name,
  data_type,
  description: '',
  optional: false,
});

const def = (
  node_name: string,
  inputs: PortDefinition[],
  outputs: PortDefinition[],
): NodeDefinition => ({ node_name, category: 'Utility', description: '', inputs, outputs, params: [] });

// The two cards of the issue's repro, and the Start node.
const LINEAR = def('Linear', [port('input', 'TENSOR')], [port('output', 'TENSOR')]);
const FLATTEN = def('Flatten', [port('tensor', 'TENSOR')], [port('output', 'TENSOR')]);
const START = def('Start', [], [port('trigger', 'TRIGGER')]);

function card(id: string, definition: NodeDefinition): Node<NodeData> {
  return {
    id,
    type: definition.node_name === 'Start' ? 'start' : 'baseNode',
    position: { x: 0, y: 0 },
    data: { label: id, type: definition.node_name, params: {}, definition },
  };
}

type Point = { x: number; y: number };
type HandleBox = NonNullable<Node['handles']>[number];

const CARD = { width: 180, height: 120 };
const DOT = 17; // --handle-size

/**
 * `node` at `at`, with handle boxes laid out as BaseNode lays them out. jsdom
 * does no layout, so React Flow gets them through the node's `handles`, which
 * it reads in place of a measurement. `__trigger` is a 0x0 box 6 px left of
 * the card's top-left corner (`.triggerHandle` in BaseNode.module.css); each
 * port dot is --handle-size wide and centred on the card's edge, one ~32 px
 * row per port below a ~34 px header, inputs first. The Start pill has only
 * its trigger.
 */
function placed(node: Node<NodeData>, at: Point): Node<NodeData> {
  const { inputs, outputs } = node.data.definition!;
  const row = (i: number) => ({ y: 49 + i * 32, width: DOT, height: DOT });
  const handles: HandleBox[] = [];
  if (node.type !== 'start') {
    handles.push({ id: '__trigger', type: 'target', position: Position.Left, x: -6, y: 0, width: 0, height: 0 });
    inputs.forEach((p, i) =>
      handles.push({ id: p.name, type: 'target', position: Position.Left, x: -DOT / 2, ...row(i) }),
    );
  }
  outputs.forEach((p, i) =>
    handles.push({
      id: p.name,
      type: 'source',
      position: Position.Right,
      x: CARD.width - DOT / 2,
      ...row(inputs.length + i),
    }),
  );
  return { ...node, position: at, ...CARD, handles };
}

/**
 * The centre of a placed node's handle. Flow and screen coordinates are the
 * same here: the canvas sits at the page origin, and the viewport stays at
 * zoom 1 with no pan, because `fitView` waits for measured nodes and jsdom
 * measures none.
 */
function centreOf(node: Node<NodeData>, handleId: string): Point {
  const box = node.handles!.find((h) => h.id === handleId)!;
  return {
    x: node.position.x + box.x + (box.width ?? 0) / 2,
    y: node.position.y + box.y + (box.height ?? 0) / 2,
  };
}

// The release point of the issue's repro: over the card's header, 10 px in
// from its left edge and 10 px down. `__trigger` is 18.9 px away, inside the
// 20 px radius; the input dot is 48.5 px away, outside it.
const nearCorner = (node: Node<NodeData>): Point => ({
  x: node.position.x + 10,
  y: node.position.y + 10,
});

const lin = placed(card('lin', LINEAR), { x: 0, y: 0 });
const flat = placed(card('flat', FLATTEN), { x: 400, y: 0 });
const flat2 = placed(card('flat2', FLATTEN), { x: 400, y: 300 });
const start = placed(card('start', START), { x: 0, y: 300 });

// ── Store ───────────────────────────────────────────────────────────────────

const ORIGINAL_TABS = useTabStore.getState().tabs;
const ORIGINAL_ACTIVE = useTabStore.getState().activeTabId;
const TAB_ID = 'tab-trigger-handle-test';

function setGraph(nodes: Node<NodeData>[], edges: Edge[] = []) {
  useTabStore.setState((s) => ({
    tabs: s.tabs.map((t) => (t.id === TAB_ID ? { ...t, nodes, edges } : t)),
  }));
}

const edges = () => useTabStore.getState().tabs.find((t) => t.id === TAB_ID)!.edges;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  const base = ORIGINAL_TABS[0];
  useTabStore.setState({
    tabs: [{ ...base, id: TAB_ID, name: 'test', nodes: [], edges: [], outputSummaries: {} }],
    activeTabId: TAB_ID,
  });
  useUIStore.setState({ gridSnapEnabled: false, draggingSourceType: null, reconnectingHandle: null });
  useNodeDefStore.setState({ definitions: [LINEAR, FLATTEN, START], presets: [] });
  captured.rf = {};
  realFlow.value = false;
});

afterEach(() => {
  // Unmount first: resetting the stores under a mounted canvas re-renders it
  // outside act().
  cleanup();
  useTabStore.setState({ tabs: ORIGINAL_TABS, activeTabId: ORIGINAL_ACTIVE });
});

// ── The rule ────────────────────────────────────────────────────────────────

describe('which wires the __trigger handle takes', () => {
  const valid = (connection: Partial<Edge>) => captured.rf.isValidConnection(connection) as boolean;

  beforeEach(() => {
    const unregistered = card('bare', LINEAR);
    delete unregistered.data.definition;
    setGraph([card('start', START), card('lin', LINEAR), card('flat', FLATTEN), unregistered]);
    renderWithFlow(<FlowCanvas />);
  });

  it('refuses a data output', () => {
    expect(valid({ source: 'lin', sourceHandle: 'output', target: 'flat', targetHandle: '__trigger' })).toBe(false);
  });

  // Where the ports cannot be looked up, the validator allows the wire; each
  // of these reached that fallback.
  it.each<[string, Partial<Edge>]>([
    ['a source card with no definition', { source: 'bare', sourceHandle: 'output', target: 'flat' }],
    ['cards the tab does not hold', { source: 'gone', sourceHandle: 'output', target: 'gone2' }],
    ['a source handle with no id', { source: 'lin', sourceHandle: null, target: 'flat' }],
  ])('refuses it from %s', (_, connection) => {
    expect(valid({ ...connection, targetHandle: '__trigger' })).toBe(false);
  });

  it("takes the Start node's trigger, which goes nowhere else", () => {
    expect(valid({ source: 'start', sourceHandle: 'trigger', target: 'flat', targetHandle: '__trigger' })).toBe(true);
    expect(valid({ source: 'start', sourceHandle: 'trigger', target: 'flat', targetHandle: 'tensor' })).toBe(false);
  });

  it('treats a wire from a Start node that names no handle as its trigger', () => {
    // A trigger edge saved without `sourceHandle` loads with none, and React
    // Flow asks about moving it with none.
    expect(valid({ source: 'start', sourceHandle: null, target: 'flat', targetHandle: '__trigger' })).toBe(true);
    expect(valid({ source: 'start', sourceHandle: null, target: 'flat', targetHandle: 'tensor' })).toBe(false);
  });

  it('leaves a data output on a real input alone', () => {
    expect(valid({ source: 'lin', sourceHandle: 'output', target: 'flat', targetHandle: 'tensor' })).toBe(true);
  });
});

// ── Dragging on the real canvas ─────────────────────────────────────────────

function handleOf(nodeId: string, handleId: string): Element {
  const el = document.querySelector(
    `.react-flow__handle[data-nodeid="${nodeId}"][data-handleid="${handleId}"]`,
  );
  if (!el) throw new Error(`no handle ${nodeId}.${handleId} on the canvas`);
  return el;
}

/**
 * A left-button drag as a browser delivers one: the press on `from`, then the
 * move and the release on the document, where React Flow listens once a drag
 * has begun.
 */
function drag(from: Element, pressAt: Point, releaseAt: Point) {
  fireEvent.mouseDown(from, { button: 0, buttons: 1, clientX: pressAt.x, clientY: pressAt.y });
  fireEvent.mouseMove(document, { buttons: 1, clientX: releaseAt.x, clientY: releaseAt.y });
  fireEvent.mouseUp(document, { button: 0, clientX: releaseAt.x, clientY: releaseAt.y });
}

// In each block below, the cases that expect a wire are the controls: if a
// drag stopped reaching React Flow, the cases that expect none would pass for
// nothing, and the controls would fail.
describe('dragging a wire on the real canvas', () => {
  beforeEach(() => {
    realFlow.value = true;
    // React Flow asks what is under the pointer, to prefer a handle the
    // pointer is on over the nearest one; jsdom has neither layout nor
    // elementFromPoint. Answering "nothing" sends React Flow to the nearest
    // handle inside its radius. By a card's corner a browser answers the same,
    // since the pointer is over the header and the hidden `__trigger` takes no
    // pointer events; on an input's dot, the nearest handle is that dot.
    Object.defineProperty(document, 'elementFromPoint', { configurable: true, value: () => null });
  });

  afterEach(() => {
    delete (document as { elementFromPoint?: unknown }).elementFromPoint;
  });

  describe('a new wire', () => {
    it("connects nothing when a data wire is released by a card's top-left corner", () => {
      setGraph([lin, flat]);
      renderWithFlow(<FlowCanvas />);
      drag(handleOf('lin', 'output'), centreOf(lin, 'output'), nearCorner(flat));
      // React Flow tries the nearest handle only; refused, the release
      // connects nothing.
      expect(edges()).toEqual([]);
    });

    it("still connects the Start node's trigger released there", () => {
      setGraph([start, flat]);
      renderWithFlow(<FlowCanvas />);
      drag(handleOf('start', 'trigger'), centreOf(start, 'trigger'), nearCorner(flat));
      expect(edges()).toMatchObject([
        { source: 'start', sourceHandle: 'trigger', target: 'flat', targetHandle: '__trigger', type: 'triggerEdge' },
      ]);
    });

    it("still connects a data wire released on an input's dot", () => {
      setGraph([lin, flat]);
      renderWithFlow(<FlowCanvas />);
      drag(handleOf('lin', 'output'), centreOf(lin, 'output'), centreOf(flat, 'tensor'));
      expect(edges()).toMatchObject([
        { source: 'lin', sourceHandle: 'output', target: 'flat', targetHandle: 'tensor' },
      ]);
    });
  });

  describe('a wire moved to another card', () => {
    const wire: Edge = { id: 'e1', source: 'lin', sourceHandle: 'output', target: 'flat', targetHandle: 'tensor' };
    const triggerWire: Edge = {
      id: 't1',
      source: 'start',
      sourceHandle: 'trigger',
      target: 'flat',
      targetHandle: '__trigger',
      type: 'triggerEdge',
      data: { type: 'trigger' },
    };

    // Pressing a connected input grabs its wire: BaseNode hands the press to
    // the wire's reconnect anchor, and React Flow drags the loose end.
    it("is not moved onto __trigger when released by the other card's corner", () => {
      setGraph([lin, flat, flat2], [wire]);
      renderWithFlow(<FlowCanvas />);
      drag(handleOf('flat', 'tensor'), centreOf(flat, 'tensor'), nearCorner(flat2));
      // A grabbed wire released where it connects to nothing is removed, as
      // on the empty canvas; the red ring on the grabbed end says so during
      // the drag.
      expect(edges()).toEqual([]);
    });

    it("is moved onto the other card's input", () => {
      setGraph([lin, flat, flat2], [wire]);
      renderWithFlow(<FlowCanvas />);
      drag(handleOf('flat', 'tensor'), centreOf(flat, 'tensor'), centreOf(flat2, 'tensor'));
      expect(edges()).toMatchObject([
        { id: 'e1', source: 'lin', sourceHandle: 'output', target: 'flat2', targetHandle: 'tensor' },
      ]);
    });

    it("moves the Start node's trigger from one card's __trigger to another's", () => {
      setGraph([start, flat, flat2], [triggerWire]);
      renderWithFlow(<FlowCanvas />);
      // `__trigger` takes no presses itself, so the trigger wire is grabbed
      // at its own reconnect anchor, which sits on that corner.
      const anchor = document.querySelector(
        '.react-flow__edge[data-id="t1"] .react-flow__edgeupdater-target',
      );
      expect(anchor).not.toBeNull();
      drag(anchor!, centreOf(flat, '__trigger'), nearCorner(flat2));
      expect(edges()).toMatchObject([
        { id: 't1', source: 'start', target: 'flat2', targetHandle: '__trigger', type: 'triggerEdge' },
      ]);
    });

    it('moves a trigger wire saved without its handle name', () => {
      // What the loader makes of a hand-written `{"type": "trigger",
      // "source": "start", "target": "flat"}`: no source handle at all.
      const unnamed: Edge = { ...triggerWire, id: 't0', sourceHandle: undefined };
      setGraph([start, flat, flat2], [unnamed]);
      renderWithFlow(<FlowCanvas />);
      const anchor = document.querySelector(
        '.react-flow__edge[data-id="t0"] .react-flow__edgeupdater-target',
      );
      expect(anchor).not.toBeNull();
      drag(anchor!, centreOf(flat, '__trigger'), nearCorner(flat2));
      expect(edges()).toMatchObject([
        { id: 't0', source: 'start', target: 'flat2', targetHandle: '__trigger', type: 'triggerEdge' },
      ]);
    });

    it("is not moved off the Start node onto a data output", () => {
      setGraph([start, lin, flat], [triggerWire]);
      renderWithFlow(<FlowCanvas />);
      // The wire's Start end: React Flow drags it from the `__trigger` end,
      // which stays where it is.
      const anchor = document.querySelector(
        '.react-flow__edge[data-id="t1"] .react-flow__edgeupdater-source',
      );
      expect(anchor).not.toBeNull();
      drag(anchor!, centreOf(start, 'trigger'), centreOf(lin, 'output'));
      // Refused, the grabbed wire is removed, as on the empty canvas. An
      // untouched trigger wire would fail this too, so the drag is known to
      // have happened.
      expect(edges()).toEqual([]);
    });
  });
});
