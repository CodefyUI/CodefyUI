import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, cleanup, fireEvent } from '@testing-library/react';
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

// Two inputs, and two outputs: the same two cards joined again by other ports
// is not a duplicate (#619). Each stands in for `flat` or `lin` in its tests.
// Not named Split: Split's outputs follow its `chunks` param.
const add = placed(
  card('add', def('Add', [port('tensor_a', 'TENSOR'), port('tensor_b', 'TENSOR')], [port('output', 'TENSOR')])),
  { x: 400, y: 0 },
);
const pair = placed(
  card('pair', def('Pair', [port('tensor', 'TENSOR')], [port('first', 'TENSOR'), port('second', 'TENSOR')])),
  { x: 0, y: 0 },
);

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
const undoFrames = () => useTabStore.getState().tabs.find((t) => t.id === TAB_ID)!.undoStack.length;

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

// Every drop asks the validity check before onReconnect; onReconnect refuses
// a copy as well, so a drop that ever reached it unasked could not stack one
// (#619).
describe('onReconnect, asked directly', () => {
  const wire: Edge = { id: 'e1', source: 'lin', sourceHandle: 'output', target: 'flat', targetHandle: 'tensor' };
  const other: Edge = { ...wire, id: 'e2', target: 'flat2' };

  beforeEach(() => {
    setGraph([card('lin', LINEAR), card('flat', FLATTEN), card('flat2', FLATTEN)], [wire, other]);
    renderWithFlow(<FlowCanvas />);
  });

  it('moves a wire to ports no wire joins', () => {
    act(() =>
      captured.rf.onReconnect(other, { source: 'flat', sourceHandle: 'output', target: 'flat2', targetHandle: 'tensor' }),
    );
    expect(edges()).toMatchObject([wire, { id: 'e2', source: 'flat', target: 'flat2', targetHandle: 'tensor' }]);
  });

  it('leaves a wire where it is rather than on the ports another already joins', () => {
    act(() =>
      captured.rf.onReconnect(other, { source: 'lin', sourceHandle: 'output', target: 'flat', targetHandle: 'tensor' }),
    );
    expect(edges()).toEqual([wire, other]);
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

    // A wire that is already there is not added again (#619). The two cases
    // above that connect are the controls: the same drags, with no wire yet.
    it('adds no second trigger released by the corner of a card its Start node already triggers', () => {
      setGraph([start, flat], [triggerWire]);
      renderWithFlow(<FlowCanvas />);
      drag(handleOf('start', 'trigger'), centreOf(start, 'trigger'), nearCorner(flat));
      expect(edges()).toEqual([triggerWire]);
    });

    it('adds no second data wire released on the input its output already feeds', () => {
      setGraph([lin, flat], [wire]);
      renderWithFlow(<FlowCanvas />);
      drag(handleOf('lin', 'output'), centreOf(lin, 'output'), centreOf(flat, 'tensor'));
      expect(edges()).toEqual([wire]);
    });

    it('still connects another output to an input that already has a wire', () => {
      // Fan-in, not a duplicate: branches merge this way, and the engine and
      // the exported script both take the last source that produced a value.
      setGraph([lin, flat, flat2], [wire]);
      renderWithFlow(<FlowCanvas />);
      drag(handleOf('flat2', 'output'), centreOf(flat2, 'output'), centreOf(flat, 'tensor'));
      expect(edges()).toMatchObject([
        wire,
        { source: 'flat2', sourceHandle: 'output', target: 'flat', targetHandle: 'tensor' },
      ]);
    });

    // A rule that compared cards and not ports would refuse these two.
    it('still connects an output to a second input of the card it already feeds', () => {
      // `x` into both inputs of Add.
      const first: Edge = { id: 'e1', source: 'lin', sourceHandle: 'output', target: 'add', targetHandle: 'tensor_a' };
      setGraph([lin, add], [first]);
      renderWithFlow(<FlowCanvas />);
      drag(handleOf('lin', 'output'), centreOf(lin, 'output'), centreOf(add, 'tensor_b'));
      expect(edges()).toMatchObject([
        first,
        { source: 'lin', sourceHandle: 'output', target: 'add', targetHandle: 'tensor_b' },
      ]);
    });

    it("still connects a card's other output to an input that card already feeds", () => {
      const first: Edge = { id: 'e1', source: 'pair', sourceHandle: 'first', target: 'flat', targetHandle: 'tensor' };
      setGraph([pair, flat], [first]);
      renderWithFlow(<FlowCanvas />);
      drag(handleOf('pair', 'second'), centreOf(pair, 'second'), centreOf(flat, 'tensor'));
      expect(edges()).toMatchObject([
        first,
        { source: 'pair', sourceHandle: 'second', target: 'flat', targetHandle: 'tensor' },
      ]);
    });
  });

  // React Flow's click-to-connect, on by default: a click on an output, then
  // one on an input, goes through the same validity check and onConnect as a
  // drag (#562).
  describe('click-to-connect', () => {
    const clickConnect = () => {
      fireEvent.click(handleOf('lin', 'output'));
      fireEvent.click(handleOf('flat', 'tensor'));
    };

    it('connects an output to an input', () => {
      setGraph([lin, flat]);
      renderWithFlow(<FlowCanvas />);
      clickConnect();
      expect(edges()).toMatchObject([
        { source: 'lin', sourceHandle: 'output', target: 'flat', targetHandle: 'tensor' },
      ]);
    });

    it('adds nothing when the two are already joined', () => {
      setGraph([lin, flat], [wire]);
      renderWithFlow(<FlowCanvas />);
      clickConnect();
      expect(edges()).toEqual([wire]);
    });
  });

  describe('a wire moved to another card', () => {
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

    it('is removed, not doubled, on an input its output already feeds', () => {
      // The wire in hand would repeat `wire` exactly, so it is removed and
      // `wire` stays, as with a trigger moved onto a card its Start node
      // already triggers, below (#619).
      const other: Edge = { ...wire, id: 'e2', target: 'flat2' };
      setGraph([lin, flat, flat2], [wire, other]);
      renderWithFlow(<FlowCanvas />);
      drag(handleOf('flat2', 'tensor'), centreOf(flat2, 'tensor'), centreOf(flat, 'tensor'));
      expect(edges()).toEqual([wire]);
    });

    it('stays when released back on its own input', () => {
      // The wire in hand is not a duplicate of itself. Released a few pixels
      // from where it was pressed: a press released on the spot is a click.
      setGraph([lin, flat, flat2], [wire]);
      renderWithFlow(<FlowCanvas />);
      const { x, y } = centreOf(flat, 'tensor');
      drag(handleOf('flat', 'tensor'), { x, y }, { x: x + 3, y: y + 3 });
      expect(edges()).toMatchObject([
        { id: 'e1', source: 'lin', sourceHandle: 'output', target: 'flat', targetHandle: 'tensor' },
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

    it("removes the Start node's trigger moved onto the __trigger of a card it already triggers", () => {
      const other: Edge = { ...triggerWire, id: 't2', target: 'flat2' };
      setGraph([start, flat, flat2], [triggerWire, other]);
      renderWithFlow(<FlowCanvas />);
      const anchor = document.querySelector(
        '.react-flow__edge[data-id="t1"] .react-flow__edgeupdater-target',
      );
      expect(anchor).not.toBeNull();
      drag(anchor!, centreOf(flat, '__trigger'), nearCorner(flat2));
      // Refused as the drop on that card's body is: the grabbed wire is
      // removed rather than stacked on the one already there.
      expect(edges()).toEqual([other]);
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

    // React Flow gives an edge a reconnect anchor at each end. The one at the
    // source end lies just outside the output port, under the node layer, so
    // a press just beside a Start node's diamond or an output's dot took the
    // wire there, and a release on the empty canvas deleted it (#593). Only
    // the target end, the one dragging a connected input grabs, has one.
    it.each<[string, Edge]>([
      ["the Start node's trigger wire", triggerWire],
      ['a data wire', wire],
    ])('leaves %s no anchor at its source end to grab by mistake', (_, edge) => {
      setGraph([start, lin, flat], [edge]);
      renderWithFlow(<FlowCanvas />);
      // Booleans, not the elements: printing an SVG element on a failure
      // crashes the reporter.
      const hasAnchor = (end: 'source' | 'target') =>
        document.querySelector(
          `.react-flow__edge[data-id="${edge.id}"] .react-flow__edgeupdater-${end}`,
        ) !== null;
      // The target end's anchor is the control: the edge is drawn.
      expect(hasAnchor('target')).toBe(true);
      expect(hasAnchor('source')).toBe(false);
    });
  });
});

// ── Dropping a trigger on a card ────────────────────────────────────────────

// While a trigger is dragged, every card that can take one glows as a drop
// target, but React Flow connects only on or near a handle, and a card's
// `__trigger` is a 13 px diamond at its top-left corner: a few screen pixels
// at the zoom a fitted graph opens at. FlowCanvas connects a trigger released
// anywhere on such a card.

/**
 * What a browser's `elementFromPoint` answers over the placed nodes: the body
 * of the card under the point, never one of its handles, or else the pane.
 */
function hitTest(nodes: Node<NodeData>[]) {
  return (x: number, y: number): Element | null => {
    const hit = nodes.find(
      (n) =>
        x >= n.position.x &&
        x <= n.position.x + (n.width ?? 0) &&
        y >= n.position.y &&
        y <= n.position.y + (n.height ?? 0),
    );
    if (!hit) return document.querySelector('.react-flow__pane');
    return document.querySelector(`.react-flow__node[data-id="${hit.id}"]`)?.firstElementChild ?? null;
  };
}

/** The middle of a placed node: no handle is within 20 px of it. */
const middleOf = (node: Node<NodeData>): Point => ({
  x: node.position.x + (node.width ?? 0) / 2,
  y: node.position.y + (node.height ?? 0) / 2,
});

describe('dropping a trigger on a card', () => {
  const note: Node<NodeData> = {
    id: 'note',
    type: 'noteNode',
    position: { x: 400, y: 300 },
    ...CARD,
    data: { label: 'note', type: 'note', params: {}, noteKind: 'text', noteContent: 'a note' },
  };
  const start2 = placed(card('start2', START), { x: 400, y: 300 });
  const preset: Node<NodeData> = {
    id: 'preset',
    type: 'presetNode',
    position: { x: 400, y: 0 },
    ...CARD,
    data: {
      label: 'preset',
      type: 'preset:Block',
      params: {},
      isPreset: true,
      presetDefinition: {
        preset_name: 'Block',
        category: 'Utility',
        description: '',
        tags: [],
        nodes: [{ id: 'n0', type: 'Linear', params: {} }],
        edges: [],
        exposed_inputs: [],
        exposed_outputs: [],
        exposed_params: [],
      },
    },
  };
  const triggerWire: Edge = {
    id: 't1',
    source: 'start',
    sourceHandle: 'trigger',
    target: 'flat',
    targetHandle: '__trigger',
    type: 'triggerEdge',
    data: { type: 'trigger' },
  };

  function show(nodes: Node<NodeData>[], wires: Edge[] = []) {
    setGraph(nodes, wires);
    Object.defineProperty(document, 'elementFromPoint', { configurable: true, value: hitTest(nodes) });
    renderWithFlow(<FlowCanvas />);
  }

  const dragTrigger = (releaseAt: Point) =>
    drag(handleOf('start', 'trigger'), centreOf(start, 'trigger'), releaseAt);

  beforeEach(() => {
    realFlow.value = true;
  });

  afterEach(() => {
    delete (document as { elementFromPoint?: unknown }).elementFromPoint;
  });

  it('connects a trigger released in the middle of a card to that card', () => {
    show([start, flat]);
    dragTrigger(middleOf(flat));
    expect(edges()).toMatchObject([
      {
        source: 'start',
        sourceHandle: 'trigger',
        target: 'flat',
        targetHandle: '__trigger',
        type: 'triggerEdge',
        data: { type: 'trigger' },
      },
    ]);
  });

  it("connects a trigger released on a preset card's body", () => {
    // A preset card has its own `__trigger`; what the engine runs for a
    // trigger into a card is a separate question (#561).
    show([start, preset]);
    dragTrigger(middleOf(preset));
    expect(edges()).toMatchObject([
      { source: 'start', sourceHandle: 'trigger', target: 'preset', targetHandle: '__trigger', type: 'triggerEdge' },
    ]);
  });

  it('connects it once, however often it is dropped there', () => {
    show([start, flat]);
    dragTrigger(middleOf(flat));
    dragTrigger(middleOf(flat));
    expect(edges()).toHaveLength(1);
  });

  // The first case above is the control for these: it fails if a drag stops
  // reaching React Flow.
  it.each<[string, Node<NodeData>[], Point]>([
    ['the empty canvas', [start, flat], { x: 300, y: 200 }],
    ['a note', [start, note], middleOf(note)],
    ['another Start node', [start, start2], middleOf(start2)],
    ['the Start node it comes from', [start, flat], middleOf(start)],
  ])('connects nothing when it is released over %s', (_, nodes, releaseAt) => {
    show(nodes);
    dragTrigger(releaseAt);
    expect(edges()).toEqual([]);
  });

  it('connects nothing when the cards were not lit as trigger targets', () => {
    // The glow comes from onConnectStart, which looks the dragged output's
    // type up in the node's definition. With none, no card glows, so a
    // release on a card's body connects nothing.
    const unlit: Node<NodeData> = { ...start, data: { ...start.data, definition: undefined } };
    show([unlit, flat]);
    dragTrigger(middleOf(flat));
    expect(useUIStore.getState().draggingSourceType).toBeNull();
    expect(edges()).toEqual([]);
  });

  it('leaves a data wire released in the middle of a card unconnected', () => {
    show([lin, flat]);
    drag(handleOf('lin', 'output'), centreOf(lin, 'output'), middleOf(flat));
    expect(edges()).toEqual([]);
  });

  // A wire that is moved lands by a card's body the same way (#593).
  describe('a trigger wire moved off its card', () => {
    /**
     * Grab trigger wire `id` on the corner of `card`, the card it goes into,
     * and release it at `releaseAt`. `__trigger` takes no presses itself, so
     * the press lands on the wire's own reconnect anchor on that corner.
     */
    const moveTriggerWire = (id: string, card: Node<NodeData>, releaseAt: Point) => {
      const anchor = document.querySelector(
        `.react-flow__edge[data-id="${id}"] .react-flow__edgeupdater-target`,
      );
      expect(anchor).not.toBeNull();
      drag(anchor!, centreOf(card, '__trigger'), releaseAt);
    };

    it("moves onto another card when released in the middle of it", () => {
      show([start, flat, flat2], [triggerWire]);
      moveTriggerWire('t1', flat, middleOf(flat2));
      expect(edges()).toMatchObject([
        {
          id: 't1',
          source: 'start',
          sourceHandle: 'trigger',
          target: 'flat2',
          targetHandle: '__trigger',
          type: 'triggerEdge',
          data: { type: 'trigger' },
        },
      ]);
    });

    it('goes back with one undo', () => {
      show([start, flat, flat2], [triggerWire]);
      const frames = undoFrames();
      moveTriggerWire('t1', flat, middleOf(flat2));
      expect(edges()).toMatchObject([{ id: 't1', target: 'flat2' }]);
      expect(undoFrames()).toBe(frames + 1);
      act(() => useTabStore.getState().undo());
      expect(edges()).toEqual([triggerWire]);
    });

    it('stays when released back on its own card', () => {
      // The wire being moved is not one its Start node already has there.
      show([start, flat, flat2], [triggerWire]);
      moveTriggerWire('t1', flat, middleOf(flat));
      expect(edges()).toMatchObject([
        { id: 't1', source: 'start', target: 'flat', targetHandle: '__trigger', type: 'triggerEdge' },
      ]);
    });

    it('is not brought back when it was deleted while in hand', () => {
      show([start, flat, flat2], [triggerWire]);
      const anchor = document.querySelector(
        '.react-flow__edge[data-id="t1"] .react-flow__edgeupdater-target',
      )!;
      const pressAt = centreOf(flat, '__trigger');
      const releaseAt = middleOf(flat2);
      fireEvent.mouseDown(anchor, { button: 0, buttons: 1, clientX: pressAt.x, clientY: pressAt.y });
      fireEvent.mouseMove(document, { buttons: 1, clientX: releaseAt.x, clientY: releaseAt.y });
      // Say Delete took it, as it does a selected wire, before the release.
      act(() => setGraph([start, flat, flat2], []));
      fireEvent.mouseUp(document, { button: 0, clientX: releaseAt.x, clientY: releaseAt.y });
      expect(edges()).toEqual([]);
    });

    it('is removed, not doubled, on a card its Start node already triggers', () => {
      const other: Edge = { ...triggerWire, id: 't2', target: 'flat2' };
      show([start, flat, flat2], [triggerWire, other]);
      moveTriggerWire('t1', flat, middleOf(flat2));
      expect(edges()).toEqual([other]);
    });

    // As before, a moved wire that lands nowhere is removed; the red ring on
    // the grabbed end says so during the drag.
    it.each<[string, Node<NodeData>[], Point]>([
      ['the empty canvas', [start, flat], { x: 300, y: 200 }],
      ['a note', [start, flat, note], middleOf(note)],
      ['another Start node', [start, flat, start2], middleOf(start2)],
      ['the Start node it comes from', [start, flat], middleOf(start)],
    ])('is removed when released over %s', (_, nodes, releaseAt) => {
      show(nodes, [triggerWire]);
      moveTriggerWire('t1', flat, releaseAt);
      expect(edges()).toEqual([]);
    });
  });
});
