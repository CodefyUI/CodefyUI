import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  computeDetachedEndpoint,
  findDetachableEdge,
  redirectMouseDownToReconnectAnchor,
} from './reconnect';

// `handleType` semantics (verified against @xyflow/react 12.10.1
// EdgeUpdateAnchors): it names the end that STAYS connected — the drag-origin
// handle — so the detached endpoint is always the OPPOSITE end.
describe('computeDetachedEndpoint', () => {
  const edge = { source: 'a', target: 'b', sourceHandle: 'out', targetHandle: 'in' };

  it("handleType 'source' (source stays) -> the grabbed/detached endpoint is the target end", () => {
    expect(computeDetachedEndpoint(edge, 'source')).toEqual({
      nodeId: 'b',
      handleId: 'in',
      type: 'target',
    });
  });

  it("handleType 'target' (target stays) -> the grabbed/detached endpoint is the source end", () => {
    expect(computeDetachedEndpoint(edge, 'target')).toEqual({
      nodeId: 'a',
      handleId: 'out',
      type: 'source',
    });
  });

  it('returns null when the detached end has no handle id', () => {
    expect(computeDetachedEndpoint({ source: 'a', target: 'b' }, 'source')).toBeNull();
    expect(computeDetachedEndpoint({ source: 'a', target: 'b' }, 'target')).toBeNull();
  });

  it('a missing handle id on the STAYING end does not matter', () => {
    // Only the detached end's handle id is required.
    expect(
      computeDetachedEndpoint({ source: 'a', target: 'b', targetHandle: 'in' }, 'source'),
    ).toEqual({ nodeId: 'b', handleId: 'in', type: 'target' });
    expect(
      computeDetachedEndpoint({ source: 'a', target: 'b', sourceHandle: 'out' }, 'target'),
    ).toEqual({ nodeId: 'a', handleId: 'out', type: 'source' });
  });
});

describe('findDetachableEdge', () => {
  const edges = [
    { id: 'e1', source: 's1', sourceHandle: 'out', target: 'n1', targetHandle: 'a' },
    { id: 'e2', source: 's2', sourceHandle: 'out', target: 'n1', targetHandle: 'b' },
    { id: 'e3', source: 's3', sourceHandle: 'out', target: 'n1', targetHandle: 'a' },
  ];

  it('returns the LAST edge into the given input (topmost-rendered wins)', () => {
    expect(findDetachableEdge(edges, 'n1', 'a')?.id).toBe('e3');
  });

  it('returns a single match', () => {
    expect(findDetachableEdge(edges, 'n1', 'b')?.id).toBe('e2');
  });

  it('returns null when nothing targets the handle', () => {
    expect(findDetachableEdge(edges, 'n1', 'c')).toBeNull();
    expect(findDetachableEdge(edges, 'n2', 'a')).toBeNull();
    expect(findDetachableEdge([], 'n1', 'a')).toBeNull();
  });

  it('ignores edges that merely ORIGINATE at the node/handle', () => {
    const outgoing = [
      { id: 'e4', source: 'n1', sourceHandle: 'a', target: 'x', targetHandle: 'in' },
    ];
    expect(findDetachableEdge(outgoing, 'n1', 'a')).toBeNull();
  });
});

describe('redirectMouseDownToReconnectAnchor', () => {
  const SVG_NS = 'http://www.w3.org/2000/svg';

  /**
   * Build the real React Flow edge DOM shape (verified against
   * @xyflow/react 12.10.1 EdgeWrapper/EdgeAnchor): an svg containing
   * `<g class="react-flow__edge" data-id="<edgeId>">` with the invisible
   * `<circle class="react-flow__edgeupdater react-flow__edgeupdater-<type>">`
   * reconnect anchor inside. Mounted under `parent` (a `.react-flow`
   * canvas container in the scoping tests, document.body otherwise).
   */
  function mountAnchorFixture(
    edgeId: string,
    anchorType: 'target' | 'source' = 'target',
    parent: Element = document.body,
  ) {
    const svg = document.createElementNS(SVG_NS, 'svg');
    const group = document.createElementNS(SVG_NS, 'g');
    group.setAttribute('class', 'react-flow__edge react-flow__edge-default');
    group.setAttribute('data-id', edgeId);
    group.setAttribute('data-testid', `rf__edge-${edgeId}`);
    const circle = document.createElementNS(SVG_NS, 'circle');
    circle.setAttribute(
      'class',
      `react-flow__edgeupdater react-flow__edgeupdater-${anchorType}`,
    );
    circle.setAttribute('r', '10');
    group.appendChild(circle);
    svg.appendChild(group);
    parent.appendChild(svg);
    const received: MouseEvent[] = [];
    circle.addEventListener('mousedown', (e) => received.push(e as MouseEvent));
    return { circle, received };
  }

  /** Create a `.react-flow` canvas container div (one per open tab in the app). */
  function mountCanvas() {
    const canvas = document.createElement('div');
    canvas.className = 'react-flow';
    document.body.appendChild(canvas);
    return canvas;
  }

  // NOTE: the default currentTarget (document.body) has no `.react-flow`
  // ancestor, so tests using this stub exercise the document-wide FALLBACK
  // path — real handles always sit inside a canvas (covered by the
  // canvas-scoping tests below and the component tests).
  function stubEvent(overrides: Partial<{
    clientX: number;
    clientY: number;
    currentTarget: EventTarget | null;
  }> = {}) {
    return {
      clientX: 111,
      clientY: 222,
      currentTarget: document.body as EventTarget | null,
      preventDefault: vi.fn(),
      stopPropagation: vi.fn(),
      ...overrides,
    };
  }

  afterEach(() => {
    document.body.innerHTML = '';
  });

  it('dispatches a left-button mousedown with the source coordinates on the anchor and suppresses the original event', () => {
    const { received } = mountAnchorFixture('e1');
    const event = stubEvent({ clientX: 120, clientY: 45 });

    expect(redirectMouseDownToReconnectAnchor(event, 'e1')).toBe(true);

    expect(received).toHaveLength(1);
    expect(received[0].clientX).toBe(120);
    expect(received[0].clientY).toBe(45);
    expect(received[0].button).toBe(0);
    expect(received[0].buttons).toBe(1);
    expect(received[0].bubbles).toBe(true);
    expect(received[0].cancelable).toBe(true);
    expect(event.preventDefault).toHaveBeenCalledTimes(1);
    expect(event.stopPropagation).toHaveBeenCalledTimes(1);
  });

  it('carries a `view`, so the d3 handlers on the canvas can read `event.view.document` (#219)', () => {
    // The dispatched press bubbles into React Flow's canvas, where BOTH d3
    // gesture recognizers dereference `event.view` on every mousedown:
    // d3-drag's `mousedowned` calls `nodrag(event.view)` and d3-zoom's calls
    // `dragDisable(event.view)` — the same function, whose first statement
    // is `view.document.documentElement`, unguarded, in a dependency we do
    // not own. A `MouseEvent` built without `view` has `view === null`, and
    // that read is exactly the "Cannot read properties of null (reading
    // 'document')" #219 recorded. The listener below is that read, verbatim.
    const canvas = mountCanvas();
    const { received } = mountAnchorFixture('e1', 'target', canvas);
    const handle = document.createElement('div');
    canvas.appendChild(handle);

    let thrown: unknown = null;
    canvas.addEventListener('mousedown', (e) => {
      try {
        void (e as MouseEvent).view!.document.documentElement;
      } catch (err) {
        // jsdom reports a listener throw to window.onerror rather than
        // propagating it out of dispatchEvent, so it has to be captured
        // here or the test would pass while the canvas was breaking.
        thrown = err;
      }
    });

    const event = stubEvent({ currentTarget: handle });
    expect(redirectMouseDownToReconnectAnchor(event, 'e1')).toBe(true);

    expect(received).toHaveLength(1);
    expect(received[0].view).not.toBeNull();
    expect(received[0].view).toBe(canvas.ownerDocument.defaultView);
    expect(thrown).toBeNull();
  });

  it('returns false and leaves the event untouched when no anchor exists', () => {
    const event = stubEvent();
    expect(redirectMouseDownToReconnectAnchor(event, 'e1')).toBe(false);
    expect(event.preventDefault).not.toHaveBeenCalled();
    expect(event.stopPropagation).not.toHaveBeenCalled();
  });

  it("does not match a DIFFERENT edge's anchor", () => {
    const { received } = mountAnchorFixture('other-edge');
    const event = stubEvent();
    expect(redirectMouseDownToReconnectAnchor(event, 'e1')).toBe(false);
    expect(received).toHaveLength(0);
    expect(event.preventDefault).not.toHaveBeenCalled();
  });

  it('only targets the TARGET anchor, never the source anchor', () => {
    const { received } = mountAnchorFixture('e1', 'source');
    const event = stubEvent();
    expect(redirectMouseDownToReconnectAnchor(event, 'e1')).toBe(false);
    expect(received).toHaveLength(0);
    expect(event.stopPropagation).not.toHaveBeenCalled();
  });

  it('falls back to the global document when currentTarget is not an element', () => {
    const { received } = mountAnchorFixture('e1');
    const event = stubEvent({ currentTarget: null });
    expect(redirectMouseDownToReconnectAnchor(event, 'e1')).toBe(true);
    expect(received).toHaveLength(1);
  });

  it('escapes CSS-hostile characters in the edge id', () => {
    const hostileId = 'xy-edge__a"b\\c';
    const { received } = mountAnchorFixture(hostileId);
    const event = stubEvent();
    expect(redirectMouseDownToReconnectAnchor(event, hostileId)).toBe(true);
    expect(received).toHaveLength(1);
  });

  it("scopes the lookup to the handle's OWN canvas when edge ids collide across canvases", () => {
    // The app mounts one hidden <FlowCanvas> per open tab, and graphs
    // loaded from example files reuse fixed edge ids — the same id can
    // exist in several canvases at once. The decoy canvas comes FIRST in
    // document order, so an unscoped document query would dispatch into
    // the wrong (hidden) canvas.
    const decoyCanvas = mountCanvas();
    const decoy = mountAnchorFixture('dup-edge', 'target', decoyCanvas);

    const ownCanvas = mountCanvas();
    const own = mountAnchorFixture('dup-edge', 'target', ownCanvas);
    const handle = document.createElement('div');
    ownCanvas.appendChild(handle);

    const event = stubEvent({ currentTarget: handle });
    expect(redirectMouseDownToReconnectAnchor(event, 'dup-edge')).toBe(true);

    expect(own.received).toHaveLength(1);
    expect(decoy.received).toHaveLength(0);
    expect(event.preventDefault).toHaveBeenCalledTimes(1);
    expect(event.stopPropagation).toHaveBeenCalledTimes(1);
  });

  it('does NOT borrow an anchor from outside the canvas when the handle sits inside one', () => {
    // Anchor exists only OUTSIDE the handle's canvas (e.g. another tab's
    // edge) — the scoped lookup must miss and leave the event untouched
    // rather than falling back to the whole document.
    const outside = mountAnchorFixture('e1');
    const canvas = mountCanvas();
    const handle = document.createElement('div');
    canvas.appendChild(handle);

    const event = stubEvent({ currentTarget: handle });
    expect(redirectMouseDownToReconnectAnchor(event, 'e1')).toBe(false);

    expect(outside.received).toHaveLength(0);
    expect(event.preventDefault).not.toHaveBeenCalled();
    expect(event.stopPropagation).not.toHaveBeenCalled();
  });
});
