import type { Edge } from '@xyflow/react';

/**
 * The edge endpoint being detached during an in-progress reconnect drag.
 * Matches the shape of `uiStore.reconnectingHandle`.
 */
export interface DetachedEndpoint {
  nodeId: string;
  handleId: string;
  type: 'source' | 'target';
}

/**
 * Map React Flow's `onReconnectStart(event, edge, handleType)` arguments to
 * the endpoint the user grabbed the edge off — the one being detached and
 * therefore the one that should show the red warning ring.
 *
 * Semantics verified against the installed @xyflow/react 12.10.1 source
 * (dist/esm/index.js, `EdgeUpdateAnchors`): the `handleType` argument is
 * `oppositeHandle.type` — the type of the handle that STAYS connected and
 * from which the replacement connection is dragged — NOT the end the user
 * grabbed. Grabbing the target end starts a drag from the source handle and
 * reports `handleType === 'source'`; grabbing the source end reports
 * `handleType === 'target'`. The detached endpoint is therefore always the
 * opposite end of `handleType`.
 *
 * Returns null when the detached end has no handle id. Edges created by this
 * app always carry handle ids (port names, 'trigger', '__trigger'), but the
 * Edge type allows them to be absent.
 */
export function computeDetachedEndpoint(
  edge: Pick<Edge, 'source' | 'target' | 'sourceHandle' | 'targetHandle'>,
  handleType: 'source' | 'target',
): DetachedEndpoint | null {
  if (handleType === 'source') {
    // The source end stays connected -> the user grabbed the TARGET endpoint.
    if (edge.targetHandle == null) return null;
    return { nodeId: edge.target, handleId: edge.targetHandle, type: 'target' };
  }
  // The target end stays connected -> the user grabbed the SOURCE endpoint.
  if (edge.sourceHandle == null) return null;
  return { nodeId: edge.source, handleId: edge.sourceHandle, type: 'source' };
}

// ── Detach-drag redirect ────────────────────────────────────────────────────
//
// ComfyUI-style detach: grabbing a CONNECTED input port should pull the
// existing edge's endpoint off (React Flow "reconnect"), not start a second
// connection. React Flow's invisible reconnect anchor (`<circle
// class="react-flow__edgeupdater react-flow__edgeupdater-target" r=10>` in
// the edge SVG layer) sits BELOW the node layer, so the port `Handle` DOM
// always wins the hit-test and the anchor can never be grabbed directly.
// The two helpers below fix that by redirecting the connection-starting
// mousedown from the Handle to the matching edge's anchor.
//
// Event bindings verified against the installed libraries
// (@xyflow/react 12.10.1 dist/esm/index.js, @xyflow/system 0.0.75):
//
// - `HandleComponent` binds exactly `onMouseDown: onPointerDown,
//   onTouchStart: onPointerDown` (React synthetic, bubble phase). The mouse
//   branch requires `event.button === 0` and calls
//   `XYHandle.onPointerDown(event.nativeEvent, ...)` — so the ONLY mouse
//   event that starts a connection from a Handle is the synthetic bubble
//   `mousedown`. Intercepting `onMouseDownCapture` on our own <Handle> fires
//   first (the prop reaches the DOM via the component's `...rest` spread);
//   React's synthetic `stopPropagation()` also calls the NATIVE event's
//   `stopPropagation()`, halting it at the React root's capture listener, so
//   the Handle's internal bubble `onMouseDown` never runs. Touch
//   (`onTouchStart`) is deliberately left alone.
//
// - `EdgeUpdateAnchors`/`EdgeAnchor` render the anchor as `<circle
//   onMouseDown=... class="react-flow__edgeupdater
//   react-flow__edgeupdater-<target|source>">`; the handler ignores
//   `event.button !== 0` and calls `XYHandle.onPointerDown` wired with the
//   reconnect callbacks. One synthetic native `mousedown` dispatched on the
//   circle therefore drives the entire native reconnect lifecycle
//   (`onReconnectStart`, original edge unmounts, document-level move/up
//   listeners follow the REAL mouse, `onReconnect`/`onReconnectEnd` on drop).
//
// - The edge group is `<g class="react-flow__edge ..." data-id="<edge.id>"
//   data-testid="rf__edge-<edge.id>">` (`EdgeWrapper`) — the `data-id`
//   attribute is confirmed present, so the anchor is addressable as
//   `.react-flow__edge[data-id="<edgeId>"] .react-flow__edgeupdater-target`.

/**
 * Find the edge a grab on a connected input handle should detach: the LAST
 * edge in the array whose target endpoint is `nodeId`/`handleId`. Later edges
 * render on top in React Flow, so the last match is the one visually grabbed.
 * Returns null when nothing is connected to that input.
 */
export function findDetachableEdge<
  E extends Pick<Edge, 'id' | 'target' | 'targetHandle'>,
>(edges: readonly E[], nodeId: string, handleId: string): E | null {
  for (let i = edges.length - 1; i >= 0; i--) {
    const edge = edges[i];
    if (edge.target === nodeId && edge.targetHandle === handleId) return edge;
  }
  return null;
}

/**
 * The subset of a React synthetic mouse event the redirect needs. Structural
 * so the helper stays DOM-light and unit-testable with a plain stub object.
 */
export interface RedirectableMouseDownEvent {
  clientX: number;
  clientY: number;
  currentTarget: EventTarget | null;
  preventDefault: () => void;
  stopPropagation: () => void;
}

/**
 * Escape a string for use inside a double-quoted CSS attribute selector.
 * Per the CSS string grammar only backslash and the quote character need
 * escaping (edge ids never contain newlines).
 */
const escapeCssString = (value: string): string =>
  value.replace(/\\/g, '\\\\').replace(/"/g, '\\"');

/**
 * Build the synthetic left-button `mousedown` that starts a reconnect.
 *
 * `view` is load-bearing (#219), and the comment that used to sit at the
 * dispatch site — "nothing downstream reads `event.view`" — was wrong. The
 * press BUBBLES into React Flow's canvas, and both d3 gesture recognizers
 * mounted there dereference `view` on every mousedown they see: d3-drag's
 * `mousedowned` calls `nodrag(event.view)` and d3-zoom's calls
 * `dragDisable(event.view)` — the same function, whose first statement is
 * `view.document.documentElement`, with no null check, in a dependency we do
 * not own (d3-drag/src/nodrag.js). A `MouseEvent` constructed without `view`
 * has `view === null`, so omitting it throws exactly the "Cannot read
 * properties of null (reading 'document')" #219 recorded — from inside React
 * Flow, on a press we synthesized ourselves.
 *
 * The window is taken from the anchor's OWN document rather than the ambient
 * global, because that is the realm the element actually lives in.
 *
 * The fallback is what the original omission was really working around.
 * jsdom validates `view` by identity against its own global proxy, and under
 * vitest the copied globals mean NO window object passes — not `window`, not
 * `globalThis`, not `document.defaultView`, which are all the same rejected
 * object there. So the standard init is tried first (that is what a real
 * browser gets, byte for byte what a user's press looks like) and only when
 * the constructor refuses is `view` installed as an own property, which
 * shadows the prototype getter and reads back identically to every consumer.
 * Without that second path the guard would be untestable, which is how it
 * went missing in the first place.
 */
function createReconnectPress(
  doc: Document,
  clientX: number,
  clientY: number,
): MouseEvent {
  const init: MouseEventInit = {
    bubbles: true,
    cancelable: true,
    button: 0,
    buttons: 1,
    clientX,
    clientY,
  };
  // Null for a detached document: no window to name, and `view: null` is the
  // constructor's own default, so that degrades to the old behavior rather
  // than throwing.
  const view = doc.defaultView;
  if (!view) return new MouseEvent('mousedown', init);

  try {
    return new MouseEvent('mousedown', { ...init, view });
  } catch {
    const press = new MouseEvent('mousedown', init);
    Object.defineProperty(press, 'view', { value: view, configurable: true });
    return press;
  }
}

/**
 * Redirect a connection-starting mousedown from a connected input Handle to
 * the given edge's target reconnect anchor.
 *
 * Locates `.react-flow__edge[data-id="<edgeId>"] .react-flow__edgeupdater-target`
 * inside the handle's OWN canvas — `event.currentTarget.closest('.react-flow')`
 * — falling back to the handle's whole document only when no such container
 * exists (real handles always live inside one; the fallback is effectively
 * test-only). Either way the anchor found lives in the SAME document as the
 * handle. If found, dispatches a fresh left-button native `mousedown`
 * carrying the original pointer coordinates on the circle — which starts the
 * full native reconnect lifecycle — then suppresses the original event
 * (`preventDefault` + `stopPropagation`) and returns true.
 *
 * If the anchor cannot be found, returns false WITHOUT touching the event so
 * the Handle keeps today's start-a-new-connection behavior as a graceful
 * fallback.
 */
export function redirectMouseDownToReconnectAnchor(
  event: RedirectableMouseDownEvent,
  edgeId: string,
): boolean {
  const handleEl =
    event.currentTarget instanceof Element ? event.currentTarget : null;
  const doc =
    handleEl?.ownerDocument ??
    (typeof document === 'undefined' ? null : document);
  if (!doc) return false;

  // Scope the lookup to the handle's own canvas. Since #125 only the active
  // tab's canvas is mounted, so the several-hidden-canvases case this was
  // originally written for no longer arises. More than one ReactFlow instance
  // can still be in the document at once (the SubgraphEditorModal mounts its own
  // provider over the main canvas), and edge ids are only unique PER GRAPH
  // (graphs loaded from example files carry fixed ids). A document-wide query
  // could therefore match an identically-named edge in the other instance:
  // querySelector returns the FIRST match in DOM order. The ReactFlow root
  // container carries the class "react-flow" (verified in @xyflow/react
  // 12.10.1 dist/esm/index.js: the ReactFlow component renders
  // `<div data-testid="rf__wrapper" ... className={cc(['react-flow',
  // className, colorModeClassName])}>`), and every real Handle lives
  // inside it. When the handle sits inside a canvas, ONLY that canvas is
  // searched — an anchor is never borrowed from another canvas.
  const scope: ParentNode = handleEl?.closest('.react-flow') ?? doc;

  const anchor = scope.querySelector(
    `.react-flow__edge[data-id="${escapeCssString(edgeId)}"] .react-flow__edgeupdater-target`,
  );
  if (!anchor) return false;

  // The anchor's own handler requires button === 0 and reads the pointer
  // position from the event, so mirror the original press exactly.
  anchor.dispatchEvent(
    createReconnectPress(doc, event.clientX, event.clientY),
  );
  event.preventDefault();
  event.stopPropagation();
  return true;
}
