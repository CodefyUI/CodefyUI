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
 * Redirect a connection-starting mousedown from a connected input Handle to
 * the given edge's target reconnect anchor.
 *
 * Locates `.react-flow__edge[data-id="<edgeId>"] .react-flow__edgeupdater-target`
 * in the handle's own document (resolved from `event.currentTarget`, so the
 * anchor found is guaranteed to live in the SAME document as the handle).
 * If found, dispatches a fresh left-button native `mousedown` carrying the
 * original pointer coordinates on the circle — which starts the full native
 * reconnect lifecycle — then suppresses the original event
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
  const doc =
    event.currentTarget instanceof Element
      ? event.currentTarget.ownerDocument
      : typeof document === 'undefined'
        ? null
        : document;
  if (!doc) return false;

  const anchor = doc.querySelector(
    `.react-flow__edge[data-id="${escapeCssString(edgeId)}"] .react-flow__edgeupdater-target`,
  );
  if (!anchor) return false;

  // The anchor's own handler requires button === 0 and reads the pointer
  // position from the event, so mirror the original press exactly. `view` is
  // intentionally omitted: nothing downstream reads `event.view`
  // (@xyflow/system resolves the document from `event.target`), and jsdom's
  // UIEvent init rejects window objects passed through vitest's copied
  // globals (same reason @testing-library/dom omits it).
  anchor.dispatchEvent(
    new MouseEvent('mousedown', {
      bubbles: true,
      cancelable: true,
      button: 0,
      buttons: 1,
      clientX: event.clientX,
      clientY: event.clientY,
    }),
  );
  event.preventDefault();
  event.stopPropagation();
  return true;
}
