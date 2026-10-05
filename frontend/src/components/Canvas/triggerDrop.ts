import type { Connection, Edge, FinalConnectionState } from '@xyflow/react';

/**
 * The trigger connection a released Start-node wire makes when React Flow
 * made none, or null.
 *
 * While a trigger is dragged, every card that can take one glows as a drop
 * target (BaseNode's `.triggerDropTarget`), but React Flow connects only when
 * the wire is released on or near a handle, and a card's `__trigger` is a
 * 13 px diamond at its top-left corner: a few screen pixels at the zoom a
 * fitted graph opens at. For a trigger released anywhere else on the card
 * React Flow connects nothing, so this finds the card under the release point.
 *
 * Null when:
 * - the wire is not a Start node's trigger, or React Flow already connected
 *   it (`isValid`);
 * - the pointer is not over a card, or is over the wire's own Start node;
 * - the card has no `__trigger` handle to land on (a Start node, a note);
 * - this Start node already triggers that card. The store's `onConnect`
 *   appends without checking, so a second drop would stack a duplicate wire
 *   under the first.
 *
 * `doc` is the document the canvas lives in; tests pass their own.
 */
export function triggerDropConnection(
  event: MouseEvent | TouchEvent,
  state: FinalConnectionState,
  edges: readonly Pick<Edge, 'source' | 'target' | 'targetHandle'>[],
  doc: Document = document,
): Connection | null {
  const from = state.fromHandle;
  if (from?.type !== 'source' || from.id !== 'trigger' || state.isValid === true) return null;

  // A touchend lists no current touches; the lifted finger is in changedTouches.
  const point = 'changedTouches' in event ? event.changedTouches[0] : event;
  if (!point) return null;

  const card = doc.elementFromPoint(point.clientX, point.clientY)?.closest('.react-flow__node');
  const target = card?.getAttribute('data-id');
  if (!card || !target || target === from.nodeId) return null;
  if (!card.querySelector('.react-flow__handle[data-handleid="__trigger"]')) return null;
  if (
    edges.some(
      (e) => e.source === from.nodeId && e.target === target && e.targetHandle === '__trigger',
    )
  ) {
    return null;
  }

  return { source: from.nodeId, sourceHandle: 'trigger', target, targetHandle: '__trigger' };
}
