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
