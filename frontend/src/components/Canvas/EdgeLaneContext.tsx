import { createContext, useContext, useMemo, type ReactNode } from 'react';
import {
  computeEdgeLanes,
  edgeLaneSignature,
  EMPTY_LANE_MAP,
  SOLO_LANE,
  type EdgeLane,
  type EdgeLaneMap,
  type LaneEdgeInput,
} from '../../utils/edgeLanes';

/**
 * Lane map for the edges of the canvas currently being rendered.
 *
 * An edge component only ever receives its own geometry, so it cannot know how
 * many siblings share its route. This context is the one place that sees the whole
 * edge list; `SmartDataEdge` and `TriggerEdge` read their slot out of it.
 *
 * The default is an empty map, so an edge rendered without a provider (unit tests,
 * or any future canvas that forgets to wrap) falls back to `SOLO_LANE` and routes
 * exactly as it did before lanes existed.
 */
const EdgeLaneContext = createContext<EdgeLaneMap>(EMPTY_LANE_MAP);

export interface EdgeLaneProviderProps {
  edges: readonly LaneEdgeInput[];
  children: ReactNode;
}

export function EdgeLaneProvider({ edges, children }: EdgeLaneProviderProps) {
  const signature = useMemo(() => edgeLaneSignature(edges), [edges]);
  // Keyed on the topology signature rather than on `edges` on purpose: selecting an
  // edge, recolouring it, or any `applyEdgeChanges` pass produces a fresh array but
  // the same signature, so the lane map keeps its identity and not one edge
  // re-renders. Node drags never touch the edge array at all, so a drag costs zero
  // here. The map is only rebuilt when an edge is actually added, removed or
  // rewired - which is also the only time a lane can change.
  const lanes = useMemo(() => computeEdgeLanes(edges), [signature]); // eslint-disable-line react-hooks/exhaustive-deps
  return <EdgeLaneContext.Provider value={lanes}>{children}</EdgeLaneContext.Provider>;
}

/** Lane of one edge; `SOLO_LANE` when the edge has no siblings or no provider. */
export function useEdgeLane(id: string): EdgeLane {
  return useContext(EdgeLaneContext).get(id) ?? SOLO_LANE;
}
