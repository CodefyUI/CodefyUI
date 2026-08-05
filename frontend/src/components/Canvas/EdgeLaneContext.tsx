import { createContext, useContext, useMemo, useRef, type ReactNode } from 'react';
import {
  columnKey,
  computeEdgeLanes,
  type LaneNodeGeometry,
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

/** Only the node fields lane assignment reads. */
export interface LaneNodeInput {
  id: string;
  position: { x: number; y: number };
  measured?: { height?: number | null } | null;
  height?: number | null;
  dragging?: boolean;
}

/**
 * Height to assume before React Flow has measured a node. Over-reporting a band
 * only separates a little more than needed, which is the safe direction, so a
 * generous default costs nothing.
 */
const ASSUMED_NODE_HEIGHT = 120;

function toGeometry(nodes: readonly LaneNodeInput[]): Map<string, LaneNodeGeometry> {
  const geometry = new Map<string, LaneNodeGeometry>();
  for (const node of nodes) {
    const height = node.measured?.height ?? node.height ?? ASSUMED_NODE_HEIGHT;
    geometry.set(node.id, {
      column: columnKey(node.position.x),
      top: node.position.y,
      bottom: node.position.y + height,
    });
  }
  return geometry;
}

export interface EdgeLaneProviderProps {
  edges: readonly LaneEdgeInput[];
  /**
   * Nodes, for grouping lanes by column. Omit and lanes are grouped per node,
   * which leaves two nodes sharing an x drawing their k-th edge on one line.
   */
  nodes?: readonly LaneNodeInput[];
  children: ReactNode;
}

function sameLanes(a: EdgeLaneMap, b: EdgeLaneMap): boolean {
  if (a.size !== b.size) return false;
  for (const [id, lane] of a) {
    const other = b.get(id);
    if (
      !other ||
      other.outSlot !== lane.outSlot ||
      other.outCount !== lane.outCount ||
      other.inSlot !== lane.inSlot ||
      other.inCount !== lane.inCount
    ) {
      return false;
    }
  }
  return true;
}

export function EdgeLaneProvider({ edges, nodes, children }: EdgeLaneProviderProps) {
  // Rebuilt on every render, which during a drag means every frame - but it is one
  // pass over the nodes and edges, and it is what decides whether the expensive
  // part runs at all.
  const frozen = useRef<string | null>(null);
  const signature = useMemo(() => {
    // Lanes are frozen for the duration of a drag. Recomputing them mid-drag is
    // what a position-dependent pass costs: the grouping genuinely changes as a
    // node crosses other columns, every change re-renders every edge, and on a
    // 300-node canvas that showed up as +5.6ms on the p95 frame. Nothing is lost
    // by waiting - the wires are all moving anyway, and a transient overlap under
    // the cursor is not what anyone is looking at. The moment the drag ends the
    // signature moves again and the lanes settle.
    if (nodes && frozen.current !== null && nodes.some((n) => n.dragging)) {
      return frozen.current;
    }
    let sig = edgeLaneSignature(edges);
    if (nodes) {
      for (const node of nodes) {
        const height = node.measured?.height ?? node.height ?? ASSUMED_NODE_HEIGHT;
        // Bucketed, so a drag re-decides the grouping once per bucket crossed
        // rather than once per pixel. Separators are escaped control characters
        // so two different node lists cannot produce one signature.
        sig += `${node.id}\u0001${columnKey(node.position.x)}\u0001`;
        sig += `${columnKey(node.position.y)}\u0001${columnKey(node.position.y + height)}\u0002`;
      }
    }
    frozen.current = sig;
    return sig;
  }, [edges, nodes]);

  const previous = useRef<EdgeLaneMap>(EMPTY_LANE_MAP);
  // Keyed on the signature, not on the arrays: selecting an edge, recolouring it,
  // or any `applyEdgeChanges` pass produces fresh objects but the same signature.
  // Dragging changes a node's x continuously, so the signature does move - but
  // only once per column bucket crossed, and the lanes it produces almost never
  // change, because the dragged node is nearly always alone in its bucket. When
  // the result is unchanged the previous map is handed back by identity, so the
  // context value is stable and not one edge re-renders. Edges re-render only when
  // a lane genuinely changed, which is exactly when their route changed too.
  const lanes = useMemo(() => {
    const next = computeEdgeLanes(edges, nodes ? toGeometry(nodes) : undefined);
    if (sameLanes(previous.current, next)) return previous.current;
    previous.current = next;
    return next;
  }, [signature]); // eslint-disable-line react-hooks/exhaustive-deps

  return <EdgeLaneContext.Provider value={lanes}>{children}</EdgeLaneContext.Provider>;
}

/** Lane of one edge; `SOLO_LANE` when the edge has no siblings or no provider. */
export function useEdgeLane(id: string): EdgeLane {
  return useContext(EdgeLaneContext).get(id) ?? SOLO_LANE;
}
