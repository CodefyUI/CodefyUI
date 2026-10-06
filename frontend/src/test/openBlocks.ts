import { useTabStore, type TabState } from '../store/tabStore';

/**
 * Standing inside open blocks, for tests of what reads a run's captures
 * (#621). Only the entered instance ids matter to those reads, so the frames
 * are empty: nothing here builds a block or swaps a canvas.
 */

export type SubgraphFrame = TabState['subgraphStack'][number];

/** One open level, entered through instance `instanceId`. */
export function subgraphFrame(instanceId: string): SubgraphFrame {
  return {
    subgraphId: 'outer',
    instanceId,
    nodes: [],
    edges: [],
    presets: [],
    undoStack: [],
    redoStack: [],
    selectedNodeId: null,
    subgraphs: [],
    segmentGroups: [],
    activeSegment: null,
  };
}

/**
 * Put the active tab inside the given instances, outermost first; no ids puts
 * it back at the top level. A component still mounted re-renders, so call it
 * inside `act`, or unmount first.
 */
export function enterBlocks(...instanceIds: string[]): void {
  const { tabs, activeTabId } = useTabStore.getState();
  useTabStore.setState({
    tabs: tabs.map((t) =>
      t.id === activeTabId ? { ...t, subgraphStack: instanceIds.map(subgraphFrame) } : t,
    ),
  });
}
