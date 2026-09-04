import { useMemo } from 'react';
import { useTabStore } from '../../store/tabStore';
import { HeatmapModal } from '../shared/HeatmapModal';
import { ScatterModal } from '../shared/ScatterModal';
import { VIZ_VIEWERS } from './vizViewers';

/**
 * The one host for every visualization card's "View full" viewer (core#324).
 *
 * Mounted once at the app root and driven by `tab.vizModalNodeId`, the way
 * `NodeDetailModal` and the preset and layers editors are. The viewer used to
 * be a `useState` flag inside each card, and `onlyRenderVisibleElements`
 * (#162) unmounts a card the moment it leaves the viewport -- a window resize,
 * a browser zoom, Shift+L or Ctrl+Z can all do that while the viewer is up --
 * so the viewer closed under the user's cursor. Lifting the flag into the
 * store alone would have been worse: a stored `true` reopens the dialog on
 * remount with no gesture behind it. So a card only asks for its viewer, and
 * this host, which nothing culls, reads the node's output from the store and
 * renders it.
 */
export function VizViewerModal() {
  const nodeId = useTabStore(
    (s) => s.tabs.find((t) => t.id === s.activeTabId)?.vizModalNodeId ?? null,
  );
  if (nodeId === null) return null;
  return <VizViewerModalBody nodeId={nodeId} />;
}

function VizViewerModalBody({ nodeId }: { nodeId: string }) {
  // The whole active tab, the way `NodeDetailModalBody` reads it: the outer
  // component mounts this one only while that tab names a node.
  const tab = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)!);
  const closeVizModal = useTabStore((s) => s.closeVizModal);
  const node = tab.nodes.find((n) => n.id === nodeId);
  const summaries = tab.outputSummaries[nodeId];
  const runId = tab.lastRunId;

  // Rebuilt only when the node, its outputs or the run change: the scatter
  // viewer keys its fit-to-view and its wheel listener on the identity of the
  // points it is handed, so a fresh array on every render would refit under
  // the user's zoom.
  const spec = useMemo(() => {
    const build = node?.type ? VIZ_VIEWERS[node.type] : undefined;
    return node && build ? build(node, summaries, runId) : null;
  }, [node, summaries, runId]);

  // The store nulls the id when the node is deleted, collapsed into a block
  // or replaced with the document; what remains is a node the canvas is not
  // showing right now (the user stepped into a block) or one with no viewer.
  if (spec === null) return null;
  if (spec.kind === 'scatter') {
    return <ScatterModal isOpen onClose={closeVizModal} {...spec.props} />;
  }
  return <HeatmapModal isOpen onClose={closeVizModal} {...spec.props} />;
}
