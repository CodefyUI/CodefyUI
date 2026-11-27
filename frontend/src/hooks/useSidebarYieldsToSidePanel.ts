import { useEffect } from 'react';
import { useTabStore } from '../store/tabStore';
import { useUIStore } from '../store/uiStore';
import { SIDE_COLLAPSES_PALETTE_BELOW, useSideDockStore } from '../plugins/sideDock';
import { useWindowWidth } from './useWindowWidth';

/**
 * Collapse the left node panel to its rail while the canvas is squeezed from
 * both sides: a side panel (apiVersion 6) is open, a node is selected (so the
 * node config column is showing), and the window is narrower than
 * SIDE_COLLAPSES_PALETTE_BELOW. At 1280px that is the difference between a
 * 220px canvas and a 500px one.
 *
 * The same rule as the Inspector's: only a collapse made here is undone when
 * the crowding ends, a panel the user collapsed stays collapsed, and a panel
 * the user reopens during the crowding stays open until it ends. The store
 * keeps this apart from the user's stored choice (`sidebarYielded`), so a
 * reload never remembers a collapse the user did not make.
 */
export function useSidebarYieldsToSidePanel(): void {
  const sideOpen = useSideDockStore((s) => s.activeKey !== null);
  const hasSelection = useTabStore(
    (s) => (s.tabs.find((t) => t.id === s.activeTabId)?.selectedNodeId ?? null) !== null,
  );
  const width = useWindowWidth();
  const crowded = sideOpen && hasSelection && width < SIDE_COLLAPSES_PALETTE_BELOW;
  useEffect(() => {
    useUIStore.getState().yieldSidebar(crowded);
  }, [crowded]);
}
