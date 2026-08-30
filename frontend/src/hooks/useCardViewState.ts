import { useCallback } from 'react';
import { useTabStore } from '../store/tabStore';

/**
 * `useState`-shaped access to one card's `expanded` flag, kept in the tab
 * store instead of inside the card (core#324).
 *
 * `onlyRenderVisibleElements` (#321) unmounts a card the moment it scrolls
 * out of the viewport, so a `useState` flag there is forgotten every time the
 * learner pans away — they open a heatmap, look at the other end of the
 * graph, come back, and it is closed. The store outlives the card.
 *
 * Deliberately NOT `node.data`: this is how one person is looking at the
 * graph, not part of the graph, and it must never be serialized into a saved
 * file. The store drops it with the node and with the document.
 */
export function useCardExpanded(nodeId: string): [boolean, (value: boolean) => void] {
  const expanded = useTabStore((s) => {
    const tab = s.tabs.find((t) => t.id === s.activeTabId);
    return tab?.cardViewState?.[nodeId]?.expanded ?? false;
  });
  const setExpanded = useCallback(
    (value: boolean) => useTabStore.getState().setCardViewState(nodeId, { expanded: value }),
    [nodeId],
  );
  return [expanded, setExpanded];
}
