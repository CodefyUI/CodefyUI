import { useEffect } from 'react';
import { useReactFlow, useStoreApi } from '@xyflow/react';
import { isTypingTarget } from './useKeyboardShortcuts';

/**
 * Delete removes the selection of the React Flow canvas this hook runs under
 * (#501). Both canvases, the main one and the Model Architecture editor's,
 * pass `deleteKeyCode={null}` and call this instead.
 *
 * WHY NOT REACT FLOW'S OWN BINDING. It keeps a set of the keys it has seen go
 * down and deletes only when that set holds Delete alone. It adds a key under
 * its keydown's `event.key` and removes it under its keyup's, and the two
 * differ once Shift comes up first: `?` goes down and `/` comes up, `L` goes
 * down and `l` comes up. The `?` then stayed in the set, the next Delete read
 * as `?` + Delete and matched nothing, and only the Delete after that worked.
 * Upgrading does not help: @xyflow/react 12.11.6 tracks keys the same way. A
 * handler that looks at the one keydown it is given has no set to go stale.
 *
 * The rules are the binding's own: a Delete with no modifier held (React Flow
 * never acted on Shift+Delete or Ctrl+Delete, and Backspace does nothing),
 * never while the user types in a field, and once per press however long the
 * key is held. The deletion is the binding's too, `deleteElements` on the
 * selected nodes and edges, so each canvas's `onBeforeDelete` still refuses
 * it while a modal is in the way and the change handlers run as before.
 */
export function useDeleteKey(): void {
  const store = useStoreApi();
  const { deleteElements } = useReactFlow();

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Delete' || e.repeat) return;
      if (e.ctrlKey || e.metaKey || e.shiftKey || e.altKey) return;
      if (isTypingTarget(e.target)) return;
      e.preventDefault();
      const { nodes, edges } = store.getState();
      void deleteElements({
        nodes: nodes.filter((node) => node.selected),
        edges: edges.filter((edge) => edge.selected),
      });
      store.setState({ nodesSelectionActive: false });
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [store, deleteElements]);
}
