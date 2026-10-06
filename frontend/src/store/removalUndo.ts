/**
 * When a removal opens a new undo step, and when it joins the one before it.
 *
 * React Flow reports one deletion in two calls. `deleteElements`, which the
 * Delete key goes through (`useDeleteKey`), hands the wires to
 * `onEdgesChange` and then the nodes to `onNodesChange`, one straight after
 * the other, and both reducers pushed an undo snapshot. The first Ctrl+Z put
 * the node back with none of its wires, and only a second one brought them
 * back: a student who stopped after one undo was left with an unwired graph.
 *
 * So a removal that pushes leaves a mark, and a later removal joins its step,
 * pushing nothing, only while all of these hold:
 *
 * - Same synchronous run. `deleteElements` makes its two calls with nothing
 *   awaited between them, the mark is dropped at the next microtask
 *   checkpoint, and every key press is a task of its own: the two calls of
 *   one deletion are one step, and two presses of Delete are two.
 * - Same tab.
 * - The tab's undo stack still has, on top, the very snapshot that removal
 *   pushed, and the redo stack is empty. Anything that pushed or undid in
 *   between ends the step without having to know this rule exists.
 *
 * The order of the two calls does not matter: whichever comes first pushes
 * the graph as it was before the deletion, which is what undo puts back.
 *
 * Module state rather than store state, for the reason `paramEditUndo.ts`
 * gives: the mark is not part of the document.
 */

/** The two stacks of a tab, as the rule reads them. */
interface RemovalHistory {
  undoStack: readonly unknown[];
  redoStack: readonly unknown[];
}

/** What the rule reads from the tab store, and the one action it calls. */
interface RemovalStore {
  activeTabId: string;
  getTab: (tabId: string) => RemovalHistory | undefined;
  pushUndoSnapshotFor: (tabId: string) => void;
}

/** The last removal that pushed: where it landed and the undo-stack top it left. */
interface RemovalMark {
  tabId: string;
  /** An undo frame, compared by identity and never read. */
  top: unknown;
}

let lastRemoval: RemovalMark | null = null;

function topOf(undoStack: readonly unknown[]): unknown {
  return undoStack[undoStack.length - 1];
}

/**
 * Push the undo snapshot for a removal on the active tab, unless the removal
 * belongs to a deletion that has pushed one already. Call it before the
 * removal is applied.
 */
export function pushRemovalStep(get: () => RemovalStore): void {
  const tabId = get().activeTabId;
  const history = get().getTab(tabId);
  if (!history) return;
  if (
    lastRemoval !== null &&
    lastRemoval.tabId === tabId &&
    history.undoStack.length > 0 &&
    lastRemoval.top === topOf(history.undoStack) &&
    history.redoStack.length === 0
  ) {
    return;
  }
  get().pushUndoSnapshotFor(tabId);
  const mark: RemovalMark = { tabId, top: topOf(get().getTab(tabId)?.undoStack ?? []) };
  lastRemoval = mark;
  queueMicrotask(() => {
    // A later removal's mark is not this one's to drop.
    if (lastRemoval === mark) lastRemoval = null;
  });
}
