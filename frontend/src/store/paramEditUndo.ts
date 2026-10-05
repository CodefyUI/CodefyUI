/**
 * When a param edit opens a new undo step, and when it joins the last one.
 *
 * `updateNodeParams` used to push no snapshot at all, so Ctrl+Z after typing
 * a value restored the frame of the structural action BEFORE it: connect an
 * edge, change a param, press Ctrl+Z, and the edge went with the value. A
 * snapshot per keystroke is no better -- typing `0.001` would take five
 * presses to undo. So a RUN of edits is one step. An edit continues the run,
 * pushing nothing, only while all of these hold:
 *
 * - Same tab and same NODE. The node rather than the param, because
 *   TensorGridEditor writes `weights` right after a `kernel_size` edit to
 *   reshape the grid. As a step of its own, that write would be the one Ctrl+Z
 *   undoes, the reshape would run again, and the key would look dead.
 * - The tab's undo stack still has, on top, the very snapshot the run pushed.
 *   Any other action that pushes onto the stack or swaps it (an edge, a drag,
 *   redo, entering a block or leaving an edited one) changes that top, so it
 *   ends the run without having to know this rule exists.
 * - The redo stack is empty. The top alone cannot see an undo: edit, connect,
 *   undo, and the top is the run's own snapshot again, with the connect in
 *   redo. Continuing would push nothing, so redo would survive the next edit
 *   and put the old value back over it.
 * - The previous edit was at most `PARAM_EDIT_IDLE_MS` ago.
 *
 * Module state rather than store state: the mark is not part of the document,
 * means nothing after a reload, and writing it into the store would notify
 * every subscriber on every keystroke for no visible change.
 */

/** A pause longer than this (ms) ends a run. Each edit restarts the wait. */
export const PARAM_EDIT_IDLE_MS = 1000;

/** The two stacks of a tab, as the rule reads them. */
export interface ParamEditHistory {
  undoStack: readonly unknown[];
  redoStack: readonly unknown[];
}

/** The last param edit: where it landed and the undo-stack top it left. */
interface ParamEditMark {
  tabId: string;
  nodeId: string;
  /** An undo frame, compared by identity and never read. */
  top: unknown;
  at: number;
}

let lastEdit: ParamEditMark | null = null;

function topOf(undoStack: readonly unknown[]): unknown {
  return undoStack[undoStack.length - 1];
}

/**
 * Does an edit to `nodeId` at `now` join the previous edit's undo step?
 * `history` is the tab's as it stands before this edit.
 */
export function paramEditContinues(
  tabId: string,
  nodeId: string,
  history: ParamEditHistory,
  now: number,
): boolean {
  if (lastEdit === null || history.undoStack.length === 0) return false;
  return (
    lastEdit.tabId === tabId &&
    lastEdit.nodeId === nodeId &&
    lastEdit.top === topOf(history.undoStack) &&
    history.redoStack.length === 0 &&
    now - lastEdit.at <= PARAM_EDIT_IDLE_MS
  );
}

/**
 * Record an edit for the next one to be compared against. Call it AFTER any
 * snapshot the edit pushed, with the stack that snapshot landed on.
 */
export function markParamEdit(
  tabId: string,
  nodeId: string,
  undoStack: readonly unknown[],
  now: number,
): void {
  lastEdit = { tabId, nodeId, top: topOf(undoStack), at: now };
}

/**
 * An edit that changed nothing. Inside a run it restarts the wait, so the run
 * survives it: typing "-0.5" can commit -0 at "-0" and the same -0 again at
 * "-0.", and only the pause before the 5 should count. Outside one it leaves
 * the mark alone: a no-op neither starts a run nor moves one to another node.
 */
export function paramEditKeepAlive(
  tabId: string,
  nodeId: string,
  history: ParamEditHistory,
  now: number,
): void {
  if (lastEdit !== null && paramEditContinues(tabId, nodeId, history, now)) {
    lastEdit = { ...lastEdit, at: now };
  }
}

/** Forget the last edit. For tests: the mark outlives a store reset. */
export function resetParamEditMark(): void {
  lastEdit = null;
}
