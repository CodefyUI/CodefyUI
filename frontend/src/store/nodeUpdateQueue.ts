import { useTabStore, type PendingNodeUpdates } from './tabStore';
import type { NodeData, NodeProgress } from '../types';

/**
 * Frame-coalesced buffer for execution updates arriving over the WebSocket
 * (#125).
 *
 * The problem it solves: every `node_status` frame used to be its own store
 * commit, and each commit rebuilt the whole `nodes` array (`nodes.map(...)`
 * to touch a single node). React Flow re-diffs all N nodes whenever that
 * array reference changes, so a training run streaming status and progress
 * for a 300-node graph spent most of a frame re-diffing nodes that had not
 * changed. The events themselves are cheap; the fan-out was not.
 *
 * The fix: frames only ever change what the user SEES once per paint, so
 * buffer the per-node patches and commit them together on the next
 * animation frame — one store commit, one nodes-array rebuild, regardless of
 * how many events arrived in between. Nodes with no patch keep their exact
 * object identity, so React Flow's reconciliation stays proportional to what
 * actually changed.
 *
 * Ordering is preserved where it matters: a later status overwrites an
 * earlier status, a later progress overwrites an earlier progress, and the
 * two never overwrite each other (they write disjoint fields). That is
 * exactly what applying every event in arrival order would have produced,
 * because the intermediate values were never painted.
 */

/** Per-node patch accumulated between flushes. */
export interface PendingNodePatch {
  /** Latest `node_status` for this node — status and its (optional) error. */
  status?: { executionStatus: NodeData['executionStatus']; error?: string };
  /** Latest progress payload for this node. */
  progress?: NodeProgress;
}

// tabId -> nodeId -> patch. Null when nothing is pending, so the common
// "flush with nothing to do" path allocates nothing.
let _pending: PendingNodeUpdates | null = null;

// Handle of the scheduled flush, plus which scheduler produced it — rAF and
// setTimeout hand back ids from different spaces and must be cancelled with
// their own canceller.
let _handle: number | null = null;
let _handleIsRaf = false;

function _hasRaf(): boolean {
  return typeof requestAnimationFrame === 'function';
}

function _cancelScheduled(): void {
  if (_handle === null) return;
  if (_handleIsRaf) {
    // A test can strip rAF between scheduling and cancelling; dropping the
    // handle is still correct because the flush re-checks `_pending`.
    if (typeof cancelAnimationFrame === 'function') cancelAnimationFrame(_handle);
  } else {
    clearTimeout(_handle);
  }
  _handle = null;
}

function _schedule(): void {
  if (_handle !== null) return;
  if (_hasRaf()) {
    _handleIsRaf = true;
    _handle = requestAnimationFrame(() => {
      _handle = null;
      flushTabNodeUpdates();
    });
  } else {
    // Environments with no rAF AT ALL: jsdom without `pretendToBeVisual`, a
    // worker. A macrotask still coalesces a burst arriving in the same tick,
    // which is the point.
    //
    // Note what this branch is NOT for. A hidden, backgrounded or occluded
    // document still HAS `requestAnimationFrame` — it simply never calls it
    // back — so the branch above is taken and the flush waits, indefinitely,
    // until the document is painted again. That is deliberate: there is no
    // reason to rebuild a nodes array for pixels nobody is looking at, and
    // the buffer is bounded (one patch per node) however long the wait runs.
    // The consequence to know about is that node badges and progress values
    // FREEZE in a background tab while logs, which are written synchronously,
    // keep arriving; everything catches up in one commit on the next frame.
    _handleIsRaf = false;
    _handle = setTimeout(() => {
      _handle = null;
      flushTabNodeUpdates();
    }, 0) as unknown as number;
  }
}

function _patchFor(tabId: string, nodeId: string): PendingNodePatch {
  if (_pending === null) _pending = new Map();
  let forTab = _pending.get(tabId);
  if (!forTab) {
    forTab = new Map();
    _pending.set(tabId, forTab);
  }
  let patch = forTab.get(nodeId);
  if (!patch) {
    patch = {};
    forTab.set(nodeId, patch);
  }
  _schedule();
  return patch;
}

/** Buffer a node's execution status; applied on the next frame. */
export function queueTabNodeStatus(
  tabId: string,
  nodeId: string,
  status: NodeData['executionStatus'],
  error?: string,
): void {
  _patchFor(tabId, nodeId).status = { executionStatus: status, error };
}

/** Buffer a node's progress payload; applied on the next frame. */
export function queueTabNodeProgress(
  tabId: string,
  nodeId: string,
  progress: NodeProgress,
): void {
  _patchFor(tabId, nodeId).progress = progress;
}

/**
 * Apply everything buffered right now and cancel the pending frame.
 *
 * Called by the scheduler itself, and directly wherever a caller needs the
 * store to be current before reading it.
 */
export function flushTabNodeUpdates(): void {
  _cancelScheduled();
  const updates = _pending;
  _pending = null;
  if (updates === null || updates.size === 0) return;
  useTabStore.getState().applyTabNodeUpdates(updates);
}

/**
 * Drop buffered updates without applying them — for one tab, or all of them.
 *
 * Used when a tab is about to have its execution state reset (a new run
 * clears every node back to idle): a patch left over from the previous run
 * would otherwise land one frame later and paint a stale status onto the
 * fresh run.
 */
export function discardTabNodeUpdates(tabId?: string): void {
  if (tabId === undefined) {
    _pending = null;
    _cancelScheduled();
    return;
  }
  _pending?.delete(tabId);
  if (_pending && _pending.size === 0) {
    _pending = null;
    _cancelScheduled();
  }
}

/** Number of buffered node patches across all tabs. Exposed for tests. */
export function pendingTabNodeUpdateCount(): number {
  if (_pending === null) return 0;
  let total = 0;
  for (const forTab of _pending.values()) total += forTab.size;
  return total;
}
