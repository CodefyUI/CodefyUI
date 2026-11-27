import { useEffect } from 'react';
import { fetchHealth } from '../api/rest';
import { useProjectStore } from '../store/projectStore';

/** At most one health read per this many milliseconds. */
export const DRIFT_CHECK_INTERVAL_MS = 5_000;

let lastCheck = Number.NEGATIVE_INFINITY;

/** Test-only: forget when the last check ran. */
export function _resetDriftCheckForTesting(): void {
  lastCheck = Number.NEGATIVE_INFINITY;
}

/**
 * Re-read `/api/health` when the window regains focus or becomes visible,
 * and mark the page stale when the server's project is no longer the one
 * this page loaded with (phase 4A).
 *
 * Another window may have switched the server while this one sat in the
 * background. The execution socket would hear the switch, but an editor
 * opens it only on its first run, so an idle window learns it here instead,
 * the moment the user comes back to it. The 409 `project_changed` on a
 * write stays as the backstop. Throttled, because focus and visibility
 * events arrive in bursts.
 */
export async function checkProjectDrift(now: number = Date.now()): Promise<void> {
  const state = useProjectStore.getState();
  if (!state.loaded || state.stale) return;
  if (now - lastCheck < DRIFT_CHECK_INTERVAL_MS) return;
  lastCheck = now;
  let project: string | null;
  try {
    project = (await fetchHealth()).project;
  } catch {
    return; // unreachable: the connection banner says so, not this one
  }
  const loaded = useProjectStore.getState();
  if (!loaded.stale && project !== loaded.projectDir) loaded.markStale();
}

export function useProjectDriftCheck(): void {
  useEffect(() => {
    const onFocus = () => void checkProjectDrift();
    const onVisible = () => {
      if (document.visibilityState === 'visible') void checkProjectDrift();
    };
    window.addEventListener('focus', onFocus);
    document.addEventListener('visibilitychange', onVisible);
    return () => {
      window.removeEventListener('focus', onFocus);
      document.removeEventListener('visibilitychange', onVisible);
    };
  }, []);
}
