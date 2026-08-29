import { useEffect } from 'react';
import { usePackStore } from '../store/packStore';

/**
 * Kick off the one-time pack catalog read.
 *
 * MUST be called from a component that is always mounted, for the same reason
 * `useNodeDefinitionsBootstrap` must: the catalog decides which LLM nodes are
 * usable and which params are greyed out, so a page booted onto any sidebar
 * tab — or with the sidebar collapsed — still needs it. It is called from the
 * sidebar SHELL, which is mounted in all of those states.
 *
 * It goes through `checkInProgress` rather than `refresh` because the boot
 * read is where two things that only make sense at boot happen: adopting an
 * install that outlived the last page load, and reporting a restart-mode
 * install that finished while this page did not exist.
 *
 * Deliberately returns nothing: consumers read the catalog straight from
 * `usePackStore` (or, for node components, through `utils/packAvailability`),
 * so nothing subscribes to store slices just to get the fetch started.
 *
 * Idempotent twice over — it reads the LATEST store state rather than
 * closure-captured flags, and `checkInProgress` keeps its own once-per-page
 * flag — so StrictMode's double mount cannot double fetch.
 */
export function usePackCatalogBootstrap(): void {
  useEffect(() => {
    const state = usePackStore.getState();
    if (!state.loaded && !state.loading) void state.checkInProgress();
  }, []);
}
