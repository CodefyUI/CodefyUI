import { useEffect } from 'react';
import { useNodeDefStore } from '../store/nodeDefStore';

/**
 * Kick off the one-time node + preset catalog load.
 *
 * MUST be called from a component that is always mounted. This used to live in
 * the node palette, which was safe while the palette was a single always-visible
 * column. Since #126 the node list is one of four sidebar tabs and mounts only
 * when `sidebarTab === 'nodes'` AND the sidebar is expanded — both persisted —
 * so hanging the bootstrap off it would mean that booting collapsed, or onto
 * any other tab, left the WHOLE app with an empty catalog for the session: no
 * presets, examples resolving against an empty definition map, empty quick
 * search, untyped edges, and the plugin host burning its full 15s
 * `waitForNodeDefinitions` timeout. It is called from the sidebar SHELL, which
 * is mounted in every one of those states.
 *
 * Deliberately returns nothing: consumers read the catalog straight from
 * `useNodeDefStore`, so nothing subscribes to store slices just to get the
 * fetch started.
 *
 * Idempotent. It reads the LATEST store state rather than closure-captured
 * flags, so a second caller — or StrictMode's double mount — cannot double
 * fetch.
 */
export function useNodeDefinitionsBootstrap(): void {
  useEffect(() => {
    const state = useNodeDefStore.getState();
    if (state.definitions.length === 0 && !state.loading) {
      state.fetchDefinitions();
    }
  }, []);
}
