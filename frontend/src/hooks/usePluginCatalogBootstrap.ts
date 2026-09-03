import { useEffect } from 'react';
import { usePluginStore } from '../store/pluginStore';

/**
 * Kick off the one-time plugin catalog read.
 *
 * Called from the sidebar SHELL, beside `usePackCatalogBootstrap`, for the
 * same reason that one is: the shell is mounted whatever tab is open and
 * whether or not the sidebar is collapsed, and this catalog is not the
 * sidebar's — it is what tells the whole app which plugins exist, whether the
 * server has a Plugin Center at all, and whether an install started before
 * this page load is still running.
 *
 * It goes through `checkInProgress` rather than `refresh` because the boot
 * read is where the thing that only makes sense at boot happens: adopting an
 * install that outlived the last page, and saying so.
 *
 * A server without the route answers 404, which the store records as
 * `unsupported` and nothing reports — this build talks to older servers too,
 * and a missing feature is not an error the user did anything about.
 *
 * Deliberately returns nothing: consumers read the catalog straight from
 * `usePluginStore`, so nothing subscribes to store slices just to get the
 * fetch started. Idempotent twice over — it reads the LATEST store state
 * rather than closure-captured flags, and `checkInProgress` keeps its own
 * once-per-page flag — so StrictMode's double mount cannot double fetch.
 */
export function usePluginCatalogBootstrap(): void {
  useEffect(() => {
    const state = usePluginStore.getState();
    if (!state.loaded && !state.loading) void state.checkInProgress();
  }, []);
}
