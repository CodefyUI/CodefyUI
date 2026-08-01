import type { Viewport } from '@xyflow/react';

/**
 * Where each tab's pan/zoom was when you last looked at it (#125).
 *
 * Before #125 every tab mounted its own `<ReactFlowProvider>` and only the
 * active one was visible, so each tab kept its own viewport for free — at the
 * cost of mounting every tab's canvas, palette and panels at once. Now a
 * single canvas serves every tab, which means the viewport has to be handed
 * over explicitly on each switch: stash the outgoing tab's, restore the
 * incoming tab's.
 *
 * A plain module-level Map rather than store state, deliberately:
 *  - writing it must not notify a single subscriber (it happens mid-switch,
 *    and nothing renders from it);
 *  - it is not persisted, which matches the old behaviour exactly — a page
 *    reload has always re-fit the canvas rather than restoring pan/zoom.
 */
const _viewports = new Map<string, Viewport>();

/** Remember where `tabId` was looking. */
export function rememberViewport(tabId: string, viewport: Viewport): void {
  _viewports.set(tabId, viewport);
}

/** The stored viewport for `tabId`, or `undefined` if it was never viewed. */
export function recallViewport(tabId: string): Viewport | undefined {
  return _viewports.get(tabId);
}

/** Drop a closed tab's viewport so the map cannot grow without bound. */
export function forgetViewport(tabId: string): void {
  _viewports.delete(tabId);
}

/** Clear everything. Exposed for tests, which share module state. */
export function _resetViewportMemory(): void {
  _viewports.clear();
}
