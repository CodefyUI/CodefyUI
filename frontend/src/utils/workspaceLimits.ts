/**
 * Limits on opening graphs as tabs.
 *
 * Two callers share them -- the plugin API's `workspace.openGraphs` and the
 * `.cduiworkspace` importer -- so they are declared once, here, and the two
 * cannot drift apart.
 */

/** Serialized JSON one graph may carry. */
export const MAX_WORKSPACE_GRAPH_BYTES = 8 * 1024 * 1024;
/** How many tabs the editor will hold before either caller starts refusing. */
export const MAX_WORKSPACE_TABS = 32;
/** A `.cduiworkspace` file larger than this is refused rather than imported. */
export const MAX_WORKSPACE_FILE_BYTES = 64 * 1024 * 1024;
