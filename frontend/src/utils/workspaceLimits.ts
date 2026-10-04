/**
 * Limits on opening graphs as tabs.
 *
 * Declared once, here, so the callers that draw on them -- the plugin API's
 * `workspace.openGraphs` (`plugins/api.ts`), the `.cduiworkspace` import and
 * the graph-file import (`utils/importGraphFile.ts`) -- cannot drift apart.
 * They do NOT all apply to every caller: each constant below names who
 * enforces it.
 */

/**
 * Serialized JSON one graph may carry. Enforced by the plugin API ALONE.
 *
 * The workspace importer applies no per-graph cap BY DESIGN: its export end
 * measures nothing, so a tab that exported must import -- a cap here would
 * refuse a graph this same build had just written, and an image note embedding
 * its PNG as a data URL is how one tab passes 8 MiB. The file cap below is
 * what bounds that input. Do not "restore" this one to the importer.
 */
export const MAX_WORKSPACE_GRAPH_BYTES = 8 * 1024 * 1024;
/**
 * How many tabs the editor will hold before the callers start refusing.
 * Enforced by ALL THREE -- the workspace importer counting the lone empty tab
 * it is about to close as already gone, the plugin API counting every open
 * tab, and the graph-file import only when the graph needs a tab of its own
 * (an empty active tab is filled instead, which adds none).
 */
export const MAX_WORKSPACE_TABS = 32;
/**
 * A `.cduiworkspace` file larger than this is refused rather than imported.
 * Enforced by the `importFile` router ALONE; the plugin API is handed graphs,
 * never a file.
 */
export const MAX_WORKSPACE_FILE_BYTES = 64 * 1024 * 1024;
