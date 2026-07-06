/** The graph format version this build knows how to WRITE. A loaded graph
 * whose format_version exceeds this opens read-only (ID8). Keep in lockstep
 * with backend app/core/project.py FORMAT_VERSION. */
export const GRAPH_FORMAT_VERSION = 1;

/** True when `formatVersion` (an untrusted `format_version` field read off a
 * loaded or imported graph payload) is newer than this build understands --
 * i.e. the graph was written by a newer CodefyUI and must open read-only
 * (ID8). Shared by every load/import call site so the comparison lives in
 * exactly one place instead of being copy-pasted at each one. */
export function isFormatTooNew(formatVersion: unknown): boolean {
  return typeof formatVersion === 'number' && formatVersion > GRAPH_FORMAT_VERSION;
}
