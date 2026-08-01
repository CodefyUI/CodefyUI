import {
  idbGet,
  idbGetByPrefix,
  idbSetMany,
  idbDeleteMany,
} from '../utils/idb';
// Type-only, so the module graph stays one-directional at runtime
// (tabStore -> tabPersistence -> idb) with no import cycle.
import type { PersistedTab } from './tabStore';

/**
 * Where the tab tree lives on disk since #125: IndexedDB, one record per tab.
 *
 * Before, the whole tree was JSON-stringified into a single localStorage
 * value every 250ms of activity. That had two costs. The visible one was the
 * ~5MB per-origin quota, a hard ceiling on graph size: cross it and every
 * subsequent save threw, so the user's work silently stopped being saved. The
 * quieter one was that a single dirty tab forced every OTHER tab to be walked
 * and re-serialized on every save, because there was only one blob.
 *
 * The record layout fixes both:
 *
 *     <scope>|meta        -> { activeTabId, tabIds }   (order + selection)
 *     <scope>|tab|<id>    -> PersistedTab              (one per tab)
 *
 * `<scope>` is the same project-scoped storage key localStorage used
 * (`codefyui-tabs`, or `codefyui-tabs::<projectDir>`), so the two backends
 * name the same thing and the migration is a straight copy.
 *
 * IndexedDB stores structured clones, so nothing here stringifies; and
 * {@link writeSnapshot} skips any record whose object identity is unchanged
 * since the last durable write, so an idle tab costs nothing at all.
 */

const META_SUFFIX = '|meta';
const TAB_INFIX = '|tab|';

/** Key of the record holding tab order and the active tab, for one scope. */
export function tabMetaKey(scope: string): string {
  return `${scope}${META_SUFFIX}`;
}

/** Key of one tab's record within a scope. */
export function tabRecordKey(scope: string, tabId: string): string {
  return `${scope}${TAB_INFIX}${tabId}`;
}

function tabRecordPrefix(scope: string): string {
  return `${scope}${TAB_INFIX}`;
}

/** What `<scope>|meta` holds. */
export interface PersistedTabsMeta {
  activeTabId: string;
  tabIds: string[];
}

/** A whole scope's worth of persisted state. */
export interface PersistedSnapshot {
  tabs: PersistedTab[];
  activeTabId: string;
}

// ── Durability bookkeeping ───────────────────────────────────────────────────
//
// Per scope, the record objects we have actually made durable. Compared by
// IDENTITY: the store hands back the same `PersistedTab` object for a tab it
// has not touched (see tabStore's record memo), so `written.get(id) === rec`
// means "already on disk, byte for byte". Only committed AFTER the
// transaction completes, so a failed write is retried rather than assumed.
const _written = new Map<string, Map<string, PersistedTab>>();
// Per scope, a signature of the last meta record written, so tab order and
// the active tab are only rewritten when one of them actually moved.
const _writtenMeta = new Map<string, string>();

function metaSignature(activeTabId: string, tabIds: string[]): string {
  return JSON.stringify([activeTabId, tabIds]);
}

/**
 * Persist `records` for `scope`, writing only what changed.
 *
 * Rejects if IndexedDB is unavailable or the transaction fails; the caller
 * decides what to do about that (tabStore falls back to localStorage).
 */
export async function writeSnapshot(
  scope: string,
  records: PersistedTab[],
  activeTabId: string,
): Promise<void> {
  const written = _written.get(scope);
  const entries: Array<[string, unknown]> = [];
  for (const rec of records) {
    if (written?.get(rec.id) === rec) continue;
    entries.push([tabRecordKey(scope, rec.id), rec]);
  }

  const tabIds = records.map((r) => r.id);
  const signature = metaSignature(activeTabId, tabIds);
  const metaChanged = _writtenMeta.get(scope) !== signature;
  if (metaChanged) {
    entries.push([tabMetaKey(scope), { activeTabId, tabIds }]);
  }

  const live = new Set(tabIds);
  const removed = written
    ? [...written.keys()].filter((id) => !live.has(id))
    : [];

  // An idle editor reaches this on every debounce tick. Bail before touching
  // IndexedDB at all rather than opening an empty transaction.
  if (entries.length === 0 && removed.length === 0) return;

  if (entries.length > 0) await idbSetMany(entries);
  if (removed.length > 0) {
    await idbDeleteMany(removed.map((id) => tabRecordKey(scope, id)));
  }

  const next = new Map<string, PersistedTab>();
  for (const rec of records) next.set(rec.id, rec);
  _written.set(scope, next);
  _writtenMeta.set(scope, signature);
}

/**
 * Read a scope back. `null` means "IndexedDB holds nothing usable for this
 * scope" — a first run, or a store that predates #125 — which is the caller's
 * cue to migrate whatever localStorage still has.
 *
 * Rejects only when IndexedDB itself is unreachable. Damaged or partial data
 * resolves `null` (or the readable subset) rather than throwing, because
 * refusing to open the app is a worse answer than starting fresh.
 */
export async function readSnapshot(scope: string): Promise<PersistedSnapshot | null> {
  const meta = await idbGet<PersistedTabsMeta>(tabMetaKey(scope));
  if (!meta || typeof meta !== 'object' || !Array.isArray(meta.tabIds)) return null;
  if (meta.tabIds.length === 0) return null;

  const records = await idbGetByPrefix<PersistedTab>(tabRecordPrefix(scope));
  if (records.length === 0) return null;

  const byId = new Map<string, PersistedTab>();
  for (const rec of records) {
    if (rec && typeof rec === 'object' && typeof rec.id === 'string') {
      byId.set(rec.id, rec);
    }
  }
  // Meta owns the ORDER; a prefix scan comes back in key order, which is not
  // the order the tabs sit in the tab bar.
  const tabs = meta.tabIds
    .map((id) => byId.get(id))
    .filter((rec): rec is PersistedTab => rec !== undefined);
  if (tabs.length === 0) return null;

  const activeTabId = tabs.some((t) => t.id === meta.activeTabId)
    ? meta.activeTabId
    : tabs[0].id;

  // Adopt what we just read as the durability baseline so the first save of
  // the session only writes what the user actually changes. The record
  // objects differ by identity from the ones tabStore will build, so each tab
  // is rewritten once; what this buys is correct deletion tracking for tabs
  // that were closed in an earlier session.
  _written.set(scope, byId);
  _writtenMeta.set(scope, metaSignature(activeTabId, tabs.map((t) => t.id)));

  return { tabs, activeTabId };
}

/**
 * Forget what is durable. Tests share module state and swap the IndexedDB
 * factory between cases; production calls it when the storage scope changes
 * under it (opening a project), where the previous scope's bookkeeping says
 * nothing about the new one.
 */
export function _resetTabPersistenceForTests(): void {
  _written.clear();
  _writtenMeta.clear();
}
