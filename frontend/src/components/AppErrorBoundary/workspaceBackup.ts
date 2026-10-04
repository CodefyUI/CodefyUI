/**
 * The recovery screen's storage work (#555): the autosaved workspace as a
 * `.cduiworkspace` file, and the empty workspace that ends a crash loop.
 *
 * Read from STORAGE, never from the tab store. The screen is up because a
 * render threw, and the tabs in memory are the first suspect: a backup built
 * from them could throw the same way, or carry the same damage into the file.
 * Storage holds what a reload would restore, and when that is what throws,
 * this file is the only way the user's graphs get out of the browser.
 *
 * The file is the one Export Workspace writes for the same tabs, so Import
 * opens it like any other.
 */

import { readSnapshot, writeSnapshot, type PersistedSnapshot } from '../../store/tabPersistence';
import {
  autosaveScope,
  runSettingsOf,
  tabFromPersisted,
  tabHasContent,
  useTabStore,
  type PersistedTab,
  type TabState,
} from '../../store/tabStore';
import { cachedAppVersion } from '../../utils/appVersion';
import { GRAPH_FORMAT_VERSION } from '../../utils/formatVersion';
import { idbAvailable, idbDeleteMany, idbGetKeysByPrefix } from '../../utils/idb';
import {
  buildWorkspaceFile,
  type WorkspaceFile,
  type WorkspaceTabEntry,
} from '../../utils/workspaceFile';

export interface WorkspaceBackup {
  /** Null when no autosaved tab holds a graph that can go in the file. */
  file: WorkspaceFile | null;
  /** Tabs with a graph left out because they are read-only, as Export Workspace leaves them out. */
  skippedReadOnly: number;
  /** Records that could not be turned back into a graph, or into JSON. */
  unreadable: number;
}

/** The records `readAutosavedTabs` found, as stored: nothing about them is trusted. */
export interface AutosavedTabs {
  records: unknown[];
  activeTabId: unknown;
}

/**
 * The records a reload would restore, from the tier hydration would take
 * them from: IndexedDB when it holds this scope -- an empty workspace
 * included -- and otherwise the localStorage copy, which is where autosave
 * writes when IndexedDB is missing or failing.
 *
 * Rejects only when neither tier can be read, with the reason.
 */
export async function readAutosavedTabs(): Promise<AutosavedTabs> {
  const scope = autosaveScope();
  let idbFailure: { reason: unknown } | null = null;
  if (idbAvailable()) {
    try {
      // What hydration calls, with the same side effect: what it reads
      // becomes the baseline the next autosave compares against, so that
      // save rewrites every tab once.
      const snapshot = await readSnapshot(scope);
      if (snapshot !== null) {
        return { records: snapshot.tabs, activeTabId: snapshot.activeTabId };
      }
    } catch (reason) {
      idbFailure = { reason };
    }
  }
  const raw = localStorage.getItem(scope);
  if (raw === null) {
    if (idbFailure !== null) throw idbFailure.reason;
    return { records: [], activeTabId: null };
  }
  // A blob that is not JSON throws here, and the user is told so. One that
  // is JSON but holds no tab list has nothing to restore, as on a reload.
  const data = JSON.parse(raw) as { tabs?: unknown; activeTabId?: unknown } | null;
  return {
    records: Array.isArray(data?.tabs) ? data.tabs : [],
    activeTabId: data?.activeTabId,
  };
}

/**
 * One record as the tab a reload would make of it.
 *
 * `tabFromPersisted` takes the id, and the session-only fields a live tab
 * carries (socket, logs, undo stacks), from its `base`. Nothing below reads
 * those, so the id is all this base holds.
 */
function restoreTab(record: unknown): TabState {
  const persisted = record as PersistedTab;
  return tabFromPersisted(persisted, { id: persisted.id } as TabState);
}

/** `value` when it is text; '' for whatever a damaged record holds instead. */
function text(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

/**
 * One file entry, key for key what Export Workspace writes for a tab
 * (`entryOf` in `utils/exportWorkspace.ts`, which is private to it), through
 * the store's one serializer.
 *
 * Unlike a live tab, a stored record can hold anything in its text fields --
 * a name the tab strip cannot render is one way a workspace crashes every
 * load -- so those are written as text or not at all.
 */
function entryOf(tab: TabState): WorkspaceTabEntry {
  const { nodes, edges, presets, segmentGroups, subgraphs, settings } =
    useTabStore.getState().getSerializedGraphOf(tab);
  const name = text(tab.name);
  return {
    title: name,
    graph: {
      name: name || 'graph',
      description: text(tab.description),
      nodes,
      edges,
      presets,
      segmentGroups,
      subgraphs,
      ...(settings ? { settings } : {}),
      format_version: GRAPH_FORMAT_VERSION,
    },
    run: runSettingsOf(tab),
  };
}

/**
 * Build the backup from what `readAutosavedTabs` found. Every record is
 * handled on its own, so one that cannot be read is counted and left out
 * instead of sinking the rest.
 *
 * No `preferences`: they are not work, and the UI store they would be read
 * from is the kind of in-memory state this file stays away from.
 */
export function buildWorkspaceBackup(saved: AutosavedTabs, now: Date): WorkspaceBackup {
  const ids: string[] = [];
  const tabs: WorkspaceTabEntry[] = [];
  let skippedReadOnly = 0;
  let unreadable = 0;
  for (const record of saved.records) {
    try {
      const tab = restoreTab(record);
      // Empty tabs are left out silently, as Export Workspace leaves them.
      if (!tabHasContent(tab)) continue;
      if (tab.readOnly) {
        skippedReadOnly += 1;
        continue;
      }
      const entry = entryOf(tab);
      // The download writes the whole file as JSON in one go, and a value
      // JSON cannot hold (a cycle, a BigInt: IndexedDB stores both) would
      // sink every tab with it. Found here, it costs this record only.
      JSON.stringify(entry);
      tabs.push(entry);
      ids.push(tab.id);
    } catch {
      unreadable += 1;
    }
  }
  if (tabs.length === 0) return { file: null, skippedReadOnly, unreadable };

  // An index into what is IN the file.
  const { activeTabId } = saved;
  const active = typeof activeTabId === 'string' ? ids.indexOf(activeTabId) : -1;
  const file = buildWorkspaceFile({
    appVersion: cachedAppVersion(),
    exportedAt: now.toISOString(),
    active: active === -1 ? null : active,
    tabs,
  });
  return { file, skippedReadOnly, unreadable };
}

/**
 * `application/octet-stream`, as Export Workspace writes it: a browser has no
 * preferred extension for it, so the `.cduiworkspace` name is kept as given.
 */
const WORKSPACE_MIME = 'application/octet-stream';

/** Blob + anchor, the way Export Workspace downloads. */
export function downloadWorkspaceFile(file: WorkspaceFile, name: string): void {
  const blob = new Blob([JSON.stringify(file, null, 2)], { type: WORKSPACE_MIME });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

export type EmptyWorkspaceResult =
  /** Storage holds an empty workspace now, and a reload opens it. */
  | 'reset'
  /** No copy of the current tabs could be kept, and no complete backup was downloaded: nothing changed. */
  | 'no_copy'
  /** What a reload would restore did not change: do not reload. */
  | 'not_changed';

/** The localStorage form of a workspace with no tabs, as `loadTabs` reads it. */
const EMPTY_WORKSPACE_BLOB = JSON.stringify({ activeTabId: '', tabs: [] });

/** Copies of a scope are kept under `<scope>-aside::<time>`, in both tiers. */
const ASIDE_INFIX = '-aside::';

/** Whether a localStorage blob holds tabs worth a copy. One JSON cannot read might. */
function holdsTabs(blob: string): boolean {
  try {
    const data = JSON.parse(blob) as { tabs?: unknown } | null;
    return Array.isArray(data?.tabs) && data.tabs.length > 0;
  } catch {
    return true;
  }
}

/**
 * The copy a stored key belongs to, `<scope>-aside::<time>`: the key itself
 * in localStorage, `<copy>|...` in IndexedDB. Null for any other key, which
 * is then never deleted.
 */
function copyOf(key: string, prefix: string): { name: string; time: number } | null {
  const match = /^(\d+)(?:\||$)/.exec(key.slice(prefix.length));
  return match === null ? null : { name: `${prefix}${match[1]}`, time: Number(match[1]) };
}

/**
 * Delete whole copies of `scope`, from both tiers: the ones `pick` names
 * when handed every copy's name, oldest first.
 */
async function deleteCopies(scope: string, pick: (copies: string[]) => string[]): Promise<void> {
  const prefix = `${scope}${ASIDE_INFIX}`;
  // Straight through `utils/idb`: `tabPersistence` writes and reads a scope
  // but has no way to list or delete one. Only reached after a copy made by
  // this page has read back, i.e. in a page whose writes land.
  let idbKeys: string[] = [];
  if (idbAvailable()) {
    try {
      idbKeys = await idbGetKeysByPrefix(prefix);
    } catch {
      // A copy left behind costs space, never work.
    }
  }
  const localKeys: string[] = [];
  try {
    for (let i = 0; i < localStorage.length; i += 1) {
      const key = localStorage.key(i);
      if (key !== null && key.startsWith(prefix)) localKeys.push(key);
    }
  } catch {
    // As above.
  }

  const times = new Map<string, number>();
  for (const key of [...idbKeys, ...localKeys]) {
    const copy = copyOf(key, prefix);
    if (copy !== null) times.set(copy.name, copy.time);
  }
  const doomed = new Set(pick([...times.keys()].sort((a, b) => times.get(a)! - times.get(b)!)));
  if (doomed.size === 0) return;
  const isDoomed = (key: string) => {
    const copy = copyOf(key, prefix);
    return copy !== null && doomed.has(copy.name);
  };
  try {
    const keys = idbKeys.filter(isDoomed);
    if (keys.length > 0) await idbDeleteMany(keys);
  } catch {
    // As above.
  }
  try {
    for (const key of localKeys.filter(isDoomed)) localStorage.removeItem(key);
  } catch {
    // As above.
  }
}

/**
 * Whether the scope still holds what was copied from it -- the same tabs in
 * IndexedDB, the same blob in localStorage -- so that the copy only repeats
 * it. A read that fails passes through, and the copy stays.
 */
async function stillAsCopied(
  scope: string,
  current: PersistedSnapshot | null,
  blob: string | null,
): Promise<boolean> {
  if (current !== null) {
    const now = await readSnapshot(scope);
    return (
      now !== null &&
      now.tabs.length === current.tabs.length &&
      now.tabs.every((tab, i) => tab.id === current.tabs[i].id)
    );
  }
  return localStorage.getItem(scope) === blob;
}

/** Empty the tier a reload restores from, and read it back. */
async function writeEmptyWorkspace(
  scope: string,
  idbReadable: boolean,
): Promise<'reset' | 'not_changed'> {
  if (idbReadable) {
    try {
      await writeSnapshot(scope, [], '');
    } catch {
      // Read back below like a write that did not land.
    }
    // A reload restores IndexedDB whenever it holds this scope, so this is
    // the write that counts, and localStorage stays as it was if it failed.
    // A read that fails passes through: whether the write landed is unknown.
    const after = await readSnapshot(scope);
    if (after === null || after.tabs.length > 0) return 'not_changed';
  }
  try {
    localStorage.setItem(scope, EMPTY_WORKSPACE_BLOB);
  } catch {
    // Read back below.
  }
  const restored = await readAutosavedTabs();
  return restored.records.length === 0 ? 'reset' : 'not_changed';
}

/**
 * Replace the autosaved workspace with an empty one, after keeping a copy.
 *
 * For a crash that comes from what is stored: every reload restores it and
 * crashes again, and the way back is to start empty and Import the backup.
 * The current tabs are first copied under `<scope>-aside::<time>`, from the
 * tier a reload restores -- IndexedDB when it holds the scope, where
 * localStorage's blob is only what it held on migration day -- and the copy
 * must read back, or nothing changes. On the localStorage tier, a complete
 * backup file the user downloaded stands in for a copy that does not fit;
 * one that left tabs out does not.
 *
 * Two copies per scope are kept: the oldest, the workspace before the first
 * reset and so the only one holding tabs a backup left out, and the newest.
 * A reset that worked deletes those in between, one that changed nothing
 * takes its own copy back, and one that failed part way deletes nothing.
 *
 * Every write is checked by reading it back, because a write can resolve
 * without writing: `writeSnapshot` does that in a page that does not hold
 * the editing lock (#554).
 */
export async function startEmptyWorkspace(
  now: Date,
  { completeBackup = false }: { completeBackup?: boolean } = {},
): Promise<EmptyWorkspaceResult> {
  const scope = autosaveScope();
  const aside = `${scope}${ASIDE_INFIX}${now.getTime()}`;

  let idbReadable = false;
  let current: PersistedSnapshot | null = null;
  if (idbAvailable()) {
    try {
      current = await readSnapshot(scope);
      idbReadable = true;
    } catch {
      // Unreadable now is unreadable at the next load too, and hydration then
      // restores localStorage's copy: that is the tier to change below.
    }
  }

  let copied = false;
  let blob: string | null = null;
  if (current !== null) {
    if (current.tabs.length > 0) {
      await writeSnapshot(aside, current.tabs, current.activeTabId);
      copied = (await readSnapshot(aside))?.tabs.length === current.tabs.length;
      // A write that resolved without writing: a page without the editing
      // lock, whose empty write would be skipped as well.
      if (!copied) return 'not_changed';
    }
  } else {
    blob = localStorage.getItem(scope);
    if (blob !== null && holdsTabs(blob)) {
      try {
        localStorage.setItem(aside, blob);
      } catch {
        // No room: a workspace over half the quota cannot be copied beside itself.
      }
      copied = localStorage.getItem(aside) === blob;
      if (!copied && !completeBackup) return 'no_copy';
    }
  }

  // An exception passes straight through, deleting nothing: the workspace may
  // be emptied by then, and the copy is what holds it.
  const outcome = await writeEmptyWorkspace(scope, idbReadable);
  if (copied && outcome === 'reset') {
    // All but the oldest of the others, and this one.
    await deleteCopies(scope, (copies) => copies.filter((copy) => copy !== aside).slice(1));
  } else if (copied && (await stillAsCopied(scope, current, blob))) {
    // Nothing changed after all, so the copy only repeats what is there. Any
    // other "not changed" (a tier emptied while a reload restores the other)
    // leaves the copy as the one place those tabs are.
    await deleteCopies(scope, (copies) => copies.filter((copy) => copy === aside));
  }
  return outcome;
}
