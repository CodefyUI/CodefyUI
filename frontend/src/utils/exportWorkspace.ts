/**
 * Exporting every open tab as one `.cduiworkspace` file.
 *
 * This is the half of the format that reads the stores; the format itself is
 * `workspaceFile.ts`, which knows nothing about them.
 */

import { useI18n } from '../i18n';
import { runSettingsOf, tabHasContent, useTabStore, type TabState } from '../store/tabStore';
import { useToastStore } from '../store/toastStore';
import { useUIStore } from '../store/uiStore';
import { cachedAppVersion } from './appVersion';
import { GRAPH_FORMAT_VERSION } from './formatVersion';
import {
  buildWorkspaceFile,
  workspaceFileName,
  type WorkspaceFile,
  type WorkspaceTabEntry,
} from './workspaceFile';

/**
 * `application/octet-stream`, not `application/json`: a browser has no
 * preferred extension for it, so the `download` name -- which has to end in
 * `.cduiworkspace` -- is kept exactly as given.
 */
const WORKSPACE_MIME = 'application/octet-stream';

export interface CollectedWorkspace {
  /** Null when no tab is exportable. */
  file: WorkspaceFile | null;
  /** Tabs that hold a graph and were left out only because they are read-only. */
  skippedReadOnly: number;
}

/**
 * Transient tabs are plugin scratch that is not persisted today either. A
 * read-only tab is plugin-owned reference material, or a graph from a newer
 * format this build only partly understands -- re-serializing that would drop
 * fields and stamp a false `format_version`.
 */
function isExportable(tab: TabState): boolean {
  return !tab.transient && !tab.readOnly && tabHasContent(tab);
}

function entryOf(tab: TabState): WorkspaceTabEntry {
  // The ONE serializer: it strips SECRET param values, flushes an open
  // subgraph editor and leaves embedded node definitions out. `TabState` and
  // `PersistedTab` are never serialized directly -- both carry runtime fields,
  // and a live tab holds typed secrets.
  const { nodes, edges, presets, segmentGroups, subgraphs, settings } =
    useTabStore.getState().getSerializedGraphOf(tab);
  return {
    title: tab.name,
    // Key for key what `Toolbar.handleExportJson` writes, plus format_version.
    graph: {
      name: tab.name || 'graph',
      description: tab.description ?? '',
      nodes,
      edges,
      presets,
      segmentGroups,
      subgraphs,
      ...(settings ? { settings } : {}),
      format_version: GRAPH_FORMAT_VERSION,
    },
    // Lifted by the store, not listed here: a run setting added to
    // `TabRunSettings` then fails to compile there instead of going unwritten.
    run: runSettingsOf(tab),
  };
}

/** Build the file from the live stores. `exportWorkspace` is the UI's door. */
export function collectWorkspaceFile(now: Date = new Date()): CollectedWorkspace {
  const { tabs, activeTabId } = useTabStore.getState();
  const skippedReadOnly = tabs.filter(
    (tab) => !tab.transient && tab.readOnly && tabHasContent(tab),
  ).length;
  const exportable = tabs.filter(isExportable);
  if (exportable.length === 0) return { file: null, skippedReadOnly };

  // Re-indexed after skipping: an index into what is IN the file.
  const active = exportable.findIndex((tab) => tab.id === activeTabId);
  const ui = useUIStore.getState();
  const file = buildWorkspaceFile({
    appVersion: cachedAppVersion(),
    exportedAt: now.toISOString(),
    active: active === -1 ? null : active,
    tabs: exportable.map(entryOf),
    // Exactly these six. The Settings device, the sidebar's layout and
    // anything a plugin stored belong to this machine and stay on it.
    preferences: {
      locale: useI18n.getState().locale,
      fontSize: ui.fontSize,
      edgeStyle: ui.edgeStyle,
      gridSnap: ui.gridSnapEnabled,
      tooltips: ui.tooltipsEnabled,
      beginnerMode: ui.beginnerMode,
    },
  });
  return { file, skippedReadOnly };
}

/** The Export menu's Workspace item: toast, or download, and nothing else. */
export function exportWorkspace(): void {
  const t = useI18n.getState().t;
  const addToast = useToastStore.getState().addToast;
  const now = new Date();
  const { file, skippedReadOnly } = collectWorkspaceFile(now);
  if (file === null) {
    addToast(t('workspace.export.empty'), 'warning');
    return;
  }
  // Blob + anchor, the way `Toolbar.handleExportJson` downloads.
  const blob = new Blob([JSON.stringify(file, null, 2)], { type: WORKSPACE_MIME });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = workspaceFileName(now);
  a.click();
  URL.revokeObjectURL(url);
  // No success toast: Export JSON has none, and the browser shows the download.
  if (skippedReadOnly > 0) {
    addToast(t('workspace.export.skippedReadOnly', { count: skippedReadOnly }), 'info');
  }
}
