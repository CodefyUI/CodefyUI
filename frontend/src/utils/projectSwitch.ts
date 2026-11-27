/**
 * The editor's side of switching projects (phase 4A).
 *
 * A switch changes every path the server resolves, so the page reloads
 * after one: every store, socket and cache then starts on the new project,
 * and the tabs are the new project's own (they are stored per project).
 * Before asking the server, the tabs with unsaved changes are listed and the
 * user confirms; they are not lost (they stay in this project's tabs and
 * come back when it is reopened), but the user should know they are leaving
 * them.
 */

import type { SwitchResult } from '../api/projects';
import { useI18n } from '../i18n';
import { AFTER_SWITCH_KEY } from '../store/projectStore';
import { tabHasUnsavedWork, useTabStore } from '../store/tabStore';
import { whenSnapshotWritesSettle } from '../store/tabPersistence';
import { confirm } from './dialog';

/** What the reloaded page says once, after the switch. */
export interface AfterSwitchNotice {
  project: string | null;
  restart: string[];
  usesLfs: boolean;
  imported: number;
  copied: number;
  skipped: number;
}

/** Names of the open tabs with unsaved changes. */
export function unsavedTabNames(): string[] {
  return useTabStore.getState().tabs
    .filter((tab) => tabHasUnsavedWork(tab))
    .map((tab) => tab.name);
}

/** Ask before leaving unsaved tabs behind; true when there are none. */
export async function confirmLeavingUnsavedTabs(): Promise<boolean> {
  const names = unsavedTabNames();
  if (names.length === 0) return true;
  const { t } = useI18n.getState();
  return confirm({
    title: t('projects.switch.unsavedTitle'),
    message: t('projects.switch.unsavedMessage', { names: names.join(', ') }),
    confirmText: t('projects.switch.unsavedConfirm'),
  });
}

/** Remember what to say after the reload, then reload. */
export async function finishSwitch(result: SwitchResult): Promise<void> {
  const notice: AfterSwitchNotice = {
    project: result.project?.name ?? null,
    restart: result.env_restart_recommended ?? [],
    usesLfs: result.uses_lfs === true,
    imported: result.imported?.length ?? 0,
    copied: result.copied?.length ?? 0,
    skipped: (result.skipped?.length ?? 0) + (result.kept?.length ?? 0),
  };
  try {
    sessionStorage.setItem(AFTER_SWITCH_KEY, JSON.stringify(notice));
  } catch {
    // Private mode: the reload just says nothing.
  }
  // The tabs of the project being left are written before the page goes.
  await whenSnapshotWritesSettle().catch(() => undefined);
  window.location.reload();
}

/** The notice a switch left for this page load, read once. */
export function takeAfterSwitchNotice(): AfterSwitchNotice | null {
  try {
    const raw = sessionStorage.getItem(AFTER_SWITCH_KEY);
    if (raw === null) return null;
    sessionStorage.removeItem(AFTER_SWITCH_KEY);
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed as AfterSwitchNotice : null;
  } catch {
    return null;
  }
}
