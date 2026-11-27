import { lazy, Suspense, useEffect, useState } from 'react';
import { ActionMenu, type ActionMenuItem } from '../shared/ActionMenu';
import { useI18n } from '../../i18n';
import { useGitStore } from '../../store/gitStore';
import { useProjectStore } from '../../store/projectStore';
import { useUIStore } from '../../store/uiStore';
import { confirmLeavingUnsavedTabs, finishSwitch } from '../../utils/projectSwitch';
import { openProject } from '../../api/projects';
import { useToastStore } from '../../store/toastStore';
import type { ProjectDialogMode } from './ProjectDialog';
import styles from './Projects.module.css';

const ProjectDialog = lazy(() =>
  import('./ProjectDialog').then((m) => ({ default: m.ProjectDialog })));

/** How many recent projects the menu lists; the Open dialog lists all. */
const MENU_RECENTS = 5;

/**
 * The toolbar's project chip (phase 4A): the open project's name and git
 * branch, with a dot while the working tree has changes. Its menu switches
 * projects -- a recent one, a folder, a new one, a clone -- and, in flat mode,
 * offers to import the graphs that are still in the pre-project folder.
 */
export function ProjectChip() {
  const { t } = useI18n();
  const projectName = useProjectStore((s) => s.projectName);
  const overview = useProjectStore((s) => s.overview);
  const status = useGitStore((s) => s.status);
  const [dialog, setDialog] = useState<ProjectDialogMode | null>(null);
  const repoState = useGitStore((s) => s.repoState);

  // The branch is read once when a project is open; the Source Control
  // panel keeps it current from then on (it refreshes after every write).
  useEffect(() => {
    if (projectName && repoState === 'unknown') void useGitStore.getState().refresh();
  }, [projectName, repoState]);

  const branch = projectName ? status?.branch ?? null : null;
  const dirty = !!status && (status.staged.length + status.unstaged.length
    + status.untracked.length + status.conflicted.length) > 0;
  const recent = (overview?.recent ?? [])
    .filter((entry) => entry.path !== overview?.current?.path)
    .slice(0, MENU_RECENTS);

  const switchToRecent = async (path: string) => {
    if (!(await confirmLeavingUnsavedTabs())) return;
    try {
      const result = await openProject(path);
      if (!result.unchanged) await finishSwitch(result);
    } catch (err) {
      useToastStore.getState().addToast(
        err instanceof Error ? err.message : String(err), 'error');
    }
  };

  const items: ActionMenuItem[] = [
    ...recent.map((entry) => ({
      id: `recent:${entry.path}`,
      label: t('projects.menu.recent', { name: entry.name }),
      disabled: entry.missing,
      hint: entry.missing ? t('projects.menu.missing', { path: entry.path }) : undefined,
      onSelect: () => void switchToRecent(entry.path),
    })),
    { id: 'open', label: t('projects.menu.open'), onSelect: () => setDialog('open') },
    { id: 'create', label: t('projects.menu.create'), onSelect: () => setDialog('create') },
    { id: 'clone', label: t('projects.menu.clone'), onSelect: () => setDialog('clone') },
  ];
  if (!projectName && (overview?.flat_graphs ?? 0) > 0) {
    items.push({
      id: 'import',
      label: t('projects.menu.import', { count: overview!.flat_graphs }),
      onSelect: () => setDialog('import'),
    });
  }
  if (projectName) {
    items.push({
      id: 'source-control',
      label: t('projects.menu.sourceControl'),
      onSelect: () => useUIStore.getState().setSidebarTab('git'),
    });
  }

  const label = projectName ?? t('projects.chip.none');
  return (
    <>
      <ActionMenu
        label={t('projects.chip.menu', { name: label })}
        items={items}
        className={`${styles.chip} ${projectName ? '' : styles.chipFlat}`}
      >
        <span className={styles.chipName} data-testid="project-chip-name">{label}</span>
        {branch && (
          <span className={styles.chipBranch} title={t('projects.chip.branch', { branch })}>
            {branch}
            {dirty && (
              <span className={styles.chipDot} role="img"
                aria-label={t('projects.chip.dirty')} title={t('projects.chip.dirty')} />
            )}
          </span>
        )}
      </ActionMenu>
      {dialog && (
        <Suspense fallback={null}>
          <ProjectDialog mode={dialog} onClose={() => setDialog(null)} />
        </Suspense>
      )}
    </>
  );
}
