import { useEffect, useId, useRef, useState, type FormEvent, type KeyboardEvent } from 'react';
import { createPortal } from 'react-dom';
import {
  browseFolders,
  cloneProject,
  createProject,
  formatBytes,
  getImportPlan,
  importFlatGraphs,
  openProject,
  ProjectError,
  removeRecentProject,
  type BrowseResult,
  type ImportPlan,
  type SwitchResult,
} from '../../api/projects';
import { useI18n } from '../../i18n';
import { useProjectStore } from '../../store/projectStore';
import { confirmLeavingUnsavedTabs, finishSwitch } from '../../utils/projectSwitch';
import styles from './Projects.module.css';

export type ProjectDialogMode = 'open' | 'create' | 'clone' | 'import';

const FOCUSABLE = 'button:not([disabled]), input:not([disabled]), [tabindex="0"]';
/** What `create` accepts; the server checks the same and says why. */
const NAME_PATTERN = /^[A-Za-z0-9._-]{1,64}$/;

interface Props {
  mode: ProjectDialogMode;
  onClose: () => void;
}

/**
 * Open a folder, create a project, clone one, or import the flat graphs.
 *
 * Every action ends in a switch: unsaved tabs are confirmed first, the
 * server's refusal (a run in progress, a folder that is not a project) is
 * shown in the dialog as it is, and success reloads the page.
 */
export function ProjectDialog({ mode, onClose }: Props) {
  const { t } = useI18n();
  const overview = useProjectStore((s) => s.overview);
  const baseId = useId();
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const firstRef = useRef<HTMLInputElement | null>(null);
  const [value, setValue] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hint, setHint] = useState<string | null>(null);
  const [browse, setBrowse] = useState<BrowseResult | null>(null);
  const [plan, setPlan] = useState<ImportPlan | null>(null);
  const [includeAssets, setIncludeAssets] = useState(true);

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const timer = setTimeout(() => firstRef.current?.focus(), 0);
    return () => {
      clearTimeout(timer);
      if (previous?.isConnected) previous.focus();
    };
  }, []);

  useEffect(() => {
    if (mode === 'import') {
      setValue('workspace');
      getImportPlan().then(setPlan, (err: unknown) => setError(messageOf(err)));
    }
    if (mode === 'open' && overview?.browse_allowed) {
      browseFolders().then(setBrowse, () => setBrowse(null));
    }
  }, [mode, overview?.browse_allowed]);

  const title = t(`projects.dialog.${mode}.title`);

  const handleKeys = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      if (!busy) onClose();
      return;
    }
    if (event.key !== 'Tab') return;
    const focusable = Array.from(dialogRef.current!.querySelectorAll<HTMLElement>(FOCUSABLE));
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  const run = async (action: () => Promise<SwitchResult>) => {
    setError(null);
    setHint(null);
    if (!(await confirmLeavingUnsavedTabs())) return;
    setBusy(true);
    try {
      const result = await action();
      if (result.unchanged) {
        onClose();
        return;
      }
      await finishSwitch(result);
    } catch (err) {
      setError(messageOf(err));
      setHint(err instanceof ProjectError ? err.hint : null);
      setBusy(false);
    }
  };

  const nameProblem = (name: string): string | null => {
    if (!NAME_PATTERN.test(name)) return t('projects.dialog.name.invalid');
    if (name.startsWith('.')) return t('projects.dialog.name.dot');
    return null;
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const text = value.trim();
    if (mode === 'open') {
      if (!text) return setError(t('projects.dialog.open.empty'));
      void run(() => openProject(text));
    } else if (mode === 'create') {
      const problem = nameProblem(text);
      if (problem) return setError(problem);
      void run(() => createProject(text));
    } else if (mode === 'clone') {
      if (!/^https:\/\//i.test(text)) return setError(t('projects.dialog.clone.httpsOnly'));
      void run(() => cloneProject(text));
    } else {
      const problem = nameProblem(text);
      if (problem) return setError(problem);
      void run(() => importFlatGraphs({ name: text }, includeAssets));
    }
  };

  const removeRecent = async (path: string) => {
    try {
      await removeRecentProject(path);
      await useProjectStore.getState().loadOverview();
    } catch (err) {
      setError(messageOf(err));
    }
  };

  const recent = (overview?.recent ?? []).filter(
    (entry) => entry.path !== overview?.current?.path,
  );
  const root = overview?.projects_root ?? '';

  return createPortal(
    <div className={styles.backdrop} onMouseDown={(event) => {
      if (event.target === event.currentTarget && !busy) onClose();
    }}>
      <div
        ref={dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby={`${baseId}-title`}
        aria-busy={busy}
        tabIndex={-1}
        onKeyDown={handleKeys}
        data-testid={`project-dialog-${mode}`}
      >
        <form noValidate onSubmit={submit}>
          <header className={styles.header}>
            <h2 id={`${baseId}-title`}>{title}</h2>
            <button type="button" className={styles.close} onClick={onClose}
              disabled={busy} aria-label={t('projects.dialog.close')}>&#215;</button>
          </header>
          <div className={styles.body}>
            {mode === 'open' && (
              <>
                {recent.length > 0 && (
                  <section aria-label={t('projects.dialog.open.recent')}>
                    <h3 className={styles.sectionLabel}>{t('projects.dialog.open.recent')}</h3>
                    <ul className={styles.list}>
                      {recent.map((entry) => (
                        <li key={entry.path} className={styles.row}>
                          <button type="button" className={styles.rowMain}
                            disabled={busy || entry.missing}
                            onClick={() => void run(() => openProject(entry.path))}
                            title={entry.path}>
                            <span className={styles.rowName}>{entry.name}</span>
                            <span className={styles.rowPath}>{entry.path}</span>
                            {entry.missing && (
                              <span className={styles.rowWarn}>{t('projects.dialog.open.missing')}</span>
                            )}
                          </button>
                          <button type="button" className={styles.rowAside}
                            disabled={busy}
                            aria-label={t('projects.dialog.open.forget', { name: entry.name })}
                            title={t('projects.dialog.open.forgetTitle')}
                            onClick={() => void removeRecent(entry.path)}>
                            {t('projects.dialog.open.forgetShort')}
                          </button>
                        </li>
                      ))}
                    </ul>
                  </section>
                )}
                <label className={styles.field}>
                  {t('projects.dialog.open.path')}
                  <input ref={firstRef} value={value} spellCheck={false}
                    placeholder={root ? `${root}/…` : ''}
                    onChange={(event) => setValue(event.target.value)} />
                </label>
                {browse ? (
                  <section aria-label={t('projects.dialog.open.browse')}>
                    <h3 className={styles.browseLabel}>
                      {t('projects.dialog.open.browseIn', { dir: browse.dir })}
                    </h3>
                    <ul className={styles.list}>
                      {browse.parent && (
                        <li className={styles.row}>
                          <button type="button" className={styles.rowMain}
                            onClick={() => browseFolders(browse.parent!).then(setBrowse, (e) => setError(messageOf(e)))}>
                            <span className={styles.rowName}>..</span>
                          </button>
                        </li>
                      )}
                      {browse.entries.map((entry) => (
                        <li key={entry.path} className={styles.row}>
                          <button type="button" className={styles.rowMain}
                            onClick={() => {
                              setValue(entry.path);
                              browseFolders(entry.path).then(setBrowse, (e) => setError(messageOf(e)));
                            }}>
                            <span className={styles.rowName}>{entry.name}</span>
                            {entry.is_project && (
                              <span className={styles.badge}>{t('projects.dialog.open.isProject')}</span>
                            )}
                          </button>
                        </li>
                      ))}
                      {browse.entries.length === 0 && (
                        <li className={styles.empty}>{t('projects.dialog.open.noFolders')}</li>
                      )}
                    </ul>
                  </section>
                ) : (
                  overview && !overview.browse_allowed && (
                    <p className={styles.hint}>{t('projects.dialog.open.noBrowse')}</p>
                  )
                )}
              </>
            )}

            {mode === 'create' && (
              <>
                <label className={styles.field}>
                  {t('projects.dialog.create.name')}
                  <input ref={firstRef} value={value} spellCheck={false}
                    onChange={(event) => setValue(event.target.value)} />
                </label>
                <p className={styles.hint}>
                  {t('projects.dialog.create.where', { path: `${root}/${value.trim() || '…'}` })}
                </p>
              </>
            )}

            {mode === 'clone' && (
              <>
                <label className={styles.field}>
                  {t('projects.dialog.clone.url')}
                  <input ref={firstRef} value={value} spellCheck={false} inputMode="url"
                    placeholder="https://"
                    onChange={(event) => setValue(event.target.value)} />
                </label>
                <p className={styles.hint}>{t('projects.dialog.clone.hint', { root })}</p>
              </>
            )}

            {mode === 'import' && (
              <>
                <p className={styles.hint}>{t('projects.dialog.import.intro')}</p>
                <label className={styles.field}>
                  {t('projects.dialog.create.name')}
                  <input ref={firstRef} value={value} spellCheck={false}
                    onChange={(event) => setValue(event.target.value)} />
                </label>
                {plan === null && !error && (
                  <p className={styles.hint}>{t('projects.dialog.import.loading')}</p>
                )}
                {plan && (
                  <>
                    <p className={styles.summary}>
                      {t('projects.dialog.import.graphs', {
                        count: plan.graphs.length, size: formatBytes(plan.graphs_bytes),
                      })}
                    </p>
                    <label className={styles.check}>
                      <input type="checkbox" checked={includeAssets}
                        disabled={plan.assets.length === 0}
                        onChange={(event) => setIncludeAssets(event.target.checked)} />
                      {plan.assets.length === 0
                        ? t('projects.dialog.import.noAssets')
                        : t('projects.dialog.import.assets', {
                          count: plan.assets.length, size: formatBytes(plan.assets_bytes),
                        })}
                    </label>
                    {plan.assets.length > 0 && (
                      <ul className={`${styles.list} ${styles.assetList}`}
                        aria-label={t('projects.dialog.import.assetList')}>
                        {plan.assets.map((asset) => (
                          <li key={asset.path} className={styles.assetRow}>
                            <span className={styles.rowPath}>
                              {asset.path}{asset.is_dir ? '/' : ''}
                            </span>
                            <span className={styles.size}>
                              {asset.is_dir
                                ? t('projects.dialog.import.folderSize', {
                                  size: formatBytes(asset.size), count: asset.files,
                                })
                                : formatBytes(asset.size)}
                            </span>
                          </li>
                        ))}
                      </ul>
                    )}
                    <p className={styles.hint}>{t('projects.dialog.import.copyOnly', { source: plan.source })}</p>
                  </>
                )}
              </>
            )}

            {error && (
              <div role="alert" className={styles.error}>
                <p>{error}</p>
                {hint && <p className={styles.errorHint}>{hint}</p>}
              </div>
            )}
          </div>
          <footer className={styles.footer}>
            <button type="button" onClick={onClose} disabled={busy}>
              {t('projects.dialog.cancel')}
            </button>
            <button type="submit" className={styles.primary} disabled={busy || (mode === 'import' && plan === null)}>
              {busy ? t(`projects.dialog.${mode}.busy`) : t(`projects.dialog.${mode}.submit`)}
            </button>
          </footer>
        </form>
      </div>
    </div>,
    document.body,
  );
}

function messageOf(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
