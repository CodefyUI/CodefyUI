import { useEffect, useId, useRef, useState, type FormEvent } from 'react';
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
import { projectErrorText } from './projectErrors';
import { useDialogKeys } from './useDialogKeys';
import styles from './Projects.module.css';

export type ProjectDialogMode = 'open' | 'create' | 'clone' | 'import';

/** What `create` accepts; the server checks the same and says why. */
const NAME_PATTERN = /^[A-Za-z0-9._-]{1,64}$/;
/** Names Windows keeps for devices, with or without an extension. */
const WINDOWS_RESERVED = new Set([
  'CON', 'PRN', 'AUX', 'NUL',
  ...Array.from({ length: 9 }, (_, i) => `COM${i + 1}`),
  ...Array.from({ length: 9 }, (_, i) => `LPT${i + 1}`),
]);

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
  const [detail, setDetail] = useState<string | null>(null);
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
      getImportPlan().then(setPlan, (err: unknown) => showError(err));
    }
    if (mode === 'open' && overview?.browse_allowed) {
      browseFolders().then(setBrowse, () => setBrowse(null));
    }
  }, [mode, overview?.browse_allowed]);

  const title = t(`projects.dialog.${mode}.title`);

  useDialogKeys(dialogRef, () => {
    if (!busy) onClose();
  });

  // A refused request leaves the focus on <body> (the submit button was
  // disabled under it): put it back where the user can fix what was wrong.
  useEffect(() => {
    if (error === null) return;
    const input = firstRef.current;
    if (input) {
      input.focus();
      input.select();
    } else {
      dialogRef.current?.focus();
    }
  }, [error]);

  const showError = (err: unknown) => {
    const text = projectErrorText(err, t);
    setError(text.message);
    setDetail(text.detail);
    setHint(err instanceof ProjectError ? err.hint : null);
  };

  const showProblem = (message: string) => {
    setError(message);
    setDetail(null);
    setHint(null);
  };

  const run = async (action: () => Promise<SwitchResult>) => {
    setError(null);
    setDetail(null);
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
      setBusy(false);
      showError(err);
    }
  };

  const nameProblem = (name: string): string | null => {
    if (!NAME_PATTERN.test(name)) return t('projects.dialog.name.invalid');
    if (name.startsWith('.')) return t('projects.dialog.name.dot');
    if (name.endsWith('.')) return t('projects.dialog.name.trailingDot');
    if (WINDOWS_RESERVED.has(name.split('.')[0].toUpperCase())) {
      return t('projects.dialog.name.reserved', { name });
    }
    return null;
  };

  const naming = mode === 'create' || mode === 'import';
  const typedName = value.trim();
  /** Said while typing, so a name that cannot work never reaches Submit. */
  const liveProblem = naming && typedName !== '' ? nameProblem(typedName) : null;
  const nameUnusable = naming && (typedName === '' || liveProblem !== null);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const text = value.trim();
    if (mode === 'open') {
      if (!text) return showProblem(t('projects.dialog.open.empty'));
      void run(() => openProject(text));
    } else if (mode === 'create') {
      const problem = nameProblem(text);
      if (problem) return showProblem(problem);
      void run(() => createProject(text));
    } else if (mode === 'clone') {
      if (!/^https:\/\//i.test(text)) return showProblem(t('projects.dialog.clone.httpsOnly'));
      void run(() => cloneProject(text));
    } else {
      const problem = nameProblem(text);
      if (problem) return showProblem(problem);
      void run(() => importFlatGraphs({ name: text }, includeAssets));
    }
  };

  const removeRecent = async (path: string) => {
    try {
      await removeRecentProject(path);
      await useProjectStore.getState().loadOverview();
    } catch (err) {
      showError(err);
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
                            onClick={() => browseFolders(browse.parent!).then(setBrowse, (e) => showError(e))}>
                            <span className={styles.rowName}>..</span>
                          </button>
                        </li>
                      )}
                      {browse.entries.map((entry) => (
                        <li key={entry.path} className={styles.row}>
                          <button type="button" className={styles.rowMain}
                            onClick={() => {
                              setValue(entry.path);
                              browseFolders(entry.path).then(setBrowse, (e) => showError(e));
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
                    aria-invalid={liveProblem !== null}
                    aria-describedby={`${baseId}-name-note`}
                    onChange={(event) => setValue(event.target.value)} />
                </label>
                <p id={`${baseId}-name-note`}
                  className={liveProblem ? styles.fieldProblem : styles.hint}
                  data-testid="project-name-note">
                  {liveProblem
                    ?? t('projects.dialog.create.where', { path: `${root}/${typedName || '…'}` })}
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
                    aria-invalid={liveProblem !== null}
                    aria-describedby={liveProblem ? `${baseId}-name-note` : undefined}
                    onChange={(event) => setValue(event.target.value)} />
                </label>
                {liveProblem && (
                  <p id={`${baseId}-name-note`} className={styles.fieldProblem}
                    data-testid="project-name-note">{liveProblem}</p>
                )}
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
                {detail && <p className={styles.errorHint}>{detail}</p>}
                {hint && <p className={styles.errorHint}>{hint}</p>}
              </div>
            )}
          </div>
          <footer className={styles.footer}>
            <button type="button" onClick={onClose} disabled={busy}>
              {t('projects.dialog.cancel')}
            </button>
            <button type="submit" className={styles.primary} disabled={busy || nameUnusable || (mode === 'import' && plan === null)}>
              {busy ? t(`projects.dialog.${mode}.busy`) : t(`projects.dialog.${mode}.submit`)}
            </button>
          </footer>
        </form>
      </div>
    </div>,
    document.body,
  );
}

