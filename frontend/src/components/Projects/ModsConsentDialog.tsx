import { useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  formatBytes, getProjectMods, ProjectError, trustProjectMods,
  type ModsFile, type ModsFinding, type ModsScan,
} from '../../api/projects';
import { useI18n } from '../../i18n';
import { capabilityKey } from '../PluginCenter/pluginStatus';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useProjectStore } from '../../store/projectStore';
import { useTabStore } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import { useDialogKeys } from './useDialogKeys';
import styles from './Projects.module.css';

/** sessionStorage: the mods digest the user said "not now" to. */
export const MODS_DECLINED_KEY = 'codefyui-mods-declined';

type Mark = 'added' | 'changed' | null;

/**
 * The mods/ trust prompt (phase 4C). Lists every file with its size, line
 * count, scan result and node names (read from the source, nothing is
 * imported to answer), and the entries that are never loaded. Consent names
 * the digest of exactly what was shown; a tree that changed since is refused
 * by the server. Its own chunk: loaded only when mods need a decision.
 */
export function ModsConsentDialog({ project, onClose }: { project: string; onClose: () => void }) {
  const { t } = useI18n();
  const baseId = useId();
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const trustRef = useRef<HTMLButtonElement | null>(null);
  const [scan, setScan] = useState<ModsScan | null>(null);
  const [full, setFull] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    getProjectMods()
      .then((answer) => { if (live) setScan(answer); })
      .catch((err) => { if (live) setError(err instanceof Error ? err.message : String(err)); });
    return () => { live = false; };
  }, []);

  useEffect(() => {
    if (scan) trustRef.current?.focus();
  }, [scan]);

  const notNow = () => {
    try {
      if (scan?.digest) sessionStorage.setItem(MODS_DECLINED_KEY, scan.digest);
    } catch {
      /* asked again next load */
    }
    onClose();
  };

  useDialogKeys(dialogRef, () => {
    if (!busy) notNow();
  });

  const trust = async () => {
    if (!scan?.digest) return;
    setBusy(true);
    setError(null);
    try {
      const result = await trustProjectMods(scan.digest, full);
      const store = useProjectStore.getState();
      if (store.overview && result.mods) store.setOverview({ ...store.overview, mods: result.mods });
      // The new node types, then every open card that was waiting for one.
      await useNodeDefStore.getState().fetchDefinitions();
      useTabStore.getState().refreshNodeDefinitions('project:');
      const toast = useToastStore.getState().addToast;
      toast(t('projects.mods.loaded', { count: result.mods?.nodes?.length ?? 0 }), 'success');
      const skipped = result.mods?.skipped ?? [];
      if (skipped.length > 0) {
        toast(t('projects.mods.skipped', {
          count: skipped.length, files: skipped.map((s) => s.path).join(', '),
        }), 'info');
      }
      onClose();
    } catch (err) {
      setError(err instanceof ProjectError && err.code === 'mods_changed'
        ? t('projects.mods.changedAgain')
        : err instanceof ProjectError && err.code === 'mods_blocked'
          ? t('projects.mods.blocked')
          : err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  };

  const changed = scan?.trust.changed ?? null;
  const markOf = (path: string): Mark =>
    changed?.added.includes(path) ? 'added' : changed?.changed.includes(path) ? 'changed' : null;
  const blocked = (scan?.refused.length ?? 0) > 0;
  const anyFinding = scan?.files.some((f) => f.finding) ?? false;

  return createPortal(
    <div className={styles.backdrop}>
      <div ref={dialogRef} className={`${styles.dialog} ${styles.dialogWide}`} role="alertdialog"
        aria-modal="true" aria-labelledby={`${baseId}-title`} aria-describedby={`${baseId}-warn`}
        tabIndex={-1} data-testid="project-mods-consent">
        <header className={styles.header}>
          <h2 id={`${baseId}-title`}>
            {t(blocked ? 'projects.mods.refusedTitle' : 'projects.mods.title')}
          </h2>
        </header>
        <div className={styles.body}>
          <p>
            {t(scan?.trust.state === 'changed' ? 'projects.mods.introChanged' : 'projects.mods.intro',
              { project })}
          </p>
          <p id={`${baseId}-warn`} className={styles.modsWarning}>{t('projects.mods.warning')}</p>
          {!scan && !error && <p className={styles.hint}>{t('projects.mods.loading')}</p>}
          {scan && blocked && (
            <>
              <p>{t('projects.mods.refused')}</p>
              <ul className={styles.modsList} data-testid="mods-refused">
                {scan.refused.map((r) => (
                  <li key={r.path} className={styles.modsRow}>
                    <code className={styles.modsPath}>mods/{r.path}</code>
                    <span className={styles.rowWarn}>{r.reason}</span>
                  </li>
                ))}
              </ul>
            </>
          )}
          {scan && (
            <ul className={styles.modsList} aria-label={t('projects.mods.files')}>
              {scan.files.map((file) => (
                <ModsFileRow key={file.path} file={file} mark={markOf(file.path)} />
              ))}
              {(changed?.removed ?? []).map((path) => (
                <li key={`removed:${path}`} className={styles.modsRow}>
                  <code className={styles.modsPath}>mods/{path}</code>
                  <span className={styles.badge}>{t('projects.mods.removed')}</span>
                </li>
              ))}
            </ul>
          )}
          {scan && !blocked && anyFinding && (
            <label className={styles.check}>
              <input type="checkbox" checked={full} onChange={(e) => setFull(e.target.checked)}
                data-testid="mods-trust-full" />
              {t('projects.mods.full')}
            </label>
          )}
          {scan && !blocked && anyFinding && !full && (
            <p className={styles.hint}>{t('projects.mods.fullHint')}</p>
          )}
          {scan && !blocked && <p className={styles.hint}>{t('projects.mods.unloadNote')}</p>}
          {error && <p role="alert" className={styles.error}>{error}</p>}
        </div>
        <footer className={styles.footer}>
          {blocked ? (
            <button type="button" className={styles.primary} onClick={notNow}>
              {t('projects.mods.close')}
            </button>
          ) : (
            <>
              <button type="button" onClick={notNow} disabled={busy}>
                {t('projects.mods.notNow')}
              </button>
              <button ref={trustRef} type="button" className={styles.primary}
                disabled={busy || !scan?.digest} onClick={() => void trust()}
                data-testid="mods-trust">
                {t('projects.mods.trust')}
              </button>
            </>
          )}
        </footer>
      </div>
    </div>,
    document.body,
  );
}

type Translate = ReturnType<typeof useI18n.getState>['t'];

/** A capability in the page's language: its sentence, or the raw id. */
function capabilityText(t: Translate, id: string): string {
  const key = capabilityKey(id);
  return key ? t(key) : id;
}

/**
 * The scan's finding in the words of a project's mods (phase 4C review,
 * round 1): the plugin gate's own text tells an author to edit a plugin
 * manifest, which a project does not have, and it is English only.
 */
export function findingText(t: Translate, finding: ModsFinding): string {
  const line = finding.line ?? 0;
  const at = finding.line !== null;
  switch (finding.kind) {
    case 'import':
      return t('projects.mods.finding.import', {
        line, module: finding.module ?? '',
        capability: capabilityText(t, finding.capability ?? ''),
      });
    case 'blocked_import':
      return t('projects.mods.finding.blockedImport', { line, module: finding.module ?? '' });
    case 'refused':
      return finding.construct
        ? t('projects.mods.finding.refusedNamed', { line, construct: finding.construct })
        : t('projects.mods.finding.refused', { line });
    case 'syntax':
      return at ? t('projects.mods.finding.syntax', { line }) : t('projects.mods.finding.syntaxNoLine');
    default:
      return t('projects.mods.finding.unreadable');
  }
}

function ModsFileRow({ file, mark }: { file: ModsFile; mark: Mark }) {
  const { t } = useI18n();
  const facts = [formatBytes(file.size)];
  if (file.lines !== undefined) facts.push(t('projects.mods.lines', { count: file.lines }));
  const never = file.loads === false;
  return (
    <li className={`${styles.modsRow} ${never ? styles.modsRowNever : ''}`} data-testid="mods-file">
      <div className={styles.modsHead}>
        <code className={styles.modsPath}>mods/{file.path}</code>
        <span className={styles.hint}>{facts.join(' · ')}</span>
        {mark && <span className={styles.badge}>{t(`projects.mods.${mark}`)}</span>}
        {never && (
          <span className={styles.modsNever} title={t('projects.mods.neverLoadedTitle')}
            data-testid="mods-never-loaded">
            {t('projects.mods.neverLoaded')}
          </span>
        )}
      </div>
      {never && <span className={styles.hint}>{t('projects.mods.neverLoadedTitle')}</span>}
      {file.role === 'node' && (
        <div className={styles.modsFacts}>
          {file.finding ? (
            <span className={styles.rowWarn}>{findingText(t, file.finding)}</span>
          ) : (
            <span className={styles.hint}>{t('projects.mods.clean')}</span>
          )}
          {/* The finding already names the capability its import needs. */}
          {!file.finding && (file.needs?.length ?? 0) > 0 && (
            <span className={styles.hint}>
              {t('projects.mods.needs', {
                names: file.needs!.map((n) => capabilityText(t, n.capability)).join('; '),
              })}
            </span>
          )}
          {(file.nodes?.length ?? 0) > 0 && (
            <span className={styles.hint}>
              {t('projects.mods.nodes', { names: file.nodes!.map((n) => `project:${n}`).join(', ') })}
            </span>
          )}
          {(file.duplicates?.length ?? 0) > 0 && (
            <span className={styles.rowWarn}>{t('projects.mods.duplicate')}</span>
          )}
        </div>
      )}
    </li>
  );
}
