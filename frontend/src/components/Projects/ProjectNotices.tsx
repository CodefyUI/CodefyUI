import { useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { PROJECT_CHANGED_EVENT } from '../../api/_auth';
import { ProjectError, trustProjectEnv, type ProjectEnv } from '../../api/projects';
import { useI18n } from '../../i18n';
import { useProjectStore } from '../../store/projectStore';
import { useToastStore } from '../../store/toastStore';
import { takeAfterSwitchNotice } from '../../utils/projectSwitch';
import { useProjectDriftCheck } from '../../hooks/useProjectDriftCheck';
import styles from './Projects.module.css';

/** sessionStorage: the .env digest the user said "not now" to. */
const ENV_DECLINED_KEY = 'codefyui-env-declined';

function declined(sha: string | null): boolean {
  try {
    return sha !== null && sessionStorage.getItem(ENV_DECLINED_KEY) === sha;
  } catch {
    return false;
  }
}

/**
 * Everything the open project has to tell the user once (phase 4A):
 *
 * - a banner when this window shows a project that is no longer open
 *   (seen when the window regains focus, from a switch broadcast, or from
 *   a write that came back `project_changed`);
 * - the `.env` consent prompt for a project this install did not create;
 * - after a switch, what the reload should say (variables that need a
 *   restart, files left in git LFS, what an import copied);
 * - start-up warnings (a `workspace` folder that is not a project).
 */
export function ProjectNotices() {
  const { t } = useI18n();
  const stale = useProjectStore((s) => s.stale);
  const overview = useProjectStore((s) => s.overview);
  const [consentOpen, setConsentOpen] = useState(false);
  useProjectDriftCheck();

  useEffect(() => {
    const onChanged = () => useProjectStore.getState().markStale();
    window.addEventListener(PROJECT_CHANGED_EVENT, onChanged);
    return () => window.removeEventListener(PROJECT_CHANGED_EVENT, onChanged);
  }, []);

  useEffect(() => {
    const notice = takeAfterSwitchNotice();
    if (!notice) return;
    const toast = useToastStore.getState().addToast;
    if (notice.imported > 0) {
      toast(t('projects.notice.imported', {
        count: notice.imported, files: notice.copied, kept: notice.skipped,
      }), 'success');
    }
    if (notice.restart.length > 0) {
      toast(t('projects.notice.restart', { names: notice.restart.join(', ') }), 'info');
    }
    if (notice.usesLfs) toast(t('projects.notice.lfs'), 'info');
    // Mount-time only: a later render must not repeat what the reload said.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!overview) return;
    if (overview.env.needs_consent && !declined(overview.env.sha256)) {
      setConsentOpen(true);
    }
    const warnings = overview.startup?.warnings ?? [];
    if (warnings.length > 0) {
      try {
        if (sessionStorage.getItem('codefyui-startup-warned') === '1') return;
        sessionStorage.setItem('codefyui-startup-warned', '1');
      } catch {
        /* show it anyway */
      }
      for (const warning of warnings) useToastStore.getState().addToast(warning, 'info');
    }
  }, [overview]);

  return (
    <>
      {stale && (
        <div role="alert" className={styles.staleBanner} data-testid="project-stale-banner">
          <span>{t('projects.stale.message')}</span>
          <button type="button" onClick={() => window.location.reload()}>
            {t('projects.stale.reload')}
          </button>
        </div>
      )}
      {consentOpen && overview && (
        <EnvConsentDialog env={overview.env} project={overview.current?.name ?? ''}
          onClose={() => setConsentOpen(false)} />
      )}
    </>
  );
}

function EnvConsentDialog({ env, project, onClose }: {
  env: ProjectEnv; project: string; onClose: () => void;
}) {
  const { t } = useI18n();
  const baseId = useId();
  const loadRef = useRef<HTMLButtonElement | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const blocked = new Set(env.blocked);
  const loadable = env.names.filter((name) => !blocked.has(name));

  useEffect(() => {
    const timer = setTimeout(() => loadRef.current?.focus(), 0);
    return () => clearTimeout(timer);
  }, []);

  const notNow = () => {
    try {
      if (env.sha256) sessionStorage.setItem(ENV_DECLINED_KEY, env.sha256);
    } catch {
      /* asked again next load */
    }
    onClose();
  };

  const load = async () => {
    if (!env.sha256) return;
    setBusy(true);
    setError(null);
    try {
      const result = await trustProjectEnv(env.sha256);
      useProjectStore.getState().setOverview({
        ...useProjectStore.getState().overview!, env: result.env,
      });
      const toast = useToastStore.getState().addToast;
      toast(t('projects.env.loaded', { count: result.env.applied.length }), 'success');
      if (result.env_restart_recommended.length > 0) {
        toast(t('projects.notice.restart', {
          names: result.env_restart_recommended.join(', '),
        }), 'info');
      }
      onClose();
    } catch (err) {
      setError(err instanceof ProjectError && err.code === 'env_changed'
        ? t('projects.env.changed')
        : err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  };

  return createPortal(
    <div className={styles.backdrop}>
      <div className={styles.dialog} role="alertdialog" aria-modal="true"
        aria-labelledby={`${baseId}-title`} aria-describedby={`${baseId}-intro`}
        data-testid="project-env-consent"
        onKeyDown={(event) => {
          if (event.key === 'Escape' && !busy) {
            event.preventDefault();
            notNow();
          }
        }}>
        <header className={styles.header}>
          <h2 id={`${baseId}-title`}>{t('projects.env.title')}</h2>
        </header>
        <div className={styles.body}>
          <p id={`${baseId}-intro`}>
            {t(env.state === 'changed' ? 'projects.env.introChanged' : 'projects.env.intro',
              { project })}
          </p>
          <ul className={styles.envList} aria-label={t('projects.env.names')}>
            {env.names.map((name) => (
              <li key={name} className={styles.envRow}>
                <code>{name}</code>
                {blocked.has(name) && (
                  <span className={styles.rowWarn}>{t('projects.env.blocked')}</span>
                )}
              </li>
            ))}
          </ul>
          {env.blocked.length > 0 && (
            <p className={styles.hint}>{t('projects.env.blockedHint')}</p>
          )}
          <p className={styles.hint}>{t('projects.env.valuesHidden')}</p>
          {error && <p role="alert" className={styles.error}>{error}</p>}
        </div>
        <footer className={styles.footer}>
          <button type="button" onClick={notNow} disabled={busy}>
            {t('projects.env.notNow')}
          </button>
          <button ref={loadRef} type="button" className={styles.primary}
            disabled={busy || loadable.length === 0} onClick={() => void load()}>
            {t('projects.env.load', { count: loadable.length })}
          </button>
        </footer>
      </div>
    </div>,
    document.body,
  );
}
