import { useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { ProjectError, trustProjectEnv, type ProjectEnv } from '../../api/projects';
import { useI18n } from '../../i18n';
import { useProjectStore } from '../../store/projectStore';
import { useToastStore } from '../../store/toastStore';
import { useDialogKeys } from './useDialogKeys';
import styles from './Projects.module.css';

/** sessionStorage: the .env digest the user said "not now" to. */
export const ENV_DECLINED_KEY = 'codefyui-env-declined';

/**
 * The `.env` consent prompt (phase 4A): the variable names, never the
 * values, with the blocked ones and the ones that choose a server marked.
 * Its own module, loaded only when a project's `.env` needs consent: it is
 * not part of the first screen (the main chunk is held under 500 kB).
 */
export function EnvConsentDialog({ env, project, onClose }: {
  env: ProjectEnv; project: string; onClose: () => void;
}) {
  const { t } = useI18n();
  const baseId = useId();
  const loadRef = useRef<HTMLButtonElement | null>(null);
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const blocked = new Set(env.blocked);
  const loadable = env.names.filter((name) => !blocked.has(name));
  const endpoints = new Set(env.endpoints ?? []);

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

  useDialogKeys(dialogRef, () => {
    if (!busy) notNow();
  });

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
      <div ref={dialogRef} className={styles.dialog} role="alertdialog" aria-modal="true"
        aria-labelledby={`${baseId}-title`} aria-describedby={`${baseId}-intro`}
        tabIndex={-1} data-testid="project-env-consent">
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
                {endpoints.has(name) && (
                  <span className={styles.envEndpoint} data-testid="env-endpoint">
                    <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true">
                      <path d="M9 2h5v5M14 2 7 9M12 9.5V13a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h3.5"
                        fill="none" stroke="currentColor" strokeWidth="1.5"
                        strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    {t('projects.env.endpoint')}
                  </span>
                )}
              </li>
            ))}
          </ul>
          {env.blocked.length > 0 && (
            <p className={styles.hint}>{t('projects.env.blockedHint')}</p>
          )}
          {endpoints.size > 0 && (
            <p className={styles.envEndpointHint}>{t('projects.env.endpointsHint')}</p>
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
