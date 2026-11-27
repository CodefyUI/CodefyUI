import { Suspense, useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { PROJECT_CHANGED_EVENT } from '../../api/_auth';
import { useI18n } from '../../i18n';
import { useProjectStore } from '../../store/projectStore';
import { useToastStore } from '../../store/toastStore';
import { takeAfterSwitchNotice } from '../../utils/projectSwitch';
import { useProjectDriftCheck } from '../../hooks/useProjectDriftCheck';
import { lazyRetry } from '../shared/lazyRetry';
import styles from './Projects.module.css';

/** sessionStorage: the .env digest the user said "not now" to (kept in
 * step with EnvConsentDialog's ENV_DECLINED_KEY, which is not imported here
 * so that the dialog stays out of the main chunk). */
const ENV_DECLINED_KEY = 'codefyui-env-declined';

const EnvConsentDialog = lazyRetry(
  () => import('./EnvConsentDialog').then((m) => m.EnvConsentDialog),
  (props) => props.onClose(),
);

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

  // A stale window can only be reloaded: everything behind the banner is
  // made inert, so nothing is clicked or typed into that would be refused
  // (or, before the window learned, applied to the wrong project).
  useEffect(() => {
    const app = document.getElementById('root');
    if (!stale || app === null) return;
    // The attribute rather than the property: the same thing in a browser,
    // and what an older engine (or jsdom) without `inert` still records.
    app.setAttribute('inert', '');
    return () => app.removeAttribute('inert');
  }, [stale]);

  return (
    <>
      {stale && createPortal(
        <div role="alert" className={styles.staleBanner} data-testid="project-stale-banner">
          <span>{t('projects.stale.message')}</span>
          <button type="button" onClick={() => window.location.reload()}>
            {t('projects.stale.reload')}
          </button>
        </div>,
        document.body,
      )}
      {consentOpen && overview && (
        <Suspense fallback={null}>
          <EnvConsentDialog env={overview.env} project={overview.current?.name ?? ''}
            onClose={() => setConsentOpen(false)} />
        </Suspense>
      )}
    </>
  );
}
