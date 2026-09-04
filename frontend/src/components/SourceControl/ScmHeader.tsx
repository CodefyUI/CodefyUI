import { useState } from 'react';
import type { GitErrorCode } from '../../api/git';
import { gitOpKey, useGitStore, type GitStoreError } from '../../store/gitStore';
import { useI18n, type TranslationKey } from '../../i18n';
import { docsUrl } from '../../utils/docsUrl';
import { MoreHorizontalIcon, RefreshIcon } from '../shared/Icons';
import { ActionMenu, type ActionMenuItem } from '../shared/ActionMenu';
import { ProgressBar } from '../shared/ProgressBar';
import shell from '../Sidebar/NodePalette.module.css';
import styles from './SourceControl.module.css';

/** The documentation page the Setup guide link and menu row point at. */
export const SCM_DOCS_PATH = '/usage/source-control';

/**
 * A layout file: the half of a saved graph that holds positions and notes.
 *
 * One Save writes a PAIR (`graphs/<name>.graph.json` and
 * `layout/<name>.layout.json`), and only the first of the two is a change
 * anybody reviews -- which is what the "Hide layout files" filter is for.
 */
export function isLayoutFile(path: string): boolean {
  return /^layout\/.+\.layout\.json$/.test(path);
}

/** Translate, with the store's own signature. */
type Translate = (key: TranslationKey, vars?: Record<string, string | number>) => string;

/**
 * The codes that have a sentence of their own.
 *
 * Everything absent from this map falls to `git.error.generic`, which shows
 * git's own words -- and that is deliberately where `invalid` (FastAPI's 422),
 * `unknown` (a code from a newer server, or a body that was not the git
 * envelope) and `git_service_unavailable` land: a fixed sentence for those
 * would replace the only description of the problem that exists.
 */
const ERROR_KEY: Partial<Record<GitErrorCode, TranslationKey>> = {
  busy: 'git.error.busy',
  nothing_to_commit: 'git.error.nothingToCommit',
  identity_missing: 'git.error.identityMissing',
  detached_head: 'git.error.detachedHead',
  merge_in_progress: 'git.error.mergeInProgress',
  not_repo: 'git.error.notRepo',
  invalid_value: 'git.error.invalid',
};

/**
 * The sentence for one refusal.
 *
 * `timeout` is the exception that is not a lookup: the 504 body carries a code
 * and nothing else, so the store has already written the finished sentence
 * (it is the only place that knows which of the three deadlines applied) and
 * re-mapping the code here would throw that number away.
 */
export function errorSentence(err: GitStoreError, t: Translate): string {
  if (err.code === 'timeout') return err.message;
  if (err.code === 'not_found') return t('git.error.notFound', { what: err.message });
  const key = ERROR_KEY[err.code];
  return key === undefined ? t('git.error.generic', { message: err.message }) : t(key);
}

/**
 * The tab's title row, the branch line, the busy bar and the error line.
 *
 * Drawn in EVERY repository state, not only in `ready`. Two reasons: the title
 * row is the shell the other four sidebar tabs share, so dropping it would
 * leave the fifth tab's header a few pixels off everyone else's; and a server
 * that stops answering after a good first read has to be able to say so, which
 * means the error line cannot live inside the branch of a switch that a
 * failed read never reaches.
 */
export function ScmHeader() {
  const { t } = useI18n();
  const status = useGitStore((s) => s.status);
  const busyOp = useGitStore((s) => s.busyOp);
  const lastError = useGitStore((s) => s.lastError);
  const loadError = useGitStore((s) => s.loadError);
  const hideLayout = useGitStore((s) => s.hideLayout);
  const refresh = useGitStore((s) => s.refresh);
  const setHideLayout = useGitStore((s) => s.setHideLayout);
  const openIdentityForm = useGitStore((s) => s.openIdentityForm);
  const dismissError = useGitStore((s) => s.dismissError);
  const [detailsOpen, setDetailsOpen] = useState(false);

  // What the filter is swallowing RIGHT NOW: zero while it is off, so the
  // count appears as it is switched on and the menu stays open to show it.
  const hiddenCount = hideLayout
    ? [...(status?.unstaged ?? []), ...(status?.untracked ?? [])].filter((file) =>
      isLayoutFile(file.path),
    ).length
    : 0;

  const items: ActionMenuItem[] = [
    {
      id: 'hideLayout',
      label:
        hiddenCount > 0
          ? `${t('git.menu.hideLayout')} ${t('git.menu.hiddenCount', { count: hiddenCount })}`
          : t('git.menu.hideLayout'),
      checked: hideLayout,
      onSelect: () => setHideLayout(!hideLayout),
    },
    {
      id: 'identity',
      label: t('git.action.identity'),
      onSelect: () => openIdentityForm(),
    },
    {
      id: 'docs',
      label: t('git.action.docs'),
      onSelect: () => window.open(docsUrl(SCM_DOCS_PATH), '_blank', 'noopener,noreferrer'),
    },
  ];

  const branchText = status === null
    ? null
    : status.detached
      ? t('git.detached')
      : status.unborn
        ? t('git.unborn')
        : t('git.branch.label', { name: status.branch ?? '' });

  // A branch with no commits has nothing to push and no upstream to lack, so
  // it gets no tracking half at all rather than "Not published" beside "No
  // commits yet", which says the same thing twice.
  const trackingText = status === null || status.unborn
    ? null
    : status.upstream_gone
      ? t('git.upstreamGone')
      : status.upstream === null
        ? t('git.noUpstream')
        : t('git.aheadBehind', { ahead: status.ahead ?? 0, behind: status.behind ?? 0 });

  const stderr = lastError?.stderr ?? null;

  return (
    <div className={shell.header}>
      <div className={shell.headerRow}>
        <div className={shell.headerTitle}>{t('sidebar.tab.git')}</div>
        <button
          type="button"
          className={shell.toolbarButton}
          onClick={() => void refresh()}
          aria-label={t('sidebar.refresh')}
          title={t('sidebar.refresh')}
        >
          <RefreshIcon size={13} />
        </button>
        <ActionMenu
          label={t('git.action.more')}
          items={items}
          align="end"
          className={shell.toolbarButton}
        >
          <MoreHorizontalIcon size={13} />
        </ActionMenu>
      </div>
      {branchText !== null && (
        <div className={styles.branchRow}>
          <span className={styles.branchName} title={branchText}>
            {branchText}
          </span>
          {trackingText !== null && (
            <span className={styles.tracking} title={trackingText}>
              {trackingText}
            </span>
          )}
        </div>
      )}
      {busyOp !== null && (
        <div className={styles.busy}>
          <ProgressBar
            value={null}
            size="sm"
            label={t('git.busy', { op: t(gitOpKey(busyOp)) })}
          />
        </div>
      )}
      {(lastError !== null || loadError !== null) && (
        <div className={styles.error} role="alert">
          {loadError !== null && (
            <div className={styles.errorMessage}>
              {t('git.error.loadFail', { error: loadError })}
            </div>
          )}
          {lastError !== null && (
            <>
              <div className={styles.errorMessage}>{errorSentence(lastError, t)}</div>
              {lastError.hint !== null && (
                <div className={styles.errorHint}>{lastError.hint}</div>
              )}
              <div className={styles.errorActions}>
                {stderr !== null && stderr !== '' && (
                  <button
                    type="button"
                    className={styles.linkButton}
                    aria-expanded={detailsOpen}
                    onClick={() => setDetailsOpen((was) => !was)}
                  >
                    {t('git.error.details')}
                  </button>
                )}
                <button
                  type="button"
                  className={styles.linkButton}
                  onClick={() => {
                    setDetailsOpen(false);
                    dismissError();
                  }}
                >
                  {t('git.error.dismiss')}
                </button>
              </div>
              {detailsOpen && stderr !== null && stderr !== '' && (
                <pre className={styles.errorStderr}>{stderr}</pre>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
