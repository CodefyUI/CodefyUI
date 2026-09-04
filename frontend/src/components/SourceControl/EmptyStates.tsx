import type { RepoState } from '../../api/git';
import { useI18n } from '../../i18n';
import { docsUrl } from '../../utils/docsUrl';
import { CommandBlock } from '../PackCenter/CommandBlock';
import { SCM_DOCS_PATH } from './ScmHeader';
import styles from './SourceControl.module.css';

/** The two commands that turn a plain server into one this tab can work in. */
const INIT_COMMAND = 'cdui project init my-project';
const START_COMMAND = 'cdui start --project my-project';

export interface EmptyStatesProps {
  /** Every repository state except `ready`, which is the panel itself. */
  state: Exclude<RepoState, 'ready'>;
  /** The outer repository this project sits inside, when it does. */
  nestedToplevel: string | null;
  /** `git --version`'s answer; null when the server could not read it. */
  gitVersion: string | null;
  onInit: () => void;
}

/**
 * What the tab shows when there is no repository to show.
 *
 * Four screens, and each one is a sentence plus the single next step:
 *
 *  - **no project** -- the server was started without one. Nothing in the app
 *    can fix that, so the screen is the two commands to run and a link to the
 *    guide. It deliberately does not offer to pick a directory: the project is
 *    a server argument, and a button here would be a button that cannot work.
 *  - **not a repository** -- one button, which is the whole fix. When the
 *    directory sits inside ANOTHER repository the screen says so first,
 *    because "Initialize" then means a second, separate repository and that is
 *    a decision, not a formality.
 *  - **git missing / too old** -- both are the server computer's problem and
 *    both end the same way: install it and restart the server.
 */
export function EmptyStates({
  state,
  nestedToplevel,
  gitVersion,
  onInit,
}: EmptyStatesProps) {
  const { t } = useI18n();

  if (state === 'no_project') {
    return (
      <div className={styles.emptyBody}>
        <p className={styles.emptyText}>{t('git.empty.noProject')}</p>
        <p className={styles.emptyHint}>{t('git.empty.noProjectHint')}</p>
        <CommandBlock command={INIT_COMMAND} />
        <CommandBlock command={START_COMMAND} />
        <a
          className={styles.docsLink}
          href={docsUrl(SCM_DOCS_PATH)}
          target="_blank"
          rel="noopener noreferrer"
        >
          {t('git.action.docs')}
        </a>
      </div>
    );
  }

  if (state === 'not_repo') {
    return (
      <div className={styles.emptyBody}>
        <p className={styles.emptyText}>{t('git.empty.notRepo')}</p>
        {nestedToplevel !== null && (
          <p className={styles.emptyHint}>
            {t('git.empty.nested', { path: nestedToplevel })}
          </p>
        )}
        <button type="button" className={styles.primaryButton} onClick={onInit}>
          {t('git.empty.init')}
        </button>
      </div>
    );
  }

  return (
    <div className={styles.emptyBody}>
      <p className={styles.emptyText}>
        {state === 'git_missing'
          ? t('git.empty.gitMissing')
          // A `git_too_old` with no version is git answering `--version` with
          // something unreadable, which is still too old for this tab. The
          // placeholder says the number is the part nobody could read.
          : t('git.empty.gitTooOld', { version: gitVersion ?? '?' })}
      </p>
      <p className={styles.emptyHint}>{t('git.empty.gitMissingHint')}</p>
    </div>
  );
}
