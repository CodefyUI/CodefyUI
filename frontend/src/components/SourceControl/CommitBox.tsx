import { useCallback, useLayoutEffect, useRef, type KeyboardEvent } from 'react';
import { useGitStore } from '../../store/gitStore';
import { useI18n } from '../../i18n';
import { MOD_LABEL } from '../../utils/platform';
import { ChevronDownIcon } from '../shared/Icons';
import { ActionMenu, type ActionMenuItem } from '../shared/ActionMenu';
import { SCM_FOCUS } from './ScmHeader';
import styles from './SourceControl.module.css';

/**
 * The message box and the button that commits it.
 *
 * A split button, the way every git client draws it: the common action is one
 * press, and the two that need a decision (commit everything, replace the last
 * commit) live behind the chevron rather than as two more buttons competing
 * for a 180px row.
 *
 * The Commit button says WHY it is off rather than just being off -- an empty
 * message and an empty index are different problems with different fixes, and
 * a disabled button with no reason is the thing people file bugs about.
 */
export function CommitBox() {
  const { t } = useI18n();
  const status = useGitStore((s) => s.status);
  const message = useGitStore((s) => s.commitMessage);
  const amend = useGitStore((s) => s.amend);
  const setCommitMessage = useGitStore((s) => s.setCommitMessage);
  const setAmend = useGitStore((s) => s.setAmend);
  const commit = useGitStore((s) => s.commit);
  const announce = useGitStore((s) => s.announce);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Grows with the text up to the `max-height` in the stylesheet, then
  // scrolls. Reset to `auto` first, or the box could only ever get taller.
  // `scrollHeight` is the CONTENT box and the element is `border-box`, so the
  // border and padding have to be added back or the box lands two pixels
  // short and scrolls a line early.
  useLayoutEffect(() => {
    const el = inputRef.current;
    if (el === null) return;
    el.style.height = 'auto';
    el.style.height = `${el.scrollHeight + el.offsetHeight - el.clientHeight}px`;
  }, [message]);

  // git refuses `--amend` before it runs when the branch has no commit yet,
  // and rewriting a commit that is already on a remote is the mistake that
  // costs somebody else a forced pull.
  const unborn = status?.unborn === true;
  const alreadyPushed = status !== null && status.upstream !== null && status.ahead === 0;
  const amendBlocked = unborn || alreadyPushed;

  const hasMessage = message.trim() !== '';
  const stagedCount = status?.staged.length ?? 0;
  // A merge commit is the only way out of MERGE_HEAD, and resolving every
  // conflict as "mine" changes no file -- so the index is empty and git still
  // wants the commit. "Nothing staged" would be a dead end with the tree in a
  // state only a commit or an abort can leave.
  const merging = status?.merge_in_progress === true;
  // An amend with an empty index is a real commit: it rewrites the last one's
  // MESSAGE, which is the commonest reason to amend at all, and the backend
  // takes it. So "nothing staged" only blocks a new commit.
  const canCommit = hasMessage && (stagedCount > 0 || amend || merging);
  // One reason, the first one that applies: a box with no message and nothing
  // staged has one thing to do next, not two.
  const blockedBecause = !hasMessage
    ? t('git.commit.needMessage')
    : stagedCount === 0 && !amend && !merging
      ? t('git.commit.nothingStaged')
      : undefined;

  const runCommit = useCallback(
    (all: boolean) => {
      void commit({ all });
    },
    [commit],
  );

  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key !== 'Enter' || !(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      if (canCommit) {
        runCommit(false);
        return;
      }
      // The chord goes past the button, and so past the button's tooltip: the
      // hand that never touched the mouse would get a keystroke that does
      // nothing and says nothing. The same reason the button carries, spoken.
      if (blockedBecause !== undefined) announce(blockedBecause);
    },
    [announce, blockedBecause, canCommit, runCommit],
  );

  const items: ActionMenuItem[] = [
    {
      id: 'all',
      // Commit All stages the tracked changes itself, so it is the one entry
      // that does not care whether anything is in the index yet.
      label: t('git.commit.all'),
      disabled: !hasMessage,
      onSelect: () => runCommit(true),
    },
    {
      id: 'amend',
      // A blocked row says what is wrong in place of its own name: the menu
      // has no tooltip, and a row that is merely grey explains nothing.
      label: alreadyPushed ? t('git.commit.amendPushed') : t('git.commit.amend'),
      checked: amend,
      disabled: amendBlocked,
      onSelect: () => setAmend(!amend),
    },
  ];

  return (
    <div className={styles.commitBox}>
      <textarea
        ref={inputRef}
        className={styles.commitInput}
        data-scm-focus={SCM_FOCUS.commit}
        rows={1}
        value={message}
        placeholder={t('git.commit.placeholder', { mod: MOD_LABEL })}
        onChange={(e) => setCommitMessage(e.target.value)}
        onKeyDown={onKeyDown}
      />
      <div className={styles.commitRow}>
        {amend && <span className={styles.amendChip}>{t('git.commit.amending')}</span>}
        {/*
          `aria-disabled`, not `disabled`. A disabled button opens no tooltip
          in Chrome and takes no focus, so the very button that has a reason to
          give would be the one that could not give it. This one is hoverable,
          focusable and announced as unavailable, and the press is refused in
          the handler.
        */}
        <button
          type="button"
          className={styles.commitButton}
          aria-disabled={!canCommit}
          title={blockedBecause}
          onClick={() => {
            if (!canCommit) return;
            runCommit(false);
          }}
        >
          {t('git.commit.button')}
        </button>
        <ActionMenu
          label={t('git.commit.options')}
          items={items}
          className={styles.commitChevron}
        >
          <ChevronDownIcon size={13} />
        </ActionMenu>
      </div>
    </div>
  );
}
