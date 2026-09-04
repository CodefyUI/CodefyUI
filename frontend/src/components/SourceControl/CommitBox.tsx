import { useCallback, useLayoutEffect, useRef, type KeyboardEvent } from 'react';
import { useGitStore } from '../../store/gitStore';
import { useI18n } from '../../i18n';
import { MOD_LABEL } from '../../utils/platform';
import { ChevronDownIcon } from '../shared/Icons';
import { ActionMenu, type ActionMenuItem } from '../shared/ActionMenu';
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
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Grows with the text up to the `max-height` in the stylesheet, then
  // scrolls. Reset to `auto` first, or the box could only ever get taller.
  useLayoutEffect(() => {
    const el = inputRef.current;
    if (el === null) return;
    el.style.height = 'auto';
    el.style.height = `${el.scrollHeight}px`;
  }, [message]);

  const hasMessage = message.trim() !== '';
  const stagedCount = status?.staged.length ?? 0;
  const canCommit = hasMessage && stagedCount > 0;
  // One reason, the first one that applies: a box with no message and nothing
  // staged has one thing to do next, not two.
  const blockedBecause = !hasMessage
    ? t('git.commit.needMessage')
    : stagedCount === 0
      ? t('git.commit.nothingStaged')
      : undefined;

  // git refuses `--amend` before it runs when the branch has no commit yet,
  // and rewriting a commit that is already on a remote is the mistake that
  // costs somebody else a forced pull.
  const unborn = status?.unborn === true;
  const alreadyPushed = status?.upstream != null && status.ahead === 0;
  const amendBlocked = unborn || alreadyPushed;

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
      if (canCommit) runCommit(false);
    },
    [canCommit, runCommit],
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
        rows={1}
        value={message}
        placeholder={t('git.commit.placeholder', { mod: MOD_LABEL })}
        onChange={(e) => setCommitMessage(e.target.value)}
        onKeyDown={onKeyDown}
      />
      <div className={styles.commitRow}>
        {amend && <span className={styles.amendChip}>{t('git.commit.amending')}</span>}
        <button
          type="button"
          className={styles.commitButton}
          disabled={!canCommit}
          title={blockedBecause}
          onClick={() => runCommit(false)}
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
