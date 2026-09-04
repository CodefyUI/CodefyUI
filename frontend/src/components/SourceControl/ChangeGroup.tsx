import { useCallback, useId, useRef, useState } from 'react';
import type { GitFile } from '../../api/git';
import { useGitStore } from '../../store/gitStore';
import { useI18n, type TranslationKey } from '../../i18n';
import { confirm } from '../../utils/dialog';
import { ChevronDownIcon, DiscardIcon, MinusIcon, PlusIcon } from '../shared/Icons';
import { FileRow, type ChangeGroupKind } from './FileRow';
import styles from './SourceControl.module.css';

const TITLE_KEY: Record<ChangeGroupKind, TranslationKey> = {
  merge: 'git.group.merge',
  staged: 'git.group.staged',
  changes: 'git.group.changes',
};

export interface ChangeGroupProps {
  kind: ChangeGroupKind;
  /** The rows to draw. Already filtered -- see `SourceControlTab`. */
  files: GitFile[];
}

/**
 * One titled, collapsible list of files, with the actions that apply to all of
 * them.
 *
 * The Merge group has no group-level actions at all. "Stage All" in the middle
 * of a merge means "mark every conflict resolved", which is a decision per
 * file and not one a group header should be able to make by accident; the
 * per-row Stage is still there for the file the user has actually fixed.
 *
 * "Discard All" asks with the counts from the WHOLE status, not from the rows
 * on screen: `discard('all')` is a whole-tree operation, so a filter that is
 * hiding layout files must not make the dialog under-report what is about to
 * be thrown away.
 */
export function ChangeGroup({ kind, files }: ChangeGroupProps) {
  const { t } = useI18n();
  const status = useGitStore((s) => s.status);
  const stage = useGitStore((s) => s.stage);
  const unstage = useGitStore((s) => s.unstage);
  const discard = useGitStore((s) => s.discard);
  const [open, setOpen] = useState(true);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const domId = useId();
  const headingId = `${domId}-heading`;
  const listId = `${domId}-list`;

  // A row that was staged, unstaged or discarded is no longer in this list, so
  // the button that did it is gone and focus would land on the document body.
  const returnFocus = useCallback(() => toggleRef.current?.focus(), []);

  const askThenDiscardAll = useCallback(async () => {
    const ok = await confirm({
      title: t('git.discard.confirmAll', {
        changed: status?.unstaged.length ?? 0,
        untracked: status?.untracked.length ?? 0,
      }),
      confirmText: t('git.discard.action'),
      variant: 'danger',
    });
    if (!ok) return;
    if (await discard('all')) returnFocus();
  }, [discard, returnFocus, status, t]);

  return (
    <section className={styles.group} aria-labelledby={headingId}>
      <div className={styles.groupHeader}>
        <button
          type="button"
          ref={toggleRef}
          id={headingId}
          className={styles.groupToggle}
          aria-expanded={open}
          aria-controls={listId}
          // The heading is the first thing to lose room at a 180px panel
          // width, where the group actions have already claimed 50 of it.
          title={t(TITLE_KEY[kind])}
          onClick={() => setOpen((was) => !was)}
        >
          <span
            className={`${styles.chevron} ${open ? '' : styles.chevronCollapsed}`}
          >
            <ChevronDownIcon size={12} />
          </span>
          <span className={styles.groupTitle}>{t(TITLE_KEY[kind])}</span>
        </button>
        <span className={styles.groupCount}>{files.length}</span>
        <div className={styles.groupActions}>
          {kind === 'changes' && (
            <>
              <button
                type="button"
                className={styles.iconButton}
                aria-label={t('git.group.stageAll')}
                title={t('git.group.stageAll')}
                onClick={() => {
                  void stage('all').then((ok) => {
                    if (ok) returnFocus();
                  });
                }}
              >
                <PlusIcon size={13} />
              </button>
              <button
                type="button"
                className={`${styles.iconButton} ${styles.dangerButton}`}
                aria-label={t('git.group.discardAll')}
                title={t('git.group.discardAll')}
                onClick={() => void askThenDiscardAll()}
              >
                <DiscardIcon size={13} />
              </button>
            </>
          )}
          {kind === 'staged' && (
            <button
              type="button"
              className={styles.iconButton}
              aria-label={t('git.group.unstageAll')}
              title={t('git.group.unstageAll')}
              onClick={() => {
                void unstage('all').then((ok) => {
                  if (ok) returnFocus();
                });
              }}
            >
              <MinusIcon size={13} />
            </button>
          )}
        </div>
      </div>
      {/* `role="list"` is spelled out because `list-style: none` takes the
          list semantics away from a `<ul>` in Safari. */}
      <ul id={listId} className={styles.list} role="list" hidden={!open}>
        {files.map((file) => (
          <FileRow
            // One path can be in the status twice with the same two letters
            // and a different kind, so the kind is part of the identity.
            key={`${file.path}:${file.xy}:${file.kind}`}
            file={file}
            group={kind}
            onActed={returnFocus}
          />
        ))}
      </ul>
    </section>
  );
}
