import { useCallback, useId, useRef, useState } from 'react';
import type { GitFile, GitResolveSide } from '../../api/git';
import { useGitStore } from '../../store/gitStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n, type TranslationKey } from '../../i18n';
import { confirm } from '../../utils/dialog';
import { ActionMenu } from '../shared/ActionMenu';
import { ChevronDownIcon, CloseIcon, MoreHorizontalIcon } from '../shared/Icons';
import { displayPath, FileKindChip, FileRowName, fileRowLabel } from './FileRow';
import { focusScmFallback } from './ScmHeader';
import styles from './SourceControl.module.css';

/** The three ways out of one conflict, in the order they are offered. */
const SIDES: Array<{ side: GitResolveSide; key: TranslationKey }> = [
  { side: 'ours', key: 'git.merge.ours' },
  { side: 'theirs', key: 'git.merge.theirs' },
  { side: 'mark', key: 'git.merge.mark' },
];

export interface MergeGroupProps {
  /** The conflicted files. Already filtered -- see `SourceControlTab`. */
  files: GitFile[];
}

/**
 * The conflict list, and the two ways a merge ends.
 *
 * Not a `ChangeGroup`: every other group's rows are bookkeeping -- stage this,
 * unstage that, and the file on disk never moves -- while each row here is a
 * DECISION that overwrites the file. So the verbs are git's own ("ours",
 * "theirs", and the manual resolution that is already saved), and Discard is
 * absent rather than disabled: `discard` refuses a conflicted path outright
 * (400 `path_not_in_status` -- a conflict is in neither the unstaged nor the
 * untracked list the discard is built from), so the button could only offer an
 * error.
 *
 * The banner belongs to this group and not to the tab above it. It is one
 * sentence about one situation, and the tab drew it a few pixels above the
 * group it describes, which is the same fact twice as soon as the group has a
 * heading of its own.
 *
 * Drawn while a merge is running even with nothing left to resolve. Settling
 * every file as "mine" changes no file, so the index is empty and the list is
 * empty -- and the only two ways out of `MERGE_HEAD` from there are the commit
 * box (which `CommitBox` allows for exactly this reason) and this Abort.
 */
export function MergeGroup({ files }: MergeGroupProps) {
  const { t } = useI18n();
  const merging = useGitStore((s) => s.status?.merge_in_progress === true);
  const resolve = useGitStore((s) => s.resolve);
  const abortMerge = useGitStore((s) => s.abortMerge);
  const openGitDiff = useUIStore((s) => s.openGitDiff);
  const [open, setOpen] = useState(true);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const domId = useId();
  const headingId = `${domId}-heading`;
  const listId = `${domId}-list`;

  // A file that has been settled is no longer conflicted, so the button that
  // settled it is gone and focus would land on the document body. On the next
  // frame, because the whole GROUP can be what disappears -- the last
  // resolution ends the list, and an abort ends the merge. See the same note
  // in `ChangeGroup`.
  const returnFocus = useCallback(() => {
    requestAnimationFrame(() => {
      const toggle = toggleRef.current;
      if (toggle !== null && toggle.isConnected) {
        toggle.focus();
        return;
      }
      focusScmFallback();
    });
  }, []);

  const askThenAbort = useCallback(async () => {
    const ok = await confirm({
      // A QUESTION, the way every other destructive confirm in this panel
      // asks one ("Delete branch work?", "Drop stash stash@{0}?"). The
      // banner's own sentence stood here, which made the heading of a modal
      // opened by Abort Merge an instruction to resolve and commit -- the
      // opposite of what was pressed, and the same sentence twice on screen,
      // since the banner is still behind it.
      title: t('git.merge.abortConfirm'),
      confirmText: t('git.action.abortMerge'),
      variant: 'danger',
    });
    if (!ok) return;
    if (await abortMerge()) returnFocus();
  }, [abortMerge, returnFocus, t]);

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
          title={t('git.group.merge')}
          onClick={() => setOpen((was) => !was)}
        >
          <span className={`${styles.chevron} ${open ? '' : styles.chevronCollapsed}`}>
            <ChevronDownIcon size={12} />
          </span>
          <span className={styles.groupTitle}>{t('git.group.merge')}</span>
        </button>
        {/* Title, then the actions, then the count -- see `ChangeGroup`. */}
        <div className={styles.groupActions}>
          {merging && (
            <button
              type="button"
              className={`${styles.iconButton} ${styles.dangerButton}`}
              aria-label={t('git.action.abortMerge')}
              title={t('git.action.abortMerge')}
              onClick={() => void askThenAbort()}
            >
              <CloseIcon size={13} />
            </button>
          )}
        </div>
        <span className={styles.groupCount}>{files.length}</span>
      </div>
      {/* Outside the list, and outside the disclosure: collapsing the rows
          away does not end the merge, and this line is the state of the
          repository rather than one of the rows. */}
      {merging && <div className={styles.banner}>{t('git.merge.banner')}</div>}
      {/* `role="list"` is spelled out because `list-style: none` takes the
          list semantics away from a `<ul>` in Safari. */}
      <ul id={listId} className={styles.list} role="list" hidden={!open}>
        {files.map((file) => (
          <MergeRow
            key={`${file.path}:${file.xy}`}
            file={file}
            onOpen={() => openGitDiff({
              // The working tree, like any other change -- and flagged, because
              // a conflicted path has no stage 0 until it is settled, so its
              // index side cannot be read and the modal shows Unified only.
              path: file.path,
              scope: 'worktree',
              conflicted: true,
            })}
            onResolve={(side) => {
              void resolve(file.path, side).then((ok) => {
                if (ok) returnFocus();
              });
            }}
          />
        ))}
      </ul>
    </section>
  );
}

/**
 * One conflicted file, with both shapes of its three actions.
 *
 * Both are always in the DOM and CSS picks one (`@container` on the group; see
 * `.rowChoices` in the stylesheet, which every list's rows share): "Keep mine
 * / Take incoming / Mark resolved" is about 210px of buttons in either
 * language, which is more than a 180px panel HAS, and a menu is the only
 * honest way to offer three choices in a row that narrow. The hidden half is
 * `display: none`, so it is out of the accessibility tree as well as off the
 * screen and nothing is announced twice.
 *
 * The actions are not hidden behind a hover the way every other row's are. A
 * conflicted row exists in order to be settled; a list of them with no visible
 * way to settle any looks like a dead end, which is the opposite of what a
 * person in the middle of a merge needs.
 */
function MergeRow({
  file,
  onOpen,
  onResolve,
}: {
  file: GitFile;
  onOpen: () => void;
  onResolve: (side: GitResolveSide) => void;
}) {
  const { t } = useI18n();
  const label = fileRowLabel(file);

  return (
    // The path lives on the ROW -- see `FileRow`.
    <li className={styles.row} title={displayPath(file)}>
      <FileKindChip kind={file.kind} />
      <FileRowName file={file} onOpen={onOpen} />
      <div className={styles.mergeActions}>
        <div className={styles.rowChoices}>
          {SIDES.map(({ side, key }) => (
            <button
              key={side}
              type="button"
              className={styles.rowAction}
              // The verb NAMES the file it overwrites, and only the tooltip is
              // the bare word: three identical verbs per row, twenty rows
              // deep, is nothing for a reader to tell apart.
              aria-label={`${t(key)} ${label}`}
              title={t(key)}
              onClick={() => onResolve(side)}
            >
              {t(key)}
            </button>
          ))}
        </div>
        <div className={styles.rowMenu}>
          {/* The trigger carries the file name, so the rows inside it do not
              have to: by then the menu's own name has established which file
              this is, and a row called "Keep mine train.py" inside a menu
              called "More actions train.py" says it twice. */}
          <ActionMenu
            label={`${t('git.action.more')} ${label}`}
            items={SIDES.map(({ side, key }) => ({
              id: side,
              label: t(key),
              onSelect: () => onResolve(side),
            }))}
            align="end"
            className={styles.iconButton}
          >
            <MoreHorizontalIcon size={13} />
          </ActionMenu>
        </div>
      </div>
    </li>
  );
}
