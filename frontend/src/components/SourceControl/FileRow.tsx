import { useCallback } from 'react';
import type { FileKind, GitFile } from '../../api/git';
import { useGitStore } from '../../store/gitStore';
import { useI18n, type TranslationKey } from '../../i18n';
import { confirm } from '../../utils/dialog';
import { DiscardIcon, MinusIcon, PlusIcon } from '../shared/Icons';
import styles from './SourceControl.module.css';

/**
 * Which group a row is being drawn in, which is what decides its actions.
 *
 * No `merge`: a conflicted file's row is `MergeGroup`'s, because settling one
 * is a decision that overwrites the file rather than a move between two lists.
 */
export type ChangeGroupKind = 'staged' | 'changes';

/**
 * The letter in the chip.
 *
 * git's own vocabulary, so a reader who has used any other git client already
 * knows it. `!` for a conflict rather than `U`: `U` is the porcelain record
 * type, and every other letter here is what the file HAD done to it.
 */
const KIND_LETTER: Record<FileKind, string> = {
  modified: 'M',
  added: 'A',
  deleted: 'D',
  renamed: 'R',
  copied: 'C',
  typechange: 'T',
  untracked: 'U',
  conflict: '!',
};

const KIND_KEY: Record<FileKind, TranslationKey> = {
  modified: 'git.status.modified',
  added: 'git.status.added',
  deleted: 'git.status.deleted',
  renamed: 'git.status.renamed',
  copied: 'git.status.copied',
  typechange: 'git.status.typechange',
  untracked: 'git.status.untracked',
  conflict: 'git.status.conflict',
};

/** The part after the last separator, and everything before it. */
function splitPath(path: string): { name: string; dir: string } {
  const cut = path.lastIndexOf('/');
  return cut < 0
    ? { name: path, dir: '' }
    : { name: path.slice(cut + 1), dir: path.slice(0, cut) };
}

/**
 * What the row says it is.
 *
 * A rename (and a copy, which carries the same field) is TWO paths, and only
 * showing the new one loses the fact that a file moved rather than appeared.
 * The names are joined with an ASCII arrow so the pair survives a terminal,
 * a bug report and a locale with no such glyph.
 */
export function displayPath(file: GitFile): string {
  return file.orig_path === null ? file.path : `${file.orig_path} -> ${file.path}`;
}

/**
 * What the row shows for a file, which is what its buttons are named after.
 *
 * The basename, or `old -> new` for a rename. A panel is twenty rows deep and
 * every row carries the same verbs, so an action's accessible name is the verb
 * plus THIS -- and the string a reader hears has to be the string they can see.
 */
export function fileRowLabel(file: GitFile): string {
  const { name } = splitPath(file.path);
  return file.orig_path === null
    ? name
    : `${splitPath(file.orig_path).name} -> ${name}`;
}

/** One letter, tinted by what happened to the file, named in full for a reader. */
export function FileKindChip({ kind }: { kind: FileKind }) {
  const { t } = useI18n();
  return (
    <span
      className={styles.chip}
      data-kind={kind}
      role="img"
      aria-label={t(KIND_KEY[kind])}
      title={t(KIND_KEY[kind])}
    >
      {KIND_LETTER[kind]}
    </span>
  );
}

/**
 * The name half of a row: the file, with its directory under it.
 *
 * One button, inert here and the diff view's target later. It carries no
 * tooltip of its own -- the row around it holds the path, because a `title` on
 * a disabled element never opens in Chrome and a truncated name is exactly
 * when the reader needs it.
 */
export function FileRowName({ file }: { file: GitFile }) {
  const { dir } = splitPath(file.path);
  return (
    <button type="button" className={styles.openButton} disabled>
      <span className={styles.rowName}>{fileRowLabel(file)}</span>
      {dir !== '' && <span className={styles.rowDir}>{dir}</span>}
    </button>
  );
}

export interface FileRowProps {
  file: GitFile;
  group: ChangeGroupKind;
  /**
   * Called after an action that succeeded. The row it was pressed on has
   * moved to another group by then, so the focus it held no longer exists —
   * the group moves it to its own header rather than letting it fall to the
   * document body.
   */
  onActed?: () => void;
}

/**
 * One file in one group.
 *
 * The actions on it are decided by the group it is in and by the backend's
 * rules, not by taste:
 *
 *  - a STAGED row can only be unstaged (its worktree copy is somebody else's
 *    row, further down);
 *  - a CHANGES row can be staged or discarded;
 *  - a CONFLICTED file is never discarded, whichever list it turns up in:
 *    `discard` refuses the path outright (400 `path_not_in_status` -- it is in
 *    neither the unstaged nor the untracked list the discard is built from),
 *    so offering the button would be offering an error. The status keeps
 *    conflicts in a list of their own, which `MergeGroup` draws.
 */
export function FileRow({ file, group, onActed }: FileRowProps) {
  const { t } = useI18n();
  const stage = useGitStore((s) => s.stage);
  const unstage = useGitStore((s) => s.unstage);
  const discard = useGitStore((s) => s.discard);

  const shown = displayPath(file);
  const label = fileRowLabel(file);

  const canStage = group === 'changes';
  const canUnstage = group === 'staged';
  // Never on a conflict, whichever group one turns up in.
  const canDiscard = group === 'changes' && file.kind !== 'conflict';

  const run = useCallback(
    async (action: () => Promise<boolean>) => {
      const ok = await action();
      if (ok) onActed?.();
    },
    [onActed],
  );

  const askThenDiscard = useCallback(async () => {
    const ok = await confirm({
      title:
        file.kind === 'untracked'
          ? t('git.discard.confirmUntracked', { name: file.path })
          : t('git.discard.confirm', { name: file.path }),
      confirmText: t('git.discard.action'),
      variant: 'danger',
    });
    if (!ok) return;
    await run(() => discard([file.path]));
  }, [discard, file.kind, file.path, run, t]);

  return (
    // The path lives on the ROW, not on the button inside it: a `title` on a
    // disabled element never opens in Chrome, and a truncated name is exactly
    // when the reader needs it.
    <li className={styles.row} title={shown}>
      <FileKindChip kind={file.kind} />
      {/*
        Opening the change is the diff view's job and that is not in this
        build, so the name button is inert -- see `FileRowName`.
      */}
      <FileRowName file={file} />
      {/*
        The verb NAMES the file it acts on, and only the tooltip is the bare
        word. A panel is twenty rows deep and every one of them carries a
        Stage button: a reader moving through them by keyboard would otherwise
        hear "Stage, Stage, Stage" with nothing to tell them apart, and the
        row's `title` is not part of any of those names. The composed label is
        the displayed name -- the basename, or `old -> new` for a rename --
        because that is the string on screen beside the button.
      */}
      <div className={styles.rowActions}>
        {canStage && (
          <button
            type="button"
            className={styles.iconButton}
            aria-label={`${t('git.file.stage')} ${label}`}
            title={t('git.file.stage')}
            onClick={() => void run(() => stage([file.path]))}
          >
            <PlusIcon size={13} />
          </button>
        )}
        {canUnstage && (
          <button
            type="button"
            className={styles.iconButton}
            aria-label={`${t('git.file.unstage')} ${label}`}
            title={t('git.file.unstage')}
            onClick={() => void run(() => unstage([file.path]))}
          >
            <MinusIcon size={13} />
          </button>
        )}
        {canDiscard && (
          <button
            type="button"
            className={`${styles.iconButton} ${styles.dangerButton}`}
            aria-label={`${t('git.file.discard')} ${label}`}
            title={t('git.file.discard')}
            onClick={() => void askThenDiscard()}
          >
            <DiscardIcon size={13} />
          </button>
        )}
      </div>
    </li>
  );
}
