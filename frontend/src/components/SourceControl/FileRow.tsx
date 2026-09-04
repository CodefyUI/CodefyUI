import { useCallback } from 'react';
import type { FileKind, GitFile } from '../../api/git';
import { useGitStore } from '../../store/gitStore';
import { useI18n, type TranslationKey } from '../../i18n';
import { confirm } from '../../utils/dialog';
import { DiscardIcon, MinusIcon, PlusIcon } from '../shared/Icons';
import styles from './SourceControl.module.css';

/** Which group a row is being drawn in, which is what decides its actions. */
export type ChangeGroupKind = 'merge' | 'staged' | 'changes';

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
 *  - a MERGE row can only be staged, because staging is how git marks a
 *    conflict resolved -- and `discard` refuses a conflicted path outright
 *    (400 `path_not_in_status`: it is in neither the unstaged nor the
 *    untracked list the discard is built from), so offering the button would
 *    be offering an error.
 */
export function FileRow({ file, group, onActed }: FileRowProps) {
  const { t } = useI18n();
  const stage = useGitStore((s) => s.stage);
  const unstage = useGitStore((s) => s.unstage);
  const discard = useGitStore((s) => s.discard);

  const shown = displayPath(file);
  const { name, dir } = splitPath(file.path);
  const label = file.orig_path === null
    ? name
    : `${splitPath(file.orig_path).name} -> ${name}`;

  const canStage = group === 'changes' || group === 'merge';
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
    <li className={styles.row}>
      <span
        className={styles.chip}
        data-kind={file.kind}
        role="img"
        aria-label={t(KIND_KEY[file.kind])}
        title={t(KIND_KEY[file.kind])}
      >
        {KIND_LETTER[file.kind]}
      </span>
      {/*
        The row's own button. Disabled: opening the change is the diff view's
        job and that is not in this build, and a button that looks live and
        does nothing is worse than one that says it cannot yet. It carries no
        tooltip of its own beyond the path, which is the fact a truncated row
        is actually missing.
      */}
      <button type="button" className={styles.openButton} title={shown} disabled>
        <span className={styles.rowName}>{label}</span>
        {dir !== '' && <span className={styles.rowDir}>{dir}</span>}
      </button>
      <div className={styles.rowActions}>
        {canStage && (
          <button
            type="button"
            className={styles.iconButton}
            aria-label={t('git.file.stage')}
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
            aria-label={t('git.file.unstage')}
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
            aria-label={t('git.file.discard')}
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
