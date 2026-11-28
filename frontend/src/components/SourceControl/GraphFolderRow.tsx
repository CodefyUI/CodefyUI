import { useCallback, useEffect, useId, useMemo, useState } from 'react';
import { getGitGraph, type GitFile } from '../../api/git';
import { useGitStore } from '../../store/gitStore';
import { useI18n } from '../../i18n';
import { confirm } from '../../utils/dialog';
import { folderKind } from '../../utils/graphFolders';
import {
  summarizeGraphDiff,
  type GraphDiffSummary as GraphDiff,
} from '../../utils/graphDiff';
import { ChevronDownIcon, DiscardIcon, MinusIcon, PlusIcon } from '../shared/Icons';
import { FileKindChip, FileRow, type ChangeGroupKind } from './FileRow';
import { GraphDiffSummary } from './GraphDiffSummary';
import styles from './SourceControl.module.css';

export interface GraphFolderRowProps {
  /** The graph's file stem: the folder is `graphs/<name>/`. */
  name: string;
  /** The changed files under that folder, in this group. */
  files: GitFile[];
  group: ChangeGroupKind;
  onActed?: () => void;
  onOpen: (file: GitFile) => void;
}

/**
 * One graph stored as a folder, as ONE row: a save that changed two nodes is
 * two node files and an edge list, and the reader wants "this graph changed,
 * like this" rather than three file names. The row says what changed in the
 * graph (the two sides from `GET /api/git/graph`), opens into the files, and
 * stages, unstages or discards the whole folder plus its layout file.
 */
export function GraphFolderRow({ name, files, group, onActed, onOpen }: GraphFolderRowProps) {
  const { t } = useI18n();
  const status = useGitStore((s) => s.status);
  const stage = useGitStore((s) => s.stage);
  const unstage = useGitStore((s) => s.unstage);
  const discard = useGitStore((s) => s.discard);
  const [open, setOpen] = useState(false);
  const [summary, setSummary] = useState<GraphDiff | null>(null);
  const listId = useId();

  // The folder's files plus its layout file when that changed in the same
  // group: the pair rule a single-file graph has always followed.
  const paths = useMemo(() => {
    const layout = `layout/${name}.layout.json`;
    const pool = status === null
      ? []
      : group === 'staged'
        ? status.staged
        : [...status.unstaged, ...status.untracked];
    const own = files.map((f) => f.path);
    return pool.some((f) => f.path === layout) ? [...own, layout] : own;
  }, [files, group, name, status]);

  // Refetched when the set of changes under the folder does.
  const signature = files.map((f) => `${f.path}:${f.xy}`).join('|');
  useEffect(() => {
    let cancelled = false;
    const [before, after] = group === 'staged' ? ['HEAD', 'index'] : ['index', 'worktree'];
    void (async () => {
      try {
        const [oldSide, newSide] = await Promise.all([
          getGitGraph(name, before),
          getGitGraph(name, after),
        ]);
        if (cancelled) return;
        setSummary(summarizeGraphDiff(
          oldSide === null ? null : JSON.stringify(oldSide.logic),
          newSide === null ? null : JSON.stringify(newSide.logic),
          'graph',
        ));
      } catch {
        // A summary nobody can build is a summary not drawn; the files are
        // still listed and can still be opened one by one.
        if (!cancelled) setSummary(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [group, name, signature]);

  const run = useCallback(
    async (action: () => Promise<boolean>) => {
      if (await action()) onActed?.();
    },
    [onActed],
  );

  const askThenDiscard = useCallback(async () => {
    const ok = await confirm({
      title: t('git.discard.confirmGraph', { name, count: paths.length }),
      confirmText: t('git.discard.action'),
      variant: 'danger',
    });
    if (ok) await run(() => discard(paths));
  }, [discard, name, paths, run, t]);

  const label = `${name}/`;
  const conflicted = files.some((f) => f.kind === 'conflict');

  return (
    <li className={styles.graphFolder}>
      <div className={styles.row} title={`graphs/${name}/`}>
        <FileKindChip kind={folderKind(files)} />
        <button
          type="button"
          className={styles.openButton}
          aria-expanded={open}
          aria-controls={listId}
          aria-label={`${t('git.graphFolder.toggle')} ${name}`}
          onClick={() => setOpen((was) => !was)}
        >
          <span className={`${styles.chevron} ${open ? '' : styles.chevronCollapsed}`}>
            <ChevronDownIcon size={12} />
          </span>
          <span className={styles.rowName}>{label}</span>
          <span className={styles.rowDir}>
            {t('git.graphFolder.files', { count: files.length })}
          </span>
        </button>
        <div className={styles.rowActions}>
          {group === 'changes' && (
            <button
              type="button"
              className={styles.iconButton}
              aria-label={`${t('git.file.stage')} ${label}`}
              title={t('git.file.stage')}
              onClick={() => void run(() => stage(paths))}
            >
              <PlusIcon size={13} />
            </button>
          )}
          {group === 'staged' && (
            <button
              type="button"
              className={styles.iconButton}
              aria-label={`${t('git.file.unstage')} ${label}`}
              title={t('git.file.unstage')}
              onClick={() => void run(() => unstage(paths))}
            >
              <MinusIcon size={13} />
            </button>
          )}
          {group === 'changes' && !conflicted && (
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
      </div>
      {summary !== null && (
        <div className={styles.graphFolderSummary}>
          <GraphDiffSummary summary={summary} />
        </div>
      )}
      <ul id={listId} className={styles.list} role="list" hidden={!open}>
        {files.map((file) => (
          <FileRow
            key={`${file.path}:${file.xy}:${file.kind}`}
            file={file}
            group={group}
            onActed={onActed}
            onOpen={onOpen}
          />
        ))}
      </ul>
    </li>
  );
}
