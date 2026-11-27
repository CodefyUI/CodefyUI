import type { GitFile } from '../api/git';

/**
 * The graph a repository path belongs to when that graph is stored as a
 * folder (storage v2, `graphs/<name>/...`), or null.
 */
export function graphFolderOf(path: string): string | null {
  const match = /^graphs\/([^/]+)\/.+$/.exec(path);
  return match === null ? null : match[1];
}

/** One row of a change list: a file, or every file of one folder graph. */
export type ChangeItem =
  | { type: 'file'; file: GitFile }
  | { type: 'graph'; name: string; files: GitFile[] };

/**
 * Fold the files of each folder graph into one row, placed where the first
 * of its files was. Every other file stays a row of its own, in order.
 */
export function groupChanges(files: GitFile[]): ChangeItem[] {
  const items: ChangeItem[] = [];
  const graphs = new Map<string, { type: 'graph'; name: string; files: GitFile[] }>();
  for (const file of files) {
    const name = graphFolderOf(file.path);
    if (name === null) {
      items.push({ type: 'file', file });
      continue;
    }
    let group = graphs.get(name);
    if (group === undefined) {
      group = { type: 'graph', name, files: [] };
      graphs.set(name, group);
      items.push(group);
    }
    group.files.push(file);
  }
  return items;
}

/**
 * The one letter a folder row shows: added when every file is new, deleted
 * when every file is gone, a conflict when any file is in one, otherwise
 * modified.
 */
export function folderKind(files: GitFile[]): GitFile['kind'] {
  if (files.some((f) => f.kind === 'conflict')) return 'conflict';
  if (files.every((f) => f.kind === 'untracked')) return 'untracked';
  if (files.every((f) => f.kind === 'added' || f.kind === 'untracked')) return 'added';
  if (files.every((f) => f.kind === 'deleted')) return 'deleted';
  return 'modified';
}
