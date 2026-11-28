import { describe, it, expect } from 'vitest';
import type { FileKind, GitFile } from '../api/git';
import { folderKind, graphFolderOf, groupChanges } from './graphFolders';

function file(path: string, kind: FileKind = 'modified'): GitFile {
  return { path, orig_path: null, kind, xy: '.M', score: null };
}

describe('graphFolderOf', () => {
  it('names the graph for any file under graphs/<name>/', () => {
    expect(graphFolderOf('graphs/train/index.json')).toBe('train');
    expect(graphFolderOf('graphs/train/nodes/conv.code.py')).toBe('train');
    expect(graphFolderOf('graphs/train.graph.json')).toBeNull();
    expect(graphFolderOf('layout/train.layout.json')).toBeNull();
    expect(graphFolderOf('nested/graphs/train/index.json')).toBeNull();
  });
});

describe('groupChanges', () => {
  it('folds each folder graph into one row where its first file was', () => {
    const items = groupChanges([
      file('README.md'),
      file('graphs/a/nodes/x.json'),
      file('graphs/b.graph.json'),
      file('graphs/a/edges.jsonl'),
      file('graphs/c/index.json', 'untracked'),
    ]);
    expect(items.map((i) => (i.type === 'file' ? i.file.path : `${i.name}/${i.files.length}`)))
      .toEqual(['README.md', 'a/2', 'graphs/b.graph.json', 'c/1']);
  });
});

describe('folderKind', () => {
  it('reads the whole folder as added, deleted, conflicted or modified', () => {
    expect(folderKind([file('a', 'untracked'), file('b', 'untracked')])).toBe('untracked');
    expect(folderKind([file('a', 'added'), file('b', 'untracked')])).toBe('added');
    expect(folderKind([file('a', 'deleted')])).toBe('deleted');
    expect(folderKind([file('a', 'deleted'), file('b')])).toBe('modified');
    expect(folderKind([file('a', 'conflict'), file('b')])).toBe('conflict');
  });
});
