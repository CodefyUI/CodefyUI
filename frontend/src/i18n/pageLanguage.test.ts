import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// The page's `<html lang>` follows the UI language (#504). A screen reader
// picks its voice from it, so a page that always said "en" had the Chinese UI
// read out with English pronunciation (WCAG 3.1.1, Language of Page).
//
// The store picks its first locale at module load, so every case imports a
// fresh copy of the module graph after `vi.resetModules()`.

/** A serialized one-node graph, in the shape a workspace file's tab carries. */
function graph(): Record<string, unknown> {
  return {
    name: 'g',
    description: '',
    nodes: [{ id: 'n1', type: 'Add', position: { x: 0, y: 0 }, data: { params: {} } }],
    edges: [],
    presets: [],
    segmentGroups: [],
    subgraphs: [],
    format_version: 1,
  };
}

/** The stores `importWorkspaceFile` touches, from the fresh module graph. */
async function freshImport() {
  const { importWorkspaceFile } = await import('../utils/importWorkspaceFile');
  const { useI18n } = await import('./index');
  const { useTabStore } = await import('../store/tabStore');
  const { useNodeDefStore } = await import('../store/nodeDefStore');
  const { useProjectStore } = await import('../store/projectStore');
  useNodeDefStore.setState({ definitions: [], presets: [] });
  useProjectStore.setState({ projectDir: null, projectName: null, loaded: true });
  // One EMPTY tab, which is what a fresh browser holds.
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('Tab 1');
  return { importWorkspaceFile, useI18n };
}

describe('the page language follows the UI language (#504)', () => {
  beforeEach(() => {
    vi.resetModules();
    localStorage.clear();
    // What `index.html` ships.
    document.documentElement.lang = 'en';
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.resetModules();
    localStorage.clear();
  });

  it('starts as the language the store starts with', async () => {
    localStorage.setItem('codefyui-locale', 'zh-TW');
    const { useI18n } = await import('./index');
    expect(useI18n.getState().locale).toBe('zh-TW');
    expect(document.documentElement.lang).toBe('zh-TW');
  });

  it('follows the language menu, both ways', async () => {
    const { useI18n } = await import('./index');
    useI18n.getState().setLocale('zh-TW');
    expect(document.documentElement.lang).toBe('zh-TW');
    useI18n.getState().setLocale('en');
    expect(document.documentElement.lang).toBe('en');
  });

  it('follows a locale written to the store directly', async () => {
    const { useI18n } = await import('./index');
    useI18n.setState({ locale: 'zh-TW' });
    expect(document.documentElement.lang).toBe('zh-TW');
  });

  it('follows a workspace import that carries a language', async () => {
    const { importWorkspaceFile, useI18n } = await freshImport();
    await importWorkspaceFile({
      version: 1,
      appVersion: null,
      exportedAt: null,
      active: null,
      preferences: { locale: 'zh-TW' },
      tabs: [{ title: 'A', graph: graph(), run: {} }],
    });
    expect(useI18n.getState().locale).toBe('zh-TW');
    expect(document.documentElement.lang).toBe('zh-TW');
  });

  it('stays with the store when the browser refuses to store the choice', async () => {
    // An import whose language could not be saved does not apply it (the
    // import's own rule), and the page must not claim a language the UI is
    // not in.
    const { importWorkspaceFile, useI18n } = await freshImport();
    const setItem = Storage.prototype.setItem;
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(function (
      this: Storage,
      key: string,
      value: string,
    ) {
      if (key === 'codefyui-locale') throw new DOMException('storage is full', 'QuotaExceededError');
      setItem.call(this, key, value);
    });
    await importWorkspaceFile({
      version: 1,
      appVersion: null,
      exportedAt: null,
      active: null,
      preferences: { locale: 'zh-TW' },
      tabs: [{ title: 'A', graph: graph(), run: {} }],
    });
    expect(useI18n.getState().locale).toBe('en');
    expect(document.documentElement.lang).toBe('en');
  });
});
