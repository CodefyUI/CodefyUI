/**
 * What an export is called, on the disk and inside the file.
 *
 * A saved graph exports under its own file's name, so `CF2A01.json` in the
 * Graphs panel downloads as `CF2A01.py` with no question, and the graph in it
 * carries the title it was saved under. A tab bound to no file -- a new
 * canvas, an imported starter, an example -- is asked first, with the tab's
 * name filled in: that name is exactly what used to go out silently as
 * "Tab_1.py", or as the starter's own `CF2D01.py`. The answer names the file,
 * the graph inside it, and, once the export has gone out, the tab.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { exportTarget, exportFileStemNow, nameTabAfterExport } from './exportFileName';
import { useDialogStore, type PromptRequest } from '../store/dialogStore';
import { useTabStore, type TabState } from '../store/tabStore';
import { useI18n } from '../i18n';

/** The store's own first tab, so a tab put in the store has every field. */
const REAL = useTabStore.getState().tabs[0];

/** A tab bound to the saved graph whose file stem is `file`, titled `title`. */
const saved = (file: string, name = 'Tab 1', title: string | null = file) => ({
  id: 'tab-1', name, currentGraphFile: file, currentGraphName: title,
});
/** A tab bound to no saved graph. */
const unsaved = (name: string) => ({
  id: 'tab-1', name, currentGraphFile: null, currentGraphName: null,
});

/** Make `tab` the store's only tab, in front, and hand back the store's copy. */
function open(tab: ReturnType<typeof saved> | ReturnType<typeof unsaved>): TabState {
  useTabStore.setState({ tabs: [{ ...REAL, ...tab }], activeTabId: tab.id });
  return useTabStore.getState().getActiveTab();
}

const labelOf = (id: string) => useTabStore.getState().tabs.find((tab) => tab.id === id)?.name;

/** The open dialog, which must be the export's file-name question. */
function question() {
  const active = useDialogStore.getState().active;
  expect(active?.kind).toBe('prompt');
  return active!;
}

/** Answer the open question (null is Cancel) and hand back what the export settled on. */
async function answer<T>(pending: Promise<T>, value: string | null): Promise<T> {
  question();
  useDialogStore.getState().close(value);
  return pending;
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useDialogStore.setState({ active: null, resolve: null });
  useTabStore.setState({ tabs: [REAL], activeTabId: REAL.id });
});

describe('exportTarget: a saved graph', () => {
  it("exports under its file's name and its title, without asking", async () => {
    await expect(exportTarget(saved('CF2A01'), 'py')).resolves.toEqual({
      stem: 'CF2A01',
      name: 'CF2A01',
      asked: false,
    });
    expect(useDialogStore.getState().active).toBeNull();
  });

  // The stem is the address, sanitized by the server; the title is what the
  // user typed when the graph was saved, spaces and all.
  it('keeps the letters of a CJK name, and the title its spaces', async () => {
    await expect(exportTarget(saved('期中考_第一題', 'Tab 1', '期中考 第一題'), 'json')).resolves.toEqual({
      stem: '期中考_第一題',
      name: '期中考 第一題',
      asked: false,
    });
    expect(useDialogStore.getState().active).toBeNull();
  });

  // The server writes stems with nothing but letters, digits, - and _, so this
  // is a no-op for every binding it made; it keeps a separator from an older
  // record out of the download name all the same.
  it('never lets a separator from a saved stem into the download name', async () => {
    await expect(exportTarget(saved('old/stem'), 'py')).resolves.toMatchObject({ stem: 'old_stem' });
  });

  // The label is the user's to type (double-click, F2) and an in-place save
  // never changes it, so it can say anything; the file is the saved graph.
  it("names the graph after the title it was saved under, whatever the tab's label says", async () => {
    await expect(exportTarget(saved('CF2A01', 'Tab 1', 'CF2A01 title'), 'py')).resolves.toEqual({
      stem: 'CF2A01',
      name: 'CF2A01 title',
      asked: false,
    });
  });

  // A tab restored from a 2.8.0 record knows its file but not its title.
  it.each([
    ['no title', 'Exam', null, 'Exam'],
    ['an empty title', 'Exam', '', 'Exam'],
    ['no title and an empty label', '', null, 'graph'],
  ])('falls back to the label, then "graph", for a graph with %s', async (_case, label, title, name) => {
    await expect(exportTarget(saved('CF2A01', label, title), 'py')).resolves.toEqual({
      stem: 'CF2A01',
      name,
      asked: false,
    });
  });
});

describe('exportTarget: a tab not saved yet', () => {
  it("asks for a name, offering the tab's name", async () => {
    const pending = exportTarget(unsaved('CF2D01'), 'py');
    expect(question()).toMatchObject({
      title: 'File name for the .py file',
      defaultValue: 'CF2D01',
    });
    await expect(answer(pending, 'CF2A01')).resolves.toEqual({
      stem: 'CF2A01',
      name: 'CF2A01',
      asked: true,
    });
  });

  it("asks in the user's language", async () => {
    useI18n.setState({ locale: 'zh-TW' });
    const pending = exportTarget(unsaved('Tab 1'), 'json');
    expect(question().title).toBe('匯出的 .json 檔名');
    await answer(pending, null);
  });

  // `name` is the answer as typed, which is what the graph is called; only
  // the file stem is sanitized.
  it.each([
    ['letters of any script', '期中考 第一題', '期中考_第一題', '期中考 第一題'],
    ['path and drive separators', 'a/b:c', 'a_b_c', 'a/b:c'],
    ['spaces around the name', '  CF2A01  ', 'CF2A01', 'CF2A01'],
    // The question names the extension, so typing it out is natural: it must
    // not come back as `CF2A01_py.py`, nor name the graph "CF2A01.py".
    ['the extension typed out', 'CF2A01.py', 'CF2A01', 'CF2A01'],
    ['the extension in capitals', 'CF2A01.PY', 'CF2A01', 'CF2A01'],
    ['a space before the extension', 'CF2A01 .py', 'CF2A01', 'CF2A01'],
    ['a different extension, which stays', 'CF2A01.json', 'CF2A01_json', 'CF2A01.json'],
    ['only the extension', '.py', 'graph', 'graph'],
  ])('makes a file name and a graph name of an answer with %s', async (_case, typed, stem, name) => {
    await expect(answer(exportTarget(unsaved('Tab 1'), 'py'), typed)).resolves.toEqual({
      stem,
      name,
      asked: true,
    });
  });

  it.each([
    ['Cancel', null],
    ['an empty answer', ''],
    ['a blank answer', '   '],
  ])('exports nothing after %s', async (_case, typed) => {
    await expect(answer(exportTarget(unsaved('CF2D01'), 'py'), typed)).resolves.toBeNull();
  });

  // Asking renames nothing: the tab takes the name once the export has gone
  // out (`nameTabAfterExport`), so a refused export leaves it as it was.
  it("leaves the tab's name alone, answered or cancelled", async () => {
    const tab = open(unsaved('CF2D01'));

    await answer(exportTarget(tab, 'py'), 'CF2A01');
    expect(labelOf(tab.id)).toBe('CF2D01');

    await answer(exportTarget(tab, 'py'), null);
    expect(labelOf(tab.id)).toBe('CF2D01');
  });

  // The question waits on the user, who can close the tab meanwhile. The
  // export still goes ahead, as it always did.
  it('still answers when the tab was closed while the question was open', async () => {
    const tab = open(unsaved('CF2D01'));
    const pending = exportTarget(tab, 'py');
    useTabStore.setState({ tabs: [REAL], activeTabId: REAL.id });

    await expect(answer(pending, 'CF2A01')).resolves.toEqual({
      stem: 'CF2A01',
      name: 'CF2A01',
      asked: true,
    });
  });

  // `con.py` on Windows is the console, not a file: the question refuses a
  // device name, with or without an extension, while the box is still open.
  it('refuses a name Windows keeps for a device', async () => {
    const pending = exportTarget(unsaved('Tab 1'), 'py');
    const { validate } = question() as PromptRequest;
    const refusal = 'Windows reserves this name; pick another';
    for (const name of ['con', 'CON', 'Nul.py', 'aux.tar.gz', ' prn ', 'com1', 'LPT9.json']) {
      expect(validate?.(name), name).toBe(refusal);
    }
    for (const name of ['CF2A01', 'console', 'con_1', 'com10', 'lpt', 'my.con', '']) {
      expect(validate?.(name), name).toBeNull();
    }
    useI18n.setState({ locale: 'zh-TW' });
    expect(validate?.('con')).toBe('這是 Windows 保留的名稱，請換一個');
    await answer(pending, null);
  });
});

/**
 * Once the file is downloading, a tab not saved yet takes the name its export
 * was asked for, so the next export or Save offers it rather than "Tab 1".
 */
describe('nameTabAfterExport', () => {
  it('gives a tab not saved yet the name it was asked for, and binds it to nothing', async () => {
    const tab = open(unsaved('CF2D01'));
    const target = await answer(exportTarget(tab, 'py'), 'CF2A01.py');

    nameTabAfterExport(tab.id, target!);

    expect(labelOf(tab.id)).toBe('CF2A01');
    expect(useTabStore.getState().getActiveTab().currentGraphFile).toBeNull();
  });

  it("keeps a saved graph's tab label", async () => {
    const tab = open(saved('CF2A01', 'my label', 'CF2A01 title'));
    const target = await exportTarget(tab, 'py');

    nameTabAfterExport(tab.id, target!);

    expect(labelOf(tab.id)).toBe('my label');
  });

  it('renames the tab it is given, not the one in front', async () => {
    const tab = open(unsaved('CF2D01'));
    const target = await answer(exportTarget(tab, 'py'), 'CF2A01');
    useTabStore.setState((s) => ({ tabs: [...s.tabs, REAL], activeTabId: REAL.id }));

    nameTabAfterExport(tab.id, target!);

    expect(labelOf(tab.id)).toBe('CF2A01');
    expect(labelOf(REAL.id)).toBe(REAL.name);
  });

  // The question and the request both wait, and the user can close the tab
  // meanwhile; no other tab is renamed, and nothing is written.
  it('renames nothing for a tab closed meanwhile', async () => {
    const tab = open(unsaved('CF2D01'));
    const target = await answer(exportTarget(tab, 'py'), 'CF2A01');
    useTabStore.setState({ tabs: [REAL], activeTabId: REAL.id });
    const before = useTabStore.getState().tabs;

    nameTabAfterExport(tab.id, target!);

    expect(useTabStore.getState().tabs).toBe(before);
  });

  // A Save while the request was out binds the tab and names it after the
  // graph it was saved as, which is the name to keep.
  it('keeps the name of a tab saved while the export was out', async () => {
    const tab = open(unsaved('CF2D01'));
    const target = await answer(exportTarget(tab, 'py'), 'CF2A01');
    useTabStore.getState().setTabGraphFile(tab.id, 'Saved_As', 'Saved As');
    useTabStore.getState().renameTab(tab.id, 'Saved As');

    nameTabAfterExport(tab.id, target!);

    expect(labelOf(tab.id)).toBe('Saved As');
  });

  it('writes nothing when the tab already has the name', async () => {
    const tab = open(unsaved('CF2A01'));
    const target = await answer(exportTarget(tab, 'py'), 'CF2A01');
    const before = useTabStore.getState().tabs;

    nameTabAfterExport(tab.id, target!);

    expect(useTabStore.getState().tabs).toBe(before);
  });
});

describe('exportFileStemNow', () => {
  it("names a saved graph after its file, whatever the tab's label says", () => {
    expect(exportFileStemNow(saved('CF2A01', 'Tab 1'))).toBe('CF2A01');
    expect(useDialogStore.getState().active).toBeNull();
  });

  it('names a tab not saved yet after the tab, without asking', () => {
    expect(exportFileStemNow(unsaved('期中考 第一題'))).toBe('期中考_第一題');
    expect(useDialogStore.getState().active).toBeNull();
  });

  it.each([
    ['an empty', ''],
    ['a blank', '   '],
  ])('falls back to "graph" for %s tab name', (_case, name) => {
    expect(exportFileStemNow(unsaved(name))).toBe('graph');
  });
});
