/**
 * What an export is called.
 *
 * A saved graph exports under its own file's name, so `CF2A01.json` in the
 * Graphs panel downloads as `CF2A01.py` with no question. A tab bound to no
 * file -- a new canvas, an imported starter, an example -- is asked first, with
 * the tab's name filled in: that name is exactly what used to go out silently
 * as "Tab_1.py", or as the starter's own `CF2D01.py`.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { exportFileStem, exportFileStemNow } from './exportFileName';
import { useDialogStore, type PromptRequest } from '../store/dialogStore';
import { useI18n } from '../i18n';

/** A tab bound to the saved graph whose file stem is `file`. */
const saved = (file: string, name = 'Tab 1') => ({ name, currentGraphFile: file });
/** A tab bound to no saved graph. */
const unsaved = (name: string) => ({ name, currentGraphFile: null });

/** The open dialog, which must be the export's file-name question. */
function question() {
  const active = useDialogStore.getState().active;
  expect(active?.kind).toBe('prompt');
  return active!;
}

/** Answer the open question (null is Cancel) and hand back the export's stem. */
async function answer(pending: Promise<string | null>, value: string | null) {
  question();
  useDialogStore.getState().close(value);
  return pending;
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useDialogStore.setState({ active: null, resolve: null });
});

describe('exportFileStem', () => {
  it("exports a saved graph under its file's name, without asking", async () => {
    await expect(exportFileStem(saved('CF2A01'), 'py')).resolves.toBe('CF2A01');
    expect(useDialogStore.getState().active).toBeNull();
  });

  it('keeps the letters of a saved CJK name', async () => {
    await expect(exportFileStem(saved('期中考_第一題'), 'json')).resolves.toBe('期中考_第一題');
    expect(useDialogStore.getState().active).toBeNull();
  });

  // The server writes stems with nothing but letters, digits, - and _, so this
  // is a no-op for every binding it made; it keeps a separator from an older
  // record out of the download name all the same.
  it('never lets a separator from a saved stem into the download name', async () => {
    await expect(exportFileStem(saved('old/stem'), 'py')).resolves.toBe('old_stem');
  });

  it("asks for a name for a tab not saved yet, offering the tab's name", async () => {
    const pending = exportFileStem(unsaved('CF2D01'), 'py');
    expect(question()).toMatchObject({
      title: 'File name for the .py file',
      defaultValue: 'CF2D01',
    });
    await expect(answer(pending, 'CF2A01')).resolves.toBe('CF2A01');
  });

  it("asks in the user's language", async () => {
    useI18n.setState({ locale: 'zh-TW' });
    const pending = exportFileStem(unsaved('Tab 1'), 'json');
    expect(question().title).toBe('匯出的 .json 檔名');
    await answer(pending, null);
  });

  it.each([
    ['letters of any script', '中文 圖', '中文_圖'],
    ['path and drive separators', 'a/b:c', 'a_b_c'],
    ['spaces around the name', '  CF2A01  ', 'CF2A01'],
    // The question names the extension, so typing it out is natural: it must
    // not come back as `CF2A01_py.py`.
    ['the extension typed out', 'CF2A01.py', 'CF2A01'],
    ['the extension in capitals', 'CF2A01.PY', 'CF2A01'],
    ['a different extension, which stays', 'CF2A01.json', 'CF2A01_json'],
    ['only the extension', '.py', 'graph'],
  ])('makes a file name of an answer with %s', async (_case, typed, stem) => {
    await expect(answer(exportFileStem(unsaved('Tab 1'), 'py'), typed)).resolves.toBe(stem);
  });

  it.each([
    ['Cancel', null],
    ['an empty answer', ''],
    ['a blank answer', '   '],
  ])('exports nothing after %s', async (_case, typed) => {
    await expect(answer(exportFileStem(unsaved('CF2D01'), 'py'), typed)).resolves.toBeNull();
  });

  // `con.py` on Windows is the console, not a file: the question refuses a
  // device name, with or without an extension, while the box is still open.
  it('refuses a name Windows keeps for a device', async () => {
    const pending = exportFileStem(unsaved('Tab 1'), 'py');
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
