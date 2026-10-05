import type { TabState } from '../store/tabStore';
import { useI18n } from '../i18n';
import { prompt } from './dialog';
import { sanitizeGraphName } from './index';

/**
 * What an exported file is called. Export as JSON, Export as Python and
 * Export Diagram all take their file name from here.
 *
 * A tab bound to a saved graph exports under that graph's file stem, as the
 * server spelled it when it wrote the file, so `CF2A01.json` in the Graphs
 * panel downloads as `CF2A01.py`, with no question. A tab bound to nothing --
 * a new canvas, an imported starter, an opened example, a plugin's tab -- is
 * asked first, with its own name filled in. That name is exactly what used to
 * decide the file silently: "Tab 1" went out as `Tab_1.py`, and a starter
 * imported as `CF2D01` exported as `CF2D01.py`, the starter's own file name,
 * which in a student's folder replaces the starter or lands beside it as
 * `CF2D01 (1).py`. Asked, the name is on screen before the download instead
 * of after it.
 *
 * Every name goes through `sanitizeGraphName`: letters and digits of any
 * script stay, and so do `-` and `_`; anything else -- a path separator, a
 * colon, a space -- becomes `_`. The rule this replaced kept ASCII only, so a
 * CJK title downloaded as a row of underscores. The download is a blob named
 * by `a.download`, and browsers accept Unicode there.
 */

/** The two fields of a tab an export's name is read from. */
export type ExportNameSource = Pick<TabState, 'name' | 'currentGraphFile'>;

/** `raw` as a file stem, or "graph" when nothing is left of it. */
function toStem(raw: string): string {
  return sanitizeGraphName(raw.trim()) || 'graph';
}

/** The saved graph's file stem, or null for a tab bound to no saved graph. */
function savedStem(tab: ExportNameSource): string | null {
  return tab.currentGraphFile ? toStem(tab.currentGraphFile) : null;
}

/**
 * Names Windows keeps for devices, which no file can have, with an extension
 * or without: `con.py` names the console. Windows reads the part before the
 * first dot, and so does the check.
 */
const WINDOWS_DEVICE = /^(con|prn|aux|nul|com\d|lpt\d)$/i;

/** The question's refusal for a device name, or null. */
function deviceNameRefusal(value: string): string | null {
  const head = value.trim().split('.')[0].trim();
  return WINDOWS_DEVICE.test(head) ? useI18n.getState().t('toolbar.exportName.reserved') : null;
}

/**
 * The stem to download an export of `tab` under, or null for "export
 * nothing": the user cancelled the question, or answered it with nothing.
 * `ext` is the file's extension without the dot (`py`, `json`); the question
 * names it, and an answer that repeats it is not given it twice.
 */
export async function exportFileStem(
  tab: ExportNameSource,
  ext: string,
): Promise<string | null> {
  const saved = savedStem(tab);
  if (saved !== null) return saved;
  const answer = await prompt({
    title: useI18n.getState().t('toolbar.exportName.prompt', { ext }),
    defaultValue: tab.name,
    placeholder: 'graph-name',
    validate: deviceNameRefusal,
  });
  const typed = answer?.trim();
  if (!typed) return null;
  // The question says ".py file", so "CF2A01.py" is a natural answer, and it
  // must not download as `CF2A01_py.py`.
  const suffix = `.${ext}`;
  const bare = typed.toLowerCase().endsWith(suffix.toLowerCase())
    ? typed.slice(0, -suffix.length)
    : typed;
  return toStem(bare);
}

/**
 * The stem without asking, for Export Diagram: a picture of the graph is not
 * worth a question. The saved graph's file stem, else the tab's own name.
 */
export function exportFileStemNow(tab: ExportNameSource): string {
  return savedStem(tab) ?? toStem(tab.name);
}
