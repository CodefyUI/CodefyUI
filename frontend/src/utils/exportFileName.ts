import { useTabStore, type TabState } from '../store/tabStore';
import { useI18n } from '../i18n';
import { prompt } from './dialog';
import { sanitizeGraphName } from './index';

/**
 * What an exported file is called, and what the graph inside it is called.
 * Export as JSON, Export as Python and Export Diagram all take their file
 * name from here; the first two also take the graph's name, which goes into
 * the JSON's `name` and the script's `GRAPH_NAME`.
 *
 * A tab bound to a saved graph exports under that graph's file stem, as the
 * server spelled it when it wrote the file, so `CF2A01.json` in the Graphs
 * panel downloads as `CF2A01.py`, with no question, and the graph in it keeps
 * the title it was saved under. A tab bound to nothing -- a new canvas, an
 * imported starter, an opened example, a plugin's tab -- is asked first, with
 * its own name filled in. That name is exactly what used to decide the file
 * silently: "Tab 1" went out as `Tab_1.py`, and a starter imported as
 * `CF2D01` exported as `CF2D01.py`, the starter's own file name, which in a
 * student's folder replaces the starter or lands beside it as
 * `CF2D01 (1).py`. Asked, the name is on screen before the download instead
 * of after it.
 *
 * The answer names the graph as well as the file, and, once the export has
 * gone out, the tab (`nameTabAfterExport`). The graph used to keep the tab's
 * name, so `CF2A01.py` said `GRAPH_NAME = 'Tab 1'` and `CF2A01.json`,
 * imported again, opened as a tab named "Tab 1".
 *
 * Every file name goes through `sanitizeGraphName`: letters and digits of any
 * script stay, and so do `-` and `_`; anything else -- a path separator, a
 * colon, a space -- becomes `_`. The rule this replaced kept ASCII only, so a
 * CJK title downloaded as a row of underscores. The download is a blob named
 * by `a.download`, and browsers accept Unicode there. The graph's name is
 * kept as typed.
 */

/** The fields of a tab an export's names are read from. */
export type ExportNameSource = Pick<TabState, 'id' | 'name' | 'currentGraphFile' | 'currentGraphName'>;

/** Where an export downloads, and what the graph inside it is called. */
export interface ExportTarget {
  /** The file name without its extension. */
  stem: string;
  /** The graph's name inside the file: the JSON's `name`, the script's `GRAPH_NAME`. */
  name: string;
  /** Whether the name was asked for, which is how a tab not saved yet is named. */
  asked: boolean;
}

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
 * Where to download an export of `tab` and what to call the graph in it, or
 * null for "export nothing": the user cancelled the question, or answered it
 * with nothing. `ext` is the file's extension without the dot (`py`, `json`);
 * the question names it, and an answer that repeats it is not given it twice.
 */
export async function exportTarget(
  tab: ExportNameSource,
  ext: string,
): Promise<ExportTarget | null> {
  const saved = savedStem(tab);
  if (saved !== null) {
    // The title the saved file carries. The label is only the fallback, for
    // a tab restored from 2.8.0 that knows its file but not its title: the
    // user can retype a label at any time, and an in-place save keeps it.
    return { stem: saved, name: tab.currentGraphName || tab.name || 'graph', asked: false };
  }
  const answer = await prompt({
    title: useI18n.getState().t('toolbar.exportName.prompt', { ext }),
    defaultValue: tab.name,
    placeholder: 'graph-name',
    validate: deviceNameRefusal,
  });
  const typed = answer?.trim();
  if (!typed) return null;
  // The question says ".py file", so "CF2A01.py" is a natural answer, and it
  // must not download as `CF2A01_py.py`, nor name the graph "CF2A01.py".
  const suffix = `.${ext}`;
  const bare = typed.toLowerCase().endsWith(suffix.toLowerCase())
    ? typed.slice(0, -suffix.length)
    : typed;
  const name = bare.trim() || 'graph';
  return { stem: toStem(name), name, asked: true };
}

/**
 * Give the tab an export started from the name the export was asked for,
 * once the file is downloading -- as a Save As renames the tab once the file
 * is written, and not before: a refused export renames nothing. The tab then
 * shows the name its export carries, and the next export or Save offers it
 * rather than "Tab 1" again. A saved graph's tab keeps its label.
 *
 * By id, because the export can outlive the tab in front: the user may have
 * moved to another tab, or closed this one, while the question or the request
 * was open. A closed tab is not renamed, and nor is one saved meanwhile: it
 * is bound now, and named after the graph it was saved as. Nothing is
 * written for a tab that already has the name.
 */
export function nameTabAfterExport(tabId: string, target: ExportTarget): void {
  if (!target.asked) return;
  const store = useTabStore.getState();
  const tab = store.getTab(tabId);
  if (tab === undefined || tab.currentGraphFile !== null || tab.name === target.name) return;
  store.renameTab(tabId, target.name);
}

/**
 * The stem without asking, for Export Diagram: a picture of the graph is not
 * worth a question. The saved graph's file stem, else the tab's own name.
 */
export function exportFileStemNow(tab: ExportNameSource): string {
  return savedStem(tab) ?? toStem(tab.name);
}
