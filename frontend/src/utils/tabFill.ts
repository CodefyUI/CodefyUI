import { tabHasContent, type TabState } from '../store/tabStore';

/**
 * May a graph being opened fill this tab instead of opening one of its own?
 *
 * One rule for every door a graph comes in by: Import (#550), a row of the
 * Graphs panel, and a starter picked on the empty-canvas gallery (#595). None
 * of them replaces work: their install pushes no undo step and autosave writes
 * it within 250 ms, so a graph replaced was a graph lost.
 *
 * Only an empty tab, by the rule the tab's close button applies before it lets
 * a tab go without asking (`tabHasContent`): any node counts, a note included,
 * and so does a graph waiting outside an open block -- which the empty-canvas
 * overlay, looking only at the level on screen, calls empty. And only one with
 * nothing to undo or redo: a tab emptied by Clear Canvas, or by deleting its
 * nodes, is one undo away from the graph it held. Undoing a delete does not
 * restore the file binding -- filled with a saved graph, that undo brought the
 * old graph back bound to the new file, for the next Save to write over it --
 * and undoing a Clear Canvas puts the cleared graph, and its binding, back
 * over the one just opened. And only one the user opened. A plugin's tab
 * goes on saying "Opened by <plugin>", on hover and in its accessible name,
 * whatever graph is put in it, and one it opened for this session only is
 * gone after a reload, the graph with it. Not one that is running, either.
 * The `.cduiworkspace` importer asks this too, before it closes the lone
 * empty tab (#625).
 */
export function canFillTab(tab: TabState | undefined): tab is TabState {
  return (
    tab !== undefined &&
    !tabHasContent(tab) &&
    !tab.undoStack?.length &&
    !tab.redoStack?.length &&
    !tab.source &&
    !tab.transient &&
    tab.status !== 'running'
  );
}
