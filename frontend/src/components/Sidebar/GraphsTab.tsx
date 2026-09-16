import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import {
  deleteGraph,
  listGraphs,
  renameGraph,
  type SavedGraphSummary,
} from '../../api/rest';
import { useI18n } from '../../i18n';
import { useProjectStore } from '../../store/projectStore';
import { useTabStore } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import { sanitizeGraphName } from '../../utils';
import { confirm, prompt } from '../../utils/dialog';
import { getGraphsWriteListener, setGraphsWriteListener } from '../../utils/graphsWrite';
import { importGraphFile } from '../../utils/importGraphFile';
import { readSavedGraphDocument } from '../../utils/openSavedGraph';
import { saveActiveGraph } from '../../utils/saveActiveGraph';
import { announceWorktreeWrite } from '../../utils/worktreeWrite';
import { relativeTime } from '../SourceControl/scm';
import { ActionMenu, type ActionMenuItem } from '../shared/ActionMenu';
import { MoreHorizontalIcon, RefreshIcon, SaveAsIcon } from '../shared/Icons';
import styles from './NodePalette.module.css';
import tabStyles from './GraphsTab.module.css';

/**
 * Most recently modified first, by name when there is nothing to date them by.
 *
 * `modified` is optional over the wire (a frontend built from source can meet
 * a backend older than this panel), so a dated row always leads an undated
 * one rather than the two comparing as equal -- which would leave the order
 * depending on which pair `sort` happened to look at. In practice the field
 * is present on every row or on none, and the all-undated case is the one
 * this fallback is really for: an alphabetical list, not an arbitrary one.
 */
export function sortGraphs(graphs: SavedGraphSummary[]): SavedGraphSummary[] {
  return [...graphs].sort((a, b) => {
    if (a.modified !== undefined && b.modified !== undefined) {
      return b.modified - a.modified || a.name.localeCompare(b.name);
    }
    if (a.modified !== undefined) return -1;
    if (b.modified !== undefined) return 1;
    return a.name.localeCompare(b.name);
  });
}

/**
 * Does this row match what was typed? `query` arrives lowercased, as
 * `exampleMatches` takes it.
 *
 * Both halves, because they stop agreeing the moment a name needs sanitizing:
 * "My Graph" is saved as `My_Graph`, and someone who typed either spelling
 * meant the same file. The toolbar picker this panel replaces matched both
 * for exactly that reason.
 */
export function graphMatches(graph: SavedGraphSummary, query: string): boolean {
  return (
    graph.name.toLowerCase().includes(query)
    || graph.file.toLowerCase().includes(query)
  );
}

// ── Graph row ──

interface GraphRowProps {
  graph: SavedGraphSummary;
  /** True when the ACTIVE tab saves back to this file. */
  current: boolean;
  onOpen: (graph: SavedGraphSummary) => void;
  onRename: (graph: SavedGraphSummary) => void;
  onDelete: (graph: SavedGraphSummary) => void;
}

/**
 * One saved graph: click to open it in a tab of its own, or reach the other
 * two verbs through the menu at the end of the row.
 *
 * Opening is ONE action rather than a choice between several. A row in a
 * list the user scrolls past must not be able to take over the canvas that
 * is in front of them, and a tab of its own is the only answer that is never
 * destructive -- which is also why the row asks nothing before opening:
 * there is no longer anything to lose by saying yes.
 *
 * Shaped like the Source Control tab's `RefRow`, which is the panel next
 * door and solved the same three problems: a name that has to ellipsise at
 * 180px, a meta line that is a description rather than part of the row's
 * name, and a set of verbs that only fit as a menu.
 */
function GraphRow({
  graph,
  current,
  onOpen,
  onRename,
  onDelete,
}: GraphRowProps) {
  const { t, locale } = useI18n();
  const metaId = useId();
  const badgeId = useId();

  // Empty when the backend predates the field, or when `st_mtime` came back
  // as something no date can be made of -- a row with no date on it says more
  // than a row with the wrong one.
  const when = graph.modified === undefined ? '' : relativeTime(graph.modified, locale);
  const meta = when === '' ? '' : t('graphs.modified', { when });
  const open = t('graphs.open');

  // Two items, and neither of them is another way to open the graph: the row
  // itself is the only door in, so the menu holds what is left -- the two
  // things that change the FILE rather than what is on screen.
  const items: ActionMenuItem[] = [
    { id: 'rename', label: t('graphs.rename'), onSelect: () => onRename(graph) },
    {
      id: 'delete',
      label: t('graphs.delete'),
      danger: true,
      onSelect: () => onDelete(graph),
    },
  ];

  // What the row's button is DESCRIBED by, in the order a reader wants it:
  // which graph the tab saves to, then when the file was last written.
  const describedBy = [current ? badgeId : '', meta === '' ? '' : metaId]
    .filter((one) => one !== '')
    .join(' ');

  return (
    // The name lives on the ROW as well: the button's own `title` would not
    // open over the menu or the padding beside it. A newline rather than a
    // separator between the two clauses, which are punctuated in their own
    // languages already.
    <li className={tabStyles.row} title={meta === '' ? graph.name : `${graph.name}\n${meta}`}>
      <button
        type="button"
        className={tabStyles.rowOpen}
        // The verb plus the row it acts on, so a list of twenty rows is
        // twenty distinct buttons rather than twenty called "Open". The meta
        // is INSIDE the button, so without the description it would be read
        // as part of that name.
        aria-label={`${open} ${graph.name}`}
        aria-describedby={describedBy === '' ? undefined : describedBy}
        onClick={() => onOpen(graph)}
      >
        <span className={tabStyles.rowName}>{graph.name}</span>
        {meta !== '' && (
          <span className={tabStyles.rowMeta} id={metaId}>{meta}</span>
        )}
      </button>
      {/* Outside the button, where `RefRow` keeps its own badge: an
          `aria-label` REPLACES the name a button takes from its content, so
          a chip inside this one was drawn on screen and announced to nobody
          -- and "which graph does Save write to" is the one thing this list
          says that its names do not. Described by, not named by, so the
          button is still the sentence a reader acts on. */}
      {current && <span className={tabStyles.rowBadge} id={badgeId}>{t('graphs.current')}</span>}
      <ActionMenu
        label={`${t('graphs.rowMenu')} ${graph.name}`}
        items={items}
        align="end"
        className={tabStyles.rowMenuButton}
      >
        <MoreHorizontalIcon size={13} />
      </ActionMenu>
    </li>
  );
}

// ── Graphs tab ──

/**
 * The saved graphs on the server, and everything that can be done to one.
 *
 * The list is `GET /api/graph/list`, read per mount the way the Templates tab
 * reads its own: the sidebar only mounts the tab that is open, and this list
 * is the one that changes behind your back. A Save from inside the app says
 * so through `graphsWrite` and the list keeps up on its own; the refresh
 * control is for the writes this app never sees -- a file dropped into
 * `graphs/` by hand, or written by a second browser.
 *
 * Opening is ONE action: a row click reads the graph into a tab of its own,
 * bound to its file so Save writes straight back over it. Nothing on screen
 * is replaced and so nothing is asked first -- a row in a list the user
 * scrolls past should not be able to take over the canvas being worked in,
 * and a new tab is the only answer that is never destructive. The one graph
 * that does not get a new tab is one a tab already holds: that tab is raised
 * instead, so two tabs never end up bound to the same file -- and when it is
 * the tab already in front, where raising it would move nothing, the click is
 * answered in words instead of in silence.
 */
export function GraphsTab() {
  const { t } = useI18n();
  const [graphs, setGraphs] = useState<SavedGraphSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  // A scalar selector rather than the tab object: the marker has to follow a
  // Save or a rename, and subscribing to the tab itself would re-render this
  // panel on every node the canvas gains.
  const boundFile = useTabStore(
    (s) => s.tabs.find((tb) => tb.id === s.activeTabId)?.currentGraphFile ?? null,
  );

  // The files whose read is in flight: one entry per row that has been
  // clicked and has no tab yet. A ref rather than state, because nothing on
  // screen is drawn from it and a re-render per click would be noise.
  const opening = useRef(new Set<string>());

  // Which list read is the current one. Every rename, delete and Save As
  // refetches and the refresh button is one click away, so two reads in
  // flight together is ordinary -- and the one that answers LAST is not the
  // one that was asked last: a read issued before a delete still carries the
  // deleted row, and would put it back on screen.
  const loadSeq = useRef(0);

  /**
   * Re-read the list. `quiet` keeps what is on screen while the read runs.
   *
   * Only the reads the user is waiting on -- the first one, the refresh
   * button, the retry after a failure -- blank the list to the loading line.
   * A refetch after a rename, a delete or a Save As changes one row, and
   * emptying the panel for it threw away the reader's place in the list.
   */
  const load = useCallback((quiet = false) => {
    const seq = loadSeq.current + 1;
    loadSeq.current = seq;
    if (!quiet) setLoading(true);
    setError(null);
    listGraphs()
      .then((all) => {
        if (seq !== loadSeq.current) return;
        setGraphs(Array.isArray(all) ? all : []);
      })
      .catch((e: Error) => {
        if (seq !== loadSeq.current) return;
        setGraphs([]);
        setError(e.message);
      })
      .finally(() => {
        if (seq !== loadSeq.current) return;
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // The toolbar's Save and this panel are on screen together, so a graph
  // saved for the first time has to appear in the list without the user
  // going looking for the refresh button -- being told to press refresh to
  // see something you just did reads as the save having failed.
  useEffect(() => {
    const listener = () => load(true);
    setGraphsWriteListener(listener);
    return () => {
      // Only while the slot still holds OUR listener. StrictMode mounts,
      // unmounts and mounts again, so this cleanup runs after the remount
      // has already registered its own -- clearing unconditionally would
      // throw that one away and leave the panel deaf for the rest of its
      // life, with nothing on screen to show for it.
      if (getGraphsWriteListener() === listener) setGraphsWriteListener(null);
    };
  }, [load]);

  // Trimmed once: it decides both what is filtered and whether an empty body
  // reads as "nothing matched" or as "nothing saved", and a box holding only
  // spaces is not a search.
  const query = searchQuery.trim();
  const rows = useMemo(() => {
    const q = query.toLowerCase();
    return sortGraphs(q ? graphs.filter((g) => graphMatches(g, q)) : graphs);
  }, [graphs, query]);

  /**
   * Bring the tab that holds this graph to the front -- or, when it is
   * already the tab in front, say so instead.
   *
   * Switching to a BACKGROUND tab is deliberately silent: the whole canvas
   * changes under the click, which answers it better than any sentence
   * could. The active tab has no such answer. `setActiveTab` would write the
   * id that is already in `activeTabId`, every selector would compare equal
   * and NOTHING on screen would move -- no tab switch, no focus, no redraw
   * -- while the button still announces "Open <name>". A click that produces
   * no response reads as a click that missed, and gets made again. It is
   * also the commonest click of all, because the row marked Current is the
   * row the user just opened.
   *
   * `announceNoOp` says whether "it is already here" is an ANSWER to the
   * click or the resolution of a race, and only the first of those is worth a
   * sentence. The check before the read answers a click: the user pressed a
   * row whose graph is open in this very tab, and without the toast nothing
   * at all happens. The recheck after the read is a race being settled --
   * something bound this file while the read was in the air -- and raising
   * the tab silently is the whole of the right answer there. Said out loud,
   * it turns ONE user click into "already open in this tab": the panel
   * unmounts mid-read when the user switches sidebar tabs, its in-flight Set
   * dies with the instance, the remounted panel issues a second read, and
   * whichever continuation loses finds the tab the winner has just created.
   */
  const raiseTab = useCallback(
    (tabId: string, name: string, opts: { announceNoOp: boolean }) => {
      if (useTabStore.getState().activeTabId === tabId) {
        if (opts.announceNoOp) {
          useToastStore.getState().addToast(t('graphs.alreadyOpen', { name }), 'info');
        }
        return;
      }
      useTabStore.getState().setActiveTab(tabId);
    },
    [t],
  );

  const handleOpen = useCallback(
    async (graph: SavedGraphSummary) => {
      // Already open somewhere? Raise that tab instead of reading a second
      // copy of the file in. Two tabs bound to one graph is a data-loss
      // shape rather than a cosmetic one -- each one's next Save silently
      // overwrites whatever the other last wrote -- and a list of rows the
      // user clicks through is the easiest place in the app to build it.
      // Deliberately NOT re-read into that tab either: it may be holding
      // unsaved edits, and once a graph is open, bringing it to the front is
      // the whole of what "open this graph" can safely mean.
      const open = useTabStore.getState().tabs.find(
        (tb) => tb.currentGraphFile === graph.file,
      );
      if (open !== undefined) {
        // Announced: this is the click's answer, and when the tab is the one
        // already in front nothing else about it moves.
        raiseTab(open.id, graph.name, { announceNoOp: true });
        return;
      }
      // The check above can only see tabs that EXIST, and this row's tab is
      // not made until its read answers -- so a second click inside that
      // window looks at a store where nothing is bound to the file yet and
      // sails straight past it. Turned away here, before it costs a second
      // read of the same file and a second run at creating a tab for it:
      // two tabs bound to one graph is the data-loss shape the check above
      // is for, and it is worse now that each tab's Save is a promptless
      // overwrite of the bound file.
      if (opening.current.has(graph.file)) return;
      opening.current.add(graph.file);
      try {
        // Read BEFORE the tab exists, so a graph that cannot be read leaves
        // no empty tab behind to close.
        const doc = await readSavedGraphDocument(graph.file, graph.file);
        // Asked again now the read is back. The mark above stops a second
        // CLICK on this row; only this stops a second BINDING that arrived
        // from somewhere else while the read was in the air -- an import, a
        // Save As, a Source Control reload -- and a tab bound that way is
        // just as real a second writer of the file. Neither guard covers
        // the other's half of the window, so dropping either leaves it open.
        const raced = useTabStore.getState().tabs.find(
          (tb) => tb.currentGraphFile === graph.file,
        );
        if (raced !== undefined) {
          // Silent: somebody bound the file while this read was in flight,
          // which is not a fact about the click. See `raiseTab` for the
          // single click this used to answer twice.
          raiseTab(raced.id, graph.name, { announceNoOp: false });
          return;
        }
        const tabId = useTabStore.getState().createTab({ title: graph.name });
        const tooNew = useTabStore.getState().loadGraphDocumentInto(tabId, doc);
        if (tooNew) {
          useToastStore.getState().addToast(
            t('project.readOnly.loadNotice', {
              version: doc.formatVersion as string | number,
            }),
            'warning',
          );
        }
        // The new tab is the active one, which is what `stampActiveTabProject`
        // writes to. The stamp is what the Source Control tab's affected-tab
        // filter reads, so a graph opened from here without one would sit
        // outside every reload offer the project ever makes.
        const projectDir = useProjectStore.getState().projectDir;
        if (projectDir !== null) useTabStore.getState().stampActiveTabProject(projectDir);
      } catch (e) {
        useToastStore.getState().addToast(
          t('toolbar.load.fail', { error: (e as Error).message }),
          'error',
        );
      } finally {
        // In a `finally`, so a read that threw does not wedge the row shut.
        // The toast above tells the user the open failed, which is an
        // invitation to try again, and a mark left behind would make every
        // later click on that row do nothing at all for the rest of the
        // session -- with nothing on screen to explain why.
        opening.current.delete(graph.file);
      }
    },
    [raiseTab, t],
  );

  const handleRename = useCallback(
    async (graph: SavedGraphSummary) => {
      const entered = await prompt({
        title: t('graphs.rename.prompt'),
        defaultValue: graph.name,
      });
      const next = entered?.trim();
      if (!next || next === graph.name) return;
      try {
        // The server answers with the sanitized stem it wrote, which is the
        // only way to learn what "My Graph" became on disk without
        // re-deriving it. `sanitizeGraphName` is the same rule, kept as the
        // fallback for a backend that answers without the field.
        const result = (await renameGraph(graph.file, next)) as { file?: unknown };
        const file = typeof result?.file === 'string' ? result.file : sanitizeGraphName(next);
        // The name a rename moves INTO can already have an owner, so give it
        // up before moving anything onto it. A tab stays bound to a file that
        // is gone on purpose: when a Source Control discard, checkout or
        // stash pop reloads the affected tabs and the file has vanished, the
        // reload keeps the binding so the tab goes on showing what it holds
        // -- and declining the reload offer keeps it too. The server has no
        // objection either, because its only guard on a rename is that the
        // destination does not exist on disk, which that file does not. Left
        // alone, the rebind below then puts TWO tabs on one physical file,
        // which is the shape the open-dedupe above and `saveActiveGraph`'s
        // collision check both exist to prevent -- and it hides: the panel
        // draws one row, the click raises the stale tab by array order, and
        // that tab's next one-click Save overwrites the graph this rename
        // just produced without asking anything. Cleared rather than moved:
        // the tab keeps its graph on screen and simply has nowhere to save
        // back to, so its next Save asks for a name. That is what
        // `handleDelete` below does for the same reason, and it is the safe
        // direction when the alternative is a silent overwrite.
        //
        // Not when the destination IS the source, which a case-only or
        // punctuation-only rename produces -- "a b" and "a.b" both sanitize
        // to `a_b`. There the holders of the destination are the very tabs
        // the rebind is about to move, and clearing them first would unbind
        // the graph the user just renamed and leave the rebind nothing to
        // find.
        if (file !== graph.file) useTabStore.getState().rebindGraphFile(file, null);
        // EVERY tab that was saving back to this file, not just the one in
        // front of the user: a tab left on the old name writes a second copy
        // of the graph there on its next Save, and a background tab is the
        // one nobody would think to look at.
        //
        // The new DISPLAY NAME travels with the new file, and `next` is it --
        // the string the user typed, which is what the server stored as the
        // graph's name. A tab moved onto `Renamed_Graph` while still holding
        // "alpha" writes "alpha" back into it on its next in-place save, so
        // the graph the user renamed renames itself back the first time they
        // press Save.
        useTabStore.getState().rebindGraphFile(graph.file, { file, name: next });
        if (useProjectStore.getState().projectDir !== null) announceWorktreeWrite();
        useToastStore.getState().addToast(t('graphs.rename.success', { name: next }), 'success');
        load(true);
      } catch (e) {
        useToastStore.getState().addToast(
          t('graphs.rename.fail', { error: (e as Error).message }),
          'error',
        );
      }
    },
    [load, t],
  );

  const handleDelete = useCallback(
    async (graph: SavedGraphSummary) => {
      const ok = await confirm({
        title: t('graphs.delete.confirm', { name: graph.name }),
        confirmText: t('graphs.delete.confirmAction'),
        variant: 'danger',
      });
      if (!ok) return;
      try {
        await deleteGraph(graph.file);
        // The graphs on screen are still whole; they simply have nowhere to
        // save back to any more, so the next Save asks for a name rather than
        // silently recreating the file the user just deleted. Every bound
        // tab, for the same reason the rename above rebinds every one. Null
        // drops the display name with the file: there is no longer a graph on
        // disk for it to be the name of.
        useTabStore.getState().rebindGraphFile(graph.file, null);
        if (useProjectStore.getState().projectDir !== null) announceWorktreeWrite();
        useToastStore.getState().addToast(
          t('graphs.delete.success', { name: graph.name }),
          'success',
        );
        load(true);
      } catch (e) {
        useToastStore.getState().addToast(
          t('graphs.delete.fail', { error: (e as Error).message }),
          'error',
        );
      }
    },
    [load, t],
  );

  const handleSaveAs = useCallback(async () => {
    await saveActiveGraph({ saveAs: true });
    // Unconditional: a save that was cancelled or refused only costs one
    // list read, and asking `saveActiveGraph` to report which it was would
    // be a change to the toolbar's Save As as well. Quiet for that reason
    // too -- a cancelled Save As must not blank the panel behind it.
    load(true);
  }, [load]);

  const handleImport = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    // Not awaited: the read finishes on its own, and clearing the input has
    // to happen now so picking the SAME file again still fires `change`.
    void importGraphFile(file);
    event.target.value = '';
  }, []);

  return (
    <>
      <div className={styles.header}>
        <div className={styles.headerRow}>
          <div className={styles.headerTitle}>{t('sidebar.tab.graphs')}</div>
          <button
            type="button"
            className={styles.toolbarButton}
            onClick={() => void handleSaveAs()}
            aria-label={t('graphs.saveAs')}
            title={t('graphs.saveAs')}
          >
            {/* Not SaveIcon: that one is the toolbar's Save, a hand's width
                away and on screen at the same time, and it overwrites the
                bound file without asking. This button always stops for a
                name, so it gets the floppy with the `+`. */}
            <SaveAsIcon size={13} />
          </button>
          <button
            type="button"
            className={styles.toolbarButton}
            // Wrapped, not passed: `load`'s first argument is `quiet`, and a
            // click handler would hand it the event.
            onClick={() => load()}
            aria-label={t('graphs.refresh')}
            title={t('graphs.refresh')}
          >
            <RefreshIcon size={13} />
          </button>
        </div>
        <input
          type="text"
          placeholder={t('graphs.search')}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className={styles.searchInput}
        />
      </div>

      <div className={styles.panelBody}>
        {loading && <div className={styles.stateMessage}>{t('graphs.loading')}</div>}

        {!loading && error !== null && (
          <div className={styles.errorWrapper}>
            <div className={styles.errorText}>{t('graphs.listFail', { error })}</div>
            <button type="button" onClick={() => load()} className={styles.retryButton}>
              {t('palette.retry')}
            </button>
          </div>
        )}

        {!loading && error === null && (
          rows.length === 0 ? (
            <div className={styles.stateMessageMuted}>
              {/* The query is not repeated back: it is in the box directly
                  above, and echoing it put a string of any length someone
                  pasted into a centred message inside an `overflow: hidden`
                  panel, where a single unbroken 200-character token had
                  nowhere to wrap. The Templates tab next door says it the
                  same way. */}
              {query ? t('graphs.noMatch') : t('graphs.empty')}
            </div>
          ) : (
            /* `role="list"`, because `list-style: none` takes list semantics
               away from a `<ul>` in Safari. */
            <ul className={tabStyles.list} role="list">
              {rows.map((graph) => (
                <GraphRow
                  key={graph.file}
                  graph={graph}
                  current={graph.file === boundFile}
                  onOpen={(g) => void handleOpen(g)}
                  onRename={(g) => void handleRename(g)}
                  onDelete={(g) => void handleDelete(g)}
                />
              ))}
            </ul>
          )
        )}
      </div>

      <div className={styles.footer}>
        <button
          type="button"
          className={tabStyles.importButton}
          onClick={() => fileInputRef.current?.click()}
        >
          {t('graphs.import')}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          className={tabStyles.fileInput}
          onChange={handleImport}
        />
      </div>
    </>
  );
}
