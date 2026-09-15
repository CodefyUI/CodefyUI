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
import { importGraphFile } from '../../utils/importGraphFile';
import {
  openSavedGraph,
  readSavedGraphDocument,
  type SavedGraphTarget,
} from '../../utils/openSavedGraph';
import { saveActiveGraph } from '../../utils/saveActiveGraph';
import { announceWorktreeWrite } from '../../utils/worktreeWrite';
import { relativeTime } from '../SourceControl/scm';
import { ActionMenu, type ActionMenuItem } from '../shared/ActionMenu';
import { MoreHorizontalIcon, RefreshIcon, SaveIcon } from '../shared/Icons';
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

/** The active tab's nodes, which is all "is there work here to lose?" means. */
function canvasHasWork(): boolean {
  const { tabs, activeTabId } = useTabStore.getState();
  return (tabs.find((tb) => tb.id === activeTabId)?.nodes.length ?? 0) > 0;
}

// ── Graph row ──

interface GraphRowProps {
  graph: SavedGraphSummary;
  /** True when the ACTIVE tab saves back to this file. */
  current: boolean;
  onOpen: (graph: SavedGraphSummary, target: SavedGraphTarget) => void;
  onOpenInNewTab: (graph: SavedGraphSummary) => void;
  onRename: (graph: SavedGraphSummary) => void;
  onDelete: (graph: SavedGraphSummary) => void;
}

/**
 * One saved graph: click to open it into this tab, or reach the rest through
 * the menu at the end of the row.
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
  onOpenInNewTab,
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

  const items: ActionMenuItem[] = [
    {
      id: 'new-tab',
      label: t('graphs.openNewTab'),
      onSelect: () => onOpenInNewTab(graph),
    },
    {
      id: 'canvas',
      label: t('graphs.openOntoCanvas'),
      onSelect: () => onOpen(graph, 'canvas'),
    },
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
        onClick={() => onOpen(graph, 'bind')}
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
 * is the one that changes behind your back -- every Save from the toolbar
 * adds to it -- which is what the refresh control is for.
 *
 * Opening is deliberately split in two. A row click BINDS, so Save writes
 * back to the file that was opened; the menu's "onto canvas" leaves the tab
 * unbound, so the next Save asks where the result should go. Both replace
 * what is on the canvas through an install that pushes no undo frame, so
 * both ask first when there is anything there to lose -- a row in a list you
 * scroll through is not a menu item you aimed at.
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

  // Trimmed once: it decides both what is filtered and whether an empty body
  // reads as "nothing matched" or as "nothing saved", and a box holding only
  // spaces is not a search.
  const query = searchQuery.trim();
  const rows = useMemo(() => {
    const q = query.toLowerCase();
    return sortGraphs(q ? graphs.filter((g) => graphMatches(g, q)) : graphs);
  }, [graphs, query]);

  const handleOpen = useCallback(
    async (graph: SavedGraphSummary, target: SavedGraphTarget) => {
      if (canvasHasWork()) {
        const ok = await confirm({
          title: t('graphs.open.confirm', { name: graph.name }),
          confirmText: t('graphs.open.confirmAction'),
          variant: 'danger',
        });
        if (!ok) return;
      }
      // Everything after the question is `openSavedGraph` -- the read, the
      // preset merge, the one install, the binding, the failure toast -- so
      // a row opens a graph the way the toolbar's menu always did.
      await openSavedGraph(graph, target);
    },
    [t],
  );

  const handleOpenInNewTab = useCallback(
    async (graph: SavedGraphSummary) => {
      try {
        // Read BEFORE the tab exists, so a graph that cannot be read leaves
        // no empty tab behind to close.
        const doc = await readSavedGraphDocument(graph.file, graph.file);
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
        // writes to -- the same stamp `openSavedGraph` puts on a graph opened
        // into the tab already in front of the user.
        const projectDir = useProjectStore.getState().projectDir;
        if (projectDir !== null) useTabStore.getState().stampActiveTabProject(projectDir);
      } catch (e) {
        useToastStore.getState().addToast(
          t('toolbar.load.fail', { error: (e as Error).message }),
          'error',
        );
      }
    },
    [t],
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
        // EVERY tab that was saving back to this file, not just the one in
        // front of the user: a tab left on the old name writes a second copy
        // of the graph there on its next Save, and a background tab is the
        // one nobody would think to look at.
        useTabStore.getState().rebindGraphFile(graph.file, file);
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
        // tab, for the same reason the rename above rebinds every one.
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
            <SaveIcon size={13} />
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
                  onOpen={(g, target) => void handleOpen(g, target)}
                  onOpenInNewTab={(g) => void handleOpenInNewTab(g)}
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
