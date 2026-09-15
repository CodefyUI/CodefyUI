import { useState, useCallback, useRef, useEffect } from 'react';
import { useGraphExecution } from '../../hooks/useGraphExecution';
import { useDeviceOptions, deviceLabel } from '../../hooks/useDeviceOptions';
import { useTabStore } from '../../store/tabStore';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useUIStore } from '../../store/uiStore';
import { listGraphs, createPreset, exportGraph } from '../../api/rest';
import { useI18n, SUPPORTED_LOCALES } from '../../i18n';
import type { TranslationKey } from '../../i18n';
import { subgraphIdOf } from '../../utils/subgraph';
import { graphToSvg, svgToPngBlob } from '../../utils/exportDiagram';
import { confirm, prompt } from '../../utils/dialog';
import { saveActiveGraph } from '../../utils/saveActiveGraph';
import { openSavedGraph } from '../../utils/openSavedGraph';
import type { SavedGraphTarget } from '../../utils/openSavedGraph';
import { importGraphFile } from '../../utils/importGraphFile';
import { CustomNodeManager } from '../CustomNodeManager/CustomNodeManager';
import { useToastStore } from '../../store/toastStore';
import type { LayoutMode } from '../../utils/autoLayout';
import { SettingsPopover } from './SettingsPopover';
import { PluginToolbarButtons } from './PluginToolbarButtons';
import { FontSizeMenu } from './FontSizeMenu';
import { ProjectBadge } from './ProjectBadge';
import styles from './Toolbar.module.css';

/* ── Shared dropdown menu ───────────────────────────────────────── */

interface MenuItem {
  label: string;
  title?: string;
  onClick: () => void;
  dividerAfter?: boolean;
}

function MenuDropdown({
  label,
  items,
  open,
  onToggle,
  onClose,
}: {
  label: string;
  items: MenuItem[];
  open: boolean;
  onToggle: () => void;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open, onClose]);

  return (
    <div ref={ref} className={styles.menuWrapper}>
      <button type="button"
        onClick={onToggle}
        className={`${styles.ghost} ${open ? styles.open : ''}`}
      >
        {label}
      </button>
      {open && (
        <div className={styles.menuPanel}>
          {items.map((item, i) => (
            <div key={i}>
              <button type="button"
                onClick={() => { item.onClick(); onClose(); }}
                className={styles.menuItem}
                title={item.title}
              >
                {item.label}
              </button>
              {/* No menu item sets dividerAfter: true, so the divider is never rendered */}
              {/* v8 ignore start */}
              {item.dividerAfter && <div className={styles.menuDivider} />}
              {/* v8 ignore stop */}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Load menu (two levels: destination, then which saved graph) ── */

/** Menu-local name for the destinations `openSavedGraph` documents. */
type LoadTarget = SavedGraphTarget;

interface SavedGraph {
  name: string;
  file: string;
}

function LoadSubMenu({
  open,
  onToggle,
  onClose,
  onLoadGraph,
  onImport,
  t,
}: {
  open: boolean;
  onToggle: () => void;
  onClose: () => void;
  onLoadGraph: (graph: SavedGraph, target: LoadTarget) => void;
  onImport: () => void;
  t: (key: TranslationKey, vars?: Record<string, string | number>) => string;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open, onClose]);

  return (
    <div ref={ref} className={styles.menuWrapper}>
      <button type="button"
        onClick={onToggle}
        className={`${styles.ghost} ${open ? styles.open : ''}`}
        aria-haspopup="menu"
        aria-expanded={open}
      >
        {t('toolbar.load')}
      </button>
      {open && (
        <LoadSubMenuPanel
          onLoadGraph={onLoadGraph}
          onImport={onImport}
          onClose={onClose}
          t={t}
        />
      )}
    </div>
  );
}

const LOAD_TARGETS: { key: LoadTarget; label: TranslationKey; title: TranslationKey }[] = [
  { key: 'canvas', label: 'toolbar.load.toCanvas', title: 'toolbar.load.toCanvas.title' },
  { key: 'bind', label: 'toolbar.load.andSave', title: 'toolbar.load.andSave.title' },
];

/**
 * The first level of {@link LoadSubMenu}: pick what the load should do, then
 * pick the graph from the flyout that opens beside it.
 *
 * Mounted only while the menu is open, so the saved-graph list is fetched
 * once on mount rather than synced off an `open` prop. The list and the
 * search box live HERE rather than in the flyout so that hovering from one
 * destination to the other neither refetches nor throws away what the user
 * has already typed.
 */
function LoadSubMenuPanel({
  onLoadGraph,
  onImport,
  onClose,
  t,
}: {
  onLoadGraph: (graph: SavedGraph, target: LoadTarget) => void;
  onImport: () => void;
  onClose: () => void;
  t: (key: TranslationKey, vars?: Record<string, string | number>) => string;
}) {
  const [graphs, setGraphs] = useState<SavedGraph[]>([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState('');
  const [target, setTarget] = useState<LoadTarget | null>(null);

  useEffect(() => {
    let cancelled = false;
    listGraphs()
      .then((result) => {
        if (!cancelled) setGraphs(Array.isArray(result) ? result : []);
      })
      .catch(() => {
        if (!cancelled) setGraphs([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className={`${styles.menuPanel} ${styles.menuPanelFlyoutHost}`}>
      {LOAD_TARGETS.map(({ key, label, title }) => (
        <div key={key} className={styles.submenuRow}>
          <button type="button"
            className={`${styles.menuItem} ${styles.menuItemSub} ${target === key ? styles.menuItemOpen : ''}`}
            title={t(title)}
            // Hover opens it the way a native submenu does; the click is what
            // keyboard and touch have instead of a hover, so it opens too --
            // never toggles, or moving the pointer across the rows would
            // leave the one under it shut.
            onClick={() => setTarget(key)}
            onMouseEnter={() => setTarget(key)}
            aria-haspopup="menu"
            aria-expanded={target === key}
          >
            <span>{t(label)}</span>
            <span className={styles.submenuCaret} aria-hidden="true">▸</span>
          </button>
          {target === key && (
            <SavedGraphPicker
              graphs={graphs}
              loading={loading}
              query={query}
              onQueryChange={setQuery}
              onPick={(graph) => { onLoadGraph(graph, key); onClose(); }}
              t={t}
            />
          )}
        </div>
      ))}
      <div className={styles.menuDivider} />
      <button type="button"
        onClick={() => { onImport(); onClose(); }}
        className={styles.menuItem}
        style={{ color: 'var(--accent)' }}
      >
        {t('toolbar.import')}
      </button>
    </div>
  );
}

/**
 * The saved-graph flyout: a search box over a list that SCROLLS.
 *
 * Both are the fix for the same bug -- the list used to render straight into
 * the shared `.menuPanel`, which clips at `overflow: hidden` with no height
 * cap, so once a project held more graphs than fit on screen the ones past
 * the bottom could be neither scrolled to nor clicked.
 */
function SavedGraphPicker({
  graphs,
  loading,
  query,
  onQueryChange,
  onPick,
  t,
}: {
  graphs: SavedGraph[];
  loading: boolean;
  query: string;
  onQueryChange: (value: string) => void;
  onPick: (graph: SavedGraph) => void;
  t: (key: TranslationKey, vars?: Record<string, string | number>) => string;
}) {
  const needle = query.trim().toLowerCase();
  // Matched on the file name as well as the label: the two differ once a name
  // has been sanitized, and the file is what the row's tooltip shows.
  const matches = needle === ''
    ? graphs
    : graphs.filter(
        (g) =>
          g.name.toLowerCase().includes(needle) || g.file.toLowerCase().includes(needle),
      );

  return (
    <div className={styles.submenuPanel} role="menu">
      <input
        type="text"
        className={styles.submenuSearch}
        value={query}
        onChange={(e) => onQueryChange(e.target.value)}
        placeholder={t('toolbar.load.search')}
        aria-label={t('toolbar.load.search')}
        autoFocus
      />
      <div className={styles.submenuList}>
        {loading ? (
          <div className={styles.menuMessage}>{t('toolbar.load.loading')}</div>
        ) : graphs.length === 0 ? (
          <div className={styles.menuMessageDim}>{t('toolbar.load.empty')}</div>
        ) : matches.length === 0 ? (
          <div className={styles.menuMessageDim}>{t('toolbar.load.noMatch', { query })}</div>
        ) : (
          matches.map((g) => (
            <button type="button"
              key={g.file}
              onClick={() => onPick(g)}
              className={`${styles.menuItem} ${styles.submenuItem}`}
              title={g.file}
              role="menuitem"
            >
              {g.name}
            </button>
          ))
        )}
      </div>
    </div>
  );
}

/* ── Main Toolbar ───────────────────────────────────────────────── */

export function Toolbar() {
  const { execute, stop } = useGraphExecution();
  const { clear, getSerializedGraph } = useTabStore();
  const activeTab = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)!);
  const status = activeTab.status;
  // Per-graph device (A8): the select writes `graphDevice`, and the empty
  // option names the Settings device a run falls back to.
  const setGraphDevice = useTabStore((s) => s.setGraphDevice);
  const globalDevice = useUIStore((s) => s.globalDevice);
  const { devices } = useDeviceOptions();
  const followDevice = devices.find((d) => d.value === globalDevice);
  const followLabel = followDevice ? deviceLabel(followDevice) : globalDevice;
  const storedDevice = activeTab.graphDevice;
  const storedDeviceListed = storedDevice === null || devices.some((d) => d.value === storedDevice);
  const { reload, fetchDefinitions } = useNodeDefStore();
  const { t, locale, setLocale } = useI18n();
  const addToast = useToastStore((s) => s.addToast);

  const [openMenu, setOpenMenu] = useState<string | null>(null);
  const [langMenuOpen, setLangMenuOpen] = useState(false);
  const [layoutMenuOpen, setLayoutMenuOpen] = useState(false);
  const [customNodeManagerOpen, setCustomNodeManagerOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [fontSizeMenuOpen, setFontSizeMenuOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const settingsTriggerRef = useRef<HTMLButtonElement>(null);
  const fontSizeTriggerRef = useRef<HTMLButtonElement>(null);
  const langTriggerRef = useRef<HTMLDivElement>(null);
  const layoutTriggerRef = useRef<HTMLDivElement>(null);

  const lastLayoutMode = useUIStore((s) => s.lastLayoutMode);
  const setLastLayoutMode = useUIStore((s) => s.setLastLayoutMode);
  const applyLayout = useTabStore((s) => s.applyLayout);
  const selectedCount = useTabStore((s) => {
    const tab = s.tabs.find((tt) => tt.id === s.activeTabId);
    // An active tab always exists while the toolbar is mounted, so the ?? 0 fallback is dead
    /* v8 ignore start */
    return tab?.nodes.filter((n) => n.selected).length ?? 0;
    /* v8 ignore stop */
  });

  const runLayout = useCallback(
    (mode: LayoutMode) => {
      setLastLayoutMode(mode);
      applyLayout(mode);
      setLayoutMenuOpen(false);
    },
    [applyLayout, setLastLayoutMode],
  );

  // Close layout dropdown on outside click
  useEffect(() => {
    if (!layoutMenuOpen) return;
    const handler = (e: MouseEvent) => {
      if (layoutTriggerRef.current && !layoutTriggerRef.current.contains(e.target as Node)) {
        setLayoutMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [layoutMenuOpen]);

  const isRunning = status === 'running';

  const closeMenus = useCallback(() => setOpenMenu(null), []);
  const toggleMenu = useCallback((name: string) => {
    setOpenMenu((prev) => (prev === name ? null : name));
  }, []);

  /* ── Handlers ─────────────────────────────────────────────────── */

  const handleRun = useCallback(() => execute(), [execute]);
  const handleStop = useCallback(() => stop(), [stop]);

  const handleSave = useCallback(() => saveActiveGraph(), []);
  const handleSaveAs = useCallback(() => saveActiveGraph({ saveAs: true }), []);

  const handleClear = useCallback(async () => {
    const ok = await confirm({
      title: t('toolbar.clear.confirm'),
      confirmText: t('toolbar.clear'),
      variant: 'danger',
    });
    if (ok) clear();
  }, [clear, t]);

  const handleLoadGraph = useCallback(
    async (graph: SavedGraph, target: LoadTarget) => {
      // Only the unbound path asks. `bind` is the load this menu has always
      // performed, and adding a confirm to it here would be a change to a
      // second thing in a change about the first one. An empty canvas has
      // nothing to lose, so it is not worth a dialog either.
      const { tabs, activeTabId } = useTabStore.getState();
      const canvasHasWork = tabs.find((tb) => tb.id === activeTabId)!.nodes.length > 0;
      if (target === 'canvas' && canvasHasWork) {
        const ok = await confirm({
          title: t('toolbar.load.toCanvas.confirm', { name: graph.name }),
          confirmText: t('toolbar.load.toCanvas.confirmAction'),
          variant: 'danger',
        });
        if (!ok) return;
      }
      // Everything after the click -- the read, the preset merge, the one
      // install, the binding, the failure toast -- is `openSavedGraph`,
      // which the Graphs panel opens its rows with too. The confirm above
      // is the one part the two surfaces disagree about, so it is the one
      // part that stayed behind.
      await openSavedGraph(graph, target);
    },
    [t],
  );

  const handleImportFile = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    // Not awaited: the read finishes on its own, and clearing the input has
    // to happen now so picking the SAME file again still fires `change`.
    void importGraphFile(file);
    event.target.value = '';
  }, []);

  const handleExportJson = useCallback(() => {
    const { nodes, edges, presets, segmentGroups, subgraphs, settings } = getSerializedGraph();
    if (nodes.length === 0) {
      addToast(t('toolbar.exportJson.empty'), 'warning');
      return;
    }
    const name = activeTab.name || 'graph';
    const data = {
      name, description: activeTab.description ?? '', nodes, edges, presets, segmentGroups, subgraphs,
      // Only when the graph assigns a device, so an unassigned export stays
      // byte-identical.
      ...(settings ? { settings } : {}),
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${name.replace(/[^a-zA-Z0-9_-]/g, '_')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [getSerializedGraph, activeTab.name, activeTab.description, t, addToast]);

  const handleExportSubgraph = useCallback(async () => {
    const { nodes, edges, subgraphs } = getSerializedGraph();
    if (nodes.length === 0) {
      addToast(t('toolbar.export.empty'), 'warning');
      return;
    }
    // core#137. A preset is stored server-side as {nodes, edges} and nothing
    // else -- there is no slot for a subgraph definition. Building one from a
    // canvas that contains an instance node would register a preset holding a
    // bare `subgraph:<id>` node whose definition can never accompany it:
    // broken for everyone who ever drops it, and broken permanently, because
    // the preset outlives the graph it came from.
    //
    // Refuse rather than strip the instances. Stripping is the silent option:
    // the user asked to turn THIS canvas into a reusable block and would get
    // one quietly missing an arbitrary piece of it (plus every edge that
    // touched the removed instance), with nothing in the result to say so.
    // Refusing costs one step the user already knows how to take -- Expand on
    // the block, from its context menu -- and the retry then yields a preset
    // that really does contain the whole graph.
    const instanceIds = Array.from(
      new Set(
        nodes
          .map((n) => subgraphIdOf(n.type))
          .filter((id): id is string => id !== null),
      ),
    );
    if (instanceIds.length > 0) {
      // `.trim()` before the fallback, and the id itself defaulted: a
      // whitespace-only name is truthy, so `|| id` alone rendered
      // "collapsed blocks ( )", and an instance whose type is a bare
      // `subgraph:` yields an EMPTY id, which rendered "collapsed blocks ()".
      // Both are messages that name nothing while looking like they do.
      const names = instanceIds.map((id) => {
        const name = subgraphs.find((d) => d.id === id)?.name?.trim();
        return name || id || t('subgraph.unnamed');
      });
      addToast(
        t('toolbar.export.subgraphRefused', { names: names.join(', ') }),
        'error',
      );
      return;
    }
    const name = await prompt({
      title: t('toolbar.export.prompt'),
      placeholder: 'preset-name',
    });
    if (!name?.trim()) return;
    try {
      await createPreset({ name: name.trim(), nodes, edges });
      await fetchDefinitions();
      addToast(t('toolbar.export.success', { name: name.trim() }), 'success');
    } catch (e) {
      addToast(t('toolbar.export.fail', { error: (e as Error).message }), 'error');
    }
  }, [getSerializedGraph, fetchDefinitions, t, addToast]);

  const handleExportPython = useCallback(async () => {
    const serialized = getSerializedGraph();
    const noteIds = new Set(
      serialized.nodes.filter((node) => node.type === 'note').map((node) => node.id),
    );
    const nodes = serialized.nodes.filter((node) => !noteIds.has(node.id));
    const edges = serialized.edges.filter(
      (edge) => !noteIds.has(edge.source) && !noteIds.has(edge.target),
    );
    if (nodes.length === 0) {
      addToast(t('toolbar.exportPython.empty'), 'warning');
      return;
    }
    const name = activeTab.name || 'graph';
    try {
      // core#137 (the trailing argument): an instance node is just
      // `subgraph:<id>` until the definition it names travels with it, and
      // definitions are graph-local -- there is no server-side registry to
      // resolve the id against. Omit them and the backend rejects any graph
      // containing a collapsed block with `Unknown subgraph: <id>`, i.e. a
      // flat 400 on Export -> Python for the entire feature.
      const result = await exportGraph(
        nodes,
        edges,
        name,
        serialized.presets,
        { seed: activeTab.seed, deterministic: activeTab.deterministic },
        serialized.subgraphs,
        // The graph's device travels with the export and becomes the
        // script's `--device` default.
        serialized.settings,
      );
      const blob = new Blob([result.script], { type: 'text/x-python' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${name.replace(/[^a-zA-Z0-9_-]/g, '_')}.py`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      addToast(t('toolbar.exportPython.fail', { error: (e as Error).message }), 'error');
    }
  }, [getSerializedGraph, activeTab.name, activeTab.seed,
      activeTab.deterministic, t, addToast]);

  const handleExportDiagram = useCallback(
    async (format: 'svg' | 'png') => {
      // Architecture diagram = nodes + their ports + connections (no param
      // values), built from the live nodes/edges rather than the serialized
      // graph (which drops the labels, category colors and ports the diagram
      // needs). Notes are annotations, not architecture, so they don't count.
      const drawable = activeTab.nodes.filter((n) => n.type !== 'noteNode');
      if (drawable.length === 0) {
        addToast(t('toolbar.exportDiagram.empty'), 'warning');
        return;
      }
      const base = (activeTab.name || 'graph').replace(/[^a-zA-Z0-9_-]/g, '_');
      const svg = graphToSvg(activeTab.nodes, activeTab.edges);
      try {
        const blob =
          format === 'png'
            ? await svgToPngBlob(svg)
            : new Blob([svg], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${base}-architecture.${format}`;
        a.click();
        URL.revokeObjectURL(url);
      } catch (e) {
        addToast(t('toolbar.exportDiagram.fail', { error: (e as Error).message }), 'error');
      }
    },
    [activeTab.nodes, activeTab.edges, activeTab.name, t, addToast],
  );

  const handleReloadNodes = useCallback(async () => {
    try { await reload(); }
    catch (e) { addToast(t('toolbar.reload.fail', { error: (e as Error).message }), 'error'); }
  }, [reload, t, addToast]);

  /* ── Menu definitions ─────────────────────────────────────────── */

  // No `title` on these three: "Save graph", "Save under a new name" and
  // "Remove every node from this canvas" were their own visible labels said
  // again at length, in the File menu of a graph editor. The one fact worth
  // stating before Clear Canvas -- that unsaved work goes -- is in the confirm
  // dialog it raises. The Export items keep theirs, which name a file format.
  const fileMenuItems: MenuItem[] = [
    { label: t('toolbar.save'), onClick: handleSave },
    { label: t('toolbar.saveAs'), onClick: handleSaveAs },
    { label: t('toolbar.clear'), onClick: handleClear },
  ];

  const exportMenuItems: MenuItem[] = [
    { label: t('toolbar.exportJson'), title: t('toolbar.exportJson.title'), onClick: handleExportJson },
    { label: t('toolbar.exportDiagram.svg'), title: t('toolbar.exportDiagram.title'), onClick: () => handleExportDiagram('svg') },
    { label: t('toolbar.exportDiagram.png'), title: t('toolbar.exportDiagram.title'), onClick: () => handleExportDiagram('png') },
    { label: t('toolbar.export'), title: t('toolbar.export.title'), onClick: handleExportSubgraph },
    { label: t('toolbar.exportPython'), title: t('toolbar.exportPython.title'), onClick: handleExportPython },
  ];

  /* ── Status visuals ───────────────────────────────────────────── */

  const statusKey = `status.${status}` as const;
  // Token-mapped onto the app's canonical run-status palette (tokens.css
  // --status-*, mirrored in styles/theme.ts STATUS_COLORS) instead of this
  // component's own drifted copy — see migration report.
  const statusDotColors: Record<string, string> = {
    idle: 'var(--status-idle)',
    running: 'var(--status-running)',
    completed: 'var(--status-completed)',
    error: 'var(--status-error)',
    cached: 'var(--status-cached)',
    skipped: 'var(--status-skipped)',
  };
  const statusDotColor = statusDotColors[status] ?? 'var(--status-idle)';
  // Must stay translucent: a solid colour here draws a hard ring, not a glow.
  const statusGlow = status === 'running' ? 'var(--glow-running)' : 'none';

  return (
    <div className={styles.root}>
      {/* Logo */}
      <div className={styles.logo}>
        <span className={styles.logoBrand}>Codefy</span>
        <span className={styles.logoSuffix}>UI</span>
      </div>
      <ProjectBadge />

      {/* Run / Stop */}
      <div className={styles.cluster}>
        {/* Both labels are always visible -- no responsive rule hides them --
            so "Run the graph" and "Stop execution" were the words on the
            button read back with the only object they could have. */}
        <button type="button"
          onClick={handleRun}
          disabled={isRunning}
          className={styles.runButton}
        >
          {isRunning ? t('toolbar.running') : t('toolbar.run')}
        </button>
        <button type="button"
          onClick={handleStop}
          disabled={!isRunning}
          className={styles.stopButton}
        >
          {t('toolbar.stop')}
        </button>
        {/* Stays enabled while a run is in flight: that run already holds
            its device, and the choice applies to the next one. */}
        <select
          aria-label={t('toolbar.device.aria')}
          title={t('toolbar.device.title')}
          className={styles.deviceSelect}
          value={storedDevice ?? ''}
          disabled={activeTab.readOnly}
          onChange={(e) => setGraphDevice(e.target.value || null)}
        >
          <option value="">{t('toolbar.device.follow', { device: followLabel })}</option>
          {/* The file names a device this server does not list (cuda:1 on a
              box without it, or auto). Shown as it is stored, so the select
              keeps the value and a Save keeps the file's assignment. */}
          {!storedDeviceListed && (
            <option value={storedDevice as string} disabled>{storedDevice}</option>
          )}
          {devices.map((d) => (
            <option key={d.value} value={d.value}>{deviceLabel(d)}</option>
          ))}
        </select>
      </div>

      <div className={styles.divider} />

      {/* File ops */}
      <div className={styles.cluster}>
        <MenuDropdown
          label={t('toolbar.menu.file')}
          items={fileMenuItems}
          open={openMenu === 'file'}
          onToggle={() => toggleMenu('file')}
          onClose={closeMenus}
        />
        <LoadSubMenu
          open={openMenu === 'load'}
          onToggle={() => toggleMenu('load')}
          onClose={closeMenus}
          onLoadGraph={handleLoadGraph}
          onImport={() => fileInputRef.current?.click()}
          t={t}
        />
        <MenuDropdown
          label={t('toolbar.menu.export')}
          items={exportMenuItems}
          open={openMenu === 'export'}
          onToggle={() => toggleMenu('export')}
          onClose={closeMenus}
        />
      </div>

      <div className={styles.divider} />

      {/* Node management */}
      <div className={styles.cluster}>
        <button type="button"
          onClick={() => useUIStore.getState().openTemplateGallery()}
          title={t('gallery.open.title')}
          className={`${styles.ghost} ${styles.ghostMuted}`}
        >
          {t('gallery.open')}
        </button>
        <button type="button"
          onClick={handleReloadNodes}
          title={t('toolbar.reloadNodes.title')}
          className={`${styles.ghost} ${styles.ghostMuted}`}
        >
          {t('toolbar.reloadNodes')}
        </button>
        {/* "Manage custom nodes" was the visible label with a verb in front,
            and the panel it opens is titled Custom Node Manager. */}
        <button type="button"
          onClick={() => setCustomNodeManagerOpen(true)}
          className={`${styles.ghost} ${styles.ghostMuted}`}
        >
          {t('toolbar.customNodes')}
        </button>
      </div>

      <div className={styles.divider} />

      {/* Auto Layout + Status */}
      <div className={styles.cluster}>
        <div ref={layoutTriggerRef} className={styles.splitButton}>
          <button type="button"
            className={styles.splitButtonMain}
            onClick={() => runLayout(lastLayoutMode)}
            title={t('toolbar.autoLayout')}
          >
            {t('toolbar.autoLayout')}
          </button>
          <button type="button"
            className={styles.splitButtonCaret}
            onClick={() => setLayoutMenuOpen((v) => !v)}
            aria-label={t('toolbar.layoutMode.aria')}
          >
            ▾
          </button>
          {layoutMenuOpen && (
            <div className={styles.layoutDropdown}>
              <div
                className={`${styles.layoutDropdownItem} ${lastLayoutMode === 'experiments' ? styles.layoutDropdownItemActive : ''}`}
                onClick={() => runLayout('experiments')}
              >
                {t('toolbar.autoLayout.experiments')}
              </div>
              <div
                className={`${styles.layoutDropdownItem} ${lastLayoutMode === 'all' ? styles.layoutDropdownItemActive : ''}`}
                onClick={() => runLayout('all')}
              >
                {t('toolbar.autoLayout.all')}
              </div>
              <div
                className={`${styles.layoutDropdownItem} ${selectedCount === 0 ? styles.layoutDropdownItemDisabled : ''} ${lastLayoutMode === 'selected' ? styles.layoutDropdownItemActive : ''}`}
                onClick={() => {
                  if (selectedCount > 0) runLayout('selected');
                }}
              >
                {t('toolbar.autoLayout.selected', { count: selectedCount })}
              </div>
            </div>
          )}
        </div>

        <div className={styles.status}>
          <span
            className={styles.statusDot}
            style={{ background: statusDotColor, boxShadow: statusGlow }}
          />
          <span style={{ color: status === 'running' ? 'var(--status-running)' : undefined }}>
            {t(statusKey)}
          </span>
        </div>
      </div>

      {/* RIGHT cluster: plugin buttons, Settings, Help, FontSize, Language */}
      <div className={`${styles.cluster} ${styles.right}`}>
        {/* Plugin buttons (#132) lead the right-hand group so an installed
            plugin never pushes Settings or Help off the row. Renders nothing
            at all — no element, no gap — when no plugin registered one. */}
        <PluginToolbarButtons />

        {/* Settings ⚙ */}
        <div className={styles.menuWrapper}>
          <button type="button"
            ref={settingsTriggerRef}
            onClick={() => setSettingsOpen((v) => !v)}
            title={t('toolbar.settings.title')}
            className={`${styles.iconBtn} ${settingsOpen ? styles.active : ''}`}
            aria-label={t('toolbar.settings')}
            aria-expanded={settingsOpen}
          >
            ⚙
          </button>
          <SettingsPopover
            open={settingsOpen}
            onClose={() => setSettingsOpen(false)}
            triggerRef={settingsTriggerRef}
          />
        </div>

        {/* Help ? — opens shortcuts modal */}
        <button type="button"
          onClick={() => useUIStore.getState().toggleShortcutsModal()}
          className={styles.iconBtn}
          title={t('shortcuts.title')}
          aria-label={t('shortcuts.title')}
        >
          ?
        </button>

        {/* Font size Aa */}
        <div className={styles.menuWrapper}>
          <button type="button"
            ref={fontSizeTriggerRef}
            onClick={() => setFontSizeMenuOpen((v) => !v)}
            className={`${styles.dropdown} ${styles.dropdownNoCaret} ${fontSizeMenuOpen ? styles.open : ''}`}
            title={t('toolbar.fontSize.title')}
            aria-label={t('toolbar.fontSize.title')}
            aria-expanded={fontSizeMenuOpen}
          >
            Aa
          </button>
          <FontSizeMenu
            open={fontSizeMenuOpen}
            onClose={() => setFontSizeMenuOpen(false)}
            triggerRef={fontSizeTriggerRef}
          />
        </div>

        {/* Language */}
        <div ref={langTriggerRef} className={styles.menuWrapper}>
          <button type="button"
            onClick={() => setLangMenuOpen((v) => !v)}
            className={`${styles.dropdown} ${langMenuOpen ? styles.open : ''}`}
            aria-label={t('toolbar.language.aria')}
            aria-expanded={langMenuOpen}
          >
            {SUPPORTED_LOCALES.find((l) => l.code === locale)?.label ?? locale}
          </button>
          {langMenuOpen && (
            <>
              <div className={styles.overlay} onClick={() => setLangMenuOpen(false)} />
              <div className={`${styles.menuPanel} ${styles.menuPanelRight}`}>
                {SUPPORTED_LOCALES.map((l) => (
                  <button type="button"
                    key={l.code}
                    onClick={() => { setLocale(l.code); setLangMenuOpen(false); }}
                    className={`${styles.langOption} ${l.code === locale ? styles.activeOption : ''}`}
                  >
                    <span>{l.nativeName}</span>
                    {l.code === locale && <span className={styles.langOptionCheck}>✓</span>}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".json"
        className={styles.fileInput}
        onChange={handleImportFile}
      />

      {customNodeManagerOpen && (
        <CustomNodeManager onClose={() => setCustomNodeManagerOpen(false)} />
      )}
    </div>
  );
}
