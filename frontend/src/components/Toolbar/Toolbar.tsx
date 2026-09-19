import { useState, useCallback, useRef, useEffect } from 'react';
import { useGraphExecution } from '../../hooks/useGraphExecution';
import { useDeviceOptions, deviceLabel } from '../../hooks/useDeviceOptions';
import { useTabStore } from '../../store/tabStore';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useUIStore } from '../../store/uiStore';
import { createPreset, exportGraph } from '../../api/rest';
import { useI18n, SUPPORTED_LOCALES } from '../../i18n';
import { subgraphIdOf } from '../../utils/subgraph';
import { graphToSvg, svgToPngBlob } from '../../utils/exportDiagram';
import { confirm, prompt } from '../../utils/dialog';
import { saveActiveGraph } from '../../utils/saveActiveGraph';
import { exportWorkspace } from '../../utils/exportWorkspace';
import { CustomNodeManager } from '../CustomNodeManager/CustomNodeManager';
import { SaveIcon } from '../shared/Icons';
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
              {item.dividerAfter && <div className={styles.menuDivider} />}
            </div>
          ))}
        </div>
      )}
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
    // The divider sets the last item apart: the five above export THIS tab's
    // graph, the one below exports every tab.
    { label: t('toolbar.exportPython'), title: t('toolbar.exportPython.title'), onClick: handleExportPython, dividerAfter: true },
    { label: t('workspace.export'), title: t('workspace.export.title'), onClick: exportWorkspace },
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
        {/* Every separator is the LAST CHILD of the cluster it trails, never
            a sibling of it. `.root` is `flex-wrap: wrap`, so a divider that
            is a root flex item in its own right can be pushed onto the next
            line alone -- the second row then opens with a 1px rule that
            separates nothing, and the cluster behind it sits indented past
            the row above. Inside the cluster the rule travels with the
            content it belongs to and can only ever follow it. Nothing needs
            respacing: `.root` and `.cluster` share `gap: var(--sp-3)` and
            `.divider` keeps its own `margin: 0 var(--sp-1)`, so the run is
            gap + margin + rule + margin + gap either way. */}
        <div className={styles.divider} />
      </div>

      {/* File ops */}
      <div className={styles.cluster}>
        {/* Save is on this row twice on purpose: the File menu is where it
            lives beside its siblings, and this is the one-click form of the
            one command people run after every edit, kept within reach of Run.
            It carries a `title` the menu's Save deliberately does not -- an
            icon-only button shows no words, so this is the only label it has.
            14px is the ⚙ and ? glyph size beside it (.iconBtn is --fs-md). */}
        <button type="button"
          onClick={handleSave}
          className={styles.iconBtn}
          title={t('toolbar.save')}
          aria-label={t('toolbar.save')}
        >
          <SaveIcon size={14} />
        </button>
        <MenuDropdown
          label={t('toolbar.menu.file')}
          items={fileMenuItems}
          open={openMenu === 'file'}
          onToggle={() => toggleMenu('file')}
          onClose={closeMenus}
        />
        <MenuDropdown
          label={t('toolbar.menu.export')}
          items={exportMenuItems}
          open={openMenu === 'export'}
          onToggle={() => toggleMenu('export')}
          onClose={closeMenus}
        />
        {/* Trails its cluster from inside it -- see the first divider. */}
        <div className={styles.divider} />
      </div>

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
        {/* Trails its cluster from inside it -- see the first divider. */}
        <div className={styles.divider} />
      </div>

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

      {customNodeManagerOpen && (
        <CustomNodeManager onClose={() => setCustomNodeManagerOpen(false)} />
      )}
    </div>
  );
}
