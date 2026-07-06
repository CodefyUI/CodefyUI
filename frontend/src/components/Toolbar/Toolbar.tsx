import { useState, useCallback, useRef, useEffect } from 'react';
import { useGraphExecution } from '../../hooks/useGraphExecution';
import { useTabStore } from '../../store/tabStore';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useUIStore } from '../../store/uiStore';
import { loadGraph, listGraphs, createPreset, exportGraph } from '../../api/rest';
import { useI18n, SUPPORTED_LOCALES } from '../../i18n';
import type { TranslationKey } from '../../i18n';
import { resolveSerializedNodes, resolveSerializedEdges } from '../../utils';
import { graphToSvg, svgToPngBlob } from '../../utils/exportDiagram';
import { confirm, prompt } from '../../utils/dialog';
import { saveActiveGraph } from '../../utils/saveActiveGraph';
import { GRAPH_FORMAT_VERSION } from '../../utils/formatVersion';
import { CustomNodeManager } from '../CustomNodeManager/CustomNodeManager';
import { useToastStore } from '../../store/toastStore';
import { useProjectStore } from '../../store/projectStore';
import { autoLayout, stackUnboundNotes, type LayoutMode } from '../../utils/autoLayout';
// Aliased: this file already casts DOM MouseEvent targets to the ambient
// lib.dom `Node` type (see the mousedown handlers below) -- importing
// @xyflow/react's `Node` unaliased would shadow that global and break them.
import type { Node as FlowNode } from '@xyflow/react';
import type { NodeData } from '../../types';
import { SettingsPopover } from './SettingsPopover';
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

/* ── Load sub-menu (lists saved graphs) ─────────────────────────── */

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
  onLoadGraph: (name: string) => void;
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

/**
 * The dropdown body for {@link LoadSubMenu}. Mounted only while the menu is
 * open, so the saved-graph list is fetched once on mount rather than synced
 * off an `open` prop.
 */
function LoadSubMenuPanel({
  onLoadGraph,
  onImport,
  onClose,
  t,
}: {
  onLoadGraph: (name: string) => void;
  onImport: () => void;
  onClose: () => void;
  t: (key: TranslationKey, vars?: Record<string, string | number>) => string;
}) {
  const [graphs, setGraphs] = useState<{ name: string; file: string }[]>([]);
  const [loading, setLoading] = useState(true);

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
    <div className={styles.menuPanel}>
      {loading ? (
        <div className={styles.menuMessage}>{t('toolbar.load.loading')}</div>
      ) : graphs.length === 0 ? (
        <div className={styles.menuMessageDim}>{t('toolbar.load.empty')}</div>
      ) : (
        graphs.map((g) => (
          <button type="button"
            key={g.file}
            onClick={() => { onLoadGraph(g.file); onClose(); }}
            className={styles.menuItem}
          >
            {g.name}
          </button>
        ))
      )}
      <div className={styles.menuDivider} />
      <button type="button"
        onClick={() => { onImport(); onClose(); }}
        className={styles.menuItem}
        style={{ color: '#06b6d4' }}
      >
        {t('toolbar.import')}
      </button>
    </div>
  );
}

/* ── Main Toolbar ───────────────────────────────────────────────── */

export function Toolbar() {
  const { execute, stop } = useGraphExecution();
  const { clear, getSerializedGraph, setNodes, setEdges, setDescription, setCurrentGraphFile, setSegmentGroups } = useTabStore();
  const activeTab = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)!);
  const status = activeTab.status;
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
    async (name: string) => {
      try {
        const graphData = await loadGraph(name);
        const rawNodes = graphData.nodes ?? [];
        const rawEdges = graphData.edges ?? [];
        const store = useNodeDefStore.getState();
        const savedPresets = Array.isArray(graphData.presets) ? graphData.presets : [];
        const mergedPresets = [...store.presets];
        for (const p of savedPresets) {
          if (!mergedPresets.some((ep) => ep.preset_name === p.preset_name)) {
            mergedPresets.push(p);
          }
        }
        const resolvedNodes = resolveSerializedNodes(rawNodes, store.definitions, mergedPresets);
        const resolvedEdges = resolveSerializedEdges(rawEdges);
        // Missing/incomplete layout (project mode): dagre-lay-out ALL nodes
        // directly -- NOT via applyLayout, which pushes an undo snapshot and a
        // toast -- then deterministically place unbound notes. The next save
        // persists the computed layout (spec 6.3).
        if (graphData.layout_missing) {
          const laid = stackUnboundNotes(
            autoLayout(resolvedNodes, resolvedEdges, 'all'),
          ) as FlowNode<NodeData>[];
          setNodes(laid);
        } else {
          setNodes(resolvedNodes);
        }
        setEdges(resolvedEdges);
        setDescription(typeof graphData.description === 'string' ? graphData.description : '');
        setSegmentGroups(Array.isArray(graphData.segmentGroups) ? graphData.segmentGroups : []);
        const tooNew = typeof graphData.format_version === 'number'
          && graphData.format_version > GRAPH_FORMAT_VERSION;
        useTabStore.getState().setTabReadOnly(tooNew);
        if (tooNew) {
          addToast(t('project.readOnly.loadNotice', { version: graphData.format_version }), 'warning');
        }
        // `name` is the sanitized file stem — bind the tab to it so re-saving
        // under the same name doesn't trigger the overwrite warning.
        setCurrentGraphFile(name);
        const projectDir = useProjectStore.getState().projectDir;
        if (projectDir !== null) useTabStore.getState().stampActiveTabProject(projectDir);
        if (savedPresets.length > 0) {
          useNodeDefStore.setState({ presets: mergedPresets });
        }
      } catch (e) {
        addToast(t('toolbar.load.fail', { error: (e as Error).message }), 'error');
      }
    },
    [setNodes, setEdges, setDescription, setSegmentGroups, setCurrentGraphFile, t, addToast],
  );

  const handleImportFile = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const data = JSON.parse(e.target?.result as string);
          const rawNodes = data.nodes ?? [];
          const edges = data.edges ?? [];
          if (!Array.isArray(rawNodes) || !Array.isArray(edges)) {
            throw new Error('Invalid graph format');
          }
          const store = useNodeDefStore.getState();
          const importedPresets = Array.isArray(data.presets) ? data.presets : [];
          const mergedPresets = [...store.presets];
          for (const p of importedPresets) {
            if (!mergedPresets.some((ep) => ep.preset_name === p.preset_name)) {
              mergedPresets.push(p);
            }
          }
          const resolvedNodes = resolveSerializedNodes(rawNodes, store.definitions, mergedPresets);
          const resolvedEdges = resolveSerializedEdges(edges);
          setNodes(resolvedNodes);
          setEdges(resolvedEdges);
          setDescription(typeof data.description === 'string' ? data.description : '');
          setSegmentGroups(Array.isArray(data.segmentGroups) ? data.segmentGroups : []);
          // An imported file is a fresh, unsaved graph — not bound to any
          // saved file yet, so the next save always runs the overwrite check.
          setCurrentGraphFile(null);
          if (importedPresets.length > 0) {
            useNodeDefStore.setState({ presets: mergedPresets });
          }
        } catch (err) {
          addToast(t('toolbar.import.fail', { error: (err as Error).message }), 'error');
        }
      };
      reader.readAsText(file);
      event.target.value = '';
    },
    [setNodes, setEdges, setDescription, setSegmentGroups, setCurrentGraphFile, t, addToast],
  );

  const handleExportJson = useCallback(() => {
    const { nodes, edges, presets, segmentGroups } = getSerializedGraph();
    if (nodes.length === 0) {
      addToast(t('toolbar.exportJson.empty'), 'warning');
      return;
    }
    const name = activeTab.name || 'graph';
    const data = { name, description: activeTab.description ?? '', nodes, edges, presets, segmentGroups };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${name.replace(/[^a-zA-Z0-9_-]/g, '_')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [getSerializedGraph, activeTab.name, activeTab.description, t, addToast]);

  const handleExportSubgraph = useCallback(async () => {
    const { nodes, edges } = getSerializedGraph();
    if (nodes.length === 0) {
      addToast(t('toolbar.export.empty'), 'warning');
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
    const { nodes, edges } = getSerializedGraph();
    if (nodes.length === 0) {
      addToast(t('toolbar.exportPython.empty'), 'warning');
      return;
    }
    const name = activeTab.name || 'graph';
    try {
      const result = await exportGraph(nodes, edges, name);
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
  }, [getSerializedGraph, activeTab.name, t, addToast]);

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

  const fileMenuItems: MenuItem[] = [
    { label: t('toolbar.save'), title: t('toolbar.save.title'), onClick: handleSave },
    { label: t('toolbar.saveAs'), title: t('toolbar.saveAs.title'), onClick: handleSaveAs },
    { label: t('toolbar.clear'), title: t('toolbar.clear.title'), onClick: handleClear },
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
  const statusDotColors: Record<string, string> = {
    idle: '#475569',
    running: '#06b6d4',
    completed: '#22c55e',
    error: '#ef4444',
    cached: '#06b6d4',
    skipped: '#64748b',
  };
  const statusDotColor = statusDotColors[status] ?? '#475569';
  const statusGlow = status === 'running'
    ? '0 0 0.375rem rgba(6, 182, 212, 0.6)'
    : 'none';

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
        <button type="button"
          onClick={handleRun}
          disabled={isRunning}
          title={t('toolbar.run.title')}
          className={styles.runButton}
        >
          {isRunning ? t('toolbar.running') : t('toolbar.run')}
        </button>
        <button type="button"
          onClick={handleStop}
          disabled={!isRunning}
          title={t('toolbar.stop.title')}
          className={styles.stopButton}
        >
          {t('toolbar.stop')}
        </button>
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
          onClick={handleReloadNodes}
          title={t('toolbar.reloadNodes.title')}
          className={`${styles.ghost} ${styles.ghostMuted}`}
        >
          {t('toolbar.reloadNodes')}
        </button>
        <button type="button"
          onClick={() => setCustomNodeManagerOpen(true)}
          title={t('toolbar.customNodes.title')}
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
          <span style={{ color: status === 'running' ? '#06b6d4' : undefined }}>
            {t(statusKey)}
          </span>
        </div>
      </div>

      {/* RIGHT cluster: Settings, Help, FontSize, Language */}
      <div className={`${styles.cluster} ${styles.right}`}>
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
