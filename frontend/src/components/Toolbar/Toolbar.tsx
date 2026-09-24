import { useState, useCallback, useRef, useEffect, useLayoutEffect } from 'react';
import { useGraphExecution } from '../../hooks/useGraphExecution';
import { useDeviceOptions, deviceLabel, isDeviceServed } from '../../hooks/useDeviceOptions';
import { useTabStore } from '../../store/tabStore';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useUIStore } from '../../store/uiStore';
import { createPreset, errorDetail, exportGraph } from '../../api/rest';
import { useI18n, type TranslationKey } from '../../i18n';
import { subgraphIdOf } from '../../utils/subgraph';
import { graphToSvg, svgToPngBlob } from '../../utils/exportDiagram';
import { confirm, prompt } from '../../utils/dialog';
import { saveActiveGraph } from '../../utils/saveActiveGraph';
import { exportWorkspace } from '../../utils/exportWorkspace';
import { SaveIcon } from '../shared/Icons';
import { useToastStore } from '../../store/toastStore';
import type { LayoutMode } from '../../utils/autoLayout';
import { ToolbarGlobalActions } from './ToolbarGlobalActions';
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
    // Capture phase, so a press on the canvas counts: React Flow's pane stops
    // mousedown from bubbling up to `document`.
    document.addEventListener('mousedown', handler, { capture: true });
    return () => document.removeEventListener('mousedown', handler, { capture: true });
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

/* ── Export as Subgraph: a name the server will not store (#476) ─── */

type Translate = (key: TranslationKey, vars?: Record<string, string | number>) => string;

/**
 * A control character as `U+0009`.
 *
 * The refusal carries `ord(character)` -- a decimal number, which names
 * nothing to anyone. The codepoint spelling is the one form of an invisible
 * character a user can look up or quote in a bug report.
 */
function codepointLabel(codepoint: number): string {
  return `U+${codepoint.toString(16).toUpperCase().padStart(4, '0')}`;
}

/** A refusal field as text, or null: the body is `Record<string, unknown>`. */
function field(detail: Record<string, unknown>, key: string): string | null {
  const value = detail[key];
  return typeof value === 'string' ? value : null;
}

/**
 * The sentence for one coded name refusal, or null for "cannot say".
 *
 * `routes_presets` answers every unstorable name with `{detail: {code,
 * ...fields}}` and deliberately NO `message`, so this is where the code
 * becomes something a user can act on. Written as a switch rather than a
 * code -> key table because half of these refusals carry the useful half of
 * their answer BESIDE the code -- which character, which reserved name, which
 * file -- and a table has nothing to interpolate.
 *
 * null for two cases, both of which fall through to `toolbar.export.name
 * .unknownRule`: a code this build has never heard of (a rule the server grew
 * later), and a known code whose field is missing (an older or partial
 * server). The second matters as much as the first -- a sentence rendered
 * with an unfilled `{character}` in it is worse than a general one.
 */
function nameRefusalMessage(
  t: Translate,
  code: string,
  detail: Record<string, unknown>,
): string | null {
  switch (code) {
    case 'name_empty':
      return t('toolbar.export.name.empty');
    case 'name_dot_segment':
      return t('toolbar.export.name.dotSegment');
    case 'name_escapes_presets_dir':
      return t('toolbar.export.name.escapesDir');
    case 'name_separator': {
      const character = field(detail, 'character');
      return character === null
        ? null
        : t('toolbar.export.name.separator', { character });
    }
    case 'name_reserved_device': {
      const reserved = field(detail, 'reserved');
      return reserved === null
        ? null
        : t('toolbar.export.name.reservedDevice', { reserved });
    }
    case 'name_reserved_character': {
      const character = field(detail, 'character');
      return character === null
        ? null
        : t('toolbar.export.name.reservedCharacter', { character });
    }
    case 'name_too_long': {
      const limit = detail.limit;
      return typeof limit === 'number'
        ? t('toolbar.export.name.tooLong', { limit })
        : null;
    }
    case 'preset_file_exists': {
      const filename = field(detail, 'filename');
      return filename === null
        ? null
        : t('toolbar.export.name.fileExists', { filename });
    }
    case 'name_control_character': {
      const codepoint = detail.codepoint;
      return typeof codepoint === 'number'
        ? t('toolbar.export.name.controlCharacter', {
            codepoint: codepointLabel(codepoint),
          })
        : null;
    }
    default:
      return null;
  }
}

/**
 * What a failed Export as Subgraph reads as, coded refusal or not.
 *
 * Three kinds of failure arrive here and only one of them is coded. A PROSE
 * refusal (no nodes, a subgraph instance, a duplicate name) and a network
 * error both keep the message they came with -- the server wrote those
 * sentences and rewriting them is not this fix.
 */
function exportFailureText(t: Translate, err: unknown): string {
  const detail = errorDetail(err);
  const code = detail === null ? null : field(detail, 'code');
  if (detail === null || code === null) {
    return err instanceof Error ? err.message : String(err);
  }
  return (
    nameRefusalMessage(t, code, detail) ??
    t('toolbar.export.name.unknownRule', { code })
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
  const storedDevice = activeTab.graphDevice;
  const storedDeviceListed = storedDevice === null || devices.some((d) => d.value === storedDevice);
  const { reload, fetchDefinitions } = useNodeDefStore();
  const { t } = useI18n();
  const followDevice = devices.find((d) => d.value === globalDevice);
  // When the Settings device is one this server does not serve, the option
  // still follows Settings -- that is what the empty value means -- but it
  // names where the run actually lands, because `resolve_device` downgrades
  // such a run to CPU on the way in and the bare "Follow Settings (cuda)"
  // promised the opposite. Not the Settings row's sentence appended: this
  // select is capped at 14rem, and the longer string clipped mid-word at
  // exactly the part that carries the news. The row explains; this reports.
  const followText = isDeviceServed(devices, globalDevice)
    ? t('toolbar.device.follow', {
        device: followDevice ? deviceLabel(followDevice) : globalDevice,
      })
    : t('toolbar.device.followFallback', { device: globalDevice });
  const addToast = useToastStore((s) => s.addToast);

  const [openMenu, setOpenMenu] = useState<string | null>(null);
  const [layoutMenuOpen, setLayoutMenuOpen] = useState(false);
  const layoutTriggerRef = useRef<HTMLDivElement>(null);
  const layoutMenuRef = useRef<HTMLDivElement>(null);
  const [layoutMenuFromRight, setLayoutMenuFromRight] = useState(false);

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

  // Close layout dropdown on outside click or Escape -- the plugin overflow
  // and font size menus close on both. The press is heard in the capture
  // phase, as in MenuDropdown, so a press on the canvas counts too.
  useEffect(() => {
    if (!layoutMenuOpen) return;
    const handler = (e: MouseEvent) => {
      if (layoutTriggerRef.current && !layoutTriggerRef.current.contains(e.target as Node)) {
        setLayoutMenuOpen(false);
      }
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setLayoutMenuOpen(false);
    };
    document.addEventListener('mousedown', handler, { capture: true });
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('mousedown', handler, { capture: true });
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [layoutMenuOpen]);

  // Hang the layout menu from the split button's right edge when its left
  // edge would carry the menu past the window (see .layoutDropdown). Measured
  // before paint, so it is never seen in the wrong place, and again on every
  // resize while it is open.
  useLayoutEffect(() => {
    if (!layoutMenuOpen) return;
    const place = () => {
      const split = layoutTriggerRef.current;
      const menu = layoutMenuRef.current;
      if (!split || !menu) return;
      // The menu's right edge when it hangs from the left edge, whichever
      // edge it hangs from now: its width is the same from both, and its
      // containing block starts inside the split button's border.
      const right =
        split.getBoundingClientRect().left + split.clientLeft + menu.getBoundingClientRect().width;
      setLayoutMenuFromRight(right > window.innerWidth);
    };
    place();
    window.addEventListener('resize', place);
    return () => window.removeEventListener('resize', place);
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
      // #476: a name the server cannot store is refused with a CODE, and
      // `(e as Error).message` on that refusal was the literal text
      // `[object Object]`. The reason the user needs -- which character, which
      // reserved name, which file -- is in the body; `exportFailureText`
      // turns it into the sentence.
      addToast(t('toolbar.export.fail', { error: exportFailureText(t, e) }), 'error');
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
          <option value="">{followText}</option>
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
          onClick={() => useUIStore.getState().openCustomNodeManager()}
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
            aria-expanded={layoutMenuOpen}
          >
            ▾
          </button>
          {/* Buttons in a menu, like the plugin overflow menu (#507). They
              were divs, which only a mouse could use: Tab never reached them. */}
          {layoutMenuOpen && (
            <div
              ref={layoutMenuRef}
              className={`${styles.layoutDropdown} ${layoutMenuFromRight ? styles.layoutDropdownRight : ''}`}
              role="menu"
              aria-label={t('toolbar.layoutMode.aria')}
            >
              <button type="button" role="menuitem"
                className={`${styles.layoutDropdownItem} ${lastLayoutMode === 'experiments' ? styles.layoutDropdownItemActive : ''}`}
                onClick={() => runLayout('experiments')}
              >
                {t('toolbar.autoLayout.experiments')}
              </button>
              <button type="button" role="menuitem"
                className={`${styles.layoutDropdownItem} ${lastLayoutMode === 'all' ? styles.layoutDropdownItemActive : ''}`}
                onClick={() => runLayout('all')}
              >
                {t('toolbar.autoLayout.all')}
              </button>
              {/* `disabled` is the only guard: a disabled button takes no
                  click and no focus, so this cannot run with nothing selected. */}
              <button type="button" role="menuitem"
                className={`${styles.layoutDropdownItem} ${lastLayoutMode === 'selected' ? styles.layoutDropdownItemActive : ''}`}
                disabled={selectedCount === 0}
                onClick={() => runLayout('selected')}
              >
                {t('toolbar.autoLayout.selected', { count: selectedCount })}
              </button>
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

      {/* RIGHT cluster: plugin buttons, Settings, Help, FontSize, Language.
          Its own component since the welcome screen (no active tab) shows
          exactly this group and none of the graph controls above it. */}
      <ToolbarGlobalActions />
    </div>
  );
}
