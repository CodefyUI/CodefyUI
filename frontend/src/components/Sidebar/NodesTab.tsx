import { useState, useRef, useCallback, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { usePluginStore } from '../../store/pluginStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import {
  localizedPackTitle,
  nodeMissingPack,
  usePackAvailability,
} from '../../utils/packAvailability';
import { pluginNameOf, type PluginIndex } from '../../utils/provider';
import type { NodeDefinition, PresetDefinition } from '../../types';
import { DIFFICULTY_COLORS } from '../../styles/theme';
import { orderCategories } from './categories';
import { CategoryList, type CategoryGroup } from './CategoryList';
import styles from './NodePalette.module.css';

// A module-scope selector, so every row of a hundred-node library compares the
// SAME function's output frame to frame. Narrow on purpose: an install running
// in the Plugin Center writes `job`, `busy` and the log on every long-poll
// turn, and none of that changes what a node is called.
type PluginStoreState = ReturnType<typeof usePluginStore.getState>;
const selectPluginsById = (state: PluginStoreState): PluginIndex => state.byId;

// ── Operation Node Item ──

interface NodeItemProps {
  definition: NodeDefinition;
}

export function NodeItem({ definition }: NodeItemProps) {
  const [hovered, setHovered] = useState(false);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);
  const itemRef = useRef<HTMLDivElement>(null);
  const tooltipsEnabled = useUIStore((s) => s.tooltipsEnabled);
  const { t, tn } = useI18n();
  // Three narrow slices that change only when a catalog refresh lands, so a
  // library of a hundred rows costs a selector compare each and nothing per
  // frame. The badge is deliberately NOT a gate: the item stays draggable,
  // and the pre-run check (plus the backend error) is the real safety net.
  const { byId, loaded, unsupported } = usePackAvailability();
  const missingPack = nodeMissingPack(definition, byId, loaded, unsupported);
  const packSentence =
    missingPack === null
      ? null
      : t('palette.needsPack.title', {
          pack: localizedPackTitle(t, byId, missingPack.packId),
        });

  // Who registered this node. Subscribed rather than read once, because the
  // catalog lands after boot: the line says `edu` until it arrives and the
  // plugin's own name from then on.
  const pluginsById = usePluginStore(selectPluginsById);
  const pluginName = pluginNameOf(pluginsById, definition.provider);
  const provenance =
    pluginName === null ? null : t('palette.fromPlugin', { plugin: pluginName });

  const desc = tn(definition.node_name, 'description', definition.description);

  const handleMouseEnter = useCallback(() => {
    setHovered(true);
    // mouseEnter fires on the element whose ref is itemRef, so it is always set
    /* v8 ignore start */
    if (itemRef.current) {
      const rect = itemRef.current.getBoundingClientRect();
      setTooltipPos({ x: rect.right + 8, y: rect.top });
    }
    /* v8 ignore stop */
  }, []);

  const handleMouseLeave = useCallback(() => {
    setHovered(false);
    setTooltipPos(null);
  }, []);

  const handleDragStart = (event: React.DragEvent) => {
    event.dataTransfer.setData('application/codefyui-node', definition.node_name);
    event.dataTransfer.effectAllowed = 'move';
  };

  // A pack-backed node with no description still earns a tooltip: the pack
  // sentence is the thing worth reading before the drag. So does a plugin's
  // node, whose one line worth reading may be where it came from.
  const showTooltip =
    tooltipsEnabled && (desc || packSentence || provenance) && hovered && tooltipPos;

  return (
    <div
      ref={itemRef}
      draggable
      onDragStart={handleDragStart}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      className={styles.nodeItem}
      style={{
        background: hovered ? 'var(--surface-hover)' : 'transparent',
        borderColor: hovered ? 'var(--border-base)' : 'transparent',
      }}
    >
      <div className={styles.nodeItemName}>
        {definition.node_name}
      </div>
      {/* A sibling of the name rather than a child of it, so the name keeps
          its own ellipsis; the item's grid puts the two on one row. */}
      {packSentence !== null && (
        <span
          className={styles.nodeItemBadge}
          // The accessible name carries the whole sentence; the visible label
          // is the two-word chip. The native tooltip is dropped whenever the
          // portal tooltip below is going to render the SAME sentence on the
          // same hover — two copies of it, one of them a browser tooltip that
          // arrives a second late, read as two different messages.
          aria-label={packSentence}
          title={tooltipsEnabled ? undefined : packSentence}
        >
          {t('palette.needsPack')}
        </span>
      )}
      {desc && (
        <div className={styles.nodeItemDesc}>
          {desc}
        </div>
      )}
      {showTooltip && createPortal(
        <div
          className={styles.nodeTooltip}
          style={{ left: tooltipPos.x, top: tooltipPos.y }}
        >
          <div className={styles.nodeTooltipTitle}>{definition.node_name}</div>
          {desc && <div className={styles.nodeTooltipDesc}>{desc}</div>}
          {packSentence !== null && (
            <div className={styles.nodeTooltipPack}>{packSentence}</div>
          )}
          {/* Last: what the node is and whether it can run come first. */}
          {provenance !== null && (
            <div className={styles.nodeTooltipProvenance}>{provenance}</div>
          )}
        </div>,
        document.body,
      )}
    </div>
  );
}

// ── Preset Item ──

interface PresetItemProps {
  preset: PresetDefinition;
}

/**
 * One composite preset, dragged onto the canvas as a whole block.
 *
 * It lived in `PresetsTab` until the Graphs panel took that rail slot; it
 * moved here rather than being rewritten, because `useDragAndDrop` reads the
 * `application/codefyui-preset` payload below and a preset dragged from the
 * node list has to behave exactly as it did from the tab.
 */
export function PresetItem({ preset }: PresetItemProps) {
  const [hovered, setHovered] = useState(false);
  const difficulty = preset.tags.find((t) => t in DIFFICULTY_COLORS) ?? 'beginner';
  const difficultyColor = DIFFICULTY_COLORS[difficulty];
  const { t } = useI18n();

  const handleDragStart = (event: React.DragEvent) => {
    event.dataTransfer.setData('application/codefyui-preset', preset.preset_name);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div
      draggable
      onDragStart={handleDragStart}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      title={preset.description}
      className={styles.presetItem}
      // Gold preset-hover tint: a semantic per-item accent (same family as
      // CATEGORY_COLORS/DIFFICULTY_COLORS below), not chrome, so it is left
      // outside the grey/accent token sweep. Close to --status-preset
      // (#e0a92b) but not identical, and there is no wash/alpha variant of
      // it to reach for; NodesTab.test.tsx also pins this exact rgba
      // string. See migration report for the token gap.
      style={{
        background: hovered ? 'rgba(212,160,23,0.08)' : 'transparent',
        borderColor: hovered ? 'rgba(212,160,23,0.3)' : 'transparent',
      }}
    >
      <div className={styles.presetHeader}>
        <div className={styles.presetName}>
          {preset.preset_name}
        </div>
        <span
          className={styles.presetDifficultyBadge}
          style={{
            background: `${difficultyColor}22`,
            color: difficultyColor,
          }}
        >
          {difficulty}
        </span>
      </div>
      <div className={styles.presetDesc}>
        {preset.description}
      </div>
      <div className={styles.presetNodeCount}>
        {t('empty.nodeCount', { count: preset.nodes.length })}
      </div>
    </div>
  );
}

// ── Nodes tab ──

/**
 * What one row of this tab can be. `CategoryList` is generic over its item, so
 * both kinds go through the one list; `preset_name` is the discriminant,
 * because a `NodeDefinition` never carries one.
 */
type PaletteEntry = NodeDefinition | PresetDefinition;

const isPreset = (entry: PaletteEntry): entry is PresetDefinition => 'preset_name' in entry;

/**
 * Category key for the pinned preset group. The server's category vocabulary
 * is plain display names, so nothing it sends can collide with this; the
 * visible label comes from `labelFor`, not from the key.
 */
const PRESET_GROUP = '__presets__';

/**
 * The node library: search, category accordions, drag-to-canvas.
 *
 * Lifted out of the old single-column `NodePalette` in #126 with its behaviour
 * intact; what changed is that the Composite/Basic sub-headers that used to
 * split each category into nodes and presets are gone. Presets are back here
 * as one group pinned after every node category — the rail slot they had is
 * now the Graphs panel — so creating a preset and using one stay on the same
 * screen.
 *
 * A pure consumer of the catalog: this tab mounts only while it is the open
 * one, so it must not be what STARTS the catalog load — that belongs to the
 * always-mounted shell (see `useNodeDefinitionsBootstrap`).
 */
export function NodesTab() {
  const categorized = useNodeDefStore((s) => s.categorized);
  const presets = useNodeDefStore((s) => s.presets);
  const loading = useNodeDefStore((s) => s.loading);
  const error = useNodeDefStore((s) => s.error);
  const refetch = useNodeDefStore((s) => s.fetchDefinitions);
  const beginnerMode = useUIStore((s) => s.beginnerMode);
  const pluginsById = usePluginStore(selectPluginsById);
  const [searchQuery, setSearchQuery] = useState('');
  const { t } = useI18n();

  const groups = useMemo<CategoryGroup<PaletteEntry>[]>(() => {
    const q = searchQuery.trim().toLowerCase();
    const out: CategoryGroup<PaletteEntry>[] = [];
    // orderCategories only ever returns keys it was given, so the lookup below
    // is always a hit.
    for (const category of orderCategories(Object.keys(categorized), beginnerMode)) {
      let items = categorized[category];
      if (q) {
        // The last field is the plugin's DISPLAY name. Its id already
        // matches through the qualified node name (`edu:FilterRows`), so what
        // this adds is the plugin as a reader knows it from the Plugin Center
        // — the only name for it that appears in no field of a definition.
        //
        // `details` is searched as well as the summary: the summary is one
        // line now, and the library a node wraps, its caveats and its formula
        // all moved down there. Shortening the row must not make the node
        // harder to find.
        items = items.filter(
          (n) =>
            n.node_name.toLowerCase().includes(q) ||
            n.description.toLowerCase().includes(q) ||
            (n.details?.toLowerCase().includes(q) ?? false) ||
            (pluginNameOf(pluginsById, n.provider)?.toLowerCase().includes(q) ?? false),
        );
      }
      if (items.length > 0) out.push({ category, items });
    }

    // Every preset in ONE group after the node categories, whatever category
    // the server filed each under: a preset is a different kind of thing from
    // a node, so the tab reads as the library with the presets under it rather
    // than as a dozen mixed sections. Beginner mode still hides a preset whose
    // category it hides in the list above — same helper, so the two cannot
    // disagree about what a beginner sees. The search predicate is the one the
    // Presets tab used, so nothing findable there stops being findable here.
    const shown = new Set(orderCategories(presets.map((p) => p.category), beginnerMode));
    let presetItems = presets.filter((p) => shown.has(p.category));
    if (q) {
      presetItems = presetItems.filter(
        (p) =>
          p.preset_name.toLowerCase().includes(q) ||
          p.description.toLowerCase().includes(q) ||
          p.tags.some((tag) => tag.toLowerCase().includes(q)),
      );
    }
    if (presetItems.length > 0) out.push({ category: PRESET_GROUP, items: presetItems });

    return out;
  }, [categorized, presets, beginnerMode, pluginsById, searchQuery]);

  return (
    <>
      <div className={styles.header}>
        <div className={styles.headerTitle}>
          {t('palette.title')}
        </div>
        <input
          type="text"
          placeholder={t('palette.search')}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className={styles.searchInput}
        />
      </div>

      <div className={styles.panelBody}>
        {loading && (
          <div className={styles.stateMessage}>
            {t('palette.loading')}
          </div>
        )}

        {error && (
          <div className={styles.errorWrapper}>
            <div className={styles.errorText}>
              {t('palette.loadFail', { error })}
            </div>
            <button type="button" onClick={refetch} className={styles.retryButton}>
              {t('palette.retry')}
            </button>
          </div>
        )}

        {!loading && !error && (
          groups.length === 0 ? (
            <div className={styles.stateMessageMuted}>
              {searchQuery ? t('palette.noMatch') : t('palette.empty')}
            </div>
          ) : (
            <CategoryList
              groups={groups}
              itemKey={(entry) => (isPreset(entry) ? entry.preset_name : entry.node_name)}
              renderItem={(entry) =>
                isPreset(entry)
                  ? <PresetItem preset={entry} />
                  : <NodeItem definition={entry} />
              }
              // Only the pinned group is renamed; a node category still shows
              // the key the server sent, as it did before presets moved in.
              labelFor={(category) =>
                category === PRESET_GROUP ? t('palette.presets.category') : category
              }
            />
          )
        )}
      </div>

      <div className={styles.footer}>
        {t('palette.hint')}
      </div>
    </>
  );
}
