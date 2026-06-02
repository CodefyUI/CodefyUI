import { useState, useEffect, useRef, useCallback } from 'react';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { CATEGORY_COLORS } from '../../styles/theme';
import type { NodeDefinition, PresetDefinition } from '../../types';
import styles from './QuickNodeSearch.module.css';

interface QuickNodeSearchProps {
  screenPos: { x: number; y: number };
  flowPos: { x: number; y: number };
  onClose: () => void;
}

type SearchResult =
  | { kind: 'node'; def: NodeDefinition }
  | { kind: 'preset'; preset: PresetDefinition };

export function QuickNodeSearch({ screenPos, flowPos, onClose }: QuickNodeSearchProps) {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const definitions = useNodeDefStore((s) => s.definitions);
  const presets = useNodeDefStore((s) => s.presets);
  const addNode = useTabStore((s) => s.addNode);
  const addPresetNode = useTabStore((s) => s.addPresetNode);
  const { t, tn } = useI18n();

  // Filter results
  const results: SearchResult[] = (() => {
    const q = query.toLowerCase().trim();
    const items: SearchResult[] = [];

    for (const def of definitions) {
      if (!q || def.node_name.toLowerCase().includes(q) || def.description.toLowerCase().includes(q)) {
        items.push({ kind: 'node', def });
      }
    }
    for (const preset of presets) {
      if (!q || preset.preset_name.toLowerCase().includes(q) || preset.description.toLowerCase().includes(q)) {
        items.push({ kind: 'preset', preset });
      }
    }

    // Boost: Start node ranks first when query is empty or matches "start"
    if (!q || 'start'.includes(q)) {
      items.sort((a, b) => {
        const aIsStart = a.kind === 'node' && a.def.node_name === 'Start';
        const bIsStart = b.kind === 'node' && b.def.node_name === 'Start';
        if (aIsStart && !bIsStart) return -1;
        if (!aIsStart && bIsStart) return 1;
        return 0;
      });
    }

    return items.slice(0, 20);
  })();

  // Auto-focus input
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Reset selected index when query changes
  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  // Scroll selected item into view
  useEffect(() => {
    const list = listRef.current;
    // listRef is always attached to the results container, which renders unconditionally
    /* v8 ignore next -- @preserve */
    if (!list) return;
    const item = list.children[selectedIndex] as HTMLElement | undefined;
    item?.scrollIntoView({ block: 'nearest' });
  }, [selectedIndex]);

  const handleSelect = useCallback(
    (result: SearchResult) => {
      if (result.kind === 'node') {
        addNode(result.def, flowPos);
      } else {
        addPresetNode(result.preset, flowPos);
      }
      // Defer close so store update completes before component unmounts
      queueMicrotask(onClose);
    },
    [addNode, addPresetNode, flowPos, onClose],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex((i) => Math.min(i + 1, results.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex((i) => Math.max(i - 1, 0));
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (results[selectedIndex]) {
          handleSelect(results[selectedIndex]);
        }
      }
    },
    [results, selectedIndex, handleSelect, onClose],
  );

  // Close on outside click
  const panelRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [onClose]);

  // Position: try to keep on screen
  const left = Math.min(screenPos.x, window.innerWidth - 300);
  const top = Math.min(screenPos.y, window.innerHeight - 400);

  return (
    <div
      ref={panelRef}
      className={styles.panel}
      style={{ left, top }}
    >
      <input
        ref={inputRef}
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
        className={styles.input}
        placeholder={t('palette.search')}
        autoComplete="off"
      />
      <div ref={listRef} className={styles.list}>
        {results.length === 0 && (
          <div className={styles.empty}>{t('palette.noMatch')}</div>
        )}
        {results.map((r, i) => {
          const name = r.kind === 'node' ? r.def.node_name : r.preset.preset_name;
          const category = r.kind === 'node' ? r.def.category : r.preset.category;
          const desc = r.kind === 'node'
            ? tn(r.def.node_name, 'description', r.def.description)
            : r.preset.description;
          const color = CATEGORY_COLORS[category] ?? '#607D8B';

          return (
            <button type="button"
              key={`${r.kind}-${name}-${i}`}
              className={`${styles.item} ${i === selectedIndex ? styles.itemSelected : ''}`}
              onClick={() => handleSelect(r)}
              onMouseEnter={() => setSelectedIndex(i)}
            >
              <span className={styles.dot} style={{ background: color }} />
              <div className={styles.itemContent}>
                <span className={styles.itemName}>{name}</span>
                {r.kind === 'preset' && <span className={styles.presetBadge}>{t('preset.badge')}</span>}
                {desc && <span className={styles.itemDesc}>{desc}</span>}
              </div>
              <span className={styles.itemCategory} style={{ color }}>{category}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
