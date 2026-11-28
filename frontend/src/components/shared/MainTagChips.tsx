import { useEffect, useRef } from 'react';
import { useI18n } from '../../i18n';
import { mainTagLabel } from '../../utils/exampleSections';
import type { MainTagEntry } from '../../utils/useExampleCatalog';
import { EXAMPLE_CATEGORY_FALLBACK, EXAMPLE_SECTION_COLORS } from '../../styles/theme';
import { useScrollEdges } from './useScrollEdges';
import styles from './MainTagChips.module.css';

/** The width of an edge fade: a chip scrolled into view clears it. */
const FADE_PX = 24;

/**
 * A vertical wheel turn over the strip, as the sideways scroll it can do
 * (phase 5). Returns how far to move, or 0 to leave the event to the page:
 * a sideways gesture (trackpad) scrolls the strip natively, and a strip
 * already at the end in that direction lets the page scroll on.
 */
export function wheelToScroll(
  event: { deltaX: number; deltaY: number; deltaMode: number },
  box: { scrollLeft: number; scrollWidth: number; clientWidth: number },
): number {
  if (Math.abs(event.deltaX) >= Math.abs(event.deltaY) || event.deltaY === 0) return 0;
  // deltaMode 1 is lines, 2 pages; 0 (pixels) is what most mice send.
  const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? box.clientWidth : 1;
  const delta = event.deltaY * unit;
  const max = box.scrollWidth - box.clientWidth;
  if (delta < 0 && box.scrollLeft <= 0) return 0;
  if (delta > 0 && box.scrollLeft >= max - 1) return 0;
  return delta;
}

/** Where to scroll the strip so `chip` is in view clear of the fades, or
 * null when it already is. */
export function scrollToShow(
  chip: { offsetLeft: number; offsetWidth: number },
  box: { scrollLeft: number; clientWidth: number },
): number | null {
  const left = chip.offsetLeft - FADE_PX;
  const right = chip.offsetLeft + chip.offsetWidth + FADE_PX;
  if (left < box.scrollLeft) return Math.max(0, left);
  if (right > box.scrollLeft + box.clientWidth) return right - box.clientWidth;
  return null;
}

/** The value of the chip row: a main tag id, or every main tag at once. */
export const ALL_MAIN_TAGS = 'all';

export interface MainTagChipsProps {
  entries: MainTagEntry[];
  value: string;
  onChange: (value: string) => void;
  /** Prefix for the tab ids, unique per surface. */
  idPrefix: string;
  /** The id of the tabpanel the chips control. */
  panelId: string;
}

/** The id of one chip, so the panel it controls can be labelled by it. */
export function mainTagTabId(idPrefix: string, value: string): string {
  return `${idPrefix}-tab-${value}`;
}

/**
 * All plus one chip per main tag that has examples, in gallery order.
 *
 * A single-select `tablist` with a roving tabindex: Tab enters on the selected
 * chip, the arrow keys (and Home/End) move AND select, as the WAI-ARIA tabs
 * pattern with automatic activation does. Each chip names its category in
 * text; the dot only repeats the accent.
 */
export function MainTagChips({ entries, value, onChange, idPrefix, panelId }: MainTagChipsProps) {
  const { t } = useI18n();
  const refs = useRef(new Map<string, HTMLButtonElement>());
  const rowRef = useRef<HTMLDivElement | null>(null);
  const edges = useScrollEdges(rowRef, 'x');

  // One row that scrolls sideways (phase 5): a mouse wheel scrolls it too.
  // A native listener, because React's onWheel is passive and cannot stop
  // the page from scrolling at the same time.
  useEffect(() => {
    const row = rowRef.current;
    if (!row) return undefined;
    const onWheel = (event: WheelEvent) => {
      const delta = wheelToScroll(event, row);
      if (delta === 0) return;
      event.preventDefault();
      row.scrollLeft += delta;
    };
    row.addEventListener('wheel', onWheel, { passive: false });
    return () => row.removeEventListener('wheel', onWheel);
  }, []);

  // The selected chip is always in view: on mount (a remembered chip far
  // down the row), on a click, and on an arrow key.
  useEffect(() => {
    const row = rowRef.current;
    const chip = refs.current.get(value);
    if (!row || !chip) return;
    const target = scrollToShow(chip, row);
    if (target !== null) row.scrollLeft = target;
  }, [value, entries]);
  const total = entries.reduce((sum, entry) => sum + entry.count, 0);
  const values = [ALL_MAIN_TAGS, ...entries.map((entry) => entry.id)];

  const move = (from: string, key: string) => {
    const index = values.indexOf(from);
    let next = index;
    if (key === 'ArrowRight' || key === 'ArrowDown') next = (index + 1) % values.length;
    else if (key === 'ArrowLeft' || key === 'ArrowUp') next = (index - 1 + values.length) % values.length;
    else if (key === 'Home') next = 0;
    else if (key === 'End') next = values.length - 1;
    else return false;
    const target = values[next];
    onChange(target);
    refs.current.get(target)?.focus();
    return true;
  };

  const chip = (id: string, label: string, count: number, color: string | null) => {
    const selected = id === value;
    return (
      <button
        key={id}
        ref={(el) => {
          if (el) refs.current.set(id, el);
          else refs.current.delete(id);
        }}
        type="button"
        role="tab"
        id={mainTagTabId(idPrefix, id)}
        aria-selected={selected}
        aria-controls={panelId}
        tabIndex={selected ? 0 : -1}
        className={`${styles.chip} ${selected ? styles.chipSelected : ''}`}
        onClick={() => onChange(id)}
        onKeyDown={(event) => {
          if (move(id, event.key)) event.preventDefault();
        }}
      >
        {color !== null && <span className={styles.dot} style={{ background: color }} aria-hidden="true" />}
        <span>{label}</span>
        <span className={styles.count}>{count}</span>
      </button>
    );
  };

  return (
    <div
      ref={rowRef}
      role="tablist"
      aria-label={t('examples.filter.mainTags')}
      className={styles.row}
      data-fade-start={edges.start || undefined}
      data-fade-end={edges.end || undefined}
    >
      {chip(ALL_MAIN_TAGS, t('examples.filter.all'), total, null)}
      {entries.map((entry) =>
        chip(
          entry.id,
          mainTagLabel(entry.id, t),
          entry.count,
          EXAMPLE_SECTION_COLORS[entry.id] ?? EXAMPLE_CATEGORY_FALLBACK,
        ),
      )}
    </div>
  );
}

/**
 * The chip to show first: the one remembered under `storageKey` if it still
 * has examples, else `fallback` (quickstart) if that does, else All.
 * Storage can be unavailable (private mode, a blocked iframe); that is
 * "nothing remembered", never an error.
 */
export function initialMainTag(storageKey: string | null, entries: MainTagEntry[], fallback: string): string {
  const available = new Set(entries.map((entry) => entry.id));
  if (storageKey !== null) {
    let remembered: string | null = null;
    try {
      remembered = window.localStorage.getItem(storageKey);
    } catch {
      remembered = null;
    }
    if (remembered === ALL_MAIN_TAGS || (remembered !== null && available.has(remembered))) return remembered;
  }
  if (fallback === ALL_MAIN_TAGS || available.has(fallback)) return fallback;
  return ALL_MAIN_TAGS;
}

export function rememberMainTag(storageKey: string, value: string): void {
  try {
    window.localStorage.setItem(storageKey, value);
  } catch {
    // Not remembered; the next tab starts on the default instead.
  }
}
