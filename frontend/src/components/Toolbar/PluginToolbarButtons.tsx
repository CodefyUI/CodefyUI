/**
 * Host chrome for plugin toolbar buttons (`api.ui.addToolbarButton`, #132).
 *
 * ── Why the host decides how many are visible ────────────────────────────
 *
 * The toolbar is a fixed row of real controls that already wraps under
 * pressure (`flex-wrap` on `.root`). Letting every installed plugin add an
 * unconditional button to it means the first user with five plugins gets a
 * two-row toolbar, and the first user on a laptop gets Run pushed off the
 * line it shares with the project badge. So plugins get a group, and the
 * group decides what fits.
 *
 * The rule is one measurement — the window width — mapped to an inline
 * capacity, with everything beyond it in a menu behind a single "..." button.
 * A width watcher rather than a media query because the overflow has to move
 * DOM (a button becomes a menu item), which CSS cannot do; a width watcher
 * rather than a per-button ResizeObserver because a capacity that depends on
 * how wide each plugin's glyph happens to render is a capacity that changes
 * while you look at it.
 *
 * At the narrowest sizes the capacity is zero: every plugin button lives in
 * the menu, and the toolbar keeps its own controls on one line.
 */
import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { useI18n } from '../../i18n';
import {
  getPluginToolbarButtons, invokePluginToolbarButton, subscribePluginToolbarButtons,
  type PluginToolbarButton,
} from '../../plugins/toolbarButtons';
import styles from './Toolbar.module.css';

/**
 * How many plugin buttons may sit inline at a given window width.
 *
 * Exported so the thresholds are testable as a table rather than through
 * fifteen render assertions.
 */
export function inlineToolbarCapacity(width: number): number {
  if (width < 900) return 0;
  if (width < 1200) return 1;
  if (width < 1500) return 2;
  return 3;
}

function useWindowWidth(): number {
  const [width, setWidth] = useState(() =>
    typeof window === 'undefined' ? 1600 : window.innerWidth,
  );
  useEffect(() => {
    const onResize = () => setWidth(window.innerWidth);
    window.addEventListener('resize', onResize);
    // The width can have changed between the initial state and this effect
    // (a mount during a resize, or a test that sets innerWidth first).
    onResize();
    return () => window.removeEventListener('resize', onResize);
  }, []);
  return width;
}

function PluginButton({ button }: { button: PluginToolbarButton }) {
  return (
    <button
      type="button"
      onClick={() => invokePluginToolbarButton(button)}
      title={button.tooltip}
      aria-label={button.tooltip}
      className={styles.iconBtn}
      data-testid={`plugin-toolbar-button-${button.key}`}
    >
      {button.icon}
    </button>
  );
}

export function PluginToolbarButtons() {
  const buttons = useSyncExternalStore(
    subscribePluginToolbarButtons, getPluginToolbarButtons,
  );
  const width = useWindowWidth();
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  const close = useCallback(() => setOpen(false), []);

  // Close on outside click or Escape — the same affordances the toolbar's own
  // menus give, so a plugin menu does not feel like a different application.
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: MouseEvent) => {
      if (!wrapperRef.current?.contains(e.target as Node)) close();
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close();
    };
    document.addEventListener('mousedown', onPointerDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('mousedown', onPointerDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [open, close]);

  if (buttons.length === 0) return null;

  const capacity = inlineToolbarCapacity(width);
  // With one button too many, the overflow button costs as much room as the
  // button it would have hidden — so the last inline slot goes to the menu.
  const inlineCount = buttons.length > capacity ? Math.max(capacity - 1, 0) : capacity;
  const inline = buttons.slice(0, inlineCount);
  const overflow = buttons.slice(inlineCount);

  return (
    <div className={styles.cluster} data-testid="plugin-toolbar-buttons">
      {inline.map((button) => <PluginButton key={button.key} button={button} />)}
      {overflow.length > 0 && (
        <div className={styles.menuWrapper} ref={wrapperRef}>
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            title={t('plugins.moreActions')}
            aria-label={t('plugins.moreActions')}
            aria-expanded={open}
            className={`${styles.iconBtn} ${open ? styles.active : ''}`}
            data-testid="plugin-toolbar-overflow"
          >
            ...
          </button>
          {open && (
            <div className={`${styles.menuPanel} ${styles.menuPanelRight}`} role="menu">
              {overflow.map((button) => (
                <button
                  key={button.key}
                  type="button"
                  role="menuitem"
                  className={styles.menuItem}
                  onClick={() => {
                    close();
                    invokePluginToolbarButton(button);
                  }}
                  data-testid={`plugin-toolbar-button-${button.key}`}
                >
                  <span aria-hidden="true">{button.icon}</span>
                  {' '}
                  {button.tooltip}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
