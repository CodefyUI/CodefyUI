import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
} from 'react';
import { createPortal } from 'react-dom';
import styles from './ActionMenu.module.css';

export interface ActionMenuItem {
  /** Stable key; also what a test can hang a query off. */
  id: string;
  label: string;
  /**
   * Present (true or false) makes the row a `menuitemcheckbox` carrying that
   * state. A checkbox row is a SETTING, so activating it leaves the menu open:
   * `aria-checked` is heard flipping where the user is, and two settings can be
   * changed without reopening. A plain `menuitem` is a command and closes.
   */
  checked?: boolean;
  /** Not activatable, and skipped by every arrow key. */
  disabled?: boolean;
  onSelect: () => void;
}

export interface ActionMenuProps {
  /** Accessible name of the trigger and of the menu it opens. */
  label: string;
  items: ActionMenuItem[];
  /** Trigger content — usually one icon. */
  children: ReactNode;
  /**
   * Styling for the trigger button. Given, it REPLACES the default ghost icon
   * button rather than adding to it, so a host never has to win a specificity
   * argument with this module.
   */
  className?: string;
  /**
   * Which edge of the trigger the panel lines up with. `end` is what keeps a
   * menu opened from a button at the right of a narrow panel on screen.
   */
  align?: 'start' | 'end';
  /** A trigger with nothing to offer yet. */
  disabled?: boolean;
}

/**
 * A dropdown menu attached to one button, with no third-party dependency.
 *
 * Keyboard: Down/Up walk the rows and wrap, skipping disabled ones; Home and
 * End jump to the first and last enabled row; Escape closes and puts focus back
 * on the trigger, as does Tab and choosing a command. Down or Up on the trigger
 * opens the menu onto its first or last row. Only one row is ever in the tab
 * order (a roving tabindex), so the menu is one stop, not N.
 *
 * The panel is portaled to `document.body` and positioned `fixed` against the
 * trigger's rect — the same trick the node tooltip uses, and for the same
 * reason: the sidebar panel this menu opens inside is `overflow: hidden` and
 * as narrow as 180px, so an absolutely-positioned panel would be clipped by
 * its own container. Being outside that box also lets a menu row be wider than
 * the panel, which "Commit All (stages every change, including new files)"
 * needs at every panel width.
 */
export function ActionMenu({
  label,
  items,
  children,
  className,
  align = 'start',
  disabled = false,
}: ActionMenuProps) {
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const itemRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const [open, setOpen] = useState(false);
  // Index into `items` of the row that owns the tab stop, or -1 when every row
  // is disabled (then the panel itself takes focus, so Escape still works).
  const [activeIndex, setActiveIndex] = useState(-1);
  const [position, setPosition] = useState<CSSProperties>({ top: 0 });

  const enabled: number[] = [];
  items.forEach((item, index) => {
    if (!item.disabled) enabled.push(index);
  });
  const hasCheckbox = items.some((item) => item.checked !== undefined);

  const place = useCallback(() => {
    const rect = triggerRef.current?.getBoundingClientRect();
    if (!rect) return;
    // `right` rather than `left` for end-alignment: it needs no measurement of
    // the panel, so there is never a frame where the menu is in the wrong spot.
    setPosition(
      align === 'end'
        ? { top: rect.bottom, right: Math.max(0, window.innerWidth - rect.right) }
        : { top: rect.bottom, left: Math.max(0, rect.left) },
    );
  }, [align]);

  useLayoutEffect(() => {
    if (!open) return undefined;
    place();
    const replace = () => place();
    window.addEventListener('resize', replace);
    // Capture: the sidebar's own scroll containers do not bubble scroll events.
    window.addEventListener('scroll', replace, true);
    return () => {
      window.removeEventListener('resize', replace);
      window.removeEventListener('scroll', replace, true);
    };
  }, [open, place]);

  useEffect(() => {
    if (!open) return;
    const target = activeIndex >= 0 ? itemRefs.current[activeIndex] : menuRef.current;
    target?.focus();
  }, [open, activeIndex]);

  const close = useCallback((returnFocus: boolean) => {
    setOpen(false);
    setActiveIndex(-1);
    if (returnFocus) triggerRef.current?.focus();
  }, []);

  useEffect(() => {
    if (!open) return undefined;
    const onPointerDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (menuRef.current?.contains(target)) return;
      // The trigger's own click already toggles; closing here too would make it
      // open and shut again in the same gesture.
      if (triggerRef.current?.contains(target)) return;
      // No focus return: the pointer has already chosen where focus goes.
      close(false);
    };
    document.addEventListener('mousedown', onPointerDown);
    return () => document.removeEventListener('mousedown', onPointerDown);
  }, [open, close]);

  const openMenu = (edge: 'first' | 'last') => {
    if (disabled) return;
    const index = edge === 'first' ? enabled[0] : enabled[enabled.length - 1];
    setActiveIndex(index ?? -1);
    setOpen(true);
  };

  const step = (delta: number) => {
    if (enabled.length === 0) return;
    const at = enabled.indexOf(activeIndex);
    const next =
      at < 0
        ? (delta > 0 ? 0 : enabled.length - 1)
        : (at + delta + enabled.length) % enabled.length;
    setActiveIndex(enabled[next]);
  };

  const onMenuKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        step(1);
        break;
      case 'ArrowUp':
        e.preventDefault();
        step(-1);
        break;
      case 'Home':
        e.preventDefault();
        if (enabled.length > 0) setActiveIndex(enabled[0]);
        break;
      case 'End':
        e.preventDefault();
        if (enabled.length > 0) setActiveIndex(enabled[enabled.length - 1]);
        break;
      case 'Escape':
        e.preventDefault();
        // Nothing outside this menu should also act on the key that dismissed
        // it — a panel or modal behind it must stay open.
        e.stopPropagation();
        close(true);
        break;
      case 'Tab':
        // The rows are the next focusable elements in the document, so letting
        // Tab run would walk the menu instead of leaving it. Hand focus back to
        // the trigger; Tab again continues from there.
        e.preventDefault();
        close(true);
        break;
      default:
        break;
    }
  };

  const onTriggerKeyDown = (e: ReactKeyboardEvent<HTMLButtonElement>) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      openMenu('first');
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      openMenu('last');
    }
  };

  const activate = (item: ActionMenuItem) => {
    if (item.disabled) return;
    item.onSelect();
    if (item.checked === undefined) close(true);
  };

  return (
    <>
      <button
        type="button"
        ref={triggerRef}
        className={className ?? styles.trigger}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={label}
        title={label}
        disabled={disabled}
        onClick={() => (open ? close(true) : openMenu('first'))}
        onKeyDown={onTriggerKeyDown}
      >
        {children}
      </button>
      {open
        && createPortal(
          <div
            ref={menuRef}
            className={styles.menu}
            role="menu"
            aria-label={label}
            tabIndex={-1}
            style={position}
            onKeyDown={onMenuKeyDown}
          >
            {items.map((item, index) => (
              <button
                key={item.id}
                type="button"
                ref={(el) => {
                  itemRefs.current[index] = el;
                }}
                className={styles.item}
                role={item.checked === undefined ? 'menuitem' : 'menuitemcheckbox'}
                aria-checked={item.checked}
                disabled={item.disabled}
                tabIndex={index === activeIndex ? 0 : -1}
                onClick={() => activate(item)}
              >
                {hasCheckbox && (
                  <span className={styles.check} aria-hidden="true">
                    {item.checked === true ? '✓' : ''}
                  </span>
                )}
                <span className={styles.label}>{item.label}</span>
              </button>
            ))}
          </div>,
          document.body,
        )}
    </>
  );
}
