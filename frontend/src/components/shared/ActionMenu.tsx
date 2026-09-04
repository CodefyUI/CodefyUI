import {
  useCallback,
  useEffect,
  useId,
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
  /**
   * Not activatable — but still reachable. The row is `aria-disabled` rather
   * than natively `disabled`, because a natively disabled button takes no
   * focus at all and a row nobody can reach is a row whose {@link hint} nobody
   * can read. Enter, Space and a click are refused in the handler instead.
   */
  disabled?: boolean;
  /**
   * Why the row is refused, shown and announced only while `disabled` is true.
   *
   * The accessible DESCRIPTION rather than part of the name, so the row is
   * still named by what it does ("Fetch") and the reason follows it ("No
   * remote yet."). An enabled row's hint is not rendered: a row that can be
   * pressed explains itself by being pressable.
   */
  hint?: string;
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
 * Distance between the trigger and the panel, above or below it. Geometry the
 * placement code owns end to end — it has to know the gap to flip across the
 * trigger — so it lives here rather than as a `margin` the module could not
 * see. Mirrors `--sp-2`.
 */
const TRIGGER_GAP_PX = 4;

/** How close to a window edge the panel is allowed to sit. */
const VIEWPORT_MARGIN_PX = 8;

/**
 * A dropdown menu attached to one button, with no third-party dependency.
 *
 * Keyboard: Down/Up walk the rows and wrap; Home and End jump to the first and
 * last row; Escape closes and puts focus back on the trigger, as does Tab and
 * choosing a command. Down or Up on the trigger opens the menu onto its first
 * or last row. Only one row is ever in the tab order (a roving tabindex), so
 * the menu is one stop, not N.
 *
 * A refused row is walked onto like any other and refuses the press, because
 * that row is where its `hint` is — "Fetch, No remote yet." is the whole
 * answer, and a row the arrows skip could never give it.
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
  // Index into `items` of the row that owns the tab stop, or -1 when there is
  // no row at all (then the panel itself takes focus, so Escape still works).
  const [activeIndex, setActiveIndex] = useState(-1);
  const [position, setPosition] = useState<CSSProperties>({ top: 0 });
  // One stem for the hint elements; each row appends its own index, so the
  // ids are stable across renders and unique across menus on the page.
  const hintStem = useId();

  const lastIndex = items.length - 1;
  const hasCheckbox = items.some((item) => item.checked !== undefined);

  // `items` belongs to the caller and can change while the menu is open: a row
  // can be dropped the moment a list refreshes. An index left pointing past the
  // end would focus nothing at all — and a menu with no focus inside it
  // swallows Escape, which is the one key that must always work. So the row
  // that actually holds focus and the tab stop is derived every render, and
  // only falls back when it has to. A row that merely turns REFUSED keeps
  // focus: the reason it was refused has just appeared on it.
  const focusIndex = activeIndex >= 0 && activeIndex <= lastIndex
    ? activeIndex
    : (items.length > 0 ? 0 : -1);

  const place = useCallback(() => {
    const rect = triggerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const height = menuRef.current?.offsetHeight ?? 0;
    const roomBelow = window.innerHeight - rect.bottom - VIEWPORT_MARGIN_PX;
    const roomAbove = rect.top - VIEWPORT_MARGIN_PX;
    // A trigger near the bottom of the window would hang its panel off the
    // edge, where it cannot be read or clicked. Flip above it — but only when
    // there is genuinely more room there, since flipping into an even smaller
    // gap trades one unreachable panel for another. When neither side fits,
    // the panel is pushed up as far as the window allows and its own
    // max-height scrolls the rest.
    const flip = height > roomBelow && roomAbove > roomBelow;
    const top = flip
      ? Math.max(VIEWPORT_MARGIN_PX, rect.top - TRIGGER_GAP_PX - height)
      : Math.min(
        rect.bottom + TRIGGER_GAP_PX,
        Math.max(VIEWPORT_MARGIN_PX, window.innerHeight - VIEWPORT_MARGIN_PX - height),
      );
    // `right` rather than `left` for end-alignment: it needs no measurement of
    // the panel, so there is never a frame where the menu is in the wrong spot.
    setPosition(
      align === 'end'
        ? { top, right: Math.max(0, window.innerWidth - rect.right) }
        : { top, left: Math.max(0, rect.left) },
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
    const target = focusIndex >= 0 ? itemRefs.current[focusIndex] : menuRef.current;
    target?.focus();
  }, [open, focusIndex]);

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
    setActiveIndex(items.length === 0 ? -1 : (edge === 'first' ? 0 : lastIndex));
    setOpen(true);
  };

  const step = (delta: number) => {
    if (items.length === 0) return;
    setActiveIndex((focusIndex + delta + items.length) % items.length);
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
        if (items.length > 0) setActiveIndex(0);
        break;
      case 'End':
        e.preventDefault();
        if (items.length > 0) setActiveIndex(lastIndex);
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
            {items.map((item, index) => {
              // Only a refused row explains itself; an enabled one is explained
              // by being pressable.
              const hint = item.disabled === true && item.hint !== undefined
                && item.hint !== ''
                ? item.hint
                : null;
              const hintId = `${hintStem}-hint-${index}`;
              return (
                <button
                  key={item.id}
                  type="button"
                  ref={(el) => {
                    itemRefs.current[index] = el;
                  }}
                  className={styles.item}
                  role={item.checked === undefined ? 'menuitem' : 'menuitemcheckbox'}
                  aria-checked={item.checked}
                  aria-disabled={item.disabled === true}
                  // The hint is INSIDE the row, so without this it would be
                  // read as part of the row's name ("Fetch No remote yet.")
                  // rather than as the description that follows it. The label
                  // is the visible text either way, so the name and what is on
                  // screen still agree.
                  aria-label={hint === null ? undefined : item.label}
                  aria-describedby={hint === null ? undefined : hintId}
                  tabIndex={index === focusIndex ? 0 : -1}
                  // A row reached by pointer or by the browser's own focus takes
                  // the roving tab stop with it, so the next arrow key steps from
                  // where the user actually is rather than from where the keyboard
                  // last was. Harmless when the focus effect is what moved it: the
                  // value it writes is the one already in state.
                  onFocus={() => setActiveIndex(index)}
                  onClick={() => activate(item)}
                >
                  {hasCheckbox && (
                    <span className={styles.check} aria-hidden="true">
                      {item.checked === true ? '✓' : ''}
                    </span>
                  )}
                  <span className={styles.text}>
                    <span className={styles.label}>{item.label}</span>
                    {hint !== null && (
                      <span className={styles.hint} id={hintId}>{hint}</span>
                    )}
                  </span>
                </button>
              );
            })}
          </div>,
          document.body,
        )}
    </>
  );
}
