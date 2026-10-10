import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useI18n } from '../../i18n';
import styles from './HelpTip.module.css';

interface HelpTipProps {
  /**
   * What the control does, in one or two short lines. A `\n` starts a new
   * line. Kept short on purpose: the reader came for one fact.
   */
  text?: string;
  /** Rich content in place of `text`, e.g. a description with math in it. */
  children?: React.ReactNode;
  /** What the tip is about, for the icon's accessible name ("About Seed"). */
  topic: string;
  className?: string;
}

/** Room between the icon and the card, and between the card and the window edge. */
const GAP = 6;
const EDGE = 8;

/**
 * A small "?" beside a control that explains it on hover or focus.
 *
 * Explanations used to sit under controls as standing prose, so a panel
 * opened onto a wall of sentences, or hid in a `title` that a keyboard never
 * reaches and a disabled control never shows. The icon is its own button, so
 * the tip still works next to a disabled control.
 *
 * - Hover or focus shows it; leaving or blurring hides it; Escape hides it.
 * - A click toggles it, for touch screens, where there is no hover.
 * - The card is portalled to `document.body` and placed below the icon, or
 *   above it when there is no room, so a scrolling or clipped panel cannot
 *   cut it off.
 * - The icon is described by the card (`aria-describedby`) while it shows.
 */
export function HelpTip({ text, children, topic, className }: HelpTipProps) {
  const { t } = useI18n();
  const id = useId();
  const iconRef = useRef<HTMLButtonElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);

  const place = useCallback(() => {
    const icon = iconRef.current?.getBoundingClientRect();
    const card = cardRef.current?.getBoundingClientRect();
    // Both are mounted whenever this runs: it is called only while open.
    /* v8 ignore start */
    if (!icon || !card) return;
    /* v8 ignore stop */
    const below = icon.bottom + GAP;
    const top =
      below + card.height > window.innerHeight - EDGE && icon.top - GAP - card.height >= EDGE
        ? icon.top - GAP - card.height
        : below;
    const centred = icon.left + icon.width / 2 - card.width / 2;
    const left = Math.max(EDGE, Math.min(centred, window.innerWidth - EDGE - card.width));
    setPos({ top, left });
  }, []);

  useLayoutEffect(() => {
    if (open) place();
    else setPos(null);
  }, [open, place, text]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        // The tip, not the dialog or popover around it.
        e.stopPropagation();
        setOpen(false);
      }
    };
    const onScroll = () => setOpen(false);
    window.addEventListener('keydown', onKey, true);
    window.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', onScroll);
    return () => {
      window.removeEventListener('keydown', onKey, true);
      window.removeEventListener('scroll', onScroll, true);
      window.removeEventListener('resize', onScroll);
    };
  }, [open]);

  return (
    <>
      <button
        type="button"
        ref={iconRef}
        className={`${styles.icon} ${className ?? ''}`}
        aria-label={t('help.about', { topic })}
        aria-describedby={open ? id : undefined}
        aria-expanded={open}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={(e) => {
          // A tip inside a clickable row must not also toggle that row.
          e.stopPropagation();
          setOpen((v) => !v);
        }}
      >
        ?
      </button>
      {open &&
        createPortal(
          <div
            ref={cardRef}
            id={id}
            role="tooltip"
            className={styles.card}
            style={pos ? { top: pos.top, left: pos.left } : { visibility: 'hidden', top: 0, left: 0 }}
          >
            {children ??
              (text ?? '').split('\n').map((line, i) => (
                <p key={i} className={styles.line}>
                  {line}
                </p>
              ))}
          </div>,
          document.body,
        )}
    </>
  );
}
