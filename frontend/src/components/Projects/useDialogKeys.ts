import { useEffect, useRef, type KeyboardEvent as ReactKeyboardEvent, type RefObject } from 'react';
import { trapTab } from '../shared/focusTrap';

const FOCUSABLE =
  'button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex="-1"])';

/**
 * Escape and Tab for a modal dialog, heard on the document rather than on
 * the dialog, so they still work when the focus has fallen out of it.
 *
 * It falls out when the control that had it is disabled under it -- the
 * submit button while a request runs -- and lands on `<body>`. A key the
 * dialog listened for itself would then never reach it: Escape did nothing
 * and Tab walked the page behind (review 4A round 1, item 3).
 *
 * A key pressed inside another modal (a confirm opened over this one) is
 * left to that modal.
 */
export function useDialogKeys(
  dialogRef: RefObject<HTMLElement | null>,
  onEscape: () => void,
): void {
  const escape = useRef(onEscape);
  escape.current = onEscape;

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const dialog = dialogRef.current;
      if (dialog === null) return;
      const active = document.activeElement;
      const inside = active !== null && dialog.contains(active);
      if (!inside && active !== null && active !== document.body
        && active.closest('[aria-modal="true"]') !== null) return;
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        escape.current();
        return;
      }
      if (event.key !== 'Tab') return;
      if (!inside) {
        const focusable = Array.from(dialog.querySelectorAll<HTMLElement>(FOCUSABLE));
        const target = event.shiftKey ? focusable[focusable.length - 1] : focusable[0];
        event.preventDefault();
        (target ?? dialog).focus();
        return;
      }
      trapTab(event as unknown as ReactKeyboardEvent<HTMLElement>, dialog);
    };
    document.addEventListener('keydown', onKey, true);
    return () => document.removeEventListener('keydown', onKey, true);
  }, [dialogRef]);
}
