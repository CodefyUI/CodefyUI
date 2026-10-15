import type { KeyboardEvent } from 'react';

const FOCUSABLE =
  'button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex="-1"])';

/**
 * Keep Tab inside *container*: from the last control to the first and,
 * with Shift, from the first (or the container itself, which holds the
 * focus after a click between controls) to the last.
 */
export function trapTab(event: KeyboardEvent<HTMLElement>, container: HTMLElement | null): void {
  if (event.key !== 'Tab' || container === null) return;
  const focusable = Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE));
  if (focusable.length === 0) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  const active = document.activeElement;
  if (event.shiftKey && (active === first || active === container)) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && active === last) {
    event.preventDefault();
    first.focus();
  }
}
