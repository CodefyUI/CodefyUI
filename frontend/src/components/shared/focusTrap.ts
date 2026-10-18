import type { KeyboardEvent } from 'react';

const FOCUSABLE =
  'button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), summary, [tabindex]:not([tabindex="-1"])';

/**
 * Whether *element* is folded away inside a closed `<details>`: everything
 * in one but its own `<summary>` is not rendered, so Tab never reaches it.
 */
function folded(element: HTMLElement): boolean {
  for (let node = element.parentElement; node !== null; node = node.parentElement) {
    if (node instanceof HTMLDetailsElement && !node.open) {
      const summary = Array.from(node.children).find((child) => child.tagName === 'SUMMARY');
      if (!summary?.contains(element)) return true;
    }
  }
  return false;
}

/**
 * Keep Tab inside *container*: from the last control to the first and,
 * with Shift, from the first (or the container itself, which holds the
 * focus after a click between controls) to the last. A control folded
 * away in a closed `<details>` is not counted; its `<summary>` is.
 */
export function trapTab(event: KeyboardEvent<HTMLElement>, container: HTMLElement | null): void {
  if (event.key !== 'Tab' || container === null) return;
  const focusable = Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE)).filter((el) => !folded(el));
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
