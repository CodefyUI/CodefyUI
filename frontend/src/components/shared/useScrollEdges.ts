import { useEffect, useState, type RefObject } from 'react';

export interface ScrollEdges {
  /** There is content before the visible part (left, or above). */
  start: boolean;
  /** There is content after it (right, or below). */
  end: boolean;
}

const NONE: ScrollEdges = { start: false, end: false };

/**
 * Whether a scroll box has more to show on either side along one axis
 * (phase 5): the edge fades of the main-tag chip strip and the "more cards
 * below" hint of the blank tab are drawn only while this says so.
 *
 * Re-read on scroll, when the box or anything inside it changes size, and
 * when its content is replaced (a different chip shows different cards).
 * A 1px tolerance absorbs sub-pixel scroll positions.
 */
export function useScrollEdges(ref: RefObject<HTMLElement | null>, axis: 'x' | 'y'): ScrollEdges {
  const [edges, setEdges] = useState<ScrollEdges>(NONE);

  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const update = () => {
      const max = axis === 'x' ? el.scrollWidth - el.clientWidth : el.scrollHeight - el.clientHeight;
      const pos = axis === 'x' ? el.scrollLeft : el.scrollTop;
      const next = { start: pos > 1, end: pos < max - 1 };
      setEdges((prev) => (prev.start === next.start && prev.end === next.end ? prev : next));
    };
    const resize = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(update);
    const observe = () => {
      if (!resize) return;
      resize.disconnect();
      resize.observe(el);
      for (const child of Array.from(el.children)) resize.observe(child);
    };
    const mutation = typeof MutationObserver === 'undefined'
      ? null
      : new MutationObserver(() => {
        observe();
        update();
      });
    observe();
    mutation?.observe(el, { childList: true, subtree: true });
    el.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
    update();
    return () => {
      resize?.disconnect();
      mutation?.disconnect();
      el.removeEventListener('scroll', update);
      window.removeEventListener('resize', update);
    };
  }, [ref, axis]);

  return edges;
}
