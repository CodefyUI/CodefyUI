import { create } from 'zustand';

export type ToastType = 'success' | 'error' | 'info' | 'warning';

/**
 * One button on a toast, for a failure whose fix is a click away.
 *
 * Deliberately a single action rather than a list: a toast is read in
 * passing, and a second choice on it is a decision the panel it points at
 * should be asking instead.
 */
export interface ToastAction {
  label: string;
  onClick: () => void;
}

export interface Toast {
  id: string;
  message: string;
  type: ToastType;
  /** Absent on almost every toast — see `ToastAction`. */
  action?: ToastAction;
}

interface ToastState {
  toasts: Toast[];
  /**
   * Put one toast on screen.
   *
   * @returns the new toast's id, for `removeToast`. Ignored by almost every
   * caller -- a toast is fire-and-forget -- but a sticky one has no timer to
   * end it, so whoever raised it needs a handle to take it back down or to
   * replace it with the next one rather than stacking a second copy.
   */
  addToast: (
    message: string,
    type?: ToastType,
    /**
     * `sticky` keeps a toast on screen until it is dismissed, whatever its
     * type. Until now the only way to do that was to call something an
     * error, which is wrong for the one the Source Control tab raises: "an
     * open graph changed on disk" is a warning with a Reload button, and a
     * warning that vanishes after four seconds is a warning nobody who
     * looked away will ever see.
     */
    opts?: { action?: ToastAction; sticky?: boolean },
  ) => string;
  removeToast: (id: string) => void;
}

let _nextId = 0;

export const useToastStore = create<ToastState>((set) => ({
  toasts: [],
  addToast: (message, type = 'info', opts) => {
    const id = String(++_nextId);
    // Spread rather than `action: opts?.action`, so a toast without one has
    // no `action` key at all — every existing caller keeps producing exactly
    // the object it produced before.
    set((state) => ({
      toasts: [
        ...state.toasts,
        { id, message, type, ...(opts?.action ? { action: opts.action } : {}) },
      ],
    }));
    // An error still never times out, and now neither does a toast that
    // asked to stay. The flag is read here and nowhere else -- the container
    // renders a close button on every toast already, so `Toast` itself has
    // nothing new to carry.
    if (type !== 'error' && !opts?.sticky) {
      setTimeout(() => {
        set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) }));
      }, 4000);
    }
    return id;
  },
  removeToast: (id) =>
    set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),
}));
