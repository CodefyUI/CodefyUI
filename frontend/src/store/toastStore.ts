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
  addToast: (
    message: string,
    type?: ToastType,
    opts?: { action?: ToastAction },
  ) => void;
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
    if (type !== 'error') {
      setTimeout(() => {
        set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) }));
      }, 4000);
    }
  },
  removeToast: (id) =>
    set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),
}));
