import { create } from 'zustand';

/**
 * In-app replacement for the browser's native ``window.confirm`` and
 * ``window.prompt``. The native dialogs visually clash with the
 * Crafted-dark IDE aesthetic and (in the case of ``confirm``) silently
 * block the entire WebSocket / Chrome-extension interaction surface
 * until dismissed — which has caused multiple "graph won't run"
 * incidents during dev.
 *
 * Only one dialog can be open at a time. Code calls ``confirm({...})``
 * or ``prompt({...})`` (in ../utils/dialog.ts) and awaits the result;
 * the resolver lives on the store so the UI can call it from a button
 * click without prop-drilling.
 */

export type DialogVariant = 'info' | 'danger';

export interface ConfirmRequest {
  kind: 'confirm';
  title: string;
  message?: string;
  confirmText?: string;
  cancelText?: string;
  variant?: DialogVariant;
}

export interface PromptRequest {
  kind: 'prompt';
  title: string;
  message?: string;
  defaultValue?: string;
  placeholder?: string;
  confirmText?: string;
  cancelText?: string;
  /** Optional input validator — return a string error message to block submit, or null when valid. */
  validate?: (value: string) => string | null;
}

export type DialogRequest = ConfirmRequest | PromptRequest;

interface DialogState {
  /** The currently-open request, or null when nothing is showing. */
  active: DialogRequest | null;
  /** Resolver attached when the request was opened — called with the user's choice. */
  resolve: ((value: boolean | string | null) => void) | null;
  open: <T extends boolean | string | null>(
    request: DialogRequest,
    resolve: (value: T) => void,
  ) => void;
  close: (value: boolean | string | null) => void;
}

export const useDialogStore = create<DialogState>((set, get) => ({
  active: null,
  resolve: null,
  open: (request, resolve) => {
    // If a dialog is already open, resolve the previous one as cancelled
    // (false / null) before replacing — keeps callers' promises from
    // hanging forever if the app fires two confirms back-to-back.
    const prior = get().resolve;
    if (prior) prior(request.kind === 'prompt' ? null : false);
    set({ active: request, resolve: resolve as (v: boolean | string | null) => void });
  },
  close: (value) => {
    const r = get().resolve;
    set({ active: null, resolve: null });
    if (r) r(value);
  },
}));
