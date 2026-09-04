import { useToastStore, type ToastAction } from './toastStore';

/**
 * The three text helpers every center's store needs, in one place.
 *
 * `packStore` and `pluginStore` grew the same three private functions --
 * "what does this thrown thing say", "is this JSON value a string", "put a
 * toast up" -- character for character, because both of them read coded
 * refusals off the wire and both of them answer in toasts. A third store
 * would have copied them again.
 *
 * Deliberately NOT a place for everything two stores share: `openCenterAction`
 * looks identical and is not, because each store names its own panel and its
 * own key. Only what is genuinely the same is here.
 */

/** What a thrown value says, whether or not it was an Error. */
export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * A JSON value as a string, or null.
 *
 * The refusal bodies both stores read are `Record<string, unknown>`: every
 * field off one is `unknown`, and this is the narrowing that keeps a number or
 * a nested object from reaching a toast as "[object Object]".
 */
export function str(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

/**
 * Put one toast up.
 *
 * `action` is spread rather than always passed, so a toast without one is
 * byte-for-byte the object every caller was already producing.
 */
export function toast(
  message: string,
  type: 'info' | 'error' | 'success' | 'warning',
  action?: ToastAction,
) {
  useToastStore.getState().addToast(message, type, action ? { action } : undefined);
}
