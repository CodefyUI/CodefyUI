import {
  useDialogStore,
  type ConfirmRequest,
  type PromptRequest,
} from '../store/dialogStore';

/**
 * Drop-in replacement for ``window.confirm``. Resolves true on confirm,
 * false on cancel/escape/backdrop. Only one dialog can be open at a
 * time — opening a second one cancels the first.
 *
 * @example
 *   const ok = await confirm({
 *     title: 'Clear canvas?',
 *     message: 'All unsaved nodes and edges will be removed.',
 *     confirmText: 'Clear',
 *     variant: 'danger',
 *   });
 *   if (!ok) return;
 *   clearCanvas();
 */
export function confirm(
  request: Omit<ConfirmRequest, 'kind'>,
): Promise<boolean> {
  return new Promise<boolean>((resolve) => {
    useDialogStore.getState().open(
      { kind: 'confirm', ...request },
      (value) => resolve(value === true),
    );
  });
}

/**
 * Drop-in replacement for ``window.prompt``. Resolves with the entered
 * string on confirm, ``null`` on cancel/escape/backdrop.
 *
 * @example
 *   const name = await prompt({
 *     title: 'Save graph as…',
 *     defaultValue: 'untitled-graph',
 *     placeholder: 'name',
 *   });
 *   if (name === null) return;
 */
export function prompt(
  request: Omit<PromptRequest, 'kind'>,
): Promise<string | null> {
  return new Promise<string | null>((resolve) => {
    useDialogStore.getState().open(
      { kind: 'prompt', ...request },
      (value) => resolve(typeof value === 'string' ? value : null),
    );
  });
}
