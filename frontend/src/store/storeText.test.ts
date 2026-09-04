import { describe, it, expect, beforeEach } from 'vitest';
import { errorMessage, str, toast } from './storeText';
import { useToastStore } from './toastStore';

/**
 * The helpers `packStore` and `pluginStore` both call.
 *
 * Each one exists for a value that arrives from outside TypeScript's reach --
 * a thrown non-Error, a field off a JSON refusal body -- so the cases worth
 * pinning are the ones a `catch` or a `detail` actually produces.
 */

beforeEach(() => {
  useToastStore.setState({ toasts: [] });
});

// ── errorMessage ─────────────────────────────────────────────────────────

describe('errorMessage', () => {
  it('reads an Error, however it was subclassed', () => {
    expect(errorMessage(new Error('Failed to fetch'))).toBe('Failed to fetch');
    expect(errorMessage(new TypeError('bad json'))).toBe('bad json');
  });

  it('stringifies what was thrown when it was not an Error', () => {
    // `throw 'boom'` and a rejected promise carrying a plain object both reach
    // a store's catch, and "[object Object]" beats an empty toast.
    expect(errorMessage('boom')).toBe('boom');
    expect(errorMessage(null)).toBe('null');
    expect(errorMessage(undefined)).toBe('undefined');
    expect(errorMessage({ code: 'busy' })).toBe('[object Object]');
  });
});

// ── str ──────────────────────────────────────────────────────────────────

describe('str', () => {
  it('passes a string through, empty one included', () => {
    expect(str('files_locked')).toBe('files_locked');
    // '' is a string the server sent, not a missing key: the caller decides
    // what an empty answer means.
    expect(str('')).toBe('');
  });

  it('answers null for everything that is not a string', () => {
    expect(str(7)).toBeNull();
    expect(str(null)).toBeNull();
    expect(str(undefined)).toBeNull();
    expect(str({ code: 'busy' })).toBeNull();
    expect(str(['busy'])).toBeNull();
  });
});

// ── toast ────────────────────────────────────────────────────────────────

describe('toast', () => {
  it('puts the message and the type on screen', () => {
    toast('Install failed: offline', 'error');

    expect(useToastStore.getState().toasts).toHaveLength(1);
    expect(useToastStore.getState().toasts[0]).toMatchObject({
      message: 'Install failed: offline', type: 'error',
    });
  });

  it('leaves out the action key entirely when there is no action', () => {
    // Spread, not `action: undefined`: a toast without a button must be the
    // same object it was before actions existed, or every equality assertion
    // written against one breaks.
    toast('Plugin UI reloaded.', 'success');

    expect('action' in useToastStore.getState().toasts[0]).toBe(false);
  });

  it('carries an action when one is passed', () => {
    const onClick = () => {};
    toast('A plugin is still installing.', 'info', {
      label: 'Open Plugin Center', onClick,
    });

    expect(useToastStore.getState().toasts[0].action).toEqual({
      label: 'Open Plugin Center', onClick,
    });
  });
});
