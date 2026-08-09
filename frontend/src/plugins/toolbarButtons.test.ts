import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  _clearPluginToolbarButtons,
  getPluginToolbarButtons,
  invokePluginToolbarButton,
  registerPluginToolbarButton,
  removePluginToolbarButton,
  removePluginToolbarButtonsFor,
  subscribePluginToolbarButtons,
} from './toolbarButtons';

const OPTS = { icon: '*', tooltip: 'Do the thing', onClick: () => {} };

beforeEach(() => _clearPluginToolbarButtons());
afterEach(() => {
  _clearPluginToolbarButtons();
  vi.restoreAllMocks();
});

describe('plugin toolbar button registry', () => {
  it('namespaces the key with the plugin id and keeps registration order', () => {
    registerPluginToolbarButton('a', { id: 'one', ...OPTS });
    registerPluginToolbarButton('b', { id: 'one', ...OPTS });
    expect(getPluginToolbarButtons().map((b) => b.key)).toEqual(['a:one', 'b:one']);
  });

  it('replaces a re-registered button in place rather than duplicating it', () => {
    registerPluginToolbarButton('a', { id: 'one', ...OPTS });
    registerPluginToolbarButton('a', { id: 'two', ...OPTS });
    registerPluginToolbarButton('a', { id: 'one', ...OPTS, tooltip: 'Updated' });
    const buttons = getPluginToolbarButtons();
    expect(buttons.map((b) => b.localId)).toEqual(['one', 'two']);
    expect(buttons[0].tooltip).toBe('Updated');
  });

  it('returns a remove function from register', () => {
    const remove = registerPluginToolbarButton('a', { id: 'one', ...OPTS });
    expect(getPluginToolbarButtons()).toHaveLength(1);
    remove();
    expect(getPluginToolbarButtons()).toHaveLength(0);
  });

  it('removing twice is harmless', () => {
    const remove = registerPluginToolbarButton('a', { id: 'one', ...OPTS });
    remove();
    expect(() => remove()).not.toThrow();
    expect(getPluginToolbarButtons()).toHaveLength(0);
  });

  it('a disposer from a superseded registration leaves the live button alone', () => {
    const undoFirst = registerPluginToolbarButton('a', {
      id: 'one', ...OPTS, tooltip: 'First',
    });
    registerPluginToolbarButton('a', { id: 'one', ...OPTS, tooltip: 'Second' });

    undoFirst();

    const buttons = getPluginToolbarButtons();
    expect(buttons).toHaveLength(1);
    expect(buttons[0].tooltip).toBe('Second');
    // The by-id remover is unchanged: it still takes down whoever holds the id.
    removePluginToolbarButton('a', 'one');
    expect(getPluginToolbarButtons()).toHaveLength(0);
  });

  it('a superseded disposer notifies nobody, since it changed nothing', () => {
    const undoFirst = registerPluginToolbarButton('a', { id: 'one', ...OPTS });
    registerPluginToolbarButton('a', { id: 'one', ...OPTS });
    const seen = vi.fn();
    const off = subscribePluginToolbarButtons(seen);
    undoFirst();
    off();
    expect(seen).not.toHaveBeenCalled();
  });

  it('removePluginToolbarButtonsFor drops only that plugin', () => {
    registerPluginToolbarButton('keep', { id: 'x', ...OPTS });
    registerPluginToolbarButton('drop', { id: 'x', ...OPTS });
    registerPluginToolbarButton('drop', { id: 'y', ...OPTS });
    removePluginToolbarButtonsFor('drop');
    expect(getPluginToolbarButtons().map((b) => b.key)).toEqual(['keep:x']);
  });

  it('notifies subscribers and stops after unsubscribe', () => {
    const seen = vi.fn();
    const off = subscribePluginToolbarButtons(seen);
    registerPluginToolbarButton('a', { id: 'one', ...OPTS });
    expect(seen).toHaveBeenCalledTimes(1);
    removePluginToolbarButton('a', 'one');
    expect(seen).toHaveBeenCalledTimes(2);
    off();
    registerPluginToolbarButton('a', { id: 'two', ...OPTS });
    expect(seen).toHaveBeenCalledTimes(2);
  });

  it('does not notify when a remove matches nothing', () => {
    const seen = vi.fn();
    subscribePluginToolbarButtons(seen);
    removePluginToolbarButton('a', 'ghost');
    removePluginToolbarButtonsFor('ghost');
    expect(seen).not.toHaveBeenCalled();
  });

  it('hands out a stable snapshot', () => {
    registerPluginToolbarButton('a', { id: 'one', ...OPTS });
    expect(getPluginToolbarButtons()).toBe(getPluginToolbarButtons());
  });
});

describe('invokePluginToolbarButton', () => {
  it('calls the handler', () => {
    const onClick = vi.fn();
    registerPluginToolbarButton('a', { id: 'one', ...OPTS, onClick });
    invokePluginToolbarButton(getPluginToolbarButtons()[0]);
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it('contains a throwing handler', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    registerPluginToolbarButton('a', {
      id: 'one', ...OPTS,
      onClick: () => { throw new Error('plugin exploded'); },
    });
    expect(() => invokePluginToolbarButton(getPluginToolbarButtons()[0])).not.toThrow();
    expect(warn).toHaveBeenCalled();
  });
});
