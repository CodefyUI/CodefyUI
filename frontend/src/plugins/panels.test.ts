import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  _clearPluginPanels,
  getPluginPanel,
  getPluginPanels,
  notifyPanelVisibility,
  registerPluginPanel,
  removePluginPanel,
  removePluginPanelsFor,
  subscribePluginPanels,
} from './panels';

beforeEach(() => _clearPluginPanels());
afterEach(() => {
  _clearPluginPanels();
  vi.restoreAllMocks();
});

describe('plugin panel registry', () => {
  it('namespaces the panel key with the plugin id', () => {
    registerPluginPanel('sweeps', { id: 'main', title: 'Sweeps' });
    expect(getPluginPanels().map((p) => p.key)).toEqual(['sweeps:main']);
    expect(getPluginPanel('sweeps:main')?.localId).toBe('main');
  });

  it('lets two plugins use the same local id', () => {
    registerPluginPanel('a', { id: 'main', title: 'A' });
    registerPluginPanel('b', { id: 'main', title: 'B' });
    expect(getPluginPanels().map((p) => p.key)).toEqual(['a:main', 'b:main']);
  });

  it('defaults the dock to bottom', () => {
    registerPluginPanel('p', { id: 'x', title: 'X' });
    expect(getPluginPanel('p:x')?.dock).toBe('bottom');
    registerPluginPanel('p', { id: 'y', title: 'Y', dock: 'right' });
    expect(getPluginPanel('p:y')?.dock).toBe('right');
  });

  it('returns the SAME element when a panel is re-registered, and updates its chrome', () => {
    const first = registerPluginPanel('p', { id: 'x', title: 'Old', icon: '-' });
    first.appendChild(document.createTextNode('plugin content'));
    const second = registerPluginPanel('p', { id: 'x', title: 'New', icon: '+' });
    expect(second).toBe(first);
    expect(second.textContent).toBe('plugin content');
    expect(getPluginPanel('p:x')?.title).toBe('New');
    expect(getPluginPanel('p:x')?.icon).toBe('+');
    expect(getPluginPanels()).toHaveLength(1);
  });

  it('preserves registration order across an update', () => {
    registerPluginPanel('p', { id: 'a', title: 'A' });
    registerPluginPanel('p', { id: 'b', title: 'B' });
    registerPluginPanel('p', { id: 'a', title: 'A2' });
    expect(getPluginPanels().map((p) => p.localId)).toEqual(['a', 'b']);
  });

  it('removePanel detaches the element from the document but keeps its children', () => {
    const el = registerPluginPanel('p', { id: 'x', title: 'X' });
    el.appendChild(document.createElement('span'));
    document.body.appendChild(el);
    expect(el.isConnected).toBe(true);

    removePluginPanel('p', 'x');
    expect(el.isConnected).toBe(false);
    expect(el.childNodes).toHaveLength(1);
    expect(getPluginPanels()).toHaveLength(0);
  });

  it('removePanel for an unknown id is a no-op', () => {
    registerPluginPanel('p', { id: 'x', title: 'X' });
    removePluginPanel('p', 'nope');
    removePluginPanel('other', 'x');
    expect(getPluginPanels()).toHaveLength(1);
  });

  it('removePluginPanelsFor drops only that plugin', () => {
    const kept = registerPluginPanel('keep', { id: 'x', title: 'K' });
    const dropped = registerPluginPanel('drop', { id: 'x', title: 'D' });
    document.body.append(kept, dropped);

    removePluginPanelsFor('drop');
    expect(getPluginPanels().map((p) => p.key)).toEqual(['keep:x']);
    expect(dropped.isConnected).toBe(false);
    expect(kept.isConnected).toBe(true);
  });

  it('notifies subscribers on register and remove, and stops after unsubscribe', () => {
    const seen = vi.fn();
    const off = subscribePluginPanels(seen);
    registerPluginPanel('p', { id: 'x', title: 'X' });
    expect(seen).toHaveBeenCalledTimes(1);
    removePluginPanel('p', 'x');
    expect(seen).toHaveBeenCalledTimes(2);
    off();
    registerPluginPanel('p', { id: 'y', title: 'Y' });
    expect(seen).toHaveBeenCalledTimes(2);
  });

  it('hands out a stable snapshot so useSyncExternalStore does not loop', () => {
    registerPluginPanel('p', { id: 'x', title: 'X' });
    expect(getPluginPanels()).toBe(getPluginPanels());
    registerPluginPanel('p', { id: 'y', title: 'Y' });
    expect(getPluginPanels()).toBe(getPluginPanels());
  });

  it('does not notify when removePluginPanelsFor matches nothing', () => {
    registerPluginPanel('p', { id: 'x', title: 'X' });
    const seen = vi.fn();
    subscribePluginPanels(seen);
    removePluginPanelsFor('somebody-else');
    expect(seen).not.toHaveBeenCalled();
  });
});

describe('panel visibility callbacks', () => {
  it('calls onShow / onHide', () => {
    const onShow = vi.fn();
    const onHide = vi.fn();
    registerPluginPanel('p', { id: 'x', title: 'X', onShow, onHide });
    const panel = getPluginPanel('p:x')!;

    notifyPanelVisibility(panel, true);
    expect(onShow).toHaveBeenCalledTimes(1);
    expect(onHide).not.toHaveBeenCalled();

    notifyPanelVisibility(panel, false);
    expect(onHide).toHaveBeenCalledTimes(1);
  });

  it('contains a throwing callback so the host can keep mounting', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    registerPluginPanel('p', {
      id: 'x', title: 'X',
      onShow: () => { throw new Error('plugin exploded'); },
    });
    const panel = getPluginPanel('p:x')!;
    expect(() => notifyPanelVisibility(panel, true)).not.toThrow();
    expect(warn).toHaveBeenCalled();
  });

  it('is a no-op when the plugin supplied no callback', () => {
    registerPluginPanel('p', { id: 'x', title: 'X' });
    const panel = getPluginPanel('p:x')!;
    expect(() => {
      notifyPanelVisibility(panel, true);
      notifyPanelVisibility(panel, false);
    }).not.toThrow();
  });
});
