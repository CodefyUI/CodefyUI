import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  registerNodeRenderer, getNodeRenderer, subscribeNodeRenderers,
  _clearNodeRenderers, type PluginNodeRenderer,
} from './nodeRenderers';

const noop: PluginNodeRenderer = { mount: () => {} };

afterEach(() => _clearNodeRenderers());

describe('nodeRenderers registry', () => {
  it('registers and looks up a renderer by node type', () => {
    registerNodeRenderer('p:Foo', noop);
    expect(getNodeRenderer('p:Foo')).toBe(noop);
    expect(getNodeRenderer('p:Bar')).toBeUndefined();
  });

  it('notifies subscribers on register and unregister', () => {
    const cb = vi.fn();
    const unsub = subscribeNodeRenderers(cb);
    const unregister = registerNodeRenderer('p:Foo', noop);
    expect(cb).toHaveBeenCalledTimes(1);
    unregister();
    expect(cb).toHaveBeenCalledTimes(2);
    expect(getNodeRenderer('p:Foo')).toBeUndefined();
    unsub();
  });

  it('re-registering the same type overwrites (no accumulation across reloads)', () => {
    const a: PluginNodeRenderer = { mount: () => {} };
    const b: PluginNodeRenderer = { mount: () => {} };
    registerNodeRenderer('p:Foo', a);
    registerNodeRenderer('p:Foo', b);
    expect(getNodeRenderer('p:Foo')).toBe(b);
  });

  it('a stale unregister does not remove a newer renderer for the same type', () => {
    const a: PluginNodeRenderer = { mount: () => {} };
    const b: PluginNodeRenderer = { mount: () => {} };
    const unregA = registerNodeRenderer('p:Foo', a);
    registerNodeRenderer('p:Foo', b); // overwrites a
    unregA(); // a is no longer current — must be a no-op
    expect(getNodeRenderer('p:Foo')).toBe(b);
  });
});
