import { describe, it, expect } from 'vitest';
import { resolveNodeComponentType } from './index';

describe('resolveNodeComponentType', () => {
  it('uses a first-party viz renderer for known bare names (incl. namespaced)', () => {
    expect(resolveNodeComponentType('Tokenizer')).toBe('tokenizerNode');
    // namespaced plugin node whose bare name has a first-party viz still uses it
    expect(resolveNodeComponentType('foundations:Edu-KNN')).toBe('eduKNNNode');
  });

  it('routes namespaced plugin nodes without a viz to the plugin bridge', () => {
    expect(resolveNodeComponentType('my-plugin:Foo')).toBe('pluginNode');
  });

  it('routes bare built-in nodes to the default card', () => {
    expect(resolveNodeComponentType('Linear')).toBe('baseNode');
  });
});
