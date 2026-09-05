import { describe, it, expect } from 'vitest';
import { pluginIdOf, pluginNameOf, type PluginIndex } from './provider';
import type { PluginCatalogEntry } from '../api/rest';

function entry(over: Partial<PluginCatalogEntry> & { id: string }): PluginCatalogEntry {
  return {
    name: over.id,
    description: '',
    kind: 'builtin',
    official: true,
    status: 'installed',
    source_kind: null,
    source: over.id,
    repo: null,
    ref: null,
    sha: null,
    url: null,
    homepage: '',
    version: null,
    installed_at: null,
    enabled: true,
    chapters: [],
    lessons: [],
    tags: [],
    nodes: [],
    node_count: 0,
    capabilities: [],
    trusted_modules: [],
    python_deps: {},
    has_frontend: false,
    consent_required: false,
    frontend_entry: null,
    job: null,
    ...over,
  };
}

const edu: PluginIndex = {
  edu: entry({ id: 'edu', name: 'EDU - hands-on teaching nodes' }),
};

describe('pluginIdOf', () => {
  it('strips the plugin: prefix off a plugin provider', () => {
    expect(pluginIdOf('plugin:edu')).toBe('edu');
    // Manifest ids keep their hyphens: the wire value is the registry key.
    expect(pluginIdOf('plugin:official-template')).toBe('official-template');
  });

  it('answers null for everything that is not a plugin', () => {
    expect(pluginIdOf('builtin')).toBeNull();
    expect(pluginIdOf('custom')).toBeNull();
    expect(pluginIdOf(undefined)).toBeNull();
    expect(pluginIdOf(null)).toBeNull();
    expect(pluginIdOf('')).toBeNull();
    // A prefix with nothing after it names no plugin.
    expect(pluginIdOf('plugin:')).toBeNull();
  });
});

describe('pluginNameOf', () => {
  it('answers the catalog name when the catalog knows the plugin', () => {
    expect(pluginNameOf(edu, 'plugin:edu')).toBe('EDU - hands-on teaching nodes');
  });

  it('falls back to the id for a plugin the catalog has not answered for', () => {
    // The three ordinary states with an empty index: before the boot fetch
    // lands, on an unsupported server, and after a network error. "From
    // plugin: edu" is still true, so the line stands rather than disappearing.
    expect(pluginNameOf({}, 'plugin:edu')).toBe('edu');
    // Same for an id this catalog does not list.
    expect(pluginNameOf(edu, 'plugin:ghost')).toBe('ghost');
  });

  it('falls back to the id when the entry carries an empty name', () => {
    expect(pluginNameOf({ edu: entry({ id: 'edu', name: '' }) }, 'plugin:edu')).toBe('edu');
  });

  it('answers null for a non-plugin provider', () => {
    expect(pluginNameOf(edu, 'builtin')).toBeNull();
    expect(pluginNameOf(edu, 'custom')).toBeNull();
    expect(pluginNameOf(edu, undefined)).toBeNull();
    expect(pluginNameOf(edu, null)).toBeNull();
    expect(pluginNameOf(edu, '')).toBeNull();
    expect(pluginNameOf(edu, 'plugin:')).toBeNull();
  });

  it('does not read a name off Object.prototype', () => {
    // The index is built from parsed JSON, so a plain byId[id] answers with an
    // inherited member for ids like `constructor` -- whose `.name` is the
    // string "Object". The id is the honest answer.
    expect(pluginNameOf({}, 'plugin:constructor')).toBe('constructor');
    expect(pluginNameOf({}, 'plugin:toString')).toBe('toString');
  });
});
