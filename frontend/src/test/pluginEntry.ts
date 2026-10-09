import type { PluginCatalogEntry } from '../api/rest';

/**
 * One Plugin Center catalog row, as `listPluginCatalog()` answers with it.
 *
 * The defaults describe a built-in plugin that is not installed yet; a test
 * names only the fields it is about. `name` and `source` follow `id` unless
 * overridden. A new required catalog field gets its default here, once.
 */
export function pluginEntry(
  over: Partial<PluginCatalogEntry> & { id: string },
): PluginCatalogEntry {
  return {
    name: over.id,
    description: '',
    kind: 'builtin',
    official: true,
    status: 'available',
    source_kind: null,
    source: over.id,
    repo: null,
    ref: null,
    sha: null,
    url: null,
    homepage: '',
    version: null,
    installed_at: null,
    enabled: false,
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
