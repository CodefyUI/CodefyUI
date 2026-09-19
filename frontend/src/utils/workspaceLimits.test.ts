import { readFileSync } from 'node:fs';
import { describe, it, expect } from 'vitest';
import {
  MAX_WORKSPACE_FILE_BYTES,
  MAX_WORKSPACE_GRAPH_BYTES,
  MAX_WORKSPACE_TABS,
} from './workspaceLimits';

describe('workspace limits', () => {
  it('are 8 MiB per graph, 32 tabs and 64 MiB per file', () => {
    expect(MAX_WORKSPACE_GRAPH_BYTES).toBe(8 * 1024 * 1024);
    expect(MAX_WORKSPACE_TABS).toBe(32);
    expect(MAX_WORKSPACE_FILE_BYTES).toBe(64 * 1024 * 1024);
  });

  it('are declared once: the plugin API imports them instead of keeping a copy', () => {
    // Read relative to the vitest cwd (frontend/), as the example-graph tests do.
    const source = readFileSync('src/plugins/api.ts', 'utf8');
    expect(source).not.toMatch(/const MAX_WORKSPACE_(GRAPH_BYTES|TABS)\s*=/);
    expect(source).toContain("from '../utils/workspaceLimits'");
  });
});
