/**
 * "Persist weights between runs" is off by default (2.8.9).
 *
 * With it on, every weight-owning node starts a run from the module the last
 * run trained, so from the second canvas run on the canvas reports
 * cumulative training while Export Python (which always starts fresh) does
 * not. A new tab now starts with it off, and every tab a 2.8.8 browser
 * stored comes back off once: 2.8.8 wrote `weightsPersistent: true` into
 * EVERY record, so the old key cannot tell a choice from the old default.
 * The setting is stored under a new key, `keepWeights`, written only when on.
 */
import { describe, it, expect, beforeEach } from 'vitest';

import {
  useTabStore,
  _buildPersistedTabForTesting,
  _persistedTabsForTesting,
  _tabFromPersistedForTesting,
  type PersistedTab,
} from './tabStore';
import { importWorkspaceFile } from '../utils/importWorkspaceFile';
import type { ParsedWorkspace } from '../utils/workspaceFile';

const store = () => useTabStore.getState();
const tab = () => useTabStore.getState().getActiveTab();

/** The record a 2.8.8 browser stored for a tab nobody touched. */
function record288(over: Record<string, unknown> = {}): PersistedTab {
  return {
    id: 'old',
    name: 'Old tab',
    nodes: [],
    edges: [],
    recordOutputs: true,
    verboseMode: false,
    graphId: 'gid-old',
    weightsPersistent: true,
    backwardMode: false,
    autoBackward: false,
    ...over,
  } as unknown as PersistedTab;
}

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('Tab 1');
  localStorage.clear();
});

describe('a new tab', () => {
  it('starts with weights not kept between runs', () => {
    expect(tab().weightsPersistent).toBe(false);
    const background = store().createTab({ title: 'Background', activate: false });
    expect(store().getTab(background)!.weightsPersistent).toBe(false);
  });
});

describe('restoring a stored tab', () => {
  it('restores a 2.8.8 record, which says weightsPersistent: true, as off', () => {
    const restored = _tabFromPersistedForTesting(record288(), tab());
    expect(restored.weightsPersistent).toBe(false);
  });

  it('restores a record that says keepWeights: true as on', () => {
    const restored = _tabFromPersistedForTesting(
      record288({ weightsPersistent: undefined, keepWeights: true }),
      tab(),
    );
    expect(restored.weightsPersistent).toBe(true);
  });

  it('restores a record with neither key as off', () => {
    const restored = _tabFromPersistedForTesting(
      record288({ weightsPersistent: undefined }),
      tab(),
    );
    expect(restored.weightsPersistent).toBe(false);
  });

  it('reads keepWeights only when it is exactly true', () => {
    const restored = _tabFromPersistedForTesting(
      record288({ keepWeights: 'yes' }),
      tab(),
    );
    expect(restored.weightsPersistent).toBe(false);
  });
});

describe('the stored record', () => {
  it('never writes the old key, and writes keepWeights only while it is on', () => {
    const off = _buildPersistedTabForTesting(tab()) as unknown as Record<string, unknown>;
    expect('weightsPersistent' in off).toBe(false);
    expect('keepWeights' in off).toBe(false);

    store().togglePersistWeights();
    const on = _buildPersistedTabForTesting(tab()) as unknown as Record<string, unknown>;
    expect('weightsPersistent' in on).toBe(false);
    expect(on.keepWeights).toBe(true);
  });

  it('round-trips through the autosave path: on stays on, off stays off', () => {
    const first = tab().id;
    store().togglePersistWeights();
    const second = store().createTab();

    const records = _persistedTabsForTesting(store().tabs);
    // The id comes from the base, as `loadTabs` builds it.
    const restored = records.map((r) => _tabFromPersistedForTesting(r, { ...tab(), id: r.id }));
    expect(restored.find((t) => t.id === first)?.weightsPersistent).toBe(true);
    expect(restored.find((t) => t.id === second)?.weightsPersistent).toBe(false);
  });
});

describe('a .cduiworkspace file', () => {
  it('still restores run.weightsPersistent: true, which the file states per tab', async () => {
    const workspace: ParsedWorkspace = {
      version: 1,
      appVersion: null,
      exportedAt: null,
      active: null,
      preferences: {},
      tabs: [
        {
          title: 'Kept',
          graph: {
            nodes: [{ id: 'n1', type: 'Add', position: { x: 0, y: 0 }, data: { params: {} } }],
            edges: [],
          },
          run: { weightsPersistent: true },
        },
      ],
    };
    const result = await importWorkspaceFile(workspace);
    const entry = result.results[0];
    if (!('tabId' in entry)) throw new Error('entry skipped');
    expect(store().getTab(entry.tabId)!.weightsPersistent).toBe(true);
  });
});
