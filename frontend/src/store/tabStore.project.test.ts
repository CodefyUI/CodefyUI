import { describe, it, expect, beforeEach } from 'vitest';
import { useTabStore } from './tabStore';
import { useProjectStore } from './projectStore';

function reset() {
  useProjectStore.setState({ projectDir: null, projectName: null, loaded: false });
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('base');
  localStorage.clear();
}

beforeEach(reset);

describe('per-project tab scoping (ID10)', () => {
  it('rehydrateForProject(null) is a no-op (keeps non-project tabs)', () => {
    const before = useTabStore.getState().tabs.map((t) => t.id);
    useTabStore.getState().rehydrateForProject(null);
    expect(useTabStore.getState().tabs.map((t) => t.id)).toEqual(before);
  });

  it('rehydrateForProject loads the project-scoped key', () => {
    localStorage.setItem('codefyui-tabs::/proj', JSON.stringify({
      activeTabId: 'p1',
      tabs: [{ id: 'p1', name: 'projtab', nodes: [], edges: [] }],
    }));
    useProjectStore.getState().setProject('/proj');
    useTabStore.getState().rehydrateForProject('/proj');
    const tabs = useTabStore.getState().tabs;
    expect(tabs).toHaveLength(1);
    expect(tabs[0].name).toBe('projtab');
  });

  it('opening a fresh project does not resurrect base-key tabs', () => {
    // Base key has tabs; scoped key for /b is empty -> a fresh default tab.
    localStorage.setItem('codefyui-tabs', JSON.stringify({
      activeTabId: 'a1', tabs: [{ id: 'a1', name: 'A-secret', nodes: [], edges: [] }],
    }));
    useProjectStore.getState().setProject('/b');
    useTabStore.getState().rehydrateForProject('/b');
    expect(useTabStore.getState().tabs.some((t) => t.name === 'A-secret')).toBe(false);
  });

  it('stampActiveTabProject records the origin', () => {
    useTabStore.getState().stampActiveTabProject('/proj');
    const tab = useTabStore.getState().tabs.find(
      (t) => t.id === useTabStore.getState().activeTabId)!;
    expect(tab.projectOrigin).toBe('/proj');
  });
});
