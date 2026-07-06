import { describe, it, expect, beforeEach } from 'vitest';
import { useTabStore } from './tabStore';
import { useProjectStore } from './projectStore';

beforeEach(() => {
  useProjectStore.setState({ projectDir: null, projectName: null, loaded: false });
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('t');
  localStorage.clear();
});

describe('tab readOnly (ID8)', () => {
  it('setTabReadOnly toggles the active tab', () => {
    useTabStore.getState().setTabReadOnly(true);
    const st = useTabStore.getState();
    expect(st.tabs.find((t) => t.id === st.activeTabId)!.readOnly).toBe(true);
  });

  it('rehydrate restores readOnly from storage', () => {
    localStorage.setItem('codefyui-tabs::/p', JSON.stringify({
      activeTabId: 'r1',
      tabs: [{ id: 'r1', name: 'ro', nodes: [], edges: [], readOnly: true }],
    }));
    useProjectStore.getState().setProject('/p');
    useTabStore.getState().rehydrateForProject('/p');
    expect(useTabStore.getState().tabs[0].readOnly).toBe(true);
  });
});
