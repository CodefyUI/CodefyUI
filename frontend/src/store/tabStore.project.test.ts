import { describe, it, expect, beforeEach } from 'vitest';
import { useTabStore } from './tabStore';
import { useProjectStore } from './projectStore';
import { DATA_TYPE_COLORS } from '../utils';

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

/**
 * A restored workspace carries its edges exactly as they were autosaved,
 * baked stroke and all, so a palette change leaves old wires in the old
 * colour beside port dots painted live from the new one (core#325). The
 * migration itself is unit-tested in `utils/index.test.ts`; these prove it is
 * wired into the one door a persisted graph comes through.
 */
describe('retired wire colours in a restored workspace (core#325)', () => {
  const sourceWith = (dataType: string) => ({
    id: 'a',
    type: 'baseNode',
    position: { x: 0, y: 0 },
    data: {
      label: 'a',
      type: 'Source',
      params: {},
      definition: {
        node_name: 'Source',
        category: 'Utility',
        description: '',
        inputs: [],
        outputs: [{ name: 'out', data_type: dataType, description: '', optional: false }],
        params: [],
      },
    },
  });

  const store = (stroke: string, dataType: string) => {
    localStorage.setItem('codefyui-tabs::/proj', JSON.stringify({
      activeTabId: 'p1',
      tabs: [{
        id: 'p1',
        name: 'saved',
        nodes: [sourceWith(dataType)],
        edges: [{
          id: 'e1', source: 'a', target: 'b',
          sourceHandle: 'out', targetHandle: 'in',
          style: { stroke, strokeWidth: 2 },
        }],
      }],
    }));
    useProjectStore.getState().setProject('/proj');
    useTabStore.getState().rehydrateForProject('/proj');
    return (useTabStore.getState().tabs[0].edges[0].style as { stroke: string }).stroke;
  };

  it('repaints a TRANSFORM wire saved in the retired amber', () => {
    expect(store('#FFC107', 'TRANSFORM')).toBe(DATA_TYPE_COLORS['TRANSFORM']);
  });

  it('leaves the retired amber alone on a wire of another type', () => {
    expect(store('#FFC107', 'MODEL')).toBe('#FFC107');
  });

  it('leaves a stroke this app never shipped alone', () => {
    expect(store('#123456', 'TRANSFORM')).toBe('#123456');
  });
});
