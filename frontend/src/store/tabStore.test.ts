import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { useTabStore } from './tabStore';
import { useToastStore } from './toastStore';
import { useUIStore } from './uiStore';
import type { NodeDefinition, PresetDefinition } from '../types';

// ── Shared helpers ──────────────────────────────────────────────────────────

/** Reset to a single fresh tab named 'test' before each test. */
function resetToSingleTab() {
  // Clear cross-test shared state (clipboard lives at the store root, not per-tab).
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('test');
}

const store = () => useTabStore.getState();
const activeTab = () => useTabStore.getState().getActiveTab();

function makeDef(overrides: Partial<NodeDefinition> = {}): NodeDefinition {
  return {
    node_name: 'Dataset',
    category: 'data',
    description: 'd',
    inputs: [],
    outputs: [],
    params: [
      { name: 'p1', param_type: 'int', default: 5, description: '', options: [], min_value: null, max_value: null },
      { name: 'p2', param_type: 'string', default: 'x', description: '', options: [], min_value: null, max_value: null },
    ],
    ...overrides,
  };
}

function makePreset(overrides: Partial<PresetDefinition> = {}): PresetDefinition {
  return {
    preset_name: 'MyPreset',
    category: 'cat',
    description: 'desc',
    tags: [],
    nodes: [
      { id: 'inner1', type: 'Dataset', params: { a: 1 } },
      { id: 'inner2', type: 'Model', params: { b: 2 } },
    ],
    edges: [],
    exposed_inputs: [
      { name: 'in', internal_node: 'inner1', internal_port: 'p', data_type: 'TENSOR', description: 'i' },
    ],
    exposed_outputs: [
      { name: 'out', internal_node: 'inner2', internal_port: 'q', data_type: 'MODEL', description: 'o' },
    ],
    exposed_params: [],
    ...overrides,
  };
}

// ── applyLayout (existing) ───────────────────────────────────────────────────

describe('applyLayout', () => {
  beforeEach(() => {
    // Reset store to known state
    useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
    useTabStore.getState().addTab('test');
  });

  it('repositions nodes in a component with a Start node for experiments mode', () => {
    const tabId = useTabStore.getState().activeTabId!;
    useTabStore.setState((state) => ({
      tabs: state.tabs.map((t) =>
        t.id === tabId
          ? {
              ...t,
              nodes: [
                {
                  id: 's',
                  type: 'start',
                  position: { x: 0, y: 0 },
                  width: 80,
                  height: 40,
                  data: { id: 's', type: 'Start' },
                },
                {
                  id: 'a',
                  type: 'baseNode',
                  position: { x: 0, y: 0 },
                  width: 200,
                  height: 80,
                  data: { id: 'a', type: 'Dataset' },
                },
                {
                  id: 'b',
                  type: 'baseNode',
                  position: { x: 0, y: 0 },
                  width: 200,
                  height: 80,
                  data: { id: 'b', type: 'DataLoader' },
                },
                {
                  id: 'c',
                  type: 'baseNode',
                  position: { x: 0, y: 0 },
                  width: 200,
                  height: 80,
                  data: { id: 'c', type: 'Model' },
                },
              ] as any,
              edges: [
                { id: 'et', source: 's', target: 'a', data: { type: 'trigger' } },
                { id: 'e1', source: 'a', target: 'b' },
                { id: 'e2', source: 'b', target: 'c' },
              ] as any,
            }
          : t,
      ),
    }));

    useTabStore.getState().applyLayout('experiments');
    const tab = useTabStore.getState().tabs.find((t) => t.id === tabId)!;
    const a = tab.nodes.find((n) => n.id === 'a')!;
    const b = tab.nodes.find((n) => n.id === 'b')!;
    const c = tab.nodes.find((n) => n.id === 'c')!;

    // Dagre LR layout: A → B → C should produce strictly increasing X
    expect(a.position.x).toBeLessThan(b.position.x);
    expect(b.position.x).toBeLessThan(c.position.x);

    // At least one node must have been moved from its original (0, 0) position
    expect(a.position.x !== 0 || b.position.x !== 0 || c.position.x !== 0).toBe(true);

    // Undo snapshot was pushed so Ctrl+Z reverts the layout
    expect(tab.undoStack.length).toBe(1);
  });

  it('publishes a one-shot layout-fit request naming the laid-out nodes and their bound notes', () => {
    const tabId = useTabStore.getState().activeTabId!;
    useTabStore.setState((state) => ({
      tabs: state.tabs.map((t) =>
        t.id === tabId
          ? {
              ...t,
              nodes: [
                {
                  id: 's',
                  type: 'start',
                  position: { x: 0, y: 0 },
                  width: 80,
                  height: 40,
                  data: { id: 's', type: 'Start' },
                },
                {
                  id: 'a',
                  type: 'baseNode',
                  position: { x: 0, y: 0 },
                  width: 200,
                  height: 80,
                  data: { id: 'a', type: 'Dataset' },
                },
                {
                  id: 'note1',
                  type: 'noteNode',
                  position: { x: 0, y: 0 },
                  width: 200,
                  height: 100,
                  data: {
                    id: 'note1',
                    type: 'note',
                    boundToNodeId: 'a',
                    boundOffset: { x: 20, y: -40 },
                  },
                },
              ] as any,
              edges: [{ id: 'et', source: 's', target: 'a', data: { type: 'trigger' } }] as any,
            }
          : t,
      ),
    }));
    useUIStore.setState({ layoutFitRequest: null });

    useTabStore.getState().applyLayout('experiments');

    const req = useUIStore.getState().layoutFitRequest;
    expect(req).not.toBeNull();
    const { bounds } = req!;
    expect(bounds.width).toBeGreaterThan(0);
    expect(bounds.height).toBeGreaterThan(0);
    // The bound note follows its parent during layout at offset (20, -40), so
    // the fitted box must extend above the computational nodes to include it.
    const tab = useTabStore.getState().getActiveTab();
    const parentY = tab.nodes.find((n) => n.id === 'a')!.position.y;
    expect(bounds.y).toBeLessThanOrEqual(parentY - 40);

    // Requests are one-shot: the consumer clears, and a later layout
    // publishes a fresh request object.
    useUIStore.getState().clearLayoutFit();
    expect(useUIStore.getState().layoutFitRequest).toBeNull();
    useTabStore.getState().applyLayout('experiments');
    expect(useUIStore.getState().layoutFitRequest).not.toBeNull();
  });
});

describe('Teaching Inspector actions', () => {
  beforeEach(() => {
    useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
    useTabStore.getState().addTab('test');
  });

  it('recordOutputs defaults to true', () => {
    const tab = useTabStore.getState().getActiveTab();
    expect(tab.recordOutputs).toBe(true);
  });

  it('toggleRecord flips the flag', () => {
    useTabStore.getState().toggleRecord();
    expect(useTabStore.getState().getActiveTab().recordOutputs).toBe(false);
    useTabStore.getState().toggleRecord();
    expect(useTabStore.getState().getActiveTab().recordOutputs).toBe(true);
  });

  it('setLastRunId stores per-tab run id', () => {
    const tabId = useTabStore.getState().activeTabId!;
    useTabStore.getState().setLastRunId(tabId, 'abc-123');
    expect(useTabStore.getState().getActiveTab().lastRunId).toBe('abc-123');
  });

  it('setActiveSegment and addSegmentGroup work independently', () => {
    const seg = { id: 'g1', headNodeId: 'n1', tailNodeId: 'n2' };
    useTabStore.getState().addSegmentGroup(seg);
    useTabStore.getState().setActiveSegment(seg);
    const tab = useTabStore.getState().getActiveTab();
    expect(tab.segmentGroups).toHaveLength(1);
    expect(tab.segmentGroups[0]).toEqual(seg);
    expect(tab.activeSegment).toEqual(seg);
  });

  it('removeSegmentGroup clears active segment when ids match', () => {
    const seg = { id: 'g1', headNodeId: 'n1', tailNodeId: 'n2' };
    useTabStore.getState().addSegmentGroup(seg);
    useTabStore.getState().setActiveSegment(seg);
    useTabStore.getState().removeSegmentGroup('g1');
    const tab = useTabStore.getState().getActiveTab();
    expect(tab.segmentGroups).toHaveLength(0);
    expect(tab.activeSegment).toBeNull();
  });

  it('addSegmentGroup replaces existing group with the same id', () => {
    useTabStore.getState().addSegmentGroup({ id: 'g1', headNodeId: 'a', tailNodeId: 'b' });
    useTabStore.getState().addSegmentGroup({ id: 'g1', headNodeId: 'c', tailNodeId: 'd' });
    const tab = useTabStore.getState().getActiveTab();
    expect(tab.segmentGroups).toHaveLength(1);
    expect(tab.segmentGroups[0].headNodeId).toBe('c');
  });

  it('getSerializedGraph includes segmentGroups', () => {
    useTabStore.getState().addSegmentGroup({ id: 'g1', headNodeId: 'a', tailNodeId: 'b' });
    const serialized = useTabStore.getState().getSerializedGraph();
    expect(serialized.segmentGroups).toHaveLength(1);
    expect(serialized.segmentGroups![0]).toEqual({ id: 'g1', headNodeId: 'a', tailNodeId: 'b' });
  });
});

// ── Tab management ───────────────────────────────────────────────────────────

describe('tab management', () => {
  beforeEach(resetToSingleTab);

  it('addTab with explicit name appends a tab and makes it active', () => {
    const before = store().tabs.length;
    store().addTab('Explicit');
    const tabs = store().tabs;
    expect(tabs.length).toBe(before + 1);
    expect(tabs[tabs.length - 1].name).toBe('Explicit');
    expect(store().activeTabId).toBe(tabs[tabs.length - 1].id);
  });

  it('addTab without a name uses the default "Tab N" naming', () => {
    // single tab present → count = 1 → next default is "Tab 2"
    store().addTab();
    const tabs = store().tabs;
    expect(tabs[tabs.length - 1].name).toBe(`Tab ${tabs.length}`);
  });

  it('removeTab is a no-op when only one tab remains', () => {
    const id = store().activeTabId;
    store().removeTab(id);
    expect(store().tabs.length).toBe(1);
    expect(store().activeTabId).toBe(id);
  });

  it('removeTab removes a non-active tab and keeps the active one', () => {
    const firstId = store().activeTabId;
    store().addTab('Second'); // becomes active
    const secondId = store().activeTabId;
    store().setActiveTab(firstId); // make first active again
    store().removeTab(secondId); // remove the non-active one
    expect(store().tabs.map((t) => t.id)).not.toContain(secondId);
    expect(store().activeTabId).toBe(firstId);
  });

  it('removeTab on the active tab moves activeTabId to a remaining neighbour', () => {
    const firstId = store().activeTabId;
    store().addTab('Second');
    const secondId = store().activeTabId; // active
    // removing the active (last) tab → newActive picks remaining at clamped index
    store().removeTab(secondId);
    expect(store().tabs.length).toBe(1);
    expect(store().activeTabId).toBe(firstId);
  });

  it('removeTab on active first tab among many picks the correct neighbour', () => {
    const firstId = store().activeTabId;
    store().addTab('Second');
    const secondId = store().activeTabId;
    store().addTab('Third');
    store().setActiveTab(firstId); // active = first (index 0)
    store().removeTab(firstId);
    // remaining = [Second, Third]; clamp(min(0, 1)) = index 0 → Second
    expect(store().activeTabId).toBe(secondId);
  });

  it('removeTab disconnects the tab websocket', () => {
    store().addTab('WithWs');
    const id = store().activeTabId;
    const tab = store().getTab(id)!;
    const spy = vi.spyOn(tab.ws, 'disconnect');
    store().removeTab(id);
    expect(spy).toHaveBeenCalled();
  });

  it('setActiveTab updates activeTabId', () => {
    const firstId = store().activeTabId;
    store().addTab('Second');
    store().setActiveTab(firstId);
    expect(store().activeTabId).toBe(firstId);
  });

  it('renameTab changes the tab name', () => {
    const id = store().activeTabId;
    store().renameTab(id, 'Renamed');
    expect(store().getTab(id)!.name).toBe('Renamed');
  });

  it('getTab returns undefined for unknown id', () => {
    expect(store().getTab('does-not-exist')).toBeUndefined();
  });
});

// ── Flow actions: setNodes / setEdges ────────────────────────────────────────

describe('setNodes / setEdges', () => {
  beforeEach(resetToSingleTab);

  it('setNodes replaces the node list of the active tab', () => {
    const nodes = [{ id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {} } }] as any;
    store().setNodes(nodes);
    expect(activeTab().nodes).toEqual(nodes);
  });

  it('setEdges replaces the edge list of the active tab', () => {
    const edges = [{ id: 'e1', source: 'a', target: 'b' }] as any;
    store().setEdges(edges);
    expect(activeTab().edges).toEqual(edges);
  });
});

// ── addNode / addPresetNode ──────────────────────────────────────────────────

describe('addNode', () => {
  beforeEach(resetToSingleTab);

  it('adds a baseNode with default params and pushes an undo snapshot', () => {
    store().addNode(makeDef(), { x: 10, y: 20 });
    const tab = activeTab();
    expect(tab.nodes).toHaveLength(1);
    const n = tab.nodes[0];
    expect(n.type).toBe('baseNode');
    expect(n.data.params).toEqual({ p1: 5, p2: 'x' });
    expect(n.data.executionStatus).toBe('idle');
    expect(n.position).toEqual({ x: 10, y: 20 });
    expect(tab.undoStack.length).toBe(1);
  });

  it('maps Start node_name to the "start" node type', () => {
    store().addNode(makeDef({ node_name: 'Start', params: [] }), { x: 0, y: 0 });
    expect(activeTab().nodes[0].type).toBe('start');
  });

  it('maps a known viz node_name to its custom node type', () => {
    store().addNode(makeDef({ node_name: 'Tokenizer', params: [] }), { x: 0, y: 0 });
    expect(activeTab().nodes[0].type).toBe('tokenizerNode');
  });

  it('falls back to baseNode for unknown node_name', () => {
    store().addNode(makeDef({ node_name: 'TotallyUnknown', params: [] }), { x: 0, y: 0 });
    expect(activeTab().nodes[0].type).toBe('baseNode');
  });
});

describe('addPresetNode', () => {
  beforeEach(resetToSingleTab);

  it('creates a presetNode with internalParams and a synthesised definition', () => {
    store().addPresetNode(makePreset(), { x: 5, y: 5 });
    const n = activeTab().nodes[0];
    expect(n.type).toBe('presetNode');
    expect(n.data.isPreset).toBe(true);
    expect(n.data.type).toBe('preset:MyPreset');
    expect(n.data.internalParams).toEqual({ inner1: { a: 1 }, inner2: { b: 2 } });
    expect(n.data.definition!.inputs).toEqual([
      { name: 'in', data_type: 'TENSOR', description: 'i', optional: false },
    ]);
    expect(n.data.definition!.outputs).toEqual([
      { name: 'out', data_type: 'MODEL', description: 'o', optional: false },
    ]);
    expect(activeTab().undoStack.length).toBe(1);
  });
});

// ── updateNodeParams / updatePresetInternalParam / updateSubgraphLayers ───────

describe('param updates', () => {
  beforeEach(resetToSingleTab);

  it('updateNodeParams merges params and marks the node dirty', () => {
    store().addNode(makeDef(), { x: 0, y: 0 });
    const id = activeTab().nodes[0].id;
    store().updateNodeParams(id, { p1: 99 });
    const n = activeTab().nodes.find((x) => x.id === id)!;
    expect(n.data.params).toEqual({ p1: 99, p2: 'x' });
    expect([...activeTab().dirtyNodeIds]).toContain(id);
  });

  it('updateNodeParams leaves other nodes untouched', () => {
    store().addNode(makeDef(), { x: 0, y: 0 });
    store().addNode(makeDef(), { x: 0, y: 0 });
    const [a, b] = activeTab().nodes;
    store().updateNodeParams(a.id, { p1: 1 });
    const bAfter = activeTab().nodes.find((x) => x.id === b.id)!;
    expect(bAfter.data.params).toEqual({ p1: 5, p2: 'x' });
  });

  it('drops edges hanging off an output port a param edit removed', () => {
    // core#131: shrinking Split's chunk count must not leave an edge on a
    // handle that no longer renders — invisible on the canvas, fatal at the
    // backend validator.
    const splitDef = makeDef({
      node_name: 'Split',
      outputs: [
        { name: 'chunk_0', data_type: 'TENSOR', description: '', optional: false },
        { name: 'chunk_1', data_type: 'TENSOR', description: '', optional: false },
      ],
      params: [
        { name: 'chunks', param_type: 'int', default: 2, description: '', options: [], min_value: 1, max_value: 32 },
      ],
    });
    store().addNode(splitDef, { x: 0, y: 0 });
    store().addNode(makeDef({ node_name: 'Print' }), { x: 200, y: 0 });
    const [split, sink] = activeTab().nodes;
    useTabStore.setState({
      tabs: useTabStore.getState().tabs.map((tab) =>
        tab.id === useTabStore.getState().activeTabId
          ? {
              ...tab,
              edges: [
                { id: 'keep', source: split.id, target: sink.id, sourceHandle: 'chunk_0', targetHandle: 'value' },
                { id: 'drop', source: split.id, target: sink.id, sourceHandle: 'chunk_2', targetHandle: 'other' },
              ],
            }
          : tab,
      ),
    });
    store().updateNodeParams(split.id, { chunks: 3 });
    expect(activeTab().edges.map((e) => e.id)).toEqual(['keep', 'drop']);

    store().updateNodeParams(split.id, { chunks: 2 });
    expect(activeTab().edges.map((e) => e.id)).toEqual(['keep']);
  });

  it('drops edges into an input port a param edit removed, and dirties the consumer', () => {
    const scriptDef = makeDef({
      node_name: 'PythonScript',
      inputs: [{ name: 'in1', data_type: 'TENSOR', description: '', optional: true }],
      outputs: [{ name: 'out1', data_type: 'ANY', description: '', optional: false }],
      params: [
        { name: 'input_ports', param_type: 'int', default: 1, description: '', options: [], min_value: 1, max_value: 8 },
        { name: 'output_ports', param_type: 'int', default: 1, description: '', options: [], min_value: 1, max_value: 8 },
      ],
    });
    store().addNode(makeDef({ node_name: 'Source' }), { x: 0, y: 0 });
    store().addNode(scriptDef, { x: 200, y: 0 });
    store().addNode(makeDef({ node_name: 'Print' }), { x: 400, y: 0 });
    const [source, script, sink] = activeTab().nodes;
    useTabStore.setState({
      tabs: useTabStore.getState().tabs.map((tab) =>
        tab.id === useTabStore.getState().activeTabId
          ? {
              ...tab,
              edges: [
                { id: 'in1', source: source.id, target: script.id, sourceHandle: 'value', targetHandle: 'in1' },
                { id: 'in2', source: source.id, target: script.id, sourceHandle: 'value', targetHandle: 'in2' },
                { id: 'out2', source: script.id, target: sink.id, sourceHandle: 'out2', targetHandle: 'value' },
              ],
            }
          : tab,
      ),
    });
    store().updateNodeParams(script.id, { input_ports: 2, output_ports: 2 });
    expect(activeTab().edges.map((e) => e.id).sort()).toEqual(['in1', 'in2', 'out2']);

    store().clearDirty();
    store().updateNodeParams(script.id, { input_ports: 1, output_ports: 1 });
    expect(activeTab().edges.map((e) => e.id)).toEqual(['in1']);
    // The consumer that just lost its input is dirty in its own right: with
    // the edge gone, the dirty walk can no longer reach it from the script.
    expect([...activeTab().dirtyNodeIds].sort()).toEqual([script.id, sink.id].sort());
  });

  it('drops a transform-chain edge when ComposeTransform loses a step', () => {
    // The third dynamic-port node, and the reason the comment on
    // `staleEdges` no longer names a closed pair: the resolver walk is
    // generic, and shrinking `steps` has to prune like `chunks` does.
    const composeDef = makeDef({
      node_name: 'ComposeTransform',
      inputs: [
        { name: 'step_1', data_type: 'TRANSFORM', description: '', optional: true },
        { name: 'step_2', data_type: 'TRANSFORM', description: '', optional: true },
      ],
      outputs: [{ name: 'transform', data_type: 'TRANSFORM', description: '', optional: false }],
      params: [
        { name: 'steps', param_type: 'int', default: 2, description: '', options: [], min_value: 2, max_value: 8 },
      ],
    });
    store().addNode(makeDef({ node_name: 'RandomFlip' }), { x: 0, y: 0 });
    store().addNode(composeDef, { x: 200, y: 0 });
    const [source, compose] = activeTab().nodes;
    useTabStore.setState({
      tabs: useTabStore.getState().tabs.map((tab) =>
        tab.id === useTabStore.getState().activeTabId
          ? {
              ...tab,
              edges: [
                { id: 'first', source: source.id, target: compose.id, sourceHandle: 'transform', targetHandle: 'step_1' },
                { id: 'third', source: source.id, target: compose.id, sourceHandle: 'transform', targetHandle: 'step_3' },
              ],
            }
          : tab,
      ),
    });
    store().updateNodeParams(compose.id, { steps: 3 });
    expect(activeTab().edges.map((e) => e.id)).toEqual(['first', 'third']);

    store().updateNodeParams(compose.id, { steps: 2 });
    expect(activeTab().edges.map((e) => e.id)).toEqual(['first']);
  });

  it('never drops a trigger edge when ports change', () => {
    const scriptDef = makeDef({
      node_name: 'PythonScript',
      inputs: [{ name: 'in1', data_type: 'TENSOR', description: '', optional: true }],
      outputs: [{ name: 'out1', data_type: 'ANY', description: '', optional: false }],
      params: [
        { name: 'input_ports', param_type: 'int', default: 1, description: '', options: [], min_value: 1, max_value: 8 },
      ],
    });
    store().addNode(makeDef({ node_name: 'Start' }), { x: 0, y: 0 });
    store().addNode(scriptDef, { x: 200, y: 0 });
    const [start, script] = activeTab().nodes;
    useTabStore.setState({
      tabs: useTabStore.getState().tabs.map((tab) =>
        tab.id === useTabStore.getState().activeTabId
          ? {
              ...tab,
              edges: [
                { id: 't', source: start.id, target: script.id, sourceHandle: 'trigger', targetHandle: '__trigger', type: 'trigger' },
              ],
            }
          : tab,
      ),
    });
    store().updateNodeParams(script.id, { input_ports: 4 });
    store().updateNodeParams(script.id, { input_ports: 1 });
    expect(activeTab().edges.map((e) => e.id)).toEqual(['t']);
  });

  it('makes edge deletion undoable, and only then pushes a snapshot', () => {
    // Dropping a script from 8 ports to 1 destroys up to 7 edges. Without a
    // snapshot Ctrl+Z would skip past their deletion entirely; with one on
    // every keystroke the undo stack would be useless.
    const scriptDef = makeDef({
      node_name: 'PythonScript',
      inputs: [{ name: 'in1', data_type: 'TENSOR', description: '', optional: true }],
      outputs: [{ name: 'out1', data_type: 'ANY', description: '', optional: false }],
      params: [
        { name: 'input_ports', param_type: 'int', default: 1, description: '', options: [], min_value: 1, max_value: 8 },
        { name: 'code', param_type: 'code', default: '', description: '', options: [], min_value: null, max_value: null },
      ],
    });
    store().addNode(makeDef({ node_name: 'Source' }), { x: 0, y: 0 });
    store().addNode(scriptDef, { x: 200, y: 0 });
    const [source, script] = activeTab().nodes;
    useTabStore.setState({
      tabs: useTabStore.getState().tabs.map((tab) =>
        tab.id === useTabStore.getState().activeTabId
          ? {
              ...tab,
              undoStack: [],
              edges: [
                { id: 'in1', source: source.id, target: script.id, sourceHandle: 'value', targetHandle: 'in1' },
                { id: 'in2', source: source.id, target: script.id, sourceHandle: 'value', targetHandle: 'in2' },
              ],
            }
          : tab,
      ),
    });

    // An ordinary param edit that changes no ports must NOT push a snapshot.
    store().updateNodeParams(script.id, { input_ports: 2 });
    store().updateNodeParams(script.id, { code: 'x' });
    expect(activeTab().undoStack.length).toBe(0);
    expect(activeTab().edges).toHaveLength(2);

    store().updateNodeParams(script.id, { input_ports: 1 });
    expect(activeTab().edges.map((e) => e.id)).toEqual(['in1']);
    expect(activeTab().undoStack.length).toBe(1);

    store().undo();
    expect(activeTab().edges.map((e) => e.id).sort()).toEqual(['in1', 'in2']);
  });

  it('leaves edges alone for a node whose ports do not depend on params', () => {
    store().addNode(makeDef({ node_name: 'Add' }), { x: 0, y: 0 });
    store().addNode(makeDef({ node_name: 'Print' }), { x: 200, y: 0 });
    const [a, b] = activeTab().nodes;
    useTabStore.setState({
      tabs: useTabStore.getState().tabs.map((tab) =>
        tab.id === useTabStore.getState().activeTabId
          ? {
              ...tab,
              edges: [
                { id: 'e', source: a.id, target: b.id, sourceHandle: 'whatever', targetHandle: 'value' },
              ],
            }
          : tab,
      ),
    });
    store().updateNodeParams(a.id, { p1: 42 });
    expect(activeTab().edges.map((e) => e.id)).toEqual(['e']);
  });

  it('updatePresetInternalParam updates a nested internal param', () => {
    store().addPresetNode(makePreset(), { x: 0, y: 0 });
    const id = activeTab().nodes[0].id;
    store().updatePresetInternalParam(id, 'inner1', 'a', 42);
    const n = activeTab().nodes[0];
    expect(n.data.internalParams!.inner1.a).toBe(42);
    expect(n.data.internalParams!.inner2).toEqual({ b: 2 });
  });

  it('updatePresetInternalParam ignores non-matching node ids', () => {
    store().addPresetNode(makePreset(), { x: 0, y: 0 });
    const before = activeTab().nodes[0].data.internalParams;
    store().updatePresetInternalParam('nope', 'inner1', 'a', 42);
    expect(activeTab().nodes[0].data.internalParams).toEqual(before);
  });

  it('updatePresetInternalParam handles a node with no prior internalParams', () => {
    // Add a plain node then force-call the preset updater on it.
    store().addNode(makeDef(), { x: 0, y: 0 });
    const id = activeTab().nodes[0].id;
    store().updatePresetInternalParam(id, 'innerX', 'k', 7);
    expect(activeTab().nodes[0].data.internalParams).toEqual({ innerX: { k: 7 } });
  });

  it('updateSubgraphLayers writes the layers param onto the matching node', () => {
    store().addNode(makeDef(), { x: 0, y: 0 });
    const id = activeTab().nodes[0].id;
    store().updateSubgraphLayers(id, '[{"a":1}]');
    expect(activeTab().nodes[0].data.params.layers).toBe('[{"a":1}]');
  });

  it('updateSubgraphLayers ignores non-matching node ids', () => {
    store().addNode(makeDef(), { x: 0, y: 0 });
    store().updateSubgraphLayers('nope', '[]');
    expect(activeTab().nodes[0].data.params.layers).toBeUndefined();
  });
});

// ── selection + modal openers ────────────────────────────────────────────────

describe('selection and modals', () => {
  beforeEach(resetToSingleTab);

  it('setSelectedNodeId sets and clears selection', () => {
    store().setSelectedNodeId('n1');
    expect(activeTab().selectedNodeId).toBe('n1');
    store().setSelectedNodeId(null);
    expect(activeTab().selectedNodeId).toBeNull();
  });

  it('setSelectedNodeId does not touch .selected -- React Flow already has, for every click (#167 follow-up)', () => {
    // The plain-click path (FlowCanvas#handleNodeClick). By the time this
    // runs, React Flow has already applied the click's own selection effect
    // to `.selected` via its own dispatch -- re-deriving it here would fight
    // that (and, for a shift+click removal specifically, get it backwards;
    // see the "shift+click that removes..." test below).
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, selected: false, data: { label: 'A', type: 'A', params: {} } },
      { id: 'n2', type: 'baseNode', position: { x: 0, y: 0 }, selected: true, data: { label: 'B', type: 'B', params: {} } },
    ] as any);
    store().setSelectedNodeId('n1');
    expect(activeTab().selectedNodeId).toBe('n1');
    // .selected is exactly as it was before this call -- untouched.
    expect(activeTab().nodes.find((n) => n.id === 'n1')!.selected).toBe(false);
    expect(activeTab().nodes.find((n) => n.id === 'n2')!.selected).toBe(true);
  });

  it('shift+click that removes a node from a selection does not re-select it (#167 follow-up)', () => {
    // The regression this branch introduced and this test guards against:
    // {a, c} selected, user shift+clicks a to remove it. React Flow's own
    // dispatch (verified against the installed @xyflow/react source)
    // synchronously sets a.selected=false BEFORE onNodeClick -- and
    // therefore setSelectedNodeId -- ever runs. At that point `a` looks
    // identical (from the nodes array alone) to a stale click landing
    // outside the current selection, which is exactly the shape
    // selectNodeExclusively exists to correct for a right-click. Using that
    // syncing action here would re-select `a` and deselect `c` -- the
    // opposite of what the user just did.
    store().setNodes([
      { id: 'a', type: 'baseNode', position: { x: 0, y: 0 }, selected: false, data: { label: 'A', type: 'A', params: {} } },
      { id: 'c', type: 'baseNode', position: { x: 0, y: 0 }, selected: true, data: { label: 'C', type: 'C', params: {} } },
    ] as any);
    store().setSelectedNodeId('a');
    expect(activeTab().nodes.find((n) => n.id === 'a')!.selected).toBe(false);
    expect(activeTab().nodes.find((n) => n.id === 'c')!.selected).toBe(true);
  });

  it('selectNodeExclusively selects the target node in React Flow, deselecting the rest (#167)', () => {
    // Mirrors the context-menu path (FlowCanvas#handleNodeContextMenu): n1
    // carries a stale `.selected` from an earlier plain click, then n2 is
    // right-clicked. `selectedNodeId` and `.selected` must agree afterwards
    // or React Flow's Delete key removes the wrong node.
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, selected: true, data: { label: 'A', type: 'A', params: {} } },
      { id: 'n2', type: 'baseNode', position: { x: 0, y: 0 }, selected: false, data: { label: 'B', type: 'B', params: {} } },
    ] as any);
    store().selectNodeExclusively('n2');
    const nodes = activeTab().nodes;
    expect(nodes.find((n) => n.id === 'n1')!.selected).toBe(false);
    expect(nodes.find((n) => n.id === 'n2')!.selected).toBe(true);
  });

  it('selectNodeExclusively(null) deselects every node', () => {
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, selected: true, data: { label: 'A', type: 'A', params: {} } },
    ] as any);
    store().selectNodeExclusively(null);
    expect(activeTab().nodes[0].selected).toBe(false);
  });

  it('open/close preset modal', () => {
    store().openPresetModal('p1');
    expect(activeTab().presetModalNodeId).toBe('p1');
    store().closePresetModal();
    expect(activeTab().presetModalNodeId).toBeNull();
  });

  it('open/close subgraph modal', () => {
    store().openSubgraphModal('s1');
    expect(activeTab().subgraphModalNodeId).toBe('s1');
    store().closeSubgraphModal();
    expect(activeTab().subgraphModalNodeId).toBeNull();
  });
});

// ── execution status ─────────────────────────────────────────────────────────

describe('execution status', () => {
  beforeEach(resetToSingleTab);

  it('setNodeExecutionStatus sets status and error on the matching node', () => {
    store().addNode(makeDef(), { x: 0, y: 0 });
    const id = activeTab().nodes[0].id;
    store().setNodeExecutionStatus(id, 'error', 'boom');
    const n = activeTab().nodes[0];
    expect(n.data.executionStatus).toBe('error');
    expect(n.data.error).toBe('boom');
  });

  it('setNodeExecutionStatus ignores non-matching ids', () => {
    store().addNode(makeDef(), { x: 0, y: 0 });
    store().setNodeExecutionStatus('nope', 'running');
    expect(activeTab().nodes[0].data.executionStatus).toBe('idle');
  });

  it('clearExecutionStatus resets every node to idle and clears errors', () => {
    store().addNode(makeDef(), { x: 0, y: 0 });
    store().addNode(makeDef(), { x: 0, y: 0 });
    const ids = activeTab().nodes.map((n) => n.id);
    store().setNodeExecutionStatus(ids[0], 'completed');
    store().setNodeExecutionStatus(ids[1], 'error', 'e');
    store().clearExecutionStatus();
    for (const n of activeTab().nodes) {
      expect(n.data.executionStatus).toBe('idle');
      expect(n.data.error).toBeUndefined();
    }
  });
});

// ── clear / getSerializedGraph ───────────────────────────────────────────────

describe('clear', () => {
  beforeEach(resetToSingleTab);

  it('clears nodes, edges, selection and modal state, pushing an undo snapshot', () => {
    store().addNode(makeDef(), { x: 0, y: 0 });
    store().setSelectedNodeId('x');
    store().openPresetModal('p');
    store().openSubgraphModal('s');
    store().clear();
    const tab = activeTab();
    expect(tab.nodes).toHaveLength(0);
    expect(tab.edges).toHaveLength(0);
    expect(tab.selectedNodeId).toBeNull();
    expect(tab.presetModalNodeId).toBeNull();
    expect(tab.subgraphModalNodeId).toBeNull();
    // addNode pushed one snapshot, clear() pushed a second.
    expect(tab.undoStack.length).toBe(2);
  });

  it('resets graph metadata (description, currentGraphFile, segments) so a cleared canvas is unbound', () => {
    store().setDescription('bound graph');
    store().setCurrentGraphFile('bound_file');
    store().addSegmentGroup({ id: 's1', headNodeId: 'h', tailNodeId: 't' });
    store().setActiveSegment({ id: 's1', headNodeId: 'h', tailNodeId: 't' });
    store().clear();
    const tab = activeTab();
    expect(tab.description).toBe('');
    expect(tab.currentGraphFile).toBeNull();
    expect(tab.segmentGroups).toEqual([]);
    expect(tab.activeSegment).toBeNull();
  });
});

describe('getSerializedGraph', () => {
  beforeEach(resetToSingleTab);

  it('serializes plain nodes and edges with default handle strings', () => {
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 1, y: 2 }, data: { label: 'A', type: 'Dataset', params: { p: 1 } } },
    ] as any);
    store().setEdges([
      { id: 'e1', source: 'n1', target: 'n2' },
    ] as any);
    const g = store().getSerializedGraph();
    expect(g.nodes[0]).toEqual({ id: 'n1', type: 'Dataset', position: { x: 1, y: 2 }, data: { params: { p: 1 } } });
    expect(g.edges[0]).toEqual({ id: 'e1', source: 'n1', target: 'n2', sourceHandle: '', targetHandle: '' });
    expect(g.presets).toEqual([]);
  });

  it('serializes preset nodes (including internalParams) and dedupes presets', () => {
    const preset = makePreset();
    store().addPresetNode(preset, { x: 0, y: 0 });
    store().addPresetNode(preset, { x: 0, y: 0 }); // same preset_name → deduped
    const g = store().getSerializedGraph();
    expect(g.presets).toHaveLength(1);
    expect(g.nodes[0].data.internalParams).toBeDefined();
    expect(g.nodes[1].data.internalParams).toBeDefined();
  });

  it('serializes note nodes with note-specific fields', () => {
    store().addNote('text', { x: 3, y: 4 });
    const g = store().getSerializedGraph();
    expect(g.nodes[0].type).toBe('note');
    expect(g.nodes[0].data).toMatchObject({
      noteKind: 'text',
      noteContent: '',
      noteColor: '#3d3d1a',
      boundToNodeId: null,
      boundOffset: null,
      noteWidth: 200,
    });
  });

  it('preserves provided sourceHandle/targetHandle and marks trigger edges (by type)', () => {
    store().setEdges([
      { id: 'e1', source: 'a', target: 'b', sourceHandle: 'out', targetHandle: 'in', type: 'triggerEdge' },
    ] as any);
    const g = store().getSerializedGraph();
    expect(g.edges[0]).toEqual({
      id: 'e1', source: 'a', target: 'b', sourceHandle: 'out', targetHandle: 'in', type: 'trigger',
    });
  });

  it('marks trigger edges detected via edge.data.type', () => {
    store().setEdges([
      { id: 'e1', source: 'a', target: 'b', data: { type: 'trigger' } },
    ] as any);
    const g = store().getSerializedGraph();
    expect((g.edges[0] as any).type).toBe('trigger');
  });

  it('does not flag normal edges as triggers', () => {
    store().setEdges([{ id: 'e1', source: 'a', target: 'b' }] as any);
    const g = store().getSerializedGraph();
    expect((g.edges[0] as any).type).toBeUndefined();
  });

  it('does not push a preset for a preset node missing presetDefinition', () => {
    store().setNodes([
      { id: 'n1', type: 'presetNode', position: { x: 0, y: 0 }, data: { label: 'P', type: 'preset:X', params: {}, isPreset: true } },
    ] as any);
    const g = store().getSerializedGraph();
    expect(g.presets).toEqual([]);
  });

  it('rounds node positions to integers (regular + note nodes)', () => {
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 12.7, y: -3.2 }, data: { label: 'A', type: 'Add', params: {} } },
    ] as any);
    store().addNote('text', { x: 5.6, y: 9.1 });
    const g = store().getSerializedGraph();
    const regular = g.nodes.find((n) => n.id === 'n1')!;
    const note = g.nodes.find((n) => n.type === 'note')!;
    expect(regular.position).toEqual({ x: 13, y: -3 });
    expect(note.position).toEqual({ x: 6, y: 9 });
  });

  it('strips SECRET-typed param values to "" (using the node definition)', () => {
    const definition = {
      node_name: 'LLMChat', category: 'LLM', description: '', inputs: [], outputs: [],
      params: [
        { name: 'openai_api_key', param_type: 'secret', default: '', description: '', options: [], min_value: null, max_value: null },
        { name: 'model', param_type: 'string', default: '', description: '', options: [], min_value: null, max_value: null },
      ],
    };
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'LLM', type: 'LLMChat', params: { openai_api_key: 'sk-secret', model: 'gpt-5.2' }, definition } },
    ] as any);
    const g = store().getSerializedGraph();
    expect(g.nodes[0].data.params.openai_api_key).toBe('');   // secret blanked
    expect(g.nodes[0].data.params.model).toBe('gpt-5.2');     // non-secret kept
  });

  it('leaves params untouched when the node has no secret-typed params', () => {
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 1, y: 2 }, data: { label: 'A', type: 'Dataset', params: { p: 1 } } },
    ] as any);
    const g = store().getSerializedGraph();
    // Identity preserved (no definition -> nothing to strip).
    expect(g.nodes[0].data.params).toEqual({ p: 1 });
  });

  it('strips secret-typed values from a preset node internalParams (old preset)', () => {
    // An OLD preset (created before secrets were withheld) exposes a secret
    // param, so its param_def.param_type === "secret" pins the inner slot.
    const preset = makePreset({
      exposed_params: [
        {
          internal_node: 'inner1', param_name: 'api_key',
          display_name: 'Dataset - api_key', group: 'Dataset',
          param_def: { name: 'api_key', param_type: 'secret', default: '', description: '', options: [], min_value: null, max_value: null },
        },
        {
          internal_node: 'inner1', param_name: 'a',
          display_name: 'Dataset - a', group: 'Dataset',
          param_def: { name: 'a', param_type: 'int', default: 1, description: '', options: [], min_value: null, max_value: null },
        },
      ],
    });
    store().addPresetNode(preset, { x: 0, y: 0 });
    const nodeId = activeTab().nodes[0].id;
    store().updatePresetInternalParam(nodeId, 'inner1', 'api_key', 'sk-typed-secret');
    const g = store().getSerializedGraph();
    const internal = g.nodes[0].data.internalParams!;
    expect(internal.inner1.api_key).toBe('');   // secret blanked
    expect(internal.inner1.a).toBe(1);          // non-secret override kept
    expect(internal.inner2).toEqual({ b: 2 });  // sibling inner untouched
  });

  it('rounds note boundOffset and width/height in serialization', () => {
    store().setNodes([
      { id: 'note1', type: 'noteNode', position: { x: 0, y: 0 }, data: {
        label: 'Note', type: 'note', params: {}, noteKind: 'text', noteContent: '', noteColor: '#fff',
        boundToNodeId: 'x', boundOffset: { x: 12.7, y: -3.2 }, noteWidth: 200.6, noteHeight: 149.4,
      } },
    ] as any);
    const g = store().getSerializedGraph();
    const note = g.nodes[0] as any;
    expect(note.data.boundOffset).toEqual({ x: 13, y: -3 });
    expect(note.data.noteWidth).toBe(201);
    expect(note.data.noteHeight).toBe(149);
  });
});

// ── graph metadata actions (description / currentGraphFile / segmentGroups) ──

describe('graph metadata actions', () => {
  beforeEach(resetToSingleTab);

  it('setDescription updates the active tab description', () => {
    expect(activeTab().description).toBe('');
    store().setDescription('an important graph');
    expect(activeTab().description).toBe('an important graph');
  });

  it('setCurrentGraphFile binds and unbinds the saved-graph file', () => {
    expect(activeTab().currentGraphFile).toBeNull();
    store().setCurrentGraphFile('my_graph');
    expect(activeTab().currentGraphFile).toBe('my_graph');
    store().setCurrentGraphFile(null);
    expect(activeTab().currentGraphFile).toBeNull();
  });

  it('setSegmentGroups replaces the whole segmentGroups array', () => {
    store().addSegmentGroup({ id: 'g0', headNodeId: 'x', tailNodeId: 'y' });
    store().setSegmentGroups([{ id: 'g1', headNodeId: 'a', tailNodeId: 'b' }]);
    expect(activeTab().segmentGroups).toEqual([{ id: 'g1', headNodeId: 'a', tailNodeId: 'b' }]);
  });
});

// ── deleteNode / duplicateNode / renameNode ──────────────────────────────────

describe('deleteNode', () => {
  beforeEach(resetToSingleTab);

  it('removes the node, its edges, and clears selection if it was selected', () => {
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {} } },
      { id: 'n2', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'B', type: 'B', params: {} } },
    ] as any);
    store().setEdges([
      { id: 'e1', source: 'n1', target: 'n2' },
      { id: 'e2', source: 'n2', target: 'n1' },
    ] as any);
    store().setSelectedNodeId('n1');
    store().deleteNode('n1');
    const tab = activeTab();
    expect(tab.nodes.map((n) => n.id)).toEqual(['n2']);
    expect(tab.edges).toHaveLength(0);
    expect(tab.selectedNodeId).toBeNull();
    expect(tab.undoStack.length).toBe(1);
  });

  it('keeps selection if a different node was selected', () => {
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {} } },
      { id: 'n2', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'B', type: 'B', params: {} } },
    ] as any);
    store().setSelectedNodeId('n2');
    store().deleteNode('n1');
    expect(activeTab().selectedNodeId).toBe('n2');
  });

  it('unbinds notes that were bound to the deleted node', () => {
    store().setNodes([
      { id: 'comp', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'C', type: 'C', params: {} } },
      { id: 'note', type: 'noteNode', position: { x: 0, y: 0 }, data: { label: 'N', type: 'note', params: {}, boundToNodeId: 'comp', boundOffset: { x: 1, y: 1 } } },
    ] as any);
    store().deleteNode('comp');
    const note = activeTab().nodes.find((n) => n.id === 'note')!;
    expect(note.data.boundToNodeId).toBeNull();
    expect(note.data.boundOffset).toBeNull();
  });

  it('leaves notes bound to other nodes untouched when deleting', () => {
    store().setNodes([
      { id: 'comp', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'C', type: 'C', params: {} } },
      { id: 'other', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'O', type: 'O', params: {} } },
      { id: 'note', type: 'noteNode', position: { x: 0, y: 0 }, data: { label: 'N', type: 'note', params: {}, boundToNodeId: 'other', boundOffset: { x: 1, y: 1 } } },
    ] as any);
    store().deleteNode('comp');
    const note = activeTab().nodes.find((n) => n.id === 'note')!;
    expect(note.data.boundToNodeId).toBe('other');
  });

  it('prunes segment groups (and a dangling active segment) referencing the deleted node', () => {
    store().setNodes([
      { id: 'a', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'Add', params: {} } },
      { id: 'b', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'B', type: 'Add', params: {} } },
      { id: 'c', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'C', type: 'Add', params: {} } },
    ] as any);
    store().addSegmentGroup({ id: 's1', headNodeId: 'a', tailNodeId: 'b' }); // references a
    store().addSegmentGroup({ id: 's2', headNodeId: 'b', tailNodeId: 'c' }); // does not
    store().setActiveSegment({ id: 's1', headNodeId: 'a', tailNodeId: 'b' });
    store().deleteNode('a');
    const tab = activeTab();
    expect(tab.segmentGroups.map((s) => s.id)).toEqual(['s2']);
    expect(tab.activeSegment).toBeNull();
  });

  it('keeps segment groups + active segment when the deleted node is unrelated', () => {
    store().setNodes([
      { id: 'a', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'Add', params: {} } },
      { id: 'b', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'B', type: 'Add', params: {} } },
      { id: 'z', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'Z', type: 'Add', params: {} } },
    ] as any);
    store().addSegmentGroup({ id: 's1', headNodeId: 'a', tailNodeId: 'b' });
    store().setActiveSegment({ id: 's1', headNodeId: 'a', tailNodeId: 'b' });
    store().deleteNode('z');
    const tab = activeTab();
    expect(tab.segmentGroups.map((s) => s.id)).toEqual(['s1']);
    expect(tab.activeSegment?.id).toBe('s1');
  });
});

describe('duplicateNode', () => {
  beforeEach(resetToSingleTab);

  it('clones a node with a new id, offset position, and reset status', () => {
    store().addNode(makeDef(), { x: 100, y: 200 });
    const original = activeTab().nodes[0];
    store().setNodeExecutionStatus(original.id, 'error', 'oops');
    store().duplicateNode(original.id);
    const tab = activeTab();
    expect(tab.nodes).toHaveLength(2);
    const dup = tab.nodes[1];
    expect(dup.id).not.toBe(original.id);
    expect(dup.position).toEqual({ x: 140, y: 240 });
    expect(dup.selected).toBe(false);
    expect(dup.data.executionStatus).toBe('idle');
    expect(dup.data.error).toBeUndefined();
    expect(tab.undoStack.length).toBeGreaterThanOrEqual(1);
  });

  it('is a no-op for an unknown node id (still pushes a snapshot)', () => {
    store().addNode(makeDef(), { x: 0, y: 0 });
    const countBefore = activeTab().nodes.length;
    store().duplicateNode('nope');
    expect(activeTab().nodes.length).toBe(countBefore);
  });
});

describe('renameNode', () => {
  beforeEach(resetToSingleTab);

  it('updates the label of the matching node', () => {
    store().addNode(makeDef(), { x: 0, y: 0 });
    const id = activeTab().nodes[0].id;
    store().renameNode(id, 'NewLabel');
    expect(activeTab().nodes[0].data.label).toBe('NewLabel');
  });

  it('ignores non-matching node ids', () => {
    store().addNode(makeDef(), { x: 0, y: 0 });
    const before = activeTab().nodes[0].data.label;
    store().renameNode('nope', 'X');
    expect(activeTab().nodes[0].data.label).toBe(before);
  });
});

// ── onConnect / onEdgesChange / onNodesChange ────────────────────────────────

describe('onConnect', () => {
  beforeEach(resetToSingleTab);

  it('adds an edge, marks the target dirty and snapshots undo', () => {
    store().onConnect({ source: 'a', target: 'b', sourceHandle: 'sh', targetHandle: 'th' });
    const tab = activeTab();
    expect(tab.edges).toHaveLength(1);
    const e = tab.edges[0];
    expect(e.source).toBe('a');
    expect(e.target).toBe('b');
    expect(e.sourceHandle).toBe('sh');
    expect(e.targetHandle).toBe('th');
    expect([...tab.dirtyNodeIds]).toContain('b');
    expect(tab.undoStack.length).toBe(1);
  });

  it('handles a connection with null handles and null target', () => {
    store().onConnect({ source: 'a', target: null as unknown as string, sourceHandle: null, targetHandle: null });
    const e = activeTab().edges[0];
    expect(e.sourceHandle).toBeUndefined();
    expect(e.targetHandle).toBeUndefined();
    // target was null → markDirty not called, dirty set stays empty
    expect(activeTab().dirtyNodeIds.size).toBe(0);
  });
});

describe('onEdgesChange', () => {
  beforeEach(resetToSingleTab);

  it('applies non-remove edge changes without a snapshot', () => {
    store().setEdges([{ id: 'e1', source: 'a', target: 'b' }] as any);
    store().onEdgesChange([{ id: 'e1', type: 'select', selected: true }]);
    expect(activeTab().edges[0].selected).toBe(true);
    expect(activeTab().undoStack.length).toBe(0);
  });

  it('snapshots undo and removes the edge on a remove change', () => {
    store().setEdges([{ id: 'e1', source: 'a', target: 'b' }] as any);
    store().onEdgesChange([{ id: 'e1', type: 'remove' }]);
    expect(activeTab().edges).toHaveLength(0);
    expect(activeTab().undoStack.length).toBe(1);
  });
});

describe('onNodesChange', () => {
  beforeEach(resetToSingleTab);

  it('applies a position change without snapshot when not a drag start', () => {
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {} } },
    ] as any);
    store().onNodesChange([{ id: 'n1', type: 'position', position: { x: 10, y: 10 } }]);
    expect(activeTab().nodes[0].position).toEqual({ x: 10, y: 10 });
    expect(activeTab().undoStack.length).toBe(0);
  });

  it('snapshots once at drag start (dragging=true and no node already dragging)', () => {
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {} } },
    ] as any);
    store().onNodesChange([{ id: 'n1', type: 'position', position: { x: 1, y: 1 }, dragging: true }]);
    expect(activeTab().undoStack.length).toBe(1);
  });

  it('does not snapshot again mid-drag when a node is already dragging', () => {
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, dragging: true, data: { label: 'A', type: 'A', params: {} } },
    ] as any);
    store().onNodesChange([{ id: 'n1', type: 'position', position: { x: 2, y: 2 }, dragging: true }]);
    expect(activeTab().undoStack.length).toBe(0);
  });

  it('snapshots and removes nodes on a remove change', () => {
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {} } },
    ] as any);
    store().onNodesChange([{ id: 'n1', type: 'remove' }]);
    expect(activeTab().nodes).toHaveLength(0);
    expect(activeTab().undoStack.length).toBe(1);
  });

  it('closes an open node-detail modal when its node is removed here', () => {
    // React Flow's Delete key emits a `remove` change straight into this
    // reducer — it never calls `deleteNode` — so the modal id has to be
    // cleared here too, or an undo that restores the node reopens the modal.
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {} } },
      { id: 'n2', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'B', type: 'B', params: {} } },
    ] as any);
    store().openNodeDetail('n1');
    store().onNodesChange([{ id: 'n1', type: 'remove' }]);
    expect(activeTab().nodeDetailNodeId).toBeNull();
  });

  it('leaves an open node-detail modal alone when a different node is removed', () => {
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {} } },
      { id: 'n2', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'B', type: 'B', params: {} } },
    ] as any);
    store().openNodeDetail('n1');
    store().onNodesChange([{ id: 'n2', type: 'remove' }]);
    expect(activeTab().nodeDetailNodeId).toBe('n1');
  });

  // ── deep-linking a tab and a port (#129) ──────────────────────────────────

  it('opens the detail modal with no tab or port requested by default', () => {
    store().openNodeDetail('n1');
    expect(activeTab().nodeDetailTab).toBeNull();
    expect(activeTab().nodeDetailPort).toBeNull();
  });

  it('records the requested tab and port when one is given', () => {
    store().openNodeDetail('n1', { tab: 'stats', port: 'src::out' });
    expect(activeTab().nodeDetailNodeId).toBe('n1');
    expect(activeTab().nodeDetailTab).toBe('stats');
    expect(activeTab().nodeDetailPort).toBe('src::out');
  });

  it('clears a previous deep link when the modal is opened plainly', () => {
    store().openNodeDetail('n1', { tab: 'stats', port: 'src::out' });
    store().openNodeDetail('n2');
    expect(activeTab().nodeDetailTab).toBeNull();
    expect(activeTab().nodeDetailPort).toBeNull();
  });

  it('bumps the request nonce on every open, target or not', () => {
    const start = activeTab().nodeDetailRequest;
    store().openNodeDetail('n1', { tab: 'stats', port: 'src::out' });
    const first = activeTab().nodeDetailRequest;
    expect(first).toBeGreaterThan(start);

    // Same node, same tab: only the nonce says anything happened.
    store().openNodeDetail('n1', { tab: 'stats', port: 'other::out' });
    expect(activeTab().nodeDetailRequest).toBeGreaterThan(first);
  });

  it('clears the deep link on close', () => {
    store().openNodeDetail('n1', { tab: 'stats', port: 'src::out' });
    store().closeNodeDetail();
    expect(activeTab().nodeDetailNodeId).toBeNull();
    expect(activeTab().nodeDetailTab).toBeNull();
    expect(activeTab().nodeDetailPort).toBeNull();
  });

  // ── selection desync between the store and React Flow (#167) ─────────────

  it('openNodeDetail also selects the target node in React Flow, deselecting the rest (#167)', () => {
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, selected: true, data: { label: 'A', type: 'A', params: {} } },
      { id: 'n2', type: 'baseNode', position: { x: 0, y: 0 }, selected: false, data: { label: 'B', type: 'B', params: {} } },
    ] as any);
    // Mirrors NodeDetailModal's arrow-key `goTo`, which calls openNodeDetail
    // with no target while walking from n1 to n2.
    store().openNodeDetail('n2');
    const nodes = activeTab().nodes;
    expect(nodes.find((n) => n.id === 'n1')!.selected).toBe(false);
    expect(nodes.find((n) => n.id === 'n2')!.selected).toBe(true);
  });

  it('lets a Delete-key removal target the node the modal is showing, not a stale click (#167)', () => {
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, selected: true, data: { label: 'A', type: 'A', params: {} } },
      { id: 'n2', type: 'baseNode', position: { x: 0, y: 0 }, selected: false, data: { label: 'B', type: 'B', params: {} } },
    ] as any);
    // Arrow-navigate the open detail modal from n1 to n2.
    store().openNodeDetail('n2');
    // React Flow's `deleteKeyCode` removes whatever node(s) it currently
    // considers `.selected` -- simulate exactly that straight into the
    // reducer, the same way the real Delete key does.
    const toRemove = activeTab().nodes.filter((n) => n.selected).map((n) => n.id);
    store().onNodesChange(toRemove.map((id) => ({ id, type: 'remove' as const })));
    const remainingIds = activeTab().nodes.map((n) => n.id);
    expect(remainingIds).not.toContain('n2');
    expect(remainingIds).toContain('n1');
  });

  it('clears selectedNodeId when the selected node is removed here', () => {
    // React Flow's Delete key never calls `deleteNode` -- it emits a `remove`
    // change straight into this reducer -- so `selectedNodeId` has to be
    // cleared here too, the same way `nodeDetailNodeId` already is above, or
    // it is left naming a node that no longer exists.
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {} } },
    ] as any);
    store().setSelectedNodeId('n1');
    store().onNodesChange([{ id: 'n1', type: 'remove' }]);
    expect(activeTab().selectedNodeId).toBeNull();
  });

  it('leaves selectedNodeId alone when a different node is removed', () => {
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {} } },
      { id: 'n2', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'B', type: 'B', params: {} } },
    ] as any);
    store().setSelectedNodeId('n1');
    store().onNodesChange([{ id: 'n2', type: 'remove' }]);
    expect(activeTab().selectedNodeId).toBe('n1');
  });

  it('recalculates a bound note offset when the note itself is dragged', () => {
    store().setNodes([
      { id: 'comp', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'C', type: 'C', params: {} } },
      { id: 'note', type: 'noteNode', position: { x: 100, y: 100 }, data: { label: 'N', type: 'note', params: {}, boundToNodeId: 'comp', boundOffset: { x: 100, y: 100 } } },
    ] as any);
    // Drag the note to (130, 140); parent stays at (0,0) → new offset (130,140)
    store().onNodesChange([{ id: 'note', type: 'position', position: { x: 130, y: 140 } }]);
    const note = activeTab().nodes.find((n) => n.id === 'note')!;
    expect(note.data.boundOffset).toEqual({ x: 130, y: 140 });
  });

  it('does not recalc offset for a bound note whose parent is missing', () => {
    store().setNodes([
      { id: 'note', type: 'noteNode', position: { x: 100, y: 100 }, data: { label: 'N', type: 'note', params: {}, boundToNodeId: 'ghost', boundOffset: { x: 5, y: 5 } } },
    ] as any);
    store().onNodesChange([{ id: 'note', type: 'position', position: { x: 130, y: 140 } }]);
    const note = activeTab().nodes[0];
    // parent not found → offset unchanged
    expect(note.data.boundOffset).toEqual({ x: 5, y: 5 });
  });

  it('skips notes with no boundOffset during the note-move branch', () => {
    store().setNodes([
      { id: 'comp', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'C', type: 'C', params: {} } },
      { id: 'note', type: 'noteNode', position: { x: 100, y: 100 }, data: { label: 'N', type: 'note', params: {}, boundToNodeId: 'comp', boundOffset: null } },
    ] as any);
    store().onNodesChange([{ id: 'note', type: 'position', position: { x: 130, y: 140 } }]);
    const note = activeTab().nodes.find((n) => n.id === 'note')!;
    expect(note.data.boundOffset).toBeNull();
    expect(note.position).toEqual({ x: 130, y: 140 });
  });

  it('repositions bound notes when their parent computational node moves', () => {
    store().setNodes([
      { id: 'comp', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'C', type: 'C', params: {} } },
      { id: 'note', type: 'noteNode', position: { x: 50, y: 50 }, data: { label: 'N', type: 'note', params: {}, boundToNodeId: 'comp', boundOffset: { x: 50, y: 50 } } },
    ] as any);
    // Move comp to (200, 300); note (not moved) → comp + offset = (250, 350)
    store().onNodesChange([{ id: 'comp', type: 'position', position: { x: 200, y: 300 } }]);
    const note = activeTab().nodes.find((n) => n.id === 'note')!;
    expect(note.position).toEqual({ x: 250, y: 350 });
  });

  it('skips repositioning a note that was moved together with its parent', () => {
    store().setNodes([
      { id: 'comp', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'C', type: 'C', params: {} } },
      { id: 'note', type: 'noteNode', position: { x: 50, y: 50 }, data: { label: 'N', type: 'note', params: {}, boundToNodeId: 'comp', boundOffset: { x: 50, y: 50 } } },
    ] as any);
    // Both comp and note get position changes — note is in movedIds so it is skipped
    // in the reposition branch (its offset is recalculated in branch 1 instead).
    store().onNodesChange([
      { id: 'comp', type: 'position', position: { x: 200, y: 300 } },
      { id: 'note', type: 'position', position: { x: 999, y: 999 } },
    ]);
    const note = activeTab().nodes.find((n) => n.id === 'note')!;
    // Note kept its own dragged position (not repositioned from parent)
    expect(note.position).toEqual({ x: 999, y: 999 });
  });

  it('does not reposition notes bound to a node that did not move', () => {
    store().setNodes([
      { id: 'comp', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'C', type: 'C', params: {} } },
      { id: 'other', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'O', type: 'O', params: {} } },
      { id: 'note', type: 'noteNode', position: { x: 50, y: 50 }, data: { label: 'N', type: 'note', params: {}, boundToNodeId: 'other', boundOffset: { x: 50, y: 50 } } },
    ] as any);
    store().onNodesChange([{ id: 'comp', type: 'position', position: { x: 200, y: 300 } }]);
    const note = activeTab().nodes.find((n) => n.id === 'note')!;
    expect(note.position).toEqual({ x: 50, y: 50 });
  });

  it('does not reposition a bound note whose parent vanished mid-update', () => {
    // moved node is comp, note bound to comp, but parent lookup in branch 2
    // succeeds; to exercise the `if (!parent) return n` we bind to a moved id
    // that is not actually present after changes. Use a remove of the parent
    // combined with a position change of another node is complex; instead bind
    // the note to a node that gets a position change but is then absent.
    // Simpler reachable case: note bound to comp, comp moved, parent found.
    // The not-found branch is covered by binding to a moved computational id
    // that is missing — emulate by referencing a moved id with no node.
    store().setNodes([
      { id: 'note', type: 'noteNode', position: { x: 50, y: 50 }, data: { label: 'N', type: 'note', params: {}, boundToNodeId: 'phantom', boundOffset: { x: 50, y: 50 } } },
    ] as any);
    // 'phantom' gets a position change but no node exists → movedComputational
    // contains nothing (node lookup fails), so this asserts the guard path is safe.
    store().onNodesChange([{ id: 'phantom', type: 'position', position: { x: 1, y: 1 } }]);
    const note = activeTab().nodes[0];
    expect(note.position).toEqual({ x: 50, y: 50 });
  });

  it('unbinds notes when their parent node is removed via a change', () => {
    store().setNodes([
      { id: 'comp', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'C', type: 'C', params: {} } },
      { id: 'note', type: 'noteNode', position: { x: 0, y: 0 }, data: { label: 'N', type: 'note', params: {}, boundToNodeId: 'comp', boundOffset: { x: 1, y: 1 } } },
    ] as any);
    store().onNodesChange([{ id: 'comp', type: 'remove' }]);
    const note = activeTab().nodes.find((n) => n.id === 'note')!;
    expect(note.data.boundToNodeId).toBeNull();
    expect(note.data.boundOffset).toBeNull();
  });

  it('keeps a note bound to a surviving node when a different node is removed', () => {
    store().setNodes([
      { id: 'comp', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'C', type: 'C', params: {} } },
      { id: 'other', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'O', type: 'O', params: {} } },
      { id: 'note', type: 'noteNode', position: { x: 0, y: 0 }, data: { label: 'N', type: 'note', params: {}, boundToNodeId: 'comp', boundOffset: { x: 1, y: 1 } } },
    ] as any);
    store().onNodesChange([{ id: 'other', type: 'remove' }]);
    const note = activeTab().nodes.find((n) => n.id === 'note')!;
    expect(note.data.boundToNodeId).toBe('comp');
  });

  it('leaves unbound notes untouched during remove unbinding', () => {
    store().setNodes([
      { id: 'comp', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'C', type: 'C', params: {} } },
      { id: 'note', type: 'noteNode', position: { x: 0, y: 0 }, data: { label: 'N', type: 'note', params: {}, boundToNodeId: null } },
    ] as any);
    store().onNodesChange([{ id: 'comp', type: 'remove' }]);
    const note = activeTab().nodes.find((n) => n.id === 'note')!;
    expect(note.data.boundToNodeId).toBeNull();
  });
});

// ── notes: addNote / updateNoteData / binding ────────────────────────────────

describe('notes', () => {
  beforeEach(resetToSingleTab);

  it('addNote("text") creates a text note with undefined height', () => {
    store().addNote('text', { x: 1, y: 2 });
    const n = activeTab().nodes[0];
    expect(n.type).toBe('noteNode');
    expect(n.data.noteKind).toBe('text');
    expect(n.data.noteHeight).toBeUndefined();
    expect(activeTab().undoStack.length).toBe(1);
  });

  it('addNote("image") creates an image note with a default height', () => {
    store().addNote('image', { x: 1, y: 2 });
    expect(activeTab().nodes[0].data.noteHeight).toBe(150);
  });

  it('updateNoteData merges updates into the matching note', () => {
    store().addNote('text', { x: 0, y: 0 });
    const id = activeTab().nodes[0].id;
    store().updateNoteData(id, { noteContent: 'hello', noteColor: '#fff' });
    const n = activeTab().nodes[0];
    expect(n.data.noteContent).toBe('hello');
    expect(n.data.noteColor).toBe('#fff');
  });

  it('updateNoteData ignores non-matching ids', () => {
    store().addNote('text', { x: 0, y: 0 });
    store().updateNoteData('nope', { noteContent: 'x' });
    expect(activeTab().nodes[0].data.noteContent).toBe('');
  });

  it('bindNoteToNode stores the relative offset', () => {
    store().setNodes([
      { id: 'comp', type: 'baseNode', position: { x: 10, y: 20 }, data: { label: 'C', type: 'C', params: {} } },
      { id: 'note', type: 'noteNode', position: { x: 60, y: 90 }, data: { label: 'N', type: 'note', params: {}, boundToNodeId: null } },
    ] as any);
    store().bindNoteToNode('note', 'comp');
    const note = activeTab().nodes.find((n) => n.id === 'note')!;
    expect(note.data.boundToNodeId).toBe('comp');
    expect(note.data.boundOffset).toEqual({ x: 50, y: 70 });
    expect(activeTab().undoStack.length).toBe(1);
  });

  it('bindNoteToNode is a no-op if the note is missing', () => {
    store().setNodes([
      { id: 'comp', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'C', type: 'C', params: {} } },
    ] as any);
    store().bindNoteToNode('ghostNote', 'comp');
    // Only the pushUndoSnapshot ran; nodes unchanged
    expect(activeTab().nodes).toHaveLength(1);
  });

  it('bindNoteToNode is a no-op if the target is missing', () => {
    store().setNodes([
      { id: 'note', type: 'noteNode', position: { x: 0, y: 0 }, data: { label: 'N', type: 'note', params: {}, boundToNodeId: null } },
    ] as any);
    store().bindNoteToNode('note', 'ghostTarget');
    expect(activeTab().nodes[0].data.boundToNodeId).toBeNull();
  });

  it('unbindNote clears binding fields', () => {
    store().setNodes([
      { id: 'note', type: 'noteNode', position: { x: 0, y: 0 }, data: { label: 'N', type: 'note', params: {}, boundToNodeId: 'x', boundOffset: { x: 1, y: 1 } } },
    ] as any);
    store().unbindNote('note');
    const note = activeTab().nodes[0];
    expect(note.data.boundToNodeId).toBeNull();
    expect(note.data.boundOffset).toBeNull();
    expect(activeTab().undoStack.length).toBe(1);
  });

  it('unbindNote ignores non-matching ids', () => {
    store().setNodes([
      { id: 'note', type: 'noteNode', position: { x: 0, y: 0 }, data: { label: 'N', type: 'note', params: {}, boundToNodeId: 'x', boundOffset: { x: 1, y: 1 } } },
    ] as any);
    store().unbindNote('other');
    expect(activeTab().nodes[0].data.boundToNodeId).toBe('x');
  });
});

describe('bindNoteToNearest', () => {
  beforeEach(resetToSingleTab);

  it('binds to the nearest non-note node using measured sizes', () => {
    store().setNodes([
      { id: 'near', type: 'baseNode', position: { x: 100, y: 100 }, measured: { width: 200, height: 80 }, data: { label: 'C', type: 'C', params: {} } },
      { id: 'far', type: 'baseNode', position: { x: 900, y: 900 }, measured: { width: 200, height: 80 }, data: { label: 'F', type: 'F', params: {} } },
      { id: 'note', type: 'noteNode', position: { x: 110, y: 110 }, measured: { width: 200, height: 80 }, data: { label: 'N', type: 'note', params: {}, boundToNodeId: null } },
    ] as any);
    store().bindNoteToNearest('note');
    const note = activeTab().nodes.find((n) => n.id === 'note')!;
    expect(note.data.boundToNodeId).toBe('near');
  });

  it('falls back to default sizes when measured is absent', () => {
    store().setNodes([
      { id: 'near', type: 'baseNode', position: { x: 100, y: 100 }, data: { label: 'C', type: 'C', params: {} } },
      { id: 'note', type: 'noteNode', position: { x: 100, y: 100 }, data: { label: 'N', type: 'note', params: {}, boundToNodeId: null } },
    ] as any);
    store().bindNoteToNearest('note');
    expect(activeTab().nodes.find((n) => n.id === 'note')!.data.boundToNodeId).toBe('near');
  });

  it('is a no-op when the note id is unknown', () => {
    store().setNodes([
      { id: 'comp', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'C', type: 'C', params: {} } },
    ] as any);
    store().bindNoteToNearest('ghost');
    expect(activeTab().nodes).toHaveLength(1);
  });

  it('does nothing when there are no candidate nodes (only notes)', () => {
    store().setNodes([
      { id: 'note', type: 'noteNode', position: { x: 0, y: 0 }, data: { label: 'N', type: 'note', params: {}, boundToNodeId: null } },
    ] as any);
    store().bindNoteToNearest('note');
    expect(activeTab().nodes[0].data.boundToNodeId).toBeNull();
  });
});

// ── undo / redo ──────────────────────────────────────────────────────────────

describe('undo / redo', () => {
  beforeEach(resetToSingleTab);

  it('pushUndoSnapshot captures current nodes/edges and clears the redo stack', () => {
    store().setNodes([{ id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {} } }] as any);
    store().pushUndoSnapshot();
    const tab = activeTab();
    expect(tab.undoStack).toHaveLength(1);
    expect(tab.undoStack[0].nodes).toHaveLength(1);
    expect(tab.redoStack).toHaveLength(0);
  });

  it('caps the undo stack at MAX_UNDO entries', () => {
    for (let i = 0; i < 60; i++) store().pushUndoSnapshot();
    expect(activeTab().undoStack.length).toBe(50);
  });

  it('undo is a no-op when the undo stack is empty', () => {
    store().undo();
    expect(activeTab().nodes).toHaveLength(0);
  });

  it('undo restores the previous snapshot and pushes the current onto redo', () => {
    store().setNodes([{ id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {} } }] as any);
    store().pushUndoSnapshot();
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {} } },
      { id: 'n2', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'B', type: 'B', params: {} } },
    ] as any);
    store().undo();
    expect(activeTab().nodes).toHaveLength(1);
    expect(activeTab().redoStack).toHaveLength(1);
    expect(activeTab().redoStack[0].nodes).toHaveLength(2);
  });

  it('redo is a no-op when the redo stack is empty', () => {
    store().redo();
    expect(activeTab().nodes).toHaveLength(0);
  });

  it('redo re-applies the undone snapshot and pushes the current onto undo', () => {
    store().setNodes([{ id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {} } }] as any);
    store().pushUndoSnapshot();
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {} } },
      { id: 'n2', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'B', type: 'B', params: {} } },
    ] as any);
    store().undo();
    store().redo();
    expect(activeTab().nodes).toHaveLength(2);
    expect(activeTab().undoStack.length).toBeGreaterThanOrEqual(1);
  });

  // ── shallow snapshot semantics (#125) ──────────────────────────────────────
  // Snapshots stopped being `JSON.parse(JSON.stringify(...))` deep clones.
  // These pin the two properties that makes safe, and the one it improves.

  it('snapshots the ARRAY, not the live reference, so later writes cannot reach it', () => {
    const first = { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {} } };
    store().setNodes([first] as any);
    store().pushUndoSnapshot();
    const snapshot = activeTab().undoStack[0].nodes;

    store().setNodes([]);
    expect(snapshot).toHaveLength(1);
    expect(activeTab().nodes).toHaveLength(0);
    // The snapshot array is its own object, not the one the store now holds.
    expect(snapshot).not.toBe(activeTab().nodes);
  });

  it('keeps node object identity through undo (React Flow re-adopts only what changed)', () => {
    const a = { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {} } };
    store().setNodes([a] as any);
    store().pushUndoSnapshot();
    const beforeUndo = activeTab().nodes[0];

    store().setNodes([
      a,
      { id: 'n2', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'B', type: 'B', params: {} } },
    ] as any);
    store().undo();

    // A deep clone would have produced a structurally-equal but distinct
    // object here, forcing React Flow to rebuild every node on every undo.
    expect(activeTab().nodes[0]).toBe(beforeUndo);
  });

  it('preserves keys whose value is undefined (JSON round-trips dropped them)', () => {
    store().setNodes([
      {
        id: 'n1',
        type: 'baseNode',
        position: { x: 0, y: 0 },
        data: { label: 'A', type: 'A', params: {}, executionStatus: 'idle', error: undefined },
      },
    ] as any);
    store().pushUndoSnapshot();
    store().setNodes([]);
    store().undo();
    expect('error' in (activeTab().nodes[0].data as object)).toBe(true);
  });

  it('does not share the node definition object with a JSON copy', () => {
    // Nodes carry the definition object owned by nodeDefStore. Deep cloning
    // replaced it with a detached copy on every undo; identity must survive.
    const definition = { node_name: 'A', category: 'X', description: '', inputs: [], outputs: [], params: [] };
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {}, definition } },
    ] as any);
    store().pushUndoSnapshot();
    store().setNodes([]);
    store().undo();
    expect(activeTab().nodes[0].data.definition).toBe(definition);
  });
});

// ── clipboard: copy / paste ──────────────────────────────────────────────────

describe('clipboard', () => {
  beforeEach(resetToSingleTab);

  it('copySelectedNodes is a no-op with no selection', () => {
    store().setNodes([{ id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, selected: false, data: { label: 'A', type: 'A', params: {} } }] as any);
    store().copySelectedNodes();
    expect(store().clipboard).toBeNull();
  });

  it('copySelectedNodes captures selected nodes and only internal edges', () => {
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, selected: true, data: { label: 'A', type: 'A', params: {} } },
      { id: 'n2', type: 'baseNode', position: { x: 0, y: 0 }, selected: true, data: { label: 'B', type: 'B', params: {} } },
      { id: 'n3', type: 'baseNode', position: { x: 0, y: 0 }, selected: false, data: { label: 'C', type: 'C', params: {} } },
    ] as any);
    store().setEdges([
      { id: 'e1', source: 'n1', target: 'n2' }, // both selected → kept
      { id: 'e2', source: 'n2', target: 'n3' }, // n3 not selected → dropped
    ] as any);
    store().copySelectedNodes();
    const clip = store().clipboard!;
    expect(clip.nodes.map((n) => n.id).sort()).toEqual(['n1', 'n2']);
    expect(clip.edges.map((e) => e.id)).toEqual(['e1']);
  });

  it('pasteNodes is a no-op when clipboard is null', () => {
    store().setNodes([] as any);
    store().pasteNodes();
    expect(activeTab().nodes).toHaveLength(0);
  });

  it('pasteNodes is a no-op when clipboard has no nodes', () => {
    useTabStore.setState({ clipboard: { nodes: [], edges: [] } });
    store().pasteNodes();
    expect(activeTab().nodes).toHaveLength(0);
  });

  it('pasteNodes inserts offset clones, deselects originals, and remaps internal edges', () => {
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, selected: true, data: { label: 'A', type: 'A', params: {} } },
      { id: 'n2', type: 'baseNode', position: { x: 0, y: 0 }, selected: true, data: { label: 'B', type: 'B', params: {} } },
    ] as any);
    store().setEdges([{ id: 'e1', source: 'n1', target: 'n2' }] as any);
    store().copySelectedNodes();
    store().pasteNodes();
    const tab = activeTab();
    // originals (2) + pasted (2)
    expect(tab.nodes).toHaveLength(4);
    // originals deselected
    expect(tab.nodes.slice(0, 2).every((n) => n.selected === false)).toBe(true);
    // pasted selected and offset by +50
    const pasted = tab.nodes.slice(2);
    expect(pasted.every((n) => n.selected === true)).toBe(true);
    expect(pasted[0].position).toEqual({ x: 50, y: 50 });
    // edge remapped to new ids
    const newEdge = tab.edges[tab.edges.length - 1];
    const pastedIds = pasted.map((n) => n.id);
    expect(pastedIds).toContain(newEdge.source);
    expect(pastedIds).toContain(newEdge.target);
    expect(tab.undoStack.length).toBeGreaterThanOrEqual(1);
  });

  it('pasteNodes remaps a note binding when its parent was also copied', () => {
    store().setNodes([
      { id: 'comp', type: 'baseNode', position: { x: 0, y: 0 }, selected: true, data: { label: 'C', type: 'C', params: {} } },
      { id: 'note', type: 'noteNode', position: { x: 0, y: 0 }, selected: true, data: { label: 'N', type: 'note', params: {}, boundToNodeId: 'comp', boundOffset: { x: 5, y: 5 } } },
    ] as any);
    store().copySelectedNodes();
    store().pasteNodes();
    const pasted = activeTab().nodes.slice(2);
    const pastedComp = pasted.find((n) => n.type === 'baseNode')!;
    const pastedNote = pasted.find((n) => n.type === 'noteNode')!;
    expect(pastedNote.data.boundToNodeId).toBe(pastedComp.id);
    expect(pastedNote.data.boundOffset).toEqual({ x: 5, y: 5 });
  });

  it('pasteNodes clears a note binding when its parent was not copied', () => {
    store().setNodes([
      { id: 'comp', type: 'baseNode', position: { x: 0, y: 0 }, selected: false, data: { label: 'C', type: 'C', params: {} } },
      { id: 'note', type: 'noteNode', position: { x: 0, y: 0 }, selected: true, data: { label: 'N', type: 'note', params: {}, boundToNodeId: 'comp', boundOffset: { x: 5, y: 5 } } },
    ] as any);
    store().copySelectedNodes();
    store().pasteNodes();
    const pastedNote = activeTab().nodes.find((n) => n.type === 'noteNode' && n.selected)!;
    expect(pastedNote.data.boundToNodeId).toBeNull();
    expect(pastedNote.data.boundOffset).toBeNull();
  });

  it('pasteNodes keeps an external edge endpoint that maps to no pasted node', () => {
    // Put an edge in the clipboard that references an id not among pasted nodes.
    useTabStore.setState({
      clipboard: {
        nodes: [
          { id: 'a', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {} } },
        ] as any,
        edges: [
          { id: 'eX', source: 'a', target: 'external' },
        ] as any,
      },
    });
    store().pasteNodes();
    const newEdge = activeTab().edges[activeTab().edges.length - 1];
    // target had no mapping → falls back to original id
    expect(newEdge.target).toBe('external');
  });
});

// ── dirty tracking ───────────────────────────────────────────────────────────

describe('dirty tracking', () => {
  beforeEach(resetToSingleTab);

  it('markDirty adds an id and clearDirty empties the set', () => {
    store().markDirty('a');
    store().markDirty('b');
    expect(activeTab().dirtyNodeIds.size).toBe(2);
    store().clearDirty();
    expect(activeTab().dirtyNodeIds.size).toBe(0);
  });

  it('getDirtyWithDownstream returns [] when nothing is dirty', () => {
    expect(store().getDirtyWithDownstream()).toEqual([]);
  });

  it('getDirtyWithDownstream walks downstream edges via BFS', () => {
    store().setEdges([
      { id: 'e1', source: 'a', target: 'b' },
      { id: 'e2', source: 'b', target: 'c' },
      { id: 'e3', source: 'x', target: 'y' }, // unrelated
    ] as any);
    store().markDirty('a');
    const result = store().getDirtyWithDownstream().sort();
    expect(result).toEqual(['a', 'b', 'c']);
  });

  it('getDirtyWithDownstream does not revisit already-seen nodes (cycle safe)', () => {
    store().setEdges([
      { id: 'e1', source: 'a', target: 'b' },
      { id: 'e2', source: 'b', target: 'a' }, // cycle
    ] as any);
    store().markDirty('a');
    expect(store().getDirtyWithDownstream().sort()).toEqual(['a', 'b']);
  });

  it('getDirtyWithDownstream handles a dirty node with no outgoing edges', () => {
    store().markDirty('lonely');
    expect(store().getDirtyWithDownstream()).toEqual(['lonely']);
  });
});

// ── execution actions (active tab) ───────────────────────────────────────────

describe('execution actions on the active tab', () => {
  beforeEach(resetToSingleTab);

  it('setStatus updates the active tab status', () => {
    store().setStatus('running');
    expect(activeTab().status).toBe('running');
  });

  it('addLog appends a timestamped entry; clearLogs empties them', () => {
    store().addLog({ message: 'hello', type: 'info' });
    expect(activeTab().logs).toHaveLength(1);
    expect(activeTab().logs[0].message).toBe('hello');
    expect(typeof activeTab().logs[0].timestamp).toBe('number');
    store().clearLogs();
    expect(activeTab().logs).toHaveLength(0);
  });
});

// ── tab-specific (WS-targeted) execution actions ─────────────────────────────

describe('tab-specific execution actions', () => {
  beforeEach(resetToSingleTab);

  it('setTabNodeExecutionStatus targets a specific tab and node', () => {
    const tabId = store().activeTabId;
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {}, executionStatus: 'idle' } },
    ] as any);
    store().setTabNodeExecutionStatus(tabId, 'n1', 'completed', undefined);
    expect(store().getTab(tabId)!.nodes[0].data.executionStatus).toBe('completed');
  });

  it('setTabNodeExecutionStatus ignores non-matching node ids', () => {
    const tabId = store().activeTabId;
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {}, executionStatus: 'idle' } },
    ] as any);
    store().setTabNodeExecutionStatus(tabId, 'other', 'running');
    expect(store().getTab(tabId)!.nodes[0].data.executionStatus).toBe('idle');
  });

  it('setTabNodeProgress writes a progress payload onto the node', () => {
    const tabId = store().activeTabId;
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {} } },
    ] as any);
    store().setTabNodeProgress(tabId, 'n1', { event: 'epoch', epoch: 1, total_epochs: 5 });
    expect(store().getTab(tabId)!.nodes[0].data.progress).toEqual({ event: 'epoch', epoch: 1, total_epochs: 5 });
  });

  it('setTabNodeProgress ignores non-matching node ids', () => {
    const tabId = store().activeTabId;
    store().setNodes([
      { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {} } },
    ] as any);
    store().setTabNodeProgress(tabId, 'other', { event: 'x' });
    expect(store().getTab(tabId)!.nodes[0].data.progress).toBeUndefined();
  });

  it('setTabOutputSummary stores a per-node output summary', () => {
    const tabId = store().activeTabId;
    store().setTabOutputSummary(tabId, 'n1', { out: { type: 'TENSOR', shape: [2, 2] } });
    expect(store().getTab(tabId)!.outputSummaries.n1).toEqual({ out: { type: 'TENSOR', shape: [2, 2] } });
  });

  it('clearOutputSummaries empties the summaries map', () => {
    const tabId = store().activeTabId;
    store().setTabOutputSummary(tabId, 'n1', { out: { type: 'SCALAR' } });
    store().clearOutputSummaries();
    expect(activeTab().outputSummaries).toEqual({});
  });

  it('setTabStatus sets the status of a specific tab', () => {
    const tabId = store().activeTabId;
    store().setTabStatus(tabId, 'completed');
    expect(store().getTab(tabId)!.status).toBe('completed');
  });

  it('addTabLog appends a timestamped log to a specific tab', () => {
    const tabId = store().activeTabId;
    store().addTabLog(tabId, { message: 'm', type: 'error', nodeId: 'n1' });
    const logs = store().getTab(tabId)!.logs;
    expect(logs).toHaveLength(1);
    expect(logs[0]).toMatchObject({ message: 'm', type: 'error', nodeId: 'n1' });
    expect(typeof logs[0].timestamp).toBe('number');
  });
});

// ── educational toggles ──────────────────────────────────────────────────────

describe('educational toggles', () => {
  beforeEach(resetToSingleTab);

  it('toggleVerbose flips verboseMode', () => {
    expect(activeTab().verboseMode).toBe(false);
    store().toggleVerbose();
    expect(activeTab().verboseMode).toBe(true);
  });

  it('togglePersistWeights flips weightsPersistent', () => {
    expect(activeTab().weightsPersistent).toBe(true);
    store().togglePersistWeights();
    expect(activeTab().weightsPersistent).toBe(false);
  });

  it('toggleBackward flips backwardMode', () => {
    expect(activeTab().backwardMode).toBe(false);
    store().toggleBackward();
    expect(activeTab().backwardMode).toBe(true);
  });

  it('toggleAutoBackward flips autoBackward', () => {
    expect(activeTab().autoBackward).toBe(false);
    store().toggleAutoBackward();
    expect(activeTab().autoBackward).toBe(true);
  });
});

// ── applyLayout edge cases ───────────────────────────────────────────────────

describe('applyLayout edge cases', () => {
  beforeEach(resetToSingleTab);

  it('returns early when activeTabId is falsy', () => {
    useTabStore.setState({ activeTabId: '' as unknown as string });
    // Should not throw and should not push an undo snapshot to any tab.
    expect(() => store().applyLayout('all')).not.toThrow();
  });

  it('warns via toast when unbound notes are present after layout', () => {
    const spy = vi.spyOn(useToastStore.getState(), 'addToast');
    store().setNodes([
      { id: 'a', type: 'baseNode', position: { x: 0, y: 0 }, width: 200, height: 80, data: { label: 'A', type: 'Dataset' } },
      { id: 'note', type: 'noteNode', position: { x: 0, y: 0 }, data: { label: 'N', type: 'note', params: {}, boundToNodeId: null } },
    ] as any);
    store().applyLayout('all');
    expect(spy).toHaveBeenCalledWith(expect.any(String), 'warning');
    spy.mockRestore();
  });

  it('does not warn when all notes are bound', () => {
    const spy = vi.spyOn(useToastStore.getState(), 'addToast');
    store().setNodes([
      { id: 'a', type: 'baseNode', position: { x: 0, y: 0 }, width: 200, height: 80, data: { label: 'A', type: 'Dataset' } },
      { id: 'note', type: 'noteNode', position: { x: 0, y: 0 }, data: { label: 'N', type: 'note', params: {}, boundToNodeId: 'a', boundOffset: { x: 0, y: 0 } } },
    ] as any);
    store().applyLayout('all');
    expect(spy).not.toHaveBeenCalled();
    spy.mockRestore();
  });

  it('leaves a non-active tab untouched when applying layout to the active one', () => {
    const firstId = store().activeTabId;
    store().addTab('Second'); // active = Second
    store().setActiveTab(firstId);
    store().setNodes([
      { id: 'a', type: 'baseNode', position: { x: 0, y: 0 }, width: 200, height: 80, data: { label: 'A', type: 'Dataset' } },
    ] as any);
    store().applyLayout('all');
    // Second tab still has no nodes
    expect(store().tabs.find((t) => t.name === 'Second')!.nodes).toHaveLength(0);
  });
});

// ── targeted branch coverage ─────────────────────────────────────────────────

describe('targeted branch coverage', () => {
  beforeEach(resetToSingleTab);

  it('applyLayout collects selected node ids (selected map callback)', () => {
    // At least one node selected so the `tab.nodes.filter(selected).map(id)`
    // callback runs over a real element.
    store().setNodes([
      { id: 'a', type: 'baseNode', position: { x: 0, y: 0 }, width: 200, height: 80, selected: true, data: { label: 'A', type: 'Dataset' } },
      { id: 'b', type: 'baseNode', position: { x: 0, y: 0 }, width: 200, height: 80, selected: false, data: { label: 'B', type: 'Model' } },
    ] as any);
    store().setEdges([{ id: 'e1', source: 'a', target: 'b' }] as any);
    expect(() => store().applyLayout('selected')).not.toThrow();
    // Selected node was part of the laid-out set.
    expect(activeTab().nodes.find((n) => n.id === 'a')).toBeDefined();
  });

  it('removeTab false-branch: removing an unknown id when >1 tab exists', () => {
    // tabs.length > 1 so we pass the guard, but find() returns undefined →
    // `if (tab) tab.ws.disconnect()` takes the false branch.
    store().addTab('Second');
    const countBefore = store().tabs.length;
    store().removeTab('does-not-exist');
    // No tab removed (filter keeps all), activeTabId unchanged.
    expect(store().tabs.length).toBe(countBefore);
  });

  it('getDirtyWithDownstream reuses an existing adjacency bucket for repeated sources', () => {
    // Two edges share the same source → second edge hits the `adj.has` true
    // branch (the bucket already exists).
    store().setEdges([
      { id: 'e1', source: 'a', target: 'b' },
      { id: 'e2', source: 'a', target: 'c' },
    ] as any);
    store().markDirty('a');
    expect(store().getDirtyWithDownstream().sort()).toEqual(['a', 'b', 'c']);
  });

  it('pasteNodes keeps an edge source that maps to no pasted node', () => {
    // Clipboard edge whose SOURCE is external → `idMap.get(e.source) ?? e.source`
    // right-hand fallback for the source side.
    useTabStore.setState({
      clipboard: {
        nodes: [
          { id: 'a', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {} } },
        ] as any,
        edges: [
          { id: 'eX', source: 'externalSource', target: 'a' },
        ] as any,
      },
    });
    store().pasteNodes();
    const newEdge = activeTab().edges[activeTab().edges.length - 1];
    expect(newEdge.source).toBe('externalSource');
  });

  it('removeSegmentGroup keeps a non-matching active segment (false branch)', () => {
    const segA = { id: 'gA', headNodeId: 'a', tailNodeId: 'b' };
    const segB = { id: 'gB', headNodeId: 'c', tailNodeId: 'd' };
    store().addSegmentGroup(segA);
    store().addSegmentGroup(segB);
    store().setActiveSegment(segB); // active is gB
    store().removeSegmentGroup('gA'); // remove a different one
    // activeSegment id (gB) !== removed id (gA) → keep it
    expect(activeTab().activeSegment).toEqual(segB);
    expect(activeTab().segmentGroups.map((s) => s.id)).toEqual(['gB']);
  });

  it('removeSegmentGroup with no active segment leaves activeSegment null (false branch)', () => {
    store().addSegmentGroup({ id: 'gA', headNodeId: 'a', tailNodeId: 'b' });
    // activeSegment is null (default) → optional chaining short-circuits to keep null
    store().removeSegmentGroup('gA');
    expect(activeTab().activeSegment).toBeNull();
  });
});

// ── localStorage persistence (saveTabs / loadTabs via module reload) ─────────

describe('persistence (module reload)', () => {
  const STORAGE_KEY = 'codefyui-tabs';

  beforeEach(() => {
    vi.resetModules();
    localStorage.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    localStorage.clear();
  });

  it('loadTabs creates a default tab when storage is empty', async () => {
    const mod = await import('./tabStore');
    const tabs = mod.useTabStore.getState().tabs;
    expect(tabs).toHaveLength(1);
    expect(tabs[0].name).toBe('Tab 1');
    expect(mod.useTabStore.getState().activeTabId).toBe(tabs[0].id);
  });

  it('loadTabs hydrates persisted tabs and honours a valid activeTabId', async () => {
    const persisted = {
      activeTabId: 'tab-b',
      tabs: [
        { id: 'tab-a', name: 'Alpha', nodes: [{ id: 'n', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'A', params: {} } }], edges: [], segmentGroups: [{ id: 's', headNodeId: 'h', tailNodeId: 't' }], recordOutputs: false, verboseMode: true, graphId: 'gid-a', weightsPersistent: false, backwardMode: true, autoBackward: true },
        { id: 'tab-b', name: 'Beta', nodes: [], edges: [{ id: 'e', source: 'a', target: 'b' }] },
      ],
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(persisted));
    const mod = await import('./tabStore');
    const st = mod.useTabStore.getState();
    expect(st.tabs.map((t) => t.name)).toEqual(['Alpha', 'Beta']);
    expect(st.activeTabId).toBe('tab-b');
    const alpha = st.tabs[0];
    expect(alpha.nodes).toHaveLength(1);
    expect(alpha.segmentGroups).toEqual([{ id: 's', headNodeId: 'h', tailNodeId: 't' }]);
    expect(alpha.recordOutputs).toBe(false);
    expect(alpha.verboseMode).toBe(true);
    expect(alpha.graphId).toBe('gid-a');
    expect(alpha.weightsPersistent).toBe(false);
    expect(alpha.backwardMode).toBe(true);
    expect(alpha.autoBackward).toBe(true);
  });

  it('loadTabs applies defaults for missing optional fields', async () => {
    const persisted = {
      activeTabId: 'tab-a',
      tabs: [{ id: 'tab-a', name: 'OnlyName' }],
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(persisted));
    const mod = await import('./tabStore');
    const t = mod.useTabStore.getState().tabs[0];
    expect(t.nodes).toEqual([]);
    expect(t.edges).toEqual([]);
    expect(t.segmentGroups).toEqual([]);
    expect(t.recordOutputs).toBe(true);
    expect(t.verboseMode).toBe(false);
    expect(typeof t.graphId).toBe('string'); // fell back to generated graphId
    expect(t.weightsPersistent).toBe(true);
    expect(t.backwardMode).toBe(false);
    expect(t.autoBackward).toBe(false);
  });

  it('loadTabs falls back to first tab id when persisted activeTabId is unknown', async () => {
    const persisted = {
      activeTabId: 'missing',
      tabs: [{ id: 'tab-a', name: 'A' }, { id: 'tab-b', name: 'B' }],
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(persisted));
    const mod = await import('./tabStore');
    expect(mod.useTabStore.getState().activeTabId).toBe('tab-a');
  });

  it('loadTabs coerces a non-array segmentGroups to []', async () => {
    const persisted = {
      activeTabId: 'tab-a',
      tabs: [{ id: 'tab-a', name: 'A', nodes: [], edges: [], segmentGroups: 'oops' }],
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(persisted));
    const mod = await import('./tabStore');
    expect(mod.useTabStore.getState().tabs[0].segmentGroups).toEqual([]);
  });

  it('loadTabs falls back to default when stored tabs array is empty', async () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ activeTabId: 'x', tabs: [] }));
    const mod = await import('./tabStore');
    expect(mod.useTabStore.getState().tabs).toHaveLength(1);
    expect(mod.useTabStore.getState().tabs[0].name).toBe('Tab 1');
  });

  it('loadTabs falls back to default on corrupted JSON', async () => {
    localStorage.setItem(STORAGE_KEY, '{not valid json');
    const mod = await import('./tabStore');
    expect(mod.useTabStore.getState().tabs).toHaveLength(1);
    expect(mod.useTabStore.getState().tabs[0].name).toBe('Tab 1');
  });

  it('createTabState uses the non-crypto graphId fallback when randomUUID is missing', async () => {
    const realCrypto = globalThis.crypto;
    // Provide a crypto WITHOUT randomUUID so the `'randomUUID' in crypto`
    // branch is false. We also stub generateId's crypto usage by giving the
    // persisted tabs explicit ids/graphId-less entries so loadTabs runs
    // createTabState without needing crypto.randomUUID at id-generation time.
    const persisted = {
      activeTabId: 'tab-a',
      // No graphId on the tab → createTabState computes one; with randomUUID
      // absent it must take the `graph-...` fallback branch.
      tabs: [{ id: 'tab-a', name: 'A', nodes: [], edges: [] }],
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(persisted));
    try {
      Object.defineProperty(globalThis, 'crypto', {
        configurable: true,
        value: {}, // no randomUUID
      });
      const mod = await import('./tabStore');
      const gid = mod.useTabStore.getState().tabs[0].graphId;
      expect(gid.startsWith('graph-')).toBe(true);
    } finally {
      Object.defineProperty(globalThis, 'crypto', {
        configurable: true,
        value: realCrypto,
      });
    }
  });

  it('saveTabs persists state changes (trailing-edge debounce)', async () => {
    vi.useFakeTimers();
    try {
      const mod = await import('./tabStore');
      mod.useTabStore.getState().addTab('Persisted');
      // Debounced save fires after 250ms.
      vi.advanceTimersByTime(300);
      const raw = localStorage.getItem(STORAGE_KEY);
      expect(raw).toBeTruthy();
      const data = JSON.parse(raw!);
      expect(data.tabs.some((t: { name: string }) => t.name === 'Persisted')).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });

  it('saveTabs strips SECRET values (params + preset internalParams) before persisting', async () => {
    vi.useFakeTimers();
    try {
      const mod = await import('./tabStore');
      const llmDef = {
        node_name: 'LLMChat', category: 'LLM', description: '', inputs: [], outputs: [],
        params: [{ name: 'openai_api_key', param_type: 'secret', default: '', description: '', options: [], min_value: null, max_value: null }],
      };
      const presetDef = {
        preset_name: 'X', category: 'c', description: '', tags: [],
        nodes: [{ id: 'inner', type: 'LLMChat', params: {} }], edges: [],
        exposed_inputs: [], exposed_outputs: [],
        exposed_params: [{
          internal_node: 'inner', param_name: 'openai_api_key', display_name: 'k', group: 'g',
          param_def: { name: 'openai_api_key', param_type: 'secret', default: '', description: '', options: [], min_value: null, max_value: null },
        }],
      };
      mod.useTabStore.getState().setNodes([
        { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'LLM', type: 'LLMChat', params: { openai_api_key: 'sk-in-storage', model: 'gpt-5.2' }, definition: llmDef } },
        { id: 'plain', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'D', type: 'Dataset', params: { p: 1 } } },
        { id: 'p1', type: 'presetNode', position: { x: 0, y: 0 }, data: { label: 'P', type: 'preset:X', params: {}, isPreset: true, presetDefinition: presetDef, internalParams: { inner: { openai_api_key: 'sk-preset-in-storage' } } } },
      ] as any);
      vi.advanceTimersByTime(300);
      const raw = localStorage.getItem(STORAGE_KEY)!;
      expect(raw).toBeTruthy();
      // No secret value survives to disk.
      expect(raw).not.toContain('sk-in-storage');
      expect(raw).not.toContain('sk-preset-in-storage');
      // Structure preserved: blanked keys present but empty, non-secrets kept.
      const persistedNodes = JSON.parse(raw).tabs[0].nodes;
      const llm = persistedNodes.find((n: any) => n.id === 'n1');
      expect(llm.data.params.openai_api_key).toBe('');
      expect(llm.data.params.model).toBe('gpt-5.2');
      const plain = persistedNodes.find((n: any) => n.id === 'plain');
      expect(plain.data.params).toEqual({ p: 1 });
      const preset = persistedNodes.find((n: any) => n.id === 'p1');
      expect(preset.data.internalParams.inner.openai_api_key).toBe('');
    } finally {
      vi.useRealTimers();
    }
  });

  it('saveTabs strips SECRET values inside subgraph definitions too (core#137)', async () => {
    vi.useFakeTimers();
    try {
      const mod = await import('./tabStore');
      const { useNodeDefStore } = await import('./nodeDefStore');
      const llmDef = {
        node_name: 'LLMChat', category: 'LLM', description: '', inputs: [], outputs: [],
        params: [{ name: 'openai_api_key', param_type: 'secret', default: '', description: '', options: [], min_value: null, max_value: null }],
      };
      // A node serialized INTO a definition keeps only its `type` string, so
      // the strip resolves the type through the registry rather than through
      // an attached `data.definition` the way the top-level strip does.
      useNodeDefStore.setState({ definitions: [llmDef] } as never);
      mod.useTabStore.getState().setNodes([
        { id: 'blk', type: 'subgraphNode', position: { x: 0, y: 0 }, data: { label: 'B', type: 'subgraph:d1', params: {} } },
      ] as any);
      mod.useTabStore.getState().setSubgraphs([
        {
          id: 'd1', name: 'Block', description: '',
          nodes: [{ id: 'inner', type: 'LLMChat', position: { x: 0, y: 0 }, data: { params: { openai_api_key: 'sk-in-block-storage', model: 'gpt-5.2' } } }],
          edges: [],
          interface: { inputs: [], outputs: [], triggerTargets: [] },
        },
      ] as any);
      vi.advanceTimersByTime(300);
      const raw = localStorage.getItem(STORAGE_KEY)!;
      expect(raw).toBeTruthy();
      expect(raw).not.toContain('sk-in-block-storage');
      const persisted = JSON.parse(raw).tabs[0].subgraphs[0];
      expect(persisted.nodes[0].data.params.openai_api_key).toBe('');
      expect(persisted.nodes[0].data.params.model).toBe('gpt-5.2');
    } finally {
      vi.useRealTimers();
    }
  });

  it('saveTabs collapses a burst of changes into a single trailing save', async () => {
    vi.useFakeTimers();
    try {
      const mod = await import('./tabStore');
      const setSpy = vi.spyOn(Storage.prototype, 'setItem');
      mod.useTabStore.getState().addTab('A');
      mod.useTabStore.getState().addTab('B');
      mod.useTabStore.getState().addTab('C');
      // Before the debounce window elapses, no save yet.
      expect(setSpy).not.toHaveBeenCalled();
      vi.advanceTimersByTime(300);
      // Exactly one trailing-edge save for the burst.
      expect(setSpy).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('saveTabs surfaces a quota toast at most once per minute when setItem keeps throwing', async () => {
    vi.useFakeTimers();
    try {
      const mod = await import('./tabStore');
      const toastMod = await import('./toastStore');
      const toastSpy = vi.spyOn(toastMod.useToastStore.getState(), 'addToast');
      vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
        throw new DOMException('quota', 'QuotaExceededError');
      });

      // First failing save → throttle window opens, one toast fires.
      mod.useTabStore.getState().addTab('Boom1');
      vi.advanceTimersByTime(300);
      expect(toastSpy).toHaveBeenCalledTimes(1);
      expect(toastSpy).toHaveBeenLastCalledWith(expect.any(String), 'error');

      // Second failing save shortly after (well within 60s) → throttle false
      // branch: no additional toast.
      mod.useTabStore.getState().addTab('Boom2');
      vi.advanceTimersByTime(300);
      expect(toastSpy).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('saveTabs swallows errors when the toast/i18n layer also throws', async () => {
    vi.useFakeTimers();
    try {
      const mod = await import('./tabStore');
      const i18nMod = await import('../i18n');
      // localStorage write fails...
      vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
        throw new DOMException('quota', 'QuotaExceededError');
      });
      // ...and the i18n lookup inside the catch ALSO throws → inner catch path.
      vi.spyOn(i18nMod.useI18n.getState(), 't').mockImplementation(() => {
        throw new Error('i18n exploded');
      });
      mod.useTabStore.getState().addTab('Boom');
      expect(() => vi.advanceTimersByTime(300)).not.toThrow();
    } finally {
      vi.useRealTimers();
    }
  });
});


// ── Bypass (core#128) ────────────────────────────────────────────────────────

/** A canvas node, defaulting to the plain (bypassable) card type. */
function bnode(id: string, over: Record<string, any> = {}) {
  const { data, ...rest } = over;
  return {
    id,
    type: 'baseNode',
    position: { x: 0, y: 0 },
    selected: false,
    data: { label: id, type: 'Dropout', params: {}, ...data },
    ...rest,
  } as any;
}

describe('bypass', () => {
  beforeEach(resetToSingleTab);

  it('toggleNodeBypass flips the flag and back again', () => {
    store().setNodes([bnode('n1')]);
    store().toggleNodeBypass('n1');
    expect(activeTab().nodes[0].data.bypassed).toBe(true);
    store().toggleNodeBypass('n1');
    expect(activeTab().nodes[0].data.bypassed).toBe(false);
  });

  it('toggleNodeBypass is one undo step', () => {
    store().setNodes([bnode('n1')]);
    store().toggleNodeBypass('n1');
    store().undo();
    expect(activeTab().nodes[0].data.bypassed).toBeUndefined();
  });

  it('toggleNodeBypass marks the node dirty so the next run redoes downstream', () => {
    store().setNodes([bnode('n1')]);
    store().setEdges([{ id: 'e1', source: 'n1', target: 'n2' }] as any);
    store().toggleNodeBypass('n1');
    // n1 itself is filtered out of the hint once muted; its consumer is not.
    expect(store().getDirtyWithDownstream()).toEqual(['n2']);
  });

  it('toggleNodeBypass refuses notes, Start nodes and presets', () => {
    store().setNodes([
      bnode('note', { type: 'noteNode' }),
      bnode('start', { type: 'start' }),
      bnode('preset', { type: 'presetNode' }),
    ]);
    for (const id of ['note', 'start', 'preset']) store().toggleNodeBypass(id);
    expect(activeTab().nodes.every((n) => n.data.bypassed === undefined)).toBe(true);
  });

  it('toggleNodeBypass ignores an id that is not on the canvas', () => {
    store().setNodes([bnode('n1')]);
    store().toggleNodeBypass('ghost');
    expect(activeTab().nodes[0].data.bypassed).toBeUndefined();
  });

  it('toggleBypassForSelection mutes every selected node and reports true', () => {
    store().setNodes([
      bnode('n1', { selected: true }),
      bnode('n2', { selected: true }),
      bnode('n3'),
    ]);
    expect(store().toggleBypassForSelection()).toBe(true);
    const byId = Object.fromEntries(activeTab().nodes.map((n) => [n.id, n.data.bypassed]));
    expect(byId).toEqual({ n1: true, n2: true, n3: undefined });
  });

  it('toggleBypassForSelection makes a mixed selection uniformly bypassed', () => {
    store().setNodes([
      bnode('n1', { selected: true, data: { bypassed: true } }),
      bnode('n2', { selected: true }),
    ]);
    store().toggleBypassForSelection();
    expect(activeTab().nodes.map((n) => n.data.bypassed)).toEqual([true, true]);
    // A second press now un-mutes the whole (uniform) selection.
    store().toggleBypassForSelection();
    expect(activeTab().nodes.map((n) => n.data.bypassed)).toEqual([false, false]);
  });

  it('toggleBypassForSelection falls back to selectedNodeId when nothing is flagged', () => {
    store().setNodes([bnode('n1'), bnode('n2')]);
    store().setSelectedNodeId('n2');
    expect(store().toggleBypassForSelection()).toBe(true);
    expect(activeTab().nodes.map((n) => n.data.bypassed)).toEqual([undefined, true]);
  });

  it('toggleBypassForSelection reports false with nothing selected', () => {
    store().setNodes([bnode('n1')]);
    expect(store().toggleBypassForSelection()).toBe(false);
    expect(activeTab().nodes[0].data.bypassed).toBeUndefined();
  });

  it('toggleBypassForSelection reports false when only un-bypassable nodes are selected', () => {
    store().setNodes([bnode('note', { type: 'noteNode', selected: true })]);
    expect(store().toggleBypassForSelection()).toBe(false);
  });

  it('getSerializedGraph writes the flag only for muted nodes', () => {
    store().setNodes([
      bnode('n1', { data: { bypassed: true } }),
      bnode('n2'),
    ]);
    const [muted, plain] = store().getSerializedGraph().nodes;
    expect(muted.data.bypassed).toBe(true);
    expect('bypassed' in plain.data).toBe(false);
  });

  it('getSerializedGraph drops the flag again once the node is un-muted', () => {
    store().setNodes([bnode('n1', { data: { bypassed: true } })]);
    store().toggleNodeBypass('n1');
    expect('bypassed' in store().getSerializedGraph().nodes[0].data).toBe(false);
  });

  it('getDirtyWithDownstream propagates through a muted node but never names it', () => {
    store().setNodes([
      bnode('a'),
      bnode('b', { data: { bypassed: true } }),
      bnode('c'),
    ]);
    store().setEdges([
      { id: 'e1', source: 'a', target: 'b' },
      { id: 'e2', source: 'b', target: 'c' },
    ] as any);
    store().markDirty('a');
    expect(store().getDirtyWithDownstream().sort()).toEqual(['a', 'c']);
  });
});

// ── insertGraph (core#128 template insertion) ────────────────────────────────

describe('insertGraph', () => {
  beforeEach(resetToSingleTab);

  it('is a no-op for an empty template', () => {
    store().setNodes([bnode('n1')]);
    store().insertGraph([], []);
    expect(activeTab().nodes).toHaveLength(1);
    expect(activeTab().undoStack).toHaveLength(0);
  });

  it('remaps ids so a template can never clobber an existing node', () => {
    // Both graphs use the id "n1" — the exact collision a template saved by
    // someone else routinely causes.
    store().setNodes([bnode('n1', { data: { label: 'MINE' } })]);
    store().insertGraph(
      [bnode('n1', { data: { label: 'TEMPLATE' } }), bnode('n2')],
      [{ id: 'e1', source: 'n1', target: 'n2' }] as any,
    );

    const tab = activeTab();
    expect(tab.nodes).toHaveLength(3);
    // The original survives untouched, under its own id.
    expect(tab.nodes.find((n) => n.id === 'n1')!.data.label).toBe('MINE');
    // Neither inserted node kept a template id.
    const inserted = tab.nodes.filter((n) => n.selected);
    expect(inserted).toHaveLength(2);
    expect(inserted.map((n) => n.id)).not.toContain('n1');
    expect(inserted.map((n) => n.id)).not.toContain('n2');
    expect(new Set(tab.nodes.map((n) => n.id)).size).toBe(3);
    // ...and the inserted edge follows the remap rather than pointing at the
    // user's own "n1".
    expect(tab.edges).toHaveLength(1);
    const insertedIds = new Set(inserted.map((n) => n.id));
    expect(insertedIds.has(tab.edges[0].source)).toBe(true);
    expect(insertedIds.has(tab.edges[0].target)).toBe(true);
    expect(tab.edges[0].id).not.toBe('e1');
  });

  it('places the template clear of the existing graph and selects only it', () => {
    store().setNodes([bnode('n1', { position: { x: 10, y: 0 } })]);
    store().insertGraph([bnode('t1', { position: { x: 500, y: 500 } })], []);

    const tab = activeTab();
    const [original, insertedNode] = tab.nodes;
    expect(original.selected).toBe(false);
    expect(insertedNode.selected).toBe(true);
    // Left-aligned with the existing graph, below its lowest point.
    expect(insertedNode.position.x).toBe(10);
    expect(insertedNode.position.y).toBeGreaterThan(0);
  });

  it('drops a template into an empty canvas at its own coordinates', () => {
    store().insertGraph([bnode('t1', { position: { x: 40, y: 60 } })], []);
    expect(activeTab().nodes[0].position).toEqual({ x: 40, y: 60 });
  });

  it('undoes the whole insertion in one step', () => {
    store().setNodes([bnode('n1')]);
    store().insertGraph([bnode('t1'), bnode('t2')], [{ id: 'e1', source: 't1', target: 't2' }] as any);
    expect(activeTab().nodes).toHaveLength(3);
    store().undo();
    expect(activeTab().nodes).toHaveLength(1);
    expect(activeTab().edges).toHaveLength(0);
  });

  it('drops an edge whose endpoint the template did not ship', () => {
    store().setNodes([bnode('n1')]);
    store().insertGraph(
      [bnode('t1')],
      [{ id: 'e1', source: 't1', target: 'nowhere' }] as any,
    );
    expect(activeTab().edges).toHaveLength(0);
  });

  it('remaps a note binding inside the template and clears a dangling one', () => {
    store().insertGraph(
      [
        bnode('t1'),
        bnode('bound', { type: 'noteNode', data: { boundToNodeId: 't1', boundOffset: { x: 1, y: 2 } } }),
        bnode('orphan', { type: 'noteNode', data: { boundToNodeId: 'gone', boundOffset: { x: 3, y: 4 } } }),
      ],
      [],
    );
    const tab = activeTab();
    const t1 = tab.nodes[0];
    const bound = tab.nodes[1];
    const orphan = tab.nodes[2];
    expect(bound.data.boundToNodeId).toBe(t1.id);
    expect(bound.data.boundOffset).toEqual({ x: 1, y: 2 });
    expect(orphan.data.boundToNodeId).toBeNull();
    expect(orphan.data.boundOffset).toBeNull();
  });

  it('resets execution state on inserted nodes', () => {
    store().insertGraph(
      [bnode('t1', { data: { executionStatus: 'error', error: 'stale' } })],
      [],
    );
    const inserted = activeTab().nodes[0];
    expect(inserted.data.executionStatus).toBe('idle');
    expect(inserted.data.error).toBeUndefined();
  });
});


describe('bypass — graph I/O contract nodes (core#128 review)', () => {
  beforeEach(resetToSingleTab);

  it('toggleNodeBypass refuses GraphInput and GraphOutput', () => {
    store().setNodes([
      bnode('gi', { data: { label: 'gi', type: 'GraphInput', params: {} } }),
      bnode('go', { data: { label: 'go', type: 'GraphOutput', params: {} } }),
    ]);
    store().toggleNodeBypass('gi');
    store().toggleNodeBypass('go');
    expect(activeTab().nodes.every((n) => n.data.bypassed === undefined)).toBe(true);
  });

  it('toggleBypassForSelection skips them and falls through', () => {
    store().setNodes([
      bnode('go', { selected: true, data: { label: 'go', type: 'GraphOutput', params: {} } }),
    ]);
    // false is what lets Ctrl+B reach the sidebar instead of doing nothing.
    expect(store().toggleBypassForSelection()).toBe(false);
    expect(activeTab().nodes[0].data.bypassed).toBeUndefined();
  });

  it('still bypasses an ordinary node sharing the selection with a contract node', () => {
    store().setNodes([
      bnode('go', { selected: true, data: { label: 'go', type: 'GraphOutput', params: {} } }),
      bnode('drop', { selected: true }),
    ]);
    expect(store().toggleBypassForSelection()).toBe(true);
    const byId = Object.fromEntries(activeTab().nodes.map((n) => [n.id, n.data.bypassed]));
    expect(byId).toEqual({ go: undefined, drop: true });
  });
});


describe('insertGraph — viewport fit (core#128 review)', () => {
  beforeEach(() => {
    resetToSingleTab();
    useUIStore.setState({ layoutFitRequest: null });
  });

  it('asks the canvas to fit the inserted block, not the whole graph', () => {
    // Without this, a template dropped below a graph the user has panned
    // away from lands entirely off-screen and the insert looks like a no-op.
    store().setNodes([bnode('n1', { position: { x: 0, y: 0 } })]);
    store().insertGraph([bnode('t1', { position: { x: 900, y: 900 } })], []);

    const request = useUIStore.getState().layoutFitRequest;
    expect(request).not.toBeNull();
    const inserted = activeTab().nodes.find((n) => n.selected)!;
    expect(request!.bounds.x).toBe(inserted.position.x);
    expect(request!.bounds.y).toBe(inserted.position.y);
  });

  it('does not request a fit for an empty template', () => {
    store().insertGraph([], []);
    expect(useUIStore.getState().layoutFitRequest).toBeNull();
  });
});

// ── Reproducibility settings (core#134) ──────────────────────────────────

describe('seed and deterministic', () => {
  beforeEach(resetToSingleTab);

  it('a new tab is unseeded and non-deterministic', () => {
    store().addTab('fresh');
    const tab = store().getActiveTab();
    expect(tab.seed).toBeNull();
    expect(tab.deterministic).toBe(false);
  });

  it('setSeed stores an integer and null clears it', () => {
    store().setSeed(1234);
    expect(store().getActiveTab().seed).toBe(1234);
    store().setSeed(null);
    expect(store().getActiveTab().seed).toBeNull();
  });

  it('setSeed keeps 0, which is a real seed and not "unset"', () => {
    store().setSeed(0);
    expect(store().getActiveTab().seed).toBe(0);
  });

  it('setSeed truncates a fractional value rather than storing it', () => {
    // The number input can emit "12.5" while someone is typing; a run needs
    // an integer, and truncating here means no call site has to remember.
    store().setSeed(12.9);
    expect(store().getActiveTab().seed).toBe(12);
  });

  it('setSeed reads NaN as "no seed"', () => {
    store().setSeed(1234);
    store().setSeed(Number('not a number'));
    expect(store().getActiveTab().seed).toBeNull();
  });

  it('toggleDeterministic flips the flag', () => {
    store().toggleDeterministic();
    expect(store().getActiveTab().deterministic).toBe(true);
    store().toggleDeterministic();
    expect(store().getActiveTab().deterministic).toBe(false);
  });
});
