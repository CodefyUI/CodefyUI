import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useDragAndDrop } from './useDragAndDrop';
import { useTabStore } from '../store/tabStore';
import { useNodeDefStore } from '../store/nodeDefStore';

// Mock @xyflow/react so useReactFlow returns a controllable screenToFlowPosition.
// Identity transform keeps the asserted positions simple.
const screenToFlowPosition = vi.fn((p: { x: number; y: number }) => ({
  x: p.x,
  y: p.y,
}));
vi.mock('@xyflow/react', () => ({
  useReactFlow: () => ({ screenToFlowPosition }),
}));

const presetA = {
  preset_name: 'MyPreset',
  category: 'Presets',
  description: 'p',
  nodes: [],
  exposed_inputs: [],
  exposed_outputs: [],
} as any;

const defA = {
  node_name: 'Dataset',
  category: 'Data',
  description: 'd',
  inputs: [],
  outputs: [],
  params: [],
} as any;

/** Build a fake React.DragEvent with controllable getData results. */
function makeDragEvent(data: Record<string, string> = {}) {
  const dataTransfer = {
    getData: vi.fn((key: string) => data[key] ?? ''),
    setData: vi.fn(),
    dropEffect: '',
  };
  return {
    preventDefault: vi.fn(),
    clientX: 10,
    clientY: 20,
    dataTransfer,
  } as unknown as React.DragEvent;
}

beforeEach(() => {
  screenToFlowPosition.mockClear();
  // Seed node-def store with one preset + one definition.
  useNodeDefStore.setState({
    definitions: [defA],
    presets: [presetA],
    loading: false,
    error: null,
    categorized: {},
    presetCategorized: {},
  });
  // Reset tab store to a single fresh tab so addNode/addPresetNode operate.
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
  useTabStore.getState().addTab('test');
});

afterEach(() => {
  vi.restoreAllMocks();
});

function activeNodes() {
  return useTabStore.getState().getActiveTab().nodes;
}

describe('useDragAndDrop - onDragOver', () => {
  it('prevents default and sets dropEffect to move', () => {
    const { result } = renderHook(() => useDragAndDrop());
    const event = makeDragEvent();
    result.current.onDragOver(event);

    expect(event.preventDefault).toHaveBeenCalledTimes(1);
    expect(event.dataTransfer.dropEffect).toBe('move');
  });
});

describe('useDragAndDrop - onDrop preset branch', () => {
  it('adds a preset node when a matching preset is dropped', () => {
    const { result } = renderHook(() => useDragAndDrop());
    const event = makeDragEvent({ 'application/codefyui-preset': 'MyPreset' });

    result.current.onDrop(event);

    expect(event.preventDefault).toHaveBeenCalledTimes(1);
    expect(screenToFlowPosition).toHaveBeenCalledWith({ x: 10, y: 20 });
    const nodes = activeNodes();
    expect(nodes).toHaveLength(1);
    expect(nodes[0].data.label).toBe('MyPreset');
    expect(nodes[0].type).toBe('presetNode');
  });

  it('does NOT add a node when the preset name has no match (and returns early)', () => {
    const { result } = renderHook(() => useDragAndDrop());
    const event = makeDragEvent({ 'application/codefyui-preset': 'Unknown' });

    result.current.onDrop(event);

    // Early return after the preset branch: the node getData is never consulted.
    expect(event.dataTransfer.getData).toHaveBeenCalledTimes(1);
    expect(activeNodes()).toHaveLength(0);
  });
});

describe('useDragAndDrop - onDrop node branch', () => {
  it('adds a node when a matching node type is dropped', () => {
    const { result } = renderHook(() => useDragAndDrop());
    const event = makeDragEvent({ 'application/codefyui-node': 'Dataset' });

    result.current.onDrop(event);

    const nodes = activeNodes();
    expect(nodes).toHaveLength(1);
    expect(nodes[0].data.type).toBe('Dataset');
  });

  it('does NOT add a node when the node type has no matching definition', () => {
    const { result } = renderHook(() => useDragAndDrop());
    const event = makeDragEvent({ 'application/codefyui-node': 'Nope' });

    result.current.onDrop(event);
    expect(activeNodes()).toHaveLength(0);
  });

  it('does nothing when neither preset nor node data is present', () => {
    const { result } = renderHook(() => useDragAndDrop());
    const event = makeDragEvent(); // both getData return ''

    result.current.onDrop(event);

    expect(event.preventDefault).toHaveBeenCalledTimes(1);
    // Both keys consulted, but no node added.
    expect(event.dataTransfer.getData).toHaveBeenCalledTimes(2);
    expect(activeNodes()).toHaveLength(0);
  });
});
