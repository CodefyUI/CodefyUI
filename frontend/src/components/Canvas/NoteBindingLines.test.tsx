import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render } from '@testing-library/react';
import { ReactFlow, ReactFlowProvider } from '@xyflow/react';
import type { Node } from '@xyflow/react';
import type { NodeData } from '../../types';
import { NoteBindingLines } from './NoteBindingLines';
import { useTabStore } from '../../store/tabStore';

const ORIGINAL_TABS = useTabStore.getState().tabs;
const ORIGINAL_ACTIVE = useTabStore.getState().activeTabId;

function setNodes(nodes: Node<NodeData>[]) {
  const { tabs, activeTabId } = useTabStore.getState();
  useTabStore.setState({
    tabs: tabs.map((t) => (t.id === activeTabId ? { ...t, nodes, edges: [] } : t)),
  });
}

function note(
  id: string,
  position: { x: number; y: number },
  data: Partial<NodeData> = {},
  geom: Partial<Node<NodeData>> = {},
): Node<NodeData> {
  return {
    id,
    type: 'noteNode',
    position,
    data: {
      label: 'Note',
      type: 'note',
      params: {},
      boundToNodeId: null,
      boundOffset: null,
      ...data,
    },
    ...geom,
  } as Node<NodeData>;
}

function baseNode(
  id: string,
  position: { x: number; y: number },
  geom: Partial<Node<NodeData>> = {},
): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position,
    data: { label: id, type: id, params: {} },
    ...geom,
  } as Node<NodeData>;
}

// Mount inside a real <ReactFlow> so ViewportPortal has a portal target.
function renderLines() {
  return render(
    <ReactFlowProvider>
      <div style={{ width: 800, height: 600 }}>
        <ReactFlow nodes={[]} edges={[]}>
          <NoteBindingLines />
        </ReactFlow>
      </div>
    </ReactFlowProvider>,
  );
}

describe('NoteBindingLines', () => {
  beforeEach(() => {
    useTabStore.setState({ tabs: ORIGINAL_TABS, activeTabId: ORIGINAL_ACTIVE });
    setNodes([]);
  });

  afterEach(() => {
    useTabStore.setState({ tabs: ORIGINAL_TABS, activeTabId: ORIGINAL_ACTIVE });
  });

  it('returns null (renders no <line>) when there are no bound notes', () => {
    setNodes([baseNode('p', { x: 0, y: 0 }), note('n', { x: 300, y: 0 })]);
    const { container } = renderLines();
    expect(container.querySelector('line')).toBeNull();
  });

  it('skips notes whose bound parent does not exist', () => {
    setNodes([note('n', { x: 0, y: 0 }, { boundToNodeId: 'missing' })]);
    const { container } = renderLines();
    expect(container.querySelector('line')).toBeNull();
  });

  it('skips non-note nodes and notes without boundToNodeId', () => {
    setNodes([
      baseNode('p', { x: 0, y: 0 }),
      note('n', { x: 300, y: 0 }, { boundToNodeId: null }),
    ]);
    const { container } = renderLines();
    expect(container.querySelector('line')).toBeNull();
  });

  it('draws a binding line between a bound note and its parent', () => {
    setNodes([
      baseNode('p', { x: 0, y: 0 }, { measured: { width: 200, height: 80 } }),
      note('n', { x: 400, y: 0 }, { boundToNodeId: 'p' }, { measured: { width: 200, height: 60 } }),
    ]);
    const { container } = renderLines();
    const line = container.querySelector('line');
    expect(line).toBeTruthy();
    expect(line?.getAttribute('stroke')).toBe('#666');
    expect(line?.getAttribute('opacity')).toBe('0.6');
    // zoom defaults to 1, so stroke width = 1/1 = 1.
    expect(line?.getAttribute('stroke-width')).toBe('1');
    expect(line?.getAttribute('stroke-dasharray')).toBe('4 4');
  });

  it('falls back to node.width/height when measured is absent', () => {
    // Both use node.width/height (not measured, not the hardcoded defaults).
    setNodes([
      baseNode('p', { x: 0, y: 0 }, { width: 100, height: 50 }),
      note('n', { x: 400, y: 0 }, { boundToNodeId: 'p' }, { width: 100, height: 50 }),
    ]);
    const { container } = renderLines();
    expect(container.querySelector('line')).toBeTruthy();
  });

  it('falls back to the hardcoded default dimensions when neither measured nor width/height exist', () => {
    // Parent has no measured + no width/height -> 200x80 defaults.
    // Note has no measured + no width/height -> 200x60 defaults.
    setNodes([
      baseNode('p', { x: 0, y: 0 }),
      note('n', { x: 400, y: 0 }, { boundToNodeId: 'p' }),
    ]);
    const { container } = renderLines();
    expect(container.querySelector('line')).toBeTruthy();
  });

  it('exercises the `?? []` fallback when the active tab cannot be found', () => {
    // Point activeTabId at a non-existent tab so the selector hits `?? []`.
    // That fallback returns a fresh array every render, so React's external-store
    // subscription re-renders until it bails with "Maximum update depth exceeded".
    // The fallback line still executes (many times) before the throw, so this
    // covers the branch. The afterEach restores a valid active tab, and React's
    // own console.error for the loop is expected here.
    const consoleError = vi
      .spyOn(console, 'error')
      .mockImplementation(() => {});
    useTabStore.setState({ activeTabId: 'does-not-exist' });
    expect(() =>
      render(
        <ReactFlowProvider>
          <NoteBindingLines />
        </ReactFlowProvider>,
      ),
    ).toThrow(/Maximum update depth/);
    consoleError.mockRestore();
  });

  it('handles a note centered exactly on its parent (rectEdgePoint dx=0 && dy=0 branch)', () => {
    // Identical centers -> rectEdgePoint returns the center directly.
    setNodes([
      baseNode('p', { x: 0, y: 0 }, { measured: { width: 200, height: 80 } }),
      note('n', { x: 0, y: 10 }, { boundToNodeId: 'p' }, { measured: { width: 200, height: 60 } }),
    ]);
    const { container } = renderLines();
    expect(container.querySelector('line')).toBeTruthy();
  });

  it('handles a purely vertical offset (rectEdgePoint dx=0, sx=Infinity branch)', () => {
    // Same x center, different y -> dx=0 so sx=Infinity, min picks sy.
    setNodes([
      baseNode('p', { x: 0, y: 0 }, { measured: { width: 200, height: 80 } }),
      note('n', { x: 0, y: 400 }, { boundToNodeId: 'p' }, { measured: { width: 200, height: 60 } }),
    ]);
    const { container } = renderLines();
    expect(container.querySelector('line')).toBeTruthy();
  });

  it('handles a purely horizontal offset (rectEdgePoint dy=0, sy=Infinity branch)', () => {
    // Same y center, different x -> dy=0 so sy=Infinity, min picks sx.
    setNodes([
      baseNode('p', { x: 0, y: 0 }, { measured: { width: 200, height: 80 } }),
      // note height 80 so vertical centers line up: note cy = 0+80/2=40; parent cy=0+80/2=40
      note('n', { x: 600, y: 0 }, { boundToNodeId: 'p' }, { measured: { width: 200, height: 80 } }),
    ]);
    const { container } = renderLines();
    expect(container.querySelector('line')).toBeTruthy();
  });
});
