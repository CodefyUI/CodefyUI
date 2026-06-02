import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { fireEvent, render, waitFor, type RenderResult } from '@testing-library/react';
import { ReactFlow, ReactFlowProvider } from '@xyflow/react';
import { SegmentBubble } from './SegmentBubble';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import type { Node as FlowNode, Edge } from '@xyflow/react';
import type { NodeData, SegmentGroup } from '../../types';

function makeNode(id: string, overrides: Partial<FlowNode<NodeData>> = {}): FlowNode<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x: 0, y: 0 },
    data: { label: id } as NodeData,
    ...overrides,
  } as FlowNode<NodeData>;
}

/**
 * `SegmentBubble` renders through `<ViewportPortal>`, whose target div only
 * exists when a real `<ReactFlow>` is mounted (a bare provider has no DOM node).
 * Mount one so the portal has somewhere to render.
 */
function renderBubble(): RenderResult {
  return render(
    <ReactFlowProvider>
      <div style={{ width: 800, height: 600 }}>
        <ReactFlow nodes={[]} edges={[]}>
          <SegmentBubble />
        </ReactFlow>
      </div>
    </ReactFlowProvider>,
  );
}

/** Reset the active tab to a known-empty state before each test. */
function resetTab(partial: Partial<{
  nodes: FlowNode<NodeData>[];
  edges: Edge[];
  segmentGroups: SegmentGroup[];
  activeSegment: SegmentGroup | null;
}> = {}) {
  const { tabs, activeTabId } = useTabStore.getState();
  useTabStore.setState({
    tabs: tabs.map((t) =>
      t.id === activeTabId
        ? {
            ...t,
            nodes: partial.nodes ?? [],
            edges: partial.edges ?? [],
            segmentGroups: partial.segmentGroups ?? [],
            activeSegment: partial.activeSegment ?? null,
          }
        : t,
    ),
  });
}

describe('SegmentBubble', () => {
  beforeEach(() => {
    useI18n.setState({ locale: 'en' });
    resetTab();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    resetTab();
  });

  it('renders no bubble svg when there are no segment groups', async () => {
    const { container } = renderBubble();
    // Give React Flow a tick to mount the portal target.
    await waitFor(() => expect(container.querySelector('.react-flow__viewport-portal')).not.toBeNull());
    expect(container.querySelector('rect[pointer-events="stroke"]')).toBeNull();
  });

  it('renders no bubble when the segment resolves to an empty node set (head not reachable to tail)', async () => {
    // head !== tail and no edges => computeSegmentNodes returns empty set
    resetTab({
      nodes: [makeNode('a'), makeNode('b')],
      edges: [],
      segmentGroups: [{ id: 'g1', headNodeId: 'a', tailNodeId: 'b' }],
    });
    const { container } = renderBubble();
    await waitFor(() => expect(container.querySelector('.react-flow__viewport-portal')).not.toBeNull());
    expect(container.querySelector('rect[pointer-events="stroke"]')).toBeNull();
  });

  it('renders no bubble when head node is missing', async () => {
    // head === tail keeps the set non-empty, but the node itself is absent
    resetTab({
      nodes: [],
      segmentGroups: [{ id: 'g1', headNodeId: 'ghost', tailNodeId: 'ghost' }],
    });
    const { container } = renderBubble();
    await waitFor(() => expect(container.querySelector('.react-flow__viewport-portal')).not.toBeNull());
    expect(container.querySelector('rect[pointer-events="stroke"]')).toBeNull();
  });

  it('renders a bubble with HEAD/TAIL badges for a single-node segment (head === tail)', async () => {
    resetTab({
      nodes: [makeNode('a', { position: { x: 10, y: 20 }, measured: { width: 120, height: 60 } })],
      segmentGroups: [{ id: 'g1', headNodeId: 'a', tailNodeId: 'a' }],
    });
    const { container, findByText } = renderBubble();
    expect(await findByText('HEAD')).toBeInTheDocument();
    expect(await findByText('TAIL')).toBeInTheDocument();
    const rect = container.querySelector('rect[pointer-events="stroke"]') as SVGRectElement;
    expect(rect).not.toBeNull();
    // inactive stroke colour
    expect(rect.getAttribute('stroke')).toBe('rgba(255, 140, 0, 0.6)');
  });

  it('uses the active stroke colour when the segment is the active one', async () => {
    const group: SegmentGroup = { id: 'g1', headNodeId: 'a', tailNodeId: 'a' };
    resetTab({
      nodes: [makeNode('a')],
      segmentGroups: [group],
      activeSegment: group,
    });
    const { container } = renderBubble();
    await waitFor(() => expect(container.querySelector('rect[pointer-events="stroke"]')).not.toBeNull());
    const rect = container.querySelector('rect[pointer-events="stroke"]') as SVGRectElement;
    expect(rect.getAttribute('stroke')).toBe('rgba(255, 149, 0, 0.95)');
  });

  it('falls back to width/height then defaults when measured size is absent', async () => {
    // node "a": no measured, but width/height set => uses n.width/n.height branch
    // node "b": no measured/width/height => default 200x80 branch (it is the tail)
    resetTab({
      nodes: [
        makeNode('a', { position: { x: 0, y: 0 }, width: 150, height: 70 }),
        makeNode('b', { position: { x: 300, y: 0 } }),
      ],
      edges: [{ id: 'e1', source: 'a', target: 'b' }],
      segmentGroups: [{ id: 'g1', headNodeId: 'a', tailNodeId: 'b' }],
    });
    const { container } = renderBubble();
    await waitFor(() => expect(container.querySelector('rect[pointer-events="stroke"]')).not.toBeNull());
    expect(container.querySelector('rect[pointer-events="stroke"]')).not.toBeNull();
  });

  it('computes the union bbox when a later node is fully inside an earlier one', async () => {
    // 'a' is the big enclosing box; 'b' sits entirely inside it. Iterating
    // boxes in order [a, b], b never extends max/min, exercising the false
    // branches of the maxX / maxY comparisons inside unionBBox.
    resetTab({
      nodes: [
        makeNode('a', { position: { x: 0, y: 0 }, measured: { width: 300, height: 300 } }),
        makeNode('b', { position: { x: 50, y: 50 }, measured: { width: 50, height: 50 } }),
      ],
      edges: [{ id: 'e1', source: 'a', target: 'b' }],
      segmentGroups: [{ id: 'g1', headNodeId: 'a', tailNodeId: 'b' }],
    });
    const { container } = renderBubble();
    await waitFor(() => expect(container.querySelector('rect[pointer-events="stroke"]')).not.toBeNull());
    const rect = container.querySelector('rect[pointer-events="stroke"]') as SVGRectElement;
    // union covers a's box (0,0..300,300), padded by BUBBLE_PAD(28) on each side.
    expect(rect.getAttribute('width')).toBe(String(300 + 28 * 2));
    expect(rect.getAttribute('height')).toBe(String(300 + 28 * 2));
  });

  it('clicking the rect border focuses the segment via setActiveSegment', async () => {
    const group: SegmentGroup = { id: 'g1', headNodeId: 'a', tailNodeId: 'a' };
    resetTab({ nodes: [makeNode('a')], segmentGroups: [group] });
    const setActiveSegment = vi.fn();
    useTabStore.setState({ setActiveSegment });

    const { container } = renderBubble();
    await waitFor(() => expect(container.querySelector('rect[pointer-events="stroke"]')).not.toBeNull());
    const rect = container.querySelector('rect[pointer-events="stroke"]') as SVGRectElement;
    fireEvent.click(rect);
    expect(setActiveSegment).toHaveBeenCalledWith(group);
  });

  it('clicking the × close button removes only that segment and stops propagation', async () => {
    const group: SegmentGroup = { id: 'g1', headNodeId: 'a', tailNodeId: 'a' };
    resetTab({ nodes: [makeNode('a')], segmentGroups: [group] });
    const removeSegmentGroup = vi.fn();
    const setActiveSegment = vi.fn();
    useTabStore.setState({ removeSegmentGroup, setActiveSegment });

    const { container } = renderBubble();
    await waitFor(() => expect(container.querySelector('g[pointer-events="all"]')).not.toBeNull());
    // The close button group carries pointer-events="all" and a <title>.
    const closeGroup = container.querySelector('g[pointer-events="all"]') as SVGGElement;
    fireEvent.click(closeGroup);
    expect(removeSegmentGroup).toHaveBeenCalledWith('g1');
    // stopPropagation means the rect's onClick (setActiveSegment) must NOT fire
    expect(setActiveSegment).not.toHaveBeenCalled();
  });

  it('renders the localized close-button title', async () => {
    const group: SegmentGroup = { id: 'g1', headNodeId: 'a', tailNodeId: 'a' };
    resetTab({ nodes: [makeNode('a')], segmentGroups: [group] });
    const { container } = renderBubble();
    await waitFor(() => expect(container.querySelector('title')).not.toBeNull());
    expect(container.querySelector('title')?.textContent).toBe('Remove this segment');
  });
});
