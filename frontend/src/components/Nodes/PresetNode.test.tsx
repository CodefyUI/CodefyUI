import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ReactFlow, ReactFlowProvider, type Edge, type Node, type NodeTypes } from '@xyflow/react';
import { renderWithFlow } from '../../test/utils';
import { useI18n } from '../../i18n';
import { useUIStore } from '../../store/uiStore';
import { useTabStore } from '../../store/tabStore';
import type { NodeDefinition, NodeData, PresetDefinition } from '../../types';
import PresetNode from './PresetNode';

function makeDef(overrides: Partial<NodeDefinition> = {}): NodeDefinition {
  return {
    node_name: 'MyPreset',
    category: 'Transformer',
    description: '',
    inputs: [{ name: 'x', data_type: 'TENSOR', description: 'in', optional: false }],
    outputs: [{ name: 'y', data_type: 'TENSOR', description: 'out', optional: false }],
    params: [],
    ...overrides,
  };
}

function makePreset(nodeCount = 2): PresetDefinition {
  return {
    preset_name: 'MyPreset',
    category: 'Transformer',
    description: '',
    tags: [],
    nodes: Array.from({ length: nodeCount }, (_, i) => ({ id: `n${i}`, type: 'Linear', params: {} })),
    edges: [],
    exposed_inputs: [],
    exposed_outputs: [],
    exposed_params: [],
  };
}

function presetData(overrides: Partial<NodeData> = {}): NodeData {
  return {
    label: 'Preset Node',
    type: 'preset:MyPreset',
    params: {},
    definition: makeDef(),
    presetDefinition: makePreset(2),
    isPreset: true,
    executionStatus: 'idle',
    ...overrides,
  };
}

const flowProps = {
  zIndex: 0,
  isConnectable: true,
  positionAbsoluteX: 0,
  positionAbsoluteY: 0,
  dragging: false,
  draggable: false,
  selectable: true,
  deletable: true,
} as const;

function renderPreset(data: NodeData, opts: { id?: string; selected?: boolean } = {}) {
  const { id = 'p1', selected = false } = opts;
  return renderWithFlow(
    <PresetNode id={id} type="presetNode" data={data} selected={selected} {...flowProps} />,
  );
}

const nodeTypes: NodeTypes = {
  presetNode: (p) => <PresetNode {...(p as React.ComponentProps<typeof PresetNode>)} />,
};

function renderPresetWithEdges(data: NodeData, edges: Edge[], id = 'p1') {
  const nodes: Node[] = [{ id, type: 'presetNode', position: { x: 0, y: 0 }, data: data as never }];
  return render(
    <div style={{ width: 800, height: 600 }}>
      <ReactFlowProvider>
        <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} />
      </ReactFlowProvider>
    </div>,
  );
}

function resetStores() {
  useI18n.setState({ locale: 'en' });
  useUIStore.setState({ draggingSourceType: null });
  const id = 'tab-preset';
  useTabStore.setState((s) => ({
    activeTabId: id,
    tabs: [{ ...s.tabs[0], id, name: 'Tab', nodes: [], edges: [], presetModalNodeId: null }],
  }));
}

beforeEach(() => {
  resetStores();
});

describe('PresetNode', () => {
  it('renders label, badge, ports, and the inside-node count', () => {
    renderPreset(presetData());
    expect(screen.getByText('Preset Node')).toBeTruthy();
    expect(screen.getByText(useI18n.getState().t('preset.badge'))).toBeTruthy();
    expect(screen.getByText('x')).toBeTruthy();
    expect(screen.getByText('y')).toBeTruthy();
    expect(screen.getByText('2')).toBeTruthy(); // 2 nodes inside
    expect(screen.getByText(useI18n.getState().t('preset.nodesInside'))).toBeTruthy();
  });

  it('renders 0 when presetDefinition is missing', () => {
    renderPreset(presetData({ presetDefinition: undefined }));
    expect(screen.getByText('0')).toBeTruthy();
  });

  it('renders no ports when definition is missing', () => {
    renderPreset(presetData({ definition: undefined }));
    expect(screen.queryByText('x')).toBeNull();
    expect(screen.queryByText('y')).toBeNull();
  });

  it('renders the divider only when both inputs and outputs exist', () => {
    const { container, rerender } = renderPreset(presetData());
    expect(container.querySelector('[class*="portDivider"]')).toBeTruthy();
    rerender(
      <PresetNode
        id="p1"
        type="presetNode"
        data={presetData({ definition: makeDef({ inputs: [] }) })}
        selected={false}
        {...flowProps}
      />,
    );
    expect(container.querySelector('[class*="portDivider"]')).toBeNull();
  });

  // ── Border branches ──
  it('selected node gets white border + gold glow', () => {
    const { container } = renderPreset(presetData(), { selected: true });
    const node = container.querySelector('[class*="node"]') as HTMLElement;
    // jsdom normalizes the border shorthand color #ffffff → rgb(255, 255, 255).
    expect(node.style.border).toBe('2px solid rgb(255, 255, 255)');
    expect(node.style.boxShadow).toContain('rgba(212,160,23,0.3)');
  });

  it('unselected idle node uses the gold default border', () => {
    const { container } = renderPreset(presetData());
    const node = container.querySelector('[class*="node"]') as HTMLElement;
    // #6B5B00 → rgb(107, 91, 0)
    expect(node.style.border).toBe('2px solid rgb(107, 91, 0)');
    expect(node.style.boxShadow).toContain('rgba(0,0,0,0.4)');
  });

  it.each([
    ['running', 'rgb(255, 193, 7)'],
    ['completed', 'rgb(76, 175, 80)'],
    ['error', 'rgb(244, 67, 54)'],
    ['cached', 'rgb(33, 150, 243)'],
  ] as const)('uses the %s status border when unselected', (status, rgb) => {
    const { container } = renderPreset(
      presetData({ executionStatus: status, error: status === 'error' ? 'x' : undefined }),
    );
    const node = container.querySelector('[class*="node"]') as HTMLElement;
    expect(node.style.border).toBe(`2px solid ${rgb}`);
  });

  // ── Status footers ──
  it('renders the error footer', () => {
    renderPreset(presetData({ executionStatus: 'error', error: 'oops' }));
    expect(screen.getByText('Error: oops')).toBeTruthy();
  });

  it('does not render the error footer when error text is absent', () => {
    renderPreset(presetData({ executionStatus: 'error', error: undefined }));
    expect(screen.queryByText(/Error:/)).toBeNull();
  });

  it('renders the running footer', () => {
    renderPreset(presetData({ executionStatus: 'running' }));
    expect(screen.getByText('Running...')).toBeTruthy();
  });

  it('renders the completed footer', () => {
    renderPreset(presetData({ executionStatus: 'completed' }));
    expect(screen.getByText('Completed')).toBeTruthy();
  });

  it('renders the cached footer', () => {
    renderPreset(presetData({ executionStatus: 'cached' }));
    expect(screen.getByText('Cached')).toBeTruthy();
  });

  // ── Click / modal ──
  it('double-click opens the preset modal', () => {
    const { container } = renderPreset(presetData());
    fireEvent.click(container.querySelector('[class*="node"]') as HTMLElement, { detail: 2 });
    expect(useTabStore.getState().getActiveTab().presetModalNodeId).toBe('p1');
  });

  it('single-click does not open the preset modal', () => {
    const { container } = renderPreset(presetData());
    fireEvent.click(container.querySelector('[class*="node"]') as HTMLElement, { detail: 1 });
    expect(useTabStore.getState().getActiveTab().presetModalNodeId).toBeNull();
  });

  // ── Dragging / trigger branches ──
  it('applies triggerDropTarget while dragging a TRIGGER source', () => {
    useUIStore.setState({ draggingSourceType: 'TRIGGER' });
    const { container } = renderPreset(presetData());
    const node = container.querySelector('[class*="node"]') as HTMLElement;
    expect(node.className).toMatch(/triggerDropTarget/);
  });

  it('adds the entryPoint class when a trigger edge targets the node', async () => {
    renderPresetWithEdges(
      presetData(),
      [{ id: 'e1', source: 's', target: 'p1', data: { type: 'trigger' } } as Edge],
      'p1',
    );
    const node = await waitFor(() => {
      const n = [...document.querySelectorAll('div')].find((d) => /entryPoint/.test(d.className));
      if (!n) throw new Error('not rendered');
      return n;
    });
    expect(node).toBeTruthy();
  });

  it('does NOT add entryPoint for a non-trigger edge', async () => {
    renderPresetWithEdges(
      presetData({ label: 'NoTrig' }),
      [{ id: 'e1', source: 's', target: 'p1', data: { type: 'data' } } as Edge],
      'p1',
    );
    await waitFor(() => expect(screen.getByText('NoTrig')).toBeTruthy());
    expect([...document.querySelectorAll('div')].some((d) => /entryPoint/.test(d.className))).toBe(false);
  });
});
