import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ReactFlow, ReactFlowProvider, type Edge, type Node, type NodeTypes } from '@xyflow/react';
import { renderWithFlow } from '../../test/utils';
import { useI18n } from '../../i18n';
import { useUIStore } from '../../store/uiStore';
import { useTabStore } from '../../store/tabStore';
import type { NodeDefinition, NodeData, PresetDefinition } from '../../types';
import { STATUS_COLORS } from '../../styles/theme';
import PresetNode from './PresetNode';

/**
 * jsdom/cssstyle normalizes ordinary color-valued CSS properties (like the
 * `border` shorthand) to `rgb(r, g, b)` on read, but leaves custom
 * properties and var() references untouched. Mirror that here so the status
 * border assertions can be computed from STATUS_COLORS instead of a
 * hand-copied literal.
 */
function hexToRgb(hex: string): string {
  const h = hex.replace('#', '');
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgb(${r}, ${g}, ${b})`;
}

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
  useUIStore.setState({ draggingSourceType: null, reconnectingHandle: null });
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
  it('selected node gets the text-primary border + gold glow', () => {
    // The selected border used to be a hardcoded '#ffffff'; it's now
    // var(--text-primary) (#f0f4f8), the same token BaseNode's selected
    // border uses. jsdom does not resolve var() references embedded in a
    // border shorthand, so the raw token string is what `.style` reports.
    const { container } = renderPreset(presetData(), { selected: true });
    const node = container.querySelector('[class*="node"]') as HTMLElement;
    expect(node.style.border).toBe('1px solid var(--text-primary)');
    // rgba(224,169,43,0.3) is --status-preset's own rgb() (#e0a92b) — no
    // glow token is paired with it, so PresetNode keeps this as a
    // hand-tuned literal at the hue (see PresetNode's own comment).
    expect(node.style.boxShadow).toContain('rgba(224, 169, 43, 0.3)');
  });

  it('unselected idle node uses the gold default border', () => {
    const { container } = renderPreset(presetData());
    const node = container.querySelector('[class*="node"]') as HTMLElement;
    // #6B5B00 → rgb(107, 91, 0). No canonical "dim preset" token exists yet
    // (see PresetNode's borderColor comment), so this stays a literal.
    expect(node.style.border).toBe('1px solid rgb(107, 91, 0)');
    // The idle drop shadow is now the shared var(--shadow) token rather
    // than a hardcoded rgba literal; jsdom does not resolve var().
    expect(node.style.boxShadow).toBe('var(--shadow)');
  });

  it.each([
    // Sourced from STATUS_COLORS (theme.test.ts pins these against
    // tokens.css) rather than hand-copied literals, so this can't silently
    // drift from the palette the way the old hardcoded rgb() values did.
    ['running', hexToRgb(STATUS_COLORS.running)],
    ['completed', hexToRgb(STATUS_COLORS.completed)],
    ['error', hexToRgb(STATUS_COLORS.error)],
    ['cached', hexToRgb(STATUS_COLORS.cached)],
    // core#122: a preset settles 'interrupted' when an internal node stopped
    // early — it must never roll up to the green 'completed'.
    ['interrupted', hexToRgb(STATUS_COLORS.interrupted)],
    // core#260: and 'skipped' when every internal node was passed over
    // because something upstream failed. Without its own branch the box
    // falls through to 'transparent' and looks like it was never reached.
    ['skipped', hexToRgb(STATUS_COLORS.skipped)],
  ] as const)('uses the %s status border when unselected', (status, rgb) => {
    const { container } = renderPreset(
      presetData({ executionStatus: status, error: status === 'error' ? 'x' : undefined }),
    );
    const node = container.querySelector('[class*="node"]') as HTMLElement;
    expect(node.style.border).toBe(`1px solid ${rgb}`);
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

  // core#260: the engine settles a preset as 'cached' when every internal
  // node was a cache hit, so this footer is the ONLY thing that tells a
  // student "this box did not actually run this time".
  it('renders the cached footer', () => {
    renderPreset(presetData({ executionStatus: 'cached' }));
    expect(screen.getByText('Cached')).toBeTruthy();
  });

  it('renders the skipped footer', () => {
    renderPreset(presetData({ executionStatus: 'skipped' }));
    expect(screen.getByText('Skipped')).toBeTruthy();
  });

  it('does not claim a cached preset completed', () => {
    renderPreset(presetData({ executionStatus: 'cached' }));
    expect(screen.queryByText('Completed')).toBeNull();
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

  // ── Reconnect detach indicator ──
  it('adds portDetaching only to the matching exposed input handle, keeping its inline styles', () => {
    useUIStore.setState({ reconnectingHandle: { nodeId: 'p1', handleId: 'x', type: 'target' } });
    const def = makeDef({
      inputs: [
        { name: 'x', data_type: 'TENSOR', description: '', optional: false },
        { name: 'x2', data_type: 'TENSOR', description: '', optional: false },
      ],
    });
    const { container } = renderPreset(presetData({ definition: def }), { id: 'p1' });
    const detaching = container.querySelectorAll('[class*="portDetaching"]');
    expect(detaching).toHaveLength(1);
    const handle = detaching[0] as HTMLElement;
    expect(handle.getAttribute('data-handleid')).toBe('x');
    // The class coexists with PresetNode's inline handle styles — the class's
    // !important border-color/box-shadow win the cascade over inline styles,
    // while the untouched inline properties (background, size) remain.
    expect(handle.style.background).not.toBe('');
    expect(handle.style.width).toBe('10px');
  });

  it('adds portDetaching to the matching exposed output handle', () => {
    useUIStore.setState({ reconnectingHandle: { nodeId: 'p1', handleId: 'y', type: 'source' } });
    const { container } = renderPreset(presetData(), { id: 'p1' });
    const detaching = container.querySelectorAll('[class*="portDetaching"]');
    expect(detaching).toHaveLength(1);
    expect((detaching[0] as HTMLElement).getAttribute('data-handleid')).toBe('y');
  });

  it('does not mark any handle when the reconnect concerns another node', () => {
    useUIStore.setState({ reconnectingHandle: { nodeId: 'other', handleId: 'x', type: 'target' } });
    const { container } = renderPreset(presetData(), { id: 'p1' });
    expect(container.querySelector('[class*="portDetaching"]')).toBeNull();
  });

  it('requires the handle type to match (same id, wrong type is not marked)', () => {
    useUIStore.setState({ reconnectingHandle: { nodeId: 'p1', handleId: 'x', type: 'source' } });
    const { container } = renderPreset(presetData(), { id: 'p1' });
    expect(container.querySelector('[class*="portDetaching"]')).toBeNull();
  });

  it('shows the red trigger diamond while the trigger target end is being detached', () => {
    useUIStore.setState({
      draggingSourceType: 'TRIGGER',
      reconnectingHandle: { nodeId: 'p1', handleId: '__trigger', type: 'target' },
    });
    const { container } = renderPreset(presetData(), { id: 'p1' });
    const trigger = container.querySelector('[data-handleid="__trigger"]') as HTMLElement;
    expect(trigger.className).toMatch(/triggerHandleActive/);
    expect(trigger.className).toMatch(/triggerHandleDetaching/);
  });

  it('does not show the red trigger diamond when another node is the origin', () => {
    useUIStore.setState({
      draggingSourceType: 'TRIGGER',
      reconnectingHandle: { nodeId: 'other', handleId: '__trigger', type: 'target' },
    });
    const { container } = renderPreset(presetData(), { id: 'p1' });
    const trigger = container.querySelector('[data-handleid="__trigger"]') as HTMLElement;
    expect(trigger.className).toMatch(/triggerHandleActive/);
    expect(trigger.className).not.toMatch(/triggerHandleDetaching/);
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

  it('mousedown on a CONNECTED exposed input redirects to the edge reconnect anchor', () => {
    // The edge whose target endpoint is p1/x — grabbing that input should
    // hand the mousedown to this edge's reconnect anchor (detach drag).
    useTabStore.setState((s) => ({
      tabs: [
        {
          ...s.tabs[0],
          edges: [
            { id: 'pe1', source: 's1', sourceHandle: 'out', target: 'p1', targetHandle: 'x' },
          ],
        },
      ],
    }));
    // Anchor fixture matching the real React Flow edge DOM shape (verified
    // against @xyflow/react 12.10.1 EdgeWrapper/EdgeAnchor).
    const SVG_NS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(SVG_NS, 'svg');
    const group = document.createElementNS(SVG_NS, 'g');
    group.setAttribute('class', 'react-flow__edge react-flow__edge-default');
    group.setAttribute('data-id', 'pe1');
    const circle = document.createElementNS(SVG_NS, 'circle');
    circle.setAttribute('class', 'react-flow__edgeupdater react-flow__edgeupdater-target');
    circle.setAttribute('r', '10');
    group.appendChild(circle);
    svg.appendChild(group);
    document.body.appendChild(svg);
    const received: MouseEvent[] = [];
    circle.addEventListener('mousedown', (e) => received.push(e as MouseEvent));

    try {
      const { container } = renderPreset(presetData(), { id: 'p1' });
      const handle = container.querySelector('[data-handleid="x"]') as HTMLElement;

      const notCancelled = fireEvent.mouseDown(handle, { button: 0, clientX: 33, clientY: 44 });

      expect(received).toHaveLength(1);
      expect(received[0].clientX).toBe(33);
      expect(received[0].clientY).toBe(44);
      // fireEvent returns false when preventDefault was called on the original.
      expect(notCancelled).toBe(false);
    } finally {
      svg.remove();
    }
  });
});
