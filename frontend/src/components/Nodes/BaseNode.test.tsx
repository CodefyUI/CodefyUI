import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ReactFlow, ReactFlowProvider, type Edge, type Node, type NodeTypes } from '@xyflow/react';
import { renderWithFlow } from '../../test/utils';
import { useI18n } from '../../i18n';
import { useUIStore } from '../../store/uiStore';
import { useTabStore } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import { _resetPackStoreForTesting, usePackStore } from '../../store/packStore';
import type { NodeDefinition, NodeData } from '../../types';
import * as rest from '../../api/rest';
import type { PackItem, PackItemStatus, PackSummary } from '../../api/rest';
import { CATEGORY_COLORS, STATUS_COLORS, NODE_HEADER_TINT, mixColor } from '../../styles/theme';
import BaseNode, { BaseNodeBody } from './BaseNode';

// ── Helpers ──────────────────────────────────────────────────────────────

function makeDef(overrides: Partial<NodeDefinition> = {}): NodeDefinition {
  return {
    node_name: 'Linear',
    category: 'CNN',
    description: 'A linear layer',
    inputs: [
      {
        name: 'in',
        data_type: 'TENSOR',
        description: 'input tensor',
        optional: false,
      },
    ],
    outputs: [
      { name: 'out', data_type: 'TENSOR', description: 'output tensor', optional: false },
    ],
    params: [
      {
        name: 'units',
        param_type: 'int',
        default: 64,
        description: '',
        options: [],
        min_value: null,
        max_value: null,
      },
    ],
    ...overrides,
  };
}

function baseData(overrides: Partial<NodeData> = {}): NodeData {
  return {
    label: 'My Linear',
    type: 'Linear',
    params: {},
    definition: makeDef(),
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

/**
 * jsdom/cssstyle normalizes ordinary color-valued CSS properties (but not
 * custom properties like `--border-color`) to `rgb(r, g, b)` on read. Mirror
 * that here so header-fill assertions can be computed from the real palette
 * constants instead of a hand-copied literal.
 */
function hexToRgb(hex: string): string {
  const h = hex.replace('#', '');
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgb(${r}, ${g}, ${b})`;
}

function renderBody(data: NodeData, opts: { id?: string; selected?: boolean; bodyExtra?: React.ReactNode } = {}) {
  const { id = 'n1', selected = false, bodyExtra } = opts;
  return renderWithFlow(
    <BaseNodeBody
      id={id}
      type="baseNode"
      data={data}
      selected={selected}
      {...flowProps}
      bodyExtra={bodyExtra}
    />,
  );
}

/**
 * `getEdges()` reads React Flow's *own* store (not our tabStore) imperatively,
 * so the edges must already be in that store on the node's first render. We
 * mount a real <ReactFlow> with the node + edges to guarantee that.
 */
const nodeTypes: NodeTypes = {
  baseNode: (p) => <BaseNodeBody {...(p as React.ComponentProps<typeof BaseNodeBody>)} />,
};

function renderBodyWithEdges(data: NodeData, edges: Edge[], opts: { id?: string } = {}) {
  const { id = 'n1' } = opts;
  const nodes: Node[] = [{ id, type: 'baseNode', position: { x: 0, y: 0 }, data: data as never }];
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
  useUIStore.setState({ tooltipsEnabled: true, draggingSourceType: null, reconnectingHandle: null });
  // Reset to a single, clean active tab so outputSummaries lookups are empty.
  const id = 'tab-test';
  useTabStore.setState((s) => ({
    activeTabId: id,
    tabs: [
      {
        ...s.tabs[0],
        id,
        name: 'Tab 1',
        nodes: [],
        edges: [],
        outputSummaries: {},
        layersModalNodeId: null,
      },
    ],
  }));
  useToastStore.setState({ toasts: [] });
}

beforeEach(() => {
  resetStores();
});

describe('BaseNode', () => {
  it('renders header label and category, input + output ports, and params', () => {
    renderBody(baseData());
    expect(screen.getByText('My Linear')).toBeTruthy();
    expect(screen.getByText('CNN')).toBeTruthy();
    // input port label, output port label, param name + value
    expect(screen.getByText('in')).toBeTruthy();
    expect(screen.getByText('out')).toBeTruthy();
    expect(screen.getByText('units')).toBeTruthy();
    expect(screen.getByText('64')).toBeTruthy(); // default since params empty
  });

  it('uses provided param value over the default', () => {
    renderBody(baseData({ params: { units: 128 } }));
    expect(screen.getByText('128')).toBeTruthy();
  });

  // ── advanced params on the card (core#134) ───────────────────────────

  function advancedDef() {
    return makeDef({
      params: [
        {
          name: 'lr', param_type: 'float', default: 0.001, description: '',
          options: [], min_value: null, max_value: null,
        },
        {
          name: 'betas', param_type: 'string', default: '0.9, 0.999',
          description: '', options: [], min_value: null, max_value: null,
          advanced: true,
        },
      ],
    });
  }

  it('keeps an untouched advanced param off the node card', () => {
    renderBody(baseData({ definition: advancedDef(), params: { lr: 0.001 } }));
    expect(screen.getByText('lr')).toBeTruthy();
    expect(screen.queryByText('betas')).toBeNull();
  });

  it('shows an advanced param on the card once it differs from its default', () => {
    renderBody(
      baseData({ definition: advancedDef(), params: { lr: 0.001, betas: '0.5, 0.9' } }),
    );
    expect(screen.getByText('betas')).toBeTruthy();
    expect(screen.getByText('0.5, 0.9')).toBeTruthy();
  });

  it('masks a non-empty SECRET param value on the node card', () => {
    const def = makeDef({
      params: [
        {
          name: 'openai_api_key',
          param_type: 'secret',
          default: '',
          description: '',
          options: [],
          min_value: null,
          max_value: null,
        },
      ],
    });
    renderBody(
      baseData({ definition: def, params: { openai_api_key: 'sk-visible-leak' } }),
    );
    expect(screen.queryByText('sk-visible-leak')).toBeNull();
    expect(screen.getByText('••••••••')).toBeTruthy();
    expect(screen.getByTitle('••••••••')).toBeTruthy();
  });

  it('renders an empty SECRET param without mask dots', () => {
    const def = makeDef({
      params: [
        {
          name: 'openai_api_key',
          param_type: 'secret',
          default: '',
          description: '',
          options: [],
          min_value: null,
          max_value: null,
        },
      ],
    });
    renderBody(baseData({ definition: def, params: {} }));
    expect(screen.queryByText('••••••••')).toBeNull();
  });

  it('falls back to Utility category color when definition has no category', () => {
    // def.category undefined → category 'Utility'; an unrecognised category
    // string exercises the same `?? CATEGORY_COLORS.Utility` fallback path.
    const def = makeDef({ category: 'TotallyUnknownCat' });
    const { container } = renderBody(baseData({ definition: def }));
    expect(screen.getByText('TotallyUnknownCat')).toBeTruthy();
    // The header is no longer painted with the raw category hue: it's that
    // hue tinted into --surface-raised by NODE_HEADER_TINT (mixColor), which
    // is what took every node title from 1.85:1-2.69:1 contrast up to
    // 9.2:1-11.7:1 (see BaseNode's headerFill comment). '#242b37' mirrors
    // --surface-raised; it's hardcoded here the same way BaseNode.tsx
    // hardcodes it, since theme.ts only mirrors the semantic data palettes,
    // not chrome/surface tokens (see theme.ts's file comment).
    const header = container.querySelector('[class*="header"]') as HTMLElement;
    const expectedFill = mixColor('#242b37', CATEGORY_COLORS.Utility, NODE_HEADER_TINT);
    expect(header.style.background).toBe(hexToRgb(expectedFill));
  });

  it('renders Utility category when no definition is present', () => {
    renderBody(baseData({ definition: undefined }));
    expect(screen.getByText('Utility')).toBeTruthy();
    // No ports / params rendered without a definition.
    expect(screen.queryByText('in')).toBeNull();
  });

  it('renders a divider only when both inputs and outputs exist', () => {
    const { container, rerender } = renderBody(baseData());
    expect(container.querySelector('[class*="divider"]')).toBeTruthy();
    // outputs only → no divider
    rerender(
      <BaseNodeBody
        id="n1"
        type="baseNode"
        data={baseData({ definition: makeDef({ inputs: [] }) })}
        selected={false}
        {...flowProps}
      />,
    );
    expect(container.querySelector('[class*="divider"]')).toBeNull();
  });

  it('marks optional inputs with the opt label', () => {
    const def = makeDef({
      inputs: [
        { name: 'maybe', data_type: 'TENSOR', description: '', optional: true },
      ],
    });
    renderBody(baseData({ definition: def }));
    expect(screen.getByText(useI18n.getState().t('node.opt'))).toBeTruthy();
  });

  it('shows tooltip on hover when tooltips enabled and there is a description', () => {
    const { container } = renderBody(baseData());
    const node = container.querySelector('[class*="node"]') as HTMLElement;
    fireEvent.mouseEnter(node);
    expect(screen.getByText('A linear layer')).toBeTruthy();
    fireEvent.mouseLeave(node);
    expect(screen.queryByText('A linear layer')).toBeNull();
  });

  it('does not show tooltip when tooltips disabled', () => {
    useUIStore.setState({ tooltipsEnabled: false });
    const { container } = renderBody(baseData());
    fireEvent.mouseEnter(container.querySelector('[class*="node"]') as HTMLElement);
    expect(screen.queryByText('A linear layer')).toBeNull();
  });

  it('does not show tooltip when description is empty', () => {
    const { container } = renderBody(baseData({ definition: makeDef({ description: '' }) }));
    fireEvent.mouseEnter(container.querySelector('[class*="node"]') as HTMLElement);
    // empty description → tooltip block not rendered (no tooltipTitle)
    expect(container.querySelector('[class*="tooltipTitle"]')).toBeNull();
  });

  // ── Border color branches ────────────────────────────────────────────────

  it('selected node gets the text-primary border + glow shadow', () => {
    // The selected border used to be a hardcoded '#ffffff'; it's now
    // var(--text-primary) (#f0f4f8, not literal white) — the same token the
    // node's own title text uses, rather than an unrelated hardcoded white.
    // jsdom does not resolve var() references on a custom property, so the
    // raw token string is what `.style` reports.
    const { container } = renderBody(baseData(), { selected: true });
    const node = container.querySelector('[class*="node"]') as HTMLElement;
    expect(node.style.getPropertyValue('--border-color')).toBe('var(--text-primary)');
    expect(node.style.boxShadow).toContain('rgba(255,255,255,0.15)');
  });

  it('unselected default node gets the default border + drop shadow', () => {
    // Both values are now CSS custom properties rather than hardcoded
    // literals ('#444444' / a literal rgba drop shadow) — jsdom does not
    // resolve var() references, so the raw token strings are what render.
    const { container } = renderBody(baseData());
    const node = container.querySelector('[class*="node"]') as HTMLElement;
    expect(node.style.getPropertyValue('--border-color')).toBe('var(--border-base)');
    expect(node.style.boxShadow).toBe('var(--shadow)');
  });

  it.each([
    // `--border-color` is a CSS custom property; jsdom does NOT normalize
    // custom-prop values, so the raw hex from STATUS_COLORS is preserved
    // verbatim (lowercase, as theme.ts defines it — theme.test.ts is the one
    // place that pins these against tokens.css, so referencing the constant
    // here keeps this test meaningful without duplicating that guarantee).
    ['running', STATUS_COLORS.running],
    ['completed', STATUS_COLORS.completed],
    ['error', STATUS_COLORS.error],
    ['cached', STATUS_COLORS.cached],
    // core#122: a node that stopped part-way with partial results. Without
    // its own branch it falls through to 'transparent' and looks exactly
    // like a node that never ran.
    ['interrupted', STATUS_COLORS.interrupted],
    // core#260: passed over because something upstream failed. A preset can
    // settle here too, so both node cards must agree on how it looks.
    ['skipped', STATUS_COLORS.skipped],
  ] as const)('uses the %s status border when unselected', (status, hex) => {
    const data = baseData({
      executionStatus: status,
      error: status === 'error' ? 'boom' : undefined,
    });
    const { container } = renderBody(data);
    const node = container.querySelector('[class*="node"]') as HTMLElement;
    expect(node.style.getPropertyValue('--border-color')).toBe(hex);
  });

  // ── Status footers ────────────────────────────────────────────────────────

  it('renders the error footer with the error message', () => {
    renderBody(baseData({ executionStatus: 'error', error: 'kaboom' }));
    expect(screen.getByText('Error: kaboom')).toBeTruthy();
  });

  it('does not render error footer when status is error but no error text', () => {
    renderBody(baseData({ executionStatus: 'error', error: undefined }));
    expect(screen.queryByText(/Error:/)).toBeNull();
  });

  it('renders the running footer (no progress)', () => {
    renderBody(baseData({ executionStatus: 'running' }));
    expect(screen.getByText('Running...')).toBeTruthy();
  });

  it('renders the running footer with epoch progress', () => {
    const { container } = renderBody(
      baseData({
        executionStatus: 'running',
        progress: { event: 'epoch', epoch: 2, total_epochs: 10, loss: 0.123456 },
      }),
    );
    expect(screen.getByText('Epoch 2/10')).toBeTruthy();
    expect(screen.getByText('Loss: 0.1235')).toBeTruthy();
    const fill = container.querySelector('[class*="progressBarFill"]') as HTMLElement;
    expect(fill.style.width).toBe('20%');
  });

  it('progress bar handles missing epoch/total_epochs (defaults to 0/1)', () => {
    const { container } = renderBody(
      baseData({
        executionStatus: 'running',
        // event epoch but epoch/total_epochs undefined → width branch uses ?? fallbacks
        progress: { event: 'epoch', loss: 0 },
      }),
    );
    const fill = container.querySelector('[class*="progressBarFill"]') as HTMLElement;
    expect(fill.style.width).toBe('0%');
  });

  it('renders the completed footer', () => {
    renderBody(baseData({ executionStatus: 'completed' }));
    expect(screen.getByText('Completed')).toBeTruthy();
  });

  it('renders the cached footer', () => {
    renderBody(baseData({ executionStatus: 'cached' }));
    expect(screen.getByText('Cached')).toBeTruthy();
  });

  it('renders the skipped footer', () => {
    renderBody(baseData({ executionStatus: 'skipped' }));
    expect(screen.getByText('Skipped')).toBeTruthy();
  });

  // ── SequentialModel branch ────────────────────────────────────────────────

  it('renders SequentialModel layer count + hint and opens the layers editor on dbl-click', () => {
    const data = baseData({
      type: 'SequentialModel',
      params: { layers: JSON.stringify([{}, {}, {}]) },
    });
    const { container } = renderBody(data);
    expect(screen.getByText('3')).toBeTruthy();
    expect(screen.getByText(useI18n.getState().t('layersEditor.hint'))).toBeTruthy();
    // cursor: pointer for sequential models
    const node = container.querySelector('[class*="node"]') as HTMLElement;
    expect(node.style.cursor).toBe('pointer');
    // Double-click opens the layers editor
    fireEvent.click(node, { detail: 2 });
    expect(useTabStore.getState().getActiveTab().layersModalNodeId).toBe('n1');
  });

  it('SequentialModel with invalid layers JSON falls back to count 0', () => {
    const data = baseData({
      type: 'SequentialModel',
      params: { layers: 'not-json' },
    });
    renderBody(data);
    expect(screen.getByText('0')).toBeTruthy();
  });

  it('SequentialModel with missing layers param defaults to []', () => {
    const data = baseData({ type: 'SequentialModel', params: {} });
    renderBody(data);
    expect(screen.getByText('0')).toBeTruthy();
  });

  it('single-click (detail=1) on SequentialModel does not open the modal', () => {
    const data = baseData({ type: 'SequentialModel', params: { layers: '[]' } });
    const { container } = renderBody(data);
    fireEvent.click(container.querySelector('[class*="node"]') as HTMLElement, { detail: 1 });
    expect(useTabStore.getState().getActiveTab().layersModalNodeId).toBeNull();
  });

  it('double-click on a non-sequential node does not open the modal', () => {
    const { container } = renderBody(baseData());
    fireEvent.click(container.querySelector('[class*="node"]') as HTMLElement, { detail: 2 });
    expect(useTabStore.getState().getActiveTab().layersModalNodeId).toBeNull();
    // Non-sequential nodes have no explicit cursor.
    const node = container.querySelector('[class*="node"]') as HTMLElement;
    expect(node.style.cursor).toBe('');
  });

  it('does not render params section for SequentialModel', () => {
    const data = baseData({ type: 'SequentialModel', params: { layers: '[]' } });
    renderBody(data);
    // params section (units) should not appear for the sequential branch
    expect(screen.queryByText('units')).toBeNull();
  });

  it('renders no params section when definition has zero params', () => {
    renderBody(baseData({ definition: makeDef({ params: [] }) }));
    expect(screen.queryByText('units')).toBeNull();
  });

  // ── Trigger / dragging branches ───────────────────────────────────────────

  it('applies the triggerDropTarget class while dragging a TRIGGER source', () => {
    useUIStore.setState({ draggingSourceType: 'TRIGGER' });
    const { container } = renderBody(baseData());
    const node = container.querySelector('[class*="node"]') as HTMLElement;
    expect(node.className).toMatch(/triggerDropTarget/);
  });

  it('marks input ports compatible/incompatible while dragging a non-trigger source', () => {
    // TENSOR source dragged onto a TENSOR input → compatible; STRING input → incompatible.
    useUIStore.setState({ draggingSourceType: 'TENSOR' });
    const def = makeDef({
      inputs: [
        { name: 'a', data_type: 'TENSOR', description: '', optional: false },
        { name: 'b', data_type: 'STRING', description: '', optional: false },
      ],
    });
    const { container } = renderBody(baseData({ definition: def }));
    expect(container.querySelector('[class*="portCompatible"]')).toBeTruthy();
    expect(container.querySelector('[class*="portIncompatible"]')).toBeTruthy();
  });

  it('marks all input ports incompatible while dragging a TRIGGER source', () => {
    useUIStore.setState({ draggingSourceType: 'TRIGGER' });
    const { container } = renderBody(baseData());
    expect(container.querySelector('[class*="portIncompatible"]')).toBeTruthy();
    expect(container.querySelector('[class*="portCompatible"]')).toBeNull();
  });

  // ── Reconnect detach indicator ────────────────────────────────────────────

  it('adds portDetaching only to the input handle matching reconnectingHandle', () => {
    useUIStore.setState({
      reconnectingHandle: { nodeId: 'n1', handleId: 'a', type: 'target' },
    });
    const def = makeDef({
      inputs: [
        { name: 'a', data_type: 'TENSOR', description: '', optional: false },
        { name: 'b', data_type: 'TENSOR', description: '', optional: false },
      ],
    });
    const { container } = renderBody(baseData({ definition: def }), { id: 'n1' });
    const detaching = container.querySelectorAll('[class*="portDetaching"]');
    expect(detaching).toHaveLength(1);
    expect((detaching[0] as HTMLElement).getAttribute('data-handleid')).toBe('a');
  });

  it('adds portDetaching to the matching output handle when type is source', () => {
    useUIStore.setState({
      reconnectingHandle: { nodeId: 'n1', handleId: 'out', type: 'source' },
    });
    const { container } = renderBody(baseData(), { id: 'n1' });
    const detaching = container.querySelectorAll('[class*="portDetaching"]');
    expect(detaching).toHaveLength(1);
    expect((detaching[0] as HTMLElement).getAttribute('data-handleid')).toBe('out');
  });

  it('does not mark any handle when reconnectingHandle points at another node', () => {
    useUIStore.setState({
      reconnectingHandle: { nodeId: 'other-node', handleId: 'in', type: 'target' },
    });
    const { container } = renderBody(baseData(), { id: 'n1' });
    expect(container.querySelector('[class*="portDetaching"]')).toBeNull();
  });

  it('requires the handle type to match (same id, wrong type is not marked)', () => {
    // 'in' exists as a target handle; a source-typed match must not hit it.
    useUIStore.setState({
      reconnectingHandle: { nodeId: 'n1', handleId: 'in', type: 'source' },
    });
    const { container } = renderBody(baseData(), { id: 'n1' });
    expect(container.querySelector('[class*="portDetaching"]')).toBeNull();
  });

  it('portDetaching wins over portCompatible on the origin handle while others keep the compat glow', () => {
    // Reconnect drag in progress: the origin input would also qualify as a
    // compatible target for the dragged TENSOR source — red must win there,
    // green stays on the other compatible input.
    useUIStore.setState({
      draggingSourceType: 'TENSOR',
      reconnectingHandle: { nodeId: 'n1', handleId: 'a', type: 'target' },
    });
    const def = makeDef({
      inputs: [
        { name: 'a', data_type: 'TENSOR', description: '', optional: false },
        { name: 'b', data_type: 'TENSOR', description: '', optional: false },
      ],
    });
    const { container } = renderBody(baseData({ definition: def }), { id: 'n1' });
    const origin = container.querySelector('[data-handleid="a"]') as HTMLElement;
    const other = container.querySelector('[data-handleid="b"]') as HTMLElement;
    expect(origin.className).toMatch(/portDetaching/);
    expect(origin.className).not.toMatch(/portCompatible/);
    expect(origin.className).not.toMatch(/portIncompatible/);
    expect(other.className).toMatch(/portCompatible/);
    expect(other.className).not.toMatch(/portDetaching/);
  });

  it('shows the red trigger diamond while the trigger target end is being detached', () => {
    useUIStore.setState({
      // During a trigger reconnect the drag origin is Start's source handle,
      // so draggingSourceType is also TRIGGER (green diamonds everywhere).
      draggingSourceType: 'TRIGGER',
      reconnectingHandle: { nodeId: 'n1', handleId: '__trigger', type: 'target' },
    });
    const { container } = renderBody(baseData(), { id: 'n1' });
    const trigger = container.querySelector('[data-handleid="__trigger"]') as HTMLElement;
    expect(trigger.className).toMatch(/triggerHandleActive/);
    expect(trigger.className).toMatch(/triggerHandleDetaching/);
  });

  it('does not show the red trigger diamond on nodes that are not the origin', () => {
    useUIStore.setState({
      draggingSourceType: 'TRIGGER',
      reconnectingHandle: { nodeId: 'other-node', handleId: '__trigger', type: 'target' },
    });
    const { container } = renderBody(baseData(), { id: 'n1' });
    const trigger = container.querySelector('[data-handleid="__trigger"]') as HTMLElement;
    expect(trigger.className).toMatch(/triggerHandleActive/);
    expect(trigger.className).not.toMatch(/triggerHandleDetaching/);
  });

  it('adds the entryPoint class when an incoming trigger edge targets this node', async () => {
    renderBodyWithEdges(
      baseData(),
      [{ id: 'e1', source: 'start', target: 'n1', data: { type: 'trigger' } } as Edge],
      { id: 'n1' },
    );
    const node = await waitFor(() => {
      const el = document.querySelector('[class*="BaseNode-module"], [class*="_node_"]');
      const n = [...document.querySelectorAll('div')].find((d) => /entryPoint/.test(d.className));
      if (!el && !n) throw new Error('not rendered yet');
      return n;
    });
    expect(node).toBeTruthy();
  });

  it('does NOT add entryPoint when the incoming edge is not a trigger edge', async () => {
    renderBodyWithEdges(
      baseData({ label: 'NonTrigger' }),
      [{ id: 'e1', source: 'start', target: 'n1', data: { type: 'data' } } as Edge],
      { id: 'n1' },
    );
    await waitFor(() => expect(screen.getByText('NonTrigger')).toBeTruthy());
    const hasEntry = [...document.querySelectorAll('div')].some((d) => /entryPoint/.test(d.className));
    expect(hasEntry).toBe(false);
  });

  // ── Download button (completed + downloadable output) ──────────────────────

  function seedDownloadableOutput(nodeId = 'n1', path = 'runs/exp1/model.pt') {
    useTabStore.setState((s) => ({
      tabs: s.tabs.map((t) =>
        t.id === s.activeTabId
          ? {
              ...t,
              outputSummaries: {
                [nodeId]: {
                  model: { type: 'string', download_path: path } as never,
                },
              },
            }
          : t,
      ),
    }));
  }

  it('shows a download button when completed with a downloadable output', () => {
    seedDownloadableOutput();
    renderBody(baseData({ executionStatus: 'completed' }));
    // basename of the path
    expect(screen.getByRole('button', { name: /model\.pt/ })).toBeTruthy();
  });

  it('clicking download calls downloadModelFile and toggles the downloading state', async () => {
    seedDownloadableOutput();
    let resolveDl: () => void = () => {};
    const spy = vi
      .spyOn(rest, 'downloadModelFile')
      .mockImplementation(() => new Promise<void>((res) => { resolveDl = res; }));
    renderBody(baseData({ executionStatus: 'completed' }));
    const btn = screen.getByRole('button', { name: /model\.pt/ }) as HTMLButtonElement;
    fireEvent.click(btn);
    // While the promise is pending, the button is disabled and shows the spinner glyph.
    await waitFor(() => expect(btn.disabled).toBe(true));
    expect(spy).toHaveBeenCalledWith('runs/exp1/model.pt');
    resolveDl();
    await waitFor(() => expect(btn.disabled).toBe(false));
    spy.mockRestore();
  });

  it('download failure adds an error toast with the error message', async () => {
    seedDownloadableOutput();
    const spy = vi
      .spyOn(rest, 'downloadModelFile')
      .mockRejectedValue(new Error('disk full'));
    renderBody(baseData({ executionStatus: 'completed' }));
    fireEvent.click(screen.getByRole('button', { name: /model\.pt/ }));
    await waitFor(() => {
      const toasts = useToastStore.getState().toasts;
      expect(toasts.some((t) => t.message === 'disk full' && t.type === 'error')).toBe(true);
    });
    spy.mockRestore();
  });

  it('download failure with no error message falls back to the localized string', async () => {
    seedDownloadableOutput();
    const spy = vi
      .spyOn(rest, 'downloadModelFile')
      .mockRejectedValue({}); // err.message is undefined
    renderBody(baseData({ executionStatus: 'completed' }));
    fireEvent.click(screen.getByRole('button', { name: /model\.pt/ }));
    await waitFor(() => {
      const toasts = useToastStore.getState().toasts;
      expect(toasts.some((t) => t.message === useI18n.getState().t('download.failed'))).toBe(true);
    });
    spy.mockRestore();
  });

  it('handleDownload no-ops when there is no downloadable path (guard branch)', () => {
    // Completed but no downloadable output → no button, and the early return
    // in handleDownload is unreachable via UI but the guard exists. Assert the
    // button is simply absent.
    renderBody(baseData({ executionStatus: 'completed' }));
    expect(screen.queryByRole('button')).toBeNull();
  });

  it('completed footer without download output renders just the completed text', () => {
    renderBody(baseData({ executionStatus: 'completed' }));
    expect(screen.getByText('Completed')).toBeTruthy();
  });

  // ── bodyExtra slot + default export ────────────────────────────────────────

  it('renders the injected bodyExtra slot', () => {
    renderBody(baseData(), { bodyExtra: <div>injected-body</div> });
    expect(screen.getByText('injected-body')).toBeTruthy();
  });

  it('default export (memoized BaseNode) renders the same card', () => {
    renderWithFlow(
      <BaseNode
        id="n9"
        type="baseNode"
        data={baseData({ label: 'Default Export Node' })}
        selected={false}
        {...flowProps}
      />,
    );
    expect(screen.getByText('Default Export Node')).toBeTruthy();
  });

  it('localizes the description via tn for non-English locale (falls back when untranslated)', () => {
    // zh-TW with NO node translation for this name → falls back to the English description.
    useI18n.setState({ locale: 'zh-TW' });
    const def = makeDef({ node_name: 'ZzzNoTranslationNode', description: 'A linear layer' });
    const { container } = renderBody(baseData({ definition: def }));
    fireEvent.mouseEnter(container.querySelector('[class*="node"]') as HTMLElement);
    const tooltip = container.querySelector('[class*="tooltipDesc"]') as HTMLElement;
    expect(tooltip).toBeTruthy();
    expect(tooltip.textContent).toContain('A linear layer');
  });
});

// ── Detach drag redirect (connected input pulls its edge off) ──────────────

describe('BaseNode detach drag redirect', () => {
  const SVG_NS = 'http://www.w3.org/2000/svg';

  /**
   * Real React Flow edge DOM shape (verified against @xyflow/react 12.10.1):
   * `<g class="react-flow__edge" data-id>` wrapping the invisible
   * `<circle class="react-flow__edgeupdater react-flow__edgeupdater-target">`
   * reconnect anchor. A spy listener records redirected mousedowns.
   *
   * Default parent is document.body: `renderBody` mounts the node WITHOUT a
   * `.react-flow` container, so those tests exercise the helper's
   * document-wide fallback; the cross-canvas test below mounts real
   * `.react-flow` containers to exercise the scoped path.
   */
  function mountAnchorFixture(edgeId: string, parent: Element = document.body) {
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('data-fixture', 'anchor');
    const group = document.createElementNS(SVG_NS, 'g');
    group.setAttribute('class', 'react-flow__edge react-flow__edge-default');
    group.setAttribute('data-id', edgeId);
    const circle = document.createElementNS(SVG_NS, 'circle');
    circle.setAttribute('class', 'react-flow__edgeupdater react-flow__edgeupdater-target');
    circle.setAttribute('r', '10');
    group.appendChild(circle);
    svg.appendChild(group);
    parent.appendChild(svg);
    const received: MouseEvent[] = [];
    circle.addEventListener('mousedown', (e) => received.push(e as MouseEvent));
    return { received };
  }

  /**
   * A `.react-flow` canvas container div — the class the ReactFlow root
   * carries (verified in @xyflow/react 12.10.1: `<div
   * data-testid="rf__wrapper" ... className={cc(['react-flow', ...])}>`).
   * The app mounts one such (hidden) canvas per open tab.
   */
  function mountCanvas() {
    const canvas = document.createElement('div');
    canvas.className = 'react-flow';
    canvas.setAttribute('data-fixture', 'anchor');
    document.body.appendChild(canvas);
    return canvas;
  }

  function seedEdges(edges: Edge[]) {
    useTabStore.setState((s) => ({ tabs: [{ ...s.tabs[0], edges }] }));
  }

  const twoInputDef = () =>
    makeDef({
      inputs: [
        { name: 'a', data_type: 'TENSOR', description: '', optional: false },
        { name: 'b', data_type: 'TENSOR', description: '', optional: false },
      ],
    });

  afterEach(() => {
    document.querySelectorAll('[data-fixture="anchor"]').forEach((el) => el.remove());
  });

  it('mousedown on a CONNECTED input redirects to the edge anchor and suppresses the default connection start', () => {
    seedEdges([{ id: 'e1', source: 's1', sourceHandle: 'out', target: 'n1', targetHandle: 'a' }]);
    const { received } = mountAnchorFixture('e1');
    const { container } = renderBody(baseData({ definition: twoInputDef() }), { id: 'n1' });
    const handle = container.querySelector('[data-handleid="a"]') as HTMLElement;

    const notCancelled = fireEvent.mouseDown(handle, { button: 0, clientX: 120, clientY: 45 });

    expect(received).toHaveLength(1);
    expect(received[0].clientX).toBe(120);
    expect(received[0].clientY).toBe(45);
    // fireEvent returns false when preventDefault was called on the original.
    expect(notCancelled).toBe(false);
  });

  it('mousedown on an UNCONNECTED input keeps the normal new-connection behavior', () => {
    seedEdges([{ id: 'e1', source: 's1', sourceHandle: 'out', target: 'n1', targetHandle: 'a' }]);
    const { received } = mountAnchorFixture('e1');
    const { container } = renderBody(baseData({ definition: twoInputDef() }), { id: 'n1' });
    const handle = container.querySelector('[data-handleid="b"]') as HTMLElement;

    const notCancelled = fireEvent.mouseDown(handle, { button: 0, clientX: 5, clientY: 6 });

    expect(received).toHaveLength(0);
    expect(notCancelled).toBe(true);
  });

  it('ctrl-modified mousedown on a connected input is NOT redirected (escape hatch)', () => {
    seedEdges([{ id: 'e1', source: 's1', sourceHandle: 'out', target: 'n1', targetHandle: 'a' }]);
    const { received } = mountAnchorFixture('e1');
    const { container } = renderBody(baseData({ definition: twoInputDef() }), { id: 'n1' });
    const handle = container.querySelector('[data-handleid="a"]') as HTMLElement;

    const notCancelled = fireEvent.mouseDown(handle, {
      button: 0,
      ctrlKey: true,
      clientX: 120,
      clientY: 45,
    });

    expect(received).toHaveLength(0);
    expect(notCancelled).toBe(true);
  });

  it('output handles never intercept, even if an edge coincidentally targets that handle id', () => {
    // Bogus edge whose target handle shares the OUTPUT handle's name — proves
    // the capture handler simply is not attached on source handles.
    seedEdges([{ id: 'e9', source: 's1', sourceHandle: 'x', target: 'n1', targetHandle: 'out' }]);
    const { received } = mountAnchorFixture('e9');
    const { container } = renderBody(baseData(), { id: 'n1' });
    const handle = container.querySelector('[data-handleid="out"]') as HTMLElement;

    const notCancelled = fireEvent.mouseDown(handle, { button: 0, clientX: 1, clientY: 2 });

    expect(received).toHaveLength(0);
    expect(notCancelled).toBe(true);
  });

  it("with duplicate edge ids across canvases, redirects to the anchor in the handle's OWN canvas", () => {
    // Edge ids are only unique PER TAB (example graphs carry fixed ids) and
    // the app mounts one hidden FlowCanvas per open tab — so the same
    // data-id can exist in several `.react-flow` containers at once. The
    // decoy canvas comes FIRST in document order; an unscoped document
    // query would dispatch into that wrong (hidden) canvas.
    seedEdges([{ id: 'dup', source: 's1', sourceHandle: 'out', target: 'n1', targetHandle: 'a' }]);

    const decoyCanvas = mountCanvas();
    const decoy = mountAnchorFixture('dup', decoyCanvas);

    const ownCanvas = mountCanvas();
    renderWithFlow(
      <BaseNodeBody
        id="n1"
        type="baseNode"
        data={baseData({ definition: twoInputDef() })}
        selected={false}
        {...flowProps}
      />,
      { container: ownCanvas },
    );
    // React's createRoot clears the container on first render — mount the
    // anchor fixture AFTER rendering so it survives inside the canvas.
    const own = mountAnchorFixture('dup', ownCanvas);

    const handle = ownCanvas.querySelector('[data-handleid="a"]') as HTMLElement;
    const notCancelled = fireEvent.mouseDown(handle, { button: 0, clientX: 9, clientY: 9 });

    expect(own.received).toHaveLength(1);
    expect(decoy.received).toHaveLength(0);
    expect(notCancelled).toBe(false);
  });
});

afterEach(() => {
  vi.restoreAllMocks();
});


// ── Bypass visuals (core#128) ────────────────────────────────────────────

describe('BaseNode bypass', () => {
  beforeEach(resetStores);

  it('renders no bypass marker on an ordinary node', () => {
    const { container } = renderBody(baseData());
    expect(screen.queryByText('BYPASS')).toBeNull();
    expect(container.querySelector('[data-bypassed]')).toBeNull();
  });

  it('dims, strikes through and badges a bypassed node', () => {
    const { container } = renderBody(baseData({ bypassed: true }));

    const badge = screen.getByText('BYPASS');
    expect(badge).toBeInTheDocument();
    expect(badge.getAttribute('title')).toBe(
      'Bypassed: this node is skipped and passes its input straight through',
    );

    // The card carries the dim/strike-through class, and says so in the DOM
    // so a test (or a screenshot diff) does not have to match on a hashed
    // CSS-module name.
    const card = container.querySelector('[data-bypassed="true"]');
    expect(card).not.toBeNull();
    expect(card!.className).toMatch(/bypassed/);
  });

  it('keeps the label and ports of a bypassed node on screen', () => {
    renderBody(baseData({ bypassed: true }));
    // Muting hides nothing: the node is still readable and still selectable,
    // which is how it gets un-muted.
    expect(screen.getByText('My Linear')).toBeInTheDocument();
    expect(screen.getByText('in')).toBeInTheDocument();
    expect(screen.getByText('out')).toBeInTheDocument();
  });
});

// ── PythonScript: dynamic ports + code preview (core#131) ────────────────

describe('BaseNodeBody — script nodes', () => {
  const scriptDef = () =>
    makeDef({
      node_name: 'PythonScript',
      category: 'Utility',
      inputs: [{ name: 'in1', data_type: 'TENSOR', description: '', optional: true }],
      outputs: [{ name: 'out1', data_type: 'ANY', description: '', optional: false }],
      params: [
        {
          name: 'code',
          param_type: 'code',
          default: '',
          description: '',
          options: [],
          min_value: null,
          max_value: null,
        },
        {
          name: 'input_ports',
          param_type: 'int',
          default: 1,
          description: '',
          options: [],
          min_value: 1,
          max_value: 8,
        },
        {
          name: 'output_ports',
          param_type: 'int',
          default: 1,
          description: '',
          options: [],
          min_value: 1,
          max_value: 8,
        },
      ],
    });

  const scriptData = (params: Record<string, unknown>) =>
    baseData({ type: 'PythonScript', definition: scriptDef(), params });

  it('renders one handle per configured port, both sides', () => {
    renderBody(scriptData({ code: '', input_ports: 3, output_ports: 2 }));
    for (const name of ['in1', 'in2', 'in3', 'out1', 'out2']) {
      expect(screen.getByText(name)).toBeInTheDocument();
    }
    expect(screen.queryByText('in4')).toBeNull();
    expect(screen.queryByText('out3')).toBeNull();
  });

  it('shows the opening lines of the script instead of an escaped one-liner', () => {
    renderBody(
      scriptData({
        code: '\n\ndef run(inputs, params):\n    x = inputs["in1"]\n    return {"out1": x}\n',
        input_ports: 1,
        output_ports: 1,
      }),
    );
    // Leading blank lines are skipped so the preview starts at real code.
    expect(screen.getByText('def run(inputs, params):')).toBeInTheDocument();
    // Exact text, indentation included: getByText normalizes whitespace by
    // default, which would hide a preview that had lost its indentation.
    expect(
      screen.getByText('    x = inputs["in1"]', { normalizer: (text) => text }),
    ).toBeInTheDocument();
    // The raw value is never rendered as one long line.
    expect(screen.queryByText(/\n/)).toBeNull();
  });

  it('counts the lines it did not show', () => {
    const code = Array.from({ length: 9 }, (_, i) => `line${i}`).join('\n');
    renderBody(scriptData({ code, input_ports: 1, output_ports: 1 }));
    expect(screen.getByText('+5 more lines')).toBeInTheDocument();
  });

  it('says the script is empty rather than showing a blank block', () => {
    renderBody(scriptData({ code: '   \n\n', input_ports: 1, output_ports: 1 }));
    expect(screen.getByText('(no code yet)')).toBeInTheDocument();
  });
});

// ── Optional packs on the card (PR 2, F8) ────────────────────────────────

function packItem(id: string, status: PackItemStatus): PackItem {
  return {
    id,
    kind: 'hf',
    repo_id: `org/${id}`,
    url: null,
    size_bytes: 1024,
    license: null,
    status,
  };
}

function packSummary(over: Partial<PackSummary> & { id: string }): PackSummary {
  return {
    title: over.id,
    description: '',
    install_mode: 'live',
    status: 'not_installed',
    pip_ready: false,
    usable: false,
    depends_on: [],
    blocked_by: [],
    pip: [],
    items: [],
    size_bytes_total: 0,
    install_command: null,
    ...over,
  };
}

/** Put a catalog in the store, the way a finished `refresh()` would. */
function seedPacks(...packs: PackSummary[]) {
  usePackStore.setState({
    loaded: true,
    unsupported: false,
    packs,
    byId: Object.fromEntries(packs.map((pack) => [pack.id, pack])),
  });
}

const wordVectors = (over: Partial<PackSummary> = {}) =>
  packSummary({ id: 'word-vectors', title: 'Word vectors', ...over });

/** A node whose SELECT names one downloaded model per option. */
const embeddingDef = () =>
  makeDef({
    node_name: 'SentenceEmbed',
    params: [
      {
        name: 'model',
        param_type: 'select',
        default: 'all-MiniLM-L6-v2',
        description: '',
        options: ['all-MiniLM-L6-v2', 'all-mpnet-base-v2'],
        option_packs: {
          'all-MiniLM-L6-v2': 'sentence-embeddings:all-MiniLM-L6-v2',
          'all-mpnet-base-v2': 'sentence-embeddings:all-mpnet-base-v2',
        },
        min_value: null,
        max_value: null,
      },
    ],
  });

const sentenceEmbeddings = () =>
  packSummary({
    id: 'sentence-embeddings',
    title: 'Sentence embeddings',
    pip_ready: true,
    // "partial": one model was downloaded, the other never was.
    usable: false,
    items: [packItem('all-MiniLM-L6-v2', 'present'), packItem('all-mpnet-base-v2', 'missing')],
  });

describe('BaseNode — optional packs', () => {
  beforeEach(() => {
    resetStores();
    _resetPackStoreForTesting();
    useUIStore.setState({ packCenterOpen: false, packCenterFocusPackId: null });
  });

  it('shows the PACK header badge and opens the Package Center on click without selecting/dragging', () => {
    seedPacks(wordVectors());
    renderBody(baseData({ definition: makeDef({ requires_pack: 'word-vectors' }) }));

    const badge = screen.getByRole('button', { name: 'PACK' });
    expect(badge.getAttribute('title')).toBe(
      'Needs the Word vectors pack. Click to open the Package Center.',
    );
    // xyflow starts a node drag from a plain mousedown; `nodrag` is how every
    // other control inside a card (the code editor, the text-input box) opts
    // out of it.
    expect(badge.className).toContain('nodrag');

    // The card is a click target of its own — selection, and a double-click
    // that opens the detail modal — so the badge's click has to stop at the
    // badge. React's delegated listener sits on the render container, so a
    // listener on <body> fires only for clicks allowed to bubble past it.
    const escaped = vi.fn();
    document.body.addEventListener('click', escaped);
    try {
      fireEvent.click(badge);
      expect(escaped).not.toHaveBeenCalled();
      // Counterfactual, so the assertion above is about stopPropagation and
      // not about the harness: an ordinary click on the card does bubble.
      fireEvent.click(screen.getByText('My Linear'));
      expect(escaped).toHaveBeenCalledTimes(1);
    } finally {
      document.body.removeEventListener('click', escaped);
    }

    expect(useUIStore.getState().packCenterOpen).toBe(true);
    // Opened on the pack id — not on the click event, and not unfocused.
    expect(useUIStore.getState().packCenterFocusPackId).toBe('word-vectors');
  });

  it('marks a select param whose value needs a missing model', () => {
    seedPacks(sentenceEmbeddings());
    renderBody(
      baseData({ definition: embeddingDef(), params: { model: 'all-mpnet-base-v2' } }),
    );

    // The value itself stays on the card: a graph saved where the model was
    // downloaded still has to be readable where it was not.
    expect(screen.getByText('all-mpnet-base-v2')).toBeInTheDocument();
    const marker = screen.getByText('needs pack');
    expect(marker.getAttribute('title')).toBe(
      '"all-mpnet-base-v2" needs the model all-mpnet-base-v2 from the Sentence embeddings pack.',
    );
    // The requirement is per-model, so the pack itself is not the problem and
    // the header carries no badge.
    expect(screen.queryByText('PACK')).toBeNull();
  });

  it('leaves the sibling value that IS downloaded unmarked', () => {
    seedPacks(sentenceEmbeddings());
    renderBody(
      baseData({ definition: embeddingDef(), params: { model: 'all-MiniLM-L6-v2' } }),
    );
    expect(screen.getByText('all-MiniLM-L6-v2')).toBeInTheDocument();
    // Only the CURRENT value is asked about: the other option being missing
    // is the config panel's business, not the card's.
    expect(screen.queryByText('needs pack')).toBeNull();
  });

  it('says nothing when the pack is installed, or the catalog cannot answer', () => {
    seedPacks(
      wordVectors({ pip_ready: true, usable: true, status: 'installed' }),
      packSummary({
        id: 'sentence-embeddings',
        pip_ready: true,
        usable: true,
        items: [packItem('all-mpnet-base-v2', 'present')],
      }),
    );
    const { unmount } = renderBody(
      baseData({
        definition: makeDef({ ...embeddingDef(), requires_pack: 'word-vectors' }),
        params: { model: 'all-mpnet-base-v2' },
      }),
    );
    expect(screen.queryByText('PACK')).toBeNull();
    expect(screen.queryByText('needs pack')).toBeNull();
    unmount();

    // An unloaded catalog (boot) and a server with no Package Center are both
    // "cannot answer", and neither greys anything out.
    for (const state of [{ loaded: false }, { loaded: true, unsupported: true }]) {
      _resetPackStoreForTesting();
      usePackStore.setState({
        byId: { 'word-vectors': wordVectors(), 'sentence-embeddings': sentenceEmbeddings() },
        ...state,
      });
      const view = renderBody(
        baseData({
          definition: makeDef({ ...embeddingDef(), requires_pack: 'word-vectors' }),
          params: { model: 'all-mpnet-base-v2' },
        }),
      );
      expect(screen.queryByText('PACK')).toBeNull();
      expect(screen.queryByText('needs pack')).toBeNull();
      view.unmount();
    }
  });

  it('badges the header in the reader locale', () => {
    useI18n.setState({ locale: 'zh-TW' });
    seedPacks(wordVectors());
    renderBody(baseData({ definition: makeDef({ requires_pack: 'word-vectors' }) }));
    // Sized by its content, not by a fixed width: the zh-TW badge is three
    // characters where the English one is four letters.
    expect(screen.getByRole('button', { name: '需套件' })).toBeInTheDocument();
  });
});
