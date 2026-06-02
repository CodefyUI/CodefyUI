import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ReactFlow, ReactFlowProvider, type Edge, type Node, type NodeTypes } from '@xyflow/react';
import { renderWithFlow } from '../../test/utils';
import { useI18n } from '../../i18n';
import { useUIStore } from '../../store/uiStore';
import { useTabStore } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import type { NodeDefinition, NodeData } from '../../types';
import * as rest from '../../api/rest';
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
  useUIStore.setState({ tooltipsEnabled: true, draggingSourceType: null });
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
        subgraphModalNodeId: null,
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

  it('falls back to Utility category color when definition has no category', () => {
    // def.category undefined → category 'Utility'; an unknown category string
    // would exercise the `?? '#607D8B'` fallback.
    const def = makeDef({ category: 'TotallyUnknownCat' });
    const { container } = renderBody(baseData({ definition: def }));
    expect(screen.getByText('TotallyUnknownCat')).toBeTruthy();
    // header background falls back to #607D8B → rgb(96, 125, 139)
    const header = container.querySelector('[class*="header"]') as HTMLElement;
    expect(header.style.background).toBe('rgb(96, 125, 139)');
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

  it('selected node gets white border + glow shadow', () => {
    const { container } = renderBody(baseData(), { selected: true });
    const node = container.querySelector('[class*="node"]') as HTMLElement;
    expect(node.style.getPropertyValue('--border-color')).toBe('#ffffff');
    expect(node.style.boxShadow).toContain('rgba(255,255,255,0.15)');
  });

  it('unselected default node gets the default border + drop shadow', () => {
    const { container } = renderBody(baseData());
    const node = container.querySelector('[class*="node"]') as HTMLElement;
    expect(node.style.getPropertyValue('--border-color')).toBe('#444444');
    expect(node.style.boxShadow).toContain('rgba(0,0,0,0.4)');
  });

  it.each([
    // `--border-color` is a CSS custom property; jsdom does NOT normalize
    // custom-prop values, so the raw hex is preserved.
    ['running', '#FFC107'],
    ['completed', '#4CAF50'],
    ['error', '#F44336'],
    ['cached', '#2196F3'],
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

  // ── SequentialModel branch ────────────────────────────────────────────────

  it('renders SequentialModel layer count + hint and opens subgraph on dbl-click', () => {
    const data = baseData({
      type: 'SequentialModel',
      params: { layers: JSON.stringify([{}, {}, {}]) },
    });
    const { container } = renderBody(data);
    expect(screen.getByText('3')).toBeTruthy();
    expect(screen.getByText(useI18n.getState().t('subgraph.hint'))).toBeTruthy();
    // cursor: pointer for sequential models
    const node = container.querySelector('[class*="node"]') as HTMLElement;
    expect(node.style.cursor).toBe('pointer');
    // Double-click opens the subgraph modal
    fireEvent.click(node, { detail: 2 });
    expect(useTabStore.getState().getActiveTab().subgraphModalNodeId).toBe('n1');
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
    expect(useTabStore.getState().getActiveTab().subgraphModalNodeId).toBeNull();
  });

  it('double-click on a non-sequential node does not open the modal', () => {
    const { container } = renderBody(baseData());
    fireEvent.click(container.querySelector('[class*="node"]') as HTMLElement, { detail: 2 });
    expect(useTabStore.getState().getActiveTab().subgraphModalNodeId).toBeNull();
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

afterEach(() => {
  vi.restoreAllMocks();
});
