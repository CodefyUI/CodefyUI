import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { PresetConfigModal } from './PresetConfigModal';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import type { Node } from '@xyflow/react';
import type { NodeData, ParamDefinition, PresetDefinition } from '../../types';

function resetToSingleTab() {
  useTabStore.setState({
    tabs: [],
    activeTabId: null as unknown as string,
    clipboard: null,
  });
  useTabStore.getState().addTab('Tab 1');
}

function intParam(name: string, def: number): ParamDefinition {
  return {
    name,
    param_type: 'int',
    default: def,
    description: '',
    options: [],
    min_value: null,
    max_value: null,
  };
}

function selectParam(name: string, options: string[], def: string): ParamDefinition {
  return {
    name,
    param_type: 'select',
    default: def,
    description: '',
    options,
    min_value: null,
    max_value: null,
  };
}

function makePreset(overrides: Partial<PresetDefinition> = {}): PresetDefinition {
  return {
    preset_name: 'MLP Block',
    category: 'CNN',
    description: 'A small MLP',
    tags: ['beginner'],
    nodes: [
      { id: 'lin1', type: 'Linear', params: {} },
      { id: 'act1', type: 'ReLU', params: {} },
    ],
    edges: [],
    exposed_inputs: [],
    exposed_outputs: [],
    exposed_params: [
      {
        internal_node: 'lin1',
        param_name: 'units',
        display_name: 'Units',
        group: 'Architecture',
        param_def: intParam('units', 64),
      },
      {
        internal_node: 'act1',
        param_name: 'kind',
        display_name: 'Activation',
        group: '', // falls back to the general group
        param_def: selectParam('kind', ['relu', 'gelu'], 'relu'),
      },
    ],
    ...overrides,
  };
}

/** Mount a preset node on the active tab and open the preset modal for it. */
function mountPresetNode(preset: PresetDefinition, internalParams?: Record<string, Record<string, any>>) {
  const tabId = useTabStore.getState().activeTabId;
  const nodeId = 'preset-node-1';
  const node: Node<NodeData> = {
    id: nodeId,
    type: 'presetNode',
    position: { x: 0, y: 0 },
    data: {
      label: preset.preset_name,
      type: `preset:${preset.preset_name}`,
      params: {},
      isPreset: true,
      presetDefinition: preset,
      internalParams: internalParams ?? {},
    },
  };
  useTabStore.setState({
    tabs: useTabStore.getState().tabs.map((t) =>
      t.id === tabId
        ? { ...t, nodes: [node], presetModalNodeId: nodeId }
        : t,
    ),
  });
  return nodeId;
}

// Fresh action mocks installed on the store each test so assertions are
// isolated and the real implementations never mutate shared store state.
let updateMock: ReturnType<typeof vi.fn>;
let closeMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  resetToSingleTab();
  updateMock = vi.fn();
  closeMock = vi.fn();
  useTabStore.setState({
    updatePresetInternalParam: updateMock as never,
    closePresetModal: closeMock as never,
  });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('PresetConfigModal', () => {
  it('renders nothing when no preset modal node is open (early return)', () => {
    const { container } = render(<PresetConfigModal />);
    expect(container.firstChild).toBeNull();
  });

  it('renders nothing when the open node id has no matching node', () => {
    const tabId = useTabStore.getState().activeTabId;
    useTabStore.setState({
      tabs: useTabStore.getState().tabs.map((t) =>
        t.id === tabId ? { ...t, presetModalNodeId: 'does-not-exist' } : t,
      ),
    });
    const { container } = render(<PresetConfigModal />);
    expect(container.firstChild).toBeNull();
  });

  it('renders nothing when the node is not a preset (no presetDefinition)', () => {
    const tabId = useTabStore.getState().activeTabId;
    const node: Node<NodeData> = {
      id: 'plain',
      type: 'baseNode',
      position: { x: 0, y: 0 },
      data: { label: 'P', type: 'X', params: {} },
    };
    useTabStore.setState({
      tabs: useTabStore.getState().tabs.map((t) =>
        t.id === tabId ? { ...t, nodes: [node], presetModalNodeId: 'plain' } : t,
      ),
    });
    const { container } = render(<PresetConfigModal />);
    expect(container.firstChild).toBeNull();
  });

  it('renders header, pipeline preview and grouped params', () => {
    mountPresetNode(makePreset());
    render(<PresetConfigModal />);
    // Header.
    expect(screen.getByText('MLP Block')).toBeTruthy();
    expect(screen.getByText('PRESET')).toBeTruthy();
    expect(screen.getByText('A small MLP')).toBeTruthy();
    // Pipeline chips for each internal node, with arrow between them.
    expect(screen.getByText('Linear')).toBeTruthy();
    expect(screen.getByText('ReLU')).toBeTruthy();
    expect(screen.getByText('→')).toBeTruthy();
    // Groups: explicit "Architecture" and fallback "General".
    expect(screen.getByText('Architecture')).toBeTruthy();
    expect(screen.getByText('General')).toBeTruthy();
    // Param labels rendered via ParamField.
    expect(screen.getByText('Units')).toBeTruthy();
    expect(screen.getByText('Activation')).toBeTruthy();
  });

  it('uses internalParams value over the param default when present', () => {
    mountPresetNode(makePreset(), { lin1: { units: 999 } });
    render(<PresetConfigModal />);
    const unitsInput = screen.getByDisplayValue('999');
    expect(unitsInput).toBeTruthy();
  });

  it('editing a param then Apply writes each edited value and closes the modal', () => {
    const nodeId = mountPresetNode(makePreset());
    render(<PresetConfigModal />);

    // Change the Units (int) field.
    const unitsInput = screen.getByDisplayValue('64');
    fireEvent.change(unitsInput, { target: { value: '128' } });

    // Change the Activation (select) field.
    const select = screen.getByDisplayValue('relu') as HTMLSelectElement;
    fireEvent.change(select, { target: { value: 'gelu' } });

    fireEvent.click(screen.getByText('Apply'));

    expect(updateMock).toHaveBeenCalledWith(nodeId, 'lin1', 'units', 128);
    expect(updateMock).toHaveBeenCalledWith(nodeId, 'act1', 'kind', 'gelu');
    expect(closeMock).toHaveBeenCalled();
  });

  it('Apply with no edits still closes (loops over empty localParams)', () => {
    mountPresetNode(makePreset());
    render(<PresetConfigModal />);
    fireEvent.click(screen.getByText('Apply'));
    expect(updateMock).not.toHaveBeenCalled();
    expect(closeMock).toHaveBeenCalled();
  });

  it('Cancel button closes without applying', () => {
    mountPresetNode(makePreset());
    render(<PresetConfigModal />);
    fireEvent.click(screen.getByText('Cancel'));
    expect(updateMock).not.toHaveBeenCalled();
    expect(closeMock).toHaveBeenCalled();
  });

  it('the ✕ header button closes the modal', () => {
    mountPresetNode(makePreset());
    render(<PresetConfigModal />);
    fireEvent.click(screen.getByText('✕'));
    expect(closeMock).toHaveBeenCalled();
  });

  it('clicking the overlay backdrop closes; clicking inside the modal does not', () => {
    mountPresetNode(makePreset());
    const { container } = render(<PresetConfigModal />);
    const overlay = container.firstChild as HTMLElement;

    // Click inside the modal (target !== overlay) → no close.
    fireEvent.click(screen.getByText('MLP Block'));
    expect(closeMock).not.toHaveBeenCalled();

    // Click the overlay itself → close.
    fireEvent.click(overlay);
    expect(closeMock).toHaveBeenCalledTimes(1);
  });

  it('skips exposed params whose param_def is null', () => {
    const preset = makePreset({
      exposed_params: [
        {
          internal_node: 'lin1',
          param_name: 'ghost',
          display_name: 'Ghost',
          group: 'Architecture',
          param_def: null,
        },
        {
          internal_node: 'lin1',
          param_name: 'units',
          display_name: 'Units',
          group: 'Architecture',
          param_def: intParam('units', 64),
        },
      ],
    });
    mountPresetNode(preset);
    render(<PresetConfigModal />);
    // The null-param row renders nothing for "Ghost".
    expect(screen.queryByText('Ghost')).toBeNull();
    expect(screen.getByText('Units')).toBeTruthy();
  });

  it('renders a single-node preset without a trailing arrow', () => {
    const preset = makePreset({
      nodes: [{ id: 'only', type: 'Solo', params: {} }],
      exposed_params: [],
    });
    mountPresetNode(preset);
    render(<PresetConfigModal />);
    expect(screen.getByText('Solo')).toBeTruthy();
    expect(screen.queryByText('→')).toBeNull();
  });

  it('reinitializes local params when the open node id changes', () => {
    // Mount node A with units=10.
    const tabId = useTabStore.getState().activeTabId;
    const nodeA: Node<NodeData> = {
      id: 'A',
      type: 'presetNode',
      position: { x: 0, y: 0 },
      data: {
        label: 'PA',
        type: 'preset:PA',
        params: {},
        isPreset: true,
        presetDefinition: makePreset({ preset_name: 'PA' }),
        internalParams: { lin1: { units: 10 } },
      },
    };
    const nodeB: Node<NodeData> = {
      id: 'B',
      type: 'presetNode',
      position: { x: 0, y: 0 },
      data: {
        label: 'PB',
        type: 'preset:PB',
        params: {},
        isPreset: true,
        presetDefinition: makePreset({ preset_name: 'PB' }),
        internalParams: { lin1: { units: 20 } },
      },
    };
    useTabStore.setState({
      tabs: useTabStore.getState().tabs.map((t) =>
        t.id === tabId
          ? { ...t, nodes: [nodeA, nodeB], presetModalNodeId: 'A' }
          : t,
      ),
    });
    const { rerender } = render(<PresetConfigModal />);
    expect(screen.getByDisplayValue('10')).toBeTruthy();

    // Switch the modal to node B; the effect re-seeds local params.
    useTabStore.setState({
      tabs: useTabStore.getState().tabs.map((t) =>
        t.id === tabId ? { ...t, presetModalNodeId: 'B' } : t,
      ),
    });
    rerender(<PresetConfigModal />);
    expect(screen.getByDisplayValue('20')).toBeTruthy();
  });
});
