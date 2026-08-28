import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, within } from '@testing-library/react';
import type { Node } from '@xyflow/react';
import type { NodeData, NodeDefinition, ParamDefinition } from '../../types';
import type { PackSummary } from '../../api/rest';
import { NodeParamList } from './NodeParamList';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { _resetPackStoreForTesting, usePackStore } from '../../store/packStore';
import { useI18n } from '../../i18n';

// Isolate the list from ParamField's REST/file backends; the mock exposes a
// button so an edit can be triggered without knowing each field's markup.
vi.mock('./ParamField', () => ({
  ParamField: ({ param, value, onChange, siblingParams }: any) => (
    <button
      type="button"
      data-testid={`field-${param.name}`}
      data-siblings={JSON.stringify(siblingParams)}
      onClick={() => onChange(param.name, 'EDITED')}
    >
      {param.name}={String(value)}
    </button>
  ),
}));

function param(over: Partial<ParamDefinition> = {}): ParamDefinition {
  return {
    name: 'p',
    param_type: 'int',
    default: 0,
    description: '',
    options: [],
    min_value: null,
    max_value: null,
    ...over,
  };
}

function def(params: ParamDefinition[]): NodeDefinition {
  return {
    node_name: 'Dense',
    category: 'CNN',
    description: '',
    inputs: [],
    outputs: [],
    params,
  };
}

function seedNode(node: Node<NodeData>) {
  useTabStore.setState((s) => ({
    tabs: s.tabs.map((t) => (t.id === s.activeTabId ? { ...t, nodes: [node] } : t)),
  }));
}

function nodeWith(params: Record<string, unknown>): Node<NodeData> {
  return {
    id: 'n1',
    type: 'baseNode',
    position: { x: 0, y: 0 },
    data: { label: 'N', type: 'Dense', params },
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

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('t');
  // Every case that does not seed a catalog runs against an empty one, which
  // is the base install: no banner anywhere.
  _resetPackStoreForTesting();
  useUIStore.setState({ packCenterOpen: false, packCenterFocusPackId: null });
});

describe('NodeParamList', () => {
  it('renders a field per visible param with its description and range hints', () => {
    seedNode(nodeWith({ lr: 0.5, units: 8 }));
    render(
      <NodeParamList
        nodeId="n1"
        definition={def([
          param({ name: 'lr', description: 'learning rate', min_value: 0, max_value: 1 }),
          param({ name: 'units', min_value: 1 }),
        ])}
        params={{ lr: 0.5, units: 8 }}
      />,
    );
    expect(screen.getByTestId('field-lr')).toHaveTextContent('lr=0.5');
    expect(screen.getByText('learning rate')).toBeInTheDocument();
    expect(screen.getByText('Range: 0 — 1')).toBeInTheDocument();
    expect(screen.getByText('Range: 1 — +∞')).toBeInTheDocument();
  });

  it('commits an edit through updateNodeParams, marking the node dirty', () => {
    seedNode(nodeWith({ lr: 0.5 }));
    render(
      <NodeParamList nodeId="n1" definition={def([param({ name: 'lr' })])} params={{ lr: 0.5 }} />,
    );
    fireEvent.click(screen.getByTestId('field-lr'));
    const tab = useTabStore.getState().getActiveTab();
    expect(tab.nodes[0].data.params.lr).toBe('EDITED');
    expect([...tab.dirtyNodeIds]).toEqual(['n1']);
    // Typing is continuous; it deliberately does not stack undo snapshots.
    expect(tab.undoStack).toHaveLength(0);
  });

  it('commits nothing when there is no node id to write to', () => {
    seedNode(nodeWith({ lr: 0.5 }));
    const spy = vi.spyOn(useTabStore.getState(), 'updateNodeParams');
    render(
      <NodeParamList nodeId={null} definition={def([param({ name: 'lr' })])} params={{ lr: 0.5 }} />,
    );
    fireEvent.click(screen.getByTestId('field-lr'));
    expect(spy).not.toHaveBeenCalled();
  });

  it('hides params whose visible_when rule does not match', () => {
    render(
      <NodeParamList
        nodeId="n1"
        definition={def([
          param({ name: 'preset' }),
          param({ name: 'weights', visible_when: { preset: 'Custom' } }),
        ])}
        params={{ preset: 'Sobel' }}
      />,
    );
    expect(screen.getByTestId('field-preset')).toBeInTheDocument();
    expect(screen.queryByTestId('field-weights')).toBeNull();
  });

  it('renders nothing for a node with no definition', () => {
    const { container } = render(
      <NodeParamList nodeId="n1" definition={undefined} params={{}} />,
    );
    expect(container.querySelectorAll('button')).toHaveLength(0);
  });

  // ── two-tier basic / advanced (core#134) ──────────────────────────────

  it('hides advanced params behind a collapsed section by default', () => {
    render(
      <NodeParamList
        nodeId="n1"
        definition={def([
          param({ name: 'lr' }),
          param({ name: 'betas', advanced: true }),
          param({ name: 'eps', advanced: true }),
        ])}
        params={{ lr: 0.1 }}
      />,
    );
    expect(screen.getByTestId('field-lr')).toBeInTheDocument();
    expect(screen.queryByTestId('field-betas')).toBeNull();
    expect(screen.queryByTestId('field-eps')).toBeNull();
    // The header still says how much is hidden, so nobody has to open it to
    // find out whether there is anything there.
    const toggle = screen.getByRole('button', { name: /Advanced/ });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    expect(toggle).toHaveTextContent('2');
  });

  it('reveals the advanced params when the section is expanded', () => {
    render(
      <NodeParamList
        nodeId="n1"
        definition={def([param({ name: 'lr' }), param({ name: 'betas', advanced: true })])}
        params={{ lr: 0.1, betas: '0.9, 0.999' }}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /Advanced/ }));
    expect(screen.getByTestId('field-betas')).toHaveTextContent('betas=0.9, 0.999');
    expect(screen.getByRole('button', { name: /Advanced/ })).toHaveAttribute(
      'aria-expanded',
      'true',
    );
  });

  it('commits an advanced edit through the same store action', () => {
    seedNode(nodeWith({ betas: '0.9, 0.999' }));
    render(
      <NodeParamList
        nodeId="n1"
        definition={def([param({ name: 'betas', advanced: true })])}
        params={{ betas: '0.9, 0.999' }}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /Advanced/ }));
    fireEvent.click(screen.getByTestId('field-betas'));
    expect(useTabStore.getState().getActiveTab().nodes[0].data.params.betas).toBe('EDITED');
  });

  it('omits the Advanced section entirely when nothing is advanced', () => {
    render(
      <NodeParamList nodeId="n1" definition={def([param({ name: 'lr' })])} params={{ lr: 1 }} />,
    );
    expect(screen.queryByRole('button', { name: /Advanced/ })).toBeNull();
  });

  it('does not count advanced params that visible_when has ruled out', () => {
    // An SGD optimizer must not advertise Adam's hidden knobs.
    render(
      <NodeParamList
        nodeId="n1"
        definition={def([
          param({ name: 'type' }),
          param({ name: 'betas', advanced: true, visible_when: { type: ['Adam', 'AdamW'] } }),
          param({ name: 'nesterov', advanced: true, visible_when: { type: 'SGD' } }),
        ])}
        params={{ type: 'SGD' }}
      />,
    );
    const toggle = screen.getByRole('button', { name: /Advanced/ });
    expect(toggle).toHaveTextContent('1');
    fireEvent.click(toggle);
    expect(screen.getByTestId('field-nesterov')).toBeInTheDocument();
    expect(screen.queryByTestId('field-betas')).toBeNull();
  });

  it('passes the sibling params through and applies an extra class', () => {
    const { container } = render(
      <NodeParamList
        nodeId="n1"
        definition={def([param({ name: 'shape' })])}
        params={{ shape: '2,2', value_mode: 'zeros' }}
        className="extra-class"
      />,
    );
    expect((container.firstChild as HTMLElement).className).toContain('extra-class');
    expect(screen.getByTestId('field-shape').dataset.siblings).toBe(
      JSON.stringify({ shape: '2,2', value_mode: 'zeros' }),
    );
  });

  // ── node-level pack banner (PR 2, F7) ─────────────────────────────────

  it('shows the node-level pack banner with a link when requires_pack is missing', () => {
    seedPacks(packSummary({ id: 'word-vectors', title: 'Word vectors', usable: false }));
    render(
      <NodeParamList
        nodeId="n1"
        definition={{ ...def([param({ name: 'lr' })]), requires_pack: 'word-vectors' }}
        params={{ lr: 0.1 }}
      />,
    );

    const banner = screen.getByRole('note');
    expect(banner).toHaveTextContent('This node needs the Word vectors pack.');
    // A missing pack is a warning, not a reason to hide the configuration:
    // the params stay editable so a saved graph can still be read.
    expect(screen.getByTestId('field-lr')).toBeInTheDocument();

    fireEvent.click(within(banner).getByRole('button', { name: 'Install pack' }));
    expect(useUIStore.getState().packCenterOpen).toBe(true);
    expect(useUIStore.getState().packCenterFocusPackId).toBe('word-vectors');
  });

  it('omits it when the pack is usable', () => {
    seedPacks(packSummary({ id: 'word-vectors', title: 'Word vectors', usable: true }));
    render(
      <NodeParamList
        nodeId="n1"
        definition={{ ...def([param({ name: 'lr' })]), requires_pack: 'word-vectors' }}
        params={{ lr: 0.1 }}
      />,
    );

    expect(screen.queryByRole('note')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Install pack' })).toBeNull();
  });

  it('omits it on a server with no Package Center', () => {
    usePackStore.setState({
      loaded: true,
      unsupported: true,
      byId: { 'word-vectors': packSummary({ id: 'word-vectors', usable: false }) },
    });
    render(
      <NodeParamList
        nodeId="n1"
        definition={{ ...def([param({ name: 'lr' })]), requires_pack: 'word-vectors' }}
        params={{ lr: 0.1 }}
      />,
    );

    expect(screen.queryByRole('note')).toBeNull();
  });
});
