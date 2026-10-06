import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { Edge, Node } from '@xyflow/react';
import type { NodeData, NodeDefinition } from '../../types';
import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import { CapturesTab } from './CapturesTab';
import type { NodeDetailTabContext } from './tabs';

// ── One port in two rows, before anything has run (#562) ─────────────────────
// One output wired into two inputs of the node (`x` into both inputs of Add)
// gives two rows for the same port. After a run the rows are PortGroup's; before
// one, this tab lists the shapes the last validation streamed, one row a port.

const tensor = { data_type: 'TENSOR', description: '', optional: false };

function node(id: string, label: string, definition: NodeDefinition): Node<NodeData> {
  return { id, type: 'baseNode', position: { x: 0, y: 0 }, data: { label, type: definition.node_name, params: {}, definition } };
}

const nodes = [
  node('src', 'Src', {
    node_name: 'Src',
    category: 'Utility',
    description: '',
    inputs: [],
    outputs: [{ name: 'out', ...tensor }, { name: 'y', ...tensor }],
    params: [],
  }),
  node('add', 'Add', {
    node_name: 'Add',
    category: 'Utility',
    description: '',
    inputs: [{ name: 'a', ...tensor }, { name: 'b', ...tensor }],
    outputs: [],
    params: [],
  }),
  node('other', 'Other', {
    node_name: 'Other',
    category: 'Utility',
    description: '',
    inputs: [{ name: 'x', ...tensor }],
    outputs: [],
    params: [],
  }),
];

const edges: Edge[] = [
  { id: 'e1', source: 'src', sourceHandle: 'out', target: 'add', targetHandle: 'a' },
  { id: 'e2', source: 'src', sourceHandle: 'out', target: 'add', targetHandle: 'b' },
  { id: 'e3', source: 'src', sourceHandle: 'y', target: 'other', targetHandle: 'x' },
];

/** The modal's context for `nodeId` before any run. */
function before(nodeId: string): NodeDetailTabContext {
  return {
    nodeId,
    node: nodes.find((n) => n.id === nodeId)!,
    runId: null,
    nodes,
    edges,
    recordOutputs: true,
    outputSummaries: {
      src: {
        out: { type: 'tensor', shape: [2, 3], dtype: 'float32' },
        y: { type: 'tensor', shape: [5], dtype: 'float32' },
      },
    },
    focusPort: null,
  };
}

const rows = () => document.querySelectorAll('[class*="summaryRow"]');

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  const id = 'tab-captures';
  useTabStore.setState((s) => ({
    activeTabId: id,
    tabs: [{ ...s.tabs[0], id, nodes, edges, logs: [], outputSummaries: {} }],
  }));
});

describe('CapturesTab before a run — the same port in two rows', () => {
  it('lists both rows with their shapes and no duplicate-key warning', () => {
    const error = vi.spyOn(console, 'error').mockImplementation(() => {});
    try {
      render(<CapturesTab kind="input" ctx={before('add')} />);
      expect(rows()).toHaveLength(2);
      expect(screen.getAllByText('Src.out')).toHaveLength(2);
      expect(screen.getAllByText('tensor · [2, 3] · float32')).toHaveLength(2);
      expect(error.mock.calls.flat().join('\n')).not.toMatch(/same key/);
    } finally {
      error.mockRestore();
    }
  });

  it("leaves only the next node's rows when the modal moves on", () => {
    const { rerender } = render(<CapturesTab kind="input" ctx={before('add')} />);
    rerender(<CapturesTab kind="input" ctx={before('other')} />);
    expect(rows()).toHaveLength(1);
    expect(screen.queryByText('Src.out')).toBeNull();
    expect(screen.getByText('Src.y')).toBeInTheDocument();
  });
});
