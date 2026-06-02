import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, within } from '@testing-library/react';
import { PortListEditor } from './PortListEditor';
import { useI18n } from '../../i18n';
import type { Node, Edge } from '@xyflow/react';
import type { LayerNodeData, PortDef } from './graphSerialization';

function makeNode(
  layerType: string,
  ports: PortDef[] | undefined,
  id = 'node1',
): Node<LayerNodeData> {
  return {
    id,
    type: layerType === 'Input' ? 'inputNode' : 'outputNode',
    position: { x: 0, y: 0 },
    data: {
      layerType,
      params: {},
      color: '#000',
      ports,
      isBoundary: true,
    },
  };
}

describe('PortListEditor', () => {
  beforeEach(() => {
    useI18n.setState({ locale: 'en' });
  });

  it('renders the header with layer type and the localized port list label', () => {
    const node = makeNode('Input', [{ id: 'p1', name: 'x' }]);
    render(
      <PortListEditor node={node} edges={[]} onUpdatePorts={vi.fn()} onRemoveEdges={vi.fn()} />,
    );
    expect(screen.getByText(/Input/)).toBeTruthy();
    expect(screen.getByText(/Ports/)).toBeTruthy();
  });

  it('falls back to an empty port list when ports is undefined', () => {
    // Covers `node.data.ports ?? []`.
    const node = makeNode('Input', undefined);
    render(
      <PortListEditor node={node} edges={[]} onUpdatePorts={vi.fn()} onRemoveEdges={vi.fn()} />,
    );
    // No text inputs rendered for ports.
    expect(screen.queryAllByRole('textbox').length).toBe(0);
  });

  it('renders one input per port with its current value', () => {
    const node = makeNode('Input', [
      { id: 'p1', name: 'x' },
      { id: 'p2', name: 'y' },
    ]);
    render(
      <PortListEditor node={node} edges={[]} onUpdatePorts={vi.fn()} onRemoveEdges={vi.fn()} />,
    );
    const inputs = screen.getAllByRole('textbox') as HTMLInputElement[];
    expect(inputs.map((i) => i.value)).toEqual(['x', 'y']);
  });

  it('setName: editing a port name calls onUpdatePorts with the updated list', () => {
    const onUpdatePorts = vi.fn();
    const node = makeNode('Input', [
      { id: 'p1', name: 'x' },
      { id: 'p2', name: 'y' },
    ]);
    render(
      <PortListEditor node={node} edges={[]} onUpdatePorts={onUpdatePorts} onRemoveEdges={vi.fn()} />,
    );
    const inputs = screen.getAllByRole('textbox');
    fireEvent.change(inputs[0], { target: { value: 'renamed' } });
    expect(onUpdatePorts).toHaveBeenCalledWith('node1', [
      { id: 'p1', name: 'renamed' },
      { id: 'p2', name: 'y' },
    ]);
  });

  it('addPort: appends a new port named portN+1', () => {
    const onUpdatePorts = vi.fn();
    const node = makeNode('Input', [{ id: 'p1', name: 'x' }]);
    render(
      <PortListEditor node={node} edges={[]} onUpdatePorts={onUpdatePorts} onRemoveEdges={vi.fn()} />,
    );
    fireEvent.click(screen.getByText('+ Add port'));
    expect(onUpdatePorts).toHaveBeenCalledTimes(1);
    const [, nextPorts] = onUpdatePorts.mock.calls[0];
    expect(nextPorts).toHaveLength(2);
    expect(nextPorts[0]).toEqual({ id: 'p1', name: 'x' });
    expect(nextPorts[1].name).toBe('port2');
    expect(typeof nextPorts[1].id).toBe('string');
  });

  it('removePort (Input): removes the port and orphaned source-handle edges', () => {
    const onUpdatePorts = vi.fn();
    const onRemoveEdges = vi.fn();
    const node = makeNode('Input', [
      { id: 'p1', name: 'x' },
      { id: 'p2', name: 'y' },
    ]);
    const edges: Edge[] = [
      { id: 'e1', source: 'node1', sourceHandle: 'p1', target: 'other', targetHandle: null },
      { id: 'e2', source: 'node1', sourceHandle: 'p2', target: 'other', targetHandle: null },
      // Unrelated edge: should NOT be orphaned.
      { id: 'e3', source: 'zzz', sourceHandle: 'p1', target: 'node1', targetHandle: 'p1' },
    ];
    render(
      <PortListEditor
        node={node}
        edges={edges}
        onUpdatePorts={onUpdatePorts}
        onRemoveEdges={onRemoveEdges}
      />,
    );
    // The first port row's Remove button.
    const rows = screen.getAllByRole('textbox').map((i) => i.closest('div')!);
    const firstRemove = within(rows[0]).getByRole('button');
    fireEvent.click(firstRemove);

    expect(onUpdatePorts).toHaveBeenCalledWith('node1', [{ id: 'p2', name: 'y' }]);
    // Only e1 references node1 as source with handle p1.
    expect(onRemoveEdges).toHaveBeenCalledWith(['e1']);
  });

  it('removePort (Output): removes orphaned target-handle edges', () => {
    const onUpdatePorts = vi.fn();
    const onRemoveEdges = vi.fn();
    const node = makeNode('Output', [
      { id: 'p1', name: 'a' },
      { id: 'p2', name: 'b' },
    ]);
    const edges: Edge[] = [
      { id: 'e1', source: 'src', sourceHandle: null, target: 'node1', targetHandle: 'p1' },
      // wrong handle, should be kept
      { id: 'e2', source: 'src', sourceHandle: null, target: 'node1', targetHandle: 'p2' },
    ];
    render(
      <PortListEditor
        node={node}
        edges={edges}
        onUpdatePorts={onUpdatePorts}
        onRemoveEdges={onRemoveEdges}
      />,
    );
    const rows = screen.getAllByRole('textbox').map((i) => i.closest('div')!);
    const firstRemove = within(rows[0]).getByRole('button');
    fireEvent.click(firstRemove);

    expect(onUpdatePorts).toHaveBeenCalledWith('node1', [{ id: 'p2', name: 'b' }]);
    expect(onRemoveEdges).toHaveBeenCalledWith(['e1']);
  });

  it('removePort: does NOT call onRemoveEdges when no edges reference the port', () => {
    // Covers the `if (orphaned.length > 0)` false branch.
    const onUpdatePorts = vi.fn();
    const onRemoveEdges = vi.fn();
    const node = makeNode('Input', [
      { id: 'p1', name: 'x' },
      { id: 'p2', name: 'y' },
    ]);
    render(
      <PortListEditor
        node={node}
        edges={[]}
        onUpdatePorts={onUpdatePorts}
        onRemoveEdges={onRemoveEdges}
      />,
    );
    const rows = screen.getAllByRole('textbox').map((i) => i.closest('div')!);
    fireEvent.click(within(rows[0]).getByRole('button'));
    expect(onUpdatePorts).toHaveBeenCalled();
    expect(onRemoveEdges).not.toHaveBeenCalled();
  });

  it('disables the Remove button when only one port remains', () => {
    const node = makeNode('Input', [{ id: 'p1', name: 'x' }]);
    render(
      <PortListEditor node={node} edges={[]} onUpdatePorts={vi.fn()} onRemoveEdges={vi.fn()} />,
    );
    const rows = screen.getAllByRole('textbox').map((i) => i.closest('div')!);
    const removeBtn = within(rows[0]).getByRole('button') as HTMLButtonElement;
    expect(removeBtn.disabled).toBe(true);
    expect(removeBtn.style.cursor).toBe('not-allowed');
    expect(removeBtn.style.opacity).toBe('0.4');
  });

  it('marks duplicate port names with the error styling and title', () => {
    const node = makeNode('Input', [
      { id: 'p1', name: 'dup' },
      { id: 'p2', name: 'dup' },
      { id: 'p3', name: 'unique' },
    ]);
    render(
      <PortListEditor node={node} edges={[]} onUpdatePorts={vi.fn()} onRemoveEdges={vi.fn()} />,
    );
    const inputs = screen.getAllByRole('textbox') as HTMLInputElement[];
    // Duplicates get the red border + title (jsdom normalizes #F44336 -> rgb()).
    expect(inputs[0].style.border).toContain('rgb(244, 67, 54)');
    expect(inputs[0].title).toBe('Duplicate port name');
    expect(inputs[1].style.border).toContain('rgb(244, 67, 54)');
    // Unique one gets the normal border + no title (#444 -> rgb(68, 68, 68)).
    expect(inputs[2].style.border).toContain('rgb(68, 68, 68)');
    expect(inputs[2].title).toBe('');
  });
});
