import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useState } from 'react';
import { act, fireEvent, render } from '@testing-library/react';
import {
  ReactFlow,
  ReactFlowProvider,
  applyEdgeChanges,
  applyNodeChanges,
  type Edge,
  type Node,
  type OnBeforeDelete,
} from '@xyflow/react';
import { useDeleteKey } from './useDeleteKey';

// A real <ReactFlow> with its own Delete binding off, as both canvases have
// it, and this hook in its place (#501).

/** What the canvas holds after the last render. */
let shown: { nodes: Node[]; edges: Edge[] } = { nodes: [], edges: [] };

function Canvas({
  initialNodes,
  initialEdges,
  onBeforeDelete,
}: {
  initialNodes: Node[];
  initialEdges: Edge[];
  onBeforeDelete?: OnBeforeDelete;
}) {
  const [nodes, setNodes] = useState(initialNodes);
  const [edges, setEdges] = useState(initialEdges);
  shown = { nodes, edges };
  useDeleteKey();
  return (
    <div style={{ width: 800, height: 600 }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={(changes) => setNodes((prev) => applyNodeChanges(changes, prev))}
        onEdgesChange={(changes) => setEdges((prev) => applyEdgeChanges(changes, prev))}
        deleteKeyCode={null}
        onBeforeDelete={onBeforeDelete}
      />
    </div>
  );
}

function node(id: string, selected = false): Node {
  return { id, position: { x: 0, y: 0 }, data: { label: id }, selected };
}

function mount(
  nodes: Node[] = [node('a', true), node('b')],
  edges: Edge[] = [],
  onBeforeDelete?: OnBeforeDelete,
) {
  return render(
    <ReactFlowProvider>
      <Canvas initialNodes={nodes} initialEdges={edges} onBeforeDelete={onBeforeDelete} />
    </ReactFlowProvider>,
  );
}

const ids = (items: { id: string }[]) => items.map((item) => item.id);

/** Let the deletion run out: it awaits `onBeforeDelete` first. */
async function settle() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

/** A key pressed on `target` and let go, with the deletion run out. */
async function press(init: KeyboardEventInit, target: Element = document.body) {
  const notPrevented = fireEvent.keyDown(target, init);
  await settle();
  fireEvent.keyUp(target, init);
  return notPrevented;
}

const DELETE: KeyboardEventInit = { key: 'Delete', code: 'Delete' };

beforeEach(() => {
  shown = { nodes: [], edges: [] };
});

describe('useDeleteKey', () => {
  it('deletes the selected nodes with their edges, and the selected edges', async () => {
    mount(
      [node('a', true), node('b'), node('c'), node('d')],
      [
        { id: 'ab', source: 'a', target: 'b' },
        { id: 'bc', source: 'b', target: 'c', selected: true },
        { id: 'cd', source: 'c', target: 'd' },
      ],
    );
    const notPrevented = await press(DELETE);
    expect(ids(shown.nodes)).toEqual(['b', 'c', 'd']);
    expect(ids(shown.edges)).toEqual(['cd']);
    // The key is claimed, as React Flow's own binding claimed it.
    expect(notPrevented).toBe(false);
  });

  it('deletes on the first Delete after a shifted key whose Shift came up first', async () => {
    // `?` goes down with Shift held and comes up as `/` once Shift is up.
    // React Flow's own binding kept the `?` as held and ignored this Delete.
    mount();
    fireEvent.keyDown(document.body, { key: 'Shift', code: 'ShiftLeft', shiftKey: true });
    fireEvent.keyDown(document.body, { key: '?', code: 'Slash', shiftKey: true });
    fireEvent.keyUp(document.body, { key: 'Shift', code: 'ShiftLeft' });
    fireEvent.keyUp(document.body, { key: '/', code: 'Slash' });
    await press(DELETE);
    expect(ids(shown.nodes)).toEqual(['b']);
  });

  it.each([
    ['Shift', { shiftKey: true }],
    ['Ctrl', { ctrlKey: true }],
    ['Alt', { altKey: true }],
    ['Meta', { metaKey: true }],
  ])('leaves %s+Delete alone, as React Flow did', async (_modifier, held) => {
    mount();
    const notPrevented = await press({ ...DELETE, ...held });
    expect(ids(shown.nodes)).toEqual(['a', 'b']);
    expect(notPrevented).toBe(true);
  });

  it('does nothing on Backspace', async () => {
    mount();
    await press({ key: 'Backspace', code: 'Backspace' });
    expect(ids(shown.nodes)).toEqual(['a', 'b']);
  });

  it.each([
    ['an input', () => document.createElement('input')],
    ['a textarea', () => document.createElement('textarea')],
    ['a select', () => document.createElement('select')],
    [
      'an editable element',
      () => {
        const div = document.createElement('div');
        // jsdom does not compute isContentEditable from the attribute.
        Object.defineProperty(div, 'isContentEditable', { value: true });
        return div;
      },
    ],
  ])('leaves a Delete typed into %s to it', async (_what, make) => {
    mount();
    const field = make();
    document.body.appendChild(field);
    const notPrevented = await press(DELETE, field);
    expect(ids(shown.nodes)).toEqual(['a', 'b']);
    expect(notPrevented).toBe(true);
    field.remove();
  });

  it('deletes once per press, however long Delete is held', async () => {
    const onBeforeDelete = vi.fn(async () => true);
    mount(undefined, undefined, onBeforeDelete);
    fireEvent.keyDown(document.body, DELETE);
    fireEvent.keyDown(document.body, { ...DELETE, repeat: true });
    fireEvent.keyDown(document.body, { ...DELETE, repeat: true });
    await settle();
    fireEvent.keyUp(document.body, DELETE);
    expect(onBeforeDelete).toHaveBeenCalledTimes(1);
  });

  it("asks the canvas's onBeforeDelete, and keeps everything when it says no", async () => {
    // How each canvas refuses a Delete while a modal is in the way (#475).
    const onBeforeDelete = vi.fn(async () => false);
    mount(undefined, undefined, onBeforeDelete);
    await press(DELETE);
    expect(onBeforeDelete).toHaveBeenCalledWith({
      nodes: [expect.objectContaining({ id: 'a' })],
      edges: [],
    });
    expect(ids(shown.nodes)).toEqual(['a', 'b']);
  });

  it('stops listening when the canvas goes away', async () => {
    const onBeforeDelete = vi.fn(async () => true);
    const { unmount } = mount(undefined, undefined, onBeforeDelete);
    unmount();
    await press(DELETE);
    expect(onBeforeDelete).not.toHaveBeenCalled();
  });
});
