import type { ReactElement, ReactNode } from 'react';
import { ReactFlowProvider, type Node, type NodeProps } from '@xyflow/react';
import { render, type RenderOptions } from '@testing-library/react';

/**
 * Wrap UI under a {@link ReactFlowProvider} so components that mount React Flow
 * primitives (`Handle`, `useReactFlow`, `useNodeId`, …) have the store context
 * they require. Use for any component in `Nodes/`, `Canvas/`, or anything that
 * renders inside the flow.
 */
export function FlowWrapper({ children }: { children: ReactNode }) {
  return <ReactFlowProvider>{children}</ReactFlowProvider>;
}

/** `render` that mounts the tree inside a {@link ReactFlowProvider}. */
export function renderWithFlow(ui: ReactElement, options?: RenderOptions) {
  return render(ui, { wrapper: FlowWrapper, ...options });
}

/**
 * Build a complete React Flow {@link NodeProps} object for tests. Supplies sane
 * defaults for every required field (including `draggable`, `selectable`,
 * `deletable`, which `@xyflow/react` v12 requires) so individual tests only need
 * to override the fields they care about (typically `id`, `type`, `data`,
 * `selected`). Generic over the node type so callers with a specific data shape
 * (e.g. `Node<LayerNodeData>`) get a correctly-typed `data` field.
 */
export function nodeProps<N extends Node = Node>(
  over: Partial<NodeProps<N>> & { data: N['data'] },
): NodeProps<N> {
  return {
    id: 'n1',
    type: 'node',
    selected: false,
    zIndex: 0,
    isConnectable: true,
    positionAbsoluteX: 0,
    positionAbsoluteY: 0,
    dragging: false,
    draggable: false,
    selectable: true,
    deletable: true,
    ...over,
  } as NodeProps<N>;
}
