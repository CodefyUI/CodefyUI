import { describe, it, expect } from 'vitest';
import { screen } from '@testing-library/react';
import type { Node } from '@xyflow/react';
import { OutputNode } from './OutputNode';
import { nodeProps, renderWithFlow } from '../../test/utils';
import type { LayerNodeData } from './graphSerialization';

function makeData(overrides: Partial<LayerNodeData> = {}): LayerNodeData {
  return {
    layerType: 'Output',
    params: {},
    color: '#F44336',
    isBoundary: true,
    ...overrides,
  };
}

function renderOutput(data: LayerNodeData, selected = false) {
  return renderWithFlow(
    <OutputNode
      {...nodeProps<Node<LayerNodeData>>({ id: 'out1', type: 'outputNode', data, selected })}
    />,
  );
}

describe('OutputNode', () => {
  it('renders the Output footer label', () => {
    renderOutput(makeData());
    expect(screen.getByText('Output')).toBeTruthy();
  });

  it('renders one port label and one target handle per port', () => {
    const { container } = renderOutput(
      makeData({
        ports: [
          { id: 'p1', name: 'y' },
          { id: 'p2', name: 'y2' },
        ],
      }),
    );
    expect(screen.getByText('y')).toBeTruthy();
    expect(screen.getByText('y2')).toBeTruthy();
    expect(container.querySelectorAll('.react-flow__handle').length).toBe(2);
  });

  it('handles missing ports (undefined) by rendering no handles', () => {
    const { container } = renderOutput(makeData());
    expect(container.querySelectorAll('.react-flow__handle').length).toBe(0);
  });

  it('applies the selected styling when selected', () => {
    const { container } = renderOutput(makeData({ ports: [{ id: 'p1', name: 'y' }] }), true);
    const root = container.querySelector('div') as HTMLElement;
    expect(root.style.border).toContain('rgb(255, 255, 255)');
    expect(root.style.boxShadow).toContain('#F4433644');
  });

  it('applies the unselected styling when not selected', () => {
    const { container } = renderOutput(makeData({ ports: [{ id: 'p1', name: 'y' }] }), false);
    const root = container.querySelector('div') as HTMLElement;
    // #F4433688 normalizes to rgba(244, 67, 54, ...).
    expect(root.style.border).toContain('rgba(244, 67, 54');
    expect(root.style.boxShadow).toContain('rgba(0,0,0,0.4)');
  });
});
