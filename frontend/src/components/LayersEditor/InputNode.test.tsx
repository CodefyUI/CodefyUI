import { describe, it, expect } from 'vitest';
import { screen } from '@testing-library/react';
import type { Node } from '@xyflow/react';
import { InputNode } from './InputNode';
import { nodeProps, renderWithFlow } from '../../test/utils';
import { colorForType } from './graphSerialization';
import type { LayerNodeData } from './graphSerialization';

function makeData(overrides: Partial<LayerNodeData> = {}): LayerNodeData {
  return {
    layerType: 'Input',
    params: {},
    color: colorForType('Input'),
    isBoundary: true,
    ...overrides,
  };
}

function renderInput(data: LayerNodeData, selected = false) {
  return renderWithFlow(
    <InputNode
      {...nodeProps<Node<LayerNodeData>>({ id: 'in1', type: 'inputNode', data, selected })}
    />,
  );
}

describe('InputNode', () => {
  it('renders the Input header label', () => {
    renderInput(makeData());
    expect(screen.getByText('Input')).toBeTruthy();
  });

  it('renders one port label and one source handle per port', () => {
    const { container } = renderInput(
      makeData({
        ports: [
          { id: 'p1', name: 'x' },
          { id: 'p2', name: 'x2' },
        ],
      }),
    );
    expect(screen.getByText('x')).toBeTruthy();
    expect(screen.getByText('x2')).toBeTruthy();
    const handles = container.querySelectorAll('.react-flow__handle');
    expect(handles.length).toBe(2);
  });

  it('handles missing ports (undefined) by rendering no handles', () => {
    // Covers the `data.ports ?? []` fallback branch.
    const { container } = renderInput(makeData());
    expect(container.querySelectorAll('.react-flow__handle').length).toBe(0);
  });

  it('applies the selected border/box-shadow styling when selected', () => {
    const { container } = renderInput(makeData({ ports: [{ id: 'p1', name: 'x' }] }), true);
    const root = container.querySelector('div') as HTMLElement;
    // Selected uses the white border branch (jsdom normalizes #fff -> rgb()).
    expect(root.style.border).toContain('rgb(255, 255, 255)');
    expect(root.style.boxShadow).toContain(colorForType('Input') + '44');
  });

  it('applies the unselected border/box-shadow styling when not selected', () => {
    const { container } = renderInput(makeData({ ports: [{ id: 'p1', name: 'x' }] }), false);
    const root = container.querySelector('div') as HTMLElement;
    // Unselected: the layer hue at full strength outlines the card. It used
    // to be drawn at 53% alpha, which composited to 2.9-3.8:1 against the
    // canvas behind it (core#228).
    expect(root.style.border).toContain('rgb(76, 175, 80)');
    expect(root.style.boxShadow).toContain('rgba(0,0,0,0.4)');
  });
});
