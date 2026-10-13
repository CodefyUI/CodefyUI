import { describe, it, expect, beforeEach } from 'vitest';
import { screen } from '@testing-library/react';
import type { Node } from '@xyflow/react';
import { LayerNode } from './LayerNode';
import { nodeProps, renderWithFlow } from '../../test/utils';
import { useI18n } from '../../i18n';
import { colorForType } from './graphSerialization';
import type { LayerNodeData } from './graphSerialization';

function makeData(overrides: Partial<LayerNodeData> = {}): LayerNodeData {
  return {
    layerType: 'Conv2d',
    params: {},
    color: colorForType('Conv2d'),
    ...overrides,
  };
}

function renderLayer(data: LayerNodeData, selected = false) {
  return renderWithFlow(
    <LayerNode
      {...nodeProps<Node<LayerNodeData>>({ id: 'l1', type: 'layerNode', data, selected })}
    />,
  );
}

describe('LayerNode', () => {
  beforeEach(() => {
    useI18n.setState({ locale: 'en' });
  });

  it('renders the layer type as the header', () => {
    renderLayer(makeData({ layerType: 'Linear' }));
    expect(screen.getByText('Linear')).toBeTruthy();
  });

  it('renders both target and source handles', () => {
    const { container } = renderLayer(makeData());
    // One target (top) + one source (bottom).
    expect(container.querySelectorAll('.react-flow__handle').length).toBe(2);
  });

  it('labels the two operand handles of a Subtract a and b', () => {
    const { container } = renderLayer(makeData({ layerType: 'Subtract', isMerge: true }));
    const targets = container.querySelectorAll('.react-flow__handle.target');
    expect([...targets].map((h) => h.getAttribute('data-handleid'))).toEqual(['a', 'b']);
    expect(screen.getByTestId('operand-label-a').textContent).toBe('a');
    expect(screen.getByTestId('operand-label-b').textContent).toBe('b');
  });

  it('gives an unwired Concat two labelled operand handles', () => {
    const { container } = renderLayer(makeData({ layerType: 'Concat', isMerge: true, params: { dim: 1 } }));
    const targets = container.querySelectorAll('.react-flow__handle.target');
    expect([...targets].map((h) => h.getAttribute('data-handleid'))).toEqual(['in0', 'in1']);
    expect(screen.getByTestId('operand-label-in1').textContent).toBe('in1');
  });

  it('keeps one unnamed target handle on an order-free merge', () => {
    const { container } = renderLayer(makeData({ layerType: 'Add', isMerge: true }));
    expect(container.querySelectorAll('.react-flow__handle.target').length).toBe(1);
    expect(screen.queryByTestId(/operand-label/)).toBeNull();
  });

  it('does not render params preview when there are no params', () => {
    renderLayer(makeData({ params: {} }));
    // No "more" hint and no param rows.
    expect(screen.queryByText(/more/)).toBeNull();
  });

  it('renders up to 3 params and stringifies their values', () => {
    renderLayer(
      makeData({
        params: { in_channels: 3, out_channels: 16, kernel_size: 3 },
      }),
    );
    expect(screen.getByText('in_channels')).toBeTruthy();
    expect(screen.getByText('out_channels')).toBeTruthy();
    expect(screen.getByText('kernel_size')).toBeTruthy();
    // values stringified (multiple "3" so use getAllByText)
    expect(screen.getAllByText('3').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('16')).toBeTruthy();
  });

  it('shows the "+N more" hint when more than 3 params exist', () => {
    renderLayer(
      makeData({
        params: { a: 1, b: 2, c: 3, d: 4, e: 5 },
      }),
    );
    // 5 params -> 2 more
    const expected = useI18n.getState().t('layersEditor.layerNode.moreParams', { count: 2 });
    expect(screen.getByText(expected)).toBeTruthy();
    // Only first 3 keys are rendered as rows.
    expect(screen.getByText('a')).toBeTruthy();
    expect(screen.getByText('b')).toBeTruthy();
    expect(screen.getByText('c')).toBeTruthy();
    expect(screen.queryByText('d')).toBeNull();
    expect(screen.queryByText('e')).toBeNull();
  });

  it('does NOT show the "+N more" hint at exactly 3 params', () => {
    renderLayer(makeData({ params: { a: 1, b: 2, c: 3 } }));
    expect(screen.queryByText(/more/)).toBeNull();
  });

  it('applies selected styling (white border + colored glow)', () => {
    const { container } = renderLayer(makeData({ color: colorForType('Linear') }), true);
    const root = container.querySelector('div') as HTMLElement;
    expect(root.style.border).toContain('rgb(255, 255, 255)');
    expect(root.style.boxShadow).toContain(colorForType('Linear') + '44');
  });

  it('applies unselected styling (color border + plain shadow)', () => {
    const { container } = renderLayer(makeData({ color: colorForType('Linear') }), false);
    const root = container.querySelector('div') as HTMLElement;
    // The hue at full strength, not the old 53%-alpha version (core#228).
    expect(root.style.border).toContain('rgb(0, 188, 212)');
    expect(root.style.boxShadow).toContain('rgba(0,0,0,0.4)');
  });
});
