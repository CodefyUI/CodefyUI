import { describe, it, expect } from 'vitest';
import { Position } from '@xyflow/react';
import type { ConnectionLineComponentProps } from '@xyflow/react';
import { renderWithFlow } from '../../test/utils';
import { CustomConnectionLine } from './CustomConnectionLine';

// `ConnectionLineComponentProps` is large; CustomConnectionLine only reads
// fromX/fromY/toX/toY, so cast a minimal object to that shape.
function makeProps(over: Partial<Record<string, unknown>> = {}): ConnectionLineComponentProps {
  return {
    fromX: 10,
    fromY: 20,
    toX: 110,
    toY: 220,
    fromPosition: Position.Right,
    toPosition: Position.Left,
    connectionLineType: 'default',
    connectionStatus: null,
    fromHandle: null,
    fromNode: null,
    toNode: null,
    toHandle: null,
    ...over,
  } as unknown as ConnectionLineComponentProps;
}

describe('CustomConnectionLine', () => {
  it('renders a bezier path and end circle using the from/to coordinates', () => {
    const { container } = renderWithFlow(
      <svg>
        <CustomConnectionLine {...makeProps()} />
      </svg>,
    );

    const path = container.querySelector('path');
    expect(path).toBeTruthy();
    // Path is built from M{fromX},{fromY} C{fromX+80}... {toX-80}... {toX},{toY}
    expect(path?.getAttribute('d')).toBe('M10,20 C90,20 30,220 110,220');
    expect(path?.getAttribute('stroke')).toBe('#888');
    expect(path?.getAttribute('stroke-width')).toBe('2');
    expect(path?.getAttribute('fill')).toBe('none');

    const circle = container.querySelector('circle');
    expect(circle?.getAttribute('cx')).toBe('110');
    expect(circle?.getAttribute('cy')).toBe('220');
    expect(circle?.getAttribute('r')).toBe('4');
    expect(circle?.getAttribute('fill')).toBe('#888');
  });
});
