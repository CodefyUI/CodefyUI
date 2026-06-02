import { describe, it, expect } from 'vitest';
import { Position, type EdgeProps } from '@xyflow/react';
import { renderWithFlow } from '../../test/utils';
import { TriggerEdge } from './TriggerEdge';

function makeProps(over: Partial<EdgeProps> = {}): EdgeProps {
  return {
    id: 'e1',
    source: 'a',
    target: 'b',
    sourceX: 0,
    sourceY: 0,
    targetX: 100,
    targetY: 100,
    sourcePosition: Position.Right,
    targetPosition: Position.Left,
    ...over,
  } as EdgeProps;
}

describe('TriggerEdge', () => {
  it('renders a dashed green BaseEdge path', () => {
    const { container } = renderWithFlow(
      <svg>
        <TriggerEdge {...makeProps()} />
      </svg>,
    );

    const path = container.querySelector('path.react-flow__edge-path');
    expect(path).toBeTruthy();
    // BaseEdge applies the provided style inline.
    const style = path?.getAttribute('style') ?? '';
    expect(style).toContain('stroke-dasharray: 6 4');
    expect(style).toContain('stroke-width: 2');
    // Path data is produced by getBezierPath and starts with a move command.
    expect(path?.getAttribute('d')?.startsWith('M')).toBe(true);
  });
});
