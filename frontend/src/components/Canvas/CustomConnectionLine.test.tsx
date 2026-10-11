import { describe, it, expect } from 'vitest';
import { Position } from '@xyflow/react';
import type { ConnectionLineComponentProps } from '@xyflow/react';
import { renderWithFlow } from '../../test/utils';
import { useUIStore } from '../../store/uiStore';
import { CustomConnectionLine } from './CustomConnectionLine';

// `ConnectionLineComponentProps` is large; CustomConnectionLine only reads
// the coordinates and positions, so cast a minimal object to that shape.
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
  it('renders a bezier path and end circle using the from/to coordinates (curve mode)', () => {
    useUIStore.setState({ edgeStyle: 'curve' });
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

  it('renders an orthogonal smoothstep preview in circuit mode, keeping the cursor dot', () => {
    useUIStore.setState({ edgeStyle: 'circuit' });
    const { container } = renderWithFlow(
      <svg>
        <CustomConnectionLine {...makeProps()} />
      </svg>,
    );

    const path = container.querySelector('path');
    const d = path?.getAttribute('d') ?? '';
    expect(d.startsWith('M')).toBe(true);
    // smoothstep emits line/quadratic segments only -- no cubics.
    expect(d).not.toContain('C');
    expect(d).toContain('L');
    expect(path?.getAttribute('stroke')).toBe('#888');

    const circle = container.querySelector('circle');
    expect(circle?.getAttribute('cx')).toBe('110');
    expect(circle?.getAttribute('cy')).toBe('220');
    expect(circle?.getAttribute('r')).toBe('4');
  });

  // Over a port, React Flow says whether the wire may land there (#685).
  function lineFor(connectionStatus: 'valid' | 'invalid' | null) {
    useUIStore.setState({ edgeStyle: 'curve' });
    const { container } = renderWithFlow(
      <svg>
        <CustomConnectionLine {...makeProps({ connectionStatus })} />
      </svg>,
    );
    return {
      path: container.querySelector('path')!,
      circle: container.querySelector('circle')!,
    };
  }

  it('turns red and dashed over a port it cannot feed (#685)', () => {
    const { path, circle } = lineFor('invalid');
    expect(path.getAttribute('stroke')).toBe('var(--status-error)');
    expect(path.getAttribute('stroke-dasharray')).toBe('6 4');
    expect(circle.getAttribute('fill')).toBe('var(--status-error)');
  });

  it('takes the active wire colour, solid, over a port it can feed (#685)', () => {
    const { path, circle } = lineFor('valid');
    expect(path.getAttribute('stroke')).toBe('var(--wire-active)');
    expect(path.getAttribute('stroke-dasharray')).toBeNull();
    expect(circle.getAttribute('fill')).toBe('var(--wire-active)');
  });

  it('stays grey and solid away from any port', () => {
    const { path } = lineFor(null);
    expect(path.getAttribute('stroke')).toBe('#888');
    expect(path.getAttribute('stroke-dasharray')).toBeNull();
  });
});
