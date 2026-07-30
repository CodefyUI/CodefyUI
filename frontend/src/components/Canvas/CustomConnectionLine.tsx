import { getSmoothStepPath, type ConnectionLineComponentProps } from '@xyflow/react';
import { useUIStore } from '../../store/uiStore';
import { CIRCUIT_BORDER_RADIUS } from './SmartDataEdge';

export function CustomConnectionLine({
  fromX,
  fromY,
  toX,
  toY,
  fromPosition,
  toPosition,
}: ConnectionLineComponentProps) {
  const circuit = useUIStore((s) => s.edgeStyle) === 'circuit';
  // Match the live edge look while dragging: orthogonal smoothstep in circuit
  // mode, the classic cubic pull in curve mode.
  const d = circuit
    ? getSmoothStepPath({
        sourceX: fromX,
        sourceY: fromY,
        sourcePosition: fromPosition,
        targetX: toX,
        targetY: toY,
        targetPosition: toPosition,
        borderRadius: CIRCUIT_BORDER_RADIUS,
      })[0]
    : `M${fromX},${fromY} C${fromX + 80},${fromY} ${toX - 80},${toY} ${toX},${toY}`;
  return (
    <g>
      <path fill="none" stroke="#888" strokeWidth={2} d={d} />
      <circle cx={toX} cy={toY} r={4} fill="#888" />
    </g>
  );
}
