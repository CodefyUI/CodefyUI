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
  connectionStatus,
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
  // Over a port, React Flow reports whether the wire may land there. A port
  // it cannot feed turns the line red and dashed, so a refusal is visible
  // before the release (#685); a port it can feed turns it the active wire
  // colour. Away from any port it stays grey.
  const stroke =
    connectionStatus === 'invalid'
      ? 'var(--status-error)'
      : connectionStatus === 'valid'
        ? 'var(--wire-active)'
        : '#888';
  return (
    <g data-status={connectionStatus ?? 'none'}>
      <path
        fill="none"
        stroke={stroke}
        strokeWidth={2}
        strokeDasharray={connectionStatus === 'invalid' ? '6 4' : undefined}
        d={d}
      />
      <circle cx={toX} cy={toY} r={4} fill={stroke} />
    </g>
  );
}
