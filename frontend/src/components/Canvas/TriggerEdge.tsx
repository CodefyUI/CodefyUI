import { BaseEdge, type EdgeProps } from '@xyflow/react';
import { useUIStore } from '../../store/uiStore';
import { useEdgeLane } from './EdgeLaneContext';
import { resolveEdgePath } from './SmartDataEdge';

/**
 * Control-flow edge from a Start node's trigger diamond to a node's `__trigger`
 * handle. Green and dashed so it reads as wiring rather than data, but routed by
 * the same router as every other line on the canvas.
 *
 * It used to draw a plain cubic with no awareness of its siblings, which made a
 * trigger fan-out the one place the no-superposition rule did not hold: on the
 * ResNet-18 / CIFAR-10 example four trigger edges leave one Start node and the two
 * heading down-right ran together for 96px. Nothing about this edge needs routing
 * of its own - the trigger diamond is a Right handle and `__trigger` is a Left
 * one, exactly like a data port, and trigger edges already sit in the same edge
 * array the lane map is built from - so it shares `resolveEdgePath` and inherits
 * the same guarantee. Only the stroke stays its own.
 */
export function TriggerEdge(props: EdgeProps) {
  const circuit = useUIStore((s) => s.edgeStyle) === 'circuit';
  const lane = useEdgeLane(props.id);

  const path = resolveEdgePath({
    sourceX: props.sourceX,
    sourceY: props.sourceY,
    sourcePosition: props.sourcePosition,
    targetX: props.targetX,
    targetY: props.targetY,
    targetPosition: props.targetPosition,
    circuit,
    lane,
  });

  return (
    <BaseEdge
      id={props.id}
      path={path}
      style={{
        stroke: 'var(--flow-trigger)',
        strokeDasharray: '6 4',
        strokeWidth: 2,
      }}
    />
  );
}
