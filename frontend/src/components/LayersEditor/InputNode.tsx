// frontend/src/components/LayersEditor/InputNode.tsx
import { memo } from 'react';
import { Handle, Node, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import { mixColor, NODE_HEADER_TINT, SURFACE_RAISED } from '../../styles/theme';
import type { LayerNodeData } from './graphSerialization';

function InputNodeComponent({ data, selected }: NodeProps<Node<LayerNodeData>>) {
  const ports = data.ports ?? [];
  // Was a hardcoded '#4CAF50' here and again in `emptyGraph()` (core#228).
  // The colour now arrives on the node, stamped once by `colorForType()`.
  const color = data.color;
  const headerFill = mixColor(SURFACE_RAISED, color, NODE_HEADER_TINT);
  return (
    <div
      style={{
        background: 'var(--surface-raised)',
        border: `2px solid ${selected ? '#fff' : color}`,
        borderRadius: 8,
        minWidth: 140,
        fontSize: '0.8125rem',
        color: 'var(--text-primary)',
        boxShadow: selected ? `0 0 12px ${color}44` : '0 3px 10px rgba(0,0,0,0.4)',
      }}
    >
      <div
        style={{
          background: headerFill,
          borderBottom: `2px solid ${color}`,
          padding: '5px 10px',
          borderRadius: '6px 6px 0 0',
          fontWeight: 600,
          fontSize: '0.8125rem',
          textAlign: 'center',
          color: 'var(--text-primary)',
        }}
      >
        Input
      </div>
      <div style={{ padding: '6px 10px 14px', display: 'flex', flexDirection: 'column', gap: 2 }}>
        {ports.map((p) => (
          <div
            key={p.id}
            style={{ fontSize: '0.6875rem', color: 'var(--text-secondary)', textAlign: 'center' }}
          >
            {p.name}
          </div>
        ))}
      </div>
      {ports.map((p, i) => {
        const left = ((i + 1) / (ports.length + 1)) * 100;
        return (
          <Handle
            key={p.id}
            id={p.id}
            type="source"
            position={Position.Bottom}
            style={{
              background: color,
              width: 'var(--handle-size)',
              height: 'var(--handle-size)',
              border: '2px solid var(--surface-raised)',
              left: `${left}%`,
              bottom: 'calc(var(--handle-size) / -2)',
            }}
          />
        );
      })}
    </div>
  );
}

export const InputNode = memo(InputNodeComponent);
