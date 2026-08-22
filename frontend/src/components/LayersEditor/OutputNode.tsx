// frontend/src/components/LayersEditor/OutputNode.tsx
import { memo } from 'react';
import { Handle, Node, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import { mixColor, NODE_HEADER_TINT, SURFACE_RAISED } from '../../styles/theme';
import type { LayerNodeData } from './graphSerialization';

function OutputNodeComponent({ data, selected }: NodeProps<Node<LayerNodeData>>) {
  const ports = data.ports ?? [];
  // Was a hardcoded '#F44336' here and again in `emptyGraph()` (core#228).
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
      {ports.map((p, i) => {
        const left = ((i + 1) / (ports.length + 1)) * 100;
        return (
          <Handle
            key={p.id}
            id={p.id}
            type="target"
            position={Position.Top}
            style={{
              background: color,
              width: 'var(--handle-size)',
              height: 'var(--handle-size)',
              border: '2px solid var(--surface-raised)',
              left: `${left}%`,
              top: 'calc(var(--handle-size) / -2)',
            }}
          />
        );
      })}
      <div style={{ padding: '14px 10px 6px', display: 'flex', flexDirection: 'column', gap: 2 }}>
        {ports.map((p) => (
          <div
            key={p.id}
            style={{ fontSize: '0.6875rem', color: 'var(--text-secondary)', textAlign: 'center' }}
          >
            {p.name}
          </div>
        ))}
      </div>
      <div
        style={{
          background: headerFill,
          borderTop: `2px solid ${color}`,
          padding: '5px 10px',
          borderRadius: '0 0 6px 6px',
          fontWeight: 600,
          fontSize: '0.8125rem',
          textAlign: 'center',
          color: 'var(--text-primary)',
        }}
      >
        Output
      </div>
    </div>
  );
}

export const OutputNode = memo(OutputNodeComponent);
