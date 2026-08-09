// frontend/src/components/LayersEditor/InputNode.tsx
import { memo } from 'react';
import { Handle, Node, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import type { LayerNodeData } from './graphSerialization';

function InputNodeComponent({ data, selected }: NodeProps<Node<LayerNodeData>>) {
  const ports = data.ports ?? [];
  return (
    <div
      style={{
        background: '#1e1e1e',
        border: `2px solid ${selected ? '#fff' : '#4CAF5088'}`,
        borderRadius: 8,
        minWidth: 140,
        fontSize: '0.8125rem',
        color: '#eee',
        boxShadow: selected ? '0 0 12px #4CAF5044' : '0 3px 10px rgba(0,0,0,0.4)',
      }}
    >
      <div
        style={{
          background: '#4CAF50',
          padding: '5px 10px',
          borderRadius: '6px 6px 0 0',
          fontWeight: 600,
          fontSize: '0.8125rem',
          textAlign: 'center',
          color: '#fff',
        }}
      >
        Input
      </div>
      <div style={{ padding: '6px 10px 14px', display: 'flex', flexDirection: 'column', gap: 2 }}>
        {ports.map((p) => (
          <div key={p.id} style={{ fontSize: '0.6875rem', color: '#bbb', textAlign: 'center' }}>
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
              background: '#4CAF50',
              width: 10,
              height: 10,
              border: '2px solid #1e1e1e',
              left: `${left}%`,
              bottom: -5,
            }}
          />
        );
      })}
    </div>
  );
}

export const InputNode = memo(InputNodeComponent);
