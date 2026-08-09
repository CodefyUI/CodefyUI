// frontend/src/components/LayersEditor/OutputNode.tsx
import { memo } from 'react';
import { Handle, Node, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import type { LayerNodeData } from './graphSerialization';

function OutputNodeComponent({ data, selected }: NodeProps<Node<LayerNodeData>>) {
  const ports = data.ports ?? [];
  return (
    <div
      style={{
        background: '#1e1e1e',
        border: `2px solid ${selected ? '#fff' : '#F4433688'}`,
        borderRadius: 8,
        minWidth: 140,
        fontSize: '0.8125rem',
        color: '#eee',
        boxShadow: selected ? '0 0 12px #F4433644' : '0 3px 10px rgba(0,0,0,0.4)',
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
              background: '#F44336',
              width: 10,
              height: 10,
              border: '2px solid #1e1e1e',
              left: `${left}%`,
              top: -5,
            }}
          />
        );
      })}
      <div style={{ padding: '14px 10px 6px', display: 'flex', flexDirection: 'column', gap: 2 }}>
        {ports.map((p) => (
          <div key={p.id} style={{ fontSize: '0.6875rem', color: '#bbb', textAlign: 'center' }}>
            {p.name}
          </div>
        ))}
      </div>
      <div
        style={{
          background: '#F44336',
          padding: '5px 10px',
          borderRadius: '0 0 6px 6px',
          fontWeight: 600,
          fontSize: '0.8125rem',
          textAlign: 'center',
          color: '#fff',
        }}
      >
        Output
      </div>
    </div>
  );
}

export const OutputNode = memo(OutputNodeComponent);
