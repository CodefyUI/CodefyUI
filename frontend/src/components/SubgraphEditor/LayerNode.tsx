import { memo } from 'react';
import { Handle, Node, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import { useI18n } from '../../i18n';
import type { LayerNodeData } from './graphSerialization';

function LayerNodeComponent({ data, selected }: NodeProps<Node<LayerNodeData>>) {
  const { t } = useI18n();
  const paramEntries = Object.entries(data.params);
  const hasParams = paramEntries.length > 0;

  return (
    <div
      style={{
        background: '#1e1e1e',
        border: `2px solid ${selected ? '#fff' : data.color + '88'}`,
        borderRadius: 8,
        minWidth: 140,
        fontSize: '0.8125rem',
        color: '#eee',
        boxShadow: selected
          ? `0 0 12px ${data.color}44`
          : '0 3px 10px rgba(0,0,0,0.4)',
        transition: 'border-color 0.2s, box-shadow 0.2s',
      }}
    >
      {/* Header */}
      <div
        style={{
          background: data.color,
          padding: '5px 10px',
          borderRadius: '6px 6px 0 0',
          fontWeight: 600,
          fontSize: '0.8125rem',
          textAlign: 'center',
          color: '#fff',
          textShadow: '0 1px 2px rgba(0,0,0,0.3)',
        }}
      >
        {data.layerType}
      </div>

      {/* Params preview */}
      {hasParams && (
        <div style={{ padding: '4px 8px' }}>
          {paramEntries.slice(0, 3).map(([key, val]) => (
            <div
              key={key}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                gap: 6,
                padding: '1px 0',
              }}
            >
              <span style={{ fontSize: '0.625rem', color: '#777' }}>{key}</span>
              <span
                style={{
                  fontSize: '0.625rem',
                  color: '#bbb',
                  fontFamily: 'monospace',
                }}
              >
                {String(val)}
              </span>
            </div>
          ))}
          {paramEntries.length > 3 && (
            <div style={{ fontSize: '0.5625rem', color: '#555', textAlign: 'center' }}>
              {t('subgraph.layerNode.moreParams', { count: paramEntries.length - 3 })}
            </div>
          )}
        </div>
      )}

      {/* Handles */}
      <Handle
        type="target"
        position={Position.Top}
        style={{
          background: '#666',
          width: 8,
          height: 8,
          border: '2px solid #1e1e1e',
          top: -4,
        }}
      />
      <Handle
        type="source"
        position={Position.Bottom}
        style={{
          background: '#666',
          width: 8,
          height: 8,
          border: '2px solid #1e1e1e',
          bottom: -4,
        }}
      />
    </div>
  );
}

export const LayerNode = memo(LayerNodeComponent);
