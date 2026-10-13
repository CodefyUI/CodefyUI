import { Fragment, memo } from 'react';
import { Handle, Node, Position, useNodeConnections } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import { useI18n } from '../../i18n';
import { mixColor, NODE_HEADER_TINT, SURFACE_RAISED } from '../../styles/theme';
import { mergeInputHandles } from './graphSerialization';
import type { LayerNodeData } from './graphSerialization';

const TARGET_HANDLE_STYLE = {
  background: 'var(--border-strong)',
  width: 'var(--handle-size)',
  height: 'var(--handle-size)',
  border: '2px solid var(--surface-raised)',
  top: 'calc(var(--handle-size) / -2)',
} as const;

function LayerNodeComponent({ id, data, selected }: NodeProps<Node<LayerNodeData>>) {
  const { t } = useI18n();
  // Order-sensitive merges (Subtract, Concat, Stack) get one labelled
  // handle per operand, so the card shows which branch comes first. The
  // backend takes the operands in handle order.
  const connections = useNodeConnections({ id, handleType: 'target' });
  const operandHandles = mergeInputHandles(
    data.layerType,
    connections.map((c) => c.targetHandle),
  );
  const paramEntries = Object.entries(data.params);
  const hasParams = paramEntries.length > 0;
  const color = data.color;
  // Same construction as a canvas node (core#228). The header used to be
  // painted with the raw layer hue and titled in #fff, which measured 2.16:1
  // to 3.09:1 on all seven hues -- under half the AA minimum, on every node in
  // the editor. Tinting the hue into the card and keeping it at full strength
  // as the header's accent rule keeps the colour coding and puts the title
  // above 4.5:1; `scripts/check-contrast.mjs` re-derives both numbers.
  const headerFill = mixColor(SURFACE_RAISED, color, NODE_HEADER_TINT);

  return (
    <div
      style={{
        background: 'var(--surface-raised)',
        border: `2px solid ${selected ? '#fff' : color}`,
        borderRadius: 8,
        minWidth: 140,
        position: 'relative',
        fontSize: '0.8125rem',
        color: 'var(--text-primary)',
        boxShadow: selected
          ? `0 0 12px ${color}44`
          : '0 3px 10px rgba(0,0,0,0.4)',
        transition: 'border-color 0.2s, box-shadow 0.2s',
      }}
    >
      {/* Header */}
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
              <span style={{ fontSize: '0.625rem', color: 'var(--text-muted)' }}>{key}</span>
              <span
                style={{
                  fontSize: '0.625rem',
                  color: 'var(--text-secondary)',
                  fontFamily: 'monospace',
                }}
              >
                {String(val)}
              </span>
            </div>
          ))}
          {paramEntries.length > 3 && (
            <div
              style={{
                fontSize: '0.5625rem',
                color: 'var(--text-muted)',
                textAlign: 'center',
              }}
            >
              {t('layersEditor.layerNode.moreParams', { count: paramEntries.length - 3 })}
            </div>
          )}
        </div>
      )}

      {/* Handles */}
      {operandHandles.length === 0 ? (
        <Handle type="target" position={Position.Top} style={TARGET_HANDLE_STYLE} />
      ) : (
        operandHandles.map((handleId, i) => {
          const left = `${((i + 1) / (operandHandles.length + 1)) * 100}%`;
          return (
            <Fragment key={handleId}>
              <Handle
                type="target"
                id={handleId}
                position={Position.Top}
                style={{ ...TARGET_HANDLE_STYLE, left }}
              />
              <span
                data-testid={`operand-label-${handleId}`}
                style={{
                  position: 'absolute',
                  left,
                  bottom: 'calc(100% + var(--handle-size) / 2 + 2px)',
                  transform: 'translateX(-50%)',
                  fontSize: '0.625rem',
                  fontFamily: 'monospace',
                  color: 'var(--text-secondary)',
                  pointerEvents: 'none',
                  whiteSpace: 'nowrap',
                }}
              >
                {handleId}
              </span>
            </Fragment>
          );
        })
      )}
      <Handle
        type="source"
        position={Position.Bottom}
        style={{
          background: 'var(--border-strong)',
          width: 'var(--handle-size)',
          height: 'var(--handle-size)',
          border: '2px solid var(--surface-raised)',
          bottom: 'calc(var(--handle-size) / -2)',
        }}
      />
    </div>
  );
}

export const LayerNode = memo(LayerNodeComponent);
