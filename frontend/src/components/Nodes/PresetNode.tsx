import { memo } from 'react';
import { Handle, Position, useReactFlow } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import type { AppNode } from '../../types';
import { getPortColor } from '../../utils';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import { STATUS_COLORS } from '../../styles/theme';
import styles from './PresetNode.module.css';
import baseStyles from './BaseNode.module.css';

function PresetNode({ id, data, selected }: NodeProps<AppNode>) {
  const openPresetModal = useTabStore((s) => s.openPresetModal);
  const draggingSourceType = useUIStore((s) => s.draggingSourceType);
  const reconnectingHandle = useUIStore((s) => s.reconnectingHandle);
  const { getEdges } = useReactFlow();
  const def = data.definition;
  const preset = data.presetDefinition;
  const { t } = useI18n();
  const isDraggingTrigger = draggingSourceType === 'TRIGGER';
  const isTriggerTarget = getEdges().some(
    (e) => e.target === id && ((e.data as { type?: string } | undefined)?.type === 'trigger'),
  );

  // True when this exact handle is the originally-connected endpoint of an
  // in-progress edge-reconnect drag; it shows the red "detaching" ring. The
  // shared baseStyles.portDetaching class uses !important declarations, which
  // beat this component's inline handle styles per the CSS cascade.
  const isDetaching = (handleId: string, type: 'source' | 'target') =>
    reconnectingHandle !== null &&
    reconnectingHandle.nodeId === id &&
    reconnectingHandle.type === type &&
    reconnectingHandle.handleId === handleId;

  const statusBorderColor =
    data.executionStatus === 'running'
      ? STATUS_COLORS.running
      : data.executionStatus === 'completed'
        ? STATUS_COLORS.completed
        : data.executionStatus === 'error'
          ? STATUS_COLORS.error
          : data.executionStatus === 'cached'
            ? STATUS_COLORS.cached
            : 'transparent';

  const borderColor = selected
    ? '#ffffff'
    : statusBorderColor !== 'transparent'
      ? statusBorderColor
      : '#6B5B00';

  const handleClick = (e: React.MouseEvent) => {
    if (e.detail === 2) {
      openPresetModal(id);
    }
  };

  return (
    <div
      onClick={handleClick}
      className={`${styles.node}${isTriggerTarget ? ` ${baseStyles.entryPoint}` : ''}${isDraggingTrigger ? ` ${baseStyles.triggerDropTarget}` : ''}`}
      style={{
        border: `2px solid ${borderColor}`,
        boxShadow: selected
          ? '0 0 16px rgba(212,160,23,0.3)'
          : '0 4px 12px rgba(0,0,0,0.4)',
      }}
    >
      {/* Trigger target handle — hidden until user drags from Start */}
      <Handle
        type="target"
        position={Position.Left}
        id="__trigger"
        className={`${baseStyles.triggerHandle}${isDraggingTrigger ? ` ${baseStyles.triggerHandleActive}` : ''}${
          isDetaching('__trigger', 'target') ? ` ${baseStyles.triggerHandleDetaching}` : ''
        }`}
      />

      {/* Header with gold gradient */}
      <div className={styles.header}>
        <span className={styles.headerLabel}>{data.label}</span>
        <span className={styles.headerBadge}>{t('preset.badge')}</span>
      </div>

      {/* Ports area */}
      <div className={styles.portsArea}>
        {/* Input handles */}
        {def?.inputs.map((input) => (
          <div
            key={`in-${input.name}`}
            className={styles.inputRow}
            title={input.description}
          >
            <Handle
              type="target"
              position={Position.Left}
              id={input.name}
              className={isDetaching(input.name, 'target') ? baseStyles.portDetaching : undefined}
              style={{
                background: getPortColor(input.data_type),
                width: 10,
                height: 10,
                border: '2px solid #1e1e1e',
                left: -5,
                top: '50%',
                transform: 'translateY(-50%)',
                borderRadius: '50%',
                cursor: 'crosshair',
              }}
            />
            <span
              className={styles.portLabel}
              style={{ color: getPortColor(input.data_type) }}
            >
              {input.name}
            </span>
          </div>
        ))}

        {/* Divider */}
        {def && def.inputs.length > 0 && def.outputs.length > 0 && (
          <div className={styles.portDivider} />
        )}

        {/* Output handles */}
        {def?.outputs.map((output) => (
          <div
            key={`out-${output.name}`}
            className={styles.outputRow}
            title={output.description}
          >
            <span
              className={styles.portLabel}
              style={{ color: getPortColor(output.data_type) }}
            >
              {output.name}
            </span>
            <Handle
              type="source"
              position={Position.Right}
              id={output.name}
              className={isDetaching(output.name, 'source') ? baseStyles.portDetaching : undefined}
              style={{
                background: getPortColor(output.data_type),
                width: 10,
                height: 10,
                border: '2px solid #1e1e1e',
                right: -5,
                top: '50%',
                transform: 'translateY(-50%)',
                borderRadius: '50%',
                cursor: 'crosshair',
              }}
            />
          </div>
        ))}
      </div>

      {/* Footer: node count hint */}
      <div className={styles.footer}>
        <span className={styles.footerCount}>{preset?.nodes.length ?? 0}</span>
        <span>{t('preset.nodesInside')}</span>
      </div>

      {/* Status footer — error */}
      {data.executionStatus === 'error' && data.error && (
        <div
          className={`${styles.statusBase} ${styles.statusError}`}
          title={data.error}
        >
          {t('node.error', { error: data.error })}
        </div>
      )}

      {/* Status footer — running */}
      {data.executionStatus === 'running' && (
        <div className={`${styles.statusBase} ${styles.statusRunning}`}>
          <span className={styles.statusRunningDot} />
          {t('node.running')}
        </div>
      )}

      {/* Status footer — completed */}
      {data.executionStatus === 'completed' && (
        <div className={`${styles.statusBase} ${styles.statusCompleted}`}>
          {t('node.completed')}
        </div>
      )}

      {/* Status footer — cached */}
      {data.executionStatus === 'cached' && (
        <div className={`${styles.statusBase} ${styles.statusCached}`}>
          {t('node.cached')}
        </div>
      )}
    </div>
  );
}

export default memo(PresetNode);
