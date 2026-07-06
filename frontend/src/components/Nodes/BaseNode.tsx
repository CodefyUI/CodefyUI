import { memo, useState, type ReactNode } from 'react';
import { Handle, Position, useReactFlow } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import type { AppNode } from '../../types';
import { getPortColor, isParamVisible, isValidConnection, resolveDynamicOutputs } from '../../utils';
import { useUIStore } from '../../store/uiStore';
import { useTabStore } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import { downloadModelFile } from '../../api/rest';
import { useI18n } from '../../i18n';
import { CATEGORY_COLORS, STATUS_COLORS } from '../../styles/theme';
import { MathText } from '../shared/MathText';
import styles from './BaseNode.module.css';

type BaseNodeProps = NodeProps<AppNode> & {
  /**
   * Custom UI rendered inside the node card, between the ports area and the
   * params summary. Used by viz nodes (e.g. TokenizerVizNode) to inject
   * inline visualizations like animated token chips. The default `BaseNode`
   * passes nothing.
   */
  bodyExtra?: ReactNode;
};

/**
 * Named export — the full node card with an injectable body slot. Use this
 * from custom xyflow node components that compose around BaseNode (e.g.
 * `<BaseNodeBody {...props} bodyExtra={<MyViz />} />`). The default export
 * below memoizes a slot-less version for the regular ``baseNode`` xyflow type.
 */
export function BaseNodeBody({ id, data, selected, bodyExtra }: BaseNodeProps) {
  const openSubgraphModal = useTabStore((s) => s.openSubgraphModal);
  const tooltipsEnabled = useUIStore((s) => s.tooltipsEnabled);
  const draggingSourceType = useUIStore((s) => s.draggingSourceType);
  const nodeOutputs = useTabStore((s) => {
    const tab = s.tabs.find((t) => t.id === s.activeTabId);
    return tab?.outputSummaries?.[id];
  });
  const [hovered, setHovered] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const { getEdges } = useReactFlow();
  const def = data.definition;
  const category = def?.category ?? 'Utility';
  const headerColor = CATEGORY_COLORS[category] ?? '#607D8B';
  const { t, tn } = useI18n();

  const isSequentialModel = data.type === 'SequentialModel';
  const isDraggingTrigger = draggingSourceType === 'TRIGGER';

  // Detect if this node is a trigger target (connected from a Start node)
  const isTriggerTarget = getEdges().some(
    (e) => e.target === id && ((e.data as { type?: string } | undefined)?.type === 'trigger'),
  );

  const handleClick = (e: React.MouseEvent) => {
    if (e.detail === 2 && isSequentialModel) {
      openSubgraphModal(id);
    }
  };

  // Dynamic border: selected > execution status > default
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
      : '#444444';

  const description = def ? tn(def.node_name, 'description', def.description) : '';

  // Any output that points to a downloadable file (set by the backend for
  // string values under MODELS_DIR). Currently ModelSaver/CheckpointSaver.
  const downloadablePath = nodeOutputs
    ? Object.values(nodeOutputs).find((s) => s?.download_path)?.download_path
    : undefined;

  const handleDownload = async (e: React.MouseEvent) => {
    e.stopPropagation();
    // The download button only renders when downloadablePath is truthy
    /* v8 ignore start */
    if (!downloadablePath) return;
    /* v8 ignore stop */
    setDownloading(true);
    try {
      await downloadModelFile(downloadablePath);
    } catch (err: any) {
      useToastStore.getState().addToast(err.message ?? t('download.failed'), 'error');
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div
      onClick={handleClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className={`${styles.node}${isTriggerTarget ? ` ${styles.entryPoint}` : ''}${isDraggingTrigger ? ` ${styles.triggerDropTarget}` : ''}`}
      style={{
        '--border-color': borderColor,
        boxShadow: selected
          ? '0 0 16px rgba(255,255,255,0.15)'
          : '0 4px 12px rgba(0,0,0,0.4)',
        cursor: isSequentialModel ? 'pointer' : undefined,
      } as React.CSSProperties}
    >
      {/* Trigger target handle — hidden until user drags from Start */}
      <Handle
        type="target"
        position={Position.Left}
        id="__trigger"
        className={`${styles.triggerHandle}${isDraggingTrigger ? ` ${styles.triggerHandleActive}` : ''}`}
      />

      {/* Tooltip */}
      {tooltipsEnabled && hovered && description && (
        <div className={styles.tooltip}>
          <div className={styles.tooltipTitle}>{data.label}</div>
          <MathText as="div" className={styles.tooltipDesc} text={description} />
        </div>
      )}

      {/* Header */}
      <div
        className={styles.header}
        style={{ background: headerColor }}
      >
        <span className={styles.headerLabel}>
          {data.label}
        </span>
        <span className={styles.headerCategory}>
          {category}
        </span>
      </div>

      {/* Ports area */}
      <div className={styles.portsArea}>
        {/* Input handles */}
        {def?.inputs.map((input) => (
          <div
            key={`in-${input.name}`}
            className={styles.portRowInput}
            title={input.description}
          >
            <Handle
              type="target"
              position={Position.Left}
              id={input.name}
              className={`${styles.portHandle} ${styles.portHandleInput}${
                draggingSourceType
                  ? isDraggingTrigger
                    ? ` ${styles.portIncompatible}`
                    : isValidConnection(draggingSourceType, input.data_type)
                      ? ` ${styles.portCompatible}`
                      : ` ${styles.portIncompatible}`
                  : ''
              }`}
              style={{ background: getPortColor(input.data_type) }}
            />
            <span
              className={styles.portLabel}
              style={{ color: getPortColor(input.data_type) }}
            >
              {input.name}
            </span>
            {input.optional && (
              <span className={styles.portOptional}>{t('node.opt')}</span>
            )}
          </div>
        ))}

        {/* Output handles — param-driven nodes (e.g. Split) expand here */}
        {(() => {
          const liveOutputs = resolveDynamicOutputs(def, data.params);
          return (
            <>
              {def && def.inputs.length > 0 && liveOutputs.length > 0 && (
                <div className={styles.divider} />
              )}
              {liveOutputs.map((output) => (
                <div
                  key={`out-${output.name}`}
                  className={styles.portRowOutput}
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
                    className={`${styles.portHandle} ${styles.portHandleOutput}`}
                    style={{ background: getPortColor(output.data_type) }}
                  />
                </div>
              ))}
            </>
          );
        })()}
      </div>

      {/* Custom body slot — viz nodes inject animated chips, scatter plots, etc. */}
      {bodyExtra}

      {/* Params display — special view for SequentialModel */}
      {isSequentialModel && (
        <div className={styles.subgraphSection}>
          {(() => {
            let count = 0;
            try { count = JSON.parse(data.params.layers ?? '[]').length; } catch { /* ignore */ }
            return (
              <>
                <div className={styles.subgraphLayerRow}>
                  <span className={styles.subgraphLayerCount}>{count}</span>
                  <span>{t('subgraph.layerCount', { count: '' }).replace(/\s*$/, '')}</span>
                </div>
                <div className={styles.subgraphHint}>
                  {t('subgraph.hint')}
                </div>
              </>
            );
          })()}
        </div>
      )}

      {/* Params display — normal nodes */}
      {!isSequentialModel && def && def.params.length > 0 && (
        <div className={styles.paramsSection}>
          {def.params
            .filter((p) => isParamVisible(p, data.params))
            .map((p) => {
              const val = data.params[p.name] ?? p.default;
              // SECRET params never render their value on the node card —
              // the canvas is visible during screen shares; only the
              // masked ParamField may hold the session value.
              const display =
                p.param_type === 'secret' && String(val ?? '') !== ''
                  ? '••••••••'
                  : String(val);
              return (
                <div key={p.name} className={styles.paramRow}>
                  <span className={styles.paramName}>{p.name}</span>
                  <span
                    className={styles.paramValue}
                    title={display}
                  >
                    {display}
                  </span>
                </div>
              );
            })}
        </div>
      )}

      {/* Status footer — error */}
      {data.executionStatus === 'error' && data.error && (
        <div
          className={`${styles.statusFooter} ${styles.statusError}`}
          title={data.error}
        >
          {t('node.error', { error: data.error })}
        </div>
      )}

      {/* Status footer — running (with optional progress) */}
      {data.executionStatus === 'running' && (
        <div className={`${styles.statusFooter} ${styles.statusRunning}`}>
          {data.progress?.event === 'epoch' ? (
            <div className={styles.progressContainer}>
              <div className={styles.progressInfo}>
                <span>Epoch {data.progress.epoch}/{data.progress.total_epochs}</span>
                <span>Loss: {Number(data.progress.loss).toFixed(4)}</span>
              </div>
              <div className={styles.progressBarTrack}>
                <div
                  className={styles.progressBarFill}
                  style={{ width: `${((data.progress.epoch ?? 0) / (data.progress.total_epochs ?? 1)) * 100}%` }}
                />
              </div>
            </div>
          ) : (
            <>
              <span className={styles.statusRunningDot} />
              {t('node.running')}
            </>
          )}
        </div>
      )}

      {/* Status footer — completed */}
      {data.executionStatus === 'completed' && (
        <div className={`${styles.statusFooter} ${styles.statusCompleted}`}>
          <div>{t('node.completed')}</div>
          {downloadablePath && (
            <button
              type="button"
              className={styles.downloadBtn}
              onClick={handleDownload}
              disabled={downloading}
              title={downloadablePath}
            >
              {downloading ? '... ' : '↓ '}
              {downloadablePath.split('/').pop()}
            </button>
          )}
        </div>
      )}

      {/* Status footer — cached */}
      {data.executionStatus === 'cached' && (
        <div className={`${styles.statusFooter} ${styles.statusCached}`}>
          <span className={styles.statusCachedDot} />
          {t('node.cached')}
        </div>
      )}
    </div>
  );
}

function BaseNode(props: NodeProps<AppNode>) {
  return <BaseNodeBody {...props} />;
}

export default memo(BaseNode);
