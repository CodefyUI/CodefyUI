import { memo, useState, type ReactNode } from 'react';
import { Handle, Position, useReactFlow } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import type { AppNode } from '../../types';
import {
  getPortColor,
  isParamVisible,
  isValidConnection,
  resolveDynamicInputs,
  resolveDynamicOutputs,
} from '../../utils';
import { findDetachableEdge, redirectMouseDownToReconnectAnchor } from '../../utils/reconnect';
import { useUIStore } from '../../store/uiStore';
import { useTabStore } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import { downloadModelFile } from '../../api/rest';
import { useI18n } from '../../i18n';
import { CATEGORY_COLORS, STATUS_COLORS } from '../../styles/theme';
import { MathText } from '../shared/MathText';
import { subgraphIdOf } from '../../utils/subgraph';
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

/** Lines of a CODE param shown on the card before the "+N more" line. */
const CODE_PREVIEW_LINES = 4;

/**
 * The opening lines of a CODE param, on the node card (core#131).
 *
 * Blank and comment-only leading lines are skipped so the preview starts at
 * something that says what the script does, which for the shipped template
 * is the `def run(...)` line.
 */
function CodePreview({ source }: { source: string }) {
  const { t } = useI18n();
  const lines = source.replace(/\t/g, '    ').split('\n');
  const start = lines.findIndex((line) => line.trim() !== '');
  const body = start < 0 ? [] : lines.slice(start);
  const shown = body.slice(0, CODE_PREVIEW_LINES);
  const hidden = Math.max(0, body.length - shown.length);

  if (shown.length === 0) {
    return <div className={styles.codePreviewEmpty}>{t('node.code.empty')}</div>;
  }
  return (
    <div className={styles.codePreview} title={source}>
      {shown.map((line, i) => (
        <div key={i} className={styles.codePreviewLine}>
          {line === '' ? ' ' : line}
        </div>
      ))}
      {hidden > 0 && (
        <div className={styles.codePreviewMore}>
          {t('node.code.moreLines', { count: hidden })}
        </div>
      )}
    </div>
  );
}

/**
 * Named export — the full node card with an injectable body slot. Use this
 * from custom xyflow node components that compose around BaseNode (e.g.
 * `<BaseNodeBody {...props} bodyExtra={<MyViz />} />`). The default export
 * below memoizes a slot-less version for the regular ``baseNode`` xyflow type.
 */
export function BaseNodeBody({ id, data, selected, bodyExtra }: BaseNodeProps) {
  const openSubgraphModal = useTabStore((s) => s.openSubgraphModal);
  const openNodeDetail = useTabStore((s) => s.openNodeDetail);
  const enterSubgraph = useTabStore((s) => s.enterSubgraph);
  const tooltipsEnabled = useUIStore((s) => s.tooltipsEnabled);
  const draggingSourceType = useUIStore((s) => s.draggingSourceType);
  const reconnectingHandle = useUIStore((s) => s.reconnectingHandle);
  const nodeOutputs = useTabStore((s) => {
    const tab = s.tabs.find((t) => t.id === s.activeTabId);
    return tab?.outputSummaries?.[id];
  });
  const [hovered, setHovered] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const { getEdges } = useReactFlow();
  const def = data.definition;
  // Live port sets: a param-driven node (Split's `chunks`, PythonScript's
  // `input_ports`, ComposeTransform's `steps`) has a different shape than
  // its palette template.
  const liveInputs = resolveDynamicInputs(def, data.params);
  const liveOutputs = resolveDynamicOutputs(def, data.params);
  const category = def?.category ?? 'Utility';
  const headerColor = CATEGORY_COLORS[category] ?? '#607D8B';
  const { t, tn } = useI18n();

  const isSequentialModel = data.type === 'SequentialModel';
  // core#137: a subgraph instance opens its definition on double-click, the
  // same way SequentialModel opens its layer editor.
  const isSubgraph = subgraphIdOf(data.type) !== null;
  const isDraggingTrigger = draggingSourceType === 'TRIGGER';
  // core#128: muted in place. The card stays fully interactive (you have to
  // be able to select it to un-mute it) — only its look says "skipped".
  const isBypassed = data.bypassed === true;

  // True when this exact handle is the originally-connected endpoint of an
  // in-progress edge-reconnect drag; it shows a red "detaching" ring warning
  // that dropping on empty space deletes the edge.
  const isDetaching = (handleId: string, type: 'source' | 'target') =>
    reconnectingHandle !== null &&
    reconnectingHandle.nodeId === id &&
    reconnectingHandle.type === type &&
    reconnectingHandle.handleId === handleId;

  // Detect if this node is a trigger target (connected from a Start node)
  const isTriggerTarget = getEdges().some(
    (e) => e.target === id && ((e.data as { type?: string } | undefined)?.type === 'trigger'),
  );

  // ComfyUI-style detach: a plain left-button press on a CONNECTED input
  // grabs the existing edge off the port instead of starting a second
  // connection. The Handle DOM sits above React Flow's invisible reconnect
  // anchor (edge SVG layer renders below nodes), so we intercept the
  // connection-starting mousedown in the CAPTURE phase — the Handle starts
  // connections from its bubble-phase onMouseDown (verified in @xyflow/react
  // 12.10.1, see utils/reconnect.ts) — and re-dispatch it onto the edge's
  // anchor, which drives the full native reconnect lifecycle
  // (onReconnectStart red ring, live drag following the real mouse,
  // drop-on-pane delete / drop-on-handle rewire). A plain click without
  // moving stays a no-op: React Flow only starts the reconnect after its
  // drag threshold.
  const interceptConnectedInputMouseDown = (
    event: React.MouseEvent,
    handleId: string,
  ) => {
    // Escape hatch: modified drags (or non-left buttons) keep the old
    // start-a-new-connection behavior.
    if (
      event.button !== 0 ||
      event.shiftKey ||
      event.ctrlKey ||
      event.altKey ||
      event.metaKey
    ) {
      return;
    }
    // Read edges at event time straight from the store (same imperative,
    // no-subscription pattern as the getEdges() trigger check above).
    const { tabs, activeTabId } = useTabStore.getState();
    const tab = tabs.find((t) => t.id === activeTabId);
    const edge = tab ? findDetachableEdge(tab.edges, id, handleId) : null;
    if (edge) redirectMouseDownToReconnectAnchor(event, edge.id);
  };

  const handleClick = (e: React.MouseEvent) => {
    if (e.detail === 2 && isSequentialModel) {
      openSubgraphModal(id);
    }
  };

  // Double-click opens the Node Detail Modal (#127) — except on
  // SequentialModel, whose double-click already belongs to the subgraph
  // editor (handled above by `handleClick`; opening two modals at once would
  // be worse than either).
  //
  // `stopPropagation` matters regardless of which modal wins: the canvas
  // listens for `dblclick` on `.react-flow__pane`, which CONTAINS the nodes,
  // and answers with QuickNodeSearch. FlowCanvas also filters on
  // `closest('.react-flow__node')`, so this is the second of two independent
  // guards — deliberately, because either one silently regressing would put
  // an "add node" palette on top of the modal.
  const handleDoubleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (isSequentialModel) return;
    if (isSubgraph) {
      enterSubgraph(id);
      return;
    }
    openNodeDetail(id);
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
            : // core#122: stopped with partial results. Without this branch an
              // interrupted node falls through to 'transparent' and renders
              // exactly like one that never ran.
              data.executionStatus === 'interrupted'
              ? STATUS_COLORS.interrupted
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
      onDoubleClick={handleDoubleClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className={`${styles.node}${isTriggerTarget ? ` ${styles.entryPoint}` : ''}${isDraggingTrigger ? ` ${styles.triggerDropTarget}` : ''}${isBypassed ? ` ${styles.bypassed}` : ''}`}
      data-bypassed={isBypassed || undefined}
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
        className={`${styles.triggerHandle}${isDraggingTrigger ? ` ${styles.triggerHandleActive}` : ''}${
          isDetaching('__trigger', 'target') ? ` ${styles.triggerHandleDetaching}` : ''
        }`}
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
        {isBypassed && (
          <span className={styles.bypassBadge} title={t('node.bypassed.title')}>
            {t('node.bypassed')}
          </span>
        )}
        <span className={styles.headerCategory}>
          {category}
        </span>
      </div>

      {/* Ports area */}
      <div className={styles.portsArea}>
        {/* Input handles — param-driven nodes (PythonScript,
            ComposeTransform) expand here */}
        {liveInputs.map((input) => (
          <div
            key={`in-${input.name}`}
            className={styles.portRowInput}
            title={input.description}
          >
            <Handle
              type="target"
              position={Position.Left}
              id={input.name}
              onMouseDownCapture={(e) => interceptConnectedInputMouseDown(e, input.name)}
              className={`${styles.portHandle} ${styles.portHandleInput}${
                isDetaching(input.name, 'target')
                  ? ` ${styles.portDetaching}`
                  : draggingSourceType
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
        {def && liveInputs.length > 0 && liveOutputs.length > 0 && (
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
              className={`${styles.portHandle} ${styles.portHandleOutput}${
                isDetaching(output.name, 'source') ? ` ${styles.portDetaching}` : ''
              }`}
              style={{ background: getPortColor(output.data_type) }}
            />
          </div>
        ))}
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
            .filter((p) => {
              if (!isParamVisible(p, data.params)) return false;
              if (!p.advanced) return true;
              // An advanced param earns a row on the CARD only once someone
              // has moved it off its default (core#134). The card is the
              // at-a-glance summary read across a projector: six untouched
              // PyTorch defaults would say less than the three values that
              // matter. A tuned one still shows, because that is exactly the
              // thing a reader of the graph needs to know.
              return String(data.params[p.name] ?? p.default) !== String(p.default);
            })
            .map((p) => {
              const val = data.params[p.name] ?? p.default;
              // CODE params are the node's whole point, not a value beside a
              // name: show the opening lines as code (core#131). One line of
              // `def run(inputs, params):` squeezed into the 120px value
              // column would say nothing about what the script does.
              if (p.param_type === 'code') {
                return (
                  <CodePreview key={p.name} source={String(val ?? '')} />
                );
              }
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
