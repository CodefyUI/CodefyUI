import { useTabStore } from '../../store/tabStore';
import { resolveDynamicInputs, resolveDynamicOutputs } from '../../utils';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import { NodeParamList } from '../shared/NodeParamList';
import { MathText } from '../shared/MathText';
import { CATEGORY_COLORS } from '../../styles/theme';
import styles from './NodeConfigPanel.module.css';

export function NodeConfigPanel() {
  const activeTab = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)!);
  const nodes = activeTab.nodes;
  const selectedNodeId = activeTab.selectedNodeId;
  const openPresetModal = useTabStore((s) => s.openPresetModal);
  const isCanvasPanning = useUIStore((s) => s.isCanvasPanning);
  const { t, tn } = useI18n();

  const selectedNode = nodes.find((n) => n.id === selectedNodeId);
  const def = selectedNode?.data.definition;
  const nodeName = def?.node_name ?? '';
  const liveInputs = resolveDynamicInputs(def, selectedNode?.data.params);
  const liveOutputs = resolveDynamicOutputs(def, selectedNode?.data.params);

  if (!selectedNode) return null;

  // Note nodes have no config panel
  if (selectedNode.type === 'noteNode') return null;

  const isPreset = selectedNode.data.isPreset;
  const category = def?.category ?? 'Utility';
  const accentColor = isPreset ? '#D4A017' : (CATEGORY_COLORS[category] ?? '#607D8B');

  return (
    <div
      className={styles.panel}
      style={{ opacity: isCanvasPanning ? 0.4 : 1 }}
    >
      {/* Panel header */}
      <div
        className={styles.header}
        style={{ borderBottom: `2px solid ${accentColor}` }}
      >
        <div className={styles.headerMeta}>
          {t('config.title')}
          {isPreset && (
            <span className={styles.presetBadge}>{t('preset.badge')}</span>
          )}
        </div>
        <div className={styles.headerName}>{selectedNode.data.label}</div>
        <div className={styles.headerCategory} style={{ color: accentColor }}>
          {category}
        </div>
        {def?.description && (
          <MathText
            as="div"
            className={styles.headerDescription}
            text={tn(nodeName, 'description', def.description)}
          />
        )}
      </div>

      {/* Scrollable content */}
      <div className={styles.content}>
        {/* Preset: Configure button */}
        {isPreset && (
          <div style={{ marginBottom: 16 }}>
            <div className={styles.presetHint}>
              {t('preset.nodeCount', { count: selectedNode.data.presetDefinition?.nodes.length ?? 0 })}
            </div>
            <button type="button"
              onClick={() => openPresetModal(selectedNode.id)}
              className={styles.presetConfigureBtn}
            >
              {t('preset.configure')}
            </button>
          </div>
        )}

        {/* Params section (for non-preset nodes) */}
        {!isPreset && def && def.params.length > 0 ? (
          <div>
            <div className={styles.sectionHeader}>{t('config.parameters')}</div>
            <NodeParamList
              nodeId={selectedNodeId}
              definition={def}
              params={selectedNode.data.params}
            />
          </div>
        ) : !isPreset ? (
          <div className={styles.noParams}>{t('config.noParams')}</div>
        ) : null}

        {/* I/O info section — live ports, so a param-driven node (Split,
            PythonScript, ComposeTransform) lists what it actually has
            right now. */}
        {def && (liveInputs.length > 0 || liveOutputs.length > 0) && (
          <div style={{ marginTop: 20 }}>
            <div className={styles.sectionHeaderPorts}>{t('config.ports')}</div>

            {liveInputs.length > 0 && (
              <div style={{ marginBottom: 8 }}>
                <div className={styles.portSubLabel}>{t('config.inputs')}</div>
                {liveInputs.map((inp) => (
                  <div key={inp.name} className={styles.portRow}>
                    <span className={styles.portDot} />
                    <span className={styles.portName}>{inp.name}</span>
                    <span className={styles.portType}>{inp.data_type}</span>
                    {inp.optional && (
                      <span className={styles.portOptional}>{t('config.optional')}</span>
                    )}
                  </div>
                ))}
              </div>
            )}

            {liveOutputs.length > 0 && (
              <div>
                <div className={styles.portSubLabel}>{t('config.outputs')}</div>
                {liveOutputs.map((out) => (
                  <div key={out.name} className={styles.portRow}>
                    <span className={styles.portDot} />
                    <span className={styles.portName}>{out.name}</span>
                    <span className={styles.portType}>{out.data_type}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Execution status */}
        {selectedNode.data.executionStatus && selectedNode.data.executionStatus !== 'idle' && (
          <div style={{ marginTop: 20 }}>
            <div className={styles.sectionHeaderExecution}>{t('config.execution')}</div>
            <div
              style={{
                padding: '6px 10px',
                borderRadius: 4,
                background:
                  selectedNode.data.executionStatus === 'error'
                    ? 'rgba(244,67,54,0.1)'
                    : selectedNode.data.executionStatus === 'completed'
                      ? 'rgba(76,175,80,0.1)'
                      : selectedNode.data.executionStatus === 'cached'
                        ? 'rgba(33,150,243,0.1)'
                        : 'rgba(255,193,7,0.1)',
                border: `1px solid ${
                  selectedNode.data.executionStatus === 'error'
                    ? '#F44336'
                    : selectedNode.data.executionStatus === 'completed'
                      ? '#4CAF50'
                      : selectedNode.data.executionStatus === 'cached'
                        ? '#2196F3'
                        : '#FFC107'
                }`,
                fontSize: '0.75rem',
                color:
                  selectedNode.data.executionStatus === 'error'
                    ? '#F44336'
                    : selectedNode.data.executionStatus === 'completed'
                      ? '#4CAF50'
                      : selectedNode.data.executionStatus === 'cached'
                        ? '#2196F3'
                        : '#FFC107',
              }}
            >
              {selectedNode.data.executionStatus === 'error' && selectedNode.data.error
                ? t('node.error', { error: selectedNode.data.error })
                : t(`status.${selectedNode.data.executionStatus}` as const)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
