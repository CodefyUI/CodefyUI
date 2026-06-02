import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import { ParamField } from '../shared/ParamField';
import { MathText } from '../shared/MathText';
import { CATEGORY_COLORS } from '../../styles/theme';
import { isParamVisible } from '../../utils';
import styles from './NodeConfigPanel.module.css';

export function NodeConfigPanel() {
  const activeTab = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)!);
  const nodes = activeTab.nodes;
  const selectedNodeId = activeTab.selectedNodeId;
  const updateNodeParams = useTabStore((s) => s.updateNodeParams);
  const openPresetModal = useTabStore((s) => s.openPresetModal);
  const isCanvasPanning = useUIStore((s) => s.isCanvasPanning);
  const { t, tn } = useI18n();

  const selectedNode = nodes.find((n) => n.id === selectedNodeId);
  const def = selectedNode?.data.definition;
  const nodeName = def?.node_name ?? '';

  const handleChange = (paramName: string, value: any) => {
    if (!selectedNodeId) return;
    updateNodeParams(selectedNodeId, { [paramName]: value });
  };

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
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {def.params
                .filter((param) => isParamVisible(param, selectedNode.data.params))
                .map((param) => (
                <div key={param.name}>
                  <ParamField
                    param={param}
                    value={selectedNode.data.params[param.name]}
                    onChange={handleChange}
                    siblingParams={selectedNode.data.params}
                  />
                  {param.description && (
                    <div className={styles.paramHint}>
                      {tn(nodeName, `param.${param.name}`, param.description)}
                    </div>
                  )}
                  {(param.min_value !== null || param.max_value !== null) && (
                    <div className={styles.paramHint}>
                      {t('config.range', {
                        min: param.min_value !== null ? param.min_value : '-∞',
                        max: param.max_value !== null ? param.max_value : '+∞',
                      })}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        ) : !isPreset ? (
          <div className={styles.noParams}>{t('config.noParams')}</div>
        ) : null}

        {/* I/O info section */}
        {def && (def.inputs.length > 0 || def.outputs.length > 0) && (
          <div style={{ marginTop: 20 }}>
            <div className={styles.sectionHeaderPorts}>{t('config.ports')}</div>

            {def.inputs.length > 0 && (
              <div style={{ marginBottom: 8 }}>
                <div className={styles.portSubLabel}>{t('config.inputs')}</div>
                {def.inputs.map((inp) => (
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

            {def.outputs.length > 0 && (
              <div>
                <div className={styles.portSubLabel}>{t('config.outputs')}</div>
                {def.outputs.map((out) => (
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
