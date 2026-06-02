import { useState, useEffect, useMemo } from 'react';
import { useTabStore } from '../../store/tabStore';
import { ParamField } from '../shared/ParamField';
import { useI18n } from '../../i18n';
import styles from './PresetConfigModal.module.css';

export function PresetConfigModal() {
  const activeTab = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)!);
  const closePresetModal = useTabStore((s) => s.closePresetModal);
  const updatePresetInternalParam = useTabStore((s) => s.updatePresetInternalParam);
  const { t } = useI18n();

  const presetModalNodeId = activeTab.presetModalNodeId;
  const node = activeTab.nodes.find((n) => n.id === presetModalNodeId);
  const preset = node?.data.presetDefinition;
  const currentInternalParams = node?.data.internalParams ?? {};

  // Local state for editing
  const [localParams, setLocalParams] = useState<Record<string, Record<string, any>>>({});

  useEffect(() => {
    // currentInternalParams is `internalParams ?? {}`, so always truthy
    /* v8 ignore start */
    if (currentInternalParams) {
      setLocalParams(JSON.parse(JSON.stringify(currentInternalParams)));
    }
    /* v8 ignore stop */
  }, [presetModalNodeId]);

  // Group exposed params
  const groupedParams = useMemo(() => {
    if (!preset) return {};
    const groups: Record<string, typeof preset.exposed_params> = {};
    for (const ep of preset.exposed_params) {
      const group = ep.group || t('preset.generalGroup');
      if (!groups[group]) groups[group] = [];
      groups[group].push(ep);
    }
    return groups;
  }, [preset, t]);

  if (!presetModalNodeId || !node || !preset) return null;

  const handleParamChange = (internalNodeId: string, paramName: string, value: any) => {
    setLocalParams((prev) => ({
      ...prev,
      [internalNodeId]: {
        ...prev[internalNodeId],
        [paramName]: value,
      },
    }));
  };

  const handleApply = () => {
    for (const [internalNodeId, params] of Object.entries(localParams)) {
      for (const [paramName, value] of Object.entries(params)) {
        updatePresetInternalParam(presetModalNodeId, internalNodeId, paramName, value);
      }
    }
    closePresetModal();
  };

  const handleCancel = () => {
    closePresetModal();
  };

  return (
    <div
      className={styles.overlay}
      onClick={(e) => {
        if (e.target === e.currentTarget) handleCancel();
      }}
    >
      <div className={styles.modal}>
        {/* Header */}
        <div className={styles.header}>
          <div className={styles.headerContent}>
            <div className={styles.headerTitleRow}>
              <span className={styles.headerTitle}>{preset.preset_name}</span>
              <span className={styles.headerBadge}>{t('preset.badge')}</span>
            </div>
            <div className={styles.headerDescription}>{preset.description}</div>
          </div>
          <button type="button" onClick={handleCancel} className={styles.closeBtn}>
            ✕
          </button>
        </div>

        {/* Pipeline preview */}
        <div className={styles.pipelinePreview}>
          <div className={styles.pipelineInner}>
            {preset.nodes.map((n, i) => (
              <div key={n.id} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span className={styles.pipelineNodeChip}>{n.type}</span>
                {i < preset.nodes.length - 1 && (
                  <span className={styles.pipelineArrow}>→</span>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Params content */}
        <div className={styles.paramsContent}>
          {Object.entries(groupedParams).map(([group, params]) => (
            <div key={group} style={{ marginBottom: 18 }}>
              <div className={styles.groupHeader}>{group}</div>
              <div className={styles.groupParams}>
                {params.map((ep) => {
                  if (!ep.param_def) return null;
                  const val = localParams[ep.internal_node]?.[ep.param_name] ?? ep.param_def.default;
                  return (
                    <div key={`${ep.internal_node}-${ep.param_name}`}>
                      <ParamField
                        param={ep.param_def}
                        value={val}
                        onChange={(_name, value) => handleParamChange(ep.internal_node, ep.param_name, value)}
                        label={ep.display_name}
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        {/* Footer buttons */}
        <div className={styles.footer}>
          <button type="button" onClick={handleCancel} className={styles.cancelBtn}>
            {t('preset.cancel')}
          </button>
          <button type="button" onClick={handleApply} className={styles.applyBtn}>
            {t('preset.apply')}
          </button>
        </div>
      </div>
    </div>
  );
}
