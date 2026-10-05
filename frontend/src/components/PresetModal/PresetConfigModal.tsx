import { useState, useEffect, useMemo } from 'react';
import { useTabStore } from '../../store/tabStore';
import { readablePresetNodes } from '../../utils';
import { ParamField } from '../shared/ParamField';
import { useI18n } from '../../i18n';
import type { PresetDefinition } from '../../types';
import styles from './PresetConfigModal.module.css';

type InternalParams = Record<string, Record<string, any>>;

/** Two param values compared the way the dialog copies them: through JSON. */
function sameValue(a: unknown, b: unknown): boolean {
  return Object.is(a, b) || JSON.stringify(a) === JSON.stringify(b);
}

/**
 * Would Apply change the card? The dialog starts from a copy of EVERY value
 * the card stores, and Apply writes all of them back, edited or not. So each
 * is compared with what the card holds now, or, for a param it does not store
 * yet, with the default its field showed.
 */
function applyChangesCard(
  local: InternalParams,
  stored: InternalParams,
  preset: PresetDefinition,
): boolean {
  const exposed: unknown = preset.exposed_params;
  const defaultOf = (internalNodeId: string, paramName: string): unknown => {
    if (!Array.isArray(exposed)) return undefined;
    const ep = exposed.find(
      (e) =>
        e !== null &&
        typeof e === 'object' &&
        e.internal_node === internalNodeId &&
        e.param_name === paramName,
    );
    return ep?.param_def?.default;
  };
  return Object.entries(local).some(([internalNodeId, params]) =>
    Object.entries(params).some(([paramName, value]) => {
      const before = stored[internalNodeId]?.[paramName];
      return !sameValue(value, before === undefined ? defaultOf(internalNodeId, paramName) : before);
    }),
  );
}

export function PresetConfigModal() {
  // Optional, like the three other root-mounted modals: this is mounted for
  // the whole session, including while no tab is open at all (the welcome
  // screen). There is no modal to show then, and the guard below already
  // says so -- what the non-null assertion did was crash on the way to it.
  const activeTab = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId) ?? null);
  const closePresetModal = useTabStore((s) => s.closePresetModal);
  const updatePresetInternalParam = useTabStore((s) => s.updatePresetInternalParam);
  const pushUndoSnapshot = useTabStore((s) => s.pushUndoSnapshot);
  const { t } = useI18n();

  const presetModalNodeId = activeTab?.presetModalNodeId ?? null;
  const node = activeTab?.nodes.find((n) => n.id === presetModalNodeId);
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

  // Group exposed params. A document's own definition is kept as it came even
  // when the backend cannot read it (#541), so a missing or junk list draws
  // no fields rather than throw.
  const groupedParams = useMemo(() => {
    if (!preset) return {};
    const groups: Record<string, typeof preset.exposed_params> = {};
    const exposed: unknown = preset.exposed_params;
    for (const ep of Array.isArray(exposed) ? exposed : []) {
      if (ep === null || typeof ep !== 'object') continue;
      const group = ep.group || t('preset.generalGroup');
      if (!groups[group]) groups[group] = [];
      groups[group].push(ep);
    }
    return groups;
  }, [preset, t]);

  if (!presetModalNodeId || !node || !preset) return null;
  const innerNodes = readablePresetNodes(preset);

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
    // One undo step for the whole Apply, taken before the first write so it
    // holds the card as it was. An Apply that changes nothing takes none.
    if (applyChangesCard(localParams, currentInternalParams, preset)) pushUndoSnapshot();
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
            {innerNodes.map((n, i) => (
              <div key={n.id} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span className={styles.pipelineNodeChip}>{n.type}</span>
                {i < innerNodes.length - 1 && (
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
