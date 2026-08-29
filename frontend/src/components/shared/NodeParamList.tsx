import { useState } from 'react';
import type { NodeDefinition, ParamDefinition } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import { isParamVisible } from '../../utils';
import {
  localizedPackTitle,
  nodeMissingPack,
  usePackAvailability,
} from '../../utils/packAvailability';
import { ParamField } from './ParamField';
import styles from './NodeParamList.module.css';

interface NodeParamListProps {
  /**
   * Canvas id of the node being edited. A falsy id renders the fields but
   * commits nothing — the node is on screen without a place to write to.
   */
  nodeId: string | null;
  definition: NodeDefinition | undefined;
  /** The node's live params, i.e. `node.data.params`. */
  params: Record<string, any>;
  /** Extra class on the list wrapper, for surface-specific spacing. */
  className?: string;
}

/**
 * The editable parameter form for one node.
 *
 * Extracted from `NodeConfigPanel` when the Node Detail Modal (#127) grew a
 * second parameter surface. Both render THIS component, so conditional
 * visibility (`visible_when`), the two-tier basic/advanced split (core#134),
 * the range / description hints, the per-field validation `ParamField`
 * applies, and — most importantly — the store action the edit lands on
 * cannot drift between the side panel and the modal.
 *
 * Edits go through `updateNodeParams`, which marks the node dirty for partial
 * re-execution. It deliberately pushes no undo snapshot: parameter typing is
 * continuous, and a snapshot per keystroke would bury real structural edits
 * under hundreds of entries. Any caller wanting different semantics must
 * change them here, for every surface at once.
 */
export function NodeParamList({ nodeId, definition, params, className }: NodeParamListProps) {
  const updateNodeParams = useTabStore((s) => s.updateNodeParams);
  const { t, tn } = useI18n();
  const { byId, loaded, unsupported } = usePackAvailability();
  // Whole-node requirement (`requires_pack`), as opposed to the per-option
  // one ParamField applies. Null on a base install and on every server
  // without a Package Center, so the banner is the rare case.
  const missingPack = nodeMissingPack(definition, byId, loaded, unsupported);
  // Collapsed on every open, on purpose. Teaching mode's default view is the
  // one a class sees; a sticky "expanded" from an earlier session would make
  // that view depend on history rather than on the node.
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const nodeName = definition?.node_name ?? '';

  const handleChange = (paramName: string, value: any) => {
    if (!nodeId) return;
    updateNodeParams(nodeId, { [paramName]: value });
  };

  // `visible_when` is applied FIRST: a param the current configuration has no
  // use for should not be counted in the Advanced badge either, or an SGD
  // node would advertise six hidden Adam knobs.
  const visible = (definition?.params ?? []).filter((param) => isParamVisible(param, params));
  const basic = visible.filter((param) => !param.advanced);
  const advanced = visible.filter((param) => param.advanced);

  const renderField = (param: ParamDefinition) => (
    <div key={param.name}>
      <ParamField
        param={param}
        value={params[param.name]}
        onChange={handleChange}
        siblingParams={params}
        // The banner below is already this node's route to the Package
        // Center; a gated select would otherwise add a second button to the
        // same place, a few rows further down. The pack ID, not a flag: the
        // banner opens exactly one pack, and an option needing a different
        // one still has to offer its own way there.
        hidePackActionFor={missingPack?.packId}
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
  );

  return (
    <div className={className ? `${styles.list} ${className}` : styles.list}>
      {/* Above the fields, and never INSTEAD of them: the params of a node
          whose pack is missing still have to be readable and editable, or a
          graph saved on a machine that had the pack could not be inspected
          on one that does not. */}
      {missingPack !== null && (
        <div role="note" className={styles.packBanner}>
          {t('config.needsPack', {
            pack: localizedPackTitle(t, byId, missingPack.packId),
          })}{' '}
          <button
            type="button"
            className={styles.packLink}
            // Named, because this banner and a gated select's hint can both
            // be on screen for the same node, each with its own "Install
            // pack": two identically named controls in one list is a choice
            // nobody navigating by control can make.
            aria-label={t('paramField.installPackFor', {
              pack: localizedPackTitle(t, byId, missingPack.packId),
            })}
            onClick={() => useUIStore.getState().openPackCenter(missingPack.packId)}
          >
            {t('paramField.installPack')}
          </button>
        </div>
      )}

      {basic.map(renderField)}

      {advanced.length > 0 && (
        <div className={styles.advanced}>
          <button
            type="button"
            className={styles.advancedToggle}
            onClick={() => setAdvancedOpen((open) => !open)}
            aria-expanded={advancedOpen}
            aria-controls={`advanced-params-${nodeId ?? 'none'}`}
          >
            <span className={styles.advancedChevron} aria-hidden="true">
              {advancedOpen ? '▾' : '▸'}
            </span>
            {t('config.advanced')}
            <span className={styles.advancedCount}>{advanced.length}</span>
          </button>
          {advancedOpen && (
            <div
              id={`advanced-params-${nodeId ?? 'none'}`}
              className={styles.advancedBody}
            >
              {advanced.map(renderField)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
