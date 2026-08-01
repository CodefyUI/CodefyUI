import type { NodeDefinition } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { isParamVisible } from '../../utils';
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
 * visibility (`visible_when`), the range / description hints, the per-field
 * validation `ParamField` applies, and — most importantly — the store action
 * the edit lands on cannot drift between the side panel and the modal.
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

  const nodeName = definition?.node_name ?? '';

  const handleChange = (paramName: string, value: any) => {
    if (!nodeId) return;
    updateNodeParams(nodeId, { [paramName]: value });
  };

  const visible = (definition?.params ?? []).filter((param) => isParamVisible(param, params));

  return (
    <div className={className ? `${styles.list} ${className}` : styles.list}>
      {visible.map((param) => (
        <div key={param.name}>
          <ParamField
            param={param}
            value={params[param.name]}
            onChange={handleChange}
            siblingParams={params}
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
  );
}
