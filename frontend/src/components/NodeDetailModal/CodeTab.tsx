import type { Node } from '@xyflow/react';
import type { NodeData, ParamDefinition } from '../../types';
import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import {
  SCRIPT_MAX_PORTS,
  SELECTABLE_DATA_TYPES,
  getPortColor,
  resolveDynamicInputs,
  resolveDynamicOutputs,
} from '../../utils';
import { ScriptCodeField } from '../shared/ScriptCodeField';
import type { NodeDetailTabContext } from './tabs';
import styles from './CodeTab.module.css';

/** The `code` param of a node, if it has one. */
export function codeParamOf(node: Node<NodeData> | undefined): ParamDefinition | undefined {
  return node?.data.definition?.params.find((p) => p.param_type === 'code');
}

/** Whether the Code tab applies to this node at all. */
export function hasCodeParam(node: Node<NodeData> | undefined): boolean {
  return codeParamOf(node) !== undefined;
}

/**
 * The Code tab: the script, full width, plus the port shape it runs against
 * (core#131).
 *
 * The same `ScriptCodeField` renders in the 320px param column, and both
 * write the one store param — so this tab is a bigger window onto the same
 * text rather than a second copy of it. The ports section lives HERE and not
 * in the param column because per-port type selects need the width, and
 * because ports and code are one decision: `inputs['in2']` only means
 * something once port 2 exists.
 */
export function CodeTab({ ctx }: { ctx: NodeDetailTabContext }) {
  const { t } = useI18n();
  const updateNodeParams = useTabStore((s) => s.updateNodeParams);
  const { node, nodeId } = ctx;
  const param = codeParamOf(node);
  const def = node.data.definition;
  const params = node.data.params;

  if (!param || !def) {
    return <div className={styles.empty}>{t('nodeDetail.code.unavailable')}</div>;
  }

  const inputs = resolveDynamicInputs(def, params);
  const outputs = resolveDynamicOutputs(def, params);

  const setCount = (countParam: string, typesParam: string, next: number) => {
    // Rewrite the type list at the same time, so shrinking then growing the
    // port count cannot resurrect a type the user no longer sees. The
    // resolver already repeats the last entry, which is what fills the new
    // ports here.
    const ports = countParam === 'input_ports' ? inputs : outputs;
    const types = Array.from(
      { length: next },
      (_, i) => ports[i]?.data_type ?? ports[ports.length - 1]?.data_type ?? 'ANY',
    );
    updateNodeParams(nodeId, { [countParam]: next, [typesParam]: types.join(',') });
  };

  const setType = (typesParam: string, ports: { data_type: string }[], index: number, value: string) => {
    const types = ports.map((port, i) => (i === index ? value : port.data_type));
    updateNodeParams(nodeId, { [typesParam]: types.join(',') });
  };

  const renderSide = (
    side: 'inputs' | 'outputs',
    ports: { name: string; data_type: string }[],
    countParam: string,
    typesParam: string,
  ) => (
    <div className={styles.side}>
      <div className={styles.sideHead}>
        <span className={styles.sideTitle}>
          {side === 'inputs' ? t('nodeDetail.code.inputs') : t('nodeDetail.code.outputs')}
        </span>
        <select
          className={styles.countSelect}
          aria-label={
            side === 'inputs'
              ? t('nodeDetail.code.inputCount')
              : t('nodeDetail.code.outputCount')
          }
          value={ports.length}
          onChange={(e) => setCount(countParam, typesParam, Number(e.target.value))}
        >
          {Array.from({ length: SCRIPT_MAX_PORTS }, (_, i) => (
            <option key={i + 1} value={i + 1} style={{ background: '#222' }}>
              {i + 1}
            </option>
          ))}
        </select>
      </div>
      {ports.map((port, index) => (
        <div key={port.name} className={styles.portRow}>
          <span className={styles.portName} style={{ color: getPortColor(port.data_type) }}>
            {port.name}
          </span>
          <select
            className={styles.typeSelect}
            aria-label={`${port.name} type`}
            value={port.data_type}
            onChange={(e) => setType(typesParam, ports, index, e.target.value)}
          >
            {SELECTABLE_DATA_TYPES.map((type) => (
              <option key={type} value={type} style={{ background: '#222' }}>
                {type}
              </option>
            ))}
          </select>
        </div>
      ))}
    </div>
  );

  return (
    <div className={styles.tab}>
      <ScriptCodeField
        param={param}
        value={params[param.name]}
        onChange={(name, value) => updateNodeParams(nodeId, { [name]: value })}
        displayLabel={t('nodeDetail.code.title')}
        variant="tab"
      />

      <div className={styles.contract}>{t('nodeDetail.code.contract')}</div>

      <div className={styles.ports}>
        {renderSide('inputs', inputs, 'input_ports', 'input_types')}
        {renderSide('outputs', outputs, 'output_ports', 'output_types')}
      </div>

      <div className={styles.security}>{t('nodeDetail.code.security')}</div>
    </div>
  );
}
