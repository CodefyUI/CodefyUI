import { useEffect, useState } from 'react';
import type { Node } from '@xyflow/react';
import type { NodeData, NodeDefinition, ParamDefinition, PortDefinition } from '../../types';
import { fetchNodeDefinition } from '../../api/rest';
import { useI18n } from '../../i18n';
import { MathText } from '../shared/MathText';
import { getPortColor } from '../../utils';
import styles from './NodeDetailModal.module.css';

function PortRows({ ports }: { ports: PortDefinition[] }) {
  const { t } = useI18n();
  return (
    <>
      {ports.map((p) => (
        <div key={p.name} className={styles.docsPortRow}>
          <span className={styles.docsPortDot} style={{ background: getPortColor(p.data_type) }} />
          <span className={styles.docsPortName}>{p.name}</span>
          <span className={styles.docsPortType}>{p.data_type}</span>
          {p.optional && <span className={styles.docsPortOptional}>{t('config.optional')}</span>}
          {p.description && <span className={styles.docsPortDesc}>{p.description}</span>}
        </div>
      ))}
    </>
  );
}

function ParamRow({ param, nodeName }: { param: ParamDefinition; nodeName: string }) {
  const { t, tn } = useI18n();
  const meta: string[] = [`${t('nodeDetail.docs.default')}: ${String(param.default)}`];
  if (param.min_value !== null || param.max_value !== null) {
    meta.push(
      `${t('nodeDetail.docs.range')}: ${param.min_value !== null ? param.min_value : '-∞'} — ` +
        `${param.max_value !== null ? param.max_value : '+∞'}`,
    );
  }
  if (param.options.length > 0) {
    meta.push(`${t('nodeDetail.docs.options')}: ${param.options.join(', ')}`);
  }
  return (
    <div className={styles.docsParamRow}>
      <div className={styles.docsParamHead}>
        <span className={styles.docsParamName}>{param.name}</span>
        <span className={styles.docsParamType}>{param.param_type}</span>
      </div>
      {param.description && (
        <div className={styles.docsParamDesc}>
          {tn(nodeName, `param.${param.name}`, param.description)}
        </div>
      )}
      <div className={styles.docsParamMeta}>{meta.join('  ·  ')}</div>
    </div>
  );
}

/**
 * The node's own documentation: description plus a full parameter and port
 * reference — the hover tooltip card, at a size you can actually read.
 *
 * Renders immediately from the definition already on the node, then refreshes
 * from `GET /api/nodes/{name}` so a reloaded plugin (or a machine whose device
 * options changed) documents what the backend will really run. A failed
 * refresh is silent by design: the local copy is still correct enough to read,
 * and a docs tab is no place to surface a transport error.
 */
export function DocsTab({ node }: { node: Node<NodeData> }) {
  const { t, tn } = useI18n();
  const local = node.data.definition;
  const nodeName = local?.node_name ?? node.data.type;
  const [remote, setRemote] = useState<NodeDefinition | null>(null);

  useEffect(() => {
    let cancelled = false;
    setRemote(null);
    fetchNodeDefinition(nodeName)
      .then((def) => {
        if (!cancelled) setRemote(def);
      })
      .catch(() => {
        /* offline / unknown node — the local definition still documents it */
      });
    return () => {
      cancelled = true;
    };
  }, [nodeName]);

  const def = remote ?? local;
  const description = def?.description ?? '';
  const inputs = def?.inputs ?? [];
  const outputs = def?.outputs ?? [];
  const params = def?.params ?? [];

  return (
    <div className={`${styles.tabBody} ${styles.docsBody}`}>
      <section className={styles.docsSection}>
        <div className={styles.docsSectionTitle}>{t('nodeDetail.docs.description')}</div>
        {description ? (
          <MathText
            as="div"
            className={styles.docsDescription}
            text={tn(nodeName, 'description', description)}
          />
        ) : (
          <div className={styles.docsEmpty}>{t('nodeDetail.docs.noDescription')}</div>
        )}
      </section>

      <section className={styles.docsSection}>
        <div className={styles.docsSectionTitle}>{t('nodeDetail.docs.params')}</div>
        {params.length === 0 ? (
          <div className={styles.docsEmpty}>{t('nodeDetail.docs.noParams')}</div>
        ) : (
          params.map((p) => <ParamRow key={p.name} param={p} nodeName={nodeName} />)
        )}
      </section>

      <section className={styles.docsSection}>
        <div className={styles.docsSectionTitle}>{t('nodeDetail.docs.inputs')}</div>
        {inputs.length === 0 ? (
          <div className={styles.docsEmpty}>{t('nodeDetail.docs.noPorts')}</div>
        ) : (
          <PortRows ports={inputs} />
        )}
      </section>

      <section className={styles.docsSection}>
        <div className={styles.docsSectionTitle}>{t('nodeDetail.docs.outputs')}</div>
        {outputs.length === 0 ? (
          <div className={styles.docsEmpty}>{t('nodeDetail.docs.noPorts')}</div>
        ) : (
          <PortRows ports={outputs} />
        )}
      </section>
    </div>
  );
}
