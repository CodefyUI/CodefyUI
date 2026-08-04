import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import { subgraphIdOf } from '../../utils/subgraph';
import type { NodeDetailTabContext } from './tabs';

/**
 * What a subgraph instance is, and the way into it (core#137, spec item 6).
 *
 * The modal is the one place that can show the boundary as DATA -- which
 * inner node and port each handle stands for -- rather than as a row of
 * coloured dots on the card.
 */
export function SubgraphTab({ ctx }: { ctx: NodeDetailTabContext }) {
  const { t } = useI18n();
  const enterSubgraph = useTabStore((s) => s.enterSubgraph);
  const closeNodeDetail = useTabStore((s) => s.closeNodeDetail);
  const subgraphId = subgraphIdOf(ctx.node.data.type);
  const definition = useTabStore((s) => {
    const tab = s.tabs.find((x) => x.id === s.activeTabId);
    return tab?.subgraphs.find((d) => d.id === subgraphId);
  });

  const inputs = definition?.interface.inputs ?? [];
  const outputs = definition?.interface.outputs ?? [];

  const portList = (
    ports: { port: string; innerNode: string; innerPort: string }[],
  ) => (
    <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
      {ports.map((p) => (
        <li key={p.port} style={{ fontSize: 12, color: '#bbb' }}>
          <code>{p.port}</code>
          {' → '}
          <span style={{ color: '#888' }}>
            {p.innerNode}.{p.innerPort}
          </span>
        </li>
      ))}
    </ul>
  );

  return (
    <div style={{ padding: 12 }} data-testid="subgraph-tab">
      <h4 style={{ margin: '0 0 8px', fontSize: 13 }}>
        {t('subgraph.detail.interface')}
      </h4>
      {inputs.length === 0 && outputs.length === 0 ? (
        <p style={{ fontSize: 12, color: '#888' }}>
          {t('subgraph.detail.empty')}
        </p>
      ) : (
        <>
          {inputs.length > 0 && (
            <section>
              <strong style={{ fontSize: 12 }}>
                {t('subgraph.detail.inputs')}
              </strong>
              {portList(inputs)}
            </section>
          )}
          {outputs.length > 0 && (
            <section style={{ marginTop: 10 }}>
              <strong style={{ fontSize: 12 }}>
                {t('subgraph.detail.outputs')}
              </strong>
              {portList(outputs)}
            </section>
          )}
        </>
      )}
      <button
        type="button"
        style={{ marginTop: 14 }}
        disabled={!definition}
        onClick={() => {
          // Close first: entering swaps the canvas underneath, and a modal
          // still pinned to the outer node would be describing a node that
          // is no longer on screen.
          closeNodeDetail();
          enterSubgraph(ctx.nodeId);
        }}
      >
        {t('subgraph.detail.enter')}
      </button>
    </div>
  );
}

export default SubgraphTab;
