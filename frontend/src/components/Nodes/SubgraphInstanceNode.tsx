import { memo } from 'react';
import type { NodeProps } from '@xyflow/react';

import type { AppNode } from '../../types';
import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import { subgraphIdOf } from '../../utils/subgraph';
import { BaseNodeBody } from './BaseNode';
import styles from './SubgraphInstanceNode.module.css';

/**
 * One use of a subgraph on the canvas (core#137).
 *
 * Deliberately a thin wrapper over `BaseNodeBody`: an instance's ports come
 * from a synthesized `NodeDefinition` (exactly as a preset's do), so every
 * behaviour the base card already has -- handle colours, edge detach, the
 * trigger target, execution status, selection -- applies with no new code.
 * The footer adds the two things the base card cannot know: that this is a
 * block, and how many nodes are inside it.
 *
 * Double-click ENTERS the subgraph; that lives in `BaseNode` alongside the
 * SequentialModel and preset cases, because the handler that would otherwise
 * open the Node Detail Modal is there.
 */
function SubgraphInstanceNode(props: NodeProps<AppNode>) {
  const { data } = props;
  const { t } = useI18n();
  const subgraphId = subgraphIdOf(data.type);
  const definition = useTabStore((s) => {
    const tab = s.tabs.find((tabState) => tabState.id === s.activeTabId);
    return tab?.subgraphs.find((d) => d.id === subgraphId);
  });

  const bodyExtra = (
    <div className={styles.footer} data-testid="subgraph-footer">
      <span className={styles.badge}>{t('subgraph.badge')}</span>
      <span className={styles.count}>
        {definition
          ? t('subgraph.nodeCount', { count: definition.nodes.length })
          : t('subgraph.missing')}
      </span>
    </div>
  );

  return <BaseNodeBody {...props} bodyExtra={bodyExtra} />;
}

export default memo(SubgraphInstanceNode);
