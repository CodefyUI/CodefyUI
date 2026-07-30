import { Handle, Position, type NodeProps } from '@xyflow/react';
import { useI18n } from '../../i18n';
import { useUIStore } from '../../store/uiStore';
import styles from './StartNode.module.css';

export function StartNode({ id }: NodeProps) {
  const { t } = useI18n();
  // Red ring while the user is dragging this node's trigger edge off its
  // source diamond (edge reconnect in progress) — warns that dropping on
  // empty space deletes the edge.
  const detaching = useUIStore(
    (s) =>
      s.reconnectingHandle !== null &&
      s.reconnectingHandle.nodeId === id &&
      s.reconnectingHandle.type === 'source' &&
      s.reconnectingHandle.handleId === 'trigger',
  );
  return (
    <div className={styles.startNode}>
      <svg className={styles.icon} viewBox="0 0 16 16" fill="none">
        <path
          d="M3 1 V15 M3 2 H12 L10 5 L12 8 H3"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinejoin="round"
          strokeLinecap="round"
          fill="currentColor"
          fillOpacity="0.4"
        />
      </svg>
      <span>{t('node.start.label')}</span>
      <Handle
        type="source"
        position={Position.Right}
        id="trigger"
        className={`${styles.handle}${detaching ? ` ${styles.handleDetaching}` : ''}`}
      />
    </div>
  );
}
