import { Handle, Position, type NodeProps } from '@xyflow/react';
import { useI18n } from '../../i18n';
import styles from './StartNode.module.css';

export function StartNode(_: NodeProps) {
  const { t } = useI18n();
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
        className={styles.handle}
      />
    </div>
  );
}
