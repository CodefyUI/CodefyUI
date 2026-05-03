import { memo, useCallback } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { AppNode } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { BaseNodeBody } from './BaseNode';
import styles from './TextInputVizNode.module.css';

function TextInputVizNode(props: NodeProps<AppNode>) {
  const { id, data } = props;
  const updateNodeParams = useTabStore((s) => s.updateNodeParams);
  const { t } = useI18n();
  const value = data.params?.value ?? '';

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      updateNodeParams(id, { value: e.target.value });
    },
    [id, updateNodeParams],
  );

  // ``nodrag`` stops xyflow from initiating a node drag when the user clicks
  // into the textarea; ``nowheel`` lets the textarea consume scroll events
  // instead of panning the canvas. Both classes are part of the xyflow API.
  const bodyExtra = (
    <div className={styles.editorArea}>
      <textarea
        className={`nodrag nowheel ${styles.textarea}`}
        value={String(value)}
        onChange={handleChange}
        placeholder={t('textInput.placeholder')}
        spellCheck={false}
      />
      <div className={styles.meta}>
        {t('textInput.charCount', { count: String(value).length })}
      </div>
    </div>
  );

  return <BaseNodeBody {...props} bodyExtra={bodyExtra} />;
}

export default memo(TextInputVizNode);
