import { useState, useEffect, useCallback, useRef } from 'react';
import { listCustomNodes, toggleCustomNode, uploadCustomNode, deleteCustomNode, type CustomNodeInfo } from '../../api/rest';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useI18n } from '../../i18n';
import { confirm } from '../../utils/dialog';
import styles from './CustomNodeManager.module.css';

interface CustomNodeManagerProps {
  onClose: () => void;
}

export function CustomNodeManager({ onClose }: CustomNodeManagerProps) {
  const [nodes, setNodes] = useState<CustomNodeInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { reload } = useNodeDefStore();
  const { t } = useI18n();

  const fetchNodes = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await listCustomNodes();
      setNodes(result);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  // The parent mounts this component only while the manager is open, so the
  // node list is fetched once on mount rather than synced off an `open` prop.
  useEffect(() => {
    fetchNodes();
  }, [fetchNodes]);

  const handleToggle = useCallback(async (filename: string) => {
    try {
      await toggleCustomNode(filename);
      await fetchNodes();
      await reload();
    } catch (e) {
      setError((e as Error).message);
    }
  }, [fetchNodes, reload]);

  const handleDelete = useCallback(async (filename: string) => {
    const ok = await confirm({
      title: t('customNodes.delete.confirm', { name: filename }),
      confirmText: 'Delete',
      variant: 'danger',
    });
    if (!ok) return;
    try {
      await deleteCustomNode(filename);
      await fetchNodes();
      await reload();
    } catch (e) {
      setError((e as Error).message);
    }
  }, [fetchNodes, reload, t]);

  const handleUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      await uploadCustomNode(file);
      await fetchNodes();
      await reload();
    } catch (e) {
      setError((e as Error).message);
    }
    event.target.value = '';
  }, [fetchNodes, reload]);

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <h2 className={styles.title}>{t('customNodes.title')}</h2>
          <button type="button" className={styles.closeButton} onClick={onClose}>x</button>
        </div>

        {error && <div className={styles.error}>{error}</div>}

        <div className={styles.body}>
          {loading && <div className={styles.message}>{t('customNodes.loading')}</div>}

          {!loading && nodes.length === 0 && (
            <div className={styles.message}>{t('customNodes.empty')}</div>
          )}

          {!loading && nodes.map((node) => (
            <div key={node.filename} className={styles.nodeRow}>
              <div className={styles.nodeInfo}>
                <span className={styles.nodeFilename}>{node.filename}</span>
                {node.nodes.length > 0 && (
                  <span className={styles.nodeNames}>
                    {node.nodes.join(', ')}
                  </span>
                )}
              </div>
              <div className={styles.nodeActions}>
                <button type="button"
                  className={`${styles.toggleButton} ${node.enabled ? styles.toggleOn : styles.toggleOff}`}
                  onClick={() => handleToggle(node.filename)}
                >
                  {node.enabled ? t('customNodes.enabled') : t('customNodes.disabled')}
                </button>
                <button type="button"
                  className={styles.deleteButton}
                  onClick={() => handleDelete(node.filename)}
                >
                  {t('customNodes.delete')}
                </button>
              </div>
            </div>
          ))}
        </div>

        <div className={styles.footer}>
          <button type="button"
            className={styles.uploadButton}
            onClick={() => fileInputRef.current?.click()}
          >
            {t('customNodes.upload')}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".py"
            style={{ display: 'none' }}
            onChange={handleUpload}
          />
        </div>
      </div>
    </div>
  );
}
