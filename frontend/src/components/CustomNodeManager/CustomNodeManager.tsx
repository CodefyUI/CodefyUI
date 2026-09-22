import { useState, useEffect, useCallback, useId, useRef } from 'react';
import { listCustomNodes, toggleCustomNode, uploadCustomNode, deleteCustomNode, type CustomNodeInfo } from '../../api/rest';
import { useDialogStore } from '../../store/dialogStore';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import { confirm } from '../../utils/dialog';
import styles from './CustomNodeManager.module.css';

/**
 * The manager as the app mounts it: once, at the root, driven by
 * `uiStore.customNodeManagerOpen`, like the Package Center. The toolbar button
 * and the Custom tab only raise the flag. At the root rather than under the
 * toolbar, because the toolbar is not mounted at all while no graph tab is
 * open (#472).
 */
export function CustomNodeManagerModal() {
  const open = useUIStore((s) => s.customNodeManagerOpen);
  const close = useUIStore((s) => s.closeCustomNodeManager);
  if (!open) return null;
  return <CustomNodeManager onClose={close} />;
}

interface CustomNodeManagerProps {
  onClose: () => void;
}

export function CustomNodeManager({ onClose }: CustomNodeManagerProps) {
  const [nodes, setNodes] = useState<CustomNodeInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const titleId = useId();
  const { reload } = useNodeDefStore();
  const { t } = useI18n();

  // Focus starts inside the panel and goes back where it came from, as in the
  // Package Center. Mounted at the app root, the manager comes after the whole
  // editor in the page, so a panel that took no focus left a keyboard user to
  // Tab through the sidebar, every node and edge on the canvas and the panels
  // to reach it (the Custom Nodes manager issue). Not a focus trap: Tab still
  // walks out into the page, as it does from the other panels.
  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    panelRef.current?.focus();
    return () => {
      if (previouslyFocused && previouslyFocused.isConnected) {
        previouslyFocused.focus();
      }
    };
  }, []);

  // Escape closes it, unless a confirm is open over it. Deleting a file asks
  // first, and that confirm renders above this panel and cancels on Escape
  // itself, so the press is the confirm's: without this, one press would
  // cancel the confirm and close the manager under it. The same guard as the
  // Package Center's.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      if (useDialogStore.getState().active !== null) return;
      e.preventDefault();
      onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

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
      confirmText: t('customNodes.delete'),
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
      <div
        ref={panelRef}
        className={styles.modal}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        onClick={(e) => e.stopPropagation()}
      >
        <div className={styles.header}>
          <h2 className={styles.title} id={titleId}>{t('customNodes.title')}</h2>
          {/* A multiplication sign with a name on it. The literal letter "x"
              this used to render was the one untranslated string in the
              modal, and a screen reader announced the button as "x". */}
          <button
            type="button"
            className={styles.closeButton}
            onClick={onClose}
            title={t('customNodes.close')}
            aria-label={t('customNodes.close')}
          >
            &#215;
          </button>
        </div>

        {error && <div className={styles.error}>{error}</div>}

        <div className={styles.body}>
          {loading && <div className={styles.message}>{t('customNodes.loading')}</div>}

          {!loading && nodes.length === 0 && (
            <div className={styles.message}>{t('customNodes.emptyUpload')}</div>
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
