import { useState, useEffect, useLayoutEffect, useCallback, useId, useRef } from 'react';
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

/**
 * One key per row, kept through Enable/Disable (#504). A toggle renames the
 * file between `x.py` and `x.py.disabled`, so a row keyed by its file name
 * came back as a new row, and the button just pressed went with the old one,
 * taking keyboard focus with it. The name without `.disabled` stays put --
 * except when both files are on disk at once (x.py uploaded again after it
 * was disabled), where each of the two rows keeps its own file name, since two
 * rows cannot share a key.
 */
function rowKeys(nodes: CustomNodeInfo[]): string[] {
  const bases = nodes.map((node) => node.filename.replace(/\.disabled$/, ''));
  return bases.map((base, i) =>
    bases.indexOf(base) === bases.lastIndexOf(base) ? base : nodes[i].filename,
  );
}

/** Where focus goes once the next list lands: the first of `keys` still listed. */
interface FocusAfterRead {
  keys: string[];
  control: 'toggle' | 'delete';
}

/** Free the rows whose request was answered before read `seq` began. */
function freeRowsAnsweredBefore(busyRows: Map<string, number>, seq: number): void {
  for (const [key, answeredAt] of busyRows) {
    if (answeredAt < seq) busyRows.delete(key);
  }
}

export function CustomNodeManager({ onClose }: CustomNodeManagerProps) {
  const [nodes, setNodes] = useState<CustomNodeInfo[]>([]);
  // Only the first read blanks the list, when there is no list yet. A re-read
  // after an action keeps the rows on screen, and the button that was pressed
  // with them (#504).
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const uploadButtonRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  // Reads are numbered and only the newest one's answer is applied (#506).
  // With the rows kept up, a second action can start a read while the first
  // is still out, and the older answer can arrive last.
  const readSeq = useRef(0);
  // The last read begun when the error on screen was set. Only a read begun
  // after it clears it: a re-read that was already out knows nothing of an
  // action that failed in the meantime.
  const errorSeq = useRef(0);
  // Rows held from the press until the list shows what the action did. Until
  // then a toggled row still shows the file name the server has just renamed
  // away, and a second Enable/Disable or a Delete sent with that name is a
  // 404. A row maps to the last read begun before its request was answered
  // (Infinity while it is out), and the newest read begun after that frees
  // it -- before the palette reload, a full rediscovery, so a second press
  // can undo at once. A failed or cancelled action frees it straight away.
  const busyRows = useRef(new Map<string, number>());
  // The newest read whose list React has drawn. Rows are freed once it is,
  // not when its answer arrives: until then the buttons still carry the old
  // file names, and a press in between (a held Enter repeating as the re-read
  // lands) would send one.
  const [appliedSeq, setAppliedSeq] = useState(0);
  // The rows' buttons by row key, for handing focus on after a re-read.
  const toggleButtons = useRef(new Map<string, HTMLButtonElement>());
  const deleteButtons = useRef(new Map<string, HTMLButtonElement>());
  const focusAfterRead = useRef<FocusAfterRead | null>(null);
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

  const showError = useCallback((message: string) => {
    errorSeq.current = readSeq.current;
    setError(message);
  }, []);

  const fetchNodes = useCallback(async () => {
    const seq = ++readSeq.current;
    try {
      const result = await listCustomNodes();
      if (seq !== readSeq.current) return;
      setNodes(result);
      setAppliedSeq(seq);
      if (errorSeq.current < seq) setError(null);
    } catch (e) {
      if (seq !== readSeq.current) return;
      // The rows stay as they were, so no focus went down with one, and no
      // new list is coming to be drawn: free the rows now.
      focusAfterRead.current = null;
      freeRowsAnsweredBefore(busyRows.current, seq);
      showError((e as Error).message);
    } finally {
      if (seq === readSeq.current) setLoading(false);
    }
  }, [showError]);

  // The parent mounts this component only while the manager is open, so the
  // node list is fetched once on mount rather than synced off an `open` prop.
  useEffect(() => {
    fetchNodes();
  }, [fetchNodes]);

  // A layout effect, so the rows are freed in the same task that commits the
  // new list: no press can come between the new names and the freeing.
  useLayoutEffect(() => {
    freeRowsAnsweredBefore(busyRows.current, appliedSeq);
  }, [appliedSeq]);

  // Focus after a re-read, once its rows are committed (#504): back on the
  // row's toggle, or after a Delete on the next row's Delete, else the
  // previous row's, else Upload. Only focus that went down with a removed
  // row moves. A toggle's button survives the re-read and keeps it, and a
  // user who moved on while the request was out stays where they went.
  useEffect(() => {
    const pending = focusAfterRead.current;
    if (pending === null) return;
    focusAfterRead.current = null;
    const active = document.activeElement;
    if (active !== null && active !== document.body) return;
    const buttons = pending.control === 'toggle' ? toggleButtons.current : deleteButtons.current;
    const target = pending.keys.map((key) => buttons.get(key)).find((el) => el !== undefined);
    (target ?? uploadButtonRef.current)?.focus();
  }, [nodes]);

  // After a change on the server: the list, then the palette.
  const refreshAfterChange = useCallback(async () => {
    await fetchNodes();
    try {
      await reload();
    } catch (e) {
      showError((e as Error).message);
    }
  }, [fetchNodes, reload, showError]);

  const handleToggle = useCallback(async (filename: string, key: string) => {
    if (busyRows.current.has(key)) return;
    busyRows.current.set(key, Infinity);
    try {
      await toggleCustomNode(filename);
    } catch (e) {
      busyRows.current.delete(key);
      showError((e as Error).message);
      return;
    }
    busyRows.current.set(key, readSeq.current);
    focusAfterRead.current = { keys: [key], control: 'toggle' };
    await refreshAfterChange();
  }, [refreshAfterChange, showError]);

  // `neighbours` are the next row's key and the previous row's, as listed
  // when Delete was pressed.
  const handleDelete = useCallback(async (filename: string, key: string, neighbours: string[]) => {
    // Held from the press, not from the answer: the confirm is not a focus
    // trap, so the row's toggle is still within Tab's reach while it is open.
    if (busyRows.current.has(key)) return;
    busyRows.current.set(key, Infinity);
    const ok = await confirm({
      title: t('customNodes.delete.confirm', { name: filename }),
      confirmText: t('customNodes.delete'),
      variant: 'danger',
    });
    if (!ok) {
      busyRows.current.delete(key);
      return;
    }
    try {
      await deleteCustomNode(filename);
    } catch (e) {
      busyRows.current.delete(key);
      showError((e as Error).message);
      return;
    }
    busyRows.current.set(key, readSeq.current);
    focusAfterRead.current = { keys: neighbours, control: 'delete' };
    await refreshAfterChange();
  }, [refreshAfterChange, showError, t]);

  const handleUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      await uploadCustomNode(file);
      await fetchNodes();
      await reload();
    } catch (e) {
      showError((e as Error).message);
    }
    event.target.value = '';
  }, [fetchNodes, reload, showError]);

  const keys = rowKeys(nodes);

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

          {!loading && nodes.map((node, index) => {
            const key = keys[index];
            const neighbours = [keys[index + 1], keys[index - 1]].filter(
              (k): k is string => k !== undefined,
            );
            return (
              <div key={key} className={styles.nodeRow}>
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
                    ref={(el) => {
                      if (el) toggleButtons.current.set(key, el);
                      else toggleButtons.current.delete(key);
                    }}
                    className={`${styles.toggleButton} ${node.enabled ? styles.toggleOn : styles.toggleOff}`}
                    onClick={() => handleToggle(node.filename, key)}
                  >
                    {node.enabled ? t('customNodes.enabled') : t('customNodes.disabled')}
                  </button>
                  <button type="button"
                    ref={(el) => {
                      if (el) deleteButtons.current.set(key, el);
                      else deleteButtons.current.delete(key);
                    }}
                    className={styles.deleteButton}
                    onClick={() => handleDelete(node.filename, key, neighbours)}
                  >
                    {t('customNodes.delete')}
                  </button>
                </div>
              </div>
            );
          })}
        </div>

        <div className={styles.footer}>
          <button type="button"
            ref={uploadButtonRef}
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
