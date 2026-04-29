import { memo, useCallback, useRef, useState } from 'react';
import { type NodeProps } from '@xyflow/react';
import type { NodeData } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import styles from './NoteNode.module.css';

const MAX_IMAGE_DIM = 800;

function resizeImage(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        let { width, height } = img;
        if (width > MAX_IMAGE_DIM || height > MAX_IMAGE_DIM) {
          const ratio = Math.min(MAX_IMAGE_DIM / width, MAX_IMAGE_DIM / height);
          width = Math.round(width * ratio);
          height = Math.round(height * ratio);
        }
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        canvas.getContext('2d')!.drawImage(img, 0, 0, width, height);
        resolve(canvas.toDataURL('image/png'));
      };
      img.onerror = reject;
      img.src = reader.result as string;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function NoteNodeInner({ id, data, selected }: NodeProps & { data: NodeData }) {
  const { t } = useI18n();
  const updateNoteData = useTabStore((s) => s.updateNoteData);
  const [editing, setEditing] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const noteColor = data.noteColor ?? '#3d3d1a';
  const noteKind = data.noteKind ?? 'text';
  const isBound = !!data.boundToNodeId;
  const noteWidth = data.noteWidth ?? 200;

  // ── Text editing ──
  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    if (noteKind === 'text') {
      e.stopPropagation();
      setEditing(true);
      // Focus the content div after React re-renders
      requestAnimationFrame(() => {
        contentRef.current?.focus();
        // Move cursor to end
        const sel = window.getSelection();
        if (sel && contentRef.current) {
          sel.selectAllChildren(contentRef.current);
          sel.collapseToEnd();
        }
      });
    } else {
      // Image: open file picker
      fileInputRef.current?.click();
    }
  }, [noteKind]);

  const handleBlur = useCallback(() => {
    setEditing(false);
    const text = contentRef.current?.innerText ?? '';
    updateNoteData(id, { noteContent: text });
  }, [id, updateNoteData]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (editing) {
      // Stop propagation to prevent XYFlow from intercepting Delete/Backspace
      e.stopPropagation();
      if (e.key === 'Escape') {
        setEditing(false);
        contentRef.current?.blur();
      }
    }
  }, [editing]);

  // ── Image upload ──
  const handleFileChange = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const dataUrl = await resizeImage(file);
      updateNoteData(id, { noteContent: dataUrl });
    } catch {
      // silently ignore
    }
    // Reset input so the same file can be re-selected
    e.target.value = '';
  }, [id, updateNoteData]);

  return (
    <div
      className={`${styles.note} ${editing ? styles.editing : ''}`}
      style={{
        width: noteWidth,
        borderColor: selected ? '#888' : undefined,
        boxShadow: selected ? '0 0 0 1px #888' : undefined,
      }}
      onDoubleClick={handleDoubleClick}
    >
      {/* Accent bar */}
      <div className={styles.accentBar} style={{ background: noteColor }} />

      {/* Bind indicator */}
      {isBound && <div className={styles.bindIcon} title={t('note.boundToNode')}>&#128279;</div>}

      {/* Text note */}
      {noteKind === 'text' && (
        <div
          ref={contentRef}
          className={styles.textContent}
          contentEditable={editing}
          suppressContentEditableWarning
          onBlur={handleBlur}
          onKeyDown={handleKeyDown}
          data-placeholder={t('note.placeholder')}
        >
          {data.noteContent || ''}
        </div>
      )}

      {/* Image note */}
      {noteKind === 'image' && (
        <div className={styles.imageContent}>
          {data.noteContent ? (
            <img
              src={data.noteContent}
              alt="Note"
              style={{
                maxHeight: data.noteHeight ?? 200,
              }}
              draggable={false}
            />
          ) : (
            <div className={styles.imagePlaceholder} onClick={() => fileInputRef.current?.click()}>
              <span className={styles.imagePlaceholderIcon}>&#128247;</span>
              <span>{t('note.imagePlaceholder')}</span>
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={handleFileChange}
          />
        </div>
      )}
    </div>
  );
}

export default memo(NoteNodeInner);
