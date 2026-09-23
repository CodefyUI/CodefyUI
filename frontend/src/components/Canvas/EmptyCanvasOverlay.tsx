import { useCallback } from 'react';
import { useI18n } from '../../i18n';
import { openExample } from '../../utils/openExample';
import { useUIStore } from '../../store/uiStore';
import type { ExampleSummary } from '../../api/rest';
import { ExampleBrowser } from '../shared/ExampleBrowser';
import styles from './EmptyCanvasOverlay.module.css';

interface EmptyCanvasOverlayProps {
  /**
   * The canvas's own drop handlers (`useDragAndDrop`), which FlowCanvas passes
   * in (#526). This gallery covers most of an empty canvas and is drawn beside
   * `<ReactFlow>`, not inside it, so a drag over it never reached the canvas's
   * handlers: nothing cancelled the dragover, and the browser refused a node,
   * preset or example dropped on it. They sit on the root, which lets the
   * pointer through to the canvas but still sees every drag over the panel
   * bubble up. `onDragOver` answers the dragenter too: on the frame the
   * pointer crosses into a new element Chrome fires only a dragenter, and an
   * uncancelled one makes that frame's drop effect "none". A drag over this
   * grid of cards crosses one element after another.
   */
  onDragOver: (event: React.DragEvent) => void;
  onDrop: (event: React.DragEvent) => void;
}

export function EmptyCanvasOverlay({ onDragOver, onDrop }: EmptyCanvasOverlayProps) {
  const { t } = useI18n();

  // Still the REPLACING reader, and the only one left (#348): this overlay
  // is shown when the canvas is empty, so there is nothing for a replace to
  // take. The sidebar's Templates tab and the gallery modal both insert
  // instead, because they are reachable while a graph is on screen. The
  // welcome screen is the other non-replacing reader -- it has no tab at all,
  // so it opens into a new one.
  const handleClick = useCallback(
    (example: ExampleSummary) => void openExample(example.path),
    [],
  );

  return (
    <div
      className={styles.overlay}
      onDragEnter={onDragOver}
      onDragOver={onDragOver}
      onDrop={onDrop}
    >
      <div className={styles.inner}>
        <div className={styles.title}>{t('empty.title')}</div>
        <div className={styles.subtitle}>{t('empty.subtitle')}</div>

        {/* The sections below stay the fast path; this is the way to the
            full, searchable list — the same modal the toolbar and the
            sidebar's Templates tab open (core#128). */}
        <button
          type="button"
          className={styles.browseButton}
          onClick={() => useUIStore.getState().openTemplateGallery()}
          title={t('gallery.open.title')}
        >
          {t('gallery.browse')}
        </button>

        <ExampleBrowser onPick={handleClick} />

        {/* No trailing "or drag a node from the left palette": the palette it
            points at is open on its default tab with its own pinned footer
            reading "Drag nodes onto the canvas", and that footer is the copy
            that has to stay -- a node item is drag-only, and the footer is
            still there after the canvas stops being empty. */}
      </div>
    </div>
  );
}
