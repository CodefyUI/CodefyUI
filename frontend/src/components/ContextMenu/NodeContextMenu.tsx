import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import styles from './NodeContextMenu.module.css';

export interface ContextMenuPosition {
  nodeId: string;
  x: number;
  y: number;
}

interface MenuItem {
  label: string;
  action: () => void;
  color?: string;
  dividerAfter?: boolean;
}

interface NodeContextMenuProps {
  position: ContextMenuPosition;
  items: MenuItem[];
  onClose: () => void;
}

const NOTE_COLORS = [
  { label: 'Yellow', value: '#3d3d1a' },
  { label: 'Blue', value: '#1a2d3d' },
  { label: 'Green', value: '#1a3d1a' },
  { label: 'Red', value: '#3d1a1a' },
  { label: 'Purple', value: '#2d1a3d' },
  { label: 'Gray', value: '#2a2a2a' },
];

export function NodeContextMenu({ position, items, onClose }: NodeContextMenuProps) {
  return (
    <>
      {/* Backdrop */}
      <div
        className={styles.backdrop}
        onClick={onClose}
        onContextMenu={(e) => { e.preventDefault(); onClose(); }}
      />
      {/* Menu — left/top are dynamic from position */}
      <div
        className={styles.menu}
        style={{ left: position.x, top: position.y }}
      >
        {items.map((item, i) => (
          <div key={i}>
            <button type="button"
              onClick={() => { item.action(); onClose(); }}
              className={styles.menuItem}
              style={{ color: item.color ?? '#ccc' }}
            >
              {item.label}
            </button>
            {item.dividerAfter && <div className={styles.divider} />}
          </div>
        ))}
      </div>
    </>
  );
}

export function useNodeContextMenuItems(
  nodeId: string,
  callbacks: {
    onDelete: (id: string) => void;
    onRename: (id: string) => void;
    onDuplicate: (id: string) => void;
  },
) {
  const { t } = useI18n();

  return [
    { label: t('contextMenu.rename'), action: () => callbacks.onRename(nodeId), dividerAfter: false },
    { label: t('contextMenu.duplicate'), action: () => callbacks.onDuplicate(nodeId), dividerAfter: true },
    { label: t('contextMenu.delete'), action: () => callbacks.onDelete(nodeId), color: '#F44336' },
  ];
}

export function useNoteContextMenuItems(
  nodeId: string,
  callbacks: {
    onDelete: (id: string) => void;
  },
) {
  const { t } = useI18n();
  const bindNoteToNearest = useTabStore((s) => s.bindNoteToNearest);
  const unbindNote = useTabStore((s) => s.unbindNote);
  const updateNoteData = useTabStore((s) => s.updateNoteData);

  const tab = useTabStore((s) => s.tabs.find((tab) => tab.id === s.activeTabId));
  const note = tab?.nodes.find((n) => n.id === nodeId);
  const isBound = !!note?.data.boundToNodeId;

  const colorItems: MenuItem[] = NOTE_COLORS.map((c) => ({
    label: `  ${c.label}`,
    action: () => updateNoteData(nodeId, { noteColor: c.value }),
    color: c.value === note?.data.noteColor ? '#fff' : '#999',
  }));

  return [
    // Bind / Unbind
    isBound
      ? { label: t('note.unbind'), action: () => unbindNote(nodeId), dividerAfter: false }
      : { label: t('note.bind'), action: () => bindNoteToNearest(nodeId), dividerAfter: false },
    // Color submenu (inline)
    { label: t('note.changeColor'), action: () => {}, color: '#888', dividerAfter: false },
    ...colorItems,
    { label: '', action: () => {}, dividerAfter: true },
    // Delete
    { label: t('contextMenu.delete'), action: () => callbacks.onDelete(nodeId), color: '#F44336' },
  ];
}
