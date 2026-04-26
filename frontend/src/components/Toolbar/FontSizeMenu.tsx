import { useEffect, useRef } from 'react';
import { useUIStore, type FontSize } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import styles from './FontSizeMenu.module.css';

interface Props {
  open: boolean;
  onClose: () => void;
  triggerRef: React.RefObject<HTMLButtonElement | null>;
}

const SIZES: { id: FontSize; sample: number }[] = [
  { id: 'small', sample: 11 },
  { id: 'default', sample: 13 },
  { id: 'large', sample: 16 },
];

export function FontSizeMenu({ open, onClose, triggerRef }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const fontSize = useUIStore((s) => s.fontSize);
  const setFontSize = useUIStore((s) => s.setFontSize);
  const { t } = useI18n();

  useEffect(() => {
    if (!open) return;
    const handleMouseDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (ref.current?.contains(target)) return;
      if (triggerRef.current?.contains(target)) return;
      onClose();
    };
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('mousedown', handleMouseDown);
    document.addEventListener('keydown', handleKey);
    return () => {
      document.removeEventListener('mousedown', handleMouseDown);
      document.removeEventListener('keydown', handleKey);
    };
  }, [open, onClose, triggerRef]);

  if (!open) return null;

  return (
    <div ref={ref} className={styles.panel} role="menu">
      <div className={styles.label}>{t('toolbar.fontSize.title')}</div>
      {SIZES.map((s) => (
        <button
          key={s.id}
          type="button"
          role="menuitemradio"
          aria-checked={fontSize === s.id}
          className={`${styles.item} ${fontSize === s.id ? styles.active : ''}`}
          onClick={() => {
            setFontSize(s.id);
            onClose();
          }}
        >
          <span>{t(`toolbar.fontSize.${s.id}` as const)}</span>
          <span className={styles.sample} style={{ fontSize: `${s.sample}px` }}>
            Aa
          </span>
        </button>
      ))}
    </div>
  );
}
