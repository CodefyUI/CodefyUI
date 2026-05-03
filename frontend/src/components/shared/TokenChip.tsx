import { useState, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { getTokenColor } from '../../styles/theme';
import styles from './TokenChip.module.css';

interface TokenChipProps {
  /** Display string for the token (BPE merges, WordPiece subwords, etc.). */
  token: string;
  /** Integer token id from the tokenizer's vocabulary. */
  id?: number;
  /** Position in the token sequence — drives color cycling. */
  index: number;
  /** Optional [start, end] character offsets; shown in the hover tooltip. */
  offset?: [number, number];
  /**
   * If true, fades in with a stagger delay proportional to `index`. Used
   * inside the tokenizer viz node so chips appear in sequence on each run.
   * Inspector renders set this to false (no animation).
   */
  animated?: boolean;
}

/**
 * Replace whitespace with visible glyphs so token boundaries don't visually
 * disappear. Newlines/tabs are rare in casual text but must be unambiguous.
 */
function displayText(token: string): string {
  return token
    .replace(/\n/g, '↵')
    .replace(/\t/g, '→')
    .replace(/ /g, '·');
}

function bytesPreview(token: string, max = 24): string {
  const enc = new TextEncoder();
  const bytes = Array.from(enc.encode(token));
  const trimmed = bytes.slice(0, max);
  const hex = trimmed.map((b) => b.toString(16).padStart(2, '0')).join(' ');
  return bytes.length > max ? `${hex} … (${bytes.length} bytes)` : hex;
}

export function TokenChip({ token, id, index, offset, animated = false }: TokenChipProps) {
  const [hovered, setHovered] = useState(false);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);
  const ref = useRef<HTMLSpanElement>(null);
  const color = getTokenColor(index);

  const handleEnter = useCallback(() => {
    setHovered(true);
    if (ref.current) {
      const rect = ref.current.getBoundingClientRect();
      setTooltipPos({ x: rect.left + rect.width / 2, y: rect.top - 8 });
    }
  }, []);

  const handleLeave = useCallback(() => {
    setHovered(false);
    setTooltipPos(null);
  }, []);

  const style: React.CSSProperties = {
    backgroundColor: `${color}26`, // ~15% alpha
    borderColor: `${color}66`,
    color,
    ...(animated ? { animationDelay: `${index * 30}ms` } : {}),
  };

  return (
    <>
      <span
        ref={ref}
        className={`${styles.chip} ${animated ? styles.animated : ''}`}
        style={style}
        onMouseEnter={handleEnter}
        onMouseLeave={handleLeave}
      >
        {displayText(token) || '·'}
      </span>
      {hovered && tooltipPos &&
        createPortal(
          <div
            className={styles.tooltip}
            style={{
              left: tooltipPos.x,
              top: tooltipPos.y,
              transform: 'translate(-50%, -100%)',
            }}
          >
            <div className={styles.tooltipRow}>
              <span className={styles.tooltipLabel}>token</span>
              <span className={styles.tooltipValue}>{JSON.stringify(token)}</span>
            </div>
            {id !== undefined && (
              <div className={styles.tooltipRow}>
                <span className={styles.tooltipLabel}>id</span>
                <span className={styles.tooltipValue}>{id}</span>
              </div>
            )}
            {offset && (
              <div className={styles.tooltipRow}>
                <span className={styles.tooltipLabel}>offset</span>
                <span className={styles.tooltipValue}>
                  [{offset[0]}, {offset[1]})
                </span>
              </div>
            )}
            <div className={styles.tooltipRow}>
              <span className={styles.tooltipLabel}>bytes</span>
              <span className={styles.tooltipValueMono}>{bytesPreview(token)}</span>
            </div>
          </div>,
          document.body,
        )}
    </>
  );
}
