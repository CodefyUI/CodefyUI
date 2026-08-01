import { useEffect, useRef } from 'react';
import type { OutputSummary } from '../../types';
import { useI18n } from '../../i18n';
import styles from '../ResultsPanel/ResultsPanel.module.css';

interface EdgeDataTooltipProps {
  x: number;
  y: number;
  sourceLabel: string;
  targetLabel: string;
  portName: string;
  summary: OutputSummary;
  onClose: () => void;
  /**
   * Open the full statistics for the value on this edge (#129). Omitted when
   * nothing has run, since there would be no captured value to summarise.
   */
  onViewStats?: () => void;
}

function formatSummary(summary: OutputSummary) {
  const rows: { label: string; value: string }[] = [];
  rows.push({ label: 'Type', value: summary.type });

  if (summary.shape) {
    rows.push({ label: 'Shape', value: `[${summary.shape.join(', ')}]` });
  }
  if (summary.dtype) {
    rows.push({ label: 'Dtype', value: summary.dtype });
  }
  if (summary.min !== undefined) {
    rows.push({ label: 'Min', value: String(summary.min) });
  }
  if (summary.max !== undefined) {
    rows.push({ label: 'Max', value: String(summary.max) });
  }
  if (summary.mean !== undefined) {
    rows.push({ label: 'Mean', value: String(summary.mean) });
  }
  if (summary.class) {
    rows.push({ label: 'Class', value: summary.class });
  }
  if (summary.params !== undefined) {
    rows.push({ label: 'Params', value: summary.params.toLocaleString() });
  }
  if (summary.trainable !== undefined) {
    rows.push({ label: 'Trainable', value: summary.trainable.toLocaleString() });
  }
  if (summary.value !== undefined) {
    rows.push({ label: 'Value', value: String(summary.value) });
  }
  if (summary.repr) {
    rows.push({ label: 'Value', value: summary.repr });
  }

  return rows;
}

export function EdgeDataTooltip({
  x,
  y,
  sourceLabel,
  targetLabel,
  portName,
  summary,
  onClose,
  onViewStats,
}: EdgeDataTooltipProps) {
  const ref = useRef<HTMLDivElement>(null);
  const { t } = useI18n();

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        onClose();
      }
    };
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('mousedown', handleClick);
    document.addEventListener('keydown', handleEsc);
    return () => {
      document.removeEventListener('mousedown', handleClick);
      document.removeEventListener('keydown', handleEsc);
    };
  }, [onClose]);

  const rows = formatSummary(summary);

  return (
    <div
      ref={ref}
      className={styles.edgeTooltip}
      style={{ left: x, top: y }}
    >
      <div className={styles.edgeTooltipTitle}>
        {sourceLabel} &rarr; {targetLabel} ({portName})
      </div>
      {rows.map((row, i) => (
        <div key={i} className={styles.edgeTooltipRow}>
          <span className={styles.edgeTooltipLabel}>{row.label}</span>
          <span className={styles.edgeTooltipValue}>{row.value}</span>
        </div>
      ))}
      {onViewStats && (
        // The summary above is whatever the run streamed — a shape, a dtype,
        // maybe a min/max. This is the way to the real distribution, computed
        // on the server, so the ANSWER is a kilobyte whatever the tensor
        // weighs (the computing still costs the server real time).
        <button type="button" className={styles.edgeTooltipLink} onClick={onViewStats}>
          {t('edge.viewStats')} &rarr;
        </button>
      )}
    </div>
  );
}
