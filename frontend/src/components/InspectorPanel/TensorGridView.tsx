import { useState, useMemo } from 'react';
import type { TensorOutput } from '../../types';
import styles from './InspectorPanel.module.css';

interface Props {
  tensor: TensorOutput;
  highlight?: (i: number, j: number) => number;
  label?: string;
}

/**
 * Drills into a multi-dim list to a 2D (or 1D) slice using leading-dim indices.
 * Values come pre-sliced from the backend if the caller supplied a slice, in
 * which case `leadingIdx` should be empty.
 */
function drillTo2D(values: unknown, leadingIdx: number[]): number[][] | number[] | number {
  let cur: unknown = values;
  for (const i of leadingIdx) {
    if (!Array.isArray(cur)) break;
    cur = cur[i];
  }
  if (!Array.isArray(cur)) return cur as number;
  if (cur.length === 0) return [];
  if (!Array.isArray(cur[0])) return cur as number[];
  return cur as number[][];
}

function fmt(v: unknown): string {
  if (v === null || v === undefined) return '·';
  if (typeof v === 'number') {
    if (Number.isInteger(v)) return String(v);
    if (Math.abs(v) < 1e-3 && v !== 0) return v.toExponential(2);
    return v.toFixed(4);
  }
  if (typeof v === 'boolean') return v ? 'T' : 'F';
  return String(v);
}

export function TensorGridView({ tensor, highlight, label }: Props) {
  const rank = tensor.sliced_shape.length;
  const leadingCount = Math.max(0, rank - 2);
  const [leading, setLeading] = useState<number[]>(() => Array(leadingCount).fill(0));

  const grid = useMemo(
    () => drillTo2D(tensor.values, leading),
    [tensor.values, leading],
  );

  const updateLeading = (dim: number, val: number) => {
    const copy = [...leading];
    copy[dim] = val;
    setLeading(copy);
  };

  const is2D = Array.isArray(grid) && grid.length > 0 && Array.isArray((grid as any[])[0]);
  const is1D = Array.isArray(grid) && !is2D;
  const isScalar = !Array.isArray(grid);

  const headerStats: string[] = [];
  if (tensor.min !== undefined) headerStats.push(`min ${fmt(tensor.min)}`);
  if (tensor.max !== undefined) headerStats.push(`max ${fmt(tensor.max)}`);
  if (tensor.mean !== undefined) headerStats.push(`mean ${fmt(tensor.mean)}`);

  return (
    <div className={styles.tensorView}>
      {label && <div className={styles.tensorLabel}>{label}</div>}
      <div className={styles.tensorMeta}>
        <span className={styles.tensorShape}>
          shape [{tensor.full_shape.join(', ')}]
        </span>
        <span className={styles.tensorDtype}>{tensor.dtype}</span>
        {headerStats.length > 0 && (
          <span className={styles.tensorStats}>{headerStats.join(' · ')}</span>
        )}
      </div>

      {leadingCount > 0 && (
        <div className={styles.tensorLeadingRow}>
          {leading.map((val, dim) => {
            const dimSize = tensor.sliced_shape[dim] ?? 1;
            return (
              <label key={dim} className={styles.tensorLeadingLabel}>
                dim {dim}
                <select
                  className={styles.tensorLeadingSelect}
                  value={val}
                  onChange={(e) => updateLeading(dim, Number(e.target.value))}
                >
                  {Array.from({ length: dimSize }, (_, i) => (
                    <option key={i} value={i}>
                      {i}
                    </option>
                  ))}
                </select>
              </label>
            );
          })}
        </div>
      )}

      <div className={styles.tensorGridScroll}>
        {isScalar && (
          <div className={styles.tensorScalar}>{fmt(grid)}</div>
        )}
        {is1D && (
          <table className={styles.tensorTable}>
            <tbody>
              <tr>
                {(grid as number[]).map((v, j) => (
                  <td
                    key={j}
                    className={styles.tensorCell}
                    style={
                      highlight ? { background: heatColor(highlight(0, j)) } : undefined
                    }
                  >
                    {fmt(v)}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        )}
        {is2D && (
          <table className={styles.tensorTable}>
            <tbody>
              {(grid as number[][]).map((row, i) => (
                <tr key={i}>
                  {row.map((v, j) => (
                    <td
                      key={j}
                      className={styles.tensorCell}
                      style={
                        highlight ? { background: heatColor(highlight(i, j)) } : undefined
                      }
                    >
                      {fmt(v)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function heatColor(intensity: number): string | undefined {
  if (intensity <= 0) return undefined;
  const alpha = Math.min(0.75, intensity);
  return `rgba(6, 182, 212, ${alpha})`;
}
