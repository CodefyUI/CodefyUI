import { useMemo, useState } from 'react';
import type { ParamDefinition } from '../../types';
import styles from './TensorGridEditor.module.css';

interface Props {
  param: ParamDefinition;
  value: any;
  onChange: (name: string, value: any) => void;
  displayLabel: string;
  siblingParams?: Record<string, any>;
}

const MAX_INLINE_NUMEL = 512;

function parseShape(shapeStr: string | undefined): number[] {
  if (!shapeStr) return [];
  return shapeStr
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => Math.max(1, parseInt(s, 10) || 1));
}

function numel(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

function zerosOf(shape: number[]): any {
  if (shape.length === 0) return 0;
  if (shape.length === 1) return Array.from({ length: shape[0] }, () => 0);
  return Array.from({ length: shape[0] }, () => zerosOf(shape.slice(1)));
}

function fillFlat(shape: number[], flat: number[], offset = 0): [any, number] {
  if (shape.length === 0) return [flat[offset] ?? 0, offset + 1];
  if (shape.length === 1) {
    const row = Array.from({ length: shape[0] }, (_, i) => flat[offset + i] ?? 0);
    return [row, offset + shape[0]];
  }
  const out: any[] = [];
  let cur = offset;
  for (let i = 0; i < shape[0]; i++) {
    const [sub, next] = fillFlat(shape.slice(1), flat, cur);
    out.push(sub);
    cur = next;
  }
  return [out, cur];
}

function reshapeValues(value: any, shape: number[]): any {
  const flat: number[] = [];
  const walk = (v: any) => {
    if (Array.isArray(v)) v.forEach(walk);
    else if (typeof v === 'number') flat.push(v);
    else flat.push(Number(v) || 0);
  };
  walk(value);
  while (flat.length < numel(shape)) flat.push(0);
  const [reshaped] = fillFlat(shape, flat);
  return reshaped;
}

function drill2D(v: any, leadingIdx: number[]): number[][] {
  let cur: any = v;
  for (const i of leadingIdx) {
    if (!Array.isArray(cur)) return [];
    cur = cur[i];
  }
  if (!Array.isArray(cur)) return [[cur as number]];
  if (!Array.isArray(cur[0])) return [cur as number[]];
  return cur as number[][];
}

function set2D(
  root: any,
  leadingIdx: number[],
  i: number,
  j: number,
  newVal: number,
): any {
  const clone = JSON.parse(JSON.stringify(root));
  let cur: any = clone;
  for (const idx of leadingIdx) {
    cur = cur[idx];
  }
  if (!Array.isArray(cur)) return clone;
  if (!Array.isArray(cur[0])) {
    // 1D at leaf
    cur[j] = newVal;
  } else {
    cur[i][j] = newVal;
  }
  return clone;
}

export function TensorGridEditor({ param, value, onChange, displayLabel, siblingParams }: Props) {
  const shape = useMemo(() => parseShape(siblingParams?.shape), [siblingParams?.shape]);
  const total = numel(shape);
  const rank = shape.length;
  const leadingCount = Math.max(0, rank - 2);
  const [leading, setLeading] = useState<number[]>(() => Array(leadingCount).fill(0));

  // If leading count changed, reset
  if (leading.length !== leadingCount) {
    setLeading(Array(leadingCount).fill(0));
  }

  const valueMode = siblingParams?.value_mode ?? 'random';
  const disabled = valueMode !== 'explicit';

  // Reshape existing value to current shape — but only if mode is explicit
  const normalized = useMemo(() => {
    if (valueMode !== 'explicit') return null;
    if (shape.length === 0 || total === 0) return null;
    if (total > MAX_INLINE_NUMEL) return null;
    return reshapeValues(value ?? zerosOf(shape), shape);
  }, [value, shape, valueMode, total]);

  const grid = useMemo(
    () => (normalized ? drill2D(normalized, leading) : []),
    [normalized, leading],
  );

  const setCell = (i: number, j: number, raw: string) => {
    if (!normalized) return;
    const n = Number(raw);
    const newVal = Number.isFinite(n) ? n : 0;
    const next = set2D(normalized, leading, i, j, newVal);
    onChange(param.name, next);
  };

  const fillAll = (v: number | 'random') => {
    if (disabled) return;
    const flat = Array.from({ length: total }, () =>
      v === 'random' ? Math.round(Math.random() * 200 - 100) / 100 : v,
    );
    const [shaped] = fillFlat(shape, flat);
    onChange(param.name, shaped);
  };

  return (
    <div className={styles.wrapper}>
      <label className={styles.label}>{displayLabel}</label>

      {disabled && (
        <div className={styles.hintDim}>
          Set <code>value_mode</code> to <code>explicit</code> to edit values inline.
        </div>
      )}

      {!disabled && total > MAX_INLINE_NUMEL && (
        <div className={styles.hintWarn}>
          Shape has {total} elements — too large for inline editing (max {MAX_INLINE_NUMEL}).
          Switch <code>value_mode</code> to <code>random</code>, <code>zeros</code>, <code>ones</code>, or <code>arange</code>.
        </div>
      )}

      {!disabled && total > 0 && total <= MAX_INLINE_NUMEL && (
        <>
          <div className={styles.toolbar}>
            <button
              type="button"
              className={styles.toolbarBtn}
              onClick={() => fillAll(0)}
            >
              Fill 0
            </button>
            <button
              type="button"
              className={styles.toolbarBtn}
              onClick={() => fillAll(1)}
            >
              Fill 1
            </button>
            <button
              type="button"
              className={styles.toolbarBtn}
              onClick={() => fillAll('random')}
            >
              Random
            </button>
            <span className={styles.shapeBadge}>
              [{shape.join(', ')}] · {total} cells
            </span>
          </div>

          {leadingCount > 0 && (
            <div className={styles.leadingRow}>
              {leading.map((val, dim) => {
                const dimSize = shape[dim] ?? 1;
                return (
                  <label key={dim} className={styles.leadingLabel}>
                    dim {dim}
                    <select
                      className={styles.leadingSelect}
                      value={val}
                      onChange={(e) => {
                        const copy = [...leading];
                        copy[dim] = Number(e.target.value);
                        setLeading(copy);
                      }}
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

          <div className={styles.gridScroll}>
            <table className={styles.table}>
              <tbody>
                {grid.map((row, i) => (
                  <tr key={i}>
                    {row.map((v, j) => (
                      <td key={j} className={styles.cell}>
                        <input
                          type="number"
                          className={styles.cellInput}
                          value={String(v ?? 0)}
                          step="any"
                          onChange={(e) => setCell(i, j, e.target.value)}
                        />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
