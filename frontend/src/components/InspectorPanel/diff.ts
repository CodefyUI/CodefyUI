import type { OutputData, TensorOutput } from '../../types';

export function isTensor(v: OutputData | null): v is TensorOutput {
  return v !== null && v.type === 'tensor';
}

export function shapesEqual(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

export function makeHighlight(
  inT: TensorOutput,
  outT: TensorOutput,
): ((i: number, j: number) => number) | undefined {
  // Callers only build this highlight after confirming shapes are equal
  /* v8 ignore start */
  if (!shapesEqual(inT.full_shape, outT.full_shape)) return undefined;
  /* v8 ignore stop */
  // Only highlight when both are 2D after drilling — we don't know the drill
  // state here, so supply a relative diff using top-level values.
  const inVals = inT.values;
  const outVals = outT.values;
  if (!Array.isArray(inVals) || !Array.isArray(outVals)) return undefined;
  // Normalize to last-2-dims view
  return (i: number, j: number) => {
    const getCell = (arr: any, ii: number, jj: number): number | undefined => {
      let cur: any = arr;
      while (Array.isArray(cur) && Array.isArray(cur[0]) && Array.isArray(cur[0][0])) {
        cur = cur[0];
      }
      // cur is seeded from values, already guarded as an array before this closure
      /* v8 ignore start */
      if (!Array.isArray(cur)) return undefined;
      /* v8 ignore stop */
      if (!Array.isArray(cur[0])) return cur[jj];
      return cur[ii]?.[jj];
    };
    const a = getCell(inVals, i, j);
    const b = getCell(outVals, i, j);
    if (typeof a !== 'number' || typeof b !== 'number') return 0;
    const diff = Math.abs(a - b);
    const scale = Math.max(Math.abs(a), Math.abs(b), 1e-6);
    return Math.min(1, diff / scale);
  };
}
