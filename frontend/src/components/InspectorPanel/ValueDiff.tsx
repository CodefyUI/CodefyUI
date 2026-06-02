import { useI18n } from '../../i18n';
import type { OutputData, TensorOutput } from '../../types';
import { TensorGridView } from './TensorGridView';
import styles from './InspectorPanel.module.css';

interface Props {
  input: OutputData | null;
  output: OutputData | null;
  inputLabel?: string;
  outputLabel?: string;
}

function isTensor(v: OutputData | null): v is TensorOutput {
  return v !== null && v.type === 'tensor';
}

function shapesEqual(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function makeHighlight(
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

export function ValueDiff({ input, output, inputLabel = 'Input', outputLabel = 'Output' }: Props) {
  const { t } = useI18n();
  if (input === null && output === null) {
    return <div className={styles.diffEmpty}>{t('inspector.valueDiff.noValues')}</div>;
  }

  let highlight: ((i: number, j: number) => number) | undefined;
  let shapeChanged = false;
  if (isTensor(input) && isTensor(output)) {
    if (shapesEqual(input.full_shape, output.full_shape)) {
      highlight = makeHighlight(input, output);
    } else {
      shapeChanged = true;
    }
  }

  return (
    <div className={styles.diffRow}>
      {shapeChanged && isTensor(input) && isTensor(output) && (
        <div className={styles.shapeChangeBanner}>
          [{input.full_shape.join(', ')}] &rarr; [{output.full_shape.join(', ')}]
        </div>
      )}
      <div className={styles.diffPair}>
        <div className={styles.diffCol}>
          {input ? (
            isTensor(input) ? (
              <TensorGridView tensor={input} label={inputLabel} />
            ) : (
              <NonTensorView value={input} label={inputLabel} />
            )
          ) : (
            <div className={styles.diffMissing}>{inputLabel}: —</div>
          )}
        </div>
        <div className={styles.diffArrow}>&darr;</div>
        <div className={styles.diffCol}>
          {output ? (
            isTensor(output) ? (
              <TensorGridView tensor={output} label={outputLabel} highlight={highlight} />
            ) : (
              <NonTensorView value={output} label={outputLabel} />
            )
          ) : (
            <div className={styles.diffMissing}>{outputLabel}: —</div>
          )}
        </div>
      </div>
    </div>
  );
}

function NonTensorView({ value, label }: { value: OutputData; label: string }) {
  return (
    <div className={styles.tensorView}>
      <div className={styles.tensorLabel}>{label}</div>
      <div className={styles.tensorMeta}>
        <span className={styles.tensorDtype}>{value.type}</span>
      </div>
      <div className={styles.tensorScalar}>
        {value.type === 'scalar' && String((value as any).value)}
        {value.type === 'string' && (value as any).value}
        {value.type === 'model' && (
          <div>
            {(value as any).class} · params {(value as any).params?.toLocaleString?.() ?? ''}
          </div>
        )}
        {!(['scalar', 'string', 'model'].includes(value.type)) && ((value as any).repr ?? value.type)}
      </div>
    </div>
  );
}
