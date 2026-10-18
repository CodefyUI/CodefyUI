import type { MemoryEstimate, StaticMemoryEstimate } from '../../api/rest';
import { useI18n } from '../../i18n';
import { formatBytes } from '../../utils/formatBytes';
import { formatGiB } from '../../utils/formatGiB';
import styles from './QueueRunDialog.module.css';

interface MemoryEstimateHintProps {
  estimate: MemoryEstimate | 'loading' | null;
  error: string | null;
}

/**
 * Where the Queue on... dialog's memory suggestion came from (#788, #794),
 * said so that a static estimate never reads as a measured peak: a
 * "Measured" or "Estimate" tag, then how the figure was reached. The server
 * applies only a measured one by default; an estimate goes with the run
 * only when it is submitted, marked as one. A rough (low confidence)
 * estimate is shown for reference with why it is not offered, and the field
 * stays empty.
 */
export function MemoryEstimateHint({ estimate, error }: MemoryEstimateHintProps) {
  const t = useI18n((s) => s.t);
  if (estimate === 'loading') {
    return <p className={styles.hint} aria-live="polite">{t('queueRun.memory.loading')}</p>;
  }
  if (error !== null) {
    return <p className={styles.hint} aria-live="polite">{t('queueRun.memory.estimateFailed', { error })}</p>;
  }
  if (estimate?.source === 'history' && estimate.peak_bytes != null) {
    return (
      <p className={styles.hint} aria-live="polite">
        <span className={`${styles.tag} ${styles.tagMeasured}`}>{t('queueRun.memory.tag.measured')}</span>{' '}
        <span>{t('queueRun.memory.suggested', { peak: formatGiB(estimate.peak_bytes) })}</span>
      </p>
    );
  }
  const statics = estimate?.static ?? null;
  if (estimate?.estimate_gib != null && statics !== null) {
    return (
      <p className={styles.hint} aria-live="polite">
        <span className={`${styles.tag} ${styles.tagEstimate}`}>{t('queueRun.memory.tag.estimate')}</span>{' '}
        <span>{t('queueRun.memory.estimated', { parts: listed(breakdown(statics, t), t) })}</span>
      </p>
    );
  }
  if (statics !== null) {
    return (
      <p className={styles.hint} aria-live="polite">
        <span className={`${styles.tag} ${styles.tagEstimate}`}>{t('queueRun.memory.tag.rough')}</span>{' '}
        <span>
          {t('queueRun.memory.rough', { size: formatGiB(statics.bytes), why: listed(roughReasons(statics, t), t) })}
        </span>
      </p>
    );
  }
  return <p className={styles.hint} aria-live="polite">{t('queueRun.memory.none')}</p>;
}

type Translate = ReturnType<typeof useI18n.getState>['t'];

/** Items joined as the locale lists things ("a, b, and c"; "a、b和c"), from
 * templates: a bare separator in the tables would be whitespace padding. */
function listed(items: string[], t: Translate): string {
  if (items.length < 2) return items.join('');
  const last = items[items.length - 1];
  const rest = items.slice(1, -1).reduce((joined, item) => t('queueRun.memory.list.next', { rest: joined, item }), items[0]);
  return t(items.length === 2 ? 'queueRun.memory.list.pair' : 'queueRun.memory.list.last', { rest, last });
}

/** Half-precision copies autocast makes of every weight, per parameter. */
const AUTOCAST_BYTES = 2;

/** How the estimate adds up, part by part, each the term the server added:
 * every parameter's float32 weight; for a graph that trains, the trainable
 * ones' gradients and optimizer state, and under bf16/fp16 autocast the
 * weights' half-precision copies; the activations of the pass that sets
 * them when there is one; the libraries' workspace; and the allocator's
 * reserve on top of the tensors, so the parts sum to the total. Each part in
 * the unit that reads best (a small model's weights are kilobytes); the
 * total stays in GiB, the unit the field takes. */
function breakdown(estimate: StaticMemoryEstimate, t: Translate): string[] {
  const count = (n: number) => n.toLocaleString('en-US');
  const params = count(estimate.parameters);
  const autocast = estimate.training && estimate.precision !== 'fp32';
  const copies = autocast ? estimate.parameters * AUTOCAST_BYTES : 0;
  const perTrainable = (estimate.multiplier - 1) * 4;
  const state = estimate.training ? estimate.trainable_parameters * perTrainable : 0;
  const parts = [
    t('queueRun.memory.part.weights', { params, weights: formatBytes(estimate.parameters * 4) }),
  ];
  if (state > 0) {
    parts.push(t('queueRun.memory.part.trainState', {
      trainable: count(estimate.trainable_parameters),
      perParam: String(perTrainable),
      bytes: formatBytes(state),
    }));
  }
  if (copies > 0) {
    parts.push(t('queueRun.memory.part.autocast', {
      params,
      precision: estimate.precision,
      bytes: formatBytes(copies),
    }));
  }
  if (estimate.activation_phase === 'training' || estimate.activation_phase === 'evaluation') {
    const evaluation = estimate.activation_phase === 'evaluation';
    parts.push(t(evaluation ? 'queueRun.memory.part.evalActivations' : 'queueRun.memory.part.trainActivations', {
      batch: String(evaluation ? estimate.eval_batch_size : estimate.batch_size),
      activations: formatBytes(estimate.activation_bytes),
    }));
  }
  parts.push(t('queueRun.memory.part.workspace', { workspace: formatBytes(estimate.workspace_bytes) }));
  const reserve = estimate.bytes - estimate.weights_bytes - estimate.activation_bytes - estimate.workspace_bytes;
  parts.push(t('queueRun.memory.part.reserve', { reserve: formatBytes(reserve) }));
  return parts;
}

/** Why a static estimate is too rough to offer, in words. */
function roughReasons(estimate: StaticMemoryEstimate, t: Translate): string[] {
  const reasons: string[] = [];
  if (estimate.uncounted.length > 0) {
    reasons.push(t('queueRun.memory.rough.uncounted', { nodes: estimate.uncounted.join(', ') }));
  }
  const unshaped = estimate.models.filter((model) => !model.activations_known && model.parameters > 0);
  if (unshaped.length > 0) {
    reasons.push(t('queueRun.memory.rough.shapes', { nodes: unshaped.map((model) => model.type).join(', ') }));
  }
  const guessed = estimate.models.filter((model) => model.activations_known && model.shape_guessed === true);
  if (guessed.length > 0) {
    reasons.push(t('queueRun.memory.rough.guessed', { nodes: guessed.map((model) => model.type).join(', ') }));
  }
  if (estimate.batch_assumed) {
    const batch = estimate.training ? estimate.batch_size : estimate.eval_batch_size;
    reasons.push(t('queueRun.memory.rough.batch', { batch: String(batch) }));
  }
  return reasons;
}
