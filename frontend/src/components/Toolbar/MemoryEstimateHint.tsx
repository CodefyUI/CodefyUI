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
 * "Measured" or "Estimate" tag, then how the figure was reached. A rough
 * (low confidence) estimate is shown for reference with why it is not used,
 * and the field stays empty.
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
  if (estimate?.source === 'estimate' && statics !== null) {
    return (
      <p className={styles.hint} aria-live="polite">
        <span className={`${styles.tag} ${styles.tagEstimate}`}>{t('queueRun.memory.tag.estimate')}</span>{' '}
        <span>{t('queueRun.memory.estimated', { parts: breakdown(statics, t) })}</span>
      </p>
    );
  }
  if (statics !== null) {
    return (
      <p className={styles.hint} aria-live="polite">
        <span className={`${styles.tag} ${styles.tagEstimate}`}>{t('queueRun.memory.tag.rough')}</span>{' '}
        <span>{t('queueRun.memory.rough', { size: formatGiB(statics.bytes), why: roughReasons(statics, t) })}</span>
      </p>
    );
  }
  return <p className={styles.hint} aria-live="polite">{t('queueRun.memory.none')}</p>;
}

type Translate = ReturnType<typeof useI18n.getState>['t'];

/** How the estimate adds up, part by part: the weights (and, for a graph
 * that trains, their gradients and optimizer state), the activations of the
 * pass that sets them when there is one, and the libraries' workspace. Each
 * part in the unit that reads best (a small model's weights are kilobytes);
 * the total stays in GiB, the unit the field takes. */
function breakdown(estimate: StaticMemoryEstimate, t: Translate): string {
  const params = estimate.parameters.toLocaleString('en-US');
  const weights = formatBytes(estimate.weights_bytes);
  const parts = [
    estimate.training
      ? t('queueRun.memory.part.trainWeights', { params, perParam: String(estimate.multiplier * 4), weights })
      : t('queueRun.memory.part.weights', { params, weights }),
  ];
  if (estimate.activation_phase === 'training' || estimate.activation_phase === 'evaluation') {
    const evaluation = estimate.activation_phase === 'evaluation';
    parts.push(t(evaluation ? 'queueRun.memory.part.evalActivations' : 'queueRun.memory.part.trainActivations', {
      batch: String(evaluation ? estimate.eval_batch_size : estimate.batch_size),
      activations: formatBytes(estimate.activation_bytes),
    }));
  }
  parts.push(t('queueRun.memory.part.workspace', { workspace: formatBytes(estimate.workspace_bytes) }));
  return parts.join(t('queueRun.memory.part.separator'));
}

/** Why a static estimate is too rough to be the default, in words. */
function roughReasons(estimate: StaticMemoryEstimate, t: Translate): string {
  const reasons: string[] = [];
  if (estimate.uncounted.length > 0) {
    reasons.push(t('queueRun.memory.rough.uncounted', { nodes: estimate.uncounted.join(', ') }));
  }
  const unshaped = estimate.models.filter((model) => !model.activations_known && model.parameters > 0);
  if (unshaped.length > 0) {
    reasons.push(t('queueRun.memory.rough.shapes', { nodes: unshaped.map((model) => model.type).join(', ') }));
  }
  if (estimate.batch_assumed) {
    reasons.push(t('queueRun.memory.rough.batch', { batch: String(estimate.batch_size) }));
  }
  return reasons.join(t('queueRun.memory.rough.separator'));
}
