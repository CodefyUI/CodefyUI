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
        <span>{t('queueRun.memory.estimated', breakdown(statics))}</span>
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

function breakdown(estimate: StaticMemoryEstimate): Record<string, string> {
  return {
    params: estimate.parameters.toLocaleString('en-US'),
    perParam: String(estimate.multiplier * 4),
    // The parts in the unit that reads best (a small model's weights are
    // kilobytes); the total stays in GiB, the unit the field takes.
    weights: formatBytes(estimate.weights_bytes),
    batch: String(estimate.training ? estimate.batch_size : estimate.eval_batch_size),
    activations: formatBytes(estimate.activation_bytes),
    workspace: formatBytes(estimate.workspace_bytes),
  };
}

type Translate = ReturnType<typeof useI18n.getState>['t'];

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
