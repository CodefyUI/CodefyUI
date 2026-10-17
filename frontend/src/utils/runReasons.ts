/**
 * The server's run reasons in the active language (#841).
 *
 * The server words why a run waits (`queue_reason`) and why a worker run
 * ended (`error`, `run_warning` details) in English, which the API and the
 * CLI keep. A waiting run also carries `queue_reason_code` and
 * `queue_reason_params`, worded here from the code. The worker messages
 * have no code: their few fixed shapes (`backend/app/core/executors.py`,
 * `workers.py`) are recognised here, and any other text, or a code this
 * build does not know, is shown as the server wrote it.
 */
import type { useI18n } from '../i18n';
import type { RunSummary } from '../api/rest';

type Translate = ReturnType<typeof useI18n.getState>['t'];

type ReasonFields = Pick<RunSummary, 'queue_reason' | 'queue_reason_code' | 'queue_reason_params'>;

function text(value: unknown): string | null {
  return typeof value === 'string' && value !== '' ? value : null;
}

function names(value: unknown): string[] | null {
  return Array.isArray(value) && value.every((name) => typeof name === 'string')
    ? value as string[]
    : null;
}

/** A list of worker names as one phrase, with the verb for its number. */
function blocked(list: string[], one: 'runs.reason.drainingOne' | 'runs.reason.skippedOne',
                 many: 'runs.reason.drainingMany' | 'runs.reason.skippedMany', t: Translate): string | null {
  if (list.length === 0) return null;
  return t(list.length === 1 ? one : many, { names: list.join(', ') });
}

function fromCode(code: string, params: Record<string, unknown>, t: Translate): string | null {
  const worker = text(params.worker);
  const devices = text(params.devices);
  const mount = text(params.mount);
  switch (code) {
    case 'worker_reconnect':
      return worker && t('runs.reason.workerReconnect', { worker });
    case 'worker_draining':
      return worker && t('runs.reason.workerDraining', { worker });
    case 'worker_no_mount': {
      if (!worker || !mount) return null;
      const transfer = text(params.transfer);
      const sentence = t('runs.reason.workerNoMount', { worker, mount });
      return transfer ? `${sentence}${t('runs.reason.orTransfer', { detail: transfer })}` : sentence;
    }
    case 'any_no_worker':
      return devices && t('runs.reason.anyNoWorker', { devices });
    case 'any_no_worker_mount':
      return devices && mount && t('runs.reason.anyNoWorkerMount', { devices, mount });
    case 'any_unavailable': {
      const draining = names(params.draining);
      const skipped = names(params.skipped);
      if (!devices || !draining || !skipped) return null;
      const drainingText = blocked(draining, 'runs.reason.drainingOne', 'runs.reason.drainingMany', t);
      const skippedText = blocked(skipped, 'runs.reason.skippedOne', 'runs.reason.skippedMany', t);
      if (drainingText && skippedText) {
        return t('runs.reason.anyUnavailableBoth', { devices, draining: drainingText, skipped: skippedText });
      }
      const detail = drainingText ?? skippedText;
      return detail && t('runs.reason.anyUnavailable', { devices, detail });
    }
    default:
      return null;
  }
}

/**
 * Why a queued run waits beyond its place in line, in the active language;
 * the server's English sentence for a code this build does not know or a
 * server that sends none; null when there is no reason.
 */
export function queueReasonText(run: ReasonFields, t: Translate): string | null {
  const fallback = run.queue_reason || null;
  const code = run.queue_reason_code;
  const params = run.queue_reason_params;
  if (!code || !params || typeof params !== 'object') return fallback;
  return fromCode(code, params, t) ?? fallback;
}

/** Why a worker went away (the part after "went away: "). */
function workerGoneReason(reason: string, t: Translate): string {
  let match: RegExpMatchArray | null;
  if (reason === 'after reconnecting it did not report this run as still running') {
    return t('runs.workerMsg.reason.notReported');
  }
  if (reason === 'the reconnected agent does not have it') return t('runs.workerMsg.reason.notOnAgent');
  if (reason === 'the connection closed') return t('runs.workerMsg.reason.connectionClosed');
  if (reason === 'the control plane is shutting down') return t('runs.workerMsg.reason.shuttingDown');
  if (reason === 'the worker was removed') return t('runs.workerMsg.reason.removed');
  if (reason === 'the worker was removed and its credential revoked') return t('runs.workerMsg.reason.revoked');
  if ((match = reason.match(/^worker (\S+) did not reconnect within (\d+) s$/))) {
    return t('runs.workerMsg.reason.noReconnect', { worker: match[1], seconds: match[2] });
  }
  if ((match = reason.match(/^no heartbeat for (\d+) s$/))) {
    return t('runs.workerMsg.reason.noHeartbeat', { seconds: match[1] });
  }
  if ((match = reason.match(/^SSH worker (\S+) was stopped$/))) {
    return t('runs.workerMsg.reason.sshStopped', { worker: match[1] });
  }
  return reason;
}

/**
 * A worker run's error or warning text in the active language: "worker X
 * went away: ...", "worker X disconnected (...); the run keeps running
 * there ...", "worker X reconnected; ...", "worker X is not connected",
 * "could not reach worker X: ...". Any other text comes back unchanged, so
 * a second pass is a no-op.
 */
export function workerMessageText(message: string, t: Translate): string {
  let match: RegExpMatchArray | null;
  if ((match = message.match(/^worker (\S+) went away: ([\s\S]+)$/))) {
    return t('runs.workerMsg.wentAway', { worker: match[1], reason: workerGoneReason(match[2], t) });
  }
  if ((match = message.match(/^worker (\S+) disconnected \(([\s\S]+)\); the run keeps running there and waits up to (\d+) s for it to reconnect$/))) {
    return t('runs.workerMsg.disconnected', {
      worker: match[1], reason: workerGoneReason(match[2], t), seconds: match[3],
    });
  }
  if ((match = message.match(/^worker (\S+) reconnected; the run continues$/))) {
    return t('runs.workerMsg.reconnected', { worker: match[1] });
  }
  if ((match = message.match(/^worker (\S+) is not connected$/))) {
    return t('runs.workerMsg.notConnected', { worker: match[1] });
  }
  if ((match = message.match(/^could not reach worker (\S+): ([\s\S]+)$/))) {
    return t('runs.workerMsg.unreachable', { worker: match[1], detail: match[2] });
  }
  return message;
}
