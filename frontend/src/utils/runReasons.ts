/**
 * The server's run reasons in the active language (#841).
 *
 * The server words why a run waits (`queue_reason`) and why a worker run
 * ended (`error`, `run_warning` details) in English, which the API and the
 * CLI keep. Beside the text each carries a code and parameters, worded
 * here: a waiting run's `queue_reason_code` / `queue_reason_params`, a
 * worker message's `code` / `params` (on the `run_warning` and
 * `execution_error` payloads, and as a row's `execution_facts.error_reason`;
 * built by `backend/app/core/workers.py`). A message without one, or with
 * a code this build does not know, is shown as the server wrote it.
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

/** A worker message's code and parameters, as the server sends them beside its text. */
export interface CodedMessage {
  code: string;
  params: Record<string, unknown>;
}

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

/** `{code, params}` read off an event payload or `execution_facts.error_reason`; undefined when absent. */
export function codedMessage(source: unknown): CodedMessage | undefined {
  const value = record(source);
  const code = value && text(value.code);
  const params = value && record(value.params);
  return code && params ? { code, params } : undefined;
}

function count(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** Why a worker went away (a message's nested `reason`), or its text, or null. */
function reasonText(source: unknown, t: Translate): string | null {
  const value = record(source);
  if (!value) return null;
  const fallback = text(value.text);
  const coded = codedMessage(value);
  if (!coded) return fallback;
  const { params } = coded;
  const worker = text(params.worker);
  const seconds = count(params.seconds);
  const revoked = params.revoked === true;
  switch (coded.code) {
    case 'connection_closed':
      return t('runs.workerMsg.reason.connectionClosed');
    case 'no_heartbeat':
      return seconds === null ? fallback : t('runs.workerMsg.reason.noHeartbeat', { seconds });
    case 'agent_lost_run':
      return t('runs.workerMsg.reason.notOnAgent');
    case 'alive_missing':
      return t('runs.workerMsg.reason.notReported');
    case 'no_reconnect':
      return worker && seconds !== null ? t('runs.workerMsg.reason.noReconnect', { worker, seconds }) : fallback;
    case 'shutting_down':
      return t('runs.workerMsg.reason.shuttingDown');
    case 'removed':
      return t(revoked ? 'runs.workerMsg.reason.revoked' : 'runs.workerMsg.reason.removed');
    case 'ssh_deleted':
      return t(revoked ? 'runs.workerMsg.reason.sshDeletedRevoked' : 'runs.workerMsg.reason.sshDeleted');
    case 'ssh_stopped':
      return worker ? t('runs.workerMsg.reason.sshStopped', { worker }) : fallback;
    case 'rekeyed':
      return t('runs.workerMsg.reason.rekeyed');
    default:
      return fallback;
  }
}

function fromWorkerCode({ code, params }: CodedMessage, t: Translate): string | null {
  const worker = text(params.worker);
  if (!worker) return null;
  const error = text(params.error);
  switch (code) {
    case 'worker_gone': {
      const reason = reasonText(params.reason, t);
      return reason && t('runs.workerMsg.wentAway', { worker, reason });
    }
    case 'worker_disconnected': {
      const reason = reasonText(params.reason, t);
      const seconds = count(params.seconds);
      return reason && seconds !== null ? t('runs.workerMsg.disconnected', { worker, reason, seconds }) : null;
    }
    case 'worker_reconnected':
      return t('runs.workerMsg.reconnected', { worker });
    case 'worker_not_connected':
      return t('runs.workerMsg.notConnected', { worker });
    case 'worker_unreachable':
      return error && t('runs.workerMsg.unreachable', { worker, detail: error });
    case 'worker_transfer_failed':
      return error && t('runs.workerMsg.transferFailed', { worker, detail: error });
    case 'worker_removed':
      return t('runs.workerMsg.removed', { worker });
    default:
      return null;
  }
}

/** A non-empty list of the run's cards as `cuda:0 = the worker's cuda:1, ...`, or null. */
function cardsText(value: unknown[], t: Translate): string | null {
  const pairs = value.map((pair) =>
    Array.isArray(pair) && pair.length === 2 && typeof pair[0] === 'string' && typeof pair[1] === 'string'
      ? t('runs.deviceFallback.card', { inRun: pair[0], physical: pair[1] })
      : null,
  );
  if (pairs.some((pair) => pair === null)) return null;
  const map = (pairs as string[]).reduce((a, b) => t('runs.deviceFallback.cardJoin', { a, b }));
  return value.length === 1
    ? t('runs.deviceFallback.cardsOne', { map })
    : t('runs.deviceFallback.cardsMany', { count: value.length, map });
}

/** A `device_fallback` warning (#839), from its code and parameters. */
function fromDeviceCode({ code, params }: CodedMessage, t: Translate): string | null {
  const requested = text(params.requested);
  const device = text(params.device);
  if (!requested || !device) return null;
  const where = params.where === 'worker' ? t('runs.deviceFallback.worker')
    : params.where === 'server' ? t('runs.deviceFallback.server') : null;
  const node = text(params.node);
  const type = text(params.type);
  switch (code) {
    case 'device_fallback_run':
      return where && t('runs.deviceFallback.run', { requested, device, where });
    case 'device_fallback_node':
      return where && node && type && t('runs.deviceFallback.node', { node, type, requested, device, where });
    case 'device_fallback_node_in_run': {
      if (!node || !type || !Array.isArray(params.cards)) return null;
      if (params.cards.length === 0) {
        return t('runs.deviceFallback.nodeNoCard', { node, type, requested, device });
      }
      const cards = cardsText(params.cards, t);
      return cards && t('runs.deviceFallback.nodeInRun', { node, type, requested, device, cards });
    }
    default:
      return null;
  }
}

/**
 * A worker run's error or warning, or a run's `device_fallback` warning, in
 * the active language, from its code and parameters; *message* (the
 * server's English) when there is no code, or one this build does not know.
 */
export function workerMessageText(message: string, coded: CodedMessage | undefined, t: Translate): string {
  if (!coded) return message;
  return (coded.code.startsWith('device_fallback') ? fromDeviceCode(coded, t) : fromWorkerCode(coded, t))
    || message;
}
