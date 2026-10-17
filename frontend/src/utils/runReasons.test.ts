import { beforeEach, describe, expect, it } from 'vitest';
import { useI18n } from '../i18n';
import { codedMessage, queueReasonText, workerMessageText } from './runReasons';

const t = (key: Parameters<ReturnType<typeof useI18n.getState>['t']>[0], vars?: Record<string, string | number>) =>
  useI18n.getState().t(key, vars);

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
});

function reason(code: string | null | undefined, params: Record<string, unknown> | null | undefined,
                text: string | null = 'server sentence') {
  return queueReasonText({ queue_reason: text, queue_reason_code: code, queue_reason_params: params }, t);
}

describe('queueReasonText', () => {
  it('words every code the server sends, in English as the server does', () => {
    expect(reason('worker_reconnect', { worker: 'gpu-box' })).toBe('waiting for worker gpu-box to reconnect');
    expect(reason('worker_draining', { worker: 'gpu-box' })).toBe('worker gpu-box is draining');
    expect(reason('worker_no_mount', { worker: 'gpu-box', mount: 'data', transfer: null }))
      .toBe('worker gpu-box has no `data` mount; add `--mount data=...` to its agent');
    expect(reason('worker_no_mount', { worker: 'gpu-box', mount: 'data', transfer: 'too big' }))
      .toBe('worker gpu-box has no `data` mount; add `--mount data=...` to its agent, '
        + 'or make the run fit the transfer backend: too big');
    expect(reason('any_no_worker', { devices: 'cuda' })).toBe('waiting for a worker that offers cuda');
    expect(reason('any_no_worker_mount', { devices: 'cuda', mount: 'data' }))
      .toBe('waiting for a worker that offers cuda and has the `data` mount');
  });

  it('names the draining and the skipped workers of an any run, one or many (U5)', () => {
    expect(reason('any_unavailable', { devices: 'cuda', draining: ['meow'], skipped: [] }))
      .toBe('waiting for a worker that offers cuda: meow is draining');
    expect(reason('any_unavailable', { devices: 'cuda', draining: ['a', 'b'], skipped: [] }))
      .toBe('waiting for a worker that offers cuda: a, b are draining');
    expect(reason('any_unavailable', { devices: 'cuda', draining: [], skipped: ['c'] }))
      .toBe('waiting for a worker that offers cuda: c is skipped by `any` (node types missing, or not reported)');
    expect(reason('any_unavailable', { devices: 'cuda', draining: ['a'], skipped: ['c', 'd'] }))
      .toBe('waiting for a worker that offers cuda: a is draining; '
        + 'c, d are skipped by `any` (node types missing, or not reported)');
  });

  it('words them in Traditional Chinese', () => {
    useI18n.setState({ locale: 'zh-TW' });
    expect(reason('worker_draining', { worker: 'meow-e2e' })).toBe('worker meow-e2e 排空中');
    expect(reason('any_unavailable', { devices: 'cuda', draining: ['meow'], skipped: ['x'] }))
      .toBe('等待提供 cuda 的 worker：meow 排空中；x 不在 `any` 的選擇範圍（缺少節點類型，或未回報）');
  });

  it("falls back to the server's sentence for what it cannot word", () => {
    // No code (a server before #841), no params, an unknown code.
    expect(reason(undefined, undefined)).toBe('server sentence');
    expect(reason('worker_draining', null)).toBe('server sentence');
    expect(reason('from_the_future', {})).toBe('server sentence');
    // Params of the wrong shape.
    expect(reason('worker_draining', { worker: '' })).toBe('server sentence');
    expect(reason('worker_reconnect', { worker: 7 })).toBe('server sentence');
    expect(reason('worker_no_mount', { worker: 'w' })).toBe('server sentence');
    expect(reason('any_no_worker', {})).toBe('server sentence');
    expect(reason('any_no_worker_mount', { devices: 'cuda' })).toBe('server sentence');
    expect(reason('any_unavailable', { devices: 'cuda', draining: 'meow', skipped: [] })).toBe('server sentence');
    expect(reason('any_unavailable', { devices: 'cuda', draining: [1], skipped: [] })).toBe('server sentence');
    expect(reason('any_unavailable', { devices: 'cuda', draining: [], skipped: [] })).toBe('server sentence');
    expect(reason('any_unavailable', { draining: [], skipped: [] })).toBe('server sentence');
    // No reason at all.
    expect(reason(null, null, null)).toBeNull();
    expect(reason(null, null, '')).toBeNull();
  });
});

const reasonOf = (code: string, text: string, params: Record<string, unknown> = {}) => ({ code, params, text });

/** The server's `worker_gone` message for *reason*: its English text and its code. */
function gone(reason: { code: string; params: Record<string, unknown>; text: string }) {
  return {
    message: `worker box went away: ${reason.text}`,
    coded: { code: 'worker_gone', params: { worker: 'box', reason } },
  };
}

describe('workerMessageText', () => {
  it('words a worker message by its code, in both languages', () => {
    const { message, coded } = gone(reasonOf('alive_missing',
      'after reconnecting it did not report this run as still running'));
    expect(workerMessageText(message, coded, t)).toBe(message);
    useI18n.setState({ locale: 'zh-TW' });
    expect(workerMessageText(message, coded, t)).toBe('worker box 已離開：重新連線後沒有回報這個 run 仍在執行');
    expect(workerMessageText('english', {
      code: 'worker_disconnected',
      params: { worker: 'box', seconds: 300, reason: reasonOf('no_heartbeat', 'no heartbeat for 31 s', { seconds: 31 }) },
    }, t)).toBe('worker box 已斷線（31 秒沒有心跳）；run 仍在該機器上執行，最多等待 300 秒讓它重新連線');
    expect(workerMessageText('english', { code: 'worker_reconnected', params: { worker: 'box' } }, t))
      .toBe('worker box 已重新連線；run 繼續執行');
    expect(workerMessageText('english', { code: 'worker_not_connected', params: { worker: 'box' } }, t))
      .toBe('worker box 未連線');
    expect(workerMessageText('english', { code: 'worker_unreachable', params: { worker: 'box', error: 'socket closed' } }, t))
      .toBe('無法連到 worker box：socket closed');
    expect(workerMessageText('english', { code: 'worker_transfer_failed', params: { worker: 'box', error: 'too big' } }, t))
      .toBe('無法把這個 run 的輸入檔案傳送到 worker box：too big');
    expect(workerMessageText('english', { code: 'worker_removed', params: { worker: 'box' } }, t))
      .toBe('worker box 已被移除');
  });

  it('words every reason a worker goes away for', () => {
    useI18n.setState({ locale: 'zh-TW' });
    const cases: Array<[ReturnType<typeof reasonOf>, string]> = [
      [reasonOf('connection_closed', 'the connection closed'), '連線已關閉'],
      [reasonOf('no_heartbeat', 'no heartbeat for 31 s', { seconds: 31 }), '31 秒沒有心跳'],
      [reasonOf('agent_lost_run', 'the reconnected agent does not have it'), '重新連線的 agent 上沒有這個 run'],
      [reasonOf('no_reconnect', 'worker box did not reconnect within 300 s', { worker: 'box', seconds: 300 }),
        'worker box 未在 300 秒內重新連線'],
      [reasonOf('shutting_down', 'the control plane is shutting down'), '控制面正在關閉'],
      [reasonOf('removed', 'the worker was removed', { revoked: false }), '這個 worker 已被移除'],
      [reasonOf('removed', 'the worker was removed and its credential revoked', { revoked: true }),
        '這個 worker 已被移除，憑證已撤銷'],
      [reasonOf('ssh_deleted', 'the SSH worker was deleted', { revoked: false }), '這個 SSH worker 已被刪除'],
      [reasonOf('ssh_deleted', 'the SSH worker was deleted and its credential revoked', { revoked: true }),
        '這個 SSH worker 已被刪除，憑證已撤銷'],
      [reasonOf('ssh_stopped', 'SSH worker box was stopped', { worker: 'box' }), 'SSH worker box 已停止'],
      [reasonOf('rekeyed', 'the worker was re-keyed; its old credential no longer works'), '這個 worker 已重新產生金鑰，舊憑證已失效'],
      // A plain-text reason, a code this build does not know, and a code
      // whose parameters are missing all keep the server's text.
      [reasonOf('other', 'the moon is full'), 'the moon is full'],
      [reasonOf('eclipse', 'the sun is gone'), 'the sun is gone'],
      [reasonOf('no_heartbeat', 'no heartbeat for a while'), 'no heartbeat for a while'],
      [reasonOf('no_reconnect', 'worker box did not reconnect'), 'worker box did not reconnect'],
      [reasonOf('ssh_stopped', 'SSH worker was stopped'), 'SSH worker was stopped'],
    ];
    for (const [reason, chinese] of cases) {
      const { message, coded } = gone(reason);
      expect(workerMessageText(message, coded, t)).toBe(`worker box 已離開：${chinese}`);
    }
  });

  it('keeps the server text when there is no code, or no way to word it', () => {
    useI18n.setState({ locale: 'zh-TW' });
    expect(workerMessageText('CUDA out of memory', undefined, t)).toBe('CUDA out of memory');
    expect(workerMessageText('new', { code: 'worker_moved', params: { worker: 'box' } }, t)).toBe('new');
    expect(workerMessageText('no worker', { code: 'worker_reconnected', params: {} }, t)).toBe('no worker');
    expect(workerMessageText('no error', { code: 'worker_unreachable', params: { worker: 'box' } }, t)).toBe('no error');
    expect(workerMessageText('no error', { code: 'worker_transfer_failed', params: { worker: 'box' } }, t)).toBe('no error');
    expect(workerMessageText('no reason', { code: 'worker_gone', params: { worker: 'box' } }, t)).toBe('no reason');
    expect(workerMessageText('no reason', { code: 'worker_gone', params: { worker: 'box', reason: 'text' } }, t))
      .toBe('no reason');
    expect(workerMessageText('textless', { code: 'worker_gone', params: { worker: 'box', reason: { code: 'eclipse' } } }, t))
      .toBe('textless');
    expect(workerMessageText('no grace', {
      code: 'worker_disconnected', params: { worker: 'box', reason: reasonOf('connection_closed', 'x') },
    }, t)).toBe('no grace');
    expect(workerMessageText('no reason', { code: 'worker_disconnected', params: { worker: 'box', seconds: 3 } }, t))
      .toBe('no reason');
  });
});

describe('codedMessage', () => {
  it('reads a code and parameters, and nothing else', () => {
    expect(codedMessage({ code: 'worker_removed', params: { worker: 'box' }, detail: 'x' }))
      .toEqual({ code: 'worker_removed', params: { worker: 'box' } });
    expect(codedMessage(undefined)).toBeUndefined();
    expect(codedMessage(null)).toBeUndefined();
    expect(codedMessage(['worker_removed'])).toBeUndefined();
    expect(codedMessage({ code: '', params: {} })).toBeUndefined();
    expect(codedMessage({ code: 'worker_removed' })).toBeUndefined();
    expect(codedMessage({ code: 'worker_removed', params: [] })).toBeUndefined();
  });
});
