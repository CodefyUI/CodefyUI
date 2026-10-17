import { beforeEach, describe, expect, it } from 'vitest';
import { useI18n } from '../i18n';
import { queueReasonText, workerMessageText } from './runReasons';

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

describe('workerMessageText', () => {
  it('words a worker run ending, in both languages', () => {
    const gone = 'worker meow-e2e went away: after reconnecting it did not report this run as still running';
    expect(workerMessageText(gone, t)).toBe(gone);
    useI18n.setState({ locale: 'zh-TW' });
    expect(workerMessageText(gone, t)).toBe('worker meow-e2e 已離開：重新連線後沒有回報這個 run 仍在執行');
    expect(workerMessageText(
      'worker box disconnected (no heartbeat for 31 s); the run keeps running there and waits up to 300 s for it to reconnect', t,
    )).toBe('worker box 已斷線（31 秒沒有心跳）；run 仍在該機器上執行，最多等待 300 秒讓它重新連線');
    expect(workerMessageText('worker box reconnected; the run continues', t)).toBe('worker box 已重新連線；run 繼續執行');
    expect(workerMessageText('worker box is not connected', t)).toBe('worker box 未連線');
    expect(workerMessageText('could not reach worker box: socket closed', t)).toBe('無法連到 worker box：socket closed');
  });

  it('words every reason a worker goes away for', () => {
    useI18n.setState({ locale: 'zh-TW' });
    const cases: Array<[string, string]> = [
      ['the reconnected agent does not have it', '重新連線的 agent 上沒有這個 run'],
      ['the connection closed', '連線已關閉'],
      ['the control plane is shutting down', '控制面正在關閉'],
      ['the worker was removed', '這個 worker 已被移除'],
      ['the worker was removed and its credential revoked', '這個 worker 已被移除，憑證已撤銷'],
      ['worker box did not reconnect within 300 s', 'worker box 未在 300 秒內重新連線'],
      ['no heartbeat for 31 s', '31 秒沒有心跳'],
      ['SSH worker box was stopped', 'SSH worker box 已停止'],
      // One this build does not know stays as the server wrote it.
      ['the moon is full', 'the moon is full'],
    ];
    for (const [english, chinese] of cases) {
      expect(workerMessageText(`worker box went away: ${english}`, t)).toBe(`worker box 已離開：${chinese}`);
    }
  });

  it('leaves any other text alone, so a second pass changes nothing', () => {
    useI18n.setState({ locale: 'zh-TW' });
    expect(workerMessageText('CUDA out of memory', t)).toBe('CUDA out of memory');
    const once = workerMessageText('worker box is not connected', t);
    expect(workerMessageText(once, t)).toBe(once);
  });
});
