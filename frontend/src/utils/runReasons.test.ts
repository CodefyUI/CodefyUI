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

describe('queueReasonText: Wave D codes (#788, #790, #794)', () => {
  const card = (code: string, params: Record<string, unknown>, text = 'card sentence') => ({ code, params, text });

  it('words the one-shot reasons as the server does, and in zh-TW', () => {
    expect(reason('one_shot_start', { worker: 'slurm' }))
      .toBe('waiting for one-shot worker slurm to start (cdui agent start --one-shot)');
    expect(reason('one_shot_taken', { worker: 'slurm', run: 'r0' }))
      .toBe('one-shot worker slurm already took run r0; start another one-shot agent');
    expect(reason('one_shot_run_id', { worker: 'slurm', run: 'r7' }))
      .toBe('one-shot worker slurm takes only run r7 (--run-id)');
    expect(reason('one_shot_pool', { devices: 'cuda' }))
      .toBe('waiting for a one-shot worker that offers cuda (cdui agent start --one-shot)');
    useI18n.setState({ locale: 'zh-TW' });
    expect(reason('one_shot_start', { worker: 'slurm' })).toBe('等待單次 worker slurm 啟動（cdui agent start --one-shot）');
    expect(reason('one_shot_pool', { devices: 'cuda' })).toBe('等待提供 cuda 的單次 worker（cdui agent start --one-shot）');
    // Missing parameters keep the server's sentence.
    expect(reason('one_shot_taken', { worker: 'slurm' })).toBe('server sentence');
    expect(reason('one_shot_run_id', { run: 'r7' })).toBe('server sentence');
  });

  it('words a memory wait card by card, and a missing budget', () => {
    expect(reason('memory_no_budget', { device: 'cuda:0', worker: 'box' })).toBe(
      'cuda:0 on box is shared by memory and this run has no memory budget: it takes one once a run of the same '
      + 'graph on the same devices reports its peak, or cancel it and queue it again with target.memory_gib');
    expect(reason('memory_no_budget', { worker: 'box' })).toBe('server sentence');
    expect(reason('memory_wait', {
      need: '6.0 GiB', context: null,
      reasons: [card('does_not_fit', { device: 'cuda:0', worker: 'box', free: '4.0 GiB', runs: ['r1', 'r2'] })],
    })).toBe('waiting for memory: needs 6.0 GiB, cuda:0 on box has 4.0 GiB free (runs r1, r2)');
    expect(reason('memory_wait', {
      need: '6.0 GiB', context: '0.8 GiB',
      reasons: [
        card('does_not_fit', { device: 'cuda:0', worker: 'a', free: '4.0 GiB', runs: [] }),
        card('no_total_reported', { device: 'cuda:1', worker: 'a' }),
        card('total_too_small', { device: 'cuda:2', worker: 'a', total: '4.0 GiB' }),
        card('held_without_budget', { device: 'cuda:3', worker: 'a' }),
        card('reserved_for_a_set', { device: 'cuda:4', worker: 'a' }),
        card('no_total', { device: 'cuda:5', worker: 'a' }),
      ],
    })).toBe('waiting for memory: needs 6.0 GiB (+0.8 GiB CUDA context), cuda:0 on a has 4.0 GiB free; '
      + 'cuda:1 on a has no memory total reported; cuda:2 on a has 4.0 GiB in total; '
      + 'cuda:3 on a is held by a run without a memory budget (a run on this server, or one started before the '
      + 'card was shared) until it ends; cuda:4 on a is reserved for an earlier run that waits for its whole device '
      + 'set; cuda:5 on a reports no memory total, so it cannot admit by memory');
    useI18n.setState({ locale: 'zh-TW' });
    expect(reason('memory_wait', {
      need: '6.0 GiB', context: '0.8 GiB',
      reasons: [card('does_not_fit', { device: 'cuda:0', worker: 'box', free: '4.0 GiB', runs: ['r1'] }),
        card('other', {}, 'plain text')],
    })).toBe('等待記憶體：需要 6.0 GiB（另加 0.8 GiB CUDA context），box 的 cuda:0 可用 4.0 GiB（run r1）；plain text');
  });

  it("keeps the server's sentence for a memory wait it cannot word", () => {
    expect(reason('memory_wait', { need: '6.0 GiB', reasons: [] })).toBe('server sentence');
    expect(reason('memory_wait', { reasons: [card('no_total', { device: 'cuda:0', worker: 'a' })] }))
      .toBe('server sentence');
    expect(reason('memory_wait', { need: '6.0 GiB', reasons: 'x' })).toBe('server sentence');
    // A card reason that is not an object, or names no card or worker.
    expect(reason('memory_wait', { need: '6.0 GiB', reasons: [3] })).toBe('server sentence');
    expect(reason('memory_wait', { need: '6.0 GiB', reasons: [{ text: 'as sent' }] }))
      .toBe('waiting for memory: needs 6.0 GiB, as sent');
    expect(reason('memory_wait', { need: '6.0 GiB', reasons: [card('no_total', { device: 'cuda:0' })] }))
      .toBe('waiting for memory: needs 6.0 GiB, card sentence');
    expect(reason('memory_wait', { need: '6.0 GiB', reasons: [card('total_too_small', { device: 'd', worker: 'w' })] }))
      .toBe('waiting for memory: needs 6.0 GiB, card sentence');
    expect(reason('memory_wait', { need: '6.0 GiB', reasons: [card('does_not_fit', { device: 'd', worker: 'w' })] }))
      .toBe('waiting for memory: needs 6.0 GiB, card sentence');
    expect(reason('memory_wait', { need: '6.0 GiB', reasons: [card('eclipse', { device: 'd', worker: 'w' })] }))
      .toBe('waiting for memory: needs 6.0 GiB, card sentence');
  });

  it('words a raised estimate alone or beside another reason (#794)', () => {
    expect(reason('budget_raised', { from: '1.0 GiB', to: '2.4 GiB', run: 'r1' })).toBe(
      'memory budget raised from the accepted estimate, 1.0 GiB, to 2.4 GiB: an earlier run of its graph (run r1) '
      + 'peaked above the estimate');
    expect(reason('budget_raised', { from: '10 MiB', to: '58 MiB', run: null })).toBe(
      'memory budget raised from the accepted estimate, 10 MiB, to 58 MiB: an earlier run of its graph peaked above '
      + 'the estimate');
    expect(reason('budget_raised', { from: '10 MiB' })).toBe('server sentence');
    expect(reason('worker_draining', { worker: 'box', raised: { from: '1.0 GiB', to: '2.4 GiB', run: null } })).toBe(
      'worker box is draining; memory budget raised from the accepted estimate, 1.0 GiB, to 2.4 GiB: an earlier '
      + 'run of its graph peaked above the estimate');
    expect(reason('worker_draining', { worker: 'box', raised: 'odd' })).toBe('server sentence');
    useI18n.setState({ locale: 'zh-TW' });
    expect(reason('worker_draining', { worker: 'box', raised: { from: '1.0 GiB', to: '2.4 GiB', run: 'r1' } })).toBe(
      'worker box 排空中；記憶體預算已從接受的估計值 1.0 GiB 調高到 2.4 GiB：同一張圖較早的 run（r1）峰值超過估計值');
  });
});

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

  it('words the adoption, shutdown and S3 messages (#791, #793)', () => {
    useI18n.setState({ locale: 'zh-TW' });
    expect(workerMessageText('english', { code: 'worker_adopted', params: { worker: 'box', seconds: 300 } }, t))
      .toBe('本伺服器在 run 於 worker box 上執行時重新啟動；run 仍在該機器上執行，最多等待 300 秒讓 worker 重新連線');
    expect(workerMessageText('english', { code: 'worker_adopted', params: { worker: 'box' } }, t)).toBe('english');
    expect(workerMessageText('english', { code: 'worker_lost_after_restart', params: { worker: 'box' } }, t))
      .toBe('本伺服器已重新啟動，而 worker box 上已沒有這個 run');
    expect(workerMessageText('english', { code: 'worker_s3_transfer_failed', params: { worker: 'box', error: 'denied' } }, t))
      .toBe('無法把這個 run 的輸入檔案傳送到 S3 bucket：denied');
    expect(workerMessageText('english', { code: 'worker_s3_transfer_failed', params: { worker: 'box' } }, t))
      .toBe('english');
    expect(workerMessageText('english', { code: 'worker_storage_fallback', params: { worker: 'box' } }, t))
      .toBe('worker box 無法使用 S3 bucket（它的 agent 沒有 boto3）；這個 run 的檔案改經由本伺服器傳送');
    const cases: Array<[ReturnType<typeof reasonOf>, string]> = [
      [reasonOf('no_reconnect_after_restart', 'it did not reconnect', { seconds: 300 }),
        '本伺服器重新啟動後，它未在 300 秒內帶著這個 run 重新連線'],
      [reasonOf('no_reconnect_after_restart', 'it did not reconnect'), 'it did not reconnect'],
      [reasonOf('shutdown_not_connected', 'shutting down', { worker: 'box' }), '本伺服器正在關閉，而 worker box 未連線'],
      [reasonOf('shutdown_not_connected', 'shutting down'), 'shutting down'],
    ];
    for (const [why, chinese] of cases) {
      const { message, coded } = gone(why);
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

describe('workerMessageText: device_fallback (#839)', () => {
  const run = { code: 'device_fallback_run', params: { requested: 'cuda', device: 'cpu', where: 'server' } };
  const node = {
    code: 'device_fallback_node',
    params: { node: 'train', type: 'TrainingLoop', requested: 'cuda:0', device: 'cpu', where: 'worker' },
  };
  const inRun = (cards: unknown) => ({
    code: 'device_fallback_node_in_run',
    params: { node: 'train', type: 'TrainingLoop', requested: 'cuda:1', device: 'cuda:0', cards },
  });

  it('words each code in English exactly as the server does', () => {
    expect(workerMessageText('x', run, t))
      .toBe('The run asked for device cuda, which this server does not have; it runs on cpu.');
    expect(workerMessageText('x', node, t)).toBe(
      'Node train (TrainingLoop) is set to device cuda:0, which this worker does not have; when it executes, it runs on cpu.',
    );
    expect(workerMessageText('x', inRun([['cuda:0', 'cuda:1']]), t)).toBe(
      "Node train (TrainingLoop) is set to device cuda:1; this run has 1 card (cuda:0 = the worker's cuda:1), so when it executes, it runs on cuda:0.",
    );
    expect(workerMessageText('x', inRun([['cuda:0', 'cuda:2'], ['cuda:1', 'cuda:0']]), t)).toBe(
      "Node train (TrainingLoop) is set to device cuda:1; this run has 2 cards (cuda:0 = the worker's cuda:2, cuda:1 = the worker's cuda:0), so when it executes, it runs on cuda:0.",
    );
    expect(workerMessageText('x', inRun([]), t)).toBe(
      'Node train (TrainingLoop) is set to device cuda:1; this run has no CUDA card, so when it executes, it runs on cuda:0.',
    );
  });

  it('words them in zh-TW', () => {
    useI18n.setState({ locale: 'zh-TW' });
    expect(workerMessageText('x', run, t)).toBe('run 要求的裝置 cuda 本伺服器沒有；改在 cpu 上執行。');
    expect(workerMessageText('x', node, t))
      .toBe('節點 train（TrainingLoop）設定為裝置 cuda:0，這個 worker 上沒有這個裝置；執行時會改用 cpu。');
    expect(workerMessageText('x', inRun([['cuda:0', 'cuda:1'], ['cuda:1', 'cuda:3']]), t))
      .toBe('節點 train（TrainingLoop）設定為裝置 cuda:1；這個 run 有 2 張卡（cuda:0 即 worker 的 cuda:1、cuda:1 即 worker 的 cuda:3），因此執行時會改用 cuda:0。');
    expect(workerMessageText('x', inRun([['cuda:0', 'cuda:1']]), t))
      .toBe('節點 train（TrainingLoop）設定為裝置 cuda:1；這個 run 有 1 張卡（cuda:0 即 worker 的 cuda:1），因此執行時會改用 cuda:0。');
    expect(workerMessageText('x', inRun([]), t))
      .toBe('節點 train（TrainingLoop）設定為裝置 cuda:1；這個 run 沒有 CUDA 卡，因此執行時會改用 cuda:0。');
  });

  it('keeps the server sentence for parameters it cannot read', () => {
    const message = 'server sentence';
    const bad = [
      { code: 'device_fallback_run', params: { requested: 'cuda', device: 'cpu', where: 'moon' } },
      { code: 'device_fallback_run', params: { device: 'cpu', where: 'server' } },
      { code: 'device_fallback_run', params: { requested: 'cuda', where: 'server' } },
      { code: 'device_fallback_node', params: { ...node.params, node: undefined } },
      { code: 'device_fallback_node', params: { ...node.params, type: undefined } },
      { code: 'device_fallback_node', params: { ...node.params, where: undefined } },
      { code: 'device_fallback_node_in_run', params: { ...inRun([]).params, node: undefined } },
      { code: 'device_fallback_node_in_run', params: { ...inRun([]).params, type: undefined } },
      inRun('not a list'),
      inRun([['cuda:0']]),
      inRun([['cuda:0', 1]]),
      inRun(['cuda:0']),
      { code: 'device_fallback_future', params: { requested: 'cuda', device: 'cpu' } },
    ];
    for (const coded of bad) expect(workerMessageText(message, coded, t)).toBe(message);
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
