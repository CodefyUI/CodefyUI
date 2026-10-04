import {
  describe,
  it,
  expect,
  beforeEach,
  afterEach,
  vi,
  type Mock,
  type MockInstance,
} from 'vitest';
import { IDBFactory, IDBKeyRange } from 'fake-indexeddb';

import {
  createWorkspaceLock,
  startWorkspaceLock,
  getWorkspaceLock,
  noteWorkspaceSuperseded,
  workspaceWriteAllowed,
  _resetWorkspaceLockForTests,
  type ChannelLike,
  type LockManagerLike,
  type WorkspaceLock,
  type WorkspaceLockEnv,
} from './workspaceLock';

// ── One origin, faked: a lock manager and a broadcast bus every page shares ──

function abortError(): DOMException {
  return new DOMException('The request was aborted.', 'AbortError');
}

interface LockWaiter {
  callback: (lock: unknown) => unknown;
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
}

/**
 * The slice of Web Locks the module uses, with the browser's rules: one
 * exclusive holder, a FIFO queue, `ifAvailable` answering null while the lock
 * is held or anyone is queued, `steal` breaking the current hold (the old
 * holder's request rejects with AbortError), and `signal` dropping a queued
 * request. The callback runs asynchronously, and the lock is held until the
 * promise it returns settles.
 */
class FakeLocks implements LockManagerLike {
  holder: LockWaiter | null = null;
  readonly queue: LockWaiter[] = [];
  readonly requests: Array<{ ifAvailable?: boolean; steal?: boolean }> = [];

  request(
    _name: string,
    options: { ifAvailable?: boolean; steal?: boolean; signal?: AbortSignal },
    callback: (lock: unknown) => unknown,
  ): Promise<unknown> {
    this.requests.push({ ifAvailable: options.ifAvailable, steal: options.steal });
    return new Promise((resolve, reject) => {
      const waiter: LockWaiter = { callback, resolve, reject };
      if (options.signal?.aborted) {
        reject(abortError());
        return;
      }
      if (options.steal) {
        const stolen = this.holder;
        this.holder = null;
        stolen?.reject(abortError());
        this.grant(waiter);
        return;
      }
      if (this.holder === null && this.queue.length === 0) {
        this.grant(waiter);
        return;
      }
      if (options.ifAvailable) {
        void Promise.resolve()
          .then(() => callback(null))
          .then(resolve, reject);
        return;
      }
      this.queue.push(waiter);
      options.signal?.addEventListener('abort', () => {
        const at = this.queue.indexOf(waiter);
        if (at === -1) return;
        this.queue.splice(at, 1);
        reject(abortError());
      });
    });
  }

  private grant(waiter: LockWaiter): void {
    this.holder = waiter;
    void Promise.resolve()
      .then(() => waiter.callback({ name: 'workspace' }))
      .then(
        (value) => {
          this.release(waiter);
          waiter.resolve(value);
        },
        (error: unknown) => {
          this.release(waiter);
          waiter.reject(error);
        },
      );
  }

  private release(waiter: LockWaiter): void {
    // Stolen, or its page went away: someone else's hold is not ours to end.
    if (this.holder !== waiter) return;
    this.holder = null;
    this.next();
  }

  private next(): void {
    const waiter = this.queue.shift();
    if (waiter) this.grant(waiter);
  }
}

/** BroadcastChannel semantics: async, cloned, never echoed to the sender. */
class FakeBus {
  private readonly channels = new Set<FakeChannel>();

  open = (name: string): FakeChannel => {
    const channel = new FakeChannel(this, name);
    this.channels.add(channel);
    return channel;
  };

  send(from: FakeChannel, message: unknown): void {
    for (const to of this.channels) {
      if (to === from || to.name !== from.name) continue;
      to.receive(structuredClone(message));
    }
  }
}

class FakeChannel implements ChannelLike {
  closed = false;
  private onMessage: ((data: unknown) => void) | null = null;
  // Non-null while the page's main thread is "busy": messages wait in its
  // task queue instead of being lost, which is what a real busy page does.
  private held: unknown[] | null = null;

  constructor(
    private readonly bus: FakeBus,
    readonly name: string,
  ) {}

  post(message: unknown): void {
    if (this.closed) throw new DOMException('Channel is closed', 'InvalidStateError');
    this.bus.send(this, message);
  }

  listen(onMessage: (data: unknown) => void): void {
    this.onMessage = onMessage;
  }

  close(): void {
    this.closed = true;
  }

  receive(message: unknown): void {
    if (this.closed) return;
    if (this.held) {
      this.held.push(message);
      return;
    }
    void Promise.resolve().then(() => {
      if (!this.closed) this.onMessage?.(message);
    });
  }

  pause(): void {
    this.held = [];
  }

  resume(): void {
    const held = this.held ?? [];
    this.held = null;
    for (const message of held) this.receive(message);
  }
}

interface Origin {
  locks: FakeLocks | null;
  bus: FakeBus | null;
}

function newOrigin({ locks = true, bus = true } = {}): Origin {
  return { locks: locks ? new FakeLocks() : null, bus: bus ? new FakeBus() : null };
}

interface Page {
  lock: WorkspaceLock;
  channel: FakeChannel | null;
  reload: Mock<() => void>;
  flush: Mock<() => Promise<void>>;
}

const openPages: Page[] = [];

/** One browser tab of the app, as far as the lock can tell. */
function openPage(origin: Origin, over: Partial<WorkspaceLockEnv> = {}): Page {
  const page = {
    channel: null,
    reload: vi.fn<() => void>(),
    flush: vi.fn<() => Promise<void>>(async () => {}),
  } as unknown as Page;
  const bus = origin.bus;
  page.lock = createWorkspaceLock({
    locks: origin.locks,
    openChannel: bus ? (name) => (page.channel = bus.open(name)) : null,
    reload: () => page.reload(),
    flush: () => page.flush(),
    ...over,
  });
  openPages.push(page);
  return page;
}

/** The tab is closed: its channel goes, and the lock it held is freed. */
function closePage(page: Page): void {
  page.lock.dispose();
}

/** A flush the test finishes by hand, so the order of events can be read. */
function deferredFlush(log: string[]): { flush: () => Promise<void>; finish: () => void } {
  let finish: () => void = () => {
    throw new Error('flush has not started');
  };
  return {
    flush: () =>
      new Promise<void>((resolve) => {
        log.push('flush started');
        finish = () => {
          log.push('flush finished');
          resolve();
        };
      }),
    finish: () => finish(),
  };
}

/**
 * A page whose main thread is busy hears what the lock manager did to it (a
 * steal) only once it is free again, the way its channel's messages wait for
 * it (see `FakeChannel.pause`).
 */
function busyLocks(locks: FakeLocks): { locks: LockManagerLike; wake: () => void } {
  let wake: () => void = () => {};
  const awake = new Promise<void>((resolve) => {
    wake = resolve;
  });
  return {
    locks: {
      request: (name, options, callback) =>
        locks.request(name, options, callback).then(
          async (value) => {
            await awake;
            return value;
          },
          async (error: unknown) => {
            await awake;
            throw error;
          },
        ),
    },
    wake: () => wake(),
  };
}

/** The editor record on one origin's disk, for pages that fence their writes. */
function editorRecord(): { stored: () => string | undefined; env: Partial<WorkspaceLockEnv> } {
  let epoch: string | undefined;
  return {
    stored: () => epoch,
    env: {
      claimEpoch: async (token) => {
        epoch = token;
      },
      readEpoch: async () => epoch,
    },
  };
}

/** Grants every request at once: for cases with a single page. */
const grantAll: LockManagerLike = {
  request: (_name, _options, callback) =>
    Promise.resolve().then(() => callback({ name: 'workspace' })),
};

afterEach(() => {
  for (const page of openPages) page.lock.dispose();
  openPages.length = 0;
  _resetWorkspaceLockForTests();
  vi.useRealTimers();
});

// ── Web Locks: the normal case, any page served from localhost or https ──

describe('workspace lock - Web Locks', () => {
  it('lets the first page edit and opens a second one read-only', async () => {
    const origin = newOrigin();
    const a = openPage(origin);
    expect(a.lock.backend).toBe('locks');
    expect(await a.lock.decided()).toBe('editor');

    const b = openPage(origin);
    expect(await b.lock.decided()).toBe('readonly');
    expect(a.lock.getRole()).toBe('editor');
  });

  it('Edit here: the editor saves, stops writing, and lets go before the new page reloads', async () => {
    const origin = newOrigin();
    const log: string[] = [];
    const slow = deferredFlush(log);
    const a = openPage(origin, { flush: slow.flush });
    await a.lock.decided();
    const b = openPage(origin, {
      reload: () => log.push(`B reloads while A is ${a.lock.getRole()}`),
    });
    await b.lock.decided();

    b.lock.takeOver();
    expect(b.lock.getRole()).toBe('claiming');
    await vi.waitFor(() => expect(a.lock.getRole()).toBe('releasing'));
    // Still allowed to write while it saves: the pending autosave has to land.
    expect(log).toEqual(['flush started']);

    slow.finish();
    await vi.waitFor(() => expect(log).toHaveLength(3));
    expect(log).toEqual(['flush started', 'flush finished', 'B reloads while A is displaced']);

    // The reloaded page asks again and gets it.
    const reloaded = openPage(origin);
    expect(await reloaded.lock.decided()).toBe('editor');
    expect(a.lock.getRole()).toBe('displaced');
  });

  it('lets a displaced page take editing back the same way', async () => {
    const origin = newOrigin();
    const a = openPage(origin);
    await a.lock.decided();
    const b = openPage(origin);
    await b.lock.decided();
    b.lock.takeOver();
    await vi.waitFor(() => expect(b.reload).toHaveBeenCalledTimes(1));
    const b2 = openPage(origin);
    await b2.lock.decided();

    a.lock.takeOver();
    await vi.waitFor(() => expect(a.reload).toHaveBeenCalledTimes(1));
    expect(b2.flush).toHaveBeenCalledTimes(1);
    expect(b2.lock.getRole()).toBe('displaced');
  });

  it('refuses writes past its deadline even before the deadline timer has run', async () => {
    // A timer can fire late -- a busy page, a throttled background tab -- and
    // must not stretch the window: the clock decides.
    vi.useFakeTimers();
    const origin = newOrigin();
    const a = openPage(origin, { flush: () => new Promise<void>(() => {}) });
    await a.lock.decided();
    const b = openPage(origin);
    await b.lock.decided();
    b.lock.takeOver();
    await vi.advanceTimersByTimeAsync(0);
    expect(a.lock.getRole()).toBe('releasing');
    expect(a.lock.canWrite()).toBe(true);

    vi.setSystemTime(Date.now() + 9_100);
    expect(a.lock.getRole()).toBe('releasing');
    expect(a.lock.canWrite()).toBe(false);
  });

  it('takes over at once when the editing page has closed', async () => {
    const origin = newOrigin();
    const a = openPage(origin);
    await a.lock.decided();
    const b = openPage(origin);
    await b.lock.decided();

    closePage(a);
    b.lock.takeOver();
    await vi.waitFor(() => expect(b.reload).toHaveBeenCalledTimes(1));
    expect(a.flush).not.toHaveBeenCalled();
  });

  it('takes over anyway when the editor does not answer in time', async () => {
    vi.useFakeTimers();
    const origin = newOrigin();
    // A page that is stuck: it hears neither the request nor, later, the steal.
    const stuck = busyLocks(origin.locks!);
    const a = openPage(origin, { locks: stuck.locks });
    await a.lock.decided();
    const b = openPage(origin);
    await b.lock.decided();
    a.channel!.pause();

    b.lock.takeOver();
    await vi.advanceTimersByTimeAsync(9_900);
    expect(b.reload).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(100);
    const requests = origin.locks!.requests;
    expect(requests[requests.length - 1]).toEqual({ ifAvailable: undefined, steal: true });
    expect(b.reload).toHaveBeenCalledTimes(1);

    // When A is free again the steal reaches it, and it stops writing.
    stuck.wake();
    await vi.advanceTimersByTimeAsync(0);
    expect(a.lock.getRole()).toBe('displaced');
  });

  it('stops writing at its deadline when its save never finishes, and lets go in time', async () => {
    vi.useFakeTimers();
    const origin = newOrigin();
    const a = openPage(origin, { flush: () => new Promise<void>(() => {}) });
    await a.lock.decided();
    const b = openPage(origin);
    await b.lock.decided();

    b.lock.takeOver();
    await vi.advanceTimersByTimeAsync(0);
    expect(a.lock.getRole()).toBe('releasing');
    expect(a.lock.canWrite()).toBe(true);
    await vi.advanceTimersByTimeAsync(8_900);
    expect(a.lock.canWrite()).toBe(true);
    expect(b.reload).not.toHaveBeenCalled();

    // A second short of B's patience: A stops writing and lets go, so B is
    // granted the lock rather than having to steal it.
    await vi.advanceTimersByTimeAsync(200);
    expect(a.lock.getRole()).toBe('displaced');
    expect(a.lock.canWrite()).toBe(false);
    expect(b.reload).toHaveBeenCalledTimes(1);
    expect(origin.locks!.requests.some((request) => request.steal)).toBe(false);
  });

  it('asks for the lock once when React StrictMode starts it twice', async () => {
    const origin = newOrigin();
    const env = { locks: origin.locks, openChannel: origin.bus!.open };
    const first = startWorkspaceLock(env);
    const second = startWorkspaceLock(env);
    expect(second).toBe(first);
    expect(getWorkspaceLock()).toBe(first);
    expect(await first.decided()).toBe('editor');
    expect(origin.locks!.requests).toHaveLength(1);
  });

  it('keeps editing as before when the lock API refuses outright', async () => {
    const refusing: LockManagerLike = {
      request: () => Promise.reject(new DOMException('opaque origin', 'SecurityError')),
    };
    const page = createWorkspaceLock({ locks: refusing, openChannel: null });
    expect(await page.decided()).toBe('editor');
  });
});

describe('workspace lock - no Web Locks and no BroadcastChannel', () => {
  it('edits as before, with nothing to coordinate', async () => {
    const page = createWorkspaceLock({ locks: null, openChannel: null });
    expect(page.backend).toBe('none');
    expect(page.getRole()).toBe('editor');
    expect(await page.decided()).toBe('editor');
  });
});

// ── BroadcastChannel: a page served over plain http, where Web Locks is withheld ──

describe('workspace lock - BroadcastChannel fallback', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  it('lets the only page edit after a short wait, and opens the next one read-only', async () => {
    const origin = newOrigin({ locks: false });
    const a = openPage(origin);
    expect(a.lock.backend).toBe('channel');
    expect(a.lock.getRole()).toBe('pending');
    await vi.advanceTimersByTimeAsync(600);
    expect(a.lock.getRole()).toBe('editor');

    const b = openPage(origin);
    // The editor answers at once; B does not wait out its window.
    await vi.advanceTimersByTimeAsync(0);
    expect(b.lock.getRole()).toBe('readonly');
  });

  it('agrees on one editor when two pages open together', async () => {
    const origin = newOrigin({ locks: false });
    const a = openPage(origin);
    const b = openPage(origin);
    await vi.advanceTimersByTimeAsync(600);
    expect([a.lock.getRole(), b.lock.getRole()].sort()).toEqual(['editor', 'readonly']);
  });

  it('Edit here: the editor saves, says it let go, and the new page reloads', async () => {
    const origin = newOrigin({ locks: false });
    const log: string[] = [];
    const slow = deferredFlush(log);
    const a = openPage(origin, { flush: slow.flush });
    await vi.advanceTimersByTimeAsync(600);
    const b = openPage(origin);
    await vi.advanceTimersByTimeAsync(0);

    b.lock.takeOver();
    await vi.advanceTimersByTimeAsync(0);
    expect(a.lock.getRole()).toBe('releasing');
    // Longer than the wait for an answer: A answered, so B keeps waiting.
    await vi.advanceTimersByTimeAsync(5_000);
    expect(b.reload).not.toHaveBeenCalled();

    slow.finish();
    await vi.advanceTimersByTimeAsync(0);
    expect(a.lock.getRole()).toBe('displaced');
    expect(b.reload).toHaveBeenCalledTimes(1);

    // The reloaded page finds nobody editing and takes it.
    const reloaded = openPage(origin);
    await vi.advanceTimersByTimeAsync(600);
    expect(reloaded.lock.getRole()).toBe('editor');
    expect(a.lock.getRole()).toBe('displaced');
  });

  it('reloads after a short wait when nobody is editing any more', async () => {
    const origin = newOrigin({ locks: false });
    const a = openPage(origin);
    await vi.advanceTimersByTimeAsync(600);
    const b = openPage(origin);
    await vi.advanceTimersByTimeAsync(0);
    closePage(a);

    b.lock.takeOver();
    await vi.advanceTimersByTimeAsync(999);
    expect(b.reload).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(b.reload).toHaveBeenCalledTimes(1);
  });

  it('without an editor record, when a busy editor missed a new page, the page open longer keeps editing', async () => {
    const origin = newOrigin({ locks: false });
    const a = openPage(origin);
    await vi.advanceTimersByTimeAsync(600);
    expect(a.lock.getRole()).toBe('editor');

    // A is busy and cannot answer, so B hears nobody and claims it too.
    a.channel!.pause();
    await vi.advanceTimersByTimeAsync(1_000);
    const b = openPage(origin);
    await vi.advanceTimersByTimeAsync(600);
    expect(b.lock.getRole()).toBe('editor');

    a.channel!.resume();
    await vi.advanceTimersByTimeAsync(0);
    expect(a.lock.getRole()).toBe('editor');
    // Read-only, not "editing moved": nobody asked B to hand anything over.
    expect(b.lock.getRole()).toBe('readonly');
    // B just opened and holds nothing A lacks; saving it would overwrite A.
    expect(b.flush).not.toHaveBeenCalled();
  });

  it('with an editor record, when a busy editor missed a new page, the page that claimed last keeps editing', async () => {
    const origin = newOrigin({ locks: false });
    const record = editorRecord();
    const a = openPage(origin, record.env);
    await vi.advanceTimersByTimeAsync(600);
    expect(a.lock.getRole()).toBe('editor');
    expect(record.stored()).toBe(a.lock.getEpoch());

    a.channel!.pause();
    await vi.advanceTimersByTimeAsync(1_000);
    const b = openPage(origin, record.env);
    await vi.advanceTimersByTimeAsync(600);
    expect(b.lock.getRole()).toBe('editor');
    expect(record.stored()).toBe(b.lock.getEpoch());

    // A wakes. The record names B, and the fence refuses A's writes from here
    // on; keeping A as the editor would leave no page able to save.
    a.channel!.resume();
    await vi.advanceTimersByTimeAsync(0);
    expect(a.lock.getRole()).toBe('displaced');
    expect(b.lock.getRole()).toBe('editor');
    expect(a.flush).not.toHaveBeenCalled();
  });
});

// ── An "Edit here" that reaches the editor after the asker stopped waiting ──

describe('workspace lock - a request read too late', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  it('BroadcastChannel: an editor that reads it after the asker reloaded stops without saving', async () => {
    // The review's probe (#554): saving here would land after the new editor
    // started, over its work.
    const origin = newOrigin({ locks: false });
    const a = openPage(origin);
    await vi.advanceTimersByTimeAsync(600);
    const b = openPage(origin);
    await vi.advanceTimersByTimeAsync(0);
    expect([a.lock.getRole(), b.lock.getRole()]).toEqual(['editor', 'readonly']);

    // A's main thread is blocked: a long task, or a frozen background tab.
    a.channel!.pause();
    b.lock.takeOver();
    await vi.advanceTimersByTimeAsync(1_000);
    expect(b.reload).toHaveBeenCalledTimes(1);
    const b2 = openPage(origin);
    await vi.advanceTimersByTimeAsync(600);
    expect(b2.lock.getRole()).toBe('editor');

    // A wakes and reads its queue: B's old request first.
    a.channel!.resume();
    await vi.advanceTimersByTimeAsync(0);
    expect(a.flush).not.toHaveBeenCalled();
    expect(a.lock.getRole()).toBe('displaced');
    expect(a.lock.canWrite()).toBe(false);
    expect(b2.lock.getRole()).toBe('editor');
  });

  it('Web Locks: an editor that reads it after being overtaken stops without saving', async () => {
    const origin = newOrigin();
    const stuck = busyLocks(origin.locks!);
    const a = openPage(origin, { locks: stuck.locks });
    await a.lock.decided();
    const b = openPage(origin);
    await b.lock.decided();

    a.channel!.pause();
    b.lock.takeOver();
    await vi.advanceTimersByTimeAsync(10_000);
    expect(b.reload).toHaveBeenCalledTimes(1);
    const b2 = openPage(origin);
    expect(await b2.lock.decided()).toBe('editor');

    // A wakes and reads B's old request before it hears about the steal.
    a.channel!.resume();
    await vi.advanceTimersByTimeAsync(0);
    expect(a.flush).not.toHaveBeenCalled();
    expect(a.lock.getRole()).toBe('displaced');
    expect(a.lock.canWrite()).toBe(false);

    stuck.wake();
    await vi.advanceTimersByTimeAsync(0);
    expect(a.lock.getRole()).toBe('displaced');
    expect(b2.lock.getRole()).toBe('editor');
  });
});

// ── The editor record that fences the writes ──

describe('workspace lock - the editor record', () => {
  it('Web Locks: the editor claims it before editing; a read-only page does not', async () => {
    const origin = newOrigin();
    const record = editorRecord();
    const a = openPage(origin, record.env);
    expect(await a.lock.decided()).toBe('editor');
    expect(a.lock.getEpoch()).not.toBeNull();
    expect(record.stored()).toBe(a.lock.getEpoch());

    const b = openPage(origin, record.env);
    expect(await b.lock.decided()).toBe('readonly');
    expect(b.lock.getEpoch()).toBeNull();
    expect(record.stored()).toBe(a.lock.getEpoch());

    // After Edit here, the reloaded page claims it afresh.
    b.lock.takeOver();
    await vi.waitFor(() => expect(b.reload).toHaveBeenCalledTimes(1));
    const b2 = openPage(origin, record.env);
    expect(await b2.lock.decided()).toBe('editor');
    expect(record.stored()).toBe(b2.lock.getEpoch());
    expect(b2.lock.getEpoch()).not.toBe(a.lock.getEpoch());
  });

  it('is not the editor until the record is claimed', async () => {
    let finishClaim: () => void = () => {};
    const page = createWorkspaceLock({
      locks: grantAll,
      openChannel: null,
      claimEpoch: () =>
        new Promise<void>((resolve) => {
          finishClaim = resolve;
        }),
    });
    openPages.push({ lock: page } as Page);
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(page.getRole()).toBe('pending');
    finishClaim();
    expect(await page.decided()).toBe('editor');
  });

  it('edits unfenced, once a few seconds pass, when the claim never answers', async () => {
    // A hung IndexedDB must not leave the page undecided: no overlay, and
    // every save held waiting for an answer that never comes.
    vi.useFakeTimers();
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const page = createWorkspaceLock({
        locks: grantAll,
        openChannel: null,
        claimEpoch: () => new Promise<void>(() => {}),
      });
      openPages.push({ lock: page } as Page);
      await vi.advanceTimersByTimeAsync(2_900);
      expect(page.getRole()).toBe('pending');

      await vi.advanceTimersByTimeAsync(200);
      expect(page.getRole()).toBe('editor');
      expect(page.getEpoch()).toBeNull();
      expect(page.canWrite()).toBe(true);
      expect(warn).toHaveBeenCalledTimes(1);

      await vi.advanceTimersByTimeAsync(60_000);
      expect(page.getRole()).toBe('editor');
      expect(warn).toHaveBeenCalledTimes(1);
    } finally {
      warn.mockRestore();
    }
  });

  it('keeps editing when a claim it stopped waiting for lands after all', async () => {
    vi.useFakeTimers();
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      let land: () => void = () => {};
      const page = createWorkspaceLock({
        locks: grantAll,
        openChannel: null,
        claimEpoch: () =>
          new Promise<void>((resolve) => {
            land = resolve;
          }),
      });
      openPages.push({ lock: page } as Page);
      await vi.advanceTimersByTimeAsync(3_000);
      expect(page.getRole()).toBe('editor');
      expect(page.getEpoch()).toBeNull();

      // The record now holds this page's token: the role stays, and the
      // token is taken up, so the page's writes are fenced again.
      land();
      await vi.advanceTimersByTimeAsync(0);
      expect(page.getRole()).toBe('editor');
      expect(page.getEpoch()).not.toBeNull();
    } finally {
      warn.mockRestore();
    }
  });

  it('edits unfenced when the record cannot be claimed', async () => {
    const page = createWorkspaceLock({
      locks: grantAll,
      openChannel: null,
      claimEpoch: () => Promise.reject(new Error('disk refused')),
    });
    openPages.push({ lock: page } as Page);
    expect(await page.decided()).toBe('editor');
    expect(page.getEpoch()).toBeNull();
  });

  it('stops editing when told its writes were fenced out', async () => {
    const page = startWorkspaceLock({
      locks: grantAll,
      openChannel: null,
      claimEpoch: async () => {},
    });
    expect(await page.decided()).toBe('editor');
    noteWorkspaceSuperseded();
    expect(page.getRole()).toBe('displaced');
    expect(workspaceWriteAllowed()).toBe(false);
  });
});

// ── The verdict the autosave asks for ──

describe('workspace lock - write verdict', () => {
  it('allows writes when no lock was ever started', () => {
    expect(getWorkspaceLock()).toBeNull();
    expect(workspaceWriteAllowed()).toBe(true);
  });

  it('holds a write while the page is still asking, then allows it for the editor', async () => {
    let grant: () => void = () => {};
    const slow: LockManagerLike = {
      request: (_name, _options, callback) =>
        new Promise((resolve) => {
          grant = () => resolve(callback({ name: 'workspace' }));
        }),
    };
    startWorkspaceLock({ locks: slow, openChannel: null });
    const verdict = workspaceWriteAllowed();
    expect(verdict).toBeInstanceOf(Promise);
    grant();
    expect(await verdict).toBe(true);
    expect(workspaceWriteAllowed()).toBe(true);
  });

  it('refuses writes in a read-only page', async () => {
    const origin = newOrigin();
    await openPage(origin).lock.decided();
    const page = startWorkspaceLock({ locks: origin.locks, openChannel: origin.bus!.open });
    await page.decided();
    expect(workspaceWriteAllowed()).toBe(false);
  });

  it('allows writes while saving for a handover, and refuses them after', async () => {
    const origin = newOrigin();
    const log: string[] = [];
    const slow = deferredFlush(log);
    const page = startWorkspaceLock({
      locks: origin.locks,
      openChannel: origin.bus!.open,
      reload: () => {},
      flush: slow.flush,
    });
    await page.decided();
    const other = openPage(origin);
    await other.lock.decided();

    other.lock.takeOver();
    await vi.waitFor(() => expect(page.getRole()).toBe('releasing'));
    expect(workspaceWriteAllowed()).toBe(true);
    slow.finish();
    await vi.waitFor(() => expect(page.getRole()).toBe('displaced'));
    expect(workspaceWriteAllowed()).toBe(false);
  });
});

// ── Two real pages: the stores, the autosave and one IndexedDB (#554) ──

type LockModule = typeof import('./workspaceLock');
type StoreModule = typeof import('../store/tabStore');
type PersistenceModule = typeof import('../store/tabPersistence');

interface AppPage {
  lock: WorkspaceLock;
  store: StoreModule;
  reload: Mock<() => void>;
}

const SCOPE = 'codefyui-tabs';

describe('two pages of the app on one workspace (#554)', () => {
  let observer: PersistenceModule;
  let warn: MockInstance<typeof console.warn>;

  beforeEach(async () => {
    localStorage.clear();
    vi.stubGlobal('indexedDB', new IDBFactory());
    vi.stubGlobal('IDBKeyRange', IDBKeyRange);
    // A third reader of the same database, so looking does not disturb the
    // durability bookkeeping of either page.
    vi.resetModules();
    observer = await import('../store/tabPersistence');
    warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    warn.mockRestore();
    vi.unstubAllGlobals();
    localStorage.clear();
  });

  async function savedTabNames(): Promise<string[]> {
    observer._resetTabPersistenceForTests();
    const snapshot = await observer.readSnapshot(SCOPE);
    return snapshot ? snapshot.tabs.map((t) => t.name) : [];
  }

  /**
   * A fresh module registry is a fresh page: its own stores, its own
   * IndexedDB connection, its own lock. The lock starts before the store is
   * imported, which is the order App.tsx gets for free: the store's
   * import-time hydration cannot write ahead of it.
   */
  async function loadAppPage(origin: Origin, onReload: () => void = () => {}): Promise<AppPage> {
    vi.resetModules();
    const lockModule: LockModule = await import('./workspaceLock');
    let settle: () => Promise<void> = async () => {};
    const reload = vi.fn<() => void>(onReload);
    const lock = lockModule.startWorkspaceLock({
      locks: origin.locks,
      openChannel: origin.bus!.open,
      reload,
      flush: () => settle(),
    });
    const store: StoreModule = await import('../store/tabStore');
    settle = (await import('../components/WorkspaceLock/settleAutosave')).settleAutosave;
    await store.whenTabsHydrated();
    await lock.decided();
    return { lock, store, reload };
  }

  it("keeps the editing page's work when a stale page acts after a takeover", async () => {
    const origin = newOrigin();
    const a = await loadAppPage(origin);
    expect(a.lock.getRole()).toBe('editor');
    await vi.waitFor(async () => expect(await savedTabNames()).toEqual(['Tab 1']));

    // A second page opens read-only, and nothing it does reaches the disk.
    let atReload: Promise<string[]> | null = null;
    const b = await loadAppPage(origin, () => {
      atReload = savedTabNames();
    });
    expect(b.lock.getRole()).toBe('readonly');
    b.store.useTabStore.getState().addTab('from read-only B');
    await new Promise((resolve) => setTimeout(resolve, 400));
    expect(await savedTabNames()).toEqual(['Tab 1']);
    expect(warn).toHaveBeenCalledTimes(1);

    // A's last edit is still inside the autosave debounce when B asks to edit.
    a.store.useTabStore.getState().addTab('last edit in A');
    b.lock.takeOver();
    await vi.waitFor(() => expect(b.reload).toHaveBeenCalledTimes(1), { timeout: 4_000 });
    expect(await atReload).toEqual(['Tab 1', 'last edit in A']);
    expect(a.lock.getRole()).toBe('displaced');

    // The reloaded page edits, starting from what A saved.
    const b2 = await loadAppPage(origin);
    expect(b2.lock.getRole()).toBe('editor');
    expect(b2.store.useTabStore.getState().tabs.map((t) => t.name)).toEqual([
      'Tab 1',
      'last edit in A',
    ]);
    b2.store.useTabStore.getState().addTab('part 2');
    await vi.waitFor(async () =>
      expect(await savedTabNames()).toEqual(['Tab 1', 'last edit in A', 'part 2']),
    );

    // The issue's probe: the stale page clicks and adds a tab. Neither lands.
    const stale = a.store.useTabStore.getState();
    stale.setActiveTab(stale.tabs[0].id);
    stale.addTab('scratch');
    await new Promise((resolve) => setTimeout(resolve, 400));
    expect(await savedTabNames()).toEqual(['Tab 1', 'last edit in A', 'part 2']);
  });
});
