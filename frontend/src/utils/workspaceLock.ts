import { generateId } from './ids';
import { idbAvailable, idbGet, idbSet } from './idb';

/**
 * One page of the app edits the workspace at a time (#554).
 *
 * Autosave is last-writer-wins: a page writes its whole tab list over the
 * stored one and sweeps the records it does not name. Two pages open on the
 * same origin -- `cdui start` prints the URL and people open it again, and
 * browsers restore old windows -- means the stale one wins the next time it
 * saves, and the other page's work is gone from disk for good.
 *
 * So one page per origin is the EDITOR and every other page is read-only:
 * `tabPersistence` refuses its writes, and `WorkspaceLockOverlay` covers it
 * and says why. "Edit here" moves editing. The editor saves what its autosave
 * still holds, stops writing, and lets go; the page that asked then reloads,
 * so it starts from what was saved rather than from what it was showing.
 *
 * Who edits is decided by, in order of preference:
 *
 * - Web Locks (`navigator.locks`). The editor holds an exclusive lock for the
 *   page's lifetime, and the browser frees it when the page goes away, crash
 *   included. A BroadcastChannel carries "please let go" to the holder.
 * - BroadcastChannel alone, where Web Locks is withheld: a page served over
 *   plain http from a LAN address is not a secure context. A new page asks
 *   who edits and claims it when nobody answers in time. That cannot be
 *   exact -- an editor too busy to answer is missed -- so when two editors
 *   meet, the editor record (below) says which one stays.
 * - Neither: the page edits, as every page did before this existed.
 *
 * A page can be too busy to hear any of it: a long task, or a background tab
 * the browser froze. It wakes with its autosave due and its role still
 * 'editor'. Two things stop it from writing over the page that took over:
 *
 * - The EDITOR RECORD. A page that becomes the editor first writes a fresh
 *   token under `WORKSPACE_EDITOR_KEY`, and `tabPersistence` checks, inside
 *   each write's own transaction, that the record still holds this page's
 *   token. A write that finds another token writes nothing, and its page
 *   stops editing.
 * - A request read too late is not a request. "Edit here" stamps when it was
 *   sent; an editor that reads it after the asker stopped waiting stops
 *   without saving, and one that reads it in time may save only until a
 *   second before the asker's patience runs out.
 *
 * A read-only page whose editor closes stays read-only until "Edit here" is
 * pressed. Taking over by itself would reload a page someone may be reading,
 * and with several read-only pages open it would have to pick one; a click
 * says which.
 */

const LOCK_NAME = 'codefyui-workspace-editor';
const CHANNEL_NAME = 'codefyui-workspace';

/**
 * The editor record: the token of the page whose writes the disk accepts.
 * One per origin, like the lock -- a project keeps its tabs under a scope of
 * its own, but an origin still has one editor. Outside every tab scope's key
 * prefix (`codefyui-tabs...`), so no prefix scan of tabs reads it.
 */
export const WORKSPACE_EDITOR_KEY = 'codefyui-workspace|editor';

/** BroadcastChannel only: how long a new page listens for an editor before it claims. */
const ELECTION_MS = 600;
/** BroadcastChannel only: how long "Edit here" waits for any editor to answer. */
const ACK_MS = 1_000;
/**
 * How long "Edit here" waits for the editor to save and let go. A live
 * editor takes far less, background tab included: its save only waits out
 * the autosave debounce (see `settleAutosave`). One that has not let go by
 * now is stuck, and is overtaken.
 */
const TAKEOVER_TIMEOUT_MS = 10_000;
/**
 * How much earlier than the asker the editor gives up on a request: room for
 * the message's trip and for timers that fire late. Past `ACK_MS` less this,
 * an answer may arrive after the asker reloaded; past `TAKEOVER_TIMEOUT_MS`
 * less this, a save may land after the asker stole the lock.
 */
const ACK_MARGIN_MS = 250;
const TAKEOVER_MARGIN_MS = 1_000;
/**
 * How long a page waits for its editor-record claim before it edits without
 * one. A hung IndexedDB must not leave the page undecided: no overlay, and
 * every save held waiting for an answer that never comes.
 */
const CLAIM_TIMEOUT_MS = 3_000;

export type WorkspaceRole =
  /** Asked who edits; no answer yet. Writes wait for the answer. */
  | 'pending'
  /** This page writes the workspace. */
  | 'editor'
  /** Another page was editing when this one opened. */
  | 'readonly'
  /** "Edit here" was pressed: waiting for the editor to let go, then reloading. */
  | 'claiming'
  /** Another page asked to edit: saving what the autosave still holds, until a deadline. */
  | 'releasing'
  /** Editing moved to another page. */
  | 'displaced';

export type WorkspaceLockBackend = 'locks' | 'channel' | 'none';

/** The part of `navigator.locks` this module uses. */
export interface LockManagerLike {
  request(
    name: string,
    options: { ifAvailable?: boolean; steal?: boolean; signal?: AbortSignal },
    callback: (lock: unknown) => unknown,
  ): Promise<unknown>;
}

/** The part of a BroadcastChannel this module uses. */
export interface ChannelLike {
  post(message: unknown): void;
  listen(onMessage: (data: unknown) => void): void;
  close(): void;
}

/** What the lock runs against. Every field defaults to the browser's own. */
export interface WorkspaceLockEnv {
  /** `navigator.locks`; null where there is none. */
  locks?: LockManagerLike | null;
  /** Opens the origin's broadcast channel; null where there is none. */
  openChannel?: ((name: string) => ChannelLike) | null;
  /** How "Edit here" ends: the page reloads and asks again. */
  reload?: () => void;
  /** Lands the pending autosave. Awaited before this page lets go of editing. */
  flush?: () => Promise<void>;
  /** Writes this page's token into the editor record; null where there is no disk to fence. */
  claimEpoch?: ((token: string) => Promise<void>) | null;
  /** Reads the editor record. */
  readEpoch?: (() => Promise<string | undefined>) | null;
}

/** This page's claim on the workspace. */
export interface WorkspaceLock {
  readonly backend: WorkspaceLockBackend;
  getRole(): WorkspaceRole;
  /** This page's token in the editor record; null until it edits, or with no record to use. */
  getEpoch(): string | null;
  /** Whether this page may write now: the editor, or releasing within its deadline. */
  canWrite(): boolean;
  /** Called on every role change. Returns the unsubscribe. */
  subscribe(listener: () => void): () => void;
  /** Resolves once the page knows whether it edits. */
  decided(): Promise<WorkspaceRole>;
  /** "Edit here". Does nothing unless the page is read-only or displaced. */
  takeOver(): void;
  /** The editor record names another page: stop editing, without saving. */
  supersede(): void;
  /** Stops listening and lets go, as closing the page would. For tests. */
  dispose(): void;
}

type MessageType = 'hello' | 'editor' | 'release' | 'ack' | 'released';

/**
 * What pages tell each other. `from` and `since` name the sender: `since` is
 * when its page opened, which is what "open longer" compares, and `from`
 * breaks a tie. `at` is when the message was sent, which is what makes a
 * request read too late recognisable. All three pages share one clock.
 */
interface Message {
  type: MessageType;
  from: string;
  since: number;
  at: number;
}

const MESSAGE_TYPES: ReadonlySet<string> = new Set(['hello', 'editor', 'release', 'ack', 'released']);

function readMessage(data: unknown): Message | null {
  if (typeof data !== 'object' || data === null) return null;
  const { type, from, since, at } = data as Record<string, unknown>;
  if (typeof type !== 'string' || !MESSAGE_TYPES.has(type)) return null;
  if (typeof from !== 'string' || typeof since !== 'number' || typeof at !== 'number') return null;
  return { type: type as MessageType, from, since, at };
}

function openedEarlier(
  a: { since: number; from: string },
  b: { since: number; from: string },
): boolean {
  return a.since < b.since || (a.since === b.since && a.from < b.from);
}

function browserLocks(): LockManagerLike | null {
  try {
    const locks = (globalThis.navigator as { locks?: LockManagerLike } | undefined)?.locks;
    return locks && typeof locks.request === 'function' ? locks : null;
  } catch {
    return null;
  }
}

function browserChannel(locks: LockManagerLike | null): ((name: string) => ChannelLike) | null {
  if (typeof BroadcastChannel !== 'function') return null;
  // With Web Locks the channel only carries "Edit here". Without them it IS
  // the mechanism, and it is meant for one case: a page served over plain
  // http, which browsers do not give Web Locks. A secure page without Web
  // Locks is a browser from before 2022 or a test environment such as jsdom,
  // and both keep the old behaviour.
  if (!locks && globalThis.isSecureContext !== false) return null;
  return (name) => {
    const channel = new BroadcastChannel(name);
    return {
      post: (message) => channel.postMessage(message),
      listen: (onMessage) => {
        channel.onmessage = (event) => onMessage(event.data);
      },
      close: () => channel.close(),
    };
  };
}

function lockRequest(
  manager: LockManagerLike,
  options: { ifAvailable?: boolean; steal?: boolean; signal?: AbortSignal },
  callback: (lock: unknown) => unknown,
): Promise<unknown> {
  try {
    return manager.request(LOCK_NAME, options, callback);
  } catch (error) {
    return Promise.reject(error);
  }
}

/** One page's claim. `startWorkspaceLock` makes the page's own; tests make several. */
export function createWorkspaceLock(env: WorkspaceLockEnv = {}): WorkspaceLock {
  const locks = env.locks !== undefined ? env.locks : browserLocks();
  const openChannel = env.openChannel !== undefined ? env.openChannel : browserChannel(locks);
  const reload = env.reload ?? (() => window.location.reload());
  const flush = env.flush ?? (() => Promise.resolve());
  // Without IndexedDB there is no record to fence with, and tabStore saves to
  // localStorage, which this does not reach.
  const disk = idbAvailable();
  const claimEpoch =
    env.claimEpoch !== undefined
      ? env.claimEpoch
      : disk
        ? (token: string) => idbSet(WORKSPACE_EDITOR_KEY, token)
        : null;
  const readEpoch =
    env.readEpoch !== undefined
      ? env.readEpoch
      : disk
        ? () => idbGet<string>(WORKSPACE_EDITOR_KEY)
        : null;

  const self = { from: generateId(), since: Date.now() };
  let channel: ChannelLike | null = null;
  try {
    channel = openChannel ? openChannel(CHANNEL_NAME) : null;
  } catch {
    channel = null;
  }
  const backend: WorkspaceLockBackend = locks ? 'locks' : channel ? 'channel' : 'none';

  let role: WorkspaceRole = backend === 'none' ? 'editor' : 'pending';
  const listeners = new Set<() => void>();
  let markDecided: (decided: WorkspaceRole) => void = () => {};
  const decided = new Promise<WorkspaceRole>((resolve) => {
    markDecided = resolve;
  });
  if (role !== 'pending') markDecided(role);

  const timers = new Set<ReturnType<typeof setTimeout>>();
  // This page's token in the editor record, once it has claimed it.
  let epoch: string | null = null;
  // While releasing: writes are allowed until then, and not after.
  let releaseDeadline = 0;
  // Ends the Web Lock this page holds as editor.
  let letGo: (() => void) | null = null;
  let disposed = false;
  let reloading = false;
  // BroadcastChannel only.
  let electionTimer: ReturnType<typeof setTimeout> | null = null;
  let deferToEarlier = false;
  let claimStarted = false;
  let answered = false;

  function setRole(next: WorkspaceRole): void {
    if (disposed || role === next) return;
    role = next;
    if (next !== 'pending') markDecided(next);
    for (const listener of [...listeners]) listener();
  }

  function canWrite(): boolean {
    if (role === 'editor') return true;
    return role === 'releasing' && Date.now() < releaseDeadline;
  }

  function after(ms: number, run: () => void): ReturnType<typeof setTimeout> {
    const timer = setTimeout(() => {
      timers.delete(timer);
      run();
    }, ms);
    timers.add(timer);
    return timer;
  }

  function cancel(timer: ReturnType<typeof setTimeout> | null): void {
    if (timer === null) return;
    clearTimeout(timer);
    timers.delete(timer);
  }

  function clearTimers(): void {
    for (const timer of timers) clearTimeout(timer);
    timers.clear();
    electionTimer = null;
  }

  function post(type: MessageType): void {
    if (channel === null || disposed) return;
    try {
      channel.post({ type, ...self, at: Date.now() });
    } catch {
      // Closed under us: the page is going away, and nobody needs telling.
    }
  }

  function reloadOnce(): void {
    if (reloading || disposed) return;
    reloading = true;
    clearTimers();
    reload();
  }

  /**
   * Start editing: claim the record first, so this page's first write passes
   * the fence and the previous editor's next one does not. Resolves whether
   * the page is now the editor -- it is not if it was closed, or its lock
   * stolen, while the claim was on its way.
   */
  async function becomeEditor(): Promise<boolean> {
    if (claimEpoch) {
      const token = generateId();
      // No record means no fence for this page: it writes as pages did
      // before the record existed.
      epoch = (await claimWithin(claimEpoch, token)) ? token : null;
    }
    if (disposed || role !== 'pending') return false;
    setRole('editor');
    return true;
  }

  /**
   * Resolves whether the claim landed within `CLAIM_TIMEOUT_MS`. One that
   * lands later leaves the role alone; it only hands the page its token, if
   * the page still edits, so its writes are fenced again from then on.
   */
  function claimWithin(
    claim: (token: string) => Promise<void>,
    token: string,
  ): Promise<boolean> {
    return new Promise<boolean>((resolve) => {
      let waiting = true;
      const timer = after(CLAIM_TIMEOUT_MS, () => {
        waiting = false;
        console.warn(
          '[CodefyUI] Editing without the editor record: IndexedDB did not answer in time (#554).',
        );
        resolve(false);
      });
      void Promise.resolve()
        .then(() => claim(token))
        .then(
          () => {
            if (waiting) {
              waiting = false;
              cancel(timer);
              resolve(true);
            } else if (role === 'editor' && epoch === null) {
              epoch = token;
            }
          },
          () => {
            if (!waiting) return;
            waiting = false;
            cancel(timer);
            resolve(false);
          },
        );
    });
  }

  /** Stop writing, then let go. `announce` tells a waiting asker. */
  function stopEditing(announce: boolean): void {
    setRole('displaced');
    releaseDeadline = 0;
    const end = letGo;
    letGo = null;
    end?.();
    if (announce) post('released');
  }

  /**
   * Another page asked to edit, and asked in time: land the pending autosave,
   * then stop. The save is bounded by `deadline`, a margin short of the
   * asker's patience; past it nothing this page writes is accepted, and it
   * lets go whether or not the save finished.
   */
  function release(deadline: number): void {
    if (role !== 'editor') return;
    releaseDeadline = deadline;
    setRole('releasing');
    let finished = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const finish = () => {
      if (finished) return;
      finished = true;
      cancel(timer);
      // Stop writing BEFORE letting go, so nothing this page writes can land
      // after the next editor has started.
      stopEditing(true);
    };
    timer = after(Math.max(0, deadline - Date.now()), finish);
    void Promise.resolve()
      .then(flush)
      .catch(() => {
        // A failed save raises its own warning; the handover must not hang on it.
      })
      .then(finish);
  }

  function onRelease(message: Message): void {
    if (role !== 'editor') return;
    // When the asker stops waiting: with Web Locks it steals the lock, on the
    // channel it reloads if nobody answered. A request read after that is
    // stale -- the asker, or the page it reloaded into, may already be
    // editing -- so this page stops without saving: a save now would land
    // over the new editor's work.
    const deadline = message.at + TAKEOVER_TIMEOUT_MS - TAKEOVER_MARGIN_MS;
    const answerBy = backend === 'channel' ? message.at + ACK_MS - ACK_MARGIN_MS : deadline;
    if (Date.now() > answerBy) {
      stopEditing(true);
      return;
    }
    post('ack');
    release(deadline);
  }

  function supersede(): void {
    if (role !== 'editor' && role !== 'releasing') return;
    stopEditing(false);
  }

  // ── Web Locks ──

  function claimWithLocks(manager: LockManagerLike): void {
    let held = false;
    lockRequest(manager, { ifAvailable: true }, async (lock) => {
      if (disposed) return undefined;
      if (lock === null) {
        setRole('readonly');
        return undefined;
      }
      held = true;
      // Held for the page's lifetime, unless another page asks for it.
      const holding = new Promise<void>((resolve) => {
        letGo = resolve;
      });
      await becomeEditor();
      if (role !== 'editor') {
        // Closed or stolen while the claim was on its way.
        const end = letGo;
        letGo = null;
        end?.();
      }
      return holding;
    }).catch(() => {
      if (!held) {
        // The API exists but refused, as it does on an opaque origin. With
        // nothing to coordinate through, edit as before.
        setRole('editor');
        return;
      }
      // Stolen: a page that asked to edit gave up waiting (see below).
      letGo = null;
      setRole('displaced');
    });
  }

  function takeOverWithLocks(manager: LockManagerLike): void {
    // Let go before the reload rather than through it: a lock held until the
    // old document is torn down can still be held when the reloaded page
    // asks, and that page would open read-only.
    const reloadAndLetGo = () => {
      reloadOnce();
      return undefined;
    };
    if (channel === null) {
      // Nothing can ask the editor to save first, so this is all there is.
      lockRequest(manager, { steal: true }, reloadAndLetGo).catch(reloadOnce);
      return;
    }
    const giveUp = new AbortController();
    const timer = after(TAKEOVER_TIMEOUT_MS, () => giveUp.abort());
    lockRequest(manager, { signal: giveUp.signal }, () => {
      cancel(timer);
      return reloadAndLetGo();
    }).catch(() => {
      if (disposed) return;
      // The editor did not let go in time: it is hung, or its page is frozen.
      // Taking the lock ends its hold; the editor record keeps its writes out.
      lockRequest(manager, { steal: true }, reloadAndLetGo).catch(reloadOnce);
    });
    // Asked after queueing, so this page is next in line when the editor lets go.
    post('release');
  }

  // ── BroadcastChannel ──

  function elect(): void {
    post('hello');
    electionTimer = after(ELECTION_MS, () => {
      electionTimer = null;
      if (role !== 'pending') return;
      if (deferToEarlier) {
        setRole('readonly');
        return;
      }
      claimStarted = true;
      void becomeEditor().then((editing) => {
        // Announced once the record is claimed: another editor that hears it
        // checks the record, and one of the two stands down.
        if (editing) post('editor');
      });
    });
  }

  function takeOverWithChannel(): void {
    answered = false;
    // No answer means nobody is editing any more.
    after(ACK_MS, () => {
      if (!answered) reloadOnce();
    });
    // An editor that answered and never finished is stuck. The reloaded page
    // asks again, and stays read-only if that editor is still there.
    after(TAKEOVER_TIMEOUT_MS, reloadOnce);
    post('release');
  }

  /** Two editors heard each other: one of them stands down. */
  function settleTwoEditors(other: Message): void {
    const mine = epoch;
    if (readEpoch && mine !== null) {
      // The record decides: it holds the token of the page that claimed last,
      // and only that page's writes pass the fence. Keeping the other one as
      // editor would leave no page able to save.
      void readEpoch().then(
        (stored) => {
          if (role !== 'editor') return;
          if (stored === mine) post('editor');
          else if (stored !== undefined) supersede();
          else if (openedEarlier(other, self)) setRole('readonly');
          else post('editor');
        },
        () => {
          // Unreadable: nothing to decide with this time; the next write's
          // fence, or the next announcement, settles it.
        },
      );
      return;
    }
    // No record to go by: the page open longer keeps editing. The other one
    // stops without saving -- it is the newer page, so what it would save is
    // mostly what it read, written over the work of the page that was
    // editing all along.
    if (openedEarlier(other, self)) setRole('readonly');
    else post('editor');
  }

  function onMessage(data: unknown): void {
    const message = readMessage(data);
    if (message === null || message.from === self.from || disposed) return;
    switch (message.type) {
      case 'release':
        onRelease(message);
        return;
      case 'ack':
        answered = true;
        return;
      case 'released':
        // With Web Locks the lock itself says so.
        if (backend === 'channel' && role === 'claiming') reloadOnce();
        return;
      case 'hello':
        if (backend !== 'channel') return;
        if (role === 'editor') post('editor');
        else if (role === 'pending' && openedEarlier(message, self)) deferToEarlier = true;
        return;
      case 'editor':
        if (backend !== 'channel') return;
        if (role === 'pending') {
          // Once this page has started claiming, its own announcement and the
          // record settle it instead (see `elect`).
          if (claimStarted) return;
          cancel(electionTimer);
          electionTimer = null;
          setRole('readonly');
        } else if (role === 'editor') {
          settleTwoEditors(message);
        }
        return;
    }
  }

  function takeOver(): void {
    if (disposed || (role !== 'readonly' && role !== 'displaced')) return;
    setRole('claiming');
    if (locks) takeOverWithLocks(locks);
    else takeOverWithChannel();
  }

  function dispose(): void {
    if (disposed) return;
    // A page going away writes nothing more; settle anyone waiting to ask.
    markDecided('displaced');
    disposed = true;
    clearTimers();
    try {
      channel?.close();
    } catch {
      // Already closed.
    }
    listeners.clear();
    const end = letGo;
    letGo = null;
    end?.();
  }

  channel?.listen(onMessage);
  if (locks) claimWithLocks(locks);
  else if (backend === 'channel') elect();

  return {
    backend,
    getRole: () => role,
    getEpoch: () => epoch,
    canWrite,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    decided: () => decided,
    takeOver,
    supersede,
    dispose,
  };
}

// ── This page's claim ──

let _page: WorkspaceLock | null = null;

/**
 * Claim the workspace for this page, once; later calls return the same claim,
 * which is what keeps React StrictMode's double start to one.
 *
 * App.tsx calls this at import time rather than from an effect. The tab store
 * starts reading IndexedDB the moment it is imported, and its first autosave
 * can fire before the first effect runs; a page that turns out read-only
 * would already have written by then.
 */
export function startWorkspaceLock(env?: WorkspaceLockEnv): WorkspaceLock {
  _page ??= createWorkspaceLock(env);
  return _page;
}

/** This page's claim, or null when nothing started one (tests, mostly). */
export function getWorkspaceLock(): WorkspaceLock | null {
  return _page;
}

/**
 * Whether this page may write the workspace now. A promise while the page is
 * still asking, so a save made in that window waits for the answer instead of
 * guessing. True when no claim was started, which keeps every caller that
 * never mounts the app (tests, scripts) writing as before.
 */
export function workspaceWriteAllowed(): boolean | Promise<boolean> {
  const page = _page;
  if (page === null) return true;
  if (page.getRole() !== 'pending') return page.canWrite();
  // Read again on arrival: the role can move on between the answer and the write.
  return page.decided().then(() => page.canWrite());
}

/**
 * The token a write must find in the editor record, or null for a write that
 * is not fenced: no claim started, or no record to claim.
 */
export function workspaceEpoch(): string | null {
  return _page?.getEpoch() ?? null;
}

/** A write found the editor record naming another page: this page stops editing. */
export function noteWorkspaceSuperseded(): void {
  _page?.supersede();
}

/** Drop this page's claim. Test-only. */
export function _resetWorkspaceLockForTests(): void {
  _page?.dispose();
  _page = null;
}
