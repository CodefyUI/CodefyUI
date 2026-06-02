import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { ExecutionWebSocket, executionWs } from './ws';
import { _setSessionTokenForTesting } from './_auth';
import { useToastStore } from '../store/toastStore';
import type { ToastType } from '../store/toastStore';

// ── Fake WebSocket ────────────────────────────────────────────────────────
// jsdom has no WebSocket. This stand-in records every instance and exposes the
// event handlers so tests can drive open / message / close / error manually.
class FakeWS {
  static instances: FakeWS[] = [];
  static OPEN = 1;
  static CLOSED = 3;
  url: string;
  readyState = 0; // CONNECTING
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  send = vi.fn();
  close = vi.fn(() => {
    this.readyState = FakeWS.CLOSED;
  });
  constructor(url: string) {
    this.url = url;
    FakeWS.instances.push(this);
  }
  // Test helpers
  fireOpen() {
    this.readyState = FakeWS.OPEN;
    this.onopen?.();
  }
  fireMessage(data: unknown) {
    this.onmessage?.({ data });
  }
  fireClose() {
    this.readyState = FakeWS.CLOSED;
    this.onclose?.();
  }
  fireError() {
    this.onerror?.();
  }
}

const g = globalThis as unknown as { WebSocket: unknown };
let originalWebSocket: unknown;
let addToastSpy: ReturnType<typeof vi.spyOn>;

/**
 * Call connect() and wait for the awaited wsUrlWithToken() microtasks to
 * settle so the FakeWS instance has been created and its onmessage/onclose
 * handlers are attached. Returns the connect() promise (still pending until
 * the caller fires open or error) and the freshly created socket.
 */
async function startConnect(ws: ExecutionWebSocket) {
  const promise = ws.connect();
  // Flush the microtask queue a few times: getSessionToken resolves, then the
  // body after `await wsUrlWithToken(...)` runs and constructs the socket.
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
  const socket = FakeWS.instances[FakeWS.instances.length - 1];
  return { promise, socket };
}

beforeEach(() => {
  originalWebSocket = g.WebSocket;
  g.WebSocket = FakeWS as unknown as typeof WebSocket;
  FakeWS.instances = [];
  // Seed the token so wsUrlWithToken() resolves without hitting the network.
  _setSessionTokenForTesting('ws-test-token');
  // Spy on the toast store so we can assert connection toasts without rendering.
  addToastSpy = vi.spyOn(useToastStore.getState(), 'addToast');
  vi.useFakeTimers();
});

afterEach(() => {
  vi.runOnlyPendingTimers();
  vi.useRealTimers();
  g.WebSocket = originalWebSocket;
  _setSessionTokenForTesting(null);
  addToastSpy.mockRestore();
  vi.restoreAllMocks();
});

describe('connect', () => {
  it('opens a socket at the token-bearing URL and resolves on onopen', async () => {
    const ws = new ExecutionWebSocket();
    const { promise, socket } = await startConnect(ws);
    expect(socket).toBeDefined();
    expect(socket.url).toContain('/ws/execution');
    expect(socket.url).toContain('token=ws-test-token');

    socket.fireOpen();
    await expect(promise).resolves.toBeUndefined();
    expect(ws.connected).toBe(true);
  });

  it('rejects when onerror fires before open', async () => {
    const ws = new ExecutionWebSocket();
    const { promise, socket } = await startConnect(ws);
    socket.fireError();
    await expect(promise).rejects.toThrow(/WebSocket connection failed/);
  });

  it('clears a pending reconnect timer when connect() is called again', async () => {
    const ws = new ExecutionWebSocket();
    // First connect + open so hasBeenConnected becomes true.
    const first = await startConnect(ws);
    first.socket.fireOpen();
    await first.promise;

    // Drop the connection → schedules a reconnect timer.
    first.socket.fireClose();
    const clearSpy = vi.spyOn(globalThis, 'clearTimeout');

    // Manually reconnecting should clear that pending timer.
    const second = ws.connect();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    expect(clearSpy).toHaveBeenCalled();

    const socket2 = FakeWS.instances[FakeWS.instances.length - 1];
    socket2.fireOpen();
    await second;
    clearSpy.mockRestore();
  });
});

describe('message dispatch', () => {
  it('routes a message to type-specific handlers and wildcard handlers', async () => {
    const ws = new ExecutionWebSocket();
    const typed = vi.fn();
    const wildcard = vi.fn();
    ws.on('progress', typed);
    ws.on('*', wildcard);

    const { socket } = await startConnect(ws);
    const payload = { type: 'progress', value: 0.5 };
    socket.fireMessage(JSON.stringify(payload));

    expect(typed).toHaveBeenCalledWith(payload);
    expect(wildcard).toHaveBeenCalledWith(payload);
  });

  it('invokes only wildcard handlers when no type handler is registered', async () => {
    const ws = new ExecutionWebSocket();
    const wildcard = vi.fn();
    ws.on('*', wildcard);
    const { socket } = await startConnect(ws);
    socket.fireMessage(JSON.stringify({ type: 'unknown_kind' }));
    expect(wildcard).toHaveBeenCalledTimes(1);
  });

  it('logs and swallows malformed JSON without throwing', async () => {
    const ws = new ExecutionWebSocket();
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const handler = vi.fn();
    ws.on('progress', handler);
    const { socket } = await startConnect(ws);

    socket.fireMessage('this is not json{');
    expect(errSpy).toHaveBeenCalledWith(
      'Failed to parse WebSocket message:',
      'this is not json{',
    );
    expect(handler).not.toHaveBeenCalled();
    errSpy.mockRestore();
  });
});

describe('on / off', () => {
  it('removes a previously registered handler so it stops firing', async () => {
    const ws = new ExecutionWebSocket();
    const handler = vi.fn();
    ws.on('evt', handler);
    ws.off('evt', handler);
    const { socket } = await startConnect(ws);
    socket.fireMessage(JSON.stringify({ type: 'evt' }));
    expect(handler).not.toHaveBeenCalled();
  });

  it('off is a no-op for a type that was never registered', () => {
    const ws = new ExecutionWebSocket();
    // Should not throw even though no handler list exists for this type.
    expect(() => ws.off('never', vi.fn())).not.toThrow();
  });

  it('appends a second handler to an existing type without recreating the list', async () => {
    const ws = new ExecutionWebSocket();
    const first = vi.fn();
    const second = vi.fn();
    // The second on('evt', ...) hits the `handlers.has(type)` === true branch.
    ws.on('evt', first);
    ws.on('evt', second);
    const { socket } = await startConnect(ws);
    socket.fireMessage(JSON.stringify({ type: 'evt' }));
    expect(first).toHaveBeenCalledTimes(1);
    expect(second).toHaveBeenCalledTimes(1);
  });
});

describe('send', () => {
  it('serializes and sends when the socket is OPEN', async () => {
    const ws = new ExecutionWebSocket();
    const { socket, promise } = await startConnect(ws);
    socket.fireOpen();
    await promise;

    ws.send({ cmd: 'run' });
    expect(socket.send).toHaveBeenCalledWith(JSON.stringify({ cmd: 'run' }));
  });

  it('warns instead of sending when the socket is not open', async () => {
    const ws = new ExecutionWebSocket();
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    // No socket at all → readyState check short-circuits.
    ws.send({ cmd: 'noop' });
    expect(warnSpy).toHaveBeenCalledWith(
      'WebSocket is not connected. Cannot send:',
      { cmd: 'noop' },
    );
    warnSpy.mockRestore();
  });
});

describe('onclose / reconnect', () => {
  it('does not reconnect on an intentional disconnect', async () => {
    const ws = new ExecutionWebSocket();
    const { socket, promise } = await startConnect(ws);
    socket.fireOpen();
    await promise;

    ws.disconnect(); // sets intentionalClose = true and closes the socket
    expect(socket.close).toHaveBeenCalled();
    // Firing close again (as the browser would) must not schedule a reconnect.
    socket.fireClose();
    vi.runOnlyPendingTimers();
    expect(FakeWS.instances).toHaveLength(1);
  });

  it('does not reconnect if the very first connection never opened', async () => {
    const ws = new ExecutionWebSocket();
    const { socket } = await startConnect(ws);
    // Never fired open → hasBeenConnected stays false.
    socket.fireClose();
    vi.runOnlyPendingTimers();
    // No second socket and no "connection lost" toast.
    expect(FakeWS.instances).toHaveLength(1);
    expect(addToastSpy).not.toHaveBeenCalled();
  });

  it('schedules a reconnect after an established connection drops', async () => {
    const ws = new ExecutionWebSocket();
    const first = await startConnect(ws);
    first.socket.fireOpen();
    await first.promise;

    first.socket.fireClose();
    // A "connection lost" warning toast is shown once.
    expect(addToastSpy).toHaveBeenCalledWith(expect.any(String), 'warning');

    // After the initial 1s backoff, connect() runs and a new socket appears.
    await vi.advanceTimersByTimeAsync(1000);
    expect(FakeWS.instances.length).toBe(2);
  });

  it('swallows a rejected reconnect attempt (handled via onclose)', async () => {
    const ws = new ExecutionWebSocket();
    const first = await startConnect(ws);
    first.socket.fireOpen();
    await first.promise;

    // Drop the established connection → schedules reconnect after 1s.
    first.socket.fireClose();
    await vi.advanceTimersByTimeAsync(1000);
    expect(FakeWS.instances.length).toBe(2);

    // The reconnect's connect() promise rejects (onerror before onopen). The
    // internal `.catch(() => {})` must swallow it — no unhandled rejection.
    const second = FakeWS.instances[1];
    second.fireError();
    // onerror does not throw; the close path drives subsequent retries.
    await vi.advanceTimersByTimeAsync(0);
    expect(ws.connected).toBe(false);
  });

  it('shows a "connection restored" success toast when a dropped link recovers', async () => {
    const ws = new ExecutionWebSocket();
    const first = await startConnect(ws);
    first.socket.fireOpen();
    await first.promise;
    addToastSpy.mockClear();

    first.socket.fireClose(); // warning toast + schedules reconnect
    await vi.advanceTimersByTimeAsync(1000); // reconnect attempt creates socket #2
    const second = FakeWS.instances[1];
    second.fireOpen(); // recovery → success toast

    expect(addToastSpy).toHaveBeenCalledWith(expect.any(String), 'success');
  });

  it('only shows one disconnect toast across repeated backoff ticks', async () => {
    const ws = new ExecutionWebSocket();
    const first = await startConnect(ws);
    first.socket.fireOpen();
    await first.promise;
    addToastSpy.mockClear();

    // Drop → schedules reconnect (#1 attempt, warning toast).
    first.socket.fireClose();
    await vi.advanceTimersByTimeAsync(1000);
    // The reconnect socket also fails to open and closes again.
    const second = FakeWS.instances[1];
    second.fireClose();
    await vi.advanceTimersByTimeAsync(2000);

    const warningToasts = addToastSpy.mock.calls.filter(
      (c: [string, ToastType?]) => c[1] === 'warning',
    );
    expect(warningToasts).toHaveLength(1);
  });

  it('gives up and shows a failure toast after MAX_RECONNECT_ATTEMPTS', async () => {
    const ws = new ExecutionWebSocket();
    const first = await startConnect(ws);
    first.socket.fireOpen();
    await first.promise;
    addToastSpy.mockClear();

    // Repeatedly drop the latest socket; backoff doubles up to a 30s cap.
    // 10 attempts are allowed; the 11th close triggers the failure toast.
    let attempts = 0;
    first.socket.fireClose();
    while (attempts < 12) {
      // Advance well past the max backoff so the scheduled connect() fires.
      await vi.advanceTimersByTimeAsync(30000);
      const latest = FakeWS.instances[FakeWS.instances.length - 1];
      // Each freshly created reconnect socket immediately closes again.
      if (latest && FakeWS.instances.length > attempts + 1) {
        latest.fireClose();
      }
      attempts++;
    }

    const errorToasts = addToastSpy.mock.calls.filter(
      (c: [string, ToastType?]) => c[1] === 'error',
    );
    expect(errorToasts.length).toBeGreaterThanOrEqual(1);
    // No further reconnect sockets are created once we give up: the instance
    // count plateaus at the cap (1 original + 10 reconnect attempts).
    expect(FakeWS.instances.length).toBeLessThanOrEqual(11);
  });
});

describe('disconnect', () => {
  it('clears a pending reconnect timer and resets state', async () => {
    const ws = new ExecutionWebSocket();
    const first = await startConnect(ws);
    first.socket.fireOpen();
    await first.promise;

    first.socket.fireClose(); // schedules a reconnect timer
    const clearSpy = vi.spyOn(globalThis, 'clearTimeout');
    ws.disconnect();
    expect(clearSpy).toHaveBeenCalled();
    expect(ws.connected).toBe(false);
    clearSpy.mockRestore();
  });

  it('is safe to call when no socket was ever created', () => {
    const ws = new ExecutionWebSocket();
    expect(() => ws.disconnect()).not.toThrow();
    expect(ws.connected).toBe(false);
  });
});

describe('connected getter', () => {
  it('is false before connecting and true once the socket is OPEN', async () => {
    const ws = new ExecutionWebSocket();
    expect(ws.connected).toBe(false);
    const { socket, promise } = await startConnect(ws);
    expect(ws.connected).toBe(false); // still CONNECTING
    socket.fireOpen();
    await promise;
    expect(ws.connected).toBe(true);
  });
});

describe('module singleton', () => {
  it('exports a shared ExecutionWebSocket instance', () => {
    expect(executionWs).toBeInstanceOf(ExecutionWebSocket);
  });
});
