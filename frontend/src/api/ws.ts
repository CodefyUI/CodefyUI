import { useToastStore } from '../store/toastStore';
import { useI18n } from '../i18n';
import { wsUrlWithToken } from './_auth';

type MessageHandler = (data: any) => void;

// Reconnect tunables. Exponential backoff capped at ~30s; we stop after
// MAX_RECONNECT_ATTEMPTS so we don't keep timers alive forever when the
// server is permanently down.
const INITIAL_RECONNECT_DELAY_MS = 1000;
const MAX_RECONNECT_DELAY_MS = 30000;
const MAX_RECONNECT_ATTEMPTS = 10;

/**
 * Synthetic event fired when a *previously established* connection comes
 * back. Not a server frame — the server has no idea a socket is a
 * replacement for an earlier one.
 *
 * Since #121 a run outlives its socket, so a reconnect leaves the browser
 * connected to a server that is no longer forwarding anything: the old
 * socket's subscription died with it. Somebody has to send `attach` again,
 * and only the hook knows which run this tab was watching — hence an event
 * rather than a re-attach baked in here.
 */
export const RECONNECTED_EVENT = 'reconnected';

/**
 * RFC 6455 close code 1009, "message too big".
 *
 * Sent by the server's WebSocket layer when a frame we sent exceeded its
 * ceiling (`CODEFYUI_WS_MAX_MESSAGE_BYTES`, handed to uvicorn as
 * `--ws-max-size`). It is refused *while the fragments are assembled*, so
 * the application never sees the message and cannot answer with the usual
 * `{"type": "error"}` frame — the close code is the only channel the
 * failure has.
 *
 * Verified end to end against a live uvicorn (core#274): the close frame
 * carries 1009 plus a reason, and both survive even a 100 MB overshoot
 * against a 4 MB ceiling. Before this constant existed the editor threw
 * both away and ran the generic reconnect path, so an oversized graph
 * looked exactly like a flaky network — "Connection lost", "Connection
 * restored", and the same silent failure on the next Run click.
 */
const WS_CLOSE_MESSAGE_TOO_BIG = 1009;

export class ExecutionWebSocket {
  private ws: WebSocket | null = null;
  private handlers: Map<string, MessageHandler[]> = new Map();
  // Set when callers explicitly disconnect() — suppresses reconnect loops
  // during teardown / tab close.
  private intentionalClose = false;
  // Set true after the first successful onopen. We only auto-reconnect if a
  // *previously established* connection drops; first-time connect failures
  // bubble up via the connect() Promise so callers can show their own error.
  private hasBeenConnected = false;
  private reconnectAttempt = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  // Toast suppression: we only want one "Connection lost" toast per outage,
  // not one per backoff tick.
  private notifiedDisconnect = false;

  async connect(): Promise<void> {
    this.intentionalClose = false;
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    // Token is appended as ?token=... because browsers cannot set custom
    // headers on WebSocket handshakes. wsUrlWithToken() awaits the bootstrap
    // exchange the first time it's called and caches the value afterwards.
    const url = await wsUrlWithToken('/ws/execution');
    this.ws = new WebSocket(url);

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data as string);
        this.dispatch(data);
      } catch {
        console.error('Failed to parse WebSocket message:', event.data);
      }
    };

    this.ws.onclose = (event?: { code?: number }) => {
      if (this.intentionalClose) return;
      // Don't loop on initial-connect failure — the connect() promise has
      // already rejected and the caller is responsible for surfacing that.
      if (!this.hasBeenConnected) return;
      if (event?.code === WS_CLOSE_MESSAGE_TOO_BIG) {
        useToastStore.getState().addToast(
          useI18n.getState().t('connection.tooLarge'),
          'error',
        );
        // Claim the one-toast-per-outage slot so scheduleReconnect's generic
        // "connection lost" does not immediately contradict the specific
        // reason we just gave. The socket itself is healthy — it was the
        // message that was refused — so we still reconnect, and the user
        // still gets "connection restored" when it comes back.
        this.notifiedDisconnect = true;
      }
      this.scheduleReconnect();
    };

    return new Promise<void>((resolve, reject) => {
      this.ws!.onopen = () => {
        // If we just recovered from a dropped connection, tell the user.
        if (this.notifiedDisconnect) {
          useToastStore.getState().addToast(
            useI18n.getState().t('connection.restored'),
            'success',
          );
          this.notifiedDisconnect = false;
        }
        // A *replacement* socket, not the first one: whatever this tab was
        // watching is still running server-side and needs re-attaching.
        const isReconnect = this.hasBeenConnected;
        this.hasBeenConnected = true;
        this.reconnectAttempt = 0;
        resolve();
        if (isReconnect) this.dispatch({ type: RECONNECTED_EVENT });
      };
      this.ws!.onerror = () => reject(new Error('WebSocket connection failed'));
    });
  }

  /**
   * Route one message (real or synthetic) to its handlers, then to '*'.
   *
   * Each handler is isolated: one that throws must not skip the ones after
   * it. That matters more since #121, because the '*' handler runs last and
   * is what tracks the run cursor — losing it would make a later reconnect
   * replay history the panel has already rendered. It also stops a handler
   * bug from being reported as "Failed to parse WebSocket message" by the
   * caller's catch, which sent every past reader looking at the wrong layer.
   */
  private dispatch(data: any): void {
    const handlers = [
      ...(this.handlers.get(data.type) ?? []),
      ...(this.handlers.get('*') ?? []),
    ];
    for (const handler of handlers) {
      try {
        handler(data);
      } catch (err) {
        console.error('WebSocket handler failed for', data.type, err);
      }
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempt >= MAX_RECONNECT_ATTEMPTS) {
      useToastStore.getState().addToast(
        useI18n.getState().t('connection.failed'),
        'error',
      );
      return;
    }

    // Only one disconnect toast per outage so a flapping server doesn't
    // flood the toast stack.
    if (!this.notifiedDisconnect) {
      useToastStore.getState().addToast(
        useI18n.getState().t('connection.lost'),
        'warning',
      );
      this.notifiedDisconnect = true;
    }

    const delay = Math.min(
      INITIAL_RECONNECT_DELAY_MS * 2 ** this.reconnectAttempt,
      MAX_RECONNECT_DELAY_MS,
    );
    this.reconnectAttempt++;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      // connect() will reject if the server is still down; in that case
      // the WebSocket's onclose fires (because hasBeenConnected is true)
      // and queues the next attempt from there.
      this.connect().catch(() => {
        /* handled via onclose → scheduleReconnect */
      });
    }, delay);
  }

  on(type: string, handler: MessageHandler): void {
    if (!this.handlers.has(type)) this.handlers.set(type, []);
    this.handlers.get(type)!.push(handler);
  }

  off(type: string, handler: MessageHandler): void {
    const handlers = this.handlers.get(type);
    if (handlers) {
      this.handlers.set(type, handlers.filter((fn) => fn !== handler));
    }
  }

  send(data: any): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket is not connected. Cannot send:', data);
    }
  }

  disconnect(): void {
    this.intentionalClose = true;
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
    this.hasBeenConnected = false;
    this.reconnectAttempt = 0;
    this.notifiedDisconnect = false;
  }

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

export const executionWs = new ExecutionWebSocket();
