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

        const typeHandlers = this.handlers.get(data.type) ?? [];
        typeHandlers.forEach((h) => h(data));

        const wildcardHandlers = this.handlers.get('*') ?? [];
        wildcardHandlers.forEach((h) => h(data));
      } catch {
        console.error('Failed to parse WebSocket message:', event.data);
      }
    };

    this.ws.onclose = () => {
      if (this.intentionalClose) return;
      // Don't loop on initial-connect failure — the connect() promise has
      // already rejected and the caller is responsible for surfacing that.
      if (!this.hasBeenConnected) return;
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
        this.hasBeenConnected = true;
        this.reconnectAttempt = 0;
        resolve();
      };
      this.ws!.onerror = () => reject(new Error('WebSocket connection failed'));
    });
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
