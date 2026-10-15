import { Suspense, useEffect, useId, useRef, useState, type KeyboardEvent } from 'react';
import { createPortal } from 'react-dom';
import type {
  SshWorker,
  TunnelState,
  WorkerDevice,
  WorkerInfo,
  WorkerNodeCatalog,
  WorkerRejection,
  WorkerStatus,
} from '../../api/workers';
import { useI18n } from '../../i18n';
import { useRunStore } from '../../store/runStore';
import { useWorkerStore } from '../../store/workerStore';
import { friendlyError } from '../../utils/errorMessages';
import { formatBytes } from '../../utils/formatBytes';
import { relativeTime } from '../SourceControl/scm';
import { trapTab } from '../shared/focusTrap';
import { lazyRetry } from '../shared/lazyRetry';
import styles from './WorkersPanel.module.css';

/** Hex characters of the environment digest shown; the tooltip has it all. */
export const FINGERPRINT_SHORT_LENGTH = 12;
/** Characters of a run id shown in a link; the tooltip has it all. */
const RUN_ID_SHORT_LENGTH = 8;

/**
 * The statuses this build has a word and a colour for. A newer server may
 * send another; it is shown as sent, in a neutral chip.
 */
const KNOWN_STATUSES: ReadonlySet<string> = new Set<WorkerStatus>([
  'online', 'degraded', 'disconnected', 'draining', 'offline',
]);

function RunLinks({ runIds }: { runIds: string[] }) {
  const { t } = useI18n();
  const revealRuns = useWorkerStore((s) => s.revealRuns);
  if (runIds.length === 0) return <span className={styles.muted}>{t('workers.running.none')}</span>;
  return (
    <span className={styles.runLinks}>
      {runIds.map((id) => (
        <button
          key={id}
          type="button"
          className={styles.linkBtn}
          title={t('workers.runLink', { id })}
          onClick={() => {
            revealRuns();
            void useRunStore.getState().select(id);
          }}
        >
          {id.slice(0, RUN_ID_SHORT_LENGTH)}
        </button>
      ))}
    </span>
  );
}

function DeviceRow({ device }: { device: WorkerDevice }) {
  const { t } = useI18n();
  const memory = device.memory;
  const hasMemory = memory != null && memory.free_bytes !== null && memory.total_bytes !== null;
  return (
    <li className={styles.device}>
      <span className={styles.deviceName}>
        <code>{device.value}</code>
        {device.name && device.name !== device.value && (
          <span className={styles.muted}> {device.name}</span>
        )}
      </span>
      {device.slots && (
        <span className={styles.fact}>
          {t('workers.device.slots', { taken: device.slots.taken, limit: device.slots.limit })}
        </span>
      )}
      {hasMemory && (
        <span className={styles.fact} title={memory.source}>
          {t('workers.device.memory', {
            free: formatBytes(memory.free_bytes as number),
            total: formatBytes(memory.total_bytes as number),
          })}
        </span>
      )}
      {device.running && <RunLinks runIds={device.running} />}
    </li>
  );
}

function VersionLine({ version, serverVersion }: { version: string; serverVersion: string | null }) {
  const { t } = useI18n();
  let check = null;
  if (serverVersion !== null) {
    check = version === serverVersion
      ? <span className={styles.ok}>{t('workers.version.matches')}</span>
      : <span className={styles.warn}>{t('workers.version.differs', { server: serverVersion })}</span>;
  }
  return (
    <span>
      {version}
      {check && <> - {check}</>}
    </span>
  );
}

/**
 * How the worker's node types compare with this server's (#781). A
 * mismatch lists the types on each side: a run naming the worker may use
 * only the types it has, and `any` never picks it.
 */
function CatalogLine({ catalog }: { catalog: WorkerNodeCatalog | undefined }) {
  const { t } = useI18n();
  if (catalog === undefined || catalog.status === 'unknown') {
    return <span className={styles.warn}>{t('workers.catalog.unknown')}</span>;
  }
  if (catalog.status === 'match') {
    return <span className={styles.ok}>{t('workers.catalog.matches')}</span>;
  }
  return (
    <span className={styles.catalog}>
      <span className={styles.warn}>{t('workers.catalog.differs')}</span>
      {catalog.missing.length > 0 && (
        <span className={styles.typeList}>
          {t('workers.catalog.missing', { count: catalog.missing.length })}{' '}
          <code>{catalog.missing.join(', ')}</code>
        </span>
      )}
      {catalog.extra.length > 0 && (
        <span className={styles.typeList}>
          {t('workers.catalog.extra', { count: catalog.extra.length })}{' '}
          <code>{catalog.extra.join(', ')}</code>
        </span>
      )}
    </span>
  );
}

/** A refused handshake: why, when, and from where (#781). */
function RejectionLine({ rejection, now }: { rejection: WorkerRejection; now: number }) {
  const { t, locale } = useI18n();
  const when = relativeTime(rejection.at, locale, now) || '-';
  return (
    <span className={styles.warn}>
      {rejection.reason}
      <span className={styles.muted}>
        {' '}({rejection.peer ? t('workers.rejected.from', { when, peer: rejection.peer }) : when})
      </span>
    </span>
  );
}

/** The tunnel states this build has a word for; another is shown as sent. */
const TUNNEL_STATES: ReadonlySet<string> = new Set<TunnelState>(['connecting', 'up', 'failed', 'stopped']);

/**
 * An SSH worker's tunnel (#784): its state, where it goes, Enable /
 * Disable, and while it is not up, the last lines ssh or the remote agent
 * wrote (the reason it failed: a refused key, an unknown host, no cdui).
 */
function TunnelLine({ entry }: { entry: SshWorker }) {
  const { t } = useI18n();
  const busy = useWorkerStore((s) => Boolean(s.busy[entry.id]));
  const setSshEnabled = useWorkerStore((s) => s.setSshEnabled);
  const where = entry.port === 22 ? entry.target : `${entry.target}:${entry.port}`;
  const showError = entry.state !== 'up' && entry.last_error;
  const known = TUNNEL_STATES.has(entry.state);
  return (
    <span className={styles.tunnel}>
      <span className={`${styles.chip} ${known ? styles[`chip_${entry.state}`] : styles.chip_neutral}`}>
        {known ? t(`workers.ssh.state.${entry.state}`) : entry.state}
      </span>
      <code title={entry.identity ?? undefined}>{where}</code>
      {entry.remote_port !== null && (
        <span className={styles.fact}>{t('workers.ssh.remotePort', { port: entry.remote_port })}</span>
      )}
      {entry.remote_pid != null && (
        <span className={styles.fact}>
          {t(entry.reattached ? 'workers.ssh.remotePidReattached' : 'workers.ssh.remotePid', { pid: entry.remote_pid })}
        </span>
      )}
      <button
        type="button"
        className={styles.btn}
        aria-disabled={busy}
        title={t(entry.enabled ? 'workers.ssh.disable.title' : 'workers.ssh.enable.title')}
        onClick={() => {
          if (!busy) void setSshEnabled(entry.id, !entry.enabled);
        }}
      >
        {t(entry.enabled ? 'workers.ssh.disable' : 'workers.ssh.enable')}
      </button>
      {entry.enabled && entry.gave_up && (
        <button
          type="button"
          className={styles.btn}
          aria-disabled={busy}
          title={t('workers.ssh.retry.title')}
          onClick={() => {
            if (!busy) void setSshEnabled(entry.id, true);
          }}
        >
          {t('workers.ssh.retry')}
        </button>
      )}
      {showError && (
        <pre className={styles.tunnelError} role="status">
          {entry.last_error}
        </pre>
      )}
      {entry.tail.length > 0 && (
        <details className={styles.tunnelTail}>
          <summary>{t('workers.ssh.output')}</summary>
          <pre>{entry.tail.join('\n')}</pre>
        </details>
      )}
    </span>
  );
}

/** An SSH worker whose agent has not connected yet (no worker entry). */
function SshOnlyCard({ entry }: { entry: SshWorker }) {
  const { t } = useI18n();
  const busy = useWorkerStore((s) => Boolean(s.busy[entry.id]));
  const deleteSsh = useWorkerStore((s) => s.deleteSsh);
  return (
    <article className={styles.card} aria-label={entry.id}>
      <header className={styles.cardHeader}>
        <h3 className={styles.workerName}>{entry.id}</h3>
        <span className={`${styles.chip} ${styles.chip_offline}`}>{t('workers.ssh.notYet')}</span>
        <span className={styles.spacer} />
        <button
          type="button"
          className={`${styles.btn} ${styles.danger}`}
          aria-disabled={busy}
          title={t('workers.ssh.delete.title')}
          onClick={() => {
            if (!busy) void deleteSsh(entry.id);
          }}
        >
          {t('workers.ssh.delete')}
        </button>
      </header>
      <dl className={styles.facts}>
        <dt>{t('workers.ssh.tunnel')}</dt>
        <dd><TunnelLine entry={entry} /></dd>
      </dl>
    </article>
  );
}

/** Statuses whose worker has no live connection: removing one ends nothing. */
const AWAY_STATUSES: ReadonlySet<string> = new Set<WorkerStatus>(['disconnected', 'offline']);

function WorkerCard({ worker, rejection, ssh, serverVersion, now, onRemoved, onRekey }: {
  worker: WorkerInfo;
  /** Its name's last refused handshake, if the server remembers one (#781). */
  rejection: WorkerRejection | undefined;
  /** Its SSH tunnel, when this server starts it over ssh (#784). */
  ssh?: SshWorker;
  serverVersion: string | null;
  now: number;
  /** Called after this card's worker was removed, to keep the focus in the dialog. */
  onRemoved: () => void;
  /** Open the Add worker dialog for a token bound to this worker (#780). */
  onRekey: (workerId: string) => void;
}) {
  const { t, locale } = useI18n();
  const busy = useWorkerStore((s) => Boolean(s.busy[worker.id]));
  const setDraining = useWorkerStore((s) => s.setDraining);
  const remove = useWorkerStore((s) => s.remove);
  const [confirming, setConfirming] = useState(false);
  const actionRef = useRef<HTMLButtonElement>(null);
  const mounts = Object.entries(worker.mounts);
  // Without per-device run ids from the server, the worker's runs are
  // listed once, under the worker. With them, a run no device lists is
  // still listed there: it is never hidden.
  const perDeviceRuns = worker.devices.some((d) => Array.isArray(d.running));
  const onDevices = new Set(worker.devices.flatMap((d) => d.running ?? []));
  const workerRuns = worker.running.filter((id) => !onDevices.has(id));
  const digest = worker.fingerprint_digest;
  const known = KNOWN_STATUSES.has(worker.status);

  // One button that changes its word, never disabled: a disabled or swapped
  // button drops the focus to the page, where the dialog's Escape and Tab
  // trap cannot hear it. A press while one is in flight does nothing.
  const toggleDrain = async () => {
    if (busy) return;
    await setDraining(worker.id, !worker.draining);
    const button = actionRef.current;
    if (button !== null && document.activeElement === document.body) button.focus();
  };

  // Removing a connected worker disconnects it and interrupts its runs, so
  // that takes a second press; one that is away goes at once.
  const removeWorker = async () => {
    if (busy) return;
    if (!AWAY_STATUSES.has(worker.status) && !confirming) {
      setConfirming(true);
      return;
    }
    await remove(worker.id);
    if (document.activeElement === document.body) onRemoved();
  };

  return (
    <article className={styles.card} aria-label={worker.name}>
      <header className={styles.cardHeader}>
        <h3 className={styles.workerName}>{worker.name}</h3>
        <span className={`${styles.chip} ${known ? styles[`chip_${worker.status}`] : styles.chip_neutral}`}>
          {known ? t(`workers.status.${worker.status}`) : worker.status}
        </span>
        {worker.plain_http && (
          <span className={`${styles.chip} ${styles.chip_degraded}`} title={t('workers.plainHttp.title')}>
            {t('workers.plainHttp')}
          </span>
        )}
        <span className={styles.spacer} />
        <button
          ref={actionRef}
          type="button"
          className={styles.btn}
          aria-disabled={busy}
          title={t(worker.draining ? 'workers.undrain.title' : 'workers.drain.title')}
          onClick={() => void toggleDrain()}
        >
          {t(worker.draining ? 'workers.undrain' : 'workers.drain')}
        </button>
        <button
          type="button"
          className={styles.btn}
          title={t('workers.rekey.title')}
          onClick={() => onRekey(worker.id)}
        >
          {t('workers.rekey')}
        </button>
        <button
          type="button"
          className={`${styles.btn} ${styles.danger}`}
          aria-disabled={busy}
          title={t(confirming ? 'workers.remove.confirmTitle' : 'workers.remove.title')}
          onClick={() => void removeWorker()}
          onBlur={() => setConfirming(false)}
        >
          {t(confirming ? 'workers.remove.confirm' : 'workers.remove')}
        </button>
      </header>
      <dl className={styles.facts}>
        {ssh && (
          <>
            <dt>{t('workers.ssh.tunnel')}</dt>
            <dd><TunnelLine entry={ssh} /></dd>
          </>
        )}
        <dt>{t('workers.version')}</dt>
        <dd><VersionLine version={worker.version} serverVersion={serverVersion} /></dd>
        <dt>{t('workers.fingerprint')}</dt>
        <dd>
          {digest
            ? <code title={digest}>{digest.slice(0, FINGERPRINT_SHORT_LENGTH)}</code>
            : <span className={styles.muted}>{t('workers.fingerprint.none')}</span>}
        </dd>
        <dt>{t('workers.catalog')}</dt>
        <dd><CatalogLine catalog={worker.node_catalog} /></dd>
        {rejection && (
          <>
            <dt>{t('workers.rejected')}</dt>
            <dd><RejectionLine rejection={rejection} now={now} /></dd>
          </>
        )}
        <dt>{t('workers.mounts')}</dt>
        <dd>
          {mounts.length === 0
            ? <span className={styles.muted}>{t('workers.mounts.none')}</span>
            : mounts.map(([label, path]) => (
                <span key={label} className={styles.mount} title={path}>{label}</span>
              ))}
        </dd>
        <dt>{t('workers.heartbeat')}</dt>
        <dd>
          {worker.last_heartbeat_at === null
            ? <span className={styles.muted}>{t('workers.heartbeat.never')}</span>
            : relativeTime(worker.last_heartbeat_at, locale, now) || '-'}
        </dd>
        {(!perDeviceRuns || workerRuns.length > 0) && (
          <>
            <dt>{t('workers.running')}</dt>
            <dd><RunLinks runIds={workerRuns} /></dd>
          </>
        )}
      </dl>
      <ul className={styles.devices}>
        {worker.devices.map((device) => (
          <DeviceRow key={device.value} device={device} />
        ))}
      </ul>
    </article>
  );
}

/**
 * The Workers panel (#776): every worker agent the server knows, polled
 * every 5 s while open, with Drain / Undrain and the Add worker dialog.
 * Opened from Settings -> Execution and from the Runs panel header; mounted
 * once in `App`.
 */
/** The Add worker dialog (#776), loaded the first time it opens. */
const AddWorkerDialog = lazyRetry(
  () => import('./AddWorkerDialog').then((module) => module.AddWorkerDialog),
  (props) => props.onClose(),
);

export function WorkersPanel() {
  const open = useWorkerStore((s) => s.panelOpen);
  if (!open) return null;
  return <WorkersDialog />;
}

function WorkersDialog() {
  const { t } = useI18n();
  const titleId = useId();
  const workers = useWorkerStore((s) => s.workers);
  const rejections = useWorkerStore((s) => s.rejections);
  const ssh = useWorkerStore((s) => s.ssh);
  const sshError = useWorkerStore((s) => s.sshError);
  const loaded = useWorkerStore((s) => s.loaded);
  const error = useWorkerStore((s) => s.error);
  const serverVersion = useWorkerStore((s) => s.serverVersion);
  const closePanel = useWorkerStore((s) => s.closePanel);
  const refresh = useWorkerStore((s) => s.refresh);
  /** The Add worker dialog: closed (null), open, or open to re-key a worker. */
  const [adding, setAdding] = useState<{ rekey: string | null } | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => useWorkerStore.getState().watch(), []);

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    return () => {
      if (previous?.isConnected) previous.focus();
    };
  }, []);

  const handleKeys = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      closePanel();
      return;
    }
    trapTab(event, dialogRef.current);
  };

  const now = Date.now();
  // A refusal of another agent's attempt to take a connected worker's name
  // is not that worker's own: it is listed with the refused attempts.
  const own = rejections.filter((r) => !r.live_owner);
  const rejectionOf = new Map(own.map((r) => [r.name, r]));
  // A refused name with no card is an agent that never appeared: listed on
  // its own, so the user sees why (#781).
  const names = new Set(workers.map((w) => w.name));
  const refusedOnly = rejections.filter((r) => r.live_owner || !names.has(r.name));
  const sshById = new Map(ssh.map((entry) => [entry.id, entry]));
  const sshOnly = ssh.filter((entry) => !workers.some((worker) => worker.id === entry.id));
  let content;
  if (!loaded) {
    content = <p className={styles.muted}>{t('workers.loading')}</p>;
  } else if (workers.length === 0 && refusedOnly.length === 0 && sshOnly.length === 0 && error === null) {
    content = <p className={styles.muted}>{t('workers.empty')}</p>;
  } else {
    content = workers.map((worker) => (
      <WorkerCard
        key={worker.id}
        worker={worker}
        rejection={rejectionOf.get(worker.name)}
        ssh={sshById.get(worker.id)}
        serverVersion={serverVersion}
        now={now}
        onRemoved={() => closeRef.current?.focus()}
        onRekey={(workerId) => setAdding({ rekey: workerId })}
      />
    ));
    content = (
      <>
        {content}
        {sshOnly.map((entry) => <SshOnlyCard key={`ssh:${entry.id}`} entry={entry} />)}
      </>
    );
  }

  return createPortal(
    <div
      className={styles.backdrop}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) closePanel();
      }}
    >
      <div
        ref={dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        onKeyDown={handleKeys}
      >
        <header className={styles.header}>
          <h2 id={titleId}>{t('workers.title')}</h2>
          <button type="button" className={styles.btn} onClick={() => void refresh()}>
            {t('workers.refresh')}
          </button>
          <button
            type="button"
            className={`${styles.btn} ${styles.accent}`}
            onClick={() => setAdding({ rekey: null })}
          >
            {t('workers.add')}
          </button>
          <button
            ref={closeRef}
            type="button"
            className={styles.close}
            onClick={closePanel}
            aria-label={t('workers.close')}
          >
            &#215;
          </button>
        </header>
        <div className={styles.body}>
          {error !== null && <div className={styles.error} role="alert">{friendlyError(error)}</div>}
          {sshError !== null && (
            <div className={styles.error} role="alert">
              {t('workers.ssh.readFailed', { error: friendlyError(sshError) })}
            </div>
          )}
          {content}
          {loaded && refusedOnly.length > 0 && (
            <section className={styles.rejections} aria-label={t('workers.rejections.title')}>
              <h3 className={styles.workerName}>{t('workers.rejections.title')}</h3>
              <p className={styles.muted}>{t('workers.rejections.intro')}</p>
              <dl className={styles.facts}>
                {refusedOnly.map((rejection) => (
                  <div key={rejection.name} className={styles.rejection}>
                    <dt>{rejection.name}</dt>
                    <dd><RejectionLine rejection={rejection} now={now} /></dd>
                  </div>
                ))}
              </dl>
            </section>
          )}
        </div>
      </div>
      {adding !== null && (
        <Suspense fallback={null}>
          <AddWorkerDialog
            mode="remote"
            rekey={adding.rekey}
            onClose={() => setAdding(null)}
            onAdded={() => void refresh()}
          />
        </Suspense>
      )}
    </div>,
    document.body,
  );
}
