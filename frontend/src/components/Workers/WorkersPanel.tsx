import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react';
import { createPortal } from 'react-dom';
import type { WorkerDevice, WorkerInfo, WorkerStatus } from '../../api/workers';
import { useI18n } from '../../i18n';
import { useRunStore } from '../../store/runStore';
import { useWorkerStore } from '../../store/workerStore';
import { friendlyError } from '../../utils/errorMessages';
import { formatBytes } from '../../utils/formatBytes';
import { relativeTime } from '../SourceControl/scm';
import { trapTab } from '../shared/focusTrap';
import { AddWorkerDialog } from './AddWorkerDialog';
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

function WorkerCard({ worker, serverVersion, now }: {
  worker: WorkerInfo;
  serverVersion: string | null;
  now: number;
}) {
  const { t, locale } = useI18n();
  const busy = useWorkerStore((s) => Boolean(s.busy[worker.id]));
  const setDraining = useWorkerStore((s) => s.setDraining);
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

  return (
    <article className={styles.card} aria-label={worker.name}>
      <header className={styles.cardHeader}>
        <h3 className={styles.workerName}>{worker.name}</h3>
        <span className={`${styles.chip} ${known ? styles[`chip_${worker.status}`] : styles.chip_neutral}`}>
          {known ? t(`workers.status.${worker.status}`) : worker.status}
        </span>
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
      </header>
      <dl className={styles.facts}>
        <dt>{t('workers.version')}</dt>
        <dd><VersionLine version={worker.version} serverVersion={serverVersion} /></dd>
        <dt>{t('workers.fingerprint')}</dt>
        <dd>
          {digest
            ? <code title={digest}>{digest.slice(0, FINGERPRINT_SHORT_LENGTH)}</code>
            : <span className={styles.muted}>{t('workers.fingerprint.none')}</span>}
        </dd>
        <dt>{t('workers.mounts')}</dt>
        <dd>
          {mounts.length === 0
            ? <span className={styles.muted}>{t('workers.mounts.none')}</span>
            : mounts.map(([label, path]) => (
                <span key={label} className={styles.mount} title={path}>{label}</span>
              ))}
        </dd>
        <dt>{t('workers.heartbeat')}</dt>
        <dd>{relativeTime(worker.last_heartbeat_at, locale, now) || '-'}</dd>
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
export function WorkersPanel() {
  const open = useWorkerStore((s) => s.panelOpen);
  if (!open) return null;
  return <WorkersDialog />;
}

function WorkersDialog() {
  const { t } = useI18n();
  const titleId = useId();
  const workers = useWorkerStore((s) => s.workers);
  const loaded = useWorkerStore((s) => s.loaded);
  const error = useWorkerStore((s) => s.error);
  const serverVersion = useWorkerStore((s) => s.serverVersion);
  const closePanel = useWorkerStore((s) => s.closePanel);
  const refresh = useWorkerStore((s) => s.refresh);
  const [addOpen, setAddOpen] = useState(false);
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
  let content;
  if (!loaded) {
    content = <p className={styles.muted}>{t('workers.loading')}</p>;
  } else if (workers.length === 0 && error === null) {
    content = <p className={styles.muted}>{t('workers.empty')}</p>;
  } else {
    content = workers.map((worker) => (
      <WorkerCard key={worker.id} worker={worker} serverVersion={serverVersion} now={now} />
    ));
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
            onClick={() => setAddOpen(true)}
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
          {content}
        </div>
      </div>
      {addOpen && <AddWorkerDialog mode="same-host" onClose={() => setAddOpen(false)} />}
    </div>,
    document.body,
  );
}
